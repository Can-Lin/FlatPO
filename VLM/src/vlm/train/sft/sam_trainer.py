from typing import Any, Dict, Optional, Union

import torch
from transformers.utils import is_sagemaker_mp_enabled
from typing_extensions import override

from src.vlm.train.sft.trainer import CustomSeq2SeqTrainer
from src.vlm.train.trainer_utils import create_custom_optimizer
from src.vlm.train.sam_optimizer import _match_peft_weight, create_sam_style_optimizer
from utils.logger import logger


class SAMSeq2SeqTrainer(CustomSeq2SeqTrainer):
    def _unwrap_sam_optimizer(self):
        optimizer = self.optimizer
        # Transformers + Accelerate may wrap the real optimizer (e.g. AcceleratedOptimizer).
        for _ in range(4):
            if hasattr(optimizer, "first_step"):
                return optimizer
            inner = getattr(optimizer, "optimizer", None)
            if inner is None:
                break
            optimizer = inner
        return None

    def _log_sam_binding_once(self) -> None:
        if not getattr(self.finetuning_args, "sam_debug", False):
            return
        if getattr(self, "_sam_binding_logged", False):
            return
        if hasattr(self, "is_world_process_zero") and not self.is_world_process_zero():
            return

        model_lora_param_ids = set()
        model_lora_trainable = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if _match_peft_weight(name, self.finetuning_args.sam_peft_weight):
                model_lora_param_ids.add(id(param))
                model_lora_trainable += 1

        sam_optimizer = self._unwrap_sam_optimizer()
        optimizer_for_stats = sam_optimizer if sam_optimizer is not None else self.optimizer
        optimizer_param_ids = set()
        optimizer_lora_bound = 0
        for group in optimizer_for_stats.param_groups:
            for param in group["params"]:
                optimizer_param_ids.add(id(param))
                if id(param) in model_lora_param_ids:
                    optimizer_lora_bound += 1

        base_opt_name = "unknown"
        if sam_optimizer is not None and hasattr(sam_optimizer, "base_optimizer"):
            base_opt_name = type(sam_optimizer.base_optimizer).__name__

        logger.info(
            (
                "[SAM DEBUG] binding model_lora_trainable=%.0f optimizer_params=%.0f "
                "optimizer_lora_bound=%.0f optimizer_cls=%s base_optimizer_cls=%s"
            )
            % (
                float(model_lora_trainable),
                float(len(optimizer_param_ids)),
                float(optimizer_lora_bound),
                type(optimizer_for_stats).__name__,
                base_opt_name,
            )
        )
        self._sam_binding_logged = True

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            if self.finetuning_args.use_sam:
                self.optimizer = create_sam_style_optimizer(
                    self.model, self.args, self.finetuning_args
                )
            else:
                self.optimizer = create_custom_optimizer(
                    self.model, self.args, self.finetuning_args
                )
        optimizer = super().create_optimizer()
        self._log_sam_binding_once()
        return optimizer

    @override
    def training_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.finetuning_args.use_sam:
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            except TypeError:
                return super().training_step(model, inputs)

        if self.args.gradient_accumulation_steps != 1:
            raise ValueError(
                "SAM training currently requires `gradient_accumulation_steps: 1`."
            )

        self._log_sam_binding_once()

        if is_sagemaker_mp_enabled():
            raise ValueError("SAM training is not implemented for SageMaker MP.")

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            try:
                loss = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch
                )
            except TypeError:
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)
        sam_optimizer = self._unwrap_sam_optimizer()
        if sam_optimizer is None:
            raise RuntimeError(
                "SAM optimizer is expected to expose `first_step` but it was not found."
            )
        sam_optimizer.first_step(zero_grad=True)

        with self.compute_loss_context_manager():
            try:
                loss_second = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch
                )
            except TypeError:
                loss_second = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss_second = loss_second.mean()

        self.accelerator.backward(loss_second)
        self._maybe_log_sam_debug(sam_optimizer, loss_second)
        return loss_second.detach()

    def _maybe_log_sam_debug(
        self,
        sam_optimizer: Any,
        loss_second: "torch.Tensor",
    ) -> None:
        if not getattr(self.finetuning_args, "sam_debug", False):
            return

        if hasattr(self, "is_world_process_zero") and not self.is_world_process_zero():
            return

        log_steps = max(1, int(getattr(self.finetuning_args, "sam_debug_log_steps", 10)))
        current_step = int(self.state.global_step) + 1
        if current_step % log_steps != 0:
            return

        first_stats = getattr(sam_optimizer, "last_first_step_stats", {})
        update_stats = getattr(sam_optimizer, "last_update_stats", {})
        logger.info(
            (
                "[SAM DEBUG] step=%d loss2=%.6f grad_norm=%.6f matched=%.0f sensitive=%.0f "
                "perturbed=%.0f perturb_l2=%.6f updated=%.0f tracked=%.0f "
                "update_l2=%.6f max_abs_update=%.6e second_step_calls=%.0f "
                "second_grad_l2=%.6f nonzero_grad_params=%.0f unique_ptrs=%.0f "
                "state_step_advanced=%.0f sample_param=[%.6e -> %.6e] "
                "sample_exp_avg_norm=%.6e sample_exp_avg_sq_norm=%.6e "
                "fallback_sgd=%.0f "
                "lr=[%.6e, %.6e] base_lr=[%.6e, %.6e]"
            )
            % (
                current_step,
                loss_second.detach().float().item(),
                float(first_stats.get("grad_norm", 0.0)),
                float(first_stats.get("matched_with_grad", 0.0)),
                float(first_stats.get("sensitive_count", 0.0)),
                float(first_stats.get("perturbed_count", 0.0)),
                float(first_stats.get("perturb_l2_norm", 0.0)),
                float(update_stats.get("updated_param_count", 0.0)),
                float(update_stats.get("tracked_param_count", 0.0)),
                float(update_stats.get("update_l2_norm", 0.0)),
                float(update_stats.get("max_abs_update", 0.0)),
                float(update_stats.get("second_step_calls", 0.0)),
                float(update_stats.get("second_grad_l2_norm", 0.0)),
                float(update_stats.get("nonzero_grad_param_count", 0.0)),
                float(update_stats.get("tracked_unique_ptr_count", 0.0)),
                float(update_stats.get("state_step_advanced_count", 0.0)),
                float(update_stats.get("sample_param_before", 0.0)),
                float(update_stats.get("sample_param_after", 0.0)),
                float(update_stats.get("sample_exp_avg_norm", 0.0)),
                float(update_stats.get("sample_exp_avg_sq_norm", 0.0)),
                float(update_stats.get("fallback_sgd_count", 0.0)),
                float(update_stats.get("lr_min", 0.0)),
                float(update_stats.get("lr_max", 0.0)),
                float(update_stats.get("base_lr_min", 0.0)),
                float(update_stats.get("base_lr_max", 0.0)),
            )
        )
