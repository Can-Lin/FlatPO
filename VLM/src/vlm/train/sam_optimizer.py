import random
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import torch
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


def _get_decay_parameter_names(model: "torch.nn.Module") -> List[str]:
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    return [name for name in decay_parameters if "bias" not in name]


def _is_lora_param(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name or "lora_embedding_" in name


def _match_peft_weight(name: str, peft_weight: Literal["all", "q", "v", "k"]) -> bool:
    if peft_weight == "all":
        return _is_lora_param(name)
    if peft_weight == "q":
        return _is_lora_param(name) and ".q_proj." in name
    if peft_weight == "v":
        return _is_lora_param(name) and ".v_proj." in name
    if peft_weight == "k":
        return _is_lora_param(name) and ".k_proj." in name
    return False


class SAMStyleOptimizer(torch.optim.Optimizer):
    """
    A SAM-compatible wrapper that supports:
    - `variant="sam"`: perturb all eligible params.
    - `variant="flatpo"`: perturb top-k (or random-k) eligible params by grad norm.
    """

    def __init__(
        self,
        *,
        base_optimizer: "torch.optim.Optimizer",
        named_params: Iterable[Tuple[str, "torch.nn.Parameter"]],
        rho: float = 0.05,
        adaptive: bool = False,
        peft_weight: Literal["all", "q", "v", "k"] = "all",
        random_weight: bool = False,
        variant: Literal["sam", "flatpo"] = "sam",
        flatpo_sensitive_ratio: float = 0.05,
        debug: bool = False,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if flatpo_sensitive_ratio <= 0.0 or flatpo_sensitive_ratio > 1.0:
            raise ValueError(
                f"Invalid flatpo_sensitive_ratio: {flatpo_sensitive_ratio}, expected (0, 1]."
            )

        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(base_optimizer.param_groups, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        for group in self.param_groups:
            group.setdefault("rho", rho)
            group.setdefault("adaptive", adaptive)

        self.param_names: Dict["torch.nn.Parameter", str] = {
            p: name for name, p in named_params
        }
        self.peft_weight = peft_weight
        self.random_weight = random_weight
        self.variant = variant
        self.flatpo_sensitive_ratio = flatpo_sensitive_ratio
        self.debug = debug
        self.last_first_step_stats: Dict[str, float] = {}
        self.last_update_stats: Dict[str, float] = {}
        self.second_step_calls: int = 0
        self._debug_tracked_param_ptrs: Optional[set[int]] = None

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        if grad_norm is None:
            if self.debug:
                self.last_first_step_stats = {
                    "grad_norm": 0.0,
                    "matched_with_grad": 0.0,
                    "sensitive_count": 0.0,
                    "perturbed_count": 0.0,
                    "perturb_l2_norm": 0.0,
                }
            return

        sensitive_params: Optional[set["torch.nn.Parameter"]] = None
        matched_with_grad = 0
        if self.variant == "flatpo":
            ranked: List[Tuple["torch.nn.Parameter", float]] = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    name = self.param_names.get(p, "")
                    if _match_peft_weight(name, self.peft_weight):
                        ranked.append((p, p.grad.norm(p=2).item()))
            matched_with_grad = len(ranked)

            if len(ranked) > 0:
                k = max(1, int(len(ranked) * self.flatpo_sensitive_ratio))
                if self.random_weight:
                    sampled = random.sample(ranked, k)
                else:
                    sampled = sorted(ranked, key=lambda x: x[1], reverse=True)[:k]
                sensitive_params = {p for p, _ in sampled}
            else:
                sensitive_params = set()
        else:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    name = self.param_names.get(p, "")
                    if _match_peft_weight(name, self.peft_weight):
                        matched_with_grad += 1

        perturbed_count = 0
        perturb_l2_sq = 0.0

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            adaptive = bool(group.get("adaptive", False))

            for p in group["params"]:
                if p.grad is None:
                    continue

                name = self.param_names.get(p, "")
                if not _match_peft_weight(name, self.peft_weight):
                    continue
                if sensitive_params is not None and p not in sensitive_params:
                    continue

                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                self.state[p]["e_w"] = e_w
                p.add_(e_w)
                perturbed_count += 1
                if self.debug:
                    perturb_l2_sq += torch.sum(e_w.detach().float() ** 2).item()

        if self.debug:
            sensitive_count = (
                len(sensitive_params)
                if sensitive_params is not None
                else matched_with_grad
            )
            self.last_first_step_stats = {
                "grad_norm": grad_norm.detach().float().item(),
                "matched_with_grad": float(matched_with_grad),
                "sensitive_count": float(sensitive_count),
                "perturbed_count": float(perturbed_count),
                "perturb_l2_norm": perturb_l2_sq ** 0.5,
            }

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        self.second_step_calls += 1
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.state[p].get("e_w", None)
                if e_w is not None:
                    p.sub_(e_w)
                    self.state[p].pop("e_w", None)

        tracked_before: List[Tuple["torch.nn.Parameter", "torch.Tensor"]] = []
        if self.debug:
            if self._debug_tracked_param_ptrs is None:
                self._debug_tracked_param_ptrs = set()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    name = self.param_names.get(p, "")
                    if _match_peft_weight(name, self.peft_weight):
                        tracked_before.append((p, p.data.detach().clone()))
                        self._debug_tracked_param_ptrs.add(p.data_ptr())

        # Keep base optimizer hyper-params aligned in case wrappers/schedulers
        # mutate the outer optimizer groups only.
        for idx, base_group in enumerate(self.base_optimizer.param_groups):
            if idx >= len(self.param_groups):
                break
            group = self.param_groups[idx]
            for key in (
                "lr",
                "betas",
                "eps",
                "weight_decay",
                "amsgrad",
                "maximize",
                "foreach",
                "capturable",
                "differentiable",
                "fused",
            ):
                if key in group:
                    base_group[key] = group[key]

        self.base_optimizer.step()

        if self.debug:
            update_l2_sq = 0.0
            max_abs_update = 0.0
            updated_param_count = 0
            grad_l2_sq = 0.0
            nonzero_grad_param_count = 0
            state_step_advanced_count = 0
            fallback_sgd_count = 0
            base_lr_min = None
            base_lr_max = None
            min_lr = None
            max_lr = None
            for group in self.param_groups:
                lr = float(group.get("lr", 0.0))
                min_lr = lr if min_lr is None else min(min_lr, lr)
                max_lr = lr if max_lr is None else max(max_lr, lr)
            for base_group in self.base_optimizer.param_groups:
                lr = float(base_group.get("lr", 0.0))
                base_lr_min = lr if base_lr_min is None else min(base_lr_min, lr)
                base_lr_max = lr if base_lr_max is None else max(base_lr_max, lr)
            for p, before in tracked_before:
                if p.grad is not None:
                    g = p.grad.detach().float()
                    g_abs_max = g.abs().max().item()
                    if g_abs_max > 0.0:
                        nonzero_grad_param_count += 1
                    grad_l2_sq += torch.sum(g ** 2).item()
                state_step = self.base_optimizer.state.get(p, {}).get("step")
                if state_step is not None and float(state_step) > 0.0:
                    state_step_advanced_count += 1
                delta = p.data.detach().float() - before.float()
                delta_abs_max = delta.abs().max().item()
                if delta_abs_max > 0.0:
                    updated_param_count += 1
                if delta_abs_max > max_abs_update:
                    max_abs_update = delta_abs_max
                update_l2_sq += torch.sum(delta ** 2).item()

            # If grads are non-zero but AdamW produced zero parameter delta,
            # apply a minimal SGD fallback in debug mode to isolate step-path issues.
            if updated_param_count == 0 and nonzero_grad_param_count > 0:
                for group in self.param_groups:
                    lr = float(group.get("lr", 0.0))
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        name = self.param_names.get(p, "")
                        if not _match_peft_weight(name, self.peft_weight):
                            continue
                        if float(p.grad.detach().abs().max().item()) == 0.0:
                            continue
                        p.add_(p.grad, alpha=-lr)
                        fallback_sgd_count += 1

                # Recompute update stats after fallback.
                update_l2_sq = 0.0
                max_abs_update = 0.0
                updated_param_count = 0
                for p, before in tracked_before:
                    delta = p.data.detach().float() - before.float()
                    delta_abs_max = delta.abs().max().item()
                    if delta_abs_max > 0.0:
                        updated_param_count += 1
                    if delta_abs_max > max_abs_update:
                        max_abs_update = delta_abs_max
                    update_l2_sq += torch.sum(delta ** 2).item()

            sample_param_before = 0.0
            sample_param_after = 0.0
            sample_exp_avg_norm = 0.0
            sample_exp_avg_sq_norm = 0.0
            if len(tracked_before) > 0:
                sample_p, sample_before = tracked_before[0]
                sample_param_before = float(sample_before.view(-1)[0].item())
                sample_param_after = float(sample_p.data.detach().view(-1)[0].item())
                sample_state = self.base_optimizer.state.get(sample_p, {})
                exp_avg = sample_state.get("exp_avg")
                exp_avg_sq = sample_state.get("exp_avg_sq")
                if exp_avg is not None:
                    sample_exp_avg_norm = float(exp_avg.detach().float().norm().item())
                if exp_avg_sq is not None:
                    sample_exp_avg_sq_norm = float(
                        exp_avg_sq.detach().float().norm().item()
                    )

            self.last_update_stats = {
                "update_l2_norm": update_l2_sq ** 0.5,
                "max_abs_update": max_abs_update,
                "updated_param_count": float(updated_param_count),
                "tracked_param_count": float(len(tracked_before)),
                "second_step_calls": float(self.second_step_calls),
                "lr_min": float(min_lr or 0.0),
                "lr_max": float(max_lr or 0.0),
                "base_lr_min": float(base_lr_min or 0.0),
                "base_lr_max": float(base_lr_max or 0.0),
                "second_grad_l2_norm": grad_l2_sq ** 0.5,
                "nonzero_grad_param_count": float(nonzero_grad_param_count),
                "tracked_unique_ptr_count": float(
                    len(self._debug_tracked_param_ptrs or set())
                ),
                "state_step_advanced_count": float(state_step_advanced_count),
                "fallback_sgd_count": float(fallback_sgd_count),
                "sample_param_before": sample_param_before,
                "sample_param_after": sample_param_after,
                "sample_exp_avg_norm": sample_exp_avg_norm,
                "sample_exp_avg_sq_norm": sample_exp_avg_sq_norm,
            }

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        # Trainer calls this after second backward; we only do restore + base step here.
        self.second_step(zero_grad=False)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self) -> Optional["torch.Tensor"]:
        norms = []
        shared_device = None
        for group in self.param_groups:
            adaptive = bool(group.get("adaptive", False))
            for p in group["params"]:
                if p.grad is None:
                    continue
                if shared_device is None:
                    shared_device = p.grad.device
                norms.append(
                    ((torch.abs(p) if adaptive else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                )

        if len(norms) == 0 or shared_device is None:
            return None
        return torch.norm(torch.stack(norms), p=2)

    def state_dict(self):
        state = super().state_dict()
        state["_sam_meta"] = {
            "peft_weight": self.peft_weight,
            "random_weight": self.random_weight,
            "variant": self.variant,
            "flatpo_sensitive_ratio": self.flatpo_sensitive_ratio,
        }
        state["_base_optimizer"] = self.base_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict):
        base_state = state_dict.pop("_base_optimizer", None)
        state_dict.pop("_sam_meta", None)
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        if base_state is not None:
            self.base_optimizer.load_state_dict(base_state)


def create_sam_style_optimizer(
    model: "torch.nn.Module",
    training_args,
    finetuning_args,
) -> "torch.optim.Optimizer":
    decay_param_names = _get_decay_parameter_names(model)
    decay_params: List["torch.nn.Parameter"] = []
    nodecay_params: List["torch.nn.Parameter"] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in decay_param_names:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=nodecay_params, weight_decay=0.0),
        dict(params=decay_params, weight_decay=training_args.weight_decay),
    ]
    base_optimizer = optim_class(param_groups, **optim_kwargs)
    return SAMStyleOptimizer(
        base_optimizer=base_optimizer,
        named_params=model.named_parameters(),
        rho=finetuning_args.sam_rho,
        adaptive=finetuning_args.sam_adaptive,
        peft_weight=finetuning_args.sam_peft_weight,
        random_weight=finetuning_args.sam_random_weight,
        variant=finetuning_args.sam_mode,
        flatpo_sensitive_ratio=finetuning_args.sam_topk_ratio,
        debug=finetuning_args.sam_debug,
    )
