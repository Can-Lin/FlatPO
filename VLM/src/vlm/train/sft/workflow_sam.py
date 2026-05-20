from typing import TYPE_CHECKING, List, Optional

from src.vlm.data.data_collactor import DataCollatorForQwenVL
from src.vlm.data.loader import get_dataset
from src.vlm.model.loader import load_model, load_tokenizer
from src.vlm.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from src.vlm.train.sft.sam_trainer import SAMSeq2SeqTrainer
from src.vlm.train.trainer_utils import create_modelcard_and_push
from utils.logger import logger
from utils.misc import calculate_tps, get_logits_processor
from utils.ploting import plot_loss


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from src.vlm.hyparams.data_args import DataArguments
    from src.vlm.hyparams.finetune_args import FinetuningArguments
    from src.vlm.hyparams.generating_args import GeneratingArguments
    from src.vlm.hyparams.model_args import ModelArguments


def run_sft_sam(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset_module = get_dataset(model_args, data_args, training_args, stage="sft")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    data_collator = DataCollatorForQwenVL(tokenizer_module["processor"])
    training_args.generation_max_length = (
        training_args.generation_max_length or data_args.cutoff_len
    )
    training_args.generation_num_beams = (
        data_args.eval_num_beams or training_args.generation_num_beams
    )
    training_args.remove_unused_columns = False

    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    trainer = SAMSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [
        tokenizer.eos_token_id
    ] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(
                training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"]
            )

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.warning(
            "Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead."
        )
        predict_results = trainer.predict(
            dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs
        )
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(
            dataset_module["eval_dataset"],
            predict_results,
            generating_args.skip_special_tokens,
        )

    create_modelcard_and_push(
        trainer, model_args, data_args, training_args, finetuning_args
    )
