from typing import TYPE_CHECKING, Any, Dict, List, Optional

from utils.packages import is_ray_available
from src.vlm.hyparams.parser import get_ray_args, get_train_args, read_args
from src.vlm.train.callback import LogCallback, PissaConvertCallback, ReporterCallback
from src.vlm.train.sft.workflow_sam import run_sft_sam
from src.vlm.train.trainer_utils import get_ray_trainer, get_swanlab_callback


if is_ray_available():
    from ray.train.huggingface.transformers import RayTrainReportCallback


if TYPE_CHECKING:
    from transformers import TrainerCallback


def _training_function(config: Dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: List[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args)
    )

    if not finetuning_args.use_sam:
        raise ValueError(
            "This entrypoint is for SAM-style LoRA training. Set `use_sam: true` in config."
        )

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    callbacks.append(
        ReporterCallback(model_args, data_args, finetuning_args, generating_args)
    )

    run_sft_sam(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
    )


def run_exp_sam(
    args: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List["TrainerCallback"]] = None,
) -> None:
    args = read_args(args)
    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        _training_function(config={"args": args, "callbacks": callbacks})
