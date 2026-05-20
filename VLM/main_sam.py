from accelerate.commands.launch import launch_command, launch_command_parser

from src.vlm.hyparams.parser import get_finetune_args, read_args
from src.train_sam import main
import sys


def do_train():
    config_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "config/vlm_config_scienceqa_multi_gpu_sam.yaml"
    )
    finetuning_args = get_finetune_args(read_args())
    if finetuning_args.use_accelerate:
        accelerate_config_file = (
            sys.argv[2] if len(sys.argv) > 2 else "config/accelerate.yaml"
        )

        accelerate_args = ["--config_file", accelerate_config_file, "--module"]
        parser = launch_command_parser()
        accelerate_args = accelerate_args + ["src.train_sam", config_file]
        args = parser.parse_args(accelerate_args)
        launch_command(args)
    else:
        main()


if __name__ == "__main__":
    do_train()
