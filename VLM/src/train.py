from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vlm.train.tuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
