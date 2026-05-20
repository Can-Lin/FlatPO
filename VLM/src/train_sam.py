from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vlm.train.tuner_sam import run_exp_sam


def main():
    run_exp_sam()


def _mp_fn(index):
    run_exp_sam()


if __name__ == "__main__":
    main()
