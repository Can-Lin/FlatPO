from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vlm.train.tuner import export_model
from typing import Optional, Dict, Any
from src.vlm.hyparams.parser import read_args


def merge_model(
    args: Optional[Dict[str, Any]] = None,
):
    args = read_args(args)
    export_model(args=args)


if __name__ == "__main__":
    merge_model()
