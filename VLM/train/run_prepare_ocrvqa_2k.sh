#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/lincan/Finetune-Qwen2.5-VL}"

cd "${PROJECT_DIR}"
python scripts/prepare_ocrvqa_train.py \
  --output-json data/train_ocrvqa_qwen_2k_for_vl.json \
  --sample-size 2000
