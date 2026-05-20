#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_lmms_eval_scienceqa.sh
#   MODEL_PATH=/path/to/merged_model TASKS=scienceqa_full NUM_PROCESSES=4 bash run_lmms_eval_scienceqa.sh
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="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"
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-/home/lincan/lmms-eval}"
MODEL_PATH="${MODEL_PATH:-/ssd/lincan/mllm_ckpt/qwen2.5-vl/lora/okvqa_flatpo/merge/checkpoint-6756}"
TASKS="${TASKS:-ok_vqa}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/lincan/lmms-eval/results}"

if [[ ! -d "${LMMS_EVAL_DIR}" ]]; then
  echo "[ERROR] lmms-eval directory not found: ${LMMS_EVAL_DIR}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] model directory not found: ${MODEL_PATH}"
  exit 1
fi

cd "${LMMS_EVAL_DIR}"

accelerate launch --num_processes="${NUM_PROCESSES}" -m lmms_eval \
  --model qwen2_5_vl \
  --model_args "pretrained=${MODEL_PATH},device_map=auto" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --output_path "${OUTPUT_PATH}"
