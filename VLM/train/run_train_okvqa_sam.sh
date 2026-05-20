#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/lincan/Finetune-Qwen2.5-VL}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/config/vlm_config_okvqa_multi_gpu_sam.yaml}"
ACC_CONFIG="${ACC_CONFIG:-${PROJECT_DIR}/config/accelerate_gpu4567.yaml}"

cd "${PROJECT_DIR}"
python main_sam.py "${CONFIG_FILE}" "${ACC_CONFIG}"
