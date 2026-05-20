#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/lincan/Finetune-Qwen2.5-VL}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/config/vlm_config_iconqa_multi_gpu.yaml}"
ACC_CONFIG="${ACC_CONFIG:-${PROJECT_DIR}/config/accelerate_gpu4567.yaml}"

cd "${PROJECT_DIR}"
python main.py "${CONFIG_FILE}" "${ACC_CONFIG}"
