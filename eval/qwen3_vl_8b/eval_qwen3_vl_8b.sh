#!/bin/bash
# LV-Bench eval launcher. Set required paths via environment before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_qwen3_vl_8b.py"

# Required:
: "${QWEN3VL_MODEL_PATH:?Set QWEN3VL_MODEL_PATH to the model weights directory.}"
: "${QWEN3VL_CLEANED_DIR:?Set QWEN3VL_CLEANED_DIR to the cleaned/ JSON directory.}"
: "${QWEN3VL_VIDEOS_DIR:?Set QWEN3VL_VIDEOS_DIR to the videos directory.}"

# Optional:
QWEN3VL_OUTPUT_DIR="${QWEN3VL_OUTPUT_DIR:-$SCRIPT_DIR/runs}"
mkdir -p "$QWEN3VL_OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN3VL_MODEL_PATH
export QWEN3VL_CLEANED_DIR
export QWEN3VL_VIDEOS_DIR
export QWEN3VL_OUTPUT_DIR
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 "$PY_SCRIPT"
