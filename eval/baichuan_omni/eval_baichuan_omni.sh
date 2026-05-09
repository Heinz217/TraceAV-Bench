#!/bin/bash
# LV-Bench eval launcher. Set required paths via environment before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_baichuan_omni.py"

# Required:
: "${BAICHUAN_MODEL_PATH:?Set BAICHUAN_MODEL_PATH to the model weights directory.}"
: "${BAICHUAN_CLEANED_DIR:?Set BAICHUAN_CLEANED_DIR to the cleaned/ JSON directory.}"
: "${BAICHUAN_VIDEOS_DIR:?Set BAICHUAN_VIDEOS_DIR to the videos directory.}"

# Optional:
BAICHUAN_OUTPUT_DIR="${BAICHUAN_OUTPUT_DIR:-$SCRIPT_DIR/runs}"
mkdir -p "$BAICHUAN_OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BAICHUAN_MODEL_PATH
export BAICHUAN_CLEANED_DIR
export BAICHUAN_VIDEOS_DIR
export BAICHUAN_OUTPUT_DIR
export BAICHUAN_CACHE_DIR="${BAICHUAN_CACHE_DIR:-}"
export BAICHUAN_MAX_FRAMES="${BAICHUAN_MAX_FRAMES:-}"
export BAICHUAN_MAX_PIXELS="${BAICHUAN_MAX_PIXELS:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 "$PY_SCRIPT"
