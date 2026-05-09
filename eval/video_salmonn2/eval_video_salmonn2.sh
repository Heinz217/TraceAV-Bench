#!/bin/bash
# LV-Bench eval launcher. Set required paths via environment before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_video_salmonn2.py"

# Required:
: "${SALMONN_MODEL_PATH:?Set SALMONN_MODEL_PATH to the model weights directory.}"
: "${SALMONN_CLEANED_DIR:?Set SALMONN_CLEANED_DIR to the cleaned/ JSON directory.}"
: "${SALMONN_VIDEOS_DIR:?Set SALMONN_VIDEOS_DIR to the videos directory.}"
: "${SALMONN_REPO_DIR:?Set SALMONN_REPO_DIR (path to the cloned model repository).}"

# Optional:
SALMONN_OUTPUT_DIR="${SALMONN_OUTPUT_DIR:-$SCRIPT_DIR/runs}"
mkdir -p "$SALMONN_OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SALMONN_MODEL_PATH
export SALMONN_CLEANED_DIR
export SALMONN_VIDEOS_DIR
export SALMONN_OUTPUT_DIR
export SALMONN_REPO_DIR
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 "$PY_SCRIPT"
