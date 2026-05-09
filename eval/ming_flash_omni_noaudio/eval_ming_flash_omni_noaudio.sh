#!/bin/bash
# LV-Bench eval launcher. Set required paths via environment before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_ming_flash_omni_noaudio.py"

# Required:
: "${MING_MODEL_PATH:?Set MING_MODEL_PATH to the model weights directory.}"
: "${MING_CLEANED_DIR:?Set MING_CLEANED_DIR to the cleaned/ JSON directory.}"
: "${MING_VIDEOS_DIR:?Set MING_VIDEOS_DIR to the videos directory.}"
: "${MING_REPO_DIR:?Set MING_REPO_DIR (path to the cloned model repository).}"

# Optional:
MING_OUTPUT_DIR="${MING_OUTPUT_DIR:-$SCRIPT_DIR/runs}"
mkdir -p "$MING_OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MING_MODEL_PATH
export MING_CLEANED_DIR
export MING_VIDEOS_DIR
export MING_OUTPUT_DIR
export MING_REPO_DIR
export MING_MAX_VIDEO_FRAMES="${MING_MAX_VIDEO_FRAMES:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 "$PY_SCRIPT"
