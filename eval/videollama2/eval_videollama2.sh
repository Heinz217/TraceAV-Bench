#!/bin/bash
# LV-Bench eval launcher. Set required paths via environment before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_videollama2.py"

# Required:
: "${VL2AV_MODEL_PATH:?Set VL2AV_MODEL_PATH to the model weights directory.}"
: "${VL2AV_CLEANED_DIR:?Set VL2AV_CLEANED_DIR to the cleaned/ JSON directory.}"
: "${VL2AV_VIDEOS_DIR:?Set VL2AV_VIDEOS_DIR to the videos directory.}"
: "${VL2AV_REPO_DIR:?Set VL2AV_REPO_DIR (path to the cloned VideoLLaMA2 repository).}"
: "${VL2AV_SIGLIP_PATH:?Set VL2AV_SIGLIP_PATH (path to the siglip-so400m weights directory).}"

# Optional:
VL2AV_OUTPUT_DIR="${VL2AV_OUTPUT_DIR:-$SCRIPT_DIR/runs}"
mkdir -p "$VL2AV_OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VL2AV_MODEL_PATH
export VL2AV_CLEANED_DIR
export VL2AV_VIDEOS_DIR
export VL2AV_OUTPUT_DIR
export VL2AV_REPO_DIR
export VL2AV_SIGLIP_PATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 "$PY_SCRIPT"
