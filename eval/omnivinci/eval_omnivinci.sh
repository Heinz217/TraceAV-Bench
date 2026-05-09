#!/bin/bash
# LV-Bench eval launcher. Set required paths via environment before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_omnivinci.py"

# Required:
: "${OMNIVINCI_MODEL_PATH:?Set OMNIVINCI_MODEL_PATH to the model weights directory.}"
: "${OMNIVINCI_CLEANED_DIR:?Set OMNIVINCI_CLEANED_DIR to the cleaned/ JSON directory.}"
: "${OMNIVINCI_VIDEOS_DIR:?Set OMNIVINCI_VIDEOS_DIR to the videos directory.}"

# Optional:
OMNIVINCI_OUTPUT_DIR="${OMNIVINCI_OUTPUT_DIR:-$SCRIPT_DIR/runs}"
mkdir -p "$OMNIVINCI_OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMNIVINCI_MODEL_PATH
export OMNIVINCI_CLEANED_DIR
export OMNIVINCI_VIDEOS_DIR
export OMNIVINCI_OUTPUT_DIR
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 "$PY_SCRIPT"
