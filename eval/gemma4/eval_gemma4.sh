#!/bin/bash
# LV-Bench eval launcher. Set required paths via environment before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_gemma4.py"

# Required:
: "${GEMMA4_MODEL_PATH:?Set GEMMA4_MODEL_PATH to the model weights directory.}"
: "${GEMMA4_CLEANED_DIR:?Set GEMMA4_CLEANED_DIR to the cleaned/ JSON directory.}"
: "${GEMMA4_VIDEOS_DIR:?Set GEMMA4_VIDEOS_DIR to the videos directory.}"

# Optional:
GEMMA4_OUTPUT_DIR="${GEMMA4_OUTPUT_DIR:-$SCRIPT_DIR/runs}"
mkdir -p "$GEMMA4_OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GEMMA4_MODEL_PATH
export GEMMA4_CLEANED_DIR
export GEMMA4_VIDEOS_DIR
export GEMMA4_OUTPUT_DIR
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 "$PY_SCRIPT"
