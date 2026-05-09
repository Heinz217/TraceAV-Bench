#!/bin/bash
# LV-Bench eval launcher. Set required paths via environment before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_humanomni.py"

# Required:
: "${HUMANOMNI_MODEL_PATH:?Set HUMANOMNI_MODEL_PATH to the model weights directory.}"
: "${HUMANOMNI_CLEANED_DIR:?Set HUMANOMNI_CLEANED_DIR to the cleaned/ JSON directory.}"
: "${HUMANOMNI_VIDEOS_DIR:?Set HUMANOMNI_VIDEOS_DIR to the videos directory.}"
: "${HUMANOMNI_REPO_DIR:?Set HUMANOMNI_REPO_DIR (path to the cloned model repository).}"

# Optional:
HUMANOMNI_OUTPUT_DIR="${HUMANOMNI_OUTPUT_DIR:-$SCRIPT_DIR/runs}"
mkdir -p "$HUMANOMNI_OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HUMANOMNI_MODEL_PATH
export HUMANOMNI_CLEANED_DIR
export HUMANOMNI_VIDEOS_DIR
export HUMANOMNI_OUTPUT_DIR
export HUMANOMNI_REPO_DIR
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 "$PY_SCRIPT"
