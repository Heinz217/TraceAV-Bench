#!/bin/bash
# LV-Bench eval launcher (OpenAI-compatible chat endpoint, e.g. vLLM server).
#
# Prerequisites:
#   1. Start your own vLLM / OpenAI-compatible server hosting the model, e.g.
#        python -m vllm.entrypoints.openai.api_server --model /path/to/qwen3-omni-instruct \
#          --served-model-name qwen3-omni-instruct --host 0.0.0.0 --port 8000 ...
#   2. Export the required environment variables below (or edit this file).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_qwen3_omni_instruct.py"

: "${BENCHMARK_DIR:?Set BENCHMARK_DIR (root containing cleaned/ and videos/).}"
: "${LVBENCH_BASE_URL:?Set LVBENCH_BASE_URL (e.g. http://127.0.0.1:8000).}"
MODEL="${MODEL:-qwen3-omni-instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/runs}"
MAX_WORKERS="${MAX_WORKERS:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"

mkdir -p "$OUTPUT_DIR"

python3 "$PY_SCRIPT" \
  --benchmark-dir "$BENCHMARK_DIR" \
  --base-url "$LVBENCH_BASE_URL" \
  --model "$MODEL" \
  --output-root "$OUTPUT_DIR" \
  --max-workers "$MAX_WORKERS" \
  --max-tokens "$MAX_TOKENS" \
  --timeout-sec "$TIMEOUT_SEC"
