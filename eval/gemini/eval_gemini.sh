#!/bin/bash
# LV-Bench Gemini evaluator launcher.
#
# Required:
#   GEMINI_API_KEY      - your Gemini (or proxy) API key
#   BENCHMARK_DIR       - root with cleaned/ + videos/
# Optional:
#   GEMINI_MODEL        - e.g. gemini-2.5-pro (default)
#   GEMINI_GENERATE_URL - custom generateContent URL (default: Google v1beta)
#   OUTPUT_DIR          - where run folders are saved
#   MAX_WORKERS         - concurrent generateContent calls (default 2)
#   TASKS               - space-separated task stems to restrict evaluation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_gemini.py"

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in the environment.}"
: "${BENCHMARK_DIR:?Set BENCHMARK_DIR (root containing cleaned/ and videos/).}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-pro}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/eval_results}"
MAX_WORKERS="${MAX_WORKERS:-2}"

mkdir -p "$OUTPUT_DIR"

ARGS=(
  --benchmark-dir "$BENCHMARK_DIR"
  --model "$GEMINI_MODEL"
  --api-key "$GEMINI_API_KEY"
  --output-root "$OUTPUT_DIR"
  --max-workers "$MAX_WORKERS"
)

if [[ -n "${GEMINI_GENERATE_URL:-}" ]]; then
  ARGS+=(--generate-url "$GEMINI_GENERATE_URL")
fi
if [[ -n "${TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  TASK_ARR=( $TASKS )
  ARGS+=(--tasks "${TASK_ARR[@]}")
fi

python3 "$PY_SCRIPT" "${ARGS[@]}"
