#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_vllm_test.sh — Quick vLLM inference smoke test
# ──────────────────────────────────────────────────────────────
#
# Examples:
#   bash run_vllm_test.sh
#   bash run_vllm_test.sh --model_path /data/models/kimi-k2 --tp_size 32
#   TP_SIZE=16 bash run_vllm_test.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/wf/ckpt/iter_0000900_test/}"
TP_SIZE="${TP_SIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
QUANTIZATION=""

# ── Parse CLI arguments ───────────────────────────────────────
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_path   PATH   HuggingFace model directory  (default: ${MODEL_PATH})"
    echo "  --tp_size      N      Tensor-parallel size          (default: ${TP_SIZE})"
    echo "  --max_tokens   N      Max tokens to generate        (default: ${MAX_TOKENS})"
    echo "  --temperature  F      Sampling temperature           (default: ${TEMPERATURE})"
    echo "  --top_p        F      Top-p sampling                 (default: ${TOP_P})"
    echo "  --quantization METHOD Quantization (awq/gptq/fp8)   (default: none)"
    echo "  -h, --help            Show this message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)    MODEL_PATH="$2";    shift 2 ;;
        --tp_size)       TP_SIZE="$2";       shift 2 ;;
        --max_tokens)    MAX_TOKENS="$2";    shift 2 ;;
        --temperature)   TEMPERATURE="$2";   shift 2 ;;
        --top_p)         TOP_P="$2";         shift 2 ;;
        --quantization)  QUANTIZATION="$2";  shift 2 ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Build python command ──────────────────────────────────────
CMD=(
    python "${SCRIPT_DIR}/test_vllm_generation.py"
    --model_path "${MODEL_PATH}"
    --tp_size "${TP_SIZE}"
    --max_tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
)

if [[ -n "${QUANTIZATION}" ]]; then
    CMD+=(--quantization "${QUANTIZATION}")
fi

echo ">>> Running: ${CMD[*]}"
exec "${CMD[@]}"
