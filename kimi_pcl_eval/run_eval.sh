#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_eval.sh — One-click OpenCompass evaluation for Kimi-K2 1T
# ──────────────────────────────────────────────────────────────
#
# Usage:
#   bash run_eval.sh --model_path /path/to/model --tp_size 32
#   bash run_eval.sh   # uses defaults (local ./data)
#
# Environment variables MODEL_PATH, TP_SIZE, WORK_DIR, DATA_DIR
# are also respected and can be used instead of CLI flags.
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults (override via CLI flags or env vars) ─────────────
MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/wf/ckpt/iter_0000900_test/}"
TP_SIZE="${TP_SIZE:-8}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/outputs}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
MAX_OUT_LEN="${MAX_OUT_LEN:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# ── Parse CLI arguments ───────────────────────────────────────
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_path  PATH  HuggingFace model directory  (default: ${MODEL_PATH})"
    echo "  --tp_size     N     Tensor-parallel size          (default: ${TP_SIZE})"
    echo "  --work_dir    DIR   Output directory               (default: ${WORK_DIR})"
    echo "  --data_dir    DIR   Local dataset directory         (default: ${DATA_DIR})"
    echo "  --max_out_len N     Max output token length        (default: ${MAX_OUT_LEN})"
    echo "  --max_seq_len N     Max sequence length            (default: ${MAX_SEQ_LEN})"
    echo "  --batch_size  N     Evaluation batch size          (default: ${BATCH_SIZE})"
    echo "  -h, --help          Show this message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)  MODEL_PATH="$2";  shift 2 ;;
        --tp_size)     TP_SIZE="$2";     shift 2 ;;
        --work_dir)    WORK_DIR="$2";    shift 2 ;;
        --data_dir)    DATA_DIR="$2";    shift 2 ;;
        --max_out_len) MAX_OUT_LEN="$2"; shift 2 ;;
        --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        -h|--help)     usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Export for the OpenCompass config ──────────────────────────
export MODEL_PATH TP_SIZE MAX_OUT_LEN MAX_SEQ_LEN BATCH_SIZE

# ── Link local data into OpenCompass's expected location ──────
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: data directory not found: ${DATA_DIR}" >&2
    echo "  Run download_datasets.sh first, or set --data_dir." >&2
    exit 1
fi

OC_PKG_DIR="$(python -c 'import opencompass, os; print(os.path.dirname(opencompass.__file__))')"
OC_DATA_LINK="${OC_PKG_DIR}/data"

if [[ -e "${OC_DATA_LINK}" && ! -L "${OC_DATA_LINK}" ]]; then
    echo "WARNING: ${OC_DATA_LINK} already exists and is not a symlink; skipping link."
else
    rm -f "${OC_DATA_LINK}"
    ln -s "$(realpath "${DATA_DIR}")" "${OC_DATA_LINK}"
    echo ">>> Linked dataset dir: ${OC_DATA_LINK} -> $(realpath "${DATA_DIR}")"
fi

# ── Print configuration summary ───────────────────────────────
echo "============================================================"
echo "  Kimi-K2 1T  —  OpenCompass Evaluation"
echo "============================================================"
echo "  Model path   : ${MODEL_PATH}"
echo "  TP size      : ${TP_SIZE}"
echo "  Work dir     : ${WORK_DIR}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Max out len  : ${MAX_OUT_LEN}"
echo "  Max seq len  : ${MAX_SEQ_LEN}"
echo "  Batch size   : ${BATCH_SIZE}"
echo "============================================================"
echo ""

# ── Launch OpenCompass ────────────────────────────────────────
echo ">>> Starting OpenCompass evaluation ..."
opencompass "${SCRIPT_DIR}/configs/eval_kimi_1t_vllm.py" \
    --work-dir "${WORK_DIR}" \
    --mode all

echo ""
echo ">>> Evaluation complete.  Results saved to: ${WORK_DIR}"
