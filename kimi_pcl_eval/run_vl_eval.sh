#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_vl_eval.sh — Multimodal evaluation via VLMEvalKit
# ──────────────────────────────────────────────────────────────
#
# Evaluates Kimi-K2 1T on 3 multimodal benchmarks:
#   MMBench_DEV_EN  (image-basic)
#   MMMU_DEV_VAL    (image-hard)
#   Video-MME       (video)
#
# Usage:
#   bash run_vl_eval.sh --model_path /path/to/model --tp_size 32
#   bash run_vl_eval.sh   # uses defaults
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/wf/ckpt/iter_0000900_test/}"
TP_SIZE="${TP_SIZE:-8}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/outputs_vl}"
VLMEVAL_DIR="${VLMEVAL_DIR:-${SCRIPT_DIR}/VLMEvalKit}"

# ── Parse CLI arguments ───────────────────────────────────────
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_path   PATH  HuggingFace model directory  (default: ${MODEL_PATH})"
    echo "  --tp_size      N     Tensor-parallel size          (default: ${TP_SIZE})"
    echo "  --work_dir     DIR   Output directory               (default: ${WORK_DIR})"
    echo "  --vlmeval_dir  DIR   VLMEvalKit repo directory      (default: ${VLMEVAL_DIR})"
    echo "  -h, --help           Show this message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)   MODEL_PATH="$2";   shift 2 ;;
        --tp_size)      TP_SIZE="$2";      shift 2 ;;
        --work_dir)     WORK_DIR="$2";     shift 2 ;;
        --vlmeval_dir)  VLMEVAL_DIR="$2";  shift 2 ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

export MODEL_PATH TP_SIZE

# ── Clone VLMEvalKit if not present ───────────────────────────
if [[ ! -d "${VLMEVAL_DIR}" ]]; then
    echo ">>> Cloning VLMEvalKit ..."
    git clone https://github.com/open-compass/VLMEvalKit.git "${VLMEVAL_DIR}"
fi

# ── Install the wrapper into VLMEvalKit ───────────────────────
echo ">>> Installing Kimi-K2 wrapper into VLMEvalKit ..."
cp "${SCRIPT_DIR}/vlmeval_wrapper/kimi_k2.py" "${VLMEVAL_DIR}/vlmeval/vlm/kimi_k2.py"

# Register in __init__.py if not already present
INIT_FILE="${VLMEVAL_DIR}/vlmeval/vlm/__init__.py"
if ! grep -q "kimi_k2" "${INIT_FILE}"; then
    echo "" >> "${INIT_FILE}"
    echo "from .kimi_k2 import KimiK2  # noqa: F401" >> "${INIT_FILE}"
    echo ">>> Registered KimiK2 in vlmeval/vlm/__init__.py"
fi

# Register in config.py if not already present
CONFIG_FILE="${VLMEVAL_DIR}/vlmeval/config.py"
if ! grep -q "Kimi-K2-1T" "${CONFIG_FILE}"; then
    # Insert into supported_VLM dict — append before the closing brace
    python -c "
import re, os

config_path = '${CONFIG_FILE}'
model_path = '${MODEL_PATH}'
tp_size = ${TP_SIZE}

with open(config_path, 'r') as f:
    content = f.read()

# Check if functools.partial is already imported
if 'from functools import partial' not in content:
    content = 'from functools import partial\n' + content

# Add KimiK2 import and entry
if 'KimiK2' not in content:
    # Add import
    content = content.replace(
        'from functools import partial',
        'from functools import partial\n'
    )

    # Find supported_VLM dict and append entry
    entry = f\"\"\"
    'Kimi-K2-1T': partial(KimiK2, model_path='{model_path}', tp_size={tp_size}),\"\"\"

    if 'supported_VLM' in content:
        # Insert before the last closing brace of supported_VLM
        idx = content.rfind('}')
        if idx != -1:
            content = content[:idx] + entry + '\n' + content[idx:]

    with open(config_path, 'w') as f:
        f.write(content)
    print('>>> Registered Kimi-K2-1T in vlmeval/config.py')
"
fi

# ── Print configuration summary ───────────────────────────────
echo "============================================================"
echo "  Kimi-K2 1T  —  VLMEvalKit Multimodal Evaluation"
echo "============================================================"
echo "  Model path    : ${MODEL_PATH}"
echo "  TP size       : ${TP_SIZE}"
echo "  Work dir      : ${WORK_DIR}"
echo "  VLMEvalKit    : ${VLMEVAL_DIR}"
echo "  Benchmarks    : MMBench_DEV_EN  MMMU_DEV_VAL  Video-MME"
echo "============================================================"
echo ""

# ── Run evaluation ────────────────────────────────────────────
echo ">>> Starting multimodal evaluation ..."
cd "${VLMEVAL_DIR}"
python run.py \
    --data MMBench_DEV_EN MMMU_DEV_VAL Video-MME \
    --model Kimi-K2-1T \
    --work-dir "${WORK_DIR}" \
    --verbose

echo ""
echo ">>> Evaluation complete.  Results saved to: ${WORK_DIR}"
