#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# download_datasets.sh — Pre-download OpenCompass datasets
# ──────────────────────────────────────────────────────────────
#
# Run this script on a machine WITH internet access, then copy
# the resulting data/ directory to the offline evaluation node.
#
# Usage:
#   bash download_datasets.sh                  # default: ./data
#   bash download_datasets.sh --data_dir /ssd/datasets
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

# ── Parse CLI arguments ───────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--data_dir DIR]"
            echo "  Downloads the 6 evaluation datasets for offline use."
            echo "  Default output: ${DATA_DIR}"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${DATA_DIR}"

echo "============================================================"
echo "  OpenCompass Dataset Downloader"
echo "============================================================"
echo "  Output directory : ${DATA_DIR}"
echo "============================================================"
echo ""

# ── Download OpenCompass official data package ────────────────
OC_DATA_URL="https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip"
OC_DATA_ZIP="${DATA_DIR}/OpenCompassData-core.zip"

echo ">>> Downloading OpenCompass official data package ..."
if command -v wget &>/dev/null; then
    wget -c -O "${OC_DATA_ZIP}" "${OC_DATA_URL}"
elif command -v curl &>/dev/null; then
    curl -L -C - -o "${OC_DATA_ZIP}" "${OC_DATA_URL}"
else
    echo "ERROR: neither wget nor curl found" >&2
    exit 1
fi

echo ">>> Extracting ..."
# The archive contains a top-level data/ folder; extract its
# contents directly into DATA_DIR to avoid double nesting.
TMP_DIR="$(mktemp -d)"
unzip -o -q "${OC_DATA_ZIP}" -d "${TMP_DIR}"
mv "${TMP_DIR}"/data/* "${DATA_DIR}/"
rm -rf "${TMP_DIR}" "${OC_DATA_ZIP}"

# ── Download TruthfulQA (not in core package) ─────────────────
echo ">>> Downloading TruthfulQA ..."
mkdir -p "${DATA_DIR}/truthfulqa"
if command -v wget &>/dev/null; then
    wget -q -O "${DATA_DIR}/truthfulqa/TruthfulQA.csv" \
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
else
    curl -sL -o "${DATA_DIR}/truthfulqa/TruthfulQA.csv" \
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
fi

# ── Verify ────────────────────────────────────────────────────
echo ""
echo ">>> Dataset directory contents:"
ls -1 "${DATA_DIR}/"
echo ""
echo "============================================================"
echo "  Done.  Datasets are ready at: ${DATA_DIR}"
echo "============================================================"
