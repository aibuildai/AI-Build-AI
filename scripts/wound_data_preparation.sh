#!/bin/bash
# Data preparation script for wound segmentation dataset
# Downloads from https://github.com/uwm-bigdata/wound-segmentation
# and extracts only the train/ and validation/ folders.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/uwm-bigdata/wound-segmentation.git"
TEMP_DIR="${SCRIPT_DIR}/.tmp_clone"
SOURCE_BASE="data/Foot Ulcer Segmentation Challenge"

# Check if data already exists
if [ -d "${SCRIPT_DIR}/train/images" ] && [ -d "${SCRIPT_DIR}/train/labels" ] \
   && [ -d "${SCRIPT_DIR}/validation/images" ] && [ -d "${SCRIPT_DIR}/validation/labels" ]; then
    echo "Data already exists in ${SCRIPT_DIR}. Skipping download."
    echo "  train/images:      $(ls "${SCRIPT_DIR}/train/images" | wc -l) files"
    echo "  train/labels:      $(ls "${SCRIPT_DIR}/train/labels" | wc -l) files"
    echo "  validation/images: $(ls "${SCRIPT_DIR}/validation/images" | wc -l) files"
    echo "  validation/labels: $(ls "${SCRIPT_DIR}/validation/labels" | wc -l) files"
    exit 0
fi

echo "==> Cloning repository (sparse checkout)..."
git clone --depth 1 --filter=blob:none --sparse "${REPO_URL}" "${TEMP_DIR}"
cd "${TEMP_DIR}"
git sparse-checkout set "data/Foot Ulcer Segmentation Challenge/train" \
                        "data/Foot Ulcer Segmentation Challenge/validation"

echo "==> Copying train/ and validation/ folders..."
cp -r "${SOURCE_BASE}/train" "${SCRIPT_DIR}/train"
cp -r "${SOURCE_BASE}/validation" "${SCRIPT_DIR}/validation"

echo "==> Cleaning up temporary clone..."
cd "${SCRIPT_DIR}"
rm -rf "${TEMP_DIR}"

echo "==> Done. Data prepared in ${SCRIPT_DIR}"
echo "  train/images:      $(ls "${SCRIPT_DIR}/train/images" | wc -l) files"
echo "  train/labels:      $(ls "${SCRIPT_DIR}/train/labels" | wc -l) files"
echo "  validation/images: $(ls "${SCRIPT_DIR}/validation/images" | wc -l) files"
echo "  validation/labels: $(ls "${SCRIPT_DIR}/validation/labels" | wc -l) files"
