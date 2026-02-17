#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${MATL_FEATURE_LOG:-${ROOT_DIR}/0TML/tests/results/ml_training_data.jsonl}"
MODEL_ROOT="${MATL_MODEL_ROOT:-${ROOT_DIR}/models}"
CONTAMINATION="${MATL_ANOMALY_CONTAMINATION:-0.08}"

if [[ ! -f "${LOG_PATH}" ]]; then
  echo "Error: feature log not found at ${LOG_PATH}" >&2
  exit 1
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${MODEL_ROOT}/byzantine_detector_${RUN_ID}"

echo "🔁 Retraining MATL detector from ${LOG_PATH}"
echo "   • Output directory: ${OUTPUT_DIR}"
echo "   • IsolationForest contamination: ${CONTAMINATION}"

nix develop --command poetry run python "${ROOT_DIR}/scripts/train_matl_detector.py" \
  --log "${LOG_PATH}" \
  --output "${OUTPUT_DIR}" \
  --anomaly-contamination "${CONTAMINATION}"

LATEST_LINK="${MODEL_ROOT}/byzantine_detector_latest"
ln -sfn "${OUTPUT_DIR}" "${LATEST_LINK}"
echo "✅ Updated ${LATEST_LINK} → ${OUTPUT_DIR}"
