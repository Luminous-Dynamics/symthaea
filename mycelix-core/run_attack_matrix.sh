#!/usr/bin/env bash
set -euo pipefail

export ML_DETECTOR_PATH="${ML_DETECTOR_PATH:-models/byzantine_detector_logged}"
export ML_FEATURE_LOG="${ML_FEATURE_LOG:-0TML/tests/results/ml_training_data.jsonl}"
export USE_ML_DETECTOR="${USE_ML_DETECTOR:-1}"

nix develop --command poetry run python scripts/run_attack_matrix.py "$@"
nix develop --command poetry run python scripts/generate_bft_matrix.py
nix develop --command poetry run python 0TML/scripts/plot_bft_matrix.py
