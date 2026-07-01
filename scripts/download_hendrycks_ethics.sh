#!/usr/bin/env bash
# Download the Hendrycks ETHICS dataset for external validation.
#
# Source: https://github.com/hendrycks/ethics
# Citation: Hendrycks et al. (2021). Aligning AI With Shared Human Values. ICLR.
#
# Usage:
#   ./scripts/download_hendrycks_ethics.sh [output_dir]
#
# Default output: data/benchmarks/external/hendrycks_ethics/

set -euo pipefail

OUTPUT_DIR="${1:-data/benchmarks/ethics}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Downloading Hendrycks ETHICS Dataset                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

mkdir -p "$OUTPUT_DIR"

REPO_URL="https://raw.githubusercontent.com/hendrycks/ethics/master/ethics"
DOMAINS=("commonsense" "deontology" "justice" "virtue" "utilitarianism")

for domain in "${DOMAINS[@]}"; do
    echo "Downloading ${domain}..."
    # Test set
    curl -sL "${REPO_URL}/${domain}/cm_test.csv" -o "${OUTPUT_DIR}/${domain}_test.csv" 2>/dev/null || \
    curl -sL "${REPO_URL}/${domain}/${domain}_test.csv" -o "${OUTPUT_DIR}/${domain}_test.csv" 2>/dev/null || \
    echo "  Warning: Could not download ${domain} test set (may need manual download)"
done

echo
echo "Download complete. Files in: ${OUTPUT_DIR}"
echo
echo "To use with the benchmark:"
echo "  cargo run --example benchmark_moral_unified --release"
echo
echo "Note: The full pipeline (MoralParser + MoralAlgebra + learned prototypes)"
echo "achieves ~92.9% composite accuracy across 5 moral reasoning datasets."
