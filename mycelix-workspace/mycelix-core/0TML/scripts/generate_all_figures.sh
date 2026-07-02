#!/usr/bin/env bash
"""
Master Figure Generation Script
================================

Generates all publication-quality figures for the paper.

Usage:
    ./scripts/generate_all_figures.sh
    ./scripts/generate_all_figures.sh --format pdf --dpi 600

Output:
    figures/ directory with:
    - ROC curves for each dataset+attack
    - DET curves (log-scale)
    - CM-Safe guard activation heatmaps
    - TTD histograms and comparisons

Author: Luminous Dynamics
Date: November 8, 2025
"""

set -euo pipefail

# Parse arguments
FORMAT="${1:-png}"
DPI="${2:-300}"
OUTPUT_DIR="figures"

echo "======================================================================"
echo "Zero-TrustML: Master Figure Generator"
echo "======================================================================"
echo ""
echo "Format: $FORMAT"
echo "DPI: $DPI"
echo "Output: $OUTPUT_DIR/"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for results
if [ ! -d "results" ] || [ -z "$(ls -A results/artifacts_* 2>/dev/null)" ]; then
    echo "❌ No experiment results found in results/"
    echo "   Run experiments first: python experiments/matrix_runner.py --config configs/sanity_slice.yaml"
    exit 1
fi

echo "✅ Found experiment results"
echo ""

# ==================================================================
# 1. ROC and DET Curves
# ==================================================================
echo "📈 Generating ROC and DET curves..."
python scripts/plot_roc_det_curves.py \
    --format "$FORMAT" \
    --dpi "$DPI" \
    --output-dir "$OUTPUT_DIR"

echo ""

# ==================================================================
# 2. Guard Activation Heatmaps
# ==================================================================
echo "🛡️  Generating CM-Safe guard activation heatmaps..."
python scripts/plot_guard_activations.py \
    --format "$FORMAT" \
    --output-dir "$OUTPUT_DIR"

echo ""

# ==================================================================
# 3. Time-to-Detection Histograms
# ==================================================================
echo "⏱️  Generating time-to-detection histograms..."
python scripts/plot_ttd_histograms.py \
    --format "$FORMAT" \
    --output-dir "$OUTPUT_DIR"

echo ""

# ==================================================================
# Summary
# ==================================================================
echo "======================================================================"
echo "✅ Figure generation complete!"
echo "======================================================================"
echo ""
echo "Output directory: $OUTPUT_DIR/"
echo ""
echo "Generated figures:"
ls -lh "$OUTPUT_DIR"/*."$FORMAT" | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Figure count: $(ls -1 "$OUTPUT_DIR"/*."$FORMAT" 2>/dev/null | wc -l)"
echo ""
echo "Next steps:"
echo "  1. Review figures: open $OUTPUT_DIR/"
echo "  2. Include in paper LaTeX: \\includegraphics{figures/roc_emnist_sign_flip.$FORMAT}"
echo "  3. Generate PDF versions: ./scripts/generate_all_figures.sh pdf 600"
echo ""
echo "======================================================================"
