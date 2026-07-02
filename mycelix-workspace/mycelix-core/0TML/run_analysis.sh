#!/usr/bin/env bash
# Automated Analysis Pipeline for FL Experiments
# Usage: ./run_analysis.sh [results_file.json]

set -e

echo "🔬 Hybrid ZeroTrustML Analysis Pipeline"
echo "===================================="
echo ""

# Find the most recent results file if not specified
if [ -z "$1" ]; then
    RESULTS_FILE=$(find results/iid -name "mnist_iid_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)

    if [ -z "$RESULTS_FILE" ]; then
        echo "❌ No results files found in results/iid/"
        echo "   Run experiment first: python -u experiments/runner.py --config experiments/configs/mnist_iid.yaml"
        exit 1
    fi

    echo "📊 Using most recent results: $RESULTS_FILE"
else
    RESULTS_FILE="$1"
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "❌ Results file not found: $RESULTS_FILE"
        exit 1
    fi
    echo "📊 Using specified results: $RESULTS_FILE"
fi

echo ""

# Extract experiment name and create output directory
EXPERIMENT_NAME=$(basename "$RESULTS_FILE" .json | sed 's/_[0-9]\{8\}_[0-9]\{6\}//')
OUTPUT_DIR="results/analysis/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "📁 Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Run analysis with Python
echo ""
echo "🔍 Running analysis..."
python -u experiments/utils/analyze_results.py \
    --results "$RESULTS_FILE" \
    --output "$OUTPUT_DIR" \
    --format png \
    --dpi 300

# Check if analysis succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis complete!"
    echo ""
    echo "📈 Generated plots:"
    ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "   No PNG files generated"
    echo ""
    echo "📋 Generated tables:"
    ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "   No CSV files generated"
    echo ""
    echo "📊 View results:"
    echo "   cd $OUTPUT_DIR"
    echo "   open *.png  # or: xdg-open *.png"
else
    echo ""
    echo "❌ Analysis failed!"
    exit 1
fi

# Generate summary report
echo ""
echo "📝 Summary Report:"
echo "=================="

if [ -f "$OUTPUT_DIR/comparison_table.csv" ]; then
    echo ""
    cat "$OUTPUT_DIR/comparison_table.csv" | column -t -s,
    echo ""
fi

echo "🎉 All done! Results saved to: $OUTPUT_DIR"
