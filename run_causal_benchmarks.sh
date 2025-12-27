#!/usr/bin/env bash
#
# Run Enhancement #4 Causal Reasoning Benchmarks
#
# This script runs comprehensive performance benchmarks for all four phases
# of Enhancement #4 and saves results for analysis.

set -euo pipefail

echo "🔬 Running Enhancement #4: Causal Reasoning Benchmarks"
echo "======================================================"
echo ""
echo "This will benchmark:"
echo "  - Phase 1: Causal Intervention"
echo "  - Phase 2: Counterfactual Reasoning"
echo "  - Phase 3: Action Planning"
echo "  - Phase 4: Causal Explanation"
echo "  - Integrated Workflows"
echo "  - Scaling Analysis"
echo ""

# Create results directory
RESULTS_DIR="benchmark_results"
mkdir -p "$RESULTS_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/causal_reasoning_$TIMESTAMP.txt"

echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Run benchmarks
cargo bench --bench causal_reasoning_benchmark 2>&1 | tee "$RESULTS_FILE"

echo ""
echo "✅ Benchmarks complete!"
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "📊 To view HTML report:"
echo "   firefox target/criterion/report/index.html"
