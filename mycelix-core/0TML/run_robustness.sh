#!/usr/bin/env bash
# Batch runner for Non-IID Robustness Study
# =========================================
# Tests PoGQ v4.1 conformal prediction under varying data heterogeneity
# Total: 3 α values × 3 attacks × 2 seeds = 18 experiments

set -e

CONFIGS=(
    "configs/robustness_alpha_0_1.yaml"   # Extreme heterogeneity
    "configs/robustness_alpha_0_3.yaml"   # Moderate (baseline)
    "configs/robustness_alpha_1_0.yaml"   # Near-IID
)

echo "===================================="
echo "Non-IID Robustness Study"
echo "===================================="
echo "Testing conformal prediction under varying heterogeneity (α)"
echo "Total configs: ${#CONFIGS[@]}"
echo "Total experiments: ~18 (3 α × 3 attacks × 2 seeds)"
echo ""
echo "Hypothesis: Mondrian conformal maintains FPR ≤ α across all conditions"
echo "Expected: Per-bucket FPR ≤ 0.12 (α=0.10 + 2% margin) for all α values"
echo ""

for config in "${CONFIGS[@]}"; do
    echo "Running: $config"
    poetry run python matrix_runner.py --config "$config" --output results/
    echo "✓ Completed: $config"
    echo ""
done

echo "===================================="
echo "Robustness study complete!"
echo "===================================="
echo "Results saved to results/robustness_alpha_*"
echo ""
echo "Next steps:"
echo "  1. Analyze per-bucket FPR across α values"
echo "  2. Compare Mondrian vs global conformal performance"
echo "  3. Measure adaptive hybrid switching frequency"
echo "  4. Generate robustness plots for paper"
