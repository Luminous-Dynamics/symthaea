#!/usr/bin/env bash
# Batch runner for PoGQ v4.1 ablation study
# =========================================
# Runs all 6 component ablation configs
# Total: 6 configs × 3 attacks × 2 seeds = 36 experiments

set -e

CONFIGS=(
    "configs/ablation_no_pca.yaml"
    "configs/ablation_no_conformal.yaml"
    "configs/ablation_no_mondrian.yaml"
    "configs/ablation_no_hybrid.yaml"
    "configs/ablation_no_ema.yaml"
    "configs/ablation_no_hysteresis.yaml"
)

echo "==================================="
echo "PoGQ v4.1 Ablation Study"
echo "==================================="
echo "Total configs: ${#CONFIGS[@]}"
echo "Total experiments: ~36 (6 configs × 3 attacks × 2 seeds)"
echo ""

for config in "${CONFIGS[@]}"; do
    echo "Running: $config"
    poetry run python matrix_runner.py --config "$config" --output results/
    echo "✓ Completed: $config"
    echo ""
done

echo "==================================="
echo "Ablation study complete!"
echo "==================================="
echo "Results saved to results/ablation_*"
echo ""
echo "Next steps:"
echo "  1. Analyze results: poetry run python tools/summarize_results.py results/ablation_*"
echo "  2. Generate comparison table for paper"
echo "  3. Identify critical vs non-critical components"
