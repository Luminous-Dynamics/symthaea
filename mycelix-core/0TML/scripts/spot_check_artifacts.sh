#!/usr/bin/env bash
# Artifact Spot-Check: Verify CI Gates on Latest Results
# =========================================================
#
# Checks the 3 critical CI gates as artifacts land:
#   1. Conformal FPR ≤ 0.12 for all buckets
#   2. EMA stability (flap count ≤ 5)
#   3. CM-Safe guard activation coverage
#
# Usage:
#   ./scripts/spot_check_artifacts.sh                    # Check latest artifact
#   ./scripts/spot_check_artifacts.sh results/artifacts_20251108_160000  # Specific dir
#   watch -n 300 ./scripts/spot_check_artifacts.sh       # Monitor every 5 min

set -euo pipefail

# Find latest artifact directory
if [ $# -eq 0 ]; then
    ART=$(ls -dt /srv/luminous-dynamics/Mycelix-Core/0TML/results/artifacts_* 2>/dev/null | head -1)
    if [ -z "$ART" ]; then
        echo "❌ No artifact directories found"
        exit 1
    fi
else
    ART="$1"
fi

echo "======================================================================="
echo "🔍 Spot-Check: $(basename "$ART")"
echo "======================================================================="
echo ""

# Check if directory exists
if [ ! -d "$ART" ]; then
    echo "❌ Directory not found: $ART"
    exit 1
fi

# Gate 1: Conformal FPR (all buckets ≤ 0.12)
echo "Gate 1: Conformal FPR Guarantee"
echo "--------------------------------"
if [ -f "$ART/per_bucket_fpr.json" ]; then
    violations=$(jq -r '.buckets[] | select(.fpr > 0.12) | "\(.profile): FPR=\(.fpr) n=\(.n)"' "$ART/per_bucket_fpr.json")
    if [ -z "$violations" ]; then
        echo "✅ All buckets ≤ 0.12 FPR"
    else
        echo "⚠️  FPR violations found:"
        echo "$violations"

        # Check if violations are on small-n buckets
        small_n=$(echo "$violations" | grep -E "n=[0-9]$|n=1[0-9]$" || true)
        if [ -n "$small_n" ]; then
            echo ""
            echo "💡 Note: Violations on small-n buckets (<20 samples)"
            echo "    Consider Mondrian profile merging for next slice"
        fi
    fi

    # Summary stats
    echo ""
    jq -r '.summary | "Total buckets: \(.total_buckets)\nMean FPR: \(.mean_fpr)\nMax FPR: \(.max_fpr)"' "$ART/per_bucket_fpr.json" 2>/dev/null || echo "No summary stats available"
else
    echo "⚠️  per_bucket_fpr.json not found"
fi

echo ""

# Gate 2: EMA Stability (flap count ≤ 5)
echo "Gate 2: EMA Stability (Flap Count)"
echo "-----------------------------------"
if [ -f "$ART/ema_vs_raw.json" ]; then
    high_flap=$(jq -r '.experiments[] | select(.flap_count > 5) | "\(.experiment_id): flaps=\(.flap_count) (raw: \(.flap_count_raw))"' "$ART/ema_vs_raw.json")
    if [ -z "$high_flap" ]; then
        echo "✅ All experiments ≤ 5 flaps with EMA"
    else
        echo "⚠️  High flap count found:"
        echo "$high_flap"
    fi

    # Variance reduction
    echo ""
    jq -r '.summary | "EMA variance reduction: \(.variance_reduction_pct)%\nMean flap count (EMA): \(.mean_flap_count)\nMean flap count (raw): \(.mean_flap_count_raw)"' "$ART/ema_vs_raw.json" 2>/dev/null || echo "No summary stats available"
else
    echo "⚠️  ema_vs_raw.json not found"
fi

echo ""

# Gate 3: CM-Safe Guard Activation Coverage
echo "Gate 3: Coordinate Median Safe Guards"
echo "--------------------------------------"
if [ -f "$ART/coord_median_diagnostics.json" ]; then
    # Check for guard activations
    cm_safe=$(jq -r '.experiments[] | select(.detector == "coord_median_safe")' "$ART/coord_median_diagnostics.json")

    if [ -z "$cm_safe" ]; then
        echo "ℹ️  No coord_median_safe experiments in this artifact"
    else
        # Check guard coverage
        zero_guards=$(jq -r '.experiments[] | select(.detector == "coord_median_safe") | select(.guard_activations.min_clients == 0 and .guard_activations.norm_clamp == 0 and .guard_activations.direction_check == 0) | .experiment_id' "$ART/coord_median_diagnostics.json")

        if [ -z "$zero_guards" ]; then
            echo "✅ Guards activated in all CM-Safe experiments"
        else
            echo "⚠️  Zero guard activations in:"
            echo "$zero_guards"
        fi

        # Show guard distribution
        echo ""
        echo "Guard activation counts:"
        jq -r '.experiments[] | select(.detector == "coord_median_safe") | "\(.experiment_id):\n  min_clients: \(.guard_activations.min_clients)\n  norm_clamp: \(.guard_activations.norm_clamp)\n  direction_check: \(.guard_activations.direction_check)"' "$ART/coord_median_diagnostics.json" | head -20
    fi
else
    echo "⚠️  coord_median_diagnostics.json not found"
fi

echo ""
echo "======================================================================="
echo "Artifact check complete"
echo "======================================================================="
