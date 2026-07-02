#!/usr/bin/env bash
################################################################################
# Full Attack Matrix Regression Test with Optimal Parameters
#
# Tests MATL/RB-BFT system across all attack types, distributions, and ratios
# with the optimized label skew parameters discovered through tuning.
#
# Optimal Configuration (validated 2025-10-28):
#   - BEHAVIOR_RECOVERY_THRESHOLD=2
#   - BEHAVIOR_RECOVERY_BONUS=0.12
#   - LABEL_SKEW_COS_MIN=-0.5
#   - LABEL_SKEW_COS_MAX=0.95
#   - Result: 3.55% FP, 91.7% detection (exceeds all targets)
################################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo "======================================================================"
echo -e "${BOLD}🧪 FULL ATTACK MATRIX REGRESSION TEST${NC}"
echo "======================================================================"
echo ""
echo "Testing with OPTIMAL PARAMETERS (validated 2025-10-28):"
echo "  ✓ Behavior Recovery Threshold: 2"
echo "  ✓ Behavior Recovery Bonus: 0.12"
echo "  ✓ Label Skew Cosine Min: -0.5"
echo "  ✓ Label Skew Cosine Max: 0.95"
echo ""
echo "Expected Performance:"
echo "  ✓ IID: >95% detection, <5% FP"
echo "  ✓ Label Skew: >90% detection, <5% FP (validated: 3.55% avg)"
echo ""
echo "======================================================================"
echo ""

# Set optimal parameters
export BEHAVIOR_RECOVERY_THRESHOLD=2
export BEHAVIOR_RECOVERY_BONUS=0.12
export LABEL_SKEW_COS_MIN=-0.5
export LABEL_SKEW_COS_MAX=0.95

# Set label skew configuration
export LABEL_SKEW_ALPHA=${LABEL_SKEW_ALPHA:-0.2}
export LABEL_SKEW_COMMITTEE_PERCENTILE=${LABEL_SKEW_COMMITTEE_PERCENTILE:-8}

# ML Detector configuration
# DISABLED: Feature mismatch with committee consensus features (7 vs 5)
# Our PoGQ + cosine + behavior recovery system achieves 3.55% FP without ML
export ML_DETECTOR_PATH=${ML_DETECTOR_PATH:-models/byzantine_detector_logged_v3}
export USE_ML_DETECTOR=${USE_ML_DETECTOR:-0}

# Attack and distribution configuration
export ATTACK_DISTRIBUTIONS=${ATTACK_DISTRIBUTIONS:-iid,label_skew}

# Enable BFT testing
export RUN_30_BFT=1

echo -e "${BLUE}📋 Configuration Summary:${NC}"
echo "  Attack Distributions: $ATTACK_DISTRIBUTIONS"
echo "  Label Skew Alpha: $LABEL_SKEW_ALPHA"
echo "  ML Detector: $ML_DETECTOR_PATH (enabled: $USE_ML_DETECTOR)"
echo "  Committee Percentile: $LABEL_SKEW_COMMITTEE_PERCENTILE"
echo ""

# Create results directory
mkdir -p results/bft-matrix
mkdir -p artifacts

echo -e "${BLUE}🚀 Starting Attack Matrix Test...${NC}"
echo ""

# Run the BFT matrix test
start_time=$(date +%s)

if nix develop -c python scripts/run_bft_matrix.py 2>&1 | tee /tmp/attack_matrix.log; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo ""
    echo "======================================================================"
    echo -e "${GREEN}✅ Attack Matrix Test Completed Successfully!${NC}"
    echo "======================================================================"
    echo ""
    echo "  Duration: ${duration}s"
    echo "  Results: results/bft-matrix/"
    echo "  Log: /tmp/attack_matrix.log"
    echo ""

    # Generate Prometheus metrics if script exists
    if [ -f scripts/export_bft_metrics.py ]; then
        echo -e "${BLUE}📊 Generating Prometheus metrics...${NC}"

        # Find the latest matrix JSON
        latest_matrix=$(ls -t results/bft-matrix/matrix_*.json 2>/dev/null | head -1)

        if [ -n "$latest_matrix" ]; then
            if python scripts/export_bft_metrics.py \
                --matrix "$latest_matrix" \
                --output artifacts/matl_metrics.prom; then
                echo -e "${GREEN}✅ Metrics exported to artifacts/matl_metrics.prom${NC}"
            else
                echo -e "${YELLOW}⚠️  Metrics export failed (non-fatal)${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  No matrix results found for metrics export${NC}"
        fi
    fi

    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    echo "  1. Review results/bft-matrix/latest_summary.md"
    echo "  2. Check artifacts/matl_metrics.prom for Prometheus metrics"
    echo "  3. Update Grafana dashboards if needed"
    echo "  4. Proceed with Holochain integration (Phase 1.5)"
    echo ""

    exit 0
else
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo ""
    echo "======================================================================"
    echo -e "${RED}❌ Attack Matrix Test FAILED${NC}"
    echo "======================================================================"
    echo ""
    echo "  Duration: ${duration}s"
    echo "  Log: /tmp/attack_matrix.log"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  1. Check /tmp/attack_matrix.log for details"
    echo "  2. Verify nix development environment: nix develop"
    echo "  3. Check if datasets are downloaded"
    echo "  4. Review LABEL_SKEW_OPTIMIZATION_COMPLETE.md"
    echo ""

    exit 1
fi
