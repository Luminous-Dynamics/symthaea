#!/usr/bin/env bash
#
# Run Byzantine Attack Robustness Experiments
#
# This script runs comprehensive tests of all 7 Byzantine attack types
# against 5 aggregation baselines (FedAvg + 4 Byzantine-robust methods).
#
# Total experiments: 7 attacks × 5 baselines = 35 runs
# Estimated time: ~2-3 hours on GPU (3-4 minutes per run)
#
# Results will show which attacks are most effective and which defenses
# are most robust - critical for validating security claims.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="experiments/configs/mnist_byzantine_attacks.yaml"
RESULTS_DIR="results/byzantine"
LOG_DIR="/tmp"

# Attack types to test
ATTACKS=(
    "gaussian_noise"
    "sign_flip"
    "label_flip"
    "targeted_poison"
    "model_replacement"
    "adaptive"
    "sybil"
)

# Baselines to test
BASELINES=(
    "fedavg"
    "krum"
    "multikrum"
    "bulyan"
    "median"
)

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Byzantine Attack Robustness Experiments                        ║${NC}"
echo -e "${BLUE}║  Testing 7 attacks × 5 baselines = 35 experiments              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check GPU availability
if nix develop --command python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')" 2>/dev/null | grep -q "GPU: True"; then
    echo -e "${GREEN}✅ GPU available - experiments will run faster${NC}"
else
    echo -e "${YELLOW}⚠️  No GPU detected - experiments will run on CPU (slower)${NC}"
fi

# Function to launch a single attack×baseline experiment
launch_attack_experiment() {
    local attack=$1
    local baseline=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local exp_name="byzantine_${attack}_${baseline}_${timestamp}"
    local log_file="${LOG_DIR}/${exp_name}.log"

    echo -e "${BLUE}🚀 Starting:${NC} Attack=${YELLOW}${attack}${NC} vs Defense=${GREEN}${baseline}${NC}"

    # Create temporary config for this specific attack×baseline combination
    local temp_config="/tmp/${exp_name}.yaml"
    cat "$CONFIG_FILE" | \
        sed "s/attack_types:/current_attack: ${attack}  # Generated\n  attack_types_disabled:/g" | \
        sed "s/baselines:/current_baseline: ${baseline}  # Generated\n  baselines_disabled:/g" \
        > "$temp_config"

    # Launch experiment
    nix develop --command python3 -u experiments/runner.py \
        --config "$temp_config" \
        --attack "$attack" \
        --baseline "$baseline" \
        &> "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Completed:${NC} $attack vs $baseline"

        # Extract key metrics from log
        local test_acc=$(grep "Test Acc" "$log_file" | tail -1 | awk '{print $NF}')
        echo -e "   Final accuracy: ${test_acc}"
    else
        echo -e "${RED}❌ Failed:${NC} $attack vs $baseline (check $log_file)"
        return 1
    fi

    rm -f "$temp_config"
}

# Main experiment loop
echo -e "\n${BLUE}Starting Byzantine attack experiments...${NC}\n"

TOTAL_EXPERIMENTS=$((${#ATTACKS[@]} * ${#BASELINES[@]}))
CURRENT=0
FAILED=0

START_TIME=$(date +%s)

for attack in "${ATTACKS[@]}"; do
    for baseline in "${BASELINES[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}Experiment ${CURRENT}/${TOTAL_EXPERIMENTS}${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

        if ! launch_attack_experiment "$attack" "$baseline"; then
            FAILED=$((FAILED + 1))
        fi

        # Progress indicator
        PROGRESS=$((CURRENT * 100 / TOTAL_EXPERIMENTS))
        echo -e "${GREEN}Progress: ${PROGRESS}%${NC} (${CURRENT}/${TOTAL_EXPERIMENTS} complete, ${FAILED} failed)"
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Byzantine Attack Experiments Complete!                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${GREEN}✅ Completed:${NC} $((TOTAL_EXPERIMENTS - FAILED))/${TOTAL_EXPERIMENTS} experiments"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ Failed:${NC} ${FAILED} experiments"
fi
echo -e "${BLUE}⏱️  Duration:${NC} ${HOURS}h ${MINUTES}m"
echo
echo -e "${BLUE}📊 Results saved to:${NC} ${RESULTS_DIR}/"
echo -e "${BLUE}📋 Logs saved to:${NC} ${LOG_DIR}/byzantine_*.log"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Analyze results: python experiments/analyze_byzantine_results.py"
echo "  2. Generate attack comparison table"
echo "  3. Compare defense effectiveness across attacks"
echo

exit $FAILED
