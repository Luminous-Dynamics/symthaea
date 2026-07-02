#!/usr/bin/env bash
################################################################################
# BFT Matrix Testing Script - Week 3 Priority 2
# Tests Byzantine resistance at progressive percentages: 0%, 10%, 20%, 30%, 40%, 50%
#
# For each Byzantine percentage, captures:
# - Detection rate
# - False positive rate
# - Precision
# - Recall
# - Time to recovery
# - Entropy metrics
# - Coherence metrics
#
# Generates: tests/results/bft_results_{percent}_byz.json
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "🧪 WEEK 3 PRIORITY 2: BFT MATRIX TESTING (0-50% Byzantine)"
echo "======================================================================"
echo ""
echo "Testing Byzantine resistance at multiple percentages:"
echo "  - 0%  (baseline - all honest)"
echo "  - 10% (2/20 Byzantine)"
echo "  - 20% (4/20 Byzantine)"
echo "  - 30% (6/20 Byzantine)"
echo "  - 33% (7/20 Byzantine - classical BFT limit)"
echo "  - 40% (8/20 Byzantine - exceeds classical limit)"
echo "  - 50% (10/20 Byzantine - majority attack)"
echo ""

# Create results directory if it doesn't exist
mkdir -p tests/results

# Array of Byzantine percentages to test
PERCENTAGES=(0 10 20 30 33 40 50)

# Activate Nix development environment
echo -e "${BLUE}🔄 Activating Nix development environment...${NC}"
export NIX_PATH="nixpkgs=/nix/var/nix/profiles/per-user/root/channels/nixos"

# Function to run BFT test for a specific percentage
run_bft_test() {
    local byz_percent=$1
    local total_nodes=20
    local byz_nodes=$(( (total_nodes * byz_percent) / 100 ))
    local honest_nodes=$(( total_nodes - byz_nodes ))

    echo ""
    echo "======================================================================"
    echo -e "${YELLOW}📊 Testing ${byz_percent}% Byzantine (${byz_nodes}/${total_nodes} nodes)${NC}"
    echo "======================================================================"
    echo ""

    # Create configuration file for this test
    local config_file="byzantine-configs/attack-strategies-${byz_percent}pct.json"

    # Generate node assignments
    local honest_ids=$(seq -s "," 0 $((honest_nodes - 1)))
    local byz_ids=""
    if [ $byz_nodes -gt 0 ]; then
        byz_ids=$(seq -s "," $honest_nodes $((total_nodes - 1)))
    fi

    # Create configuration
    cat > "$config_file" <<EOF
{
  "byzantine_configuration": {
    "total_nodes": $total_nodes,
    "honest_nodes": $honest_nodes,
    "byzantine_nodes": $byz_nodes,
    "byzantine_percentage": $byz_percent,
    "exceeds_classical_bft": $([ $byz_percent -gt 33 ] && echo "true" || echo "false")
  },
  "node_assignments": {
    "honest": [${honest_ids}],
    "byzantine": [${byz_ids}]
  },
  "attack_strategies": {
EOF

    # Add attack strategies based on number of Byzantine nodes
    # Build strategies array to avoid trailing comma issues
    strategies=""

    if [ $byz_nodes -ge 2 ]; then
        strategies+='"label_flipping": {
      "nodes": ['$honest_nodes', '$(($honest_nodes + 1))'],
      "severity": "medium",
      "detectability": "high"
    }'
    fi

    if [ $byz_nodes -ge 4 ]; then
        [ -n "$strategies" ] && strategies+=","
        strategies+='
    "gradient_reversal": {
      "nodes": ['$(($honest_nodes + 2))', '$(($honest_nodes + 3))'],
      "severity": "high",
      "detectability": "high"
    }'
    fi

    if [ $byz_nodes -ge 6 ]; then
        [ -n "$strategies" ] && strategies+=","
        strategies+='
    "random_noise": {
      "nodes": ['$(($honest_nodes + 4))', '$(($honest_nodes + 5))'],
      "severity": "medium",
      "detectability": "medium"
    }'
    fi

    if [ $byz_nodes -ge 8 ]; then
        [ -n "$strategies" ] && strategies+=","
        strategies+='
    "sybil_coordination": {
      "nodes": ['$(($honest_nodes + 6))', '$(($honest_nodes + 7))'],
      "severity": "critical",
      "detectability": "low"
    }'
    fi

    # Write strategies to file (with proper indentation)
    if [ -n "$strategies" ]; then
        echo "    $strategies" >> "$config_file"
    fi

    # Close JSON
    cat >> "$config_file" <<EOF
  },
  "validation_rules": {
    "1_dimension_validation": { "enabled": true },
    "2_magnitude_bounds": { "enabled": true, "threshold": 3.0 },
    "3_statistical_outlier_pogq": { "enabled": true, "threshold": 0.5 },
    "4_temporal_consistency": { "enabled": true, "window": 3 },
    "5_reputation_scoring": { "enabled": true, "threshold": 0.3, "decay_rate": 0.2 },
    "6_cross_validation": { "enabled": true },
    "7_gradient_noise": { "enabled": true },
    "8_pattern_anomaly": { "enabled": false }
  }
}
EOF

    echo -e "${BLUE}✅ Configuration created: $config_file${NC}"

    # Run the integrated test
    local output_file="tests/results/bft_results_${byz_percent}_byz.json"
    local log_file="/tmp/bft_test_${byz_percent}pct.log"

    echo -e "${BLUE}🚀 Running integrated Byzantine test...${NC}"

    # Run test with timeout and capture output
    if timeout 600 nix develop --command python tests/byzantine/integrated_byzantine_test.py \
        --config "$config_file" \
        --output "$output_file" \
        --rounds 5 \
        2>&1 | tee "$log_file"; then

        echo -e "${GREEN}✅ Test completed successfully!${NC}"
        echo -e "${GREEN}📊 Results saved to: $output_file${NC}"
        echo -e "${BLUE}📝 Log saved to: $log_file${NC}"

        # Display summary if results exist
        if [ -f "$output_file" ]; then
            echo ""
            echo "Summary for ${byz_percent}% Byzantine:"
            python3 -c "
import json
with open('$output_file') as f:
    results = json.load(f)
    print(f\"  Detection Rate: {results.get('detection_rate', 0):.1%}\")
    print(f\"  False Positive Rate: {results.get('false_positive_rate', 0):.1%}\")
    print(f\"  Accuracy: {results.get('accuracy', 0):.1%}\")
    print(f\"  Precision: {results.get('precision', 0):.1%}\")
"
        fi
    else
        echo -e "${RED}❌ Test failed or timed out!${NC}"
        echo -e "${YELLOW}⚠️  Check log file: $log_file${NC}"
        return 1
    fi
}

# Run tests for each percentage
echo -e "${BLUE}Starting BFT matrix testing...${NC}"
echo ""

FAILED_TESTS=()

for percent in "${PERCENTAGES[@]}"; do
    if ! run_bft_test "$percent"; then
        FAILED_TESTS+=("$percent")
        echo -e "${RED}⚠️  ${percent}% test failed, continuing with remaining tests...${NC}"
    fi

    # Small delay between tests
    sleep 2
done

# Final summary
echo ""
echo "======================================================================"
echo "🎯 BFT MATRIX TESTING COMPLETE"
echo "======================================================================"
echo ""

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ All tests completed successfully!${NC}"
    echo ""
    echo "📊 Results files generated:"
    for percent in "${PERCENTAGES[@]}"; do
        local output_file="tests/results/bft_results_${percent}_byz.json"
        if [ -f "$output_file" ]; then
            echo -e "  ${GREEN}✓${NC} $output_file"
        fi
    done
else
    echo -e "${YELLOW}⚠️  Some tests failed:${NC}"
    for percent in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}✗${NC} ${percent}% Byzantine test"
    done
fi

echo ""
echo "Next steps:"
echo "  1. Review results in tests/results/bft_results_*.json"
echo "  2. Generate visualization plots"
echo "  3. Create BYZANTINE_RESISTANCE_TESTS.md report"
echo ""

exit 0
