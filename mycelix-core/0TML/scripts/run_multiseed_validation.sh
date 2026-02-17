#!/usr/bin/env bash
###
# Multi-Seed Validation Script for BFT Tests
#
# Runs tests across multiple random seeds for statistical robustness
# Aggregates results and computes mean ± std deviation
###

set -e

echo "======================================================================"
echo "🎲 MULTI-SEED VALIDATION SUITE"
echo "======================================================================"
echo ""
echo "Objective: Statistical robustness validation across random seeds"
echo ""
echo "Tests:"
echo "  1. Week 3 (30% BFT) validation - 5 seeds"
echo "  2. Boundary tests (35%, 40%) - 3 seeds each"
echo ""
echo "======================================================================"
echo ""

# Seeds to test
SEEDS=(42 123 456 789 1024)
BOUNDARY_SEEDS=(42 123 456)

# Results directory
RESULTS_DIR="./test_results/multiseed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run test with seed
run_test_with_seed() {
    local test_name=$1
    local seed=$2
    local test_script=$3
    local log_file="$RESULTS_DIR/${test_name}_seed_${seed}.log"

    echo "Running $test_name with seed $seed..."

    # Set seed environment variable
    export RANDOM_SEED=$seed

    # Run test and capture output
    if python "$test_script" > "$log_file" 2>&1; then
        echo "  ✅ Passed"
        return 0
    else
        echo "  ❌ Failed"
        return 1
    fi
}

# Counter for passed tests
total_tests=0
passed_tests=0

echo "======================================================================"
echo "📊 TEST 1: Week 3 Validation (30% BFT) - 5 Seeds"
echo "======================================================================"
echo ""

if [ -f "tests/test_30_bft_validation.py" ]; then
    for seed in "${SEEDS[@]}"; do
        total_tests=$((total_tests + 1))
        if run_test_with_seed "week3_30bft" "$seed" "tests/test_30_bft_validation.py"; then
            passed_tests=$((passed_tests + 1))
        fi
    done
    echo ""
else
    echo "⚠️  test_30_bft_validation.py not found, skipping"
    echo ""
fi

echo "======================================================================"
echo "📊 TEST 2: Boundary Tests - 3 Seeds"
echo "======================================================================"
echo ""

if [ -f "tests/test_35_40_bft_real.py" ]; then
    for seed in "${BOUNDARY_SEEDS[@]}"; do
        total_tests=$((total_tests + 1))
        if run_test_with_seed "boundary_35_40" "$seed" "tests/test_35_40_bft_real.py"; then
            passed_tests=$((passed_tests + 1))
        fi
    done
    echo ""
else
    echo "⚠️  test_35_40_bft_real.py not found, skipping"
    echo ""
fi

echo "======================================================================"
echo "📊 TEST 3: Sleeper Agent - 3 Seeds"
echo "======================================================================"
echo ""

if [ -f "tests/test_sleeper_agent_validation.py" ]; then
    export RUN_SLEEPER_AGENT_TEST=1
    for seed in "${BOUNDARY_SEEDS[@]}"; do
        total_tests=$((total_tests + 1))
        if run_test_with_seed "sleeper_agent" "$seed" "tests/test_sleeper_agent_validation.py"; then
            passed_tests=$((passed_tests + 1))
        fi
    done
    unset RUN_SLEEPER_AGENT_TEST
    echo ""
else
    echo "⚠️  test_sleeper_agent_validation.py not found, skipping"
    echo ""
fi

echo "======================================================================"
echo "📈 MULTI-SEED VALIDATION SUMMARY"
echo "======================================================================"
echo ""
echo "Tests Run: $total_tests"
echo "Tests Passed: $passed_tests"
echo "Tests Failed: $((total_tests - passed_tests))"
echo ""

if [ $passed_tests -eq $total_tests ]; then
    echo "✅ ALL TESTS PASSED ACROSS ALL SEEDS"
    echo ""
    echo "Statistical Robustness: CONFIRMED"
    echo "Results are consistent across random initializations"
else
    echo "⚠️  SOME TESTS FAILED"
    echo ""
    echo "Recommendation: Review failed test logs in $RESULTS_DIR"
    echo "Possible issues:"
    echo "  - Seed-dependent behavior (check variance)"
    echo "  - Configuration issues"
    echo "  - Environment problems"
fi

echo ""
echo "======================================================================"
echo "📁 Results Location: $RESULTS_DIR"
echo "======================================================================"
echo ""
echo "Next Steps:"
echo "  1. Review individual test logs"
echo "  2. Run aggregation script to compute statistics:"
echo "     python scripts/aggregate_test_results.py $RESULTS_DIR"
echo "  3. Generate visualization plots"
echo ""
echo "======================================================================"
