#!/bin/bash
# Comprehensive Validation for Grant Submission
# Tests all scenarios including critical alpha ≤ 0.2 label skew cases
#
# Total scenarios: 3 datasets × 2 distributions × 4 alphas × 8 attacks × 3 ratios = 576
# Estimated runtime: 6-8 hours
#
# Usage:
#   cd /srv/luminous-dynamics/Mycelix-Core/0TML
#   source ../.env.optimal
#   export USE_ML_DETECTOR=1 ML_DETECTOR_PATH=models/byzantine_detector_v4_7features
#   nohup sh scripts/run_comprehensive_validation_grant.sh &> /tmp/validation_$(date +%Y%m%d_%H%M).log &

set -e

echo "======================================================================"
echo "COMPREHENSIVE GRANT-READY VALIDATION"
echo "======================================================================"
echo "Start Time: $(date)"
echo "ML Detector: ${ML_DETECTOR_PATH:-NOT SET}"
echo "USE_ML_DETECTOR: ${USE_ML_DETECTOR:-NOT SET}"
echo ""

# Verify environment
if [ "$USE_ML_DETECTOR" != "1" ]; then
    echo "❌ ERROR: USE_ML_DETECTOR not set to 1"
    echo "Run: export USE_ML_DETECTOR=1"
    exit 1
fi

if [ -z "$ML_DETECTOR_PATH" ]; then
    echo "❌ ERROR: ML_DETECTOR_PATH not set"
    echo "Run: export ML_DETECTOR_PATH=models/byzantine_detector_v4_7features"
    exit 1
fi

if [ ! -d "$ML_DETECTOR_PATH" ]; then
    echo "❌ ERROR: ML detector not found at $ML_DETECTOR_PATH"
    exit 1
fi

# Verify optimal parameters are loaded
if [ "$LABEL_SKEW_COS_MIN" != "-0.5" ]; then
    echo "⚠️  WARNING: LABEL_SKEW_COS_MIN=$LABEL_SKEW_COS_MIN (expected -0.5)"
    echo "Did you source ../.env.optimal?"
fi

echo "✅ Environment verified"
echo ""

# Configuration
DATASETS=("cifar10" "emnist_balanced" "breast_cancer")
DISTRIBUTIONS=("iid" "label_skew")
LABEL_SKEW_ALPHAS=("0.1" "0.2" "0.5" "1.0")  # Critical: 0.1 and 0.2 are hard cases
ATTACKS=("noise" "sign_flip" "zero" "random" "backdoor" "adaptive" "scaled_sign_flip" "stealth_backdoor")
BFT_RATIOS=("0.30" "0.40" "0.50")

# Output directory
OUTPUT_DIR="results/grant_validation_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUTPUT_DIR"

# Initialize counters
TOTAL_SCENARIOS=0
COMPLETED_SCENARIOS=0
FAILED_SCENARIOS=0

# Calculate total scenarios
for dataset in "${DATASETS[@]}"; do
    for dist in "${DISTRIBUTIONS[@]}"; do
        if [ "$dist" = "label_skew" ]; then
            # Label skew: test all alpha values
            for alpha in "${LABEL_SKEW_ALPHAS[@]}"; do
                for attack in "${ATTACKS[@]}"; do
                    for ratio in "${BFT_RATIOS[@]}"; do
                        TOTAL_SCENARIOS=$((TOTAL_SCENARIOS + 1))
                    done
                done
            done
        else
            # IID: no alpha parameter
            for attack in "${ATTACKS[@]}"; do
                for ratio in "${BFT_RATIOS[@]}"; do
                    TOTAL_SCENARIOS=$((TOTAL_SCENARIOS + 1))
                done
            done
        fi
    done
done

echo "Total scenarios to run: $TOTAL_SCENARIOS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Main validation loop
START_TIME=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    echo "======================================================================"
    echo "DATASET: $dataset"
    echo "======================================================================"

    for dist in "${DISTRIBUTIONS[@]}"; do
        echo "----------------------------------------------------------------------"
        echo "Distribution: $dist"
        echo "----------------------------------------------------------------------"

        if [ "$dist" = "label_skew" ]; then
            # Test all alpha values for label skew
            for alpha in "${LABEL_SKEW_ALPHAS[@]}"; do
                echo ""
                echo "Label Skew Alpha: $alpha ($(echo "$alpha <= 0.2" | bc) ≤ 0.2 = CRITICAL)"

                for attack in "${ATTACKS[@]}"; do
                    for ratio in "${BFT_RATIOS[@]}"; do
                        SCENARIO="${dataset}_${dist}_alpha${alpha}_${attack}_bft${ratio}"
                        COMPLETED_SCENARIOS=$((COMPLETED_SCENARIOS + 1))

                        echo ""
                        echo "[$COMPLETED_SCENARIOS/$TOTAL_SCENARIOS] Running: $SCENARIO"

                        # Set environment for this scenario
                        export BFT_DATASET="$dataset"
                        export BFT_DISTRIBUTION="$dist"
                        export LABEL_SKEW_ALPHA="$alpha"
                        export BFT_ATTACK="$attack"
                        export BFT_RATIO="$ratio"
                        export RUN_30_BFT=1

                        # Run test and capture result
                        OUTPUT_FILE="$OUTPUT_DIR/${SCENARIO}.json"
                        LOG_FILE="$OUTPUT_DIR/${SCENARIO}.log"

                        if poetry run python tests/test_30_bft_validation.py &> "$LOG_FILE"; then
                            echo "  ✅ SUCCESS"
                            # Extract results (simplified - actual extraction needs parsing)
                            echo "{\"scenario\":\"$SCENARIO\",\"status\":\"success\"}" > "$OUTPUT_FILE"
                        else
                            echo "  ❌ FAILED"
                            FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                            echo "{\"scenario\":\"$SCENARIO\",\"status\":\"failed\"}" > "$OUTPUT_FILE"
                        fi

                        # Progress update
                        ELAPSED=$(($(date +%s) - START_TIME))
                        AVG_TIME=$((ELAPSED / COMPLETED_SCENARIOS))
                        REMAINING=$((TOTAL_SCENARIOS - COMPLETED_SCENARIOS))
                        ETA=$((AVG_TIME * REMAINING))

                        echo "  Progress: $COMPLETED_SCENARIOS/$TOTAL_SCENARIOS ($(( COMPLETED_SCENARIOS * 100 / TOTAL_SCENARIOS ))%)"
                        echo "  Elapsed: $((ELAPSED / 60))m | Avg: ${AVG_TIME}s/test | ETA: $((ETA / 60))m"
                    done
                done
            done
        else
            # IID distribution: no alpha parameter
            for attack in "${ATTACKS[@]}"; do
                for ratio in "${BFT_RATIOS[@]}"; do
                    SCENARIO="${dataset}_${dist}_${attack}_bft${ratio}"
                    COMPLETED_SCENARIOS=$((COMPLETED_SCENARIOS + 1))

                    echo ""
                    echo "[$COMPLETED_SCENARIOS/$TOTAL_SCENARIOS] Running: $SCENARIO"

                    # Set environment for this scenario
                    export BFT_DATASET="$dataset"
                    export BFT_DISTRIBUTION="$dist"
                    unset LABEL_SKEW_ALPHA  # Not used for IID
                    export BFT_ATTACK="$attack"
                    export BFT_RATIO="$ratio"
                    export RUN_30_BFT=1

                    # Run test and capture result
                    OUTPUT_FILE="$OUTPUT_DIR/${SCENARIO}.json"
                    LOG_FILE="$OUTPUT_DIR/${SCENARIO}.log"

                    if poetry run python tests/test_30_bft_validation.py &> "$LOG_FILE"; then
                        echo "  ✅ SUCCESS"
                        echo "{\"scenario\":\"$SCENARIO\",\"status\":\"success\"}" > "$OUTPUT_FILE"
                    else
                        echo "  ❌ FAILED"
                        FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                        echo "{\"scenario\":\"$SCENARIO\",\"status\":\"failed\"}" > "$OUTPUT_FILE"
                    fi

                    # Progress update
                    ELAPSED=$(($(date +%s) - START_TIME))
                    AVG_TIME=$((ELAPSED / COMPLETED_SCENARIOS))
                    REMAINING=$((TOTAL_SCENARIOS - COMPLETED_SCENARIOS))
                    ETA=$((AVG_TIME * REMAINING))

                    echo "  Progress: $COMPLETED_SCENARIOS/$TOTAL_SCENARIOS ($(( COMPLETED_SCENARIOS * 100 / TOTAL_SCENARIOS ))%)"
                    echo "  Elapsed: $((ELAPSED / 60))m | Avg: ${AVG_TIME}s/test | ETA: $((ETA / 60))m"
                done
            done
        fi
    done
done

# Final summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "======================================================================"
echo "COMPREHENSIVE VALIDATION COMPLETE"
echo "======================================================================"
echo "End Time: $(date)"
echo "Total Time: $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m"
echo ""
echo "Results:"
echo "  Total Scenarios: $TOTAL_SCENARIOS"
echo "  Completed: $COMPLETED_SCENARIOS"
echo "  Failed: $FAILED_SCENARIOS"
echo "  Success Rate: $(( (COMPLETED_SCENARIOS - FAILED_SCENARIOS) * 100 / COMPLETED_SCENARIOS ))%"
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo ""

if [ $FAILED_SCENARIOS -gt 0 ]; then
    echo "⚠️  $FAILED_SCENARIOS scenarios failed. Review logs in $OUTPUT_DIR/*.log"
else
    echo "✅ All scenarios completed successfully!"
fi

echo ""
echo "Next Steps:"
echo "  1. Analyze results: python scripts/analyze_grant_validation.py $OUTPUT_DIR"
echo "  2. Generate report: python scripts/generate_grant_report.py $OUTPUT_DIR"
echo "  3. Create plots: python scripts/plot_validation_results.py $OUTPUT_DIR"
echo ""
echo "======================================================================"
