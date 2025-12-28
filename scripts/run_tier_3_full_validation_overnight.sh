#!/usr/bin/env bash
# Tier 3 Full Validation - Overnight Background Build & Execution
#
# Purpose: Run complete 19-topology validation in background for long compilation
# Expected compile time: 15-30 minutes (release mode)
# Expected run time: 30-60 seconds
# Total overnight time: Safe buffer for compilation + execution

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

BUILD_LOG="$LOG_DIR/tier3_full_build_${TIMESTAMP}.log"
RUN_LOG="$LOG_DIR/tier3_full_run_${TIMESTAMP}.log"
STATUS_LOG="$LOG_DIR/tier3_full_status_${TIMESTAMP}.log"
RESULTS_FILE="$PROJECT_ROOT/TIER_3_VALIDATION_RESULTS_${TIMESTAMP}.txt"

echo "🌙 Starting Tier 3 Full Validation (Overnight Mode)" | tee "$STATUS_LOG"
echo "==========================================" | tee -a "$STATUS_LOG"
echo "Timestamp: $TIMESTAMP" | tee -a "$STATUS_LOG"
echo "Build Log: $BUILD_LOG" | tee -a "$STATUS_LOG"
echo "Run Log: $RUN_LOG" | tee -a "$STATUS_LOG"
echo "Results: $RESULTS_FILE" | tee -a "$STATUS_LOG"
echo "" | tee -a "$STATUS_LOG"

cd "$PROJECT_ROOT"

# Step 1: Build in release mode (optimized, slow compile but fast execution)
echo "📦 Step 1: Building tier_3_exotic_topologies example (release mode)..." | tee -a "$STATUS_LOG"
echo "⏰ Estimated time: 15-30 minutes" | tee -a "$STATUS_LOG"
echo "Started at: $(date)" | tee -a "$STATUS_LOG"

if cargo build --example tier_3_exotic_topologies --release > "$BUILD_LOG" 2>&1; then
    echo "✅ Build completed successfully at $(date)" | tee -a "$STATUS_LOG"
    BUILD_SUCCESS=true
else
    echo "❌ Build failed at $(date)" | tee -a "$STATUS_LOG"
    echo "Check build log: $BUILD_LOG" | tee -a "$STATUS_LOG"
    BUILD_SUCCESS=false
fi

# Step 2: Execute validation (if build succeeded)
if [ "$BUILD_SUCCESS" = true ]; then
    echo "" | tee -a "$STATUS_LOG"
    echo "🚀 Step 2: Running validation (19 topologies × 10 samples)..." | tee -a "$STATUS_LOG"
    echo "⏰ Estimated time: 30-60 seconds" | tee -a "$STATUS_LOG"
    echo "Started at: $(date)" | tee -a "$STATUS_LOG"

    if cargo run --example tier_3_exotic_topologies --release > "$RUN_LOG" 2>&1; then
        echo "✅ Execution completed successfully at $(date)" | tee -a "$STATUS_LOG"
        RUN_SUCCESS=true

        # Copy results to dedicated file
        cp "$RUN_LOG" "$RESULTS_FILE"
        echo "" | tee -a "$STATUS_LOG"
        echo "📊 Results saved to: $RESULTS_FILE" | tee -a "$STATUS_LOG"
    else
        echo "❌ Execution failed at $(date)" | tee -a "$STATUS_LOG"
        echo "Check run log: $RUN_LOG" | tee -a "$STATUS_LOG"
        RUN_SUCCESS=false
    fi
else
    echo "⏭️  Skipping execution (build failed)" | tee -a "$STATUS_LOG"
    RUN_SUCCESS=false
fi

# Step 3: Summary
echo "" | tee -a "$STATUS_LOG"
echo "==========================================" | tee -a "$STATUS_LOG"
echo "🏁 Tier 3 Full Validation Complete" | tee -a "$STATUS_LOG"
echo "==========================================" | tee -a "$STATUS_LOG"
echo "Finished at: $(date)" | tee -a "$STATUS_LOG"
echo "" | tee -a "$STATUS_LOG"

if [ "$BUILD_SUCCESS" = true ] && [ "$RUN_SUCCESS" = true ]; then
    echo "✅ SUCCESS: All steps completed" | tee -a "$STATUS_LOG"
    echo "" | tee -a "$STATUS_LOG"
    echo "📊 Quick Summary from Results:" | tee -a "$STATUS_LOG"
    echo "" | tee -a "$STATUS_LOG"

    # Extract key results
    if grep -q "DIMENSIONAL INVARIANCE" "$RESULTS_FILE"; then
        echo "🔬 Dimensional Invariance Test:" | tee -a "$STATUS_LOG"
        grep -A 5 "DIMENSIONAL INVARIANCE" "$RESULTS_FILE" | tee -a "$STATUS_LOG"
    fi

    if grep -q "OVERALL CHAMPION" "$RESULTS_FILE"; then
        echo "" | tee -a "$STATUS_LOG"
        echo "🏆 Overall Champion:" | tee -a "$STATUS_LOG"
        grep -A 3 "OVERALL CHAMPION" "$RESULTS_FILE" | tee -a "$STATUS_LOG"
    fi

    echo "" | tee -a "$STATUS_LOG"
    echo "📄 View full results:" | tee -a "$STATUS_LOG"
    echo "   cat $RESULTS_FILE" | tee -a "$STATUS_LOG"

    exit 0
elif [ "$BUILD_SUCCESS" = true ]; then
    echo "⚠️  BUILD OK, RUN FAILED" | tee -a "$STATUS_LOG"
    echo "Check logs:" | tee -a "$STATUS_LOG"
    echo "   Build: $BUILD_LOG" | tee -a "$STATUS_LOG"
    echo "   Run: $RUN_LOG" | tee -a "$STATUS_LOG"
    exit 1
else
    echo "❌ FAILED: Build did not complete" | tee -a "$STATUS_LOG"
    echo "Check build log: $BUILD_LOG" | tee -a "$STATUS_LOG"
    exit 2
fi
