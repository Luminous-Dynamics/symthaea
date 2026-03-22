#!/usr/bin/env bash
# Parameter sweep runner for Mycelix/Symthaea simulations.
# Runs multiple configurations with 10 seeds for statistical robustness.
#
# Usage:
#   bash sweep.sh                  # Run all sweeps (skip existing)
#   bash sweep.sh --force          # Re-run everything
#   bash sweep.sh --sim commons    # Run only commons sweeps
#   bash sweep.sh --sim macro      # Run only macro-economy sweeps
#   bash sweep.sh --sim statespace # Run only state-space sweeps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SIM_DIR/sweep-results"
FORCE=false
SIM_FILTER="all"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=true; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --sim) SIM_FILTER="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash sweep.sh [--force] [--sim {all,macro,commons,statespace}] [--output-dir DIR]"
            exit 0 ;;
        *) shift ;;
    esac
done

mkdir -p "$OUTPUT_DIR/macro-economy" "$OUTPUT_DIR/commons-resource" "$OUTPUT_DIR/state-space"

TOTAL=0
DONE=0
SKIPPED=0

run_if_needed() {
    local out="$1"; shift
    TOTAL=$((TOTAL + 1))
    if [[ -f "$out" && "$FORCE" == "false" ]]; then
        SKIPPED=$((SKIPPED + 1))
        return
    fi
    DONE=$((DONE + 1))
    echo "  [$DONE] $(basename "$out")"
    "$@" > "$out" 2>/dev/null
}

# 10 seeds for statistical robustness
SEEDS="42 123 789 1337 2024 3141 4242 5555 6789 9876"

# ============================================================================
# BUILD
# ============================================================================

echo "=== Building simulations ==="

if [[ "$SIM_FILTER" == "all" || "$SIM_FILTER" == "macro" ]]; then
    echo "  Building macro-economy..."
    (cd "$SIM_DIR/macro-economy" && cargo build --release 2>/dev/null)
    MACRO_BIN="$SIM_DIR/macro-economy/target/release/macro-economy-sim"
fi

if [[ "$SIM_FILTER" == "all" || "$SIM_FILTER" == "commons" ]]; then
    echo "  Building commons-resource..."
    (cd "$SIM_DIR/commons-resource" && cargo build --release 2>/dev/null)
    COMMONS_BIN="$SIM_DIR/commons-resource/target/release/commons-resource-sim"
fi

SYMTHAEA_DIR="$SIM_DIR/../../symthaea"
STATE_SPACE_BIN="$SYMTHAEA_DIR/target/release/examples/consciousness_state_space"

echo ""

# ============================================================================
# MACRO ECONOMY SWEEPS
# ============================================================================

if [[ "$SIM_FILTER" == "all" || "$SIM_FILTER" == "macro" ]]; then
    echo "=== Macro Economy Sweeps ==="
    DONE=0
    SKIPPED=0

    # Sweep 1: Inactive rates
    INACTIVE_RATES="0.0 0.5 1.0 1.5 2.0 3.0"
    for rate in $INACTIVE_RATES; do
        for seed in $SEEDS; do
            run_if_needed "$OUTPUT_DIR/macro-economy/inactive_${rate}_seed_${seed}.csv" \
                "$MACRO_BIN" --agents 500 --days 365 --seed "$seed" --inactive-rate "$rate"
        done
    done

    # Sweep 2: Demurrage rates (constitutional bounds: 1-5%)
    DEMURRAGE_RATES="0.01 0.02 0.03 0.04 0.05"
    for rate in $DEMURRAGE_RATES; do
        for seed in $SEEDS; do
            run_if_needed "$OUTPUT_DIR/macro-economy/demurrage_${rate}_seed_${seed}.csv" \
                "$MACRO_BIN" --agents 500 --days 365 --seed "$seed" --demurrage-rate "$rate"
        done
    done

    # Sweep 3: Jubilee timing (years between compression)
    JUBILEE_YEARS="0 2 4 8"
    for years in $JUBILEE_YEARS; do
        for seed in $SEEDS; do
            run_if_needed "$OUTPUT_DIR/macro-economy/jubilee_${years}y_seed_${seed}.csv" \
                "$MACRO_BIN" --agents 500 --days 1460 --seed "$seed" --jubilee-years "$years"
        done
    done

    echo "  Macro economy: $DONE runs, $SKIPPED skipped"
    echo ""
fi

# ============================================================================
# COMMONS RESOURCE SWEEPS
# ============================================================================

if [[ "$SIM_FILTER" == "all" || "$SIM_FILTER" == "commons" ]]; then
    echo "=== Commons Resource Sweeps ==="
    DONE=0
    SKIPPED=0

    DEFECTOR_PCTS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
    VOTING_MODES="consciousness flat plutocratic random"

    # Sweep 1: Voting mode comparison (primary)
    for mode in $VOTING_MODES; do
        for defpct in $DEFECTOR_PCTS; do
            for seed in $SEEDS; do
                run_if_needed "$OUTPUT_DIR/commons-resource/${mode}_def${defpct}_seed${seed}.csv" \
                    "$COMMONS_BIN" --agents 100 --resources 10 --days 365 \
                    --defector-pct "$defpct" --seed "$seed" --voting-mode "$mode"
            done
        done
    done

    # No governance baseline
    for defpct in $DEFECTOR_PCTS; do
        for seed in $SEEDS; do
            run_if_needed "$OUTPUT_DIR/commons-resource/nogovern_def${defpct}_seed${seed}.csv" \
                "$COMMONS_BIN" --agents 100 --resources 10 --days 365 \
                --defector-pct "$defpct" --seed "$seed" --no-governance
        done
    done

    # Sweep 2: Cross-sim consciousness degradation
    DEGRADATION_RATES="0.0 0.001 0.002 0.005 0.01"
    for degrade in $DEGRADATION_RATES; do
        for seed in $SEEDS; do
            run_if_needed "$OUTPUT_DIR/commons-resource/degrade_${degrade}_seed${seed}.csv" \
                "$COMMONS_BIN" --agents 100 --resources 10 --days 365 \
                --defector-pct 0.3 --seed "$seed" --consciousness-degradation "$degrade"
        done
    done

    echo "  Commons resource: $DONE runs, $SKIPPED skipped"
    echo ""
fi

# ============================================================================
# STATE-SPACE SWEEPS
# ============================================================================

if [[ "$SIM_FILTER" == "all" || "$SIM_FILTER" == "statespace" ]]; then
    if [[ ! -x "$STATE_SPACE_BIN" ]]; then
        echo "=== State-Space Sweeps: SKIPPED ==="
        echo "  Binary not found: $STATE_SPACE_BIN"
        echo "  Build with: cd symthaea && nix develop --command cargo build --release --example consciousness_state_space"
        echo ""
    else
        echo "=== State-Space Sweeps ==="
        COMPONENTS="attention binding efficacy integration knowledge recursion workspace"
        DONE=0
        SKIPPED=0

        # All 21 pairwise heatmaps
        for a in $COMPONENTS; do
            for b in $COMPONENTS; do
                [[ "$a" > "$b" || "$a" == "$b" ]] && continue
                run_if_needed "$OUTPUT_DIR/state-space/heatmap_${a}_${b}.csv" \
                    "$STATE_SPACE_BIN" --mode heatmap --axis-a "$a" --axis-b "$b" --resolution 20
            done
        done

        # LHS sampling
        run_if_needed "$OUTPUT_DIR/state-space/lhs.csv" \
            "$STATE_SPACE_BIN" --mode lhs --samples 10000 --seed 42

        # Phase transition
        run_if_needed "$OUTPUT_DIR/state-space/transition.csv" \
            "$STATE_SPACE_BIN" --mode transition --resolution 100

        echo "  State-space: $DONE runs, $SKIPPED skipped"
        echo ""
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo "=== Sweep Complete ==="
echo "Results in: $OUTPUT_DIR"
echo "Total files: $(find "$OUTPUT_DIR" -name '*.csv' | wc -l)"
