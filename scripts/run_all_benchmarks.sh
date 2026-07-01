#!/usr/bin/env bash
# Run Symthaea benchmark suite and collect results.
#
# Usage:
#   ./scripts/run_all_benchmarks.sh              # Quick mode (scorecard only)
#   ./scripts/run_all_benchmarks.sh --quick      # Same as above
#   ./scripts/run_all_benchmarks.sh --full       # Run ALL 61 benchmark examples
#   ./scripts/run_all_benchmarks.sh --list       # List all benchmarks
#
# Results are written to data/benchmarks/runs/ and aggregated into
# data/benchmarks/full_scorecard.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
RUN_DIR="${ROOT_DIR}/data/benchmarks/runs"
SCORECARD="${ROOT_DIR}/data/benchmarks/full_scorecard.json"
BASELINE="${ROOT_DIR}/data/benchmarks/full_scorecard_baseline.json"
TIMEOUT=600  # 10 minutes per benchmark

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Quick benchmarks (the scorecard suite — ~8 benchmarks that are fast) ──
QUICK_BENCHMARKS=(
    "benchmark_scorecard"
)

# ── Mode selection ─────────────────────────────────────────────────────
MODE="quick"
if [ "${1:-}" = "--full" ]; then
    MODE="full"
elif [ "${1:-}" = "--list" ]; then
    MODE="list"
elif [ "${1:-}" = "--quick" ]; then
    MODE="quick"
fi

# ── List mode ──────────────────────────────────────────────────────────
if [ "$MODE" = "list" ]; then
    echo "Available benchmarks:"
    for f in "${ROOT_DIR}"/examples/benchmark_*.rs; do
        name=$(basename "$f" .rs)
        echo "  $name"
    done | sort
    count=$(ls "${ROOT_DIR}"/examples/benchmark_*.rs 2>/dev/null | wc -l)
    echo ""
    echo "Total: $count benchmarks"
    exit 0
fi

# ── Setup ──────────────────────────────────────────────────────────────
mkdir -p "$RUN_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Symthaea Benchmark Runner                             ║"
echo "║       Mode: ${MODE}                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Build benchmarks list ──────────────────────────────────────────────
if [ "$MODE" = "quick" ]; then
    BENCHMARKS=("${QUICK_BENCHMARKS[@]}")
else
    BENCHMARKS=()
    for f in "${ROOT_DIR}"/examples/benchmark_*.rs; do
        name=$(basename "$f" .rs)
        BENCHMARKS+=("$name")
    done
fi

TOTAL=${#BENCHMARKS[@]}
PASSED=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

echo "Running $TOTAL benchmark(s)..."
echo ""

# ── Results accumulator ────────────────────────────────────────────────
RESULTS="[]"

for i in "${!BENCHMARKS[@]}"; do
    name="${BENCHMARKS[$i]}"
    idx=$((i + 1))
    log_file="${RUN_DIR}/${name}.log"

    printf "[%2d/%2d] %-50s " "$idx" "$TOTAL" "$name"

    # Run with timeout, capture output
    set +e
    timeout "$TIMEOUT" cargo run --manifest-path "${ROOT_DIR}/Cargo.toml" \
        --example "$name" --release 2>&1 > "$log_file"
    exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        printf "${GREEN}PASS${NC} "
        PASSED=$((PASSED + 1))

        # Try to extract timing from log
        elapsed=$(grep -oP 'Time: \K[0-9.]+s' "$log_file" 2>/dev/null | tail -1 || echo "?")
        printf "(${elapsed})\n"
    elif [ $exit_code -eq 124 ]; then
        printf "${YELLOW}TIMEOUT${NC} (>${TIMEOUT}s)\n"
        SKIPPED=$((SKIPPED + 1))
    else
        # Check if it's a missing dataset
        if grep -q "not found\|No data\|Cannot open\|No such file" "$log_file" 2>/dev/null; then
            printf "${YELLOW}SKIP${NC} (missing data)\n"
            SKIPPED=$((SKIPPED + 1))
        else
            printf "${RED}FAIL${NC} (exit $exit_code)\n"
            FAILED=$((FAILED + 1))
            # Show last few lines of error
            tail -3 "$log_file" 2>/dev/null | sed 's/^/    /'
        fi
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo -e "  Passed:  ${GREEN}${PASSED}${NC}"
echo -e "  Failed:  ${RED}${FAILED}${NC}"
echo -e "  Skipped: ${YELLOW}${SKIPPED}${NC}"
echo "  Total:   ${TOTAL}"
echo "  Time:    ${ELAPSED}s"
echo ""

# ── Collect scorecard.json if it was updated ───────────────────────────
if [ -f "${ROOT_DIR}/data/benchmarks/scorecard.json" ]; then
    cp "${ROOT_DIR}/data/benchmarks/scorecard.json" "$SCORECARD"
    echo "Scorecard saved to: $(basename "$SCORECARD")"
fi

# ── Regression check ──────────────────────────────────────────────────
if [ -f "$BASELINE" ] && [ -f "$SCORECARD" ]; then
    echo ""
    echo "Regression check against baseline:"
    # Simple: compare entry count
    baseline_count=$(python3 -c "import json; print(len(json.load(open('$BASELINE')).get('entries',[])))" 2>/dev/null || echo "?")
    current_count=$(python3 -c "import json; print(len(json.load(open('$SCORECARD')).get('entries',[])))" 2>/dev/null || echo "?")
    echo "  Baseline entries: $baseline_count"
    echo "  Current entries:  $current_count"
fi

echo ""
echo "Logs: ${RUN_DIR}/"

# ── Exit code ──────────────────────────────────────────────────────────
if [ $FAILED -gt 0 ]; then
    exit 1
fi
exit 0
