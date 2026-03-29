#!/usr/bin/env bash
# check-bridge-common-features.sh
# Local CI check for mycelix-bridge-common feature matrix.
# Verifies the crate builds with and without default features.
#
# Usage: ./scripts/check-bridge-common-features.sh

set -euo pipefail

CRATE="mycelix-bridge-common"
PASS=0
FAIL=0
RESULTS=()

run_check() {
    local label="$1"
    shift
    printf "  %-50s " "$label"
    if "$@" >/dev/null 2>&1; then
        echo "OK"
        PASS=$((PASS + 1))
        RESULTS+=("  PASS  $label")
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
        RESULTS+=("  FAIL  $label")
    fi
}

echo "=== $CRATE feature matrix check ==="
echo ""

run_check "cargo check --no-default-features" \
    cargo check -p "$CRATE" --no-default-features

run_check "cargo check (default features)" \
    cargo check -p "$CRATE"

run_check "cargo check --all-features" \
    cargo check -p "$CRATE" --all-features

run_check "cargo test --no-default-features" \
    cargo test -p "$CRATE" --no-default-features --lib --no-run

run_check "cargo test (default features, compile only)" \
    cargo test -p "$CRATE" --lib --no-run

echo ""
echo "=== Results ==="
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo ""
echo "Passed: $PASS  Failed: $FAIL"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
