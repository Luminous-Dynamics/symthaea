#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

TARGET_DIR="${CARGO_TARGET_DIR:-target}"

PASS=0
FAIL=0
FAILURES=()

log()  { printf "\n=== %s ===\n" "$*"; }
pass() { PASS=$((PASS + 1)); }
fail() { FAIL=$((FAIL + 1)); FAILURES+=("$1"); }

# --- Prerequisites -----------------------------------------------------------

log "Checking prerequisites"

missing=()
command -v holochain >/dev/null 2>&1 || missing+=("holochain")
command -v hc        >/dev/null 2>&1 || missing+=("hc")

# Check for libclang (needed by datachannel-sys)
if ! ldconfig -p 2>/dev/null | grep -q libclang; then
    if [ -z "${LIBCLANG_PATH:-}" ]; then
        missing+=("libclang-dev (or set LIBCLANG_PATH)")
    fi
fi

if [ ${#missing[@]} -gt 0 ]; then
    printf "Missing prerequisites:\n"
    for m in "${missing[@]}"; do
        printf "  - %s\n" "$m"
    done
    exit 1
fi

printf "All prerequisites found.\n"
printf "  holochain : %s\n" "$(holochain --version 2>&1 || true)"
printf "  hc        : %s\n" "$(hc --version 2>&1 || true)"
printf "  target dir: %s\n" "$TARGET_DIR"

# --- Build WASM zomes --------------------------------------------------------

log "Building WASM zomes"
cargo build --release --workspace --target wasm32-unknown-unknown

# --- Pack DNA -----------------------------------------------------------------

DNA_PATH="dna/mycelix_finance.dna"
if [ -f "$DNA_PATH" ]; then
    printf "DNA bundle already exists at %s, skipping pack.\n" "$DNA_PATH"
else
    log "Packing DNA bundle"
    hc dna pack dna/
fi

# --- Integration tests (require conductor) ------------------------------------

INTEGRATION_TESTS=(
    sweettest_integration
    bridge_test
    tend_test
    payments_test
    treasury_test
    staking_test
    recognition_test
)

log "Running integration tests (sweettest, single-threaded)"

for t in "${INTEGRATION_TESTS[@]}"; do
    printf "\n--- %s ---\n" "$t"
    if cargo test --test "$t" -- --include-ignored --test-threads=1; then
        pass
        printf "  PASS: %s\n" "$t"
    else
        fail "$t"
        printf "  FAIL: %s\n" "$t"
    fi
done

# --- Stress tests (pure Rust, no conductor) -----------------------------------

log "Running economics stress tests"

if cargo test -p mycelix_finance_types --test economics_stress_test; then
    pass
    printf "  PASS: economics_stress_test\n"
else
    fail "economics_stress_test"
    printf "  FAIL: economics_stress_test\n"
fi

# --- Summary ------------------------------------------------------------------

log "Test Summary"
printf "  Passed: %d\n" "$PASS"
printf "  Failed: %d\n" "$FAIL"

if [ "$FAIL" -gt 0 ]; then
    printf "\nFailed tests:\n"
    for f in "${FAILURES[@]}"; do
        printf "  - %s\n" "$f"
    done
    exit 1
fi

printf "\nAll tests passed.\n"
