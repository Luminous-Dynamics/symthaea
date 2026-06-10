#!/usr/bin/env bash
# Regression test for the Spore WASM build.
#
# Checks:
# 1. Rust unit tests pass
# 2. WASM binary builds successfully
# 3. Binary size stays under budget (300KB raw, 150KB gzip)
# 4. JS bindings export expected symbols
# 5. Node.js smoke test (if node available)
#
# Usage:
#   ./test-wasm.sh          # Full regression
#   ./test-wasm.sh --quick  # Skip rebuild, just check existing pkg/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$SCRIPT_DIR/www/pkg"
PASS=0
FAIL=0
TOTAL=0

pass() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); echo "  FAIL: $1"; }
check() {
    if eval "$1"; then pass "$2"; else fail "$2"; fi
}

echo "=== Spore WASM Regression Tests ==="
echo ""

# ---- 1. Rust unit tests ----
if [[ "${1:-}" != "--quick" ]]; then
    echo "[1/5] Running Rust unit tests..."
    if cargo test -p symthaea-spore --lib 2>&1 | tail -5 | grep -q "test result: ok"; then
        pass "Rust unit tests"
    else
        fail "Rust unit tests"
    fi
    echo ""

    # ---- 2. WASM build ----
    echo "[2/5] Building WASM..."
    if "$SCRIPT_DIR/build-wasm.sh" > /tmp/spore-build.log 2>&1; then
        pass "WASM build"
    else
        fail "WASM build"
        cat /tmp/spore-build.log
    fi
    echo ""
else
    echo "[1-2/5] Skipped (--quick mode)"
    echo ""
fi

# ---- 3. Size checks ----
echo "[3/5] Size budget checks..."
if [[ -f "$PKG_DIR/symthaea_spore_bg.wasm" ]]; then
    RAW_SIZE=$(stat -c%s "$PKG_DIR/symthaea_spore_bg.wasm" 2>/dev/null || stat -f%z "$PKG_DIR/symthaea_spore_bg.wasm")
    GZIP_SIZE=$(gzip -c "$PKG_DIR/symthaea_spore_bg.wasm" | wc -c)

    check "[[ $RAW_SIZE -lt 307200 ]]" "WASM raw < 300KB (actual: $((RAW_SIZE/1024))KB)"
    check "[[ $GZIP_SIZE -lt 153600 ]]" "WASM gzip < 150KB (actual: $((GZIP_SIZE/1024))KB)"
else
    fail "WASM file not found at $PKG_DIR/symthaea_spore_bg.wasm"
fi
echo ""

# ---- 4. JS bindings check ----
echo "[4/5] JS binding symbol checks..."
if [[ -f "$PKG_DIR/symthaea_spore.js" ]]; then
    JS="$PKG_DIR/symthaea_spore.js"
    check "grep -q 'class SporeEngine' '$JS'" "SporeEngine class exported"
    check "grep -q 'cycle(' '$JS'" "cycle() method present"
    check "grep -q 'set_substrate(' '$JS'" "set_substrate() method present"
    check "grep -q 'inject_neuromodulator(' '$JS'" "inject_neuromodulator() method present"
    check "grep -q 'consciousness_level(' '$JS'" "consciousness_level() method present"
    check "grep -q 'honest_confidence(' '$JS'" "honest_confidence() method present"
    check "grep -q 'harmony_alignment(' '$JS'" "harmony_alignment() method present"
    check "grep -q 'anesthesia_experiment(' '$JS'" "anesthesia_experiment() method present"
    check "grep -q 'measure_pci(' '$JS'" "measure_pci() method present"
    check "grep -q 'split_brain_experiment(' '$JS'" "split_brain_experiment() method present"
    check "grep -q 'collapse_threshold_experiment(' '$JS'" "collapse_threshold_experiment() method present"
    check "grep -q 'consciousness_report(' '$JS'" "consciousness_report() method present"
    check "grep -q 'active_instance_count(' '$JS'" "active_instance_count() method present"
else
    fail "JS bindings not found at $PKG_DIR/symthaea_spore.js"
fi
echo ""

# ---- 5. TypeScript definitions check ----
echo "[5/5] TypeScript definitions check..."
if [[ -f "$PKG_DIR/symthaea_spore.d.ts" ]]; then
    DTS="$PKG_DIR/symthaea_spore.d.ts"
    check "grep -q 'export class SporeEngine' '$DTS'" "SporeEngine in .d.ts"
    check "grep -q 'anesthesia_experiment' '$DTS'" "anesthesia_experiment in .d.ts"
    check "grep -q 'measure_pci' '$DTS'" "measure_pci in .d.ts"
    check "grep -q 'split_brain_experiment' '$DTS'" "split_brain_experiment in .d.ts"
    check "grep -q 'collapse_threshold_experiment' '$DTS'" "collapse_threshold_experiment in .d.ts"
else
    fail "TypeScript definitions not found"
fi
echo ""

# ---- Summary ----
echo "=== Results: $PASS/$TOTAL passed, $FAIL failed ==="
if [[ $FAIL -gt 0 ]]; then
    echo "REGRESSION DETECTED"
    exit 1
else
    echo "ALL CLEAR"
    exit 0
fi
