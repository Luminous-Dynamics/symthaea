#!/usr/bin/env bash
# Build the Symthaea Spore WASM consciousness kernel.
#
# Pipeline: cargo build → wasm-opt → wasm-bindgen
#
# Requirements: cargo, wasm32-unknown-unknown target, wasm-opt, wasm-bindgen CLI
# On NixOS: all provided by the flake devShell.
#
# Usage:
#   ./build-wasm.sh          # Full optimized build
#   ./build-wasm.sh --debug  # Debug build (faster, larger)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$SCRIPT_DIR/www/pkg"
TMP_OPT="/tmp/symthaea_spore.wasm"

PROFILE="release"
CARGO_FLAGS="--release"
if [[ "${1:-}" == "--debug" ]]; then
    PROFILE="debug"
    CARGO_FLAGS=""
    echo "[spore] Debug build (no optimization)"
fi

echo "[spore] Step 1/3: Compiling to wasm32-unknown-unknown ($PROFILE)..."
cargo build $CARGO_FLAGS --target wasm32-unknown-unknown --features wasm,broca-pipeline -p symthaea-spore

# Find the wasm file (workspace target dir may vary)
WASM_FILE=$(find "$(cargo metadata --format-version=1 --no-deps 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])' 2>/dev/null || echo "$SCRIPT_DIR/../../target")/wasm32-unknown-unknown/$PROFILE" -name "symthaea_spore.wasm" -maxdepth 1 2>/dev/null | head -1)

if [[ -z "$WASM_FILE" ]]; then
    echo "[spore] ERROR: Could not find symthaea_spore.wasm in target directory"
    exit 1
fi
echo "[spore] Found: $WASM_FILE ($(du -h "$WASM_FILE" | cut -f1))"

# Step 2: Optimize with wasm-opt (skip for debug)
if [[ "$PROFILE" == "release" ]] && command -v wasm-opt &>/dev/null; then
    echo "[spore] Step 2/3: Optimizing with wasm-opt..."
    wasm-opt \
        --enable-bulk-memory \
        --enable-nontrapping-float-to-int \
        --enable-sign-ext \
        -O2 \
        "$WASM_FILE" \
        -o "$TMP_OPT"
    WASM_FILE="$TMP_OPT"
    echo "[spore] Optimized: $(du -h "$WASM_FILE" | cut -f1)"
else
    echo "[spore] Step 2/3: Skipping wasm-opt (${PROFILE} build or wasm-opt not found)"
fi

# Step 3: Generate JS bindings
echo "[spore] Step 3/3: Generating JS bindings with wasm-bindgen..."
mkdir -p "$PKG_DIR"
wasm-bindgen --target web --out-dir "$PKG_DIR" "$WASM_FILE"

# wasm-bindgen derives output names from the input filename.
# Since we name the temp file symthaea_spore.wasm, outputs are already correct:
# symthaea_spore_bg.wasm, symthaea_spore.js, symthaea_spore.d.ts

# Report sizes
WASM_SIZE=$(stat -c%s "$PKG_DIR/symthaea_spore_bg.wasm" 2>/dev/null || stat -f%z "$PKG_DIR/symthaea_spore_bg.wasm")
JS_SIZE=$(stat -c%s "$PKG_DIR/symthaea_spore.js" 2>/dev/null || stat -f%z "$PKG_DIR/symthaea_spore.js")

echo ""
echo "=== Spore WASM Build Complete ==="
echo "  WASM: $(du -h "$PKG_DIR/symthaea_spore_bg.wasm" | cut -f1) ($WASM_SIZE bytes)"
echo "  JS:   $(du -h "$PKG_DIR/symthaea_spore.js" | cut -f1) ($JS_SIZE bytes)"

# Gzip size estimate
if command -v gzip &>/dev/null; then
    GZIP_SIZE=$(gzip -c "$PKG_DIR/symthaea_spore_bg.wasm" | wc -c)
    echo "  WASM (gzip): $((GZIP_SIZE / 1024))KB ($GZIP_SIZE bytes)"
fi

# Brotli size estimate
if command -v brotli &>/dev/null; then
    BROTLI_SIZE=$(brotli -c "$PKG_DIR/symthaea_spore_bg.wasm" | wc -c)
    echo "  WASM (brotli): $((BROTLI_SIZE / 1024))KB ($BROTLI_SIZE bytes)"
fi

echo ""
echo "Output: $PKG_DIR/"
ls -lh "$PKG_DIR/"

# Clean up temp file
rm -f "$TMP_OPT"
