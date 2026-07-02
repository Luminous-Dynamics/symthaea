#!/usr/bin/env bash
#
# DKG Build Script
# Compiles the zomes to WASM and packages the hApp bundle
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧬 Building DKG (Distributed Knowledge Graph)..."
echo "══════════════════════════════════════════════════"

# Step 1: Build WASM
echo ""
echo "📦 Step 1: Compiling zomes to WASM..."
cargo build --release --target wasm32-unknown-unknown

# Verify WASM files exist
INTEGRITY_WASM="target/wasm32-unknown-unknown/release/dkg_integrity.wasm"
COORDINATOR_WASM="target/wasm32-unknown-unknown/release/dkg_coordinator.wasm"

if [[ ! -f "$INTEGRITY_WASM" ]]; then
    echo "❌ Error: Integrity WASM not found at $INTEGRITY_WASM"
    exit 1
fi

if [[ ! -f "$COORDINATOR_WASM" ]]; then
    echo "❌ Error: Coordinator WASM not found at $COORDINATOR_WASM"
    exit 1
fi

echo "   ✅ dkg_integrity.wasm: $(du -h "$INTEGRITY_WASM" | cut -f1)"
echo "   ✅ dkg_coordinator.wasm: $(du -h "$COORDINATOR_WASM" | cut -f1)"

# Step 2: Pack DNA (requires holonix or hc in PATH)
echo ""
echo "🧪 Step 2: Packing DNA..."
cd workdir

# Check if hc is available, if not try to use holonix
if command -v hc &> /dev/null; then
    hc dna pack . -o dkg.dna
    hc app pack . -o dkg.happ
else
    echo "   ℹ️  hc not in PATH, using holonix..."
    nix shell github:holochain/holonix#hc --accept-flake-config --command bash -c "
        hc dna pack . -o dkg.dna && hc app pack . -o dkg.happ
    "
fi

if [[ ! -f "dkg.dna" ]]; then
    echo "❌ Error: DNA pack failed"
    exit 1
fi
echo "   ✅ dkg.dna: $(du -h dkg.dna | cut -f1)"

# Step 3: Verify hApp
echo ""
echo "📱 Step 3: Verifying hApp..."

if [[ ! -f "dkg.happ" ]]; then
    echo "❌ Error: hApp pack failed"
    exit 1
fi
echo "   ✅ dkg.happ: $(du -h dkg.happ | cut -f1)"

# Summary
echo ""
echo "══════════════════════════════════════════════════"
echo "🎉 DKG Build Complete!"
echo ""
echo "Artifacts:"
echo "  - workdir/dkg.dna"
echo "  - workdir/dkg.happ"
echo ""
echo "Run tests with:"
echo "  cd tests && npm install && npm test"
echo "══════════════════════════════════════════════════"
