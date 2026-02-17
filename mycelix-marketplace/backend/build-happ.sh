#!/usr/bin/env bash
# build-happ.sh - Build Mycelix Marketplace hApp
#
# This script builds all WASM zomes, packages the DNA, and creates the hApp bundle.

set -e  # Exit on error

echo "🔧 Building Mycelix Marketplace hApp..."
echo ""

# Check if in nix-shell
if [ -z "$IN_NIX_SHELL" ]; then
    echo "⚠️  Not in nix-shell. Run 'nix develop' first!"
    exit 1
fi

# Check if wasm32-unknown-unknown target is installed
if ! rustup target list | grep -q "wasm32-unknown-unknown (installed)"; then
    echo "📦 Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Building Integrity Zomes..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for zome in listings reputation transactions arbitration messaging; do
    echo "  📦 Building ${zome}_integrity..."
    cargo build --release --target wasm32-unknown-unknown -p "${zome}_integrity"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Building Coordinator Zomes..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for zome in listings reputation transactions arbitration messaging; do
    echo "  📦 Building ${zome}..."
    cargo build --release --target wasm32-unknown-unknown -p "${zome}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Copying WASM Files..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create zome directories
for zome in listings reputation transactions arbitration messaging; do
    mkdir -p "zomes/${zome}"
done

# Copy integrity zomes
for zome in listings reputation transactions arbitration messaging; do
    echo "  📁 Copying ${zome}_integrity.wasm..."
    cp "target/wasm32-unknown-unknown/release/${zome}_integrity.wasm" \
       "zomes/${zome}/integrity.wasm"
done

# Copy coordinator zomes
for zome in listings reputation transactions arbitration messaging; do
    echo "  📁 Copying ${zome}.wasm..."
    cp "target/wasm32-unknown-unknown/release/${zome}.wasm" \
       "zomes/${zome}/coordinator.wasm"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  Packaging DNA..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
hc dna pack .

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  Packaging hApp..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
hc app pack .

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Build Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📦 Output files:"
echo "  - dna.dna (DNA bundle)"
echo "  - mycelix_marketplace.happ (hApp bundle)"
echo ""
echo "🚀 Next steps:"
echo "  1. Test with: holochain -c conductor-config.yaml"
echo "  2. Or deploy to Holochain network"
echo ""
