#!/usr/bin/env bash
# Complete WASM Build Script for Mycelix Marketplace
# Runs from project root, enters backend/, builds all zomes

set -e  # Exit on error

echo "🚀 Mycelix Marketplace - Complete WASM Build"
echo "=============================================="
echo ""

# Navigate to backend directory
cd "$(dirname "$0")/backend"

echo "📍 Working directory: $(pwd)"
echo ""

# Check if wasm32-unknown-unknown target is installed (optional check)
if command -v rustup &> /dev/null; then
    if ! rustup target list 2>/dev/null | grep -q "wasm32-unknown-unknown (installed)"; then
        echo "📦 Installing wasm32-unknown-unknown target..."
        rustup target add wasm32-unknown-unknown
    fi
else
    echo "📝 Note: rustup not found, assuming wasm32 target is available from Nix environment"
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

# Every coordinator crate's [package] name is "${zome}_coordinator", but its
# [lib] (wasm artifact) name varies per crate: some match the package name
# (transactions, arbitration), others use the bare zome name (listings,
# reputation, messaging). Cargo's -p flag needs the package name; the copy
# step below needs the lib name.
declare -A LIB_NAME=(
    [listings]="listings"
    [reputation]="reputation"
    [transactions]="transactions_coordinator"
    [arbitration]="arbitration_coordinator"
    [messaging]="messaging"
)

for zome in listings reputation transactions arbitration messaging; do
    echo "  📦 Building ${zome}_coordinator..."
    cargo build --release --target wasm32-unknown-unknown -p "${zome}_coordinator"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Copying WASM Files..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create zome directories
for zome in listings reputation transactions arbitration messaging; do
    mkdir -p "zomes/${zome}"
done

# Copy integrity zomes. Destination filenames must be globally unique
# (dna.yaml's basename-keyed bundle resources — see the note there), so
# these are named "${zome}_integrity.wasm", not the old "integrity.wasm".
for zome in listings reputation transactions arbitration messaging; do
    echo "  📁 Copying ${zome}_integrity.wasm..."
    cp "target/wasm32-unknown-unknown/release/${zome}_integrity.wasm" \
       "zomes/${zome}/${zome}_integrity.wasm"
done

# Copy coordinator zomes (source filename is the crate's [lib] name, not
# necessarily the zome name — see LIB_NAME map above). Destination is
# "${zome}_coordinator.wasm" to keep bundle resource basenames unique.
for zome in listings reputation transactions arbitration messaging; do
    lib_name="${LIB_NAME[$zome]}"
    echo "  📁 Copying ${lib_name}.wasm..."
    cp "target/wasm32-unknown-unknown/release/${lib_name}.wasm" \
       "zomes/${zome}/${zome}_coordinator.wasm"
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
echo "  - backend/dna.dna (DNA bundle)"
echo "  - backend/mycelix_marketplace.happ (hApp bundle)"
echo ""
echo "🎉 All 10 zomes successfully built and packaged!"
echo ""
echo "📊 Build Statistics:"
find backend/zomes -name "*.wasm" -type f -exec ls -lh {} \; | awk '{print "  - "$9" ("$5")"}'
echo ""
echo "🚀 Next steps:"
echo "  1. Test with: holochain -c backend/conductor-config.yaml"
echo "  2. Or deploy to Holochain network"
echo ""
