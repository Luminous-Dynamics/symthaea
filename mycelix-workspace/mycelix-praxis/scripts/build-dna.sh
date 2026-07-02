#!/usr/bin/env bash
# Build and package Mycelix EduNet DNA
# Usage: ./scripts/build-dna.sh

set -euo pipefail

echo "🎓 Mycelix EduNet - DNA Build Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Step 1: Build all WASM targets
echo "📦 Step 1/3: Building WASM targets..."
echo ""
cargo build --target wasm32-unknown-unknown --release

# Check WASM files exist
echo ""
echo "✅ WASM files built:"
find target/wasm32-unknown-unknown/release/ -name "*.wasm" -type f -exec ls -lh {} \;
echo ""

# Step 2: Package DNA
echo "📦 Step 2/3: Packaging DNA bundle..."
echo ""
cd dna
hc dna pack .

# Step 3: Verify DNA bundle
echo ""
echo "📦 Step 3/3: Verifying DNA bundle..."
echo ""
if [ -f edunet.dna ]; then
    DNA_SIZE=$(du -h edunet.dna | cut -f1)
    DNA_HASH=$(hc dna hash edunet.dna 2>/dev/null || echo "Hash calculation skipped")

    echo "✅ DNA bundle created successfully!"
    echo ""
    echo "📊 DNA Information:"
    echo "  File: dna/edunet.dna"
    echo "  Size: $DNA_SIZE"
    if [ "$DNA_HASH" != "Hash calculation skipped" ]; then
        echo "  Hash: $DNA_HASH"
    fi
    echo ""
else
    echo "❌ Error: DNA bundle not created!"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 DNA build complete!"
echo ""
echo "Next steps:"
echo "  • Run conductor: ./scripts/run-conductor.sh"
echo "  • Install DNA: hc app install dna/edunet.dna"
echo ""
