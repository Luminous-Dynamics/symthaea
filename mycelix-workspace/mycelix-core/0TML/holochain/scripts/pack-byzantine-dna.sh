#!/usr/bin/env bash
# Pack Byzantine Defense DNA and hApp bundles using Holochain 0.6
# Usage: ./scripts/pack-byzantine-dna.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

DNA_DIR="$PROJECT_ROOT/dna/byzantine_defense"
HAPP_DIR="$PROJECT_ROOT/happ/byzantine_defense"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Byzantine-Robust FL: DNA & hApp Packaging (Holochain 0.6)   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check we have the required tools
if ! command -v hc &> /dev/null; then
    echo "❌ 'hc' command not found. Enter holonix environment first:"
    echo "   nix develop .#holonix"
    exit 1
fi

echo "🔧 Holochain Tools:"
echo "   HC CLI: $(hc --version 2>/dev/null || echo 'unknown')"
echo ""

# Check WASM files exist
echo "=== Checking WASM Zomes ==="
ZOMES=(
    "gradient_storage"
    "cbd_zome"
    "reputation_tracker_v2"
    "defense_coordinator"
)

for zome in "${ZOMES[@]}"; do
    wasm_path="$PROJECT_ROOT/zomes/$zome/target/wasm32-unknown-unknown/release/${zome}.wasm"
    if [ -f "$wasm_path" ]; then
        size=$(ls -lh "$wasm_path" | awk '{print $5}')
        echo "   ✅ $zome.wasm ($size)"
    else
        echo "   ❌ $zome.wasm NOT FOUND"
        echo "   Run ./scripts/setup-conductor-tests.sh first"
        exit 1
    fi
done
echo ""

# Pack DNA
echo "=== Packing DNA ==="
cd "$DNA_DIR"
hc dna pack .
if [ -f "byzantine_defense.dna" ]; then
    size=$(ls -lh "byzantine_defense.dna" | awk '{print $5}')
    echo "   ✅ byzantine_defense.dna ($size)"
else
    echo "   ❌ DNA packing failed"
    exit 1
fi
echo ""

# Pack hApp
echo "=== Packing hApp ==="
cd "$HAPP_DIR"
hc app pack .
if [ -f "byzantine_defense.happ" ]; then
    size=$(ls -lh "byzantine_defense.happ" | awk '{print $5}')
    echo "   ✅ byzantine_defense.happ ($size)"
else
    echo "   ❌ hApp packing failed"
    exit 1
fi
echo ""

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     PACKAGING COMPLETE                          ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
echo "║  DNA:  $DNA_DIR/byzantine_defense.dna"
echo "║  hApp: $HAPP_DIR/byzantine_defense.happ"
echo "║                                                                  ║"
echo "║  Next steps:                                                     ║"
echo "║    1. Create sandbox:  hc sandbox generate --root ./conductor    ║"
echo "║    2. Run conductor:   hc sandbox run                            ║"
echo "║    3. Install hApp:    hc sandbox call install-app byzantine_defense.happ"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
