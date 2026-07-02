#!/usr/bin/env bash
#
# Setup script for Holochain Conductor Integration Tests
# Byzantine-Robust Federated Learning System
#
# This script prepares the environment for full conductor testing
# with multi-agent Byzantine detection scenarios.
#
# Usage: ./scripts/setup-conductor-tests.sh
#
# Author: Luminous Dynamics Research Team
# Date: December 30, 2025

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOCHAIN_DIR="$(dirname "$SCRIPT_DIR")"
ZOMES_DIR="$HOLOCHAIN_DIR/zomes"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Byzantine-Robust FL: Holochain Conductor Test Setup          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
echo "=== Checking Prerequisites ==="

check_command() {
    if command -v "$1" &> /dev/null; then
        echo "  ✅ $1: $(command -v "$1")"
        return 0
    else
        echo "  ❌ $1: NOT FOUND"
        return 1
    fi
}

MISSING=0
check_command rustc || MISSING=1
check_command cargo || MISSING=1
check_command go || MISSING=1

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Missing prerequisites. Enter nix develop first:"
    echo "  cd $HOLOCHAIN_DIR && nix develop"
    exit 1
fi

echo ""
echo "=== Building WASM Zomes ==="

build_zome() {
    local zome_name=$1
    local zome_dir="$ZOMES_DIR/$zome_name"

    if [ -d "$zome_dir" ]; then
        echo "  Building $zome_name..."

        # Check if zome has its own workspace (standalone) or uses parent workspace
        if grep -q '^\[workspace\]' "$zome_dir/Cargo.toml" 2>/dev/null; then
            # Standalone workspace - builds to its own target
            (cd "$zome_dir" && cargo build --release --target wasm32-unknown-unknown 2>&1 | tail -3)
            local wasm_path="$zome_dir/target/wasm32-unknown-unknown/release/${zome_name}.wasm"
        else
            # Part of parent workspace - builds to parent target
            (cd "$HOLOCHAIN_DIR" && cargo build --release --target wasm32-unknown-unknown -p "$zome_name" 2>&1 | tail -3)
            local wasm_path="$HOLOCHAIN_DIR/target/wasm32-unknown-unknown/release/${zome_name}.wasm"
        fi

        if [ -f "$wasm_path" ]; then
            local size=$(du -h "$wasm_path" | cut -f1)
            echo "    ✅ $zome_name.wasm ($size)"
        else
            echo "    ❌ Build failed for $zome_name"
            return 1
        fi
    else
        echo "    ⚠️  $zome_name directory not found"
    fi
}

# Build all Byzantine defense zomes
build_zome "cbd_zome"
build_zome "reputation_tracker_v2"
build_zome "defense_coordinator"
build_zome "gradient_storage"

echo ""
echo "=== Running Unit Tests ==="

run_tests() {
    local zome_name=$1
    local zome_dir="$ZOMES_DIR/$zome_name"

    if [ -d "$zome_dir" ]; then
        echo "  Testing $zome_name..."
        local output=$(cd "$zome_dir" && cargo test --lib 2>&1)
        local result_line=$(echo "$output" | grep "test result:")
        if [[ "$result_line" == *"passed"* ]] && [[ "$result_line" != *"failed: 1"* ]]; then
            # Extract just "X passed" from "test result: ok. X passed; 0 failed..."
            local passed=$(echo "$result_line" | grep -oP '\d+ passed')
            echo "    ✅ $passed"
        else
            echo "    ❌ Tests failed"
            echo "$output" | tail -5
        fi
    fi
}

run_tests "cbd_zome"
run_tests "reputation_tracker_v2"
run_tests "defense_coordinator"

echo ""
echo "=== Creating DNA Manifest ==="

DNA_DIR="$HOLOCHAIN_DIR/dna/byzantine_defense"
mkdir -p "$DNA_DIR"

cat > "$DNA_DIR/dna.yaml" << 'EOF'
---
manifest_version: "0"
name: byzantine_defense
integrity:
  network_seed: ~
  properties: ~
  zomes: []
coordinator:
  zomes:
    - name: gradient_storage
      path: ../../target/wasm32-unknown-unknown/release/gradient_storage.wasm
      dependencies: []
    - name: cbd_zome
      path: ../../zomes/cbd_zome/target/wasm32-unknown-unknown/release/cbd_zome.wasm
      dependencies: []
    - name: reputation_tracker_v2
      path: ../../zomes/reputation_tracker_v2/target/wasm32-unknown-unknown/release/reputation_tracker_v2.wasm
      dependencies: []
    - name: defense_coordinator
      path: ../../zomes/defense_coordinator/target/wasm32-unknown-unknown/release/defense_coordinator.wasm
      dependencies: []
EOF

echo "  ✅ Created $DNA_DIR/dna.yaml"

echo ""
echo "=== Creating hApp Manifest ==="

HAPP_DIR="$HOLOCHAIN_DIR/happ/byzantine_defense"
mkdir -p "$HAPP_DIR"

cat > "$HAPP_DIR/happ.yaml" << 'EOF'
---
manifest_version: "0"
name: byzantine_defense
description: Byzantine-robust federated learning with 45% fault tolerance
roles:
  - name: fl_node
    provisioning:
      strategy: create
      deferred: false
    dna:
      path: ../../dna/byzantine_defense/byzantine_defense.dna
      modifiers:
        network_seed: ~
        properties: ~
EOF

echo "  ✅ Created $HAPP_DIR/happ.yaml"

echo ""
echo "=== Setup Summary ==="
echo ""
echo "WASM Zomes Built:"
ls -lh "$ZOMES_DIR"/*/target/wasm32-unknown-unknown/release/*.wasm 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'

echo ""
echo "DNA Manifest: $DNA_DIR/dna.yaml"
echo "hApp Manifest: $HAPP_DIR/happ.yaml"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     NEXT STEPS                                   ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
echo "║  1. Pack the DNA:                                                ║"
echo "║     hc dna pack $DNA_DIR                                         ║"
echo "║                                                                  ║"
echo "║  2. Pack the hApp:                                               ║"
echo "║     hc app pack $HAPP_DIR                                        ║"
echo "║                                                                  ║"
echo "║  3. Start a sandbox conductor:                                   ║"
echo "║     hc sandbox generate --root ./conductor-test                  ║"
echo "║     hc sandbox run                                               ║"
echo "║                                                                  ║"
echo "║  4. Install the hApp:                                            ║"
echo "║     hc sandbox call install-app byzantine_defense.happ           ║"
echo "║                                                                  ║"
echo "║  5. Run multi-agent tests (requires holochain_test_utils)        ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "For Python simulation tests (no conductor required):"
echo "  pytest holochain/tests/test_byzantine_integration.py -v"
echo ""
