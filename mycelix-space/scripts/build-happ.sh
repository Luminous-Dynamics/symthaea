#!/usr/bin/env bash
# Build script for Mycelix Space hApp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Add cargo bin to PATH if not already there
export PATH="$HOME/.cargo/bin:$PATH"

echo "========================================"
echo "  Mycelix Space hApp Build"
echo "========================================"
echo ""

# Check for required tools
echo "Checking prerequisites..."
command -v cargo >/dev/null 2>&1 || { echo "ERROR: cargo not found"; exit 1; }
command -v hc >/dev/null 2>&1 || { echo "ERROR: hc (Holochain CLI) not found. Install with: cargo install holochain_cli"; exit 1; }

# Check for wasm32 target
if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
    echo "Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

echo "Prerequisites OK"
echo ""

# Step 1: Build all zomes to WASM
echo "Step 1/3: Compiling zomes to WASM..."
echo "  Building integrity zomes..."
cargo build --release --target wasm32-unknown-unknown \
    -p orbital_objects_integrity \
    -p observations_integrity \
    -p conjunctions_integrity \
    -p debris_bounties_integrity \
    -p traffic_control_integrity

echo "  Building coordinator zomes..."
cargo build --release --target wasm32-unknown-unknown \
    -p orbital_objects_coordinator \
    -p observations_coordinator \
    -p conjunctions_coordinator \
    -p debris_bounties_coordinator \
    -p traffic_control_coordinator

echo "  WASM compilation complete!"
echo ""

# List generated WASM files
echo "  Generated WASM files:"
ls -lh target/wasm32-unknown-unknown/release/*.wasm 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
echo ""

# Step 2: Pack the DNA
echo "Step 2/3: Packing DNA..."
hc dna pack workdir/dna -o workdir/dna/mycelix_space.dna
echo "  DNA packed: workdir/dna/mycelix_space.dna"
echo ""

# Step 3: Pack the hApp
echo "Step 3/3: Packing hApp..."
hc app pack workdir/happ -o workdir/happ/mycelix_space.happ
echo "  hApp packed: workdir/happ/mycelix_space.happ"
echo ""

echo "========================================"
echo "  Build Complete!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  DNA:  $(ls -lh workdir/dna/mycelix_space.dna | awk '{print $9 " (" $5 ")"}')"
echo "  hApp: $(ls -lh workdir/happ/mycelix_space.happ | awk '{print $9 " (" $5 ")"}')"
echo ""
echo "Next steps:"
echo "  1. Test with sandbox: hc sandbox generate workdir/happ/mycelix_space.happ"
echo "  2. Or install in conductor: hc app install workdir/happ/mycelix_space.happ"
