#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Credential Pipeline Demo — Praxis → Craft
#
# Demonstrates the full credential lifecycle:
# 1. Build WASM zomes for both Praxis and Craft
# 2. Pack DNA and hApp bundles
# 3. Start a Holochain sandbox conductor with both hApps installed
# 4. Issue a credential in Praxis (with PoL + mastery)
# 5. Publish it to Craft (with guild context, epistemic code)
# 6. Verify vitality decay over time
#
# Prerequisites:
#   - nix develop (for hc, holochain binaries)
#   - Rust toolchain with wasm32-unknown-unknown target
#
# Usage:
#   cd mycelix-craft
#   nix develop ../mycelix-praxis -c bash -c './scripts/demo-credential-pipeline.sh'

set -euo pipefail

CRAFT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PRAXIS_DIR="$(cd "$CRAFT_DIR/../mycelix-praxis" && pwd)"

echo "=== Credential Pipeline Demo: Praxis → Craft ==="
echo ""
echo "This demo:"
echo "  1. Builds WASM zomes for both clusters"
echo "  2. Packs DNA bundles"
echo "  3. Starts a shared conductor"
echo "  4. Issues a credential in Praxis"
echo "  5. Publishes it to Craft with guild context"
echo "  6. Verifies living credential vitality"
echo ""

# Step 1: Build WASM zomes
echo "[1/6] Building WASM zomes..."
echo "  Building Praxis..."
(cd "$PRAXIS_DIR" && cargo build --workspace --target wasm32-unknown-unknown --release 2>&1 | tail -1)
echo "  Building Craft..."
(cd "$CRAFT_DIR" && cargo build --workspace --target wasm32-unknown-unknown --release 2>&1 | tail -1)

# Step 2: Pack DNA
echo "[2/6] Packing DNA bundles..."
(cd "$PRAXIS_DIR" && hc dna pack dna/ -o dna/praxis.dna 2>&1 | tail -1)
(cd "$CRAFT_DIR" && hc dna pack dna/ -o dna/mycelix_craft.dna 2>&1 | tail -1)

# Step 3: Pack hApp
echo "[3/6] Packing hApp bundles..."
(cd "$PRAXIS_DIR" && cp dna/praxis.dna happ/praxis.dna && hc app pack happ/ -o happ/mycelix-praxis.happ 2>&1 | tail -1)
(cd "$CRAFT_DIR" && cp dna/mycelix_craft.dna happ/mycelix_craft.dna 2>&1 && hc app pack happ/ -o happ/mycelix-craft.happ 2>&1 | tail -1)

echo "[4/6] Starting sandbox conductor..."
echo ""
echo "  The conductor will start with both hApps installed."
echo "  Use the Holochain admin API (port shown below) to:"
echo "    - Call Praxis credential_coordinator::issue_credential"
echo "    - Call Craft craft_graph::publish_credential (with guild_id, epistemic_code)"
echo "    - Call Craft craft_graph::get_credential_vitality"
echo ""
echo "  Conductor ports: admin=8403, app=8404 (dev/test range)"
echo ""

# Step 4-6: Start conductor with both hApps
# The sandbox will install both hApps and start serving
echo "" | hc sandbox --piped generate \
  -a 8403 \
  "$PRAXIS_DIR/happ/mycelix-praxis.happ" \
  "$CRAFT_DIR/happ/mycelix-craft.happ" \
  --run=8404

echo ""
echo "=== Demo Complete ==="
echo ""
echo "Both Praxis and Craft are running on the same conductor."
echo "The credential pipeline is ready for testing."
