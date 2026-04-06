#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Activate Praxis + Craft on the shared Holochain conductor
#
# Usage (from nix develop shell):
#   nix develop ./mycelix-praxis -c bash -c './scripts/activate-conductor.sh'
#
# Prerequisites:
#   - Holochain conductor running (admin on 33800, app on 8888)
#   - nix develop shell with hc, holochain, lair-keystore

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADMIN_PORT="${ADMIN_PORT:-33800}"

echo "=== Mycelix Conductor Activation ==="
echo "Admin port: $ADMIN_PORT"
echo ""

# Check conductor is running
if ! curl -s --max-time 2 "http://127.0.0.1:$ADMIN_PORT" > /dev/null 2>&1; then
    # WebSocket check (curl won't work for WS, but port should be open)
    if ! nc -z 127.0.0.1 "$ADMIN_PORT" 2>/dev/null; then
        echo "ERROR: No conductor on port $ADMIN_PORT"
        echo "Start one with: holochain -c conductor-config.yaml"
        exit 1
    fi
fi

# 1. Build WASM
echo "[1/5] Building WASM zomes..."
echo "  Craft (14 crates)..."
(cd "$ROOT/mycelix-craft" && cargo build --workspace --target wasm32-unknown-unknown --release 2>&1 | tail -1)

echo "  Praxis (29 crates)..."
(cd "$ROOT/mycelix-praxis" && cargo build --workspace --target wasm32-unknown-unknown --release 2>&1 | tail -1)

# 2. Pack DNA
echo "[2/5] Packing DNA bundles..."
(cd "$ROOT/mycelix-craft" && hc dna pack dna/ -o dna/mycelix_craft.dna 2>&1)
(cd "$ROOT/mycelix-praxis" && hc dna pack dna/ -o dna/praxis.dna 2>&1)
echo "  Craft DNA: $(ls -lh "$ROOT/mycelix-craft/dna/mycelix_craft.dna" | awk '{print $5}')"
echo "  Praxis DNA: $(ls -lh "$ROOT/mycelix-praxis/dna/praxis.dna" | awk '{print $5}')"

# 3. Pack hApp
echo "[3/5] Packing hApp bundles..."
(cd "$ROOT/mycelix-craft" && hc app pack . -o mycelix-craft.happ 2>&1)
(cd "$ROOT/mycelix-praxis" && hc app pack happ/ -o happ/mycelix-praxis.happ 2>&1)

# 4. Install
echo "[4/5] Installing on conductor (admin=$ADMIN_PORT)..."
hc sandbox call --running="$ADMIN_PORT" install-app "$ROOT/mycelix-craft/mycelix-craft.happ" --app-id mycelix-craft 2>/dev/null && echo "  Craft: installed" || echo "  Craft: already installed or error"
hc sandbox call --running="$ADMIN_PORT" install-app "$ROOT/mycelix-praxis/happ/mycelix-praxis.happ" --app-id mycelix-praxis 2>/dev/null && echo "  Praxis: installed" || echo "  Praxis: already installed or error"

# Enable
hc sandbox call --running="$ADMIN_PORT" enable-app --app-id mycelix-craft 2>/dev/null && echo "  Craft: enabled" || echo "  Craft: already enabled or error"
hc sandbox call --running="$ADMIN_PORT" enable-app --app-id mycelix-praxis 2>/dev/null && echo "  Praxis: enabled" || echo "  Praxis: already enabled or error"

# 5. Verify
echo "[5/5] Verification..."
echo "  Installed apps:"
hc sandbox call --running="$ADMIN_PORT" list-apps 2>/dev/null || echo "  (list-apps not available via sandbox, check admin API)"

echo ""
echo "=== Activation Complete ==="
echo ""
echo "  Conductor:  ws://localhost:8888 (app) / ws://localhost:$ADMIN_PORT (admin)"
echo "  Praxis UI:  http://localhost:8107 → https://praxis.mycelix.net"
echo "  Craft UI:   http://localhost:8129 → https://craft.mycelix.net"
echo ""
echo "  Test profile round-trip:"
echo "    Open https://craft.mycelix.net/profile"
echo "    Fill in name, click 'Save & Publish'"
echo "    Check browser console for '[Craft] Connected' status"
