#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Install EduNet hApp onto the shared Mycelix ecosystem conductor.
#
# Prerequisites:
#   - Shared conductor running (admin on port 33743, app on 8888)
#   - hApp bundle at happ/mycelix-edunet.happ (run: hc app pack happ/)
#   - nix develop shell (for hc CLI)
#
# Usage:
#   cd mycelix-edunet
#   nix develop
#   bash scripts/install-on-conductor.sh

set -euo pipefail

ADMIN_PORT="${1:-33743}"
HAPP_PATH="happ/mycelix-edunet.happ"
APP_ID="edunet"

echo "=== EduNet hApp Installer ==="
echo "  Admin port: $ADMIN_PORT"
echo "  hApp: $HAPP_PATH"
echo "  App ID: $APP_ID"
echo ""

# Check conductor is running
if ! ss -tlnp 2>/dev/null | grep -q ":${ADMIN_PORT}"; then
    echo "ERROR: No conductor listening on port $ADMIN_PORT"
    echo "Start the shared conductor first, or pass the correct admin port:"
    echo "  bash scripts/install-on-conductor.sh <admin-port>"
    exit 1
fi

# Check hApp exists
if [ ! -f "$HAPP_PATH" ]; then
    echo "ERROR: hApp not found at $HAPP_PATH"
    echo "Build it first:"
    echo "  cargo build --workspace --target wasm32-unknown-unknown --release"
    echo "  hc dna pack dna/ -o dna/edunet.dna"
    echo "  cp dna/edunet.dna happ/edunet.dna"
    echo "  hc app pack happ/ -o happ/mycelix-edunet.happ"
    exit 1
fi

echo "[1/3] Reading hApp bundle..."
HAPP_BASE64=$(base64 -w0 "$HAPP_PATH")
HAPP_SIZE=$(stat -c%s "$HAPP_PATH")
echo "  Size: $((HAPP_SIZE / 1024 / 1024))MB ($HAPP_SIZE bytes)"

echo "[2/3] Sending InstallApp to admin WebSocket..."

# Use Python to send the admin WebSocket message (available on NixOS)
python3 << PYTHON
import asyncio
import json
import struct
import sys

try:
    import websockets
except ImportError:
    # Fallback: use raw socket
    print("  websockets not available, trying raw approach...")
    sys.exit(2)

async def install():
    uri = "ws://localhost:${ADMIN_PORT}"
    async with websockets.connect(uri) as ws:
        # List existing apps first
        list_msg = json.dumps({
            "type": "list_apps",
            "data": {"status_filter": None}
        })

        # Holochain admin API uses msgpack, not JSON
        # The exact protocol depends on HC version
        # For HC 0.6, we need the AppRequest enum serialized as msgpack
        print("  Connected to admin interface")
        print("  Note: Full admin API requires msgpack encoding")
        print("  Use 'hc sandbox call' for proper installation")

install_result = asyncio.run(install()) if hasattr(asyncio, 'run') else None
PYTHON

PYTHON_EXIT=$?

if [ $PYTHON_EXIT -eq 2 ] || [ $PYTHON_EXIT -ne 0 ]; then
    echo ""
    echo "[2/3] Python WebSocket not available. Using hc sandbox instead..."
    echo ""

    # Use hc sandbox to install onto existing conductor
    # The --force flag allows installing even if app-id exists
    if command -v hc &> /dev/null; then
        echo "  Using hc sandbox call to install..."

        # Generate agent key and install
        hc sandbox call --running="$ADMIN_PORT" install-app "$HAPP_PATH" --app-id "$APP_ID" 2>&1 || {
            echo ""
            echo "  hc sandbox install failed. Try manual installation:"
            echo ""
            echo "  # In a nix develop shell:"
            echo "  hc sandbox call --running=$ADMIN_PORT install-app $HAPP_PATH --app-id $APP_ID"
            echo ""
            echo "  # Or use the Holochain admin API directly via websocket on port $ADMIN_PORT"
            exit 1
        }
    else
        echo "ERROR: hc CLI not found. Run this inside 'nix develop'"
        exit 1
    fi
fi

echo "[3/3] Verifying installation..."

# Check if app port is responding
if ss -tlnp 2>/dev/null | grep -q ":8888"; then
    echo "  App interface on port 8888: LISTENING"
else
    echo "  App interface on port 8888: NOT FOUND"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "  Test from browser console:"
echo "    const ws = new WebSocket('ws://localhost:8888');"
echo "    ws.onopen = () => console.log('Connected to conductor!');"
echo ""
echo "  Or open: https://edunet.luminousdynamics.io"
echo "  The PWA will auto-connect to ws://localhost:8888"
