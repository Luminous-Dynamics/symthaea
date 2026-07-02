#!/bin/bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# DeSci Portal Deployment Script
# Builds the Leptos frontend and prepares for production serving.
#
# Usage: ./deploy.sh
#
# After running, restart the systemd services:
#   sudo systemctl restart desci-spa
#   sudo systemctl restart mycelix-api  (if API service is configured)
#
# Then seed with real claims:
#   ./scripts/seed_claims.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/apps/leptos"
DIST_DIR="$FRONTEND_DIR/dist"

echo "=== DeSci Portal Deployment ==="
echo ""

# 1. Build frontend
echo "[1/3] Building Leptos frontend (trunk build --release)..."
cd "$FRONTEND_DIR"
~/.cargo/bin/trunk build --release
echo "      Built to: $DIST_DIR"

# 2. Check dist exists
if [ ! -f "$DIST_DIR/index.html" ]; then
    echo "ERROR: dist/index.html not found!"
    exit 1
fi

WASM_SIZE=$(du -sh "$DIST_DIR"/*.wasm 2>/dev/null | head -1 | awk '{print $1}')
echo "      WASM size: $WASM_SIZE"
echo ""

# 3. Instructions
echo "[2/3] Restart services:"
echo "      sudo systemctl restart desci-spa"
echo ""
echo "[3/3] Seed with claims:"
echo "      ./scripts/seed_claims.sh"
echo ""
echo "=== Portal will be live at https://desci.mycelix.net ==="
