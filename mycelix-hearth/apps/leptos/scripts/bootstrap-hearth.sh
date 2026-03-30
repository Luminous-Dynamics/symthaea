#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Bootstrap a test Hearth with sample data.
#
# Run from within `nix develop` in the mycelix-hearth directory:
#   cd mycelix-hearth
#   nix develop
#   bash apps/leptos/scripts/bootstrap-hearth.sh
#
# This script:
# 1. Starts a Holochain sandbox with the Hearth hApp on port 8888
# 2. The frontend at :8096 auto-detects the conductor
# 3. Real zome calls replace mock data
#
# For multi-agent testing (e.g., with Pixel 8 Pro):
#   hc sandbox generate --app-port 8889 mycelix-hearth.happ --run
#   (second conductor on a different port)

set -euo pipefail

HAPP_PATH="${1:-mycelix-hearth.happ}"
APP_PORT="${2:-8888}"

echo "=== Hearth Bootstrap ==="
echo ""
echo "hApp:     ${HAPP_PATH}"
echo "App port: ${APP_PORT}"
echo ""

if [ ! -f "${HAPP_PATH}" ]; then
    echo "ERROR: hApp not found at ${HAPP_PATH}"
    echo "Run from the mycelix-hearth directory."
    exit 1
fi

if ! command -v hc &>/dev/null; then
    echo "ERROR: 'hc' not found. Enter the nix shell first:"
    echo "  cd mycelix-hearth && nix develop"
    exit 1
fi

echo "Starting Holochain sandbox..."
echo "  The frontend at http://localhost:8096 will auto-detect this conductor."
echo "  Press Ctrl+C to stop."
echo ""

# Generate sandbox and run
# --app-port: the WebSocket port the frontend connects to
# The sandbox creates a temp directory, installs the hApp, and starts the conductor
hc sandbox generate \
    --app-port "${APP_PORT}" \
    "${HAPP_PATH}" \
    --run

echo ""
echo "Conductor stopped."
