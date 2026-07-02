#!/usr/bin/env bash
# Start Holochain Conductor - Production Ready
# Config validated October 3, 2025

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/conductor-config-minimal.yaml}"
ENV_PATH="/tmp/holochain-zerotrustml"

echo "🚀 Starting Holochain Conductor"
echo "================================"
echo ""

# Create environment directory
mkdir -p "$ENV_PATH"
echo "✅ Environment: $ENV_PATH"

# Check config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config not found: $CONFIG_FILE"
    echo ""
    echo "Available configs:"
    ls -la "$SCRIPT_DIR"/*.yaml 2>/dev/null || echo "   No config files found"
    exit 1
fi
echo "✅ Config: $CONFIG_FILE"

# Check holochain is installed
if ! command -v holochain &> /dev/null; then
    echo "❌ holochain not found in PATH"
    echo ""
    echo "Install with:"
    echo "  nix develop github:holochain/holonix"
    echo "  OR"
    echo "  cargo install holochain --version 0.5.6"
    exit 1
fi

# Show version
HOLOCHAIN_VERSION=$(holochain --version 2>&1 | head -1)
echo "✅ Holochain: $HOLOCHAIN_VERSION"

# Start conductor
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Starting conductor..."
echo "   Config: $(basename "$CONFIG_FILE")"
echo "   Admin Port: 8888 (WebSocket)"
echo "   Data Path: $ENV_PATH"
echo ""
echo "   Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start with error handling
if holochain --config-path "$CONFIG_FILE"; then
    echo ""
    echo "✅ Conductor stopped gracefully"
else
    EXIT_CODE=$?
    echo ""
    echo "❌ Conductor exited with code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  - Port 8888 already in use (check: lsof -i :8888)"
    echo "  - Config file invalid (check YAML syntax)"
    echo "  - Permissions on $ENV_PATH"
    echo ""
    exit $EXIT_CODE
fi
