#!/usr/bin/env bash
set -euo pipefail

# Start Holochain conductor with Mycelix EduNet DNA
# This script must be run inside `nix develop`

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(realpath "$SCRIPT_DIR/..")"

echo "🎓 Mycelix EduNet - Starting Holochain Conductor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Project: $PROJECT_DIR"
echo ""

# Package the hApp if not already done or if DNA is newer
HAPP_FILE="$PROJECT_DIR/happ/mycelix-edunet.happ"
DNA_FILE="$PROJECT_DIR/dna/edunet.dna"

if [[ ! -f "$HAPP_FILE" ]] || [[ "$DNA_FILE" -nt "$HAPP_FILE" ]]; then
    echo "📦 Packaging hApp..."
    hc app pack "$PROJECT_DIR/happ/" -o "$HAPP_FILE"
    echo "   ✅ hApp packaged: $HAPP_FILE"
else
    echo "📦 hApp already up to date: $HAPP_FILE"
fi

echo ""
echo "🧹 Cleaning previous sandbox..."
hc sandbox clean 2>/dev/null || true

echo ""
echo "🚀 Generating new sandbox and installing hApp..."
echo "   Admin WebSocket: ws://localhost:8888"
echo "   App WebSocket:   ws://localhost:8889"
echo ""

# Generate sandbox with the hApp installed
# Using --piped to avoid interactive passphrase prompts
# Force ports 8888 (admin) and 8889 (app)
hc sandbox --piped generate \
    --root "$PROJECT_DIR/.hc-sandbox" \
    --directories 1 \
    -a 8889 \
    "$HAPP_FILE" \
    --run=8888

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Press Ctrl+C to stop the conductor"
