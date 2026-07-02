#!/usr/bin/env bash
# Preview the CAPS Matric Explorer in a headless browser and save screenshots.
# Usage: ./scripts/preview-caps.sh [output-dir]
#
# Starts a local server, takes screenshots, stops the server.
# Requires: chromium (headless), python3

set -euo pipefail
CAPS_DIR="$(cd "$(dirname "$0")/../examples/curriculum/caps" && pwd)"
OUT="${1:-/tmp}"
PORT=8092

# Start server
python3 -m http.server "$PORT" -d "$CAPS_DIR" &>/dev/null &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT
sleep 1

echo "Taking screenshots..."

# Full page
chromium --headless --disable-gpu --no-sandbox \
  --screenshot="$OUT/matric-explorer-full.png" \
  --window-size=1920,1080 \
  --virtual-time-budget=5000 \
  "http://localhost:$PORT/matric-explorer.html" 2>/dev/null

echo "  -> $OUT/matric-explorer-full.png"

# Mobile view
chromium --headless --disable-gpu --no-sandbox \
  --screenshot="$OUT/matric-explorer-mobile.png" \
  --window-size=390,844 \
  --virtual-time-budget=5000 \
  "http://localhost:$PORT/matric-explorer.html" 2>/dev/null

echo "  -> $OUT/matric-explorer-mobile.png"

echo "Done. Preview at: $OUT/matric-explorer-full.png"
