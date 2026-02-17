#!/usr/bin/env bash
# Launch Mycelix Desktop in Development Mode
# This shows detailed error messages

cd "$(dirname "$0")"

echo "🔧 Launching Mycelix Desktop in DEVELOPMENT MODE..."
echo "This will show detailed error messages"
echo ""
echo "Location: $(pwd)"
echo ""

# Disable GPU hardware acceleration to fix GBM buffer error
export WEBKIT_DISABLE_COMPOSITING_MODE=1
export LIBGL_ALWAYS_SOFTWARE=1

# Enter nix environment and run dev mode
nix develop --command bash -c "npm run tauri dev"
