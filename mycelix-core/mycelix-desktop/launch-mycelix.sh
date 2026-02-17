#!/usr/bin/env bash
# Mycelix Desktop Launcher

cd "$(dirname "$0")"

echo "🚀 Launching Mycelix Desktop..."
echo "Location: $(pwd)"
echo ""

# Disable GPU hardware acceleration to fix GBM buffer error
# This allows the app to run using software rendering
export WEBKIT_DISABLE_COMPOSITING_MODE=1
export LIBGL_ALWAYS_SOFTWARE=1

# Run the app in foreground so you see any errors
./target/release/mycelix-desktop

# If we get here, the app closed
read -p "Press Enter to close this window..."
