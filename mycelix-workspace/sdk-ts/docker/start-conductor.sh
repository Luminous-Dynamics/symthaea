#!/bin/bash
# Holochain Conductor Startup Script for Mycelix Integration Tests
#
# This script:
# 1. Starts the Holochain conductor
# 2. Waits for it to be ready
# 3. Installs the Mycelix hApp if provided
# 4. Keeps the conductor running

set -e

CONDUCTOR_CONFIG="/holochain/config/conductor-config.yaml"
DATA_DIR="/holochain/data"
HAPP_DIR="/holochain/happs"
APP_ID="${MYCELIX_APP_ID:-mycelix_ecosystem}"

echo "==================================="
echo "Mycelix Holochain Conductor"
echo "==================================="
echo "Holochain version: $(holochain --version || echo 'unknown')"
echo "Config: $CONDUCTOR_CONFIG"
echo "Data dir: $DATA_DIR"
echo "App ID: $APP_ID"
echo "==================================="

# Ensure data directory exists
mkdir -p "$DATA_DIR"

# Start conductor in background
echo "Starting Holochain conductor..."
holochain -c "$CONDUCTOR_CONFIG" -p "$DATA_DIR" &
CONDUCTOR_PID=$!

# Wait for conductor to be ready
echo "Waiting for conductor to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Try to connect to admin interface
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:8889/" 2>/dev/null | grep -q "400\|200"; then
        echo "Conductor is ready!"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for conductor... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 1
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: Conductor failed to start within $MAX_RETRIES seconds"
    exit 1
fi

# Install hApp if provided
if [ -f "$HAPP_DIR/mycelix.happ" ]; then
    echo "Installing Mycelix hApp..."
    hc sandbox call install-app \
        --app-id "$APP_ID" \
        --agent-key-hash $(hc sandbox call generate-agent-key 2>/dev/null | head -1) \
        "$HAPP_DIR/mycelix.happ" || echo "hApp installation skipped (may already be installed)"
    echo "hApp installation complete"
fi

# Generate app interface if needed
echo "Configuring app interface on port 8888..."
# The app interface is configured via admin calls during test setup

echo "==================================="
echo "Conductor is running"
echo "Admin interface: ws://localhost:8889"
echo "App interface: ws://localhost:8888"
echo "==================================="

# Keep the conductor running
wait $CONDUCTOR_PID
