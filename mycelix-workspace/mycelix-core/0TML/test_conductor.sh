#!/usr/bin/env bash
# Test conductor startup script

echo "=== Testing Conductor Config ==="
holochain --config-path conductor-config.yaml 2>&1 | head -10 &
CONDUCTOR_PID=$!
echo "Started conductor with PID: $CONDUCTOR_PID"

# Wait a bit for startup
sleep 3

# Check if still running
if ps -p $CONDUCTOR_PID > /dev/null; then
    echo "✅ Conductor is running!"
    # Kill it
    kill $CONDUCTOR_PID
    echo "Conductor stopped"
else
    echo "❌ Conductor failed to start or crashed"
fi
