#!/usr/bin/env bash
# Relaunch Grand Slam after fixing path resolution issue

set -e

echo "🔧 Relaunching Grand Slam Experimental Matrix"
echo "=============================================="

# Navigate to experiments directory
cd /srv/luminous-dynamics/Mycelix-Protocal-Framework/0TML/experiments

# Kill any existing Grand Slam processes
echo "1️⃣  Checking for existing Grand Slam processes..."
if pgrep -f "python.*run_grand_slam.py" > /dev/null; then
    echo "   Found running process, killing it..."
    pkill -f "python.*run_grand_slam.py" || true
    sleep 2
fi

# Clean up old log
echo ""
echo "2️⃣  Cleaning up old log file..."
rm -f /tmp/grand_slam_corrected.log

# Launch Grand Slam in background
echo ""
echo "3️⃣  Launching Grand Slam (this will take ~2 hours)..."
nohup nix develop --command python run_grand_slam.py > /tmp/grand_slam_corrected.log 2>&1 &
GRAND_SLAM_PID=$!

echo ""
echo "✅ Grand Slam relaunched!"
echo "   PID: $GRAND_SLAM_PID"
echo "   Log: /tmp/grand_slam_corrected.log"
echo ""
echo "Monitor progress with:"
echo "   tail -f /tmp/grand_slam_corrected.log"
echo ""
echo "Or check the last 50 lines:"
echo "   tail -50 /tmp/grand_slam_corrected.log"
