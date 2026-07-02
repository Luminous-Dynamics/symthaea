#!/usr/bin/env bash
# Monitor Grand Slam experiment progress (CORRECTED VERSION)

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                   GRAND SLAM PROGRESS MONITOR (CORRECTED)                    ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if process is running
PID=$(ps aux | grep "python run_grand_slam.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ Grand Slam process not running"
    echo ""
    echo "To start it, run:"
    echo "  PYTHONUNBUFFERED=1 nohup nix develop --command python run_grand_slam.py > /tmp/grand_slam_corrected.log 2>&1 &"
    exit 1
fi

echo "✅ Grand Slam running (PID: $PID)"
echo ""

# Show CPU and memory usage
CPU=$(ps -p $PID -o %cpu | tail -1)
MEM=$(ps -p $PID -o %mem | tail -1)
echo "📊 Resource Usage:"
echo "   CPU: ${CPU}%"
echo "   Memory: ${MEM}%"
echo ""

# Count completed experiments
COMPLETED=$(grep -c "✅ Results saved to:" /tmp/grand_slam_corrected.log 2>/dev/null || echo "0")
echo "🎯 Progress: ${COMPLETED}/10 experiments completed (CORRECTED: was 11)"
echo ""

# Show current experiment
CURRENT=$(tail -100 /tmp/grand_slam_corrected.log | grep "Running:" | tail -1)
if [ -n "$CURRENT" ]; then
    echo "🔄 Current: $CURRENT"
fi
echo ""

# Show recent output
echo "📝 Recent Output (last 30 lines):"
echo "════════════════════════════════════════════════════════════════════════════════"
tail -30 /tmp/grand_slam_corrected.log
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Estimated time remaining
if [ "$COMPLETED" -gt "0" ]; then
    # Rough estimate: ~25 minutes per experiment
    REMAINING=$((10 - COMPLETED))
    MINUTES=$((REMAINING * 25))
    HOURS=$((MINUTES / 60))
    MINS=$((MINUTES % 60))
    echo "⏱️  Estimated time remaining: ~${HOURS}h ${MINS}m"
else
    echo "⏱️  Estimated total time: ~4-5 hours (CORRECTED: was 4-6 hours)"
fi
echo ""

echo "✅ FIXES APPLIED:"
echo "   • Dataset names: mnist, cifar10 (not MNIST, CIFAR-10)"
echo "   • Baseline names: pogq (not pogqrep)"
echo "   • Shakespeare removed (not implemented)"
echo "   • num_clients: 20 (was 10 in base config)"
echo "   • Multi-Krum Byzantine tolerance: 6/20 allowed"
echo ""

echo "To watch live updates:"
echo "  tail -f /tmp/grand_slam_corrected.log"
echo ""
echo "To re-run this monitor:"
echo "  bash monitor_grand_slam.sh"
