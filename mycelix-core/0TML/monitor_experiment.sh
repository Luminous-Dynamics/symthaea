#!/usr/bin/env bash
# Monitor MNIST IID experiment progress
# Usage: ./monitor_experiment.sh

LOG_FILE="/tmp/mnist_iid_complete.log"

echo "🔍 Hybrid ZeroTrustML Experiment Monitor"
echo "======================================"
echo ""

# Check if experiment is running
if ps aux | grep -q '[p]ython -u experiments/runner.py'; then
    PID=$(ps aux | grep '[p]ython -u experiments/runner.py' | awk '{print $2}' | head -1)
    CPU=$(ps aux | grep '[p]ython -u experiments/runner.py' | awk '{print $3}' | head -1)
    RUNTIME=$(ps aux | grep '[p]ython -u experiments/runner.py' | awk '{print $10}' | head -1)

    echo "✅ Status: RUNNING"
    echo "📊 PID: $PID"
    echo "💻 CPU: ${CPU}%"
    echo "⏱️  Runtime: $RUNTIME"
    echo ""
else
    echo "❌ Status: NOT RUNNING"
    echo ""
    exit 1
fi

# Show latest progress from log
if [ -f "$LOG_FILE" ]; then
    echo "📝 Latest Progress:"
    echo "==================="

    # Extract last baseline and rounds
    BASELINE=$(grep -o 'Baseline: [A-Z]*' "$LOG_FILE" | tail -1 | cut -d' ' -f2)
    LAST_ROUND=$(grep -oP 'Round\s+\K\d+' "$LOG_FILE" | tail -1)

    if [ ! -z "$BASELINE" ]; then
        echo "🎯 Current Baseline: $BASELINE"
    fi

    if [ ! -z "$LAST_ROUND" ]; then
        echo "🔄 Last Round: $LAST_ROUND/100"

        # Calculate percentage
        PCT=$((LAST_ROUND * 100 / 100))
        echo "📈 Progress: ${PCT}% complete for this baseline"
    fi

    echo ""
    echo "📊 Last 10 Lines of Log:"
    echo "========================"
    tail -10 "$LOG_FILE" | grep -E '(Round|Baseline|Test Acc)'

else
    echo "⚠️  Log file not found: $LOG_FILE"
fi

echo ""
echo "💡 Commands:"
echo "   tail -f $LOG_FILE           # Watch live progress"
echo "   ./monitor_experiment.sh      # Check current status"
echo "   kill -9 $PID                 # Stop experiment (if needed)"
