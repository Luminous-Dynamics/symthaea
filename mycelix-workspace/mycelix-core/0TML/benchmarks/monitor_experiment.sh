#!/usr/bin/env bash
# Monitor MNIST experiment progress

PID=178071
LOG_FILE="/tmp/mnist_experiment_final.log"

echo "📊 Monitoring MNIST Experiment"
echo "================================"
echo ""

# Check if process is running
if ! ps -p $PID > /dev/null 2>&1; then
    echo "✅ Experiment COMPLETE!"
    echo ""
    echo "📄 Final output:"
    tail -50 "$LOG_FILE"
    echo ""
    echo "📁 Results location:"
    ls -lh /srv/luminous-dynamics/Mycelix-Core/0TML/benchmarks/results/
    exit 0
fi

# Show process stats
echo "🏃 Experiment RUNNING"
ps -p $PID -o pid,etime,cputime,pcpu,rss --no-headers | \
    awk '{printf "  PID: %s\n  Elapsed: %s\n  CPU Time: %s\n  CPU: %s%%\n  Memory: %d MB\n", $1, $2, $3, $4, $5/1024}'
echo ""

# Show latest log output
echo "📝 Latest output (last 30 lines):"
echo "-----------------------------------"
tail -30 "$LOG_FILE"
echo ""

# Estimate completion
ELAPSED_SECONDS=$(ps -p $PID -o etimes --no-headers)
ESTIMATED_TOTAL=1500  # ~25 minutes
REMAINING=$((ESTIMATED_TOTAL - ELAPSED_SECONDS))
if [ $REMAINING -gt 0 ]; then
    REMAINING_MIN=$((REMAINING / 60))
    echo "⏱️  Estimated time remaining: ~$REMAINING_MIN minutes"
else
    echo "⏱️  Should complete soon!"
fi

echo ""
echo "💡 Tip: Run 'watch -n 10 ./monitor_experiment.sh' to monitor continuously"
