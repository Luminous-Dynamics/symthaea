#!/usr/bin/env bash
# Resume Matrix Experiment Runner with Checkpoint Recovery
# Usage: ./resume_matrix.sh [config.yaml]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG="${1:-configs/sanity_slice.yaml}"

echo "🔄 Matrix Experiment Resume Script"
echo "=================================="
echo ""

# Count completed experiments
COMPLETED=$(ls -d results/artifacts_* 2>/dev/null | wc -l)
NEXT_EXP=$((COMPLETED + 1))

echo "📊 Progress Status:"
echo "   Completed experiments: $COMPLETED"
echo "   Next experiment to run: $NEXT_EXP"
echo "   Config: $CONFIG"
echo ""

if [ $COMPLETED -eq 0 ]; then
    echo "⚠️  No completed experiments found. Starting from beginning..."
    START=1
else
    echo "✅ Found $COMPLETED completed experiments"
    echo "🚀 Resuming from experiment $NEXT_EXP"
    START=$NEXT_EXP
fi

echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

echo ""
echo "🏃 Starting experiments from #$START..."
echo ""

# Run with nohup and tee for background execution
nohup python experiments/matrix_runner.py \
    --config "$CONFIG" \
    --start "$START" \
    2>&1 | tee "/tmp/matrix_runner_resume_$(date +%Y%m%d_%H%M%S).log" &

PID=$!
echo "✅ Experiments launched in background (PID: $PID)"
echo "📝 Log: /tmp/matrix_runner_resume_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "Monitor progress:"
echo "  tail -f /tmp/matrix_runner_resume_*.log"
echo "  ps aux | grep $PID"
echo ""
