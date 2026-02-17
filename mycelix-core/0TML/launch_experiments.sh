#!/usr/bin/env bash
#
# Launch Experiments with Automatic Monitoring
# ============================================
#
# This script:
# 1. Launches the full experiment matrix in background
# 2. Starts real-time monitoring to detect failures
# 3. Logs everything to files for later review
#
# Usage:
#   ./launch_experiments.sh configs/FINAL_working_3datasets.yaml
#

set -euo pipefail

# Configuration
CONFIG_FILE="${1:-configs/FINAL_working_3datasets.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
EXPERIMENT_LOG="${LOG_DIR}/experiments_${TIMESTAMP}.log"
MONITOR_LOG="${LOG_DIR}/monitor_${TIMESTAMP}.log"

# Create log directory
mkdir -p "${LOG_DIR}"

echo "========================================================================"
echo "🚀 LAUNCHING EXPERIMENT MATRIX"
echo "========================================================================"
echo "Config: ${CONFIG_FILE}"
echo "Timestamp: ${TIMESTAMP}"
echo "Experiment log: ${EXPERIMENT_LOG}"
echo "Monitor log: ${MONITOR_LOG}"
echo ""

# Verify config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Show what we're about to launch
echo "📋 Configuration Preview:"
echo "----------------------------------------"
grep -A 5 "datasets:" "${CONFIG_FILE}" || true
echo "----------------------------------------"
echo ""

# Ask for confirmation
read -p "Launch experiments? This will run for 20-24 hours. (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 1
fi

# Launch experiments in background
echo ""
echo "🚀 Starting matrix runner..."
poetry run python -u experiments/matrix_runner.py \
    --config "${CONFIG_FILE}" \
    2>&1 | tee "${EXPERIMENT_LOG}" &

EXPERIMENT_PID=$!
echo "   Process ID: ${EXPERIMENT_PID}"
echo "   Log file: ${EXPERIMENT_LOG}"

# Wait a moment for process to start
sleep 3

# Verify process is running
if ! ps -p ${EXPERIMENT_PID} > /dev/null; then
    echo "❌ ERROR: Experiment process failed to start!"
    echo "   Check log: ${EXPERIMENT_LOG}"
    exit 1
fi

echo "✓ Experiments started successfully"
echo ""

# Start monitor
echo "🔍 Starting experiment monitor..."
poetry run python -u experiments/monitor_experiments.py \
    --pid ${EXPERIMENT_PID} \
    --interval 300 \
    2>&1 | tee "${MONITOR_LOG}" &

MONITOR_PID=$!
echo "   Monitor PID: ${MONITOR_PID}"
echo "   Monitor log: ${MONITOR_LOG}"
echo ""

# Save PIDs for later reference
PID_FILE="${LOG_DIR}/pids_${TIMESTAMP}.txt"
cat > "${PID_FILE}" <<EOF
EXPERIMENT_PID=${EXPERIMENT_PID}
MONITOR_PID=${MONITOR_PID}
TIMESTAMP=${TIMESTAMP}
CONFIG=${CONFIG_FILE}
EXPERIMENT_LOG=${EXPERIMENT_LOG}
MONITOR_LOG=${MONITOR_LOG}
EOF

echo "✓ Monitor started successfully"
echo ""
echo "========================================================================"
echo "✅ LAUNCH COMPLETE"
echo "========================================================================"
echo ""
echo "📊 Status:"
echo "   Experiments: PID ${EXPERIMENT_PID}"
echo "   Monitor: PID ${MONITOR_PID}"
echo "   PIDs saved to: ${PID_FILE}"
echo ""
echo "📝 Logs:"
echo "   Experiments: ${EXPERIMENT_LOG}"
echo "   Monitor: ${MONITOR_LOG}"
echo ""
echo "💡 Useful commands:"
echo "   # Check experiment progress"
echo "   tail -f ${EXPERIMENT_LOG}"
echo ""
echo "   # Check monitor alerts"
echo "   tail -f ${MONITOR_LOG}"
echo ""
echo "   # Check process status"
echo "   ps aux | grep ${EXPERIMENT_PID}"
echo ""
echo "   # Stop experiments (if needed)"
echo "   kill ${EXPERIMENT_PID}"
echo "   kill ${MONITOR_PID}"
echo ""
echo "   # Quick status check"
echo "   ls -lh results/artifacts_* | tail -10"
echo ""
echo "⏱️  Estimated completion: $(date -d '+22 hours' '+%Y-%m-%d %H:%M')"
echo ""
echo "🌊 May your experiments flow smoothly!"
