#!/usr/bin/env bash
# Gen7 Validation Suite Monitor
# Real-time progress display for all running experiments

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Clear screen
clear

echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    Gen7 Validation Suite Monitor${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo

# Function to check if process is running
check_process() {
    if pgrep -f "$1" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Running${NC}"
    else
        echo -e "${RED}❌ Not running${NC}"
    fi
}

# Check all three experiments
echo -e "${YELLOW}📊 Experiment Status:${NC}"
echo
echo -n "  1. CIFAR-10 Neural Network:  "
check_process "gen7_cifar10_nn_test.py"

echo -n "  2. Extended Non-IID Test:    "
check_process "gen7_noniid_extended_test.py"

echo -n "  3. Ablation Study:           "
check_process "gen7_ablation_study.py"

echo
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────────${NC}"

# CIFAR-10 NN Progress
echo
echo -e "${YELLOW}🧠 CIFAR-10 Neural Network Test${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────────${NC}"
if [ -f /tmp/gen7_cifar10_nn.log ]; then
    tail -15 /tmp/gen7_cifar10_nn.log | grep -E "(Experiment|Accuracy|Status|complete|Error|WARNING)" || echo "  Initializing..."
else
    echo "  Log file not found"
fi

# Non-IID Extended Progress
echo
echo -e "${YELLOW}🌐 Extended Non-IID Test (50 Rounds)${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────────${NC}"
if [ -f /tmp/gen7_noniid_extended.log ]; then
    tail -15 /tmp/gen7_noniid_extended.log | grep -E "(Experiment|Accuracy|Status|complete|Error|WARNING)" || echo "  Initializing..."
else
    echo "  Log file not found"
fi

# Ablation Study Progress
echo
echo -e "${YELLOW}🔬 Ablation Study${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────────${NC}"
if [ -f /tmp/gen7_ablation.log ]; then
    tail -15 /tmp/gen7_ablation.log | grep -E "(Configuration|Accuracy|Status|complete|Error|WARNING)" || echo "  Initializing..."
else
    echo "  Log file not found"
fi

# Summary
echo
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}💡 Quick Commands:${NC}"
echo
echo "  View full logs:"
echo "    tail -f /tmp/gen7_cifar10_nn.log"
echo "    tail -f /tmp/gen7_noniid_extended.log"
echo "    tail -f /tmp/gen7_ablation.log"
echo
echo "  Watch this monitor continuously:"
echo "    watch -n 10 $0"
echo
echo "  Kill all experiments:"
echo "    pkill -f gen7_cifar10_nn_test"
echo "    pkill -f gen7_noniid_extended_test"
echo "    pkill -f gen7_ablation_study"
echo
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Last updated: $(date)${NC}"
echo
