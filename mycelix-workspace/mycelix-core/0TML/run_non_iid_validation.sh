#!/bin/bash
#
# Non-IID Byzantine Detection Validation Runner
# ==============================================
#
# This script runs comprehensive validation experiments to prove
# Byzantine detection works under realistic non-IID federated learning conditions.
#
# Usage:
#   ./run_non_iid_validation.sh [mode]
#
# Modes:
#   quick    - Core scenarios only (~5-10 minutes)
#   extended - Core + additional scenarios (~15-20 minutes)
#   full     - Complete test suite (~30-60 minutes)
#   sweep    - Alpha parameter sweep across non-IID severity
#   compare  - Compare different aggregators
#
# Examples:
#   ./run_non_iid_validation.sh quick
#   ./run_non_iid_validation.sh full
#
# Author: Luminous Dynamics
# Date: January 2026

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default mode
MODE="${1:-quick}"

echo ""
echo "=============================================================="
echo "  NON-IID BYZANTINE DETECTION VALIDATION"
echo "=============================================================="
echo ""
echo "  Mode: $MODE"
echo "  Time: $(date)"
echo ""

# Check Python environment
if [ -d ".venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source .venv/bin/activate
elif [ -d ".venv-bench" ]; then
    echo -e "${BLUE}Activating benchmark virtual environment...${NC}"
    source .venv-bench/bin/activate
fi

# Verify Python can import required modules
echo -e "${BLUE}Checking dependencies...${NC}"
python -c "import numpy; import scipy" 2>/dev/null || {
    echo -e "${RED}ERROR: Required Python packages not found${NC}"
    echo "Please ensure numpy and scipy are installed"
    exit 1
}

# Run validation based on mode
case $MODE in
    quick)
        echo -e "${GREEN}Running QUICK validation (core scenarios)...${NC}"
        echo "This will test IID baseline, mild, moderate, and severe non-IID."
        echo "Expected time: 5-10 minutes"
        echo ""
        python -m experiments.non_iid_validation.run_validation --quick
        ;;

    extended)
        echo -e "${GREEN}Running EXTENDED validation...${NC}"
        echo "This includes core scenarios plus quantity/feature/temporal skew."
        echo "Expected time: 15-20 minutes"
        echo ""
        python -m experiments.non_iid_validation.run_validation --extended
        ;;

    full)
        echo -e "${YELLOW}Running FULL validation suite...${NC}"
        echo "This runs all scenarios including stress tests."
        echo "Expected time: 30-60 minutes"
        echo ""
        python -m experiments.non_iid_validation.run_validation --full
        ;;

    sweep)
        echo -e "${GREEN}Running ALPHA SWEEP...${NC}"
        echo "Testing detection across the non-IID severity spectrum."
        echo "Alpha values: 100.0 (IID) -> 0.05 (extreme non-IID)"
        echo ""
        python -m experiments.non_iid_validation.run_validation --alpha-sweep
        ;;

    compare)
        echo -e "${GREEN}Running AGGREGATOR COMPARISON...${NC}"
        echo "Comparing Krum, TrimmedMean, Median, AEGIS, AEGIS+Gen7"
        echo "on moderate non-IID data (alpha=0.5)"
        echo ""
        python -m experiments.non_iid_validation.run_validation --compare-aggregators
        ;;

    help|--help|-h)
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  quick    - Core scenarios only (~5-10 min)"
        echo "  extended - Extended scenarios (~15-20 min)"
        echo "  full     - Complete test suite (~30-60 min)"
        echo "  sweep    - Alpha parameter sweep"
        echo "  compare  - Aggregator comparison"
        echo "  help     - Show this message"
        exit 0
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Run '$0 help' for usage information."
        exit 1
        ;;
esac

# Check exit status
EXIT_CODE=$?

echo ""
echo "=============================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "  ${GREEN}VALIDATION COMPLETED SUCCESSFULLY${NC}"
else
    echo -e "  ${YELLOW}VALIDATION COMPLETED WITH WARNINGS${NC}"
fi
echo "=============================================================="
echo ""
echo "Results saved in: validation_results/non_iid/"
echo ""

# List generated files
if [ -d "validation_results/non_iid" ]; then
    echo "Generated files:"
    ls -lt validation_results/non_iid/*.{json,txt} 2>/dev/null | head -5 || true
fi

exit $EXIT_CODE
