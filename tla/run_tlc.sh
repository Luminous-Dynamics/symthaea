#!/usr/bin/env bash
#
# TLC Model Checker Runner for CantorLtcHdc
#
# Usage: ./run_tlc.sh [options]
#
# Options:
#   -d, --download    Download TLA+ tools if not present
#   -w, --workers N   Number of worker threads (default: auto)
#   -c, --coverage    Enable coverage statistics
#   -v, --verbose     Verbose output
#   -h, --help        Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TLA_TOOLS="$SCRIPT_DIR/tla2tools.jar"
SPEC="CantorLtcHdc_MC.tla"
CONFIG="CantorLtcHdc.cfg"
WORKERS="auto"
COVERAGE=""
VERBOSE=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}       ${GREEN}TLC Model Checker: Cantor-LTC/HDC Network${NC}             ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}       ${YELLOW}Verifying Mathematical Sovereignty${NC}                    ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
}

download_tools() {
    if [[ ! -f "$TLA_TOOLS" ]]; then
        echo -e "${YELLOW}Downloading TLA+ tools...${NC}"
        curl -L -o "$TLA_TOOLS" \
            "https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar"
        echo -e "${GREEN}✓ TLA+ tools downloaded${NC}"
    else
        echo -e "${GREEN}✓ TLA+ tools already present${NC}"
    fi
}

check_java() {
    if ! command -v java &> /dev/null; then
        echo -e "${RED}✗ Java not found. Please install Java 11+${NC}"
        echo "  NixOS: nix-shell -p jdk11"
        exit 1
    fi

    JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
    if [[ "$JAVA_VERSION" -lt 11 ]]; then
        echo -e "${YELLOW}⚠ Java version $JAVA_VERSION detected. Recommend Java 11+${NC}"
    else
        echo -e "${GREEN}✓ Java $JAVA_VERSION detected${NC}"
    fi
}

run_tlc() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Running TLC Model Checker...${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

    cd "$SCRIPT_DIR"

    # Build TLC command
    TLC_CMD="java -XX:+UseParallelGC -Xmx4g -jar $TLA_TOOLS"
    TLC_CMD="$TLC_CMD -config $CONFIG $SPEC"
    TLC_CMD="$TLC_CMD -workers $WORKERS"

    if [[ -n "$COVERAGE" ]]; then
        TLC_CMD="$TLC_CMD -coverage 1"
    fi

    if [[ -n "$VERBOSE" ]]; then
        TLC_CMD="$TLC_CMD -dump states.dump"
    fi

    echo -e "${BLUE}Command:${NC} $TLC_CMD"
    echo

    # Run TLC
    START_TIME=$(date +%s)

    if $TLC_CMD; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                    ✓ VERIFICATION PASSED                     ║${NC}"
        echo -e "${GREEN}║                                                              ║${NC}"
        echo -e "${GREEN}║  All 7 safety invariants verified                            ║${NC}"
        echo -e "${GREEN}║  Fixed Core Integrity: MATHEMATICALLY SOVEREIGN              ║${NC}"
        echo -e "${GREEN}║  Duration: ${DURATION}s                                              ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    else
        echo
        echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║                    ✗ VERIFICATION FAILED                     ║${NC}"
        echo -e "${RED}║                                                              ║${NC}"
        echo -e "${RED}║  Check the error trace above to identify the violation       ║${NC}"
        echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
        exit 1
    fi
}

show_help() {
    echo "TLC Model Checker for Cantor-LTC/HDC Network"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -d, --download    Download TLA+ tools if not present"
    echo "  -w, --workers N   Number of worker threads (default: auto)"
    echo "  -c, --coverage    Enable coverage statistics"
    echo "  -v, --verbose     Verbose output (dump states)"
    echo "  -h, --help        Show this help"
    echo
    echo "Examples:"
    echo "  $0                     # Run with defaults"
    echo "  $0 -d -w 4             # Download tools, use 4 workers"
    echo "  $0 -c -v               # With coverage and verbose"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--download)
            download_tools
            shift
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE="true"
            shift
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
print_header
check_java

if [[ ! -f "$TLA_TOOLS" ]]; then
    echo -e "${YELLOW}TLA+ tools not found. Run with -d to download.${NC}"
    echo -e "  ${BLUE}$0 -d${NC}"
    exit 1
fi

run_tlc
