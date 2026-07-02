#!/bin/bash
# Comprehensive test suite for Mycelix-DeSci
# Usage: ./scripts/test.sh [options]
#
# Options:
#   --all       Run all tests (unit, integration, examples)
#   --coverage  Run with coverage report
#   --quick     Run only unit tests
#   --bench     Run benchmarks

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
RUN_UNIT=true
RUN_INTEGRATION=false
RUN_EXAMPLES=false
RUN_COVERAGE=false
RUN_BENCH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_INTEGRATION=true
            RUN_EXAMPLES=true
            shift
            ;;
        --coverage)
            RUN_COVERAGE=true
            shift
            ;;
        --quick)
            RUN_UNIT=true
            shift
            ;;
        --bench)
            RUN_BENCH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")/../src/core"

echo -e "${BLUE}🧪 Mycelix-DeSci Test Suite${NC}\n"

# Run unit tests
if [ "$RUN_UNIT" = true ]; then
    echo -e "${GREEN}Running unit tests...${NC}"
    if [ "$RUN_COVERAGE" = true ]; then
        cargo tarpaulin --lib --out Html --output-dir ../../target/coverage
        echo -e "${GREEN}✅ Coverage report generated at target/coverage/index.html${NC}\n"
    else
        cargo test --lib
        echo -e "${GREEN}✅ Unit tests passed${NC}\n"
    fi
fi

# Run integration tests
if [ "$RUN_INTEGRATION" = true ]; then
    echo -e "${GREEN}Running integration tests...${NC}"
    cargo test --test '*'
    echo -e "${GREEN}✅ Integration tests passed${NC}\n"
fi

# Test examples compile
if [ "$RUN_EXAMPLES" = true ]; then
    echo -e "${GREEN}Testing examples...${NC}"
    cargo build --example create_claim
    cargo build --example hash_dataset
    cargo build --example complete_workflow
    cargo build --example query_demo
    cargo build --example trust_demo
    echo -e "${GREEN}✅ All examples compile${NC}\n"
fi

# Run benchmarks
if [ "$RUN_BENCH" = true ]; then
    echo -e "${GREEN}Running benchmarks...${NC}"
    cargo bench --bench core_benchmarks
    echo -e "${GREEN}✅ Benchmarks complete${NC}\n"
fi

echo -e "${GREEN}🎉 All tests passed!${NC}"
