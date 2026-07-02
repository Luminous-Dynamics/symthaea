#!/bin/bash
# Code quality checks for Mycelix-DeSci
# Usage: ./scripts/lint.sh [--fix]

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

FIX_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")/../src/core"

echo -e "${BLUE}🔍 Mycelix-DeSci Code Quality Checks${NC}\n"

# Format check
echo -e "${GREEN}Checking code formatting...${NC}"
if [ "$FIX_MODE" = true ]; then
    cargo fmt --all
    echo -e "${GREEN}✅ Code formatted${NC}\n"
else
    cargo fmt --all -- --check
    echo -e "${GREEN}✅ Formatting OK${NC}\n"
fi

# Clippy
echo -e "${GREEN}Running clippy...${NC}"
if [ "$FIX_MODE" = true ]; then
    cargo clippy --all-targets --all-features --fix --allow-dirty --allow-staged
    echo -e "${GREEN}✅ Clippy fixes applied${NC}\n"
else
    cargo clippy --all-targets --all-features -- -D warnings
    echo -e "${GREEN}✅ Clippy passed${NC}\n"
fi

# Check for common issues
echo -e "${GREEN}Checking for common issues...${NC}"

# Check for TODO/FIXME
TODO_COUNT=$(grep -r "TODO\|FIXME" --include="*.rs" src/ | wc -l || true)
if [ $TODO_COUNT -gt 0 ]; then
    echo -e "${BLUE}ℹ️  Found $TODO_COUNT TODO/FIXME comments${NC}"
fi

# Check for unwrap() in src/ (not tests)
UNWRAP_COUNT=$(grep -r "\.unwrap()" --include="*.rs" src/ | grep -v "test" | grep -v "/examples/" | wc -l || true)
if [ $UNWRAP_COUNT -gt 0 ]; then
    echo -e "${BLUE}⚠️  Found $UNWRAP_COUNT .unwrap() calls in library code${NC}"
fi

echo -e "\n${GREEN}🎉 All linting checks passed!${NC}"
