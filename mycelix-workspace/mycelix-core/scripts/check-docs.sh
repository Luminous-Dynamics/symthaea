#!/usr/bin/env bash
# Documentation Health Check Script for Mycelix Protocol

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
ERRORS=0
WARNINGS=0
PASSED=0

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$BASE_DIR/docs"

echo -e "${BLUE}🔍 Mycelix Documentation Health Check${NC}"
echo -e "${BLUE}======================================${NC}\n"

section() {
    echo -e "\n${BLUE}## $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED++))
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

error() {
    echo -e "${RED}❌ $1${NC}"
    ((ERRORS++))
}

section "Required Files"

REQUIRED_FILES=(
    "CLAUDE.md"
    "README.md"
    "docs/README.md"
    "docs/VERSION_HISTORY.md"
    "docs/adr/README.md"
    "docs/02-charters/README.md"
    "docs/03-architecture/README.md"
    "0TML/docs/README.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$BASE_DIR/$file" ]]; then
        success "$file exists"
    else
        error "$file is missing"
    fi
done

section "Summary"
echo ""
echo -e "${GREEN}  ✅ Passed:   $PASSED${NC}"
echo -e "${YELLOW}  ⚠️  Warnings: $WARNINGS${NC}"
echo -e "${RED}  ❌ Errors:   $ERRORS${NC}"

[[ $ERRORS -eq 0 ]] && echo -e "\n${GREEN}PASSED${NC}" || echo -e "\n${RED}FAILED${NC}"
