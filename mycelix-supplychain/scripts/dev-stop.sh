#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  🛑 Stopping Mycelix Supply Chain Development Environment   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

docker-compose down

echo ""
echo -e "${GREEN}✅ All services stopped${NC}"
echo ""
echo -e "  Data volumes preserved (postgres, prometheus, grafana)"
echo -e "  To remove all data, run: ${BLUE}./scripts/dev-reset.sh${NC}"
echo ""
