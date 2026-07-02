#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║  🔄 Resetting Development Environment                       ║${NC}"
echo -e "${RED}║  ⚠️  WARNING: This will delete ALL data!                     ║${NC}"
echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

read -p "Are you sure you want to delete all data? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo -e "${YELLOW}Reset cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Stopping all services...${NC}"
docker-compose down

echo -e "${YELLOW}Removing all volumes (databases, metrics data)...${NC}"
docker-compose down -v

echo -e "${YELLOW}Removing any local SQLite databases...${NC}"
rm -f ./rust/data/*.db

echo ""
echo -e "${GREEN}✅ Environment reset complete${NC}"
echo ""
echo -e "  All data has been deleted:"
echo -e "    - PostgreSQL database"
echo -e "    - Prometheus metrics"
echo -e "    - Grafana dashboards"
echo -e "    - Local SQLite files"
echo ""
echo -e "  Run ${GREEN}./scripts/dev-start.sh${NC} to start fresh"
echo ""
