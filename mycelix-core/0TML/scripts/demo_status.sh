#!/usr/bin/env bash
# ZeroTrustML Grant Demo - Network Status Display
# Provides clean, branded output for video recording

set -e

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          ZeroTrustML Grant Demo - Network Status                  ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║                                                                ║"
echo "║  HONEST NODES (3)                                             ║"

# Check Boston
if nc -zv localhost 8881 2>&1 | grep -q succeeded; then
    echo -e "║  ${GREEN}✅${NC} Boston Hospital    (Port 8881)  [HEALTHY]              ║"
else
    echo -e "║  ${RED}❌${NC} Boston Hospital    (Port 8881)  [DOWN]                 ║"
fi

# Check London
if nc -zv localhost 8882 2>&1 | grep -q succeeded; then
    echo -e "║  ${GREEN}✅${NC} London Hospital    (Port 8882)  [HEALTHY]              ║"
else
    echo -e "║  ${RED}❌${NC} London Hospital    (Port 8882)  [DOWN]                 ║"
fi

# Check Tokyo
if nc -zv localhost 8883 2>&1 | grep -q succeeded; then
    echo -e "║  ${GREEN}✅${NC} Tokyo Hospital     (Port 8883)  [HEALTHY]              ║"
else
    echo -e "║  ${RED}❌${NC} Tokyo Hospital     (Port 8883)  [DOWN]                 ║"
fi

echo "║                                                                ║"
echo "║  MALICIOUS NODES (2)                                          ║"

# Check Rogue 1
if nc -zv localhost 8884 2>&1 | grep -q succeeded; then
    echo -e "║  ${YELLOW}⚠️${NC}  Rogue Hospital 1  (Port 8884)  [HEALTHY]              ║"
    echo "║     Attack Type: Gradient Inversion                           ║"
else
    echo -e "║  ${RED}❌${NC} Rogue Hospital 1  (Port 8884)  [DOWN]                 ║"
fi

# Check Rogue 2
if nc -zv localhost 8885 2>&1 | grep -q succeeded; then
    echo -e "║  ${YELLOW}⚠️${NC}  Rogue Hospital 2  (Port 8885)  [HEALTHY]              ║"
    echo "║     Attack Type: Sign Flipping                                ║"
else
    echo -e "║  ${RED}❌${NC} Rogue Hospital 2  (Port 8885)  [DOWN]                 ║"
fi

echo "║                                                                ║"
echo "║  Configuration:                                                ║"
echo "║  • Total Nodes: 5                                              ║"
echo "║  • Byzantine Ratio: 40% (2 out of 5 nodes)                    ║"
echo "║  • Attack Types: 2 different strategies                       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# PostgreSQL status
echo ""
echo "Database Status:"
if nc -zv localhost 5433 2>&1 | grep -q succeeded; then
    echo -e "  ${GREEN}✅${NC} PostgreSQL (Port 5433) - HEALTHY"
else
    echo -e "  ${RED}❌${NC} PostgreSQL (Port 5433) - DOWN"
fi

# Docker container status (detailed)
echo ""
echo "Docker Container Details:"
docker ps --format "table {{.Names}}\t{{.Status}}" --filter name=holochain 2>/dev/null || echo "  ⚠️  Docker not available"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Ready for Byzantine detection demo${NC}"
echo -e "${BLUE}Run: python tests/test_grant_demo_5nodes.py${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
