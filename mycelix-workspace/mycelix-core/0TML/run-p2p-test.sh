#!/usr/bin/env bash

# ZeroTrustML Multi-Node P2P Test Launcher
# Demonstrates 3 hospitals running federated learning via Holochain P2P

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  ZeroTrustML Multi-Node P2P Federated Learning Test${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "This will start:"
echo "  • 3 Holochain Conductors (Boston, London, Tokyo)"
echo "  • 3 ZeroTrustML Nodes (one per hospital)"
echo "  • 1 Test Orchestrator (runs FL training)"
echo "  • 1 PostgreSQL database (shared for demo)"
echo ""
echo -e "${YELLOW}Architecture: Pure P2P - No Central Server!${NC}"
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker found: $(docker --version)${NC}"
echo -e "${GREEN}✅ Docker Compose found: $(docker-compose --version)${NC}"
echo ""

# Create test_results directory if it doesn't exist
mkdir -p test_results

# Check if containers are already running
if docker-compose -f docker-compose.multi-node.yml ps | grep -q "Up"; then
    echo -e "${YELLOW}⚠️  Containers already running. Stopping them first...${NC}"
    docker-compose -f docker-compose.multi-node.yml down
    echo ""
fi

# Start the multi-node network
echo -e "${BLUE}🚀 Starting multi-node P2P network...${NC}"
echo ""

docker-compose -f docker-compose.multi-node.yml up -d postgres
echo -e "${GREEN}✅ PostgreSQL started${NC}"
sleep 5

docker-compose -f docker-compose.multi-node.yml up -d holochain-node1 holochain-node2 holochain-node3
echo -e "${GREEN}✅ Holochain conductors started${NC}"
echo "   • Boston:  ws://localhost:8881"
echo "   • London:  ws://localhost:8882"
echo "   • Tokyo:   ws://localhost:8883"
sleep 10

docker-compose -f docker-compose.multi-node.yml up -d zerotrustml-node1 zerotrustml-node2 zerotrustml-node3
echo -e "${GREEN}✅ ZeroTrustML nodes started${NC}"
sleep 5

echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
docker-compose -f docker-compose.multi-node.yml ps

echo ""
echo -e "${BLUE}🧪 Running federated learning test...${NC}"
docker-compose -f docker-compose.multi-node.yml up test-orchestrator

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Test Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if test results exist
if [ -f "test_results/multi_node_p2p_results.json" ]; then
    echo -e "${BLUE}📊 Test Results:${NC}"
    echo ""
    cat test_results/multi_node_p2p_results.json | python3 -m json.tool
    echo ""
fi

echo -e "${BLUE}📋 View detailed logs:${NC}"
echo "  docker logs zerotrustml-test-orchestrator"
echo ""

echo -e "${BLUE}🛑 Stop all services:${NC}"
echo "  docker-compose -f docker-compose.multi-node.yml down"
echo ""

echo -e "${BLUE}🔍 Explore running nodes:${NC}"
echo "  docker exec -it zerotrustml-node1-boston /bin/bash"
echo "  docker exec -it zerotrustml-node2-london /bin/bash"
echo "  docker exec -it zerotrustml-node3-tokyo /bin/bash"
echo ""

echo -e "${BLUE}📖 Read more: MULTI_NODE_P2P_TEST.md${NC}"
echo ""
