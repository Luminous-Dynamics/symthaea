#!/usr/bin/env bash
# ZeroTrustML Phase 9 - Deployment Verification Script

set -e

echo "🔍 ZeroTrustML Phase 9 - Deployment Verification"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

check_docker_service() {
    local service=$1
    echo -n "Checking Docker service: $service... "
    if docker ps --filter "name=$service" --format "{{.Status}}" | grep -q "Up"; then
        echo -e "${GREEN}✓ Running${NC}"
        return 0
    else
        echo -e "${RED}✗ Not running${NC}"
        return 1
    fi
}

# Check Docker
echo "📦 Docker Status"
if ! docker ps > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker not running${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker running${NC}"
echo ""

# Check services
echo "🐳 Docker Services"
check_docker_service "zerotrustml-postgres" || true
check_docker_service "zerotrustml-node" || true  
check_docker_service "zerotrustml-prometheus" || true
check_docker_service "zerotrustml-grafana" || true
echo ""

echo "✅ Verification complete!"
echo ""
echo "Access Grafana: http://localhost:3000"
