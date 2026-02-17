#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  🚀 Starting Mycelix Supply Chain Development Environment   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Starting services with Docker Compose...${NC}"
docker-compose up -d

echo ""
echo -e "${GREEN}✅ Services started successfully!${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Service URLs:${NC}"
echo -e "  📡 Supply Chain API:  ${BLUE}http://localhost:8080${NC}"
echo -e "  📊 Prometheus:        ${BLUE}http://localhost:9090${NC}"
echo -e "  📈 Grafana:           ${BLUE}http://localhost:3000${NC} (admin/admin)"
echo -e "  🗄️  PostgreSQL:        ${BLUE}localhost:5432${NC}"
echo ""
echo -e "${GREEN}Quick Commands:${NC}"
echo -e "  Health check:   ${YELLOW}curl http://localhost:8080/health | jq${NC}"
echo -e "  View logs:      ${YELLOW}docker-compose logs -f service${NC}"
echo -e "  Stop services:  ${YELLOW}./scripts/dev-stop.sh${NC}"
echo -e "  Reset env:      ${YELLOW}./scripts/dev-reset.sh${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"

# Wait for service to be healthy
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is ready!${NC}"
        echo ""

        # Get service version and status
        HEALTH=$(curl -s http://localhost:8080/health | jq -r '.status')
        VERSION=$(curl -s http://localhost:8080/health | jq -r '.version')

        echo -e "${GREEN}Status:${NC}  $HEALTH"
        echo -e "${GREEN}Version:${NC} $VERSION"
        echo ""

        exit 0
    fi

    ATTEMPT=$((ATTEMPT + 1))
    echo -ne "\rAttempt $ATTEMPT/$MAX_ATTEMPTS..."
    sleep 2
done

echo ""
echo -e "${YELLOW}⚠️  Service is taking longer than expected to start.${NC}"
echo -e "${YELLOW}   Check logs with: docker-compose logs service${NC}"
echo ""

exit 0
