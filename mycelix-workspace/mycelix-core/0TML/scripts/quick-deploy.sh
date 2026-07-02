#!/usr/bin/env bash
# ZeroTrustML Phase 9 - Quick Deployment Script

set -e

echo "🚀 ZeroTrustML Phase 9 - Quick Deployment"
echo "======================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and set secure passwords!"
    echo "   Especially: ZEROTRUSTML_DB_PASSWORD and GRAFANA_PASSWORD"
    echo ""
    read -p "Press Enter to continue after editing .env..." 
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data models logs backups config
echo "✓ Directories created"
echo ""

# Pull images
echo "Pulling Docker images..."
docker-compose -f docker-compose.prod.yml pull
echo "✓ Images pulled"
echo ""

# Start services
echo "Starting ZeroTrustML services..."
docker-compose -f docker-compose.prod.yml up -d
echo "✓ Services started"
echo ""

# Wait for services to be healthy
echo "Waiting for services to be healthy (30s)..."
sleep 30

# Verify deployment
echo ""
./scripts/verify-deployment.sh

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Access Grafana: http://localhost:3000"
echo "2. Connect FL clients to: ws://localhost:8765"
echo "3. View metrics: http://localhost:9090/metrics"
echo ""
