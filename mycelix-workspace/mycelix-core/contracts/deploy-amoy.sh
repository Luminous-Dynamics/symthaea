#!/bin/bash
# Mycelix Contract Deployment Script for Polygon Amoy
# Usage: ./deploy-amoy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Mycelix Contract Deployment (Amoy)"
echo "=========================================="
echo ""

# Check for .env
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Copy .env.example to .env and configure your private key"
    exit 1
fi

# Source environment
source .env

# Check for private key
if [ -z "$PRIVATE_KEY" ] || [ "$PRIVATE_KEY" = "your_private_key_here" ]; then
    echo "ERROR: PRIVATE_KEY not configured in .env"
    exit 1
fi

echo "Deploying contracts to Polygon Amoy..."
echo ""

# Deploy
forge script script/Deploy.s.sol:DeployAmoy \
    --rpc-url https://rpc-amoy.polygon.technology \
    --broadcast \
    -vvv

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "Update sdk/config.ts with the deployed addresses"
