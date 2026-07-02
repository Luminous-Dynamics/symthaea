#!/bin/bash

# Mycelix Music - Setup Validation Script
# This script verifies that all components are properly configured

set -e

echo "🔍 Mycelix Music - Setup Validation"
echo "====================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUCCESS=0
WARNINGS=0
ERRORS=0

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((SUCCESS++))
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

# Check Node.js
echo "📦 Checking Prerequisites..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js installed: $NODE_VERSION"

    # Check version is 20+
    NODE_MAJOR=$(node --version | cut -d'.' -f1 | sed 's/v//')
    if [ "$NODE_MAJOR" -ge 20 ]; then
        print_success "Node.js version is 20+ (required)"
    else
        print_error "Node.js version must be 20+, found: $NODE_VERSION"
    fi
else
    print_error "Node.js not installed"
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_success "npm installed: $NPM_VERSION"
else
    print_error "npm not installed"
fi

# Check Foundry
if command -v forge &> /dev/null; then
    FORGE_VERSION=$(forge --version | head -n1)
    print_success "Foundry installed: $FORGE_VERSION"
else
    print_warning "Foundry not installed (required for smart contracts)"
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker installed: $DOCKER_VERSION"
else
    print_warning "Docker not installed (required for backend services)"
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    print_success "Docker Compose installed: $COMPOSE_VERSION"
else
    print_warning "Docker Compose not installed (required for backend services)"
fi

echo ""
echo "📁 Checking Project Structure..."

# Check key directories
DIRS=(
    "contracts/src"
    "packages/sdk/src"
    "apps/web/pages"
    "apps/api/src"
    "scripts"
    "docs"
)

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_success "Directory exists: $dir"
    else
        print_error "Directory missing: $dir"
    fi
done

# Check key files
FILES=(
    "package.json"
    ".env.example"
    "docker-compose.yml"
    "contracts/foundry.toml"
    "contracts/src/EconomicStrategyRouter.sol"
    "contracts/src/strategies/PayPerStreamStrategy.sol"
    "contracts/src/strategies/GiftEconomyStrategy.sol"
    "packages/sdk/src/economic-strategies.ts"
    "apps/web/pages/index.tsx"
    "apps/api/src/index.ts"
    "QUICKSTART.md"
    "README.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "File exists: $file"
    else
        print_error "File missing: $file"
    fi
done

echo ""
echo "🔧 Checking Configuration..."

# Check .env file
if [ -f ".env" ]; then
    print_success ".env file exists"

    # Check for required variables
    ENV_VARS=(
        "PRIVATE_KEY"
        "RPC_URL"
        "CHAIN_ID"
    )

    for var in "${ENV_VARS[@]}"; do
        if grep -q "^${var}=" .env; then
            print_success "Environment variable set: $var"
        else
            print_warning "Environment variable not set: $var"
        fi
    done
else
    print_warning ".env file not found (copy from .env.example)"
fi

# Check node_modules
if [ -d "node_modules" ]; then
    print_success "Root dependencies installed"
else
    print_warning "Root dependencies not installed (run: npm install)"
fi

if [ -d "apps/web/node_modules" ]; then
    print_success "Frontend dependencies installed"
else
    print_warning "Frontend dependencies not installed"
fi

if [ -d "apps/api/node_modules" ]; then
    print_success "Backend dependencies installed"
else
    print_warning "Backend dependencies not installed"
fi

echo ""
echo "🔗 Checking Services..."

# Check if Anvil is running
if curl -s -X POST http://localhost:8545 \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
    > /dev/null 2>&1; then
    print_success "Anvil blockchain running (port 8545)"
else
    print_warning "Anvil not running (start with: anvil --block-time 1)"
fi

# Check PostgreSQL
if docker ps | grep -q mycelix-music-postgres; then
    print_success "PostgreSQL running (Docker)"
elif nc -z localhost 5432 2>/dev/null; then
    print_success "PostgreSQL running (port 5432)"
else
    print_warning "PostgreSQL not running"
fi

# Check Redis
if docker ps | grep -q mycelix-music-redis; then
    print_success "Redis running (Docker)"
elif nc -z localhost 6379 2>/dev/null; then
    print_success "Redis running (port 6379)"
else
    print_warning "Redis not running"
fi

# Check Frontend
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend running (http://localhost:3000)"
else
    print_warning "Frontend not running"
fi

# Check Backend API
if curl -s http://localhost:3100/health > /dev/null 2>&1; then
    print_success "Backend API running (http://localhost:3100)"
else
    print_warning "Backend API not running"
fi

echo ""
echo "====================================="
echo "📊 Validation Summary"
echo "====================================="
echo -e "${GREEN}Successful checks:${NC} $SUCCESS"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Errors:${NC} $ERRORS"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 All checks passed! Your setup is complete.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. If Anvil isn't running: anvil --block-time 1"
    echo "2. Deploy contracts: npm run contracts:deploy:local"
    echo "3. Start services: npm run services:up"
    echo "4. Start frontend: cd apps/web && npm run dev"
    echo "5. Visit: http://localhost:3000"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Setup mostly complete, but some warnings.${NC}"
    echo "Review warnings above and start missing services."
    exit 0
else
    echo -e "${RED}❌ Setup incomplete. Fix errors above before proceeding.${NC}"
    exit 1
fi
