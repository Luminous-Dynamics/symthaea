#!/bin/bash
#
# Development Environment Setup Script
# Sets up everything needed to develop on mycelix-supplychain
#

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Mycelix Supply Chain Setup  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check for required tools
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}✗ Rust not found${NC}"
    echo "Install from: https://rustup.rs/"
    exit 1
fi
echo -e "${GREEN}✓ Rust $(cargo --version | cut -d ' ' -f 2)${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠ Node.js not found (optional for TS components)${NC}"
else
    echo -e "${GREEN}✓ Node.js $(node --version)${NC}"
fi

# Check Make
if ! command -v make &> /dev/null; then
    echo -e "${YELLOW}⚠ Make not found (optional, but recommended)${NC}"
else
    echo -e "${GREEN}✓ Make$(NC}"
fi

echo ""

# Install Rust components
echo -e "${BLUE}Installing Rust components...${NC}"
rustup component add rustfmt clippy
echo -e "${GREEN}✓ Rust components installed${NC}"
echo ""

# Create data directory
echo -e "${BLUE}Creating data directory...${NC}"
mkdir -p data
echo -e "${GREEN}✓ Data directory created${NC}"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env file...${NC}"
    cp config/dotenv.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}  → Edit .env to customize configuration${NC}"
else
    echo -e "${YELLOW}ℹ .env already exists, skipping${NC}"
fi
echo ""

# Install TypeScript dependencies
if command -v npm &> /dev/null; then
    echo -e "${BLUE}Installing TypeScript dependencies...${NC}"

    echo "  → SDK..."
    cd ts/sdk && npm install --silent > /dev/null 2>&1 && cd ../..

    echo "  → Dashboard..."
    cd ts/dashboard && npm install --silent > /dev/null 2>&1 && cd ../..

    echo "  → CSV Adapter..."
    cd ts/adapters/csv && npm install --silent > /dev/null 2>&1 && cd ../../..

    echo "  → MQTT Adapter..."
    cd ts/adapters/mqtt && npm install --silent > /dev/null 2>&1 && cd ../../..

    echo -e "${GREEN}✓ TypeScript dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Skipping TypeScript dependencies (npm not found)${NC}"
fi
echo ""

# Build Rust service
echo -e "${BLUE}Building Rust service (this may take a few minutes)...${NC}"
cd rust && cargo build --release --quiet > /dev/null 2>&1 && cd ..
echo -e "${GREEN}✓ Rust service built${NC}"
echo ""

# Success!
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Setup Complete! 🎉             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo ""
echo "  1. Start the service:"
echo -e "     ${GREEN}make run${NC}"
echo ""
echo "  2. In another terminal, test it:"
echo -e "     ${GREEN}curl http://localhost:8080/health${NC}"
echo ""
echo "  3. Post an event:"
echo -e "     ${GREEN}curl -X POST http://localhost:8080/v1/events \\${NC}"
echo -e "       ${GREEN}-H 'Content-Type: application/json' \\${NC}"
echo -e "       ${GREEN}-d @specs/examples/batch_produced.json${NC}"
echo ""
echo "  4. Read the quickstart:"
echo -e "     ${GREEN}cat QUICKSTART.md${NC}"
echo ""
echo -e "${BLUE}Available commands:${NC}"
echo -e "  ${GREEN}make help${NC}     - See all available commands"
echo -e "  ${GREEN}make test${NC}     - Run tests"
echo -e "  ${GREEN}make fmt${NC}      - Format code"
echo -e "  ${GREEN}make lint${NC}     - Lint code"
echo ""
echo "Happy building! 🚀"
