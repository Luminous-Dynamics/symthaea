#!/usr/bin/env bash
# Build script for Mycelix-Marketplace Holochain Backend
# Compiles all zomes to WASM and packages the DNA/hApp

set -e

echo "🍄 Building Mycelix-Marketplace Backend..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found. Please install Rust."
    exit 1
fi

if ! command -v hc &> /dev/null; then
    echo "⚠️  Holochain CLI (hc) not found. DNA packaging will be skipped."
    echo "   Install with: cargo install holochain_cli --version 0.6.0"
fi

# Build all zomes
echo -e "${BLUE}📦 Building Zomes...${NC}"
echo ""

cd backend

# Build in release mode with optimizations
echo -e "${YELLOW}Building Listings Integrity...${NC}"
cargo build --release --target wasm32-unknown-unknown -p listings_integrity

echo -e "${YELLOW}Building Listings Coordinator...${NC}"
cargo build --release --target wasm32-unknown-unknown -p listings_coordinator

echo -e "${YELLOW}Building Reputation Integrity...${NC}"
cargo build --release --target wasm32-unknown-unknown -p reputation_integrity

echo -e "${YELLOW}Building Reputation Coordinator...${NC}"
cargo build --release --target wasm32-unknown-unknown -p reputation_coordinator

echo -e "${YELLOW}Building Transactions Integrity...${NC}"
cargo build --release --target wasm32-unknown-unknown -p transactions_integrity

echo -e "${YELLOW}Building Transactions Coordinator...${NC}"
cargo build --release --target wasm32-unknown-unknown -p transactions_coordinator

echo -e "${YELLOW}Building Arbitration Integrity...${NC}"
cargo build --release --target wasm32-unknown-unknown -p arbitration_integrity

echo -e "${YELLOW}Building Arbitration Coordinator...${NC}"
cargo build --release --target wasm32-unknown-unknown -p arbitration_coordinator

echo -e "${YELLOW}Building Messaging Integrity...${NC}"
cargo build --release --target wasm32-unknown-unknown -p messaging_integrity

echo -e "${YELLOW}Building Messaging Coordinator...${NC}"
cargo build --release --target wasm32-unknown-unknown -p messaging_coordinator

echo ""
echo -e "${GREEN}✅ All zomes built successfully!${NC}"
echo ""

# Package DNA and hApp if hc is available
if command -v hc &> /dev/null; then
    echo -e "${BLUE}📦 Packaging DNA...${NC}"
    hc dna pack .

    echo -e "${BLUE}📦 Packaging hApp...${NC}"
    hc app pack .

    echo ""
    echo -e "${GREEN}✅ DNA and hApp packaged successfully!${NC}"
    echo ""
    echo "Output files:"
    echo "  - mycelix_marketplace.dna"
    echo "  - mycelix_marketplace.happ"
else
    echo ""
    echo -e "${YELLOW}⚠️  Skipping DNA/hApp packaging (hc not installed)${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Build complete!${NC}"
echo ""
echo "To test the backend:"
echo "  1. Install Holochain: cargo install holochain --version 0.6.0"
echo "  2. Run conductor: holochain -c conductor-config.yaml"
echo "  3. Install hApp: hc app install mycelix_marketplace.happ"
echo ""
echo "To connect frontend:"
echo "  1. cd frontend"
echo "  2. npm run dev"
echo "  3. Update HOLOCHAIN_CONDUCTOR_URL in .env"
echo ""
