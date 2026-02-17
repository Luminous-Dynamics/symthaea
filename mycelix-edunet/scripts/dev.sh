#!/usr/bin/env bash
# Development environment startup script for Mycelix EduNet

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Mycelix EduNet Development Environment${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Check prerequisites
echo -e "${YELLOW}▶ Checking prerequisites...${NC}"

if ! command -v cargo &> /dev/null; then
    echo -e "${RED}✗ Rust/Cargo not found. Install from https://rustup.rs/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Rust/Cargo found${NC}"

if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js not found. Install from https://nodejs.org/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found${NC}"

# TODO: Check for Holochain tools when zomes are implemented
# if ! command -v hc &> /dev/null; then
#     echo -e "${YELLOW}⚠ Holochain CLI not found. Install for full functionality.${NC}"
# else
#     echo -e "${GREEN}✓ Holochain CLI found${NC}"
# fi

echo

# Build Rust workspace
echo -e "${YELLOW}▶ Building Rust workspace...${NC}"
if cargo build --workspace; then
    echo -e "${GREEN}✓ Rust build successful${NC}"
else
    echo -e "${RED}✗ Rust build failed${NC}"
    exit 1
fi

echo

# Build web app
echo -e "${YELLOW}▶ Installing web dependencies...${NC}"
cd apps/web
if npm install; then
    echo -e "${GREEN}✓ Web dependencies installed${NC}"
else
    echo -e "${RED}✗ Web dependency installation failed${NC}"
    exit 1
fi

cd ../..
echo

# TODO: Start Holochain conductor when DNA is implemented
# echo -e "${YELLOW}▶ Starting Holochain conductor...${NC}"
# hc sandbox create -d=edunet-dev workdir
# hc sandbox run -p 8888 workdir &
# CONDUCTOR_PID=$!
# echo -e "${GREEN}✓ Conductor started (PID: $CONDUCTOR_PID)${NC}"

echo -e "${YELLOW}▶ Starting web development server...${NC}"
echo -e "${YELLOW}  Access the app at: http://localhost:3000${NC}"
echo
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Development environment ready!${NC}"
echo -e "${GREEN}  Press Ctrl+C to stop${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

cd apps/web
npm run dev
