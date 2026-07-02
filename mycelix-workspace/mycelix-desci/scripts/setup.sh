#!/bin/bash
# Development environment setup for Mycelix-DeSci
# Usage: ./scripts/setup.sh

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Mycelix-DeSci Development Setup${NC}\n"

# Check for Rust
echo -e "${GREEN}Checking Rust installation...${NC}"
if ! command -v rustc &> /dev/null; then
    echo -e "${YELLOW}Rust not found. Installing via rustup...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

RUST_VERSION=$(rustc --version)
echo -e "${GREEN}✅ Rust installed: $RUST_VERSION${NC}\n"

# Install components
echo -e "${GREEN}Installing Rust components...${NC}"
rustup component add rustfmt clippy
echo -e "${GREEN}✅ Components installed${NC}\n"

# Install development tools
echo -e "${GREEN}Installing development tools...${NC}"

# cargo-tarpaulin for coverage
if ! command -v cargo-tarpaulin &> /dev/null; then
    echo -e "${YELLOW}Installing cargo-tarpaulin...${NC}"
    cargo install cargo-tarpaulin
fi

# cargo-audit for security
if ! command -v cargo-audit &> /dev/null; then
    echo -e "${YELLOW}Installing cargo-audit...${NC}"
    cargo install cargo-audit
fi

# cargo-watch for development
if ! command -v cargo-watch &> /dev/null; then
    echo -e "${YELLOW}Installing cargo-watch...${NC}"
    cargo install cargo-watch
fi

echo -e "${GREEN}✅ Development tools installed${NC}\n"

# Make scripts executable
echo -e "${GREEN}Making scripts executable...${NC}"
chmod +x scripts/*.sh
echo -e "${GREEN}✅ Scripts ready${NC}\n"

# Build project
echo -e "${GREEN}Building project...${NC}"
cd src/core
cargo build
echo -e "${GREEN}✅ Build successful${NC}\n"

# Run tests
echo -e "${GREEN}Running tests...${NC}"
cargo test --lib
echo -e "${GREEN}✅ All tests passed${NC}\n"

echo -e "${GREEN}🎉 Setup complete!${NC}"
echo -e "\n${BLUE}Next steps:${NC}"
echo -e "  • Run tests: ./scripts/test.sh"
echo -e "  • Run lints: ./scripts/lint.sh"
echo -e "  • Start development: cargo watch -x 'test' -x 'clippy'"
echo -e "  • Run examples: cargo run --example complete_workflow"
echo -e "\n${BLUE}Documentation:${NC}"
echo -e "  • README.md - Project overview"
echo -e "  • CONTRIBUTING.md - Contributing guide"
echo -e "  • docs/ARCHITECTURE.md - System architecture"
