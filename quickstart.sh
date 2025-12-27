#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🧠 Symthaea HLB - Quickstart${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v cargo >/dev/null 2>&1; then
    echo -e "${RED}Error: Rust not found. Install from https://rustup.rs${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Rust found ($(cargo --version))${NC}"

if ! command -v nix >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Nix not found. Some features may not work.${NC}"
else
    echo -e "${GREEN}✓ Nix found ($(nix --version | head -n1))${NC}"
fi

# Build release
echo ""
echo -e "${BLUE}Building Symthaea (release mode)...${NC}"
echo -e "${YELLOW}This may take a few minutes on first run...${NC}"

if cargo build --release 2>&1 | tee /tmp/symthaea-build.log | grep -q "error"; then
    echo -e "${RED}✗ Build failed. Check /tmp/symthaea-build.log for details${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build complete${NC}"

# Run quick tests
echo ""
echo -e "${BLUE}Running quick tests...${NC}"

if cargo test --lib --quiet -- --test-threads=1 2>&1 | tee /tmp/symthaea-test.log | grep -q "FAILED"; then
    echo -e "${YELLOW}⚠ Some tests failed. Check /tmp/symthaea-test.log for details${NC}"
    echo -e "${YELLOW}  Continuing anyway...${NC}"
else
    echo -e "${GREEN}✓ Tests passed${NC}"
fi

# Run demo if available
echo ""
echo -e "${BLUE}Running demo...${NC}"

if [ -f "examples/quickstart_demo.rs" ]; then
    echo -e "${CYAN}Demo query: 'install nginx'${NC}"
    cargo run --release --example quickstart_demo 2>/dev/null || {
        echo -e "${YELLOW}⚠ Demo not yet available${NC}"
    }
else
    echo -e "${YELLOW}⚠ Demo example not yet created (examples/quickstart_demo.rs)${NC}"
    echo -e "${CYAN}  You can create a basic query manually:${NC}"
    echo ""
    cat << 'EOF'
use symthaea::SymthaeaHLB;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut symthaea = SymthaeaHLB::new(16384, 1024)?;
    let response = symthaea.process("install nginx").await?;
    println!("Response: {}", response.content);
    println!("Confidence: {:.1}%", response.confidence * 100.0);
    Ok(())
}
EOF
fi

# Success message
echo ""
echo -e "${GREEN}🎉 Success! Symthaea is ready.${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo -e "  ${BLUE}1.${NC} Read the quickstart guide: ${YELLOW}docs/examples/01-quickstart.md${NC}"
echo -e "  ${BLUE}2.${NC} Try the benchmarks: ${YELLOW}./run_benchmarks.sh${NC}"
echo -e "  ${BLUE}3.${NC} Explore examples: ${YELLOW}ls examples/${NC}"
echo -e "  ${BLUE}4.${NC} Build the inspector (coming soon): ${YELLOW}cd tools/symthaea-inspect && cargo build${NC}"
echo ""
echo -e "${CYAN}Documentation:${NC}"
echo -e "  ${BLUE}•${NC} Architecture: ${YELLOW}docs/architecture/${NC}"
echo -e "  ${BLUE}•${NC} Parallel Development Plan: ${YELLOW}PARALLEL_DEVELOPMENT_PLAN.md${NC}"
echo -e "  ${BLUE}•${NC} Improvement Roadmap: ${YELLOW}SYMTHAEA_IMPROVEMENT_ROADMAP.md${NC}"
echo ""
