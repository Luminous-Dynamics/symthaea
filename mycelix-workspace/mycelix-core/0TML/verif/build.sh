#!/usr/bin/env bash
#
# VSV-STARK Build Script
# ======================
#
# Builds both Rust library and Python extension.
#
# Usage:
#   ./verif/build.sh          # Build everything
#   ./verif/build.sh test     # Build + run tests
#   ./verif/build.sh clean    # Clean build artifacts
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================"
echo "VSV-STARK Build Script"
echo "======================================================================"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Cargo not found. Install Rust: https://rustup.rs/${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

if ! command -v maturin &> /dev/null; then
    echo -e "${YELLOW}⚠️  maturin not found. Installing...${NC}"
    pip install maturin
fi

echo -e "${GREEN}✅ All prerequisites satisfied${NC}"
echo ""

# Clean mode
if [ "${1:-}" = "clean" ]; then
    echo "🧹 Cleaning build artifacts..."
    cd air
    cargo clean
    cd ../python
    cargo clean
    rm -rf target/
    echo -e "${GREEN}✅ Clean complete${NC}"
    exit 0
fi

# Build Rust library
echo "🦀 Building Rust AIR library..."
cd air
if cargo build --release; then
    echo -e "${GREEN}✅ Rust library built successfully${NC}"
else
    echo -e "${RED}❌ Rust library build failed${NC}"
    exit 1
fi
echo ""

# Run Rust tests
echo "🧪 Running Rust tests..."
if cargo test --release; then
    echo -e "${GREEN}✅ Rust tests passed${NC}"
else
    echo -e "${RED}❌ Rust tests failed${NC}"
    exit 1
fi
echo ""

# Build Python extension
echo "🐍 Building Python extension..."
cd ../python
if maturin develop --release; then
    echo -e "${GREEN}✅ Python extension built successfully${NC}"
else
    echo -e "${RED}❌ Python extension build failed${NC}"
    exit 1
fi
echo ""

# Verify installation
echo "✓ Verifying installation..."
if python3 -c "import vsv_stark; print(f'VSV-STARK v{vsv_stark.__version__} installed')"; then
    echo -e "${GREEN}✅ Installation verified${NC}"
else
    echo -e "${RED}❌ Installation verification failed${NC}"
    exit 1
fi
echo ""

# Test mode
if [ "${1:-}" = "test" ]; then
    echo "🧪 Running integration tests..."
    cd ..
    if [ -d "tests" ]; then
        python3 -m pytest tests/ -v
    else
        echo -e "${YELLOW}⚠️  No integration tests found${NC}"
    fi
    echo ""
fi

echo "======================================================================"
echo -e "${GREEN}✅ Build complete${NC}"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Test the installation:"
echo "     python3 -c 'import vsv_stark; print(vsv_stark.__version__)'"
echo ""
echo "  2. Generate a proof:"
echo "     python scripts/prove_decisions.py results/artifacts_run_001/"
echo ""
echo "  3. See verif/README.md for full documentation"
echo ""
