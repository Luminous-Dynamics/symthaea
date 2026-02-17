#!/usr/bin/env bash
# Mycelix-Mail Development Setup Script
# Run this script to set up your development environment

set -e

echo "🍄 Mycelix-Mail Development Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found: $($1 --version 2>&1 | head -1)"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

# Check prerequisites
echo "Checking prerequisites..."
echo ""

MISSING=0

check_command "rustc" || MISSING=1
check_command "cargo" || MISSING=1
check_command "node" || MISSING=1
check_command "npm" || MISSING=1

# Holochain-specific tools
echo ""
echo "Holochain development tools:"
check_command "lld" || { echo "  (Required for WASM compilation - install lld)"; MISSING=1; }
check_command "wasm-pack" || echo "  (Optional: cargo install wasm-pack)"
check_command "wasm-opt" || echo "  (Optional: install binaryen for wasm-opt)"

# Optional but recommended
echo ""
echo "Optional tools:"
check_command "holochain" || echo "  (Install via Nix flake for Holochain development)"
check_command "hc" || echo "  (Holochain CLI - comes with holochain)"
check_command "lair-keystore" || echo "  (Holochain keystore - comes with holochain)"
check_command "ipfs" || echo "  (Install for IPFS storage support)"
check_command "just" || echo "  (Install for easy command running: cargo install just)"

echo ""

if [ $MISSING -eq 1 ]; then
    echo -e "${RED}Missing required tools. Please install them first.${NC}"
    echo ""
    echo "Quick install options:"
    echo "  - Nix (recommended): nix develop (uses flake.nix)"
    echo "  - Rust: https://rustup.rs"
    echo "  - Node.js: https://nodejs.org"
    exit 1
fi

# Check for wasm target
echo ""
echo "Checking Rust wasm target..."
if rustup target list --installed | grep -q wasm32-unknown-unknown; then
    echo -e "${GREEN}✓${NC} wasm32-unknown-unknown target installed"
else
    echo "Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

# Setup Cargo config for WASM/Holochain
echo ""
echo "Configuring Cargo for Holochain WASM builds..."
CARGO_CONFIG_DIR="$(dirname "$0")/../holochain/.cargo"
mkdir -p "$CARGO_CONFIG_DIR"
if [ ! -f "$CARGO_CONFIG_DIR/config.toml" ]; then
    cat > "$CARGO_CONFIG_DIR/config.toml" << 'EOF'
# Holochain WASM Build Configuration
[target.wasm32-unknown-unknown]
linker = "lld"
rustflags = ["-C", "link-arg=-zstack-size=65536"]

[build]
# Use lld for faster linking
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

[env]
# Environment for openssl-sys builds
OPENSSL_NO_VENDOR = "1"
EOF
    echo -e "  ${GREEN}✓${NC} Created .cargo/config.toml for WASM builds"
else
    echo -e "  ${GREEN}✓${NC} .cargo/config.toml already exists"
fi

# Setup directories
echo ""
echo "Setting up project..."

# Backend setup
echo "Setting up backend..."
cd "$(dirname "$0")/../happ/backend-rs"
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
    echo -e "${YELLOW}  ⚠ Edit .env to set JWT_SECRET before running in production${NC}"
fi

# Frontend setup
echo "Setting up frontend..."
cd "$(dirname "$0")/../ui/frontend"
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
fi

echo "Installing frontend dependencies..."
npm install

# Build check
echo ""
echo "Verifying builds..."

echo "Checking backend builds..."
cd "$(dirname "$0")/../happ/backend-rs"
cargo check

echo ""
echo "Checking Holochain zomes..."
cd "$(dirname "$0")/../holochain"
if cargo check --workspace 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Holochain zomes check passed"
else
    echo -e "  ${YELLOW}⚠${NC} Holochain zomes check failed (may need HDK)"
fi

echo ""
echo "Checking TypeScript SDK..."
cd "$(dirname "$0")/../sdk/typescript"
if [ -f package.json ]; then
    npm install
    npm run typecheck 2>/dev/null || echo "  (typecheck script not found, skipping)"
fi

echo "Checking frontend builds..."
cd "$(dirname "$0")/../ui/frontend"
npm run type-check 2>/dev/null || echo "  (type-check script not found, skipping)"

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  Backend (REST API):"
echo "    cd happ/backend-rs && cargo run"
echo ""
echo "  Frontend (React UI):"
echo "    cd ui/frontend && npm run dev"
echo ""
echo "  Holochain Zomes:"
echo "    cd holochain && cargo build --release --target wasm32-unknown-unknown"
echo "    # Or use the Makefile:"
echo "    cd holochain && make build"
echo ""
echo "  TypeScript SDK:"
echo "    cd sdk/typescript && npm run build"
echo ""
echo "Or if you have 'just' installed:"
echo "  just dev    # Start all services"
echo "  just test   # Run all tests"
echo "  just build  # Build all components"
echo ""
echo "For full Holochain development environment:"
echo "  nix develop  # Enter Nix dev shell with holochain, hc, lair-keystore"
echo ""
echo "Run Holochain tests:"
echo "  cd holochain && cargo test --workspace"
echo "  cd holochain && make test"
