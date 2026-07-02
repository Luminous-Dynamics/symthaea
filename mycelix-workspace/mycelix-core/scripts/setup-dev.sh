#!/usr/bin/env bash
#
# Mycelix Development Environment Setup
#
# This script sets up the development environment for new contributors.
# Run once after cloning the repository.
#
# Usage: ./scripts/setup-dev.sh
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "  Mycelix Development Environment Setup"
echo "=========================================="
echo ""

# Check for required tools
log_info "Checking prerequisites..."

check_tool() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 is installed"
        return 0
    else
        log_warn "$1 is not installed"
        return 1
    fi
}

MISSING_TOOLS=()

check_tool "rustc" || MISSING_TOOLS+=("rust")
check_tool "cargo" || MISSING_TOOLS+=("rust")
check_tool "python3" || MISSING_TOOLS+=("python3")
check_tool "pip" || MISSING_TOOLS+=("pip")
check_tool "docker" || MISSING_TOOLS+=("docker")
check_tool "git" || MISSING_TOOLS+=("git")

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    log_error "Missing required tools: ${MISSING_TOOLS[*]}"
    echo ""
    echo "Please install the missing tools and run this script again."
    echo ""
    echo "Install instructions:"
    echo "  Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  Python: https://www.python.org/downloads/"
    echo "  Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Install Rust components
log_info "Installing Rust components..."
rustup component add clippy rustfmt 2>/dev/null || true
log_success "Rust components installed"

# Install pre-commit
log_info "Installing pre-commit..."
pip install --quiet pre-commit
log_success "pre-commit installed"

# Install pre-commit hooks
log_info "Setting up pre-commit hooks..."
cd "$(dirname "$0")/.."
pre-commit install
pre-commit install --hook-type commit-msg
log_success "Pre-commit hooks installed"

# Create secrets baseline if it doesn't exist
if [ ! -f ".secrets.baseline" ]; then
    log_info "Creating secrets baseline..."
    detect-secrets scan > .secrets.baseline 2>/dev/null || true
    log_success "Secrets baseline created"
fi

# Create local environment file
if [ ! -f ".env.local" ]; then
    log_info "Creating local environment file..."
    if [ -f ".env.example" ]; then
        cp .env.example .env.local
        log_success "Created .env.local from .env.example"
    else
        cat > .env.local << 'EOF'
# Mycelix Local Development Environment
# Copy this file and fill in your values

# Database (for local development)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mycelix

# Redis
REDIS_URL=redis://localhost:6379

# Logging
RUST_LOG=info,mycelix=debug

# Feature flags
ENABLE_METRICS=true
ENABLE_TRACING=false
EOF
        log_success "Created .env.local template"
    fi
    log_warn "Please edit .env.local with your configuration"
fi

# Build libraries (optional)
echo ""
read -p "Build all Rust libraries? This may take a few minutes. (y/N): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Building Rust libraries..."
    for lib in libs/*/; do
        if [ -f "$lib/Cargo.toml" ]; then
            log_info "Building $(basename "$lib")..."
            (cd "$lib" && cargo build --release 2>/dev/null) || log_warn "Failed to build $(basename "$lib")"
        fi
    done
    log_success "Libraries built"
fi

# Start local services (optional)
echo ""
read -p "Start local development services (PostgreSQL, Redis)? (y/N): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "deployment/docker-compose.dev.yml" ]; then
        log_info "Starting local services..."
        docker compose -f deployment/docker-compose.dev.yml up -d
        log_success "Local services started"
    else
        log_warn "docker-compose.dev.yml not found, skipping"
    fi
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env.local with your configuration"
echo "  2. Run 'cargo test' in any library to verify setup"
echo "  3. See CONTRIBUTING.md for development workflow"
echo ""
echo "Useful commands:"
echo "  cargo fmt          # Format code"
echo "  cargo clippy       # Run linter"
echo "  cargo test         # Run tests"
echo "  pre-commit run     # Run pre-commit checks"
echo ""
log_success "Happy coding!"
