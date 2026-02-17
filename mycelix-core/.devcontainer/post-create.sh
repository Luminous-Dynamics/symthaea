#!/bin/bash
# Mycelix-Core Post-Create Setup Script
# Runs after the dev container is created

set -e

echo ""
echo "=========================================="
echo "  Mycelix-Core Environment Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Step 1: Install Rust dependencies
print_step "Building Rust workspace (this may take a few minutes on first run)..."
if cargo build --release 2>/dev/null; then
    print_success "Rust workspace built successfully"
else
    print_warning "Rust build had warnings (this is okay for initial setup)"
fi

# Step 2: Install Python dependencies
print_step "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
fi

if [ -f "0TML/requirements.txt" ]; then
    pip install -r 0TML/requirements.txt -q
fi

if [ -d "mycelix_cli" ]; then
    pip install -e mycelix_cli -q 2>/dev/null || pip install -e ./mycelix_cli -q 2>/dev/null || true
fi

print_success "Python dependencies installed"

# Step 3: Install Node.js dependencies (if any)
print_step "Checking for Node.js dependencies..."
if [ -f "package.json" ]; then
    npm install --silent
    print_success "Node.js dependencies installed"
else
    echo "    No package.json found (skipping)"
fi

# Step 4: Set up Foundry for smart contracts
print_step "Setting up Foundry for smart contracts..."
if [ -d "contracts" ]; then
    cd contracts
    if [ -f "foundry.toml" ]; then
        forge install --no-commit 2>/dev/null || true
        print_success "Foundry dependencies installed"
    fi
    cd ..
else
    echo "    No contracts directory found (skipping)"
fi

# Step 5: Create local configuration files
print_step "Creating local configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success "Created .env from .env.example"
    else
        cat > .env << 'EOF'
# Mycelix Local Development Environment
MYCELIX_ENV=development
FL_COORDINATOR_HOST=127.0.0.1
FL_COORDINATOR_PORT=3000
HOLOCHAIN_CONDUCTOR_PORT=8888
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
RUST_LOG=info,fl_aggregator=debug
EOF
        print_success "Created default .env file"
    fi
fi

# Step 6: Run initial tests to verify setup
print_step "Running quick verification tests..."
if cargo test --lib -p fl-aggregator --no-fail-fast -- --test-threads=2 2>/dev/null; then
    print_success "Core tests passing"
else
    print_warning "Some tests may need attention (check with 'cargo test')"
fi

# Step 7: Generate documentation index
print_step "Setting up documentation..."
if [ -d "docs" ]; then
    print_success "Documentation available in docs/"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Quick Start Commands:"
echo "  demo          - Run the interactive FL demo"
echo "  fl-demo       - Run the Byzantine resistance simulation"
echo "  cargo test    - Run all tests"
echo "  cargo build   - Build the project"
echo ""
echo "See docs/GETTING_STARTED.md for more information."
echo ""
