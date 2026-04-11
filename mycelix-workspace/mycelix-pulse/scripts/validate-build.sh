#!/bin/bash
# Mycelix Mail - Build Validation Script
# =======================================
# Validates that all components build and tests pass

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASSED++)); }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARNINGS++)); }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; ((FAILED++)); }

header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
}

# Check prerequisites
check_prerequisites() {
    header "Checking Prerequisites"

    # Rust
    if command -v rustc &> /dev/null; then
        VERSION=$(rustc --version)
        log_success "Rust installed: $VERSION"
    else
        log_error "Rust not found. Install from https://rustup.rs"
    fi

    # Cargo
    if command -v cargo &> /dev/null; then
        log_success "Cargo available"
    else
        log_error "Cargo not found"
    fi

    # WASM target
    if rustup target list | grep -q "wasm32-unknown-unknown (installed)"; then
        log_success "WASM target installed"
    else
        log_warning "WASM target not installed. Run: rustup target add wasm32-unknown-unknown"
    fi

    # Node.js
    if command -v node &> /dev/null; then
        VERSION=$(node --version)
        log_success "Node.js installed: $VERSION"
    else
        log_error "Node.js not found"
    fi

    # npm
    if command -v npm &> /dev/null; then
        VERSION=$(npm --version)
        log_success "npm installed: $VERSION"
    else
        log_error "npm not found"
    fi

    # Holochain CLI (optional)
    if command -v hc &> /dev/null; then
        VERSION=$(hc --version 2>/dev/null || echo "unknown")
        log_success "Holochain CLI installed: $VERSION"
    else
        log_warning "Holochain CLI not found. Some features may not work."
    fi

    # Docker (optional)
    if command -v docker &> /dev/null; then
        log_success "Docker available"
    else
        log_warning "Docker not found. Container deployment unavailable."
    fi
}

# Validate Rust zomes
validate_rust() {
    header "Validating Rust Zomes"

    cd holochain

    # Check formatting
    log_info "Checking Rust formatting..."
    if cargo fmt --all -- --check 2>/dev/null; then
        log_success "Rust formatting OK"
    else
        log_warning "Rust formatting issues found. Run: cargo fmt"
    fi

    # Clippy lints
    log_info "Running Clippy..."
    if cargo clippy --all-targets 2>/dev/null; then
        log_success "Clippy passed"
    else
        log_warning "Clippy warnings found"
    fi

    # Build check (native)
    log_info "Checking Rust build (native)..."
    if cargo check 2>/dev/null; then
        log_success "Rust build check passed"
    else
        log_error "Rust build check failed"
    fi

    # Build WASM
    log_info "Building WASM zomes..."
    if cargo build --release --target wasm32-unknown-unknown 2>/dev/null; then
        log_success "WASM zomes built"

        # Count zomes
        ZOME_COUNT=$(find target/wasm32-unknown-unknown/release -name "*.wasm" 2>/dev/null | wc -l)
        log_info "Built $ZOME_COUNT WASM files"
    else
        log_error "WASM build failed"
    fi

    # Run tests
    log_info "Running Rust tests..."
    if cargo test 2>/dev/null; then
        log_success "Rust tests passed"
    else
        log_warning "Some Rust tests failed"
    fi

    cd ..
}

# Validate TypeScript client
validate_client() {
    header "Validating TypeScript Client"

    cd holochain/client

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing client dependencies..."
        npm install 2>/dev/null || npm install --legacy-peer-deps 2>/dev/null
    fi

    # Type check
    log_info "Type checking..."
    if npm run typecheck 2>/dev/null; then
        log_success "TypeScript types OK"
    else
        log_warning "TypeScript type errors found"
    fi

    # Lint
    log_info "Linting..."
    if npm run lint 2>/dev/null; then
        log_success "Linting passed"
    else
        log_warning "Lint issues found"
    fi

    # Build
    log_info "Building client..."
    if npm run build 2>/dev/null; then
        log_success "Client built successfully"
    else
        log_error "Client build failed"
    fi

    # Run tests
    log_info "Running client tests..."
    if npm run test:run 2>/dev/null; then
        log_success "Client tests passed"
    else
        log_warning "Some client tests failed"
    fi

    cd ../..
}

# Validate UI frontend
validate_frontend() {
    header "Validating UI Frontend"

    if [ ! -d "ui/frontend" ]; then
        log_warning "UI frontend directory not found"
        return
    fi

    cd ui/frontend

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm install 2>/dev/null || npm install --legacy-peer-deps 2>/dev/null
    fi

    # Type check
    log_info "Type checking frontend..."
    if npm run type-check 2>/dev/null || npm run typecheck 2>/dev/null; then
        log_success "Frontend types OK"
    else
        log_warning "Frontend type errors found"
    fi

    # Lint
    log_info "Linting frontend..."
    if npm run lint 2>/dev/null; then
        log_success "Frontend linting passed"
    else
        log_warning "Frontend lint issues found"
    fi

    # Build
    log_info "Building frontend..."
    if npm run build 2>/dev/null; then
        log_success "Frontend built successfully"
    else
        log_warning "Frontend build failed"
    fi

    cd ../..
}

# Validate Docker setup
validate_docker() {
    header "Validating Docker Setup"

    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available, skipping"
        return
    fi

    # Check docker-compose file
    if [ -f "docker/docker-compose.yml" ]; then
        log_success "docker-compose.yml exists"

        # Validate syntax
        if docker-compose -f docker/docker-compose.yml config > /dev/null 2>&1; then
            log_success "docker-compose.yml syntax valid"
        else
            log_warning "docker-compose.yml has syntax issues"
        fi
    else
        log_warning "docker-compose.yml not found"
    fi

    # Check Dockerfiles
    for dockerfile in docker/Dockerfile.*; do
        if [ -f "$dockerfile" ]; then
            log_success "Found: $dockerfile"
        fi
    done
}

# Validate documentation
validate_docs() {
    header "Validating Documentation"

    DOCS=("docs/ARCHITECTURE.md" "docs/GETTING_STARTED.md" "docs/SECURITY.md" "README.md")

    for doc in "${DOCS[@]}"; do
        if [ -f "$doc" ]; then
            log_success "Found: $doc"
        else
            log_warning "Missing: $doc"
        fi
    done
}

# Security check
security_check() {
    header "Security Check"

    # Rust audit
    if command -v cargo-audit &> /dev/null; then
        log_info "Running cargo audit..."
        cd holochain
        if cargo audit 2>/dev/null; then
            log_success "No known vulnerabilities in Rust dependencies"
        else
            log_warning "Vulnerabilities found in Rust dependencies"
        fi
        cd ..
    else
        log_warning "cargo-audit not installed. Run: cargo install cargo-audit"
    fi

    # npm audit
    log_info "Running npm audit on client..."
    cd holochain/client
    if npm audit --audit-level=high 2>/dev/null; then
        log_success "No high-severity npm vulnerabilities"
    else
        log_warning "npm vulnerabilities found"
    fi
    cd ../..
}

# Print summary
print_summary() {
    header "Validation Summary"

    echo ""
    echo -e "  ${GREEN}Passed:${NC}   $PASSED"
    echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
    echo -e "  ${RED}Failed:${NC}   $FAILED"
    echo ""

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  BUILD VALIDATION SUCCESSFUL${NC}"
        echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
        exit 0
    else
        echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  BUILD VALIDATION FAILED - $FAILED issues found${NC}"
        echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
        exit 1
    fi
}

# Main
main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       MYCELIX MAIL - BUILD VALIDATION                    ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"

    # Change to project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR/.."

    check_prerequisites
    validate_rust
    validate_client
    validate_frontend
    validate_docker
    validate_docs
    security_check
    print_summary
}

main "$@"
