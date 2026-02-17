# Mycelix-Mail Development Commands
# Run `just` to see all available commands

set dotenv-load

# Default: show help
default:
    @just --list

# ============================================================================
# Development
# ============================================================================

# Start all services for development
dev: ipfs-start
    @echo "🍄 Starting Mycelix-Mail development environment..."
    just backend &
    just frontend &
    @echo ""
    @echo "Services:"
    @echo "  Backend:  http://localhost:3001"
    @echo "  Frontend: http://localhost:5173"
    @echo "  IPFS:     http://localhost:5001"
    @echo ""
    wait

# Start backend only
backend:
    cd happ/backend-rs && cargo run

# Start backend with auto-reload
backend-watch:
    cd happ/backend-rs && cargo watch -x run

# Start frontend dev server
frontend:
    cd ui/frontend && npm run dev

# ============================================================================
# Building
# ============================================================================

# Build all components
build: build-dna build-backend build-frontend
    @echo "✅ All components built successfully"

# Build Holochain DNA
build-dna:
    @echo "Building DNA..."
    cd happ/dna/integrity && cargo build --release --target wasm32-unknown-unknown
    cd happ/dna/zomes/mail_messages && cargo build --release --target wasm32-unknown-unknown
    cd happ/dna/zomes/trust_filter && cargo build --release --target wasm32-unknown-unknown
    @echo "✅ DNA zomes built"

# Build Rust backend
build-backend:
    @echo "Building backend..."
    cd happ/backend-rs && cargo build --release
    @echo "✅ Backend built: happ/backend-rs/target/release/mycelix-mail-backend"

# Build frontend for production
build-frontend:
    @echo "Building frontend..."
    cd ui/frontend && npm run build
    @echo "✅ Frontend built: ui/frontend/dist/"

# ============================================================================
# Testing
# ============================================================================

# Run all tests
test: test-dna test-backend test-frontend
    @echo "✅ All tests passed"

# Test DNA zomes
test-dna:
    @echo "Testing DNA..."
    cd happ/dna && cargo test
    @echo "✅ DNA tests passed"

# Test backend
test-backend:
    @echo "Testing backend..."
    cd happ/backend-rs && cargo test
    @echo "✅ Backend tests passed"

# Test frontend
test-frontend:
    @echo "Testing frontend..."
    cd ui/frontend && npm test -- --run
    @echo "✅ Frontend tests passed"

# Run backend tests with coverage
test-coverage:
    cd happ/backend-rs && cargo tarpaulin --out Html

# ============================================================================
# Linting & Formatting
# ============================================================================

# Format all code
fmt: fmt-rust fmt-frontend
    @echo "✅ All code formatted"

# Format Rust code
fmt-rust:
    cd happ/backend-rs && cargo fmt
    cd happ/dna && cargo fmt
    cd happ/cli && cargo fmt

# Format frontend code
fmt-frontend:
    cd ui/frontend && npm run lint:fix

# Check code style
lint: lint-rust lint-frontend
    @echo "✅ All linting passed"

# Lint Rust code
lint-rust:
    cd happ/backend-rs && cargo clippy -- -D warnings
    cd happ/dna && cargo clippy -- -D warnings

# Lint frontend code
lint-frontend:
    cd ui/frontend && npm run lint

# ============================================================================
# Infrastructure
# ============================================================================

# Start IPFS daemon
ipfs-start:
    @if ! pgrep -x "ipfs" > /dev/null; then \
        echo "Starting IPFS daemon..."; \
        ipfs daemon --init &; \
        sleep 2; \
    else \
        echo "IPFS already running"; \
    fi

# Stop IPFS daemon
ipfs-stop:
    @pkill ipfs || true
    @echo "IPFS stopped"

# Start Holochain conductor
holochain-start:
    @echo "Starting Holochain conductor..."
    holochain sandbox generate --run=4444 workdir

# Install DNA to conductor
holochain-install:
    hc sandbox call install-app happ/workdir/mycelix-mail.happ

# ============================================================================
# Setup
# ============================================================================

# Initial setup for new developers
setup: setup-frontend setup-env
    @echo "✅ Setup complete! Run 'just dev' to start development"

# Install frontend dependencies
setup-frontend:
    cd ui/frontend && npm install

# Create .env files from examples
setup-env:
    @if [ ! -f happ/backend-rs/.env ]; then \
        cp happ/backend-rs/.env.example happ/backend-rs/.env; \
        echo "Created happ/backend-rs/.env"; \
    fi
    @if [ ! -f ui/frontend/.env ]; then \
        cp ui/frontend/.env.example ui/frontend/.env; \
        echo "Created ui/frontend/.env"; \
    fi

# ============================================================================
# Cleanup
# ============================================================================

# Clean all build artifacts
clean: clean-rust clean-frontend
    @echo "✅ All build artifacts cleaned"

# Clean Rust build artifacts
clean-rust:
    cd happ/backend-rs && cargo clean
    cd happ/dna && cargo clean
    cd happ/cli && cargo clean

# Clean frontend build artifacts
clean-frontend:
    rm -rf ui/frontend/dist
    rm -rf ui/frontend/node_modules/.vite

# ============================================================================
# Documentation
# ============================================================================

# Generate API documentation
docs:
    cd happ/backend-rs && cargo doc --open

# ============================================================================
# Release
# ============================================================================

# Create a release build
release: build
    @echo "Creating release..."
    mkdir -p release
    cp happ/backend-rs/target/release/mycelix-mail-backend release/
    cp -r ui/frontend/dist release/frontend
    @echo "✅ Release created in release/"

# Check release readiness
check-release:
    @echo "Checking release readiness..."
    just lint
    just test
    just build
    @echo "✅ Ready for release"
