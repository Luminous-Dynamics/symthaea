#!/usr/bin/env bash
#
# Mycelix Local Holochain Test Infrastructure
#
# Sets up Holochain conductor and runs zome integration tests.
# Requires: Holochain (holochain, hc)
#
# Usage:
#   ./scripts/local-holochain-testnet.sh start    # Start conductor
#   ./scripts/local-holochain-testnet.sh stop     # Stop conductor
#   ./scripts/local-holochain-testnet.sh status   # Check status
#   ./scripts/local-holochain-testnet.sh test     # Run zome tests
#   ./scripts/local-holochain-testnet.sh package  # Package hApps
#
# Environment:
#   HC_ADMIN_PORT     - Admin WebSocket port (default: 9000)
#   HC_APP_PORT       - App WebSocket port (default: 9001)
#   CONDUCTOR_DIR     - Conductor data directory (default: /tmp/mycelix-conductor)
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ZOMES_DIR="$PROJECT_ROOT/zomes"
HC_ADMIN_PORT="${HC_ADMIN_PORT:-9000}"
HC_APP_PORT="${HC_APP_PORT:-9001}"
CONDUCTOR_DIR="${CONDUCTOR_DIR:-/tmp/mycelix-conductor}"
CONDUCTOR_PID_FILE="$CONDUCTOR_DIR/conductor.pid"
CONDUCTOR_CONFIG="$CONDUCTOR_DIR/conductor-config.yaml"

check_holochain() {
    if ! command -v holochain &> /dev/null; then
        log_error "Holochain not found. Install Holochain or run 'nix develop' first."
        echo ""
        echo "Install Holochain:"
        echo "  nix-env -f https://github.com/holochain/holochain/archive/main-0.3.tar.gz -iA packages.holochain"
        echo ""
        echo "Or use nix:"
        echo "  nix develop"
        exit 1
    fi

    if ! command -v hc &> /dev/null; then
        log_error "hc CLI not found. Install Holochain dev tools."
        exit 1
    fi

    local version
    version=$(holochain --version 2>/dev/null | head -1)
    log_success "Holochain available: $version"
}

create_conductor_config() {
    log_step "Creating conductor configuration..."

    mkdir -p "$CONDUCTOR_DIR"

    cat > "$CONDUCTOR_CONFIG" << EOF
---
environment_path: $CONDUCTOR_DIR/databases
keystore:
  type: lair_server
  connection_url: unix://$CONDUCTOR_DIR/keystore/socket
admin_interfaces:
  - driver:
      type: websocket
      port: $HC_ADMIN_PORT
network:
  transport_pool:
    - type: webrtc
  bootstrap_service: https://bootstrap.holo.host
  network_type: quic_mdns
EOF

    log_success "Conductor config created: $CONDUCTOR_CONFIG"
}

start_conductor() {
    log_step "Starting Holochain conductor..."

    if [ -f "$CONDUCTOR_PID_FILE" ]; then
        local old_pid
        old_pid=$(cat "$CONDUCTOR_PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log_warn "Conductor already running (PID: $old_pid)"
            return 0
        fi
        rm -f "$CONDUCTOR_PID_FILE"
    fi

    # Create config if needed
    if [ ! -f "$CONDUCTOR_CONFIG" ]; then
        create_conductor_config
    fi

    # Initialize keystore if needed
    local keystore_dir="$CONDUCTOR_DIR/keystore"
    if [ ! -d "$keystore_dir" ]; then
        log_info "Initializing Lair keystore..."
        mkdir -p "$keystore_dir"
        # Lair will be started automatically by the conductor
    fi

    # Start conductor
    holochain -c "$CONDUCTOR_CONFIG" &
    local conductor_pid=$!
    echo "$conductor_pid" > "$CONDUCTOR_PID_FILE"

    # Wait for conductor to be ready
    log_info "Waiting for conductor to start..."
    local retries=30
    while ! curl -s "http://localhost:$HC_ADMIN_PORT" &>/dev/null; do
        ((retries--))
        if [ $retries -le 0 ]; then
            log_error "Conductor failed to start"
            kill "$conductor_pid" 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done

    log_success "Conductor running (PID: $conductor_pid)"
    echo ""
    echo "Admin WebSocket: ws://localhost:$HC_ADMIN_PORT"
    echo "Conductor data: $CONDUCTOR_DIR"
}

stop_conductor() {
    log_step "Stopping Holochain conductor..."

    if [ -f "$CONDUCTOR_PID_FILE" ]; then
        local pid
        pid=$(cat "$CONDUCTOR_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            log_success "Conductor stopped (PID: $pid)"
        fi
        rm -f "$CONDUCTOR_PID_FILE"
    else
        log_warn "Conductor PID file not found"
        pkill -f "holochain.*conductor-config" 2>/dev/null || true
    fi
}

check_status() {
    log_step "Checking conductor status..."

    if [ -f "$CONDUCTOR_PID_FILE" ]; then
        local pid
        pid=$(cat "$CONDUCTOR_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_success "Conductor running (PID: $pid)"
            echo "Admin port: $HC_ADMIN_PORT"
            return 0
        fi
    fi

    log_warn "Conductor not running"
    return 1
}

package_happs() {
    log_step "Packaging Mycelix hApps..."

    if [ ! -d "$ZOMES_DIR" ]; then
        log_error "Zomes directory not found: $ZOMES_DIR"
        exit 1
    fi

    local output_dir="$PROJECT_ROOT/target/happs"
    mkdir -p "$output_dir"

    # Build zomes
    local zomes=(
        "federated_learning"
        "pogq_validation"
        "bridge"
        "agents"
    )

    for zome in "${zomes[@]}"; do
        local zome_dir="$ZOMES_DIR/$zome"
        if [ -d "$zome_dir" ]; then
            log_info "Building $zome zome..."

            # Build coordinator zome
            if [ -d "$zome_dir/coordinator" ]; then
                cd "$zome_dir/coordinator"
                cargo build --release --target wasm32-unknown-unknown
            fi

            # Build integrity zome
            if [ -d "$zome_dir/integrity" ]; then
                cd "$zome_dir/integrity"
                cargo build --release --target wasm32-unknown-unknown
            fi

            log_success "$zome built"
        fi
    done

    # Package DNA if manifest exists
    local dna_manifests=("$PROJECT_ROOT"/dnas/*/dna.yaml)
    for manifest in "${dna_manifests[@]}"; do
        if [ -f "$manifest" ]; then
            local dna_dir
            dna_dir=$(dirname "$manifest")
            local dna_name
            dna_name=$(basename "$dna_dir")

            log_info "Packaging $dna_name DNA..."
            cd "$dna_dir"
            hc dna pack . -o "$output_dir/$dna_name.dna"
            log_success "$dna_name.dna created"
        fi
    done

    # Package hApp if manifest exists
    local happ_manifests=("$PROJECT_ROOT"/happs/*/happ.yaml "$PROJECT_ROOT"/*.happ.yaml)
    for manifest in "${happ_manifests[@]}"; do
        if [ -f "$manifest" ]; then
            local happ_dir
            happ_dir=$(dirname "$manifest")
            local happ_name
            happ_name=$(basename "$manifest" .happ.yaml)

            log_info "Packaging $happ_name hApp..."
            cd "$happ_dir"
            hc app pack . -o "$output_dir/$happ_name.happ"
            log_success "$happ_name.happ created"
        fi
    done

    log_success "All packages created in $output_dir"
}

run_zome_tests() {
    log_step "Running zome integration tests..."

    # Check conductor is running
    if ! check_status &>/dev/null; then
        log_warn "Conductor not running, starting..."
        start_conductor
    fi

    # Export environment for tests
    export HC_ADMIN_WS="ws://localhost:$HC_ADMIN_PORT"
    export CONDUCTOR_DIR

    # Run Rust tests for each zome
    local zomes=(
        "federated_learning"
        "pogq_validation"
        "bridge"
        "agents"
    )

    for zome in "${zomes[@]}"; do
        local zome_dir="$ZOMES_DIR/$zome"
        if [ -d "$zome_dir" ]; then
            log_info "Testing $zome zome..."

            # Run coordinator tests
            if [ -d "$zome_dir/coordinator" ]; then
                cd "$zome_dir/coordinator"
                cargo test --features conductor_tests -- --test-threads=1 || {
                    log_warn "$zome coordinator tests failed or skipped"
                }
            fi

            # Run integration tests if they exist
            local test_dir="$zome_dir/tests"
            if [ -d "$test_dir" ]; then
                cd "$test_dir"
                if [ -f "package.json" ]; then
                    log_info "Running TypeScript tests for $zome..."
                    npm install
                    npm test || log_warn "$zome TypeScript tests failed"
                fi
            fi
        fi
    done

    # Run cross-zome integration tests
    local integration_tests="$PROJECT_ROOT/tests/integration"
    if [ -d "$integration_tests" ]; then
        log_info "Running cross-zome integration tests..."
        cd "$integration_tests"
        cargo test --features holochain_integration -- --test-threads=1
    fi

    log_success "Zome tests complete"
}

install_happ() {
    local happ_path="${2:-}"

    if [ -z "$happ_path" ]; then
        log_error "Usage: $0 install <path-to-happ>"
        exit 1
    fi

    if [ ! -f "$happ_path" ]; then
        log_error "hApp file not found: $happ_path"
        exit 1
    fi

    log_step "Installing hApp: $happ_path"

    # Check conductor is running
    if ! check_status &>/dev/null; then
        log_error "Conductor not running. Start it first."
        exit 1
    fi

    # Generate agent key if needed
    log_info "Installing hApp via admin API..."

    # Use hc admin command to install
    hc sandbox call --running "$HC_ADMIN_PORT" install-app "$happ_path"

    log_success "hApp installed"
}

# Main
case "${1:-}" in
    start)
        check_holochain
        start_conductor
        ;;
    stop)
        stop_conductor
        ;;
    status)
        check_status
        ;;
    test)
        check_holochain
        run_zome_tests
        ;;
    package)
        check_holochain
        package_happs
        ;;
    install)
        check_holochain
        install_happ "$@"
        ;;
    clean)
        log_step "Cleaning conductor data..."
        stop_conductor 2>/dev/null || true
        rm -rf "$CONDUCTOR_DIR"
        log_success "Conductor data cleaned"
        ;;
    *)
        echo "Mycelix Local Holochain Test Infrastructure"
        echo ""
        echo "Usage: $0 {start|stop|status|test|package|install|clean}"
        echo ""
        echo "Commands:"
        echo "  start           - Start Holochain conductor"
        echo "  stop            - Stop conductor"
        echo "  status          - Check conductor status"
        echo "  test            - Run zome integration tests"
        echo "  package         - Package DNAs and hApps"
        echo "  install <happ>  - Install a hApp to running conductor"
        echo "  clean           - Remove all conductor data"
        exit 1
        ;;
esac
