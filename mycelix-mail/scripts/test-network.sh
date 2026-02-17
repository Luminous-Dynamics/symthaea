#!/bin/bash
# Mycelix Mail - Multi-Node Test Network
# =======================================
# Spawns multiple Holochain conductors for P2P testing

set -e

# Configuration
NUM_NODES=${1:-3}
BASE_APP_PORT=8888
BASE_ADMIN_PORT=9000
NETWORK_SEED="mycelix-test-$(date +%s)"
DATA_DIR="/tmp/mycelix-test-network"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Cleanup function
cleanup() {
    log_info "Shutting down test network..."

    # Kill all conductor processes
    pkill -f "holochain.*mycelix-test" 2>/dev/null || true

    # Clean up data directory
    rm -rf "$DATA_DIR"

    log_success "Cleanup complete"
}

# Trap for cleanup on exit
trap cleanup EXIT INT TERM

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v holochain &> /dev/null; then
        log_error "Holochain not found. Install with: cargo install holochain"
        exit 1
    fi

    if ! command -v hc &> /dev/null; then
        log_error "Holochain CLI not found. Install with: cargo install holochain_cli"
        exit 1
    fi

    if ! command -v lair-keystore &> /dev/null; then
        log_error "Lair keystore not found. Install with: cargo install lair_keystore"
        exit 1
    fi

    log_success "Prerequisites OK"
}

# Create conductor config for a node
create_conductor_config() {
    local node_id=$1
    local app_port=$((BASE_APP_PORT + node_id))
    local admin_port=$((BASE_ADMIN_PORT + node_id))
    local node_dir="$DATA_DIR/node-$node_id"

    mkdir -p "$node_dir"

    cat > "$node_dir/conductor-config.yaml" << EOF
---
environment_path: $node_dir/databases
keystore:
  type: lair_server_in_proc
admin_interfaces:
  - driver:
      type: websocket
      port: $admin_port
      allowed_origins: "*"
network:
  network_type: quic_bootstrap
  bootstrap_service: null
  transport_pool:
    - type: quic
  tuning_params:
    gossip_loop_iteration_delay_ms: 500
    gossip_peer_on_success_next_gossip_delay_ms: 5000
    gossip_peer_on_error_next_gossip_delay_ms: 10000
    gossip_local_sync_delay_ms: 5000
db_sync_strategy: Fast
EOF

    echo "$node_dir/conductor-config.yaml"
}

# Start a conductor node
start_node() {
    local node_id=$1
    local config_path=$2
    local app_port=$((BASE_APP_PORT + node_id))
    local admin_port=$((BASE_ADMIN_PORT + node_id))
    local node_dir="$DATA_DIR/node-$node_id"
    local log_file="$node_dir/conductor.log"

    log_info "Starting node $node_id (app: $app_port, admin: $admin_port)..."

    # Start conductor in background
    RUST_LOG=info holochain \
        -c "$config_path" \
        > "$log_file" 2>&1 &

    local pid=$!
    echo $pid > "$node_dir/conductor.pid"

    # Wait for startup
    sleep 3

    if kill -0 $pid 2>/dev/null; then
        log_success "Node $node_id started (PID: $pid)"
        return 0
    else
        log_error "Node $node_id failed to start"
        cat "$log_file"
        return 1
    fi
}

# Install app on a node
install_app() {
    local node_id=$1
    local admin_port=$((BASE_ADMIN_PORT + node_id))
    local app_port=$((BASE_APP_PORT + node_id))

    log_info "Installing app on node $node_id..."

    # Check if hApp exists
    local happ_path="holochain/workdir/happ/mycelix-mail.happ"
    if [ ! -f "$happ_path" ]; then
        log_info "Building hApp..."
        cd holochain
        hc dna pack workdir/dna 2>/dev/null || true
        hc app pack workdir/happ 2>/dev/null || true
        cd ..
    fi

    if [ -f "$happ_path" ]; then
        # Install via admin API
        hc sandbox call \
            --port $admin_port \
            install-app \
            --app-id "mycelix-mail-$node_id" \
            --path "$happ_path" \
            2>/dev/null || log_warning "App install skipped (may already exist)"

        # Attach app interface
        hc sandbox call \
            --port $admin_port \
            attach-app-interface \
            --port $app_port \
            2>/dev/null || true

        log_success "App installed on node $node_id"
    else
        log_warning "hApp not found, skipping install"
    fi
}

# Get agent pubkey for a node
get_agent_pubkey() {
    local node_id=$1
    local admin_port=$((BASE_ADMIN_PORT + node_id))

    hc sandbox call --port $admin_port list-agents 2>/dev/null | head -1
}

# Print network status
print_status() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  MYCELIX MAIL TEST NETWORK STATUS${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""

    for i in $(seq 0 $((NUM_NODES - 1))); do
        local app_port=$((BASE_APP_PORT + i))
        local admin_port=$((BASE_ADMIN_PORT + i))
        local node_dir="$DATA_DIR/node-$i"
        local pid_file="$node_dir/conductor.pid"

        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 $pid 2>/dev/null; then
                echo -e "  ${GREEN}Node $i:${NC} Running (PID: $pid)"
                echo -e "    App WebSocket:   ws://localhost:$app_port"
                echo -e "    Admin WebSocket: ws://localhost:$admin_port"
            else
                echo -e "  ${RED}Node $i:${NC} Stopped"
            fi
        else
            echo -e "  ${YELLOW}Node $i:${NC} Not started"
        fi
    done

    echo ""
    echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
    echo ""
    echo "  Data directory: $DATA_DIR"
    echo "  Network seed:   $NETWORK_SEED"
    echo ""
    echo "  To connect from client:"
    echo "    const client = await AppWebsocket.connect('ws://localhost:$BASE_APP_PORT');"
    echo ""
    echo "  Press Ctrl+C to stop the network"
    echo ""
}

# Main
main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║    MYCELIX MAIL - MULTI-NODE TEST NETWORK                ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    log_info "Starting $NUM_NODES node test network..."

    # Change to project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR/.."

    check_prerequisites

    # Clean previous data
    rm -rf "$DATA_DIR"
    mkdir -p "$DATA_DIR"

    # Start nodes
    for i in $(seq 0 $((NUM_NODES - 1))); do
        config_path=$(create_conductor_config $i)
        start_node $i "$config_path"
    done

    # Wait for all nodes to be ready
    sleep 5

    # Install apps
    for i in $(seq 0 $((NUM_NODES - 1))); do
        install_app $i
    done

    print_status

    # Keep running
    log_info "Network is running. Press Ctrl+C to stop."
    while true; do
        sleep 10
    done
}

main "$@"
