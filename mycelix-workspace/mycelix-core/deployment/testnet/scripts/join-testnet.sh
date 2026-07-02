#!/usr/bin/env bash
# Mycelix Testnet - Node Join Script
# This script helps node operators join the public testnet
# Usage: ./join-testnet.sh [options]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration
NODE_TYPE="${NODE_TYPE:-participant}"
NODE_NAME="${NODE_NAME:-$(hostname)-$(date +%s | tail -c 6)}"
DATA_DIR="${DATA_DIR:-$HOME/.mycelix}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Bootstrap nodes (update with actual testnet bootstrap nodes)
BOOTSTRAP_NODES="${BOOTSTRAP_NODES:-testnet.mycelix.network:9000,testnet-bootstrap-1.mycelix.network:9000}"

# Ports
P2P_PORT="${P2P_PORT:-9000}"
API_PORT="${API_PORT:-9001}"
METRICS_PORT="${METRICS_PORT:-9090}"

# Docker image
DOCKER_IMAGE="${DOCKER_IMAGE:-mycelix/fl-node:testnet-latest}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Display banner
show_banner() {
    echo ""
    echo -e "${CYAN}"
    echo "  __  __                   _ _      "
    echo " |  \/  |_   _  ___ ___| (_)_  __"
    echo " | |\/| | | | |/ __/ _ \ | \ \/ /"
    echo " | |  | | |_| | (_|  __/ | |>  < "
    echo " |_|  |_|\__, |\___\___|_|_/_/\_\\"
    echo "         |___/                    "
    echo ""
    echo "      Public Testnet Join Script"
    echo -e "${NC}"
    echo ""
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --name)
                NODE_NAME="$2"
                shift 2
                ;;
            --type)
                NODE_TYPE="$2"
                shift 2
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --p2p-port)
                P2P_PORT="$2"
                shift 2
                ;;
            --api-port)
                API_PORT="$2"
                shift 2
                ;;
            --bootstrap)
                BOOTSTRAP_NODES="$2"
                shift 2
                ;;
            --docker)
                USE_DOCKER=true
                shift
                ;;
            --native)
                USE_DOCKER=false
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --name NAME         Node name (default: hostname-random)"
    echo "  --type TYPE         Node type: participant, validator, seed (default: participant)"
    echo "  --data-dir DIR      Data directory (default: ~/.mycelix)"
    echo "  --p2p-port PORT     P2P port (default: 9000)"
    echo "  --api-port PORT     API port (default: 9001)"
    echo "  --bootstrap NODES   Bootstrap nodes (comma-separated)"
    echo "  --docker            Use Docker (default)"
    echo "  --native            Use native binary"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --name my-node --type participant"
    echo "  $0 --docker --p2p-port 9100"
    echo "  $0 --native --data-dir /opt/mycelix"
}

# Check system requirements
check_requirements() {
    log_step "Checking system requirements..."

    # Check Docker
    if [ "${USE_DOCKER:-true}" = true ]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is not installed. Please install Docker first."
            echo "  Visit: https://docs.docker.com/get-docker/"
            exit 1
        fi

        # Check if Docker daemon is running
        if ! docker info &> /dev/null; then
            log_error "Docker daemon is not running. Please start Docker."
            exit 1
        fi
    fi

    # Check ports
    check_port "$P2P_PORT" "P2P"
    check_port "$API_PORT" "API"
    check_port "$METRICS_PORT" "Metrics"

    # Check disk space (need at least 20GB)
    local available_space
    available_space=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 20 ]; then
        log_warning "Low disk space: ${available_space}GB available. Recommended: 20GB+"
    fi

    # Check memory (need at least 4GB)
    local total_memory
    total_memory=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_memory" -lt 4 ]; then
        log_warning "Low memory: ${total_memory}GB available. Recommended: 4GB+"
    fi

    log_success "System requirements met!"
}

# Check if a port is available
check_port() {
    local port=$1
    local name=$2

    if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
        log_error "Port $port ($name) is already in use."
        log_info "Try using --$( echo "$name" | tr '[:upper:]' '[:lower:]')-port to specify a different port."
        exit 1
    fi
}

# Generate node identity
generate_identity() {
    log_step "Generating node identity..."

    mkdir -p "$DATA_DIR"

    # Generate node key if not exists
    if [ ! -f "$DATA_DIR/node.key" ]; then
        openssl rand -hex 32 > "$DATA_DIR/node.key"
        chmod 600 "$DATA_DIR/node.key"
        log_info "Generated new node key: $DATA_DIR/node.key"
    else
        log_info "Using existing node key: $DATA_DIR/node.key"
    fi

    # Get public key (derive from private key)
    NODE_PUBLIC_KEY=$(cat "$DATA_DIR/node.key" | openssl dgst -sha256 | cut -d' ' -f2)
    echo "$NODE_PUBLIC_KEY" > "$DATA_DIR/node.pub"

    log_success "Node identity ready!"
    log_info "Node ID: ${NODE_PUBLIC_KEY:0:16}..."
}

# Download genesis configuration
download_genesis() {
    log_step "Downloading genesis configuration..."

    local genesis_url="https://testnet.mycelix.network/genesis.json"
    local genesis_path="$DATA_DIR/genesis.json"

    if [ -f "$genesis_path" ]; then
        log_info "Genesis file already exists. Checking for updates..."
    fi

    # Try to download genesis
    if curl -sf -o "$genesis_path.tmp" "$genesis_url" 2>/dev/null; then
        mv "$genesis_path.tmp" "$genesis_path"
        log_success "Downloaded genesis configuration!"
    else
        log_warning "Could not download genesis. Using bundled configuration."
        # Copy bundled genesis if available
        if [ -f "$SCRIPT_DIR/../config/genesis.json" ]; then
            cp "$SCRIPT_DIR/../config/genesis.json" "$genesis_path"
        fi
    fi
}

# Create node configuration
create_config() {
    log_step "Creating node configuration..."

    cat > "$DATA_DIR/config.json" << EOF
{
  "node": {
    "id": "$NODE_NAME",
    "type": "$NODE_TYPE",
    "public_key_path": "$DATA_DIR/node.pub",
    "private_key_path": "$DATA_DIR/node.key"
  },
  "network": {
    "name": "mycelix-testnet",
    "bootstrap_nodes": [$(echo "$BOOTSTRAP_NODES" | sed 's/,/","/g' | sed 's/^/"/;s/$/"/')],
    "listen_address": "0.0.0.0",
    "p2p_port": $P2P_PORT,
    "api_port": $API_PORT,
    "metrics_port": $METRICS_PORT
  },
  "storage": {
    "data_dir": "$DATA_DIR/data",
    "db_path": "$DATA_DIR/db"
  },
  "federated_learning": {
    "enabled": true,
    "model_path": "$DATA_DIR/models",
    "aggregation_method": "fedavg",
    "min_participants": 3
  },
  "metrics": {
    "enabled": true,
    "port": $METRICS_PORT
  },
  "logging": {
    "level": "$LOG_LEVEL",
    "format": "json"
  }
}
EOF

    log_success "Configuration created: $DATA_DIR/config.json"
}

# Start node with Docker
start_docker() {
    log_step "Starting node with Docker..."

    # Pull latest image
    log_info "Pulling latest image..."
    docker pull "$DOCKER_IMAGE"

    # Create Docker network if not exists
    docker network create mycelix-testnet 2>/dev/null || true

    # Stop existing container if running
    docker stop "mycelix-$NODE_NAME" 2>/dev/null || true
    docker rm "mycelix-$NODE_NAME" 2>/dev/null || true

    # Start container
    docker run -d \
        --name "mycelix-$NODE_NAME" \
        --network mycelix-testnet \
        --restart unless-stopped \
        -p "$P2P_PORT:9000" \
        -p "$API_PORT:9001" \
        -p "$METRICS_PORT:9090" \
        -v "$DATA_DIR:/data" \
        -e MYCELIX_NETWORK=testnet \
        -e MYCELIX_NODE_ID="$NODE_NAME" \
        -e MYCELIX_NODE_TYPE="$NODE_TYPE" \
        -e MYCELIX_LOG_LEVEL="$LOG_LEVEL" \
        -e BOOTSTRAP_NODES="$BOOTSTRAP_NODES" \
        -e METRICS_ENABLED=true \
        "$DOCKER_IMAGE"

    log_success "Node started in Docker container: mycelix-$NODE_NAME"
}

# Start node with native binary
start_native() {
    log_step "Starting node with native binary..."

    local binary_path="${MYCELIX_BINARY:-mycelix-node}"

    # Check if binary exists
    if ! command -v "$binary_path" &> /dev/null; then
        log_error "Mycelix binary not found: $binary_path"
        log_info "Please install Mycelix or use --docker option."
        echo ""
        echo "Installation options:"
        echo "  1. Download from: https://github.com/mycelix/core/releases"
        echo "  2. Build from source: make build"
        echo "  3. Use Docker: $0 --docker"
        exit 1
    fi

    # Start the node
    nohup "$binary_path" \
        --config "$DATA_DIR/config.json" \
        --data-dir "$DATA_DIR" \
        > "$DATA_DIR/node.log" 2>&1 &

    local pid=$!
    echo "$pid" > "$DATA_DIR/node.pid"

    log_success "Node started with PID: $pid"
    log_info "Logs: $DATA_DIR/node.log"
}

# Verify node is running
verify_node() {
    log_step "Verifying node is running..."

    sleep 5

    # Check if node is responding
    local max_attempts=12
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            log_success "Node is healthy and responding!"
            break
        fi

        log_info "Waiting for node to start... (attempt $attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done

    if [ $attempt -gt $max_attempts ]; then
        log_warning "Node may not be fully started yet. Check logs for details."
    fi
}

# Display node information
show_node_info() {
    echo ""
    echo "========================================"
    echo "         Node Information"
    echo "========================================"
    echo ""
    echo "  Node Name:      $NODE_NAME"
    echo "  Node Type:      $NODE_TYPE"
    echo "  Data Directory: $DATA_DIR"
    echo ""
    echo "  Endpoints:"
    echo "    P2P:      localhost:$P2P_PORT"
    echo "    API:      http://localhost:$API_PORT"
    echo "    Metrics:  http://localhost:$METRICS_PORT/metrics"
    echo ""
    echo "  Useful Commands:"
    echo "    Check status:  curl http://localhost:$API_PORT/health"
    echo "    View logs:     docker logs -f mycelix-$NODE_NAME"
    echo "    Stop node:     docker stop mycelix-$NODE_NAME"
    echo ""
    echo "  Resources:"
    echo "    Dashboard:     https://grafana.testnet.mycelix.network"
    echo "    Faucet:        https://faucet.testnet.mycelix.network"
    echo "    Documentation: https://docs.mycelix.network"
    echo ""
    echo "========================================"
}

# Request tokens from faucet
request_faucet() {
    log_step "Requesting tokens from faucet..."

    local faucet_url="https://faucet.testnet.mycelix.network/api/request"
    local node_address
    node_address=$(cat "$DATA_DIR/node.pub")

    local response
    response=$(curl -sf -X POST \
        -H "Content-Type: application/json" \
        -d "{\"address\": \"$node_address\"}" \
        "$faucet_url" 2>/dev/null) || true

    if [ -n "$response" ]; then
        log_success "Faucet request submitted!"
        echo "Response: $response"
    else
        log_info "Faucet request failed. You can request tokens manually at:"
        echo "  https://faucet.testnet.mycelix.network"
    fi
}

# Main function
main() {
    show_banner

    # Parse arguments
    parse_args "$@"

    log_info "Joining Mycelix testnet as: $NODE_NAME ($NODE_TYPE)"
    echo ""

    # Run setup steps
    check_requirements
    generate_identity
    download_genesis
    create_config

    # Start node
    if [ "${USE_DOCKER:-true}" = true ]; then
        start_docker
    else
        start_native
    fi

    # Verify and display info
    verify_node
    show_node_info

    # Optionally request faucet tokens
    echo ""
    read -p "Would you like to request testnet tokens from the faucet? (y/n): " request_tokens
    if [[ "$request_tokens" =~ ^[Yy]$ ]]; then
        request_faucet
    fi

    echo ""
    log_success "Congratulations! Your node has joined the Mycelix testnet!"
    echo ""
}

# Run main function
main "$@"
