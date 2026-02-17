#!/usr/bin/env bash
#
# Mycelix Local Ethereum Testnet Setup
#
# Sets up Anvil local testnet and deploys Mycelix contracts.
# Requires: Foundry (anvil, forge, cast)
#
# Usage:
#   ./scripts/local-ethereum-testnet.sh start    # Start Anvil and deploy contracts
#   ./scripts/local-ethereum-testnet.sh stop     # Stop Anvil
#   ./scripts/local-ethereum-testnet.sh status   # Check status
#   ./scripts/local-ethereum-testnet.sh test     # Run integration tests
#
# Environment:
#   ANVIL_PORT      - Anvil RPC port (default: 8545)
#   DEPLOY_KEY      - Private key for deployments (default: Anvil account 0)
#   CONTRACTS_DIR   - Path to contracts (default: ./contracts)
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
CONTRACTS_DIR="${CONTRACTS_DIR:-$PROJECT_ROOT/contracts}"
ANVIL_PORT="${ANVIL_PORT:-8545}"
ANVIL_PID_FILE="/tmp/mycelix-anvil.pid"
DEPLOYED_ADDRESSES_FILE="/tmp/mycelix-deployed-addresses.json"

# Anvil default private key (account 0) - ONLY FOR LOCAL TESTING
DEFAULT_DEPLOY_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
DEPLOY_KEY="${DEPLOY_KEY:-$DEFAULT_DEPLOY_KEY}"

check_foundry() {
    if ! command -v anvil &> /dev/null; then
        log_error "Anvil not found. Install Foundry or run 'nix develop' first."
        echo ""
        echo "Install Foundry:"
        echo "  curl -L https://foundry.paradigm.xyz | bash"
        echo "  foundryup"
        echo ""
        echo "Or use nix:"
        echo "  nix develop"
        exit 1
    fi

    if ! command -v forge &> /dev/null; then
        log_error "Forge not found. Install Foundry."
        exit 1
    fi

    log_success "Foundry tools available"
}

start_anvil() {
    log_step "Starting Anvil local testnet on port $ANVIL_PORT..."

    if [ -f "$ANVIL_PID_FILE" ]; then
        local old_pid
        old_pid=$(cat "$ANVIL_PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log_warn "Anvil already running (PID: $old_pid)"
            return 0
        fi
        rm -f "$ANVIL_PID_FILE"
    fi

    # Start Anvil with deterministic accounts for reproducible tests
    anvil \
        --port "$ANVIL_PORT" \
        --accounts 10 \
        --balance 10000 \
        --block-time 1 \
        --chain-id 31337 \
        --gas-limit 30000000 \
        --silent &

    local anvil_pid=$!
    echo "$anvil_pid" > "$ANVIL_PID_FILE"

    # Wait for Anvil to be ready
    local retries=30
    while ! curl -s "http://localhost:$ANVIL_PORT" -X POST -H "Content-Type: application/json" \
           --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' &>/dev/null; do
        ((retries--))
        if [ $retries -le 0 ]; then
            log_error "Anvil failed to start"
            kill "$anvil_pid" 2>/dev/null || true
            exit 1
        fi
        sleep 0.5
    done

    log_success "Anvil running (PID: $anvil_pid)"
    echo ""
    echo "RPC URL: http://localhost:$ANVIL_PORT"
    echo "Chain ID: 31337"
    echo ""
    echo "Test Accounts:"
    echo "  Account 0: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
    echo "  Account 1: 0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    echo ""
}

stop_anvil() {
    log_step "Stopping Anvil..."

    if [ -f "$ANVIL_PID_FILE" ]; then
        local pid
        pid=$(cat "$ANVIL_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            log_success "Anvil stopped (PID: $pid)"
        fi
        rm -f "$ANVIL_PID_FILE"
    else
        log_warn "Anvil PID file not found"
        # Try to find and kill any anvil process on our port
        pkill -f "anvil.*--port $ANVIL_PORT" 2>/dev/null || true
    fi

    rm -f "$DEPLOYED_ADDRESSES_FILE"
}

check_status() {
    log_step "Checking status..."

    if [ -f "$ANVIL_PID_FILE" ]; then
        local pid
        pid=$(cat "$ANVIL_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_success "Anvil running (PID: $pid)"

            # Check block number
            local block_num
            block_num=$(curl -s "http://localhost:$ANVIL_PORT" -X POST \
                -H "Content-Type: application/json" \
                --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
                | grep -o '"result":"[^"]*"' | cut -d'"' -f4)
            echo "Current block: $block_num"

            if [ -f "$DEPLOYED_ADDRESSES_FILE" ]; then
                echo ""
                echo "Deployed contracts:"
                cat "$DEPLOYED_ADDRESSES_FILE"
            fi
            return 0
        fi
    fi

    log_warn "Anvil not running"
    return 1
}

deploy_contracts() {
    log_step "Deploying Mycelix contracts..."

    if [ ! -d "$CONTRACTS_DIR" ]; then
        log_error "Contracts directory not found: $CONTRACTS_DIR"
        exit 1
    fi

    cd "$CONTRACTS_DIR"

    # Build contracts
    log_info "Building contracts..."
    forge build

    # Deploy contracts in order (respecting dependencies)
    local contracts=(
        "src/MycelixRegistry.sol:MycelixRegistry"
        "src/ReputationAnchor.sol:ReputationAnchor"
        "src/ContributionRegistry.sol:ContributionRegistry"
        "src/ModelRegistry.sol:ModelRegistry"
        "src/PaymentRouter.sol:PaymentRouter"
    )

    local deployed=()

    echo "{" > "$DEPLOYED_ADDRESSES_FILE"

    for contract_spec in "${contracts[@]}"; do
        local file="${contract_spec%%:*}"
        local name="${contract_spec##*:}"

        log_info "Deploying $name..."

        local output
        output=$(forge create "$file:$name" \
            --rpc-url "http://localhost:$ANVIL_PORT" \
            --private-key "$DEPLOY_KEY" \
            2>&1)

        local address
        address=$(echo "$output" | grep "Deployed to:" | awk '{print $3}')

        if [ -z "$address" ]; then
            log_error "Failed to deploy $name"
            echo "$output"
            exit 1
        fi

        log_success "$name deployed at $address"
        deployed+=("\"$name\": \"$address\"")
    done

    # Write deployed addresses JSON
    echo "  $(IFS=,; echo "${deployed[*]}" | sed 's/,/,\n  /g')" >> "$DEPLOYED_ADDRESSES_FILE"
    echo "}" >> "$DEPLOYED_ADDRESSES_FILE"

    log_success "All contracts deployed"
    echo ""
    echo "Deployed addresses saved to: $DEPLOYED_ADDRESSES_FILE"
    cat "$DEPLOYED_ADDRESSES_FILE"
}

run_integration_tests() {
    log_step "Running integration tests..."

    # Check Anvil is running
    if ! check_status &>/dev/null; then
        log_error "Anvil not running. Start it first with: $0 start"
        exit 1
    fi

    # Run contract tests with Foundry
    cd "$CONTRACTS_DIR"
    log_info "Running Foundry tests..."
    forge test -vvv --fork-url "http://localhost:$ANVIL_PORT"

    # Run Rust integration tests if they exist
    local eth_test_dir="$PROJECT_ROOT/libs/fl-aggregator/tests"
    if [ -d "$eth_test_dir" ]; then
        log_info "Running Rust integration tests..."
        cd "$PROJECT_ROOT/libs/fl-aggregator"

        # Export RPC URL for tests
        export ANVIL_RPC_URL="http://localhost:$ANVIL_PORT"
        export DEPLOYED_ADDRESSES="$DEPLOYED_ADDRESSES_FILE"

        cargo test --features ethereum_integration -- --test-threads=1
    fi

    log_success "Integration tests complete"
}

# Main
case "${1:-}" in
    start)
        check_foundry
        start_anvil
        deploy_contracts
        ;;
    stop)
        stop_anvil
        ;;
    status)
        check_status
        ;;
    test)
        check_foundry
        run_integration_tests
        ;;
    deploy)
        check_foundry
        deploy_contracts
        ;;
    *)
        echo "Mycelix Local Ethereum Testnet"
        echo ""
        echo "Usage: $0 {start|stop|status|test|deploy}"
        echo ""
        echo "Commands:"
        echo "  start   - Start Anvil and deploy contracts"
        echo "  stop    - Stop Anvil"
        echo "  status  - Check status and deployed addresses"
        echo "  test    - Run integration tests"
        echo "  deploy  - Deploy contracts (Anvil must be running)"
        exit 1
        ;;
esac
