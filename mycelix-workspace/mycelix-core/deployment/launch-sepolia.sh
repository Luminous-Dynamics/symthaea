#!/usr/bin/env bash
# =============================================================================
# Mycelix Validator Node Launch Script - Sepolia Testnet
# =============================================================================
#
# Usage:
#   ./launch-sepolia.sh [OPTIONS]
#
# Options:
#   --validate       Validate configuration only (don't start)
#   --health-check   Run health checks on running services
#   --stop           Stop all services
#   --restart        Restart all services
#   --logs           Follow validator logs
#   --full           Start with full monitoring stack
#   -h, --help       Show this help message
#
# =============================================================================

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Sepolia Contract Addresses
EXPECTED_REGISTRY="0x69411e4c9D99814952EBA9a35028c12c2bbD03cd"
EXPECTED_REPUTATION="0x042ee96Ce1CaFFF1e84d6da0585CD440d73DAF99"
EXPECTED_PAYMENT="0x3BD80003d27f9786bd4ffeaA5ffA7CC83365da1b"
SEPOLIA_CHAIN_ID="11155111"
DEFAULT_RPC="https://sepolia.drpc.org"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "=============================================="
    echo "  Mycelix Validator - Sepolia Testnet Launch"
    echo "=============================================="
    echo ""
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Mycelix Validator Node Launch Script for Sepolia Testnet"
    echo ""
    echo "Options:"
    echo "  --validate       Validate configuration only (don't start)"
    echo "  --health-check   Run health checks on running services"
    echo "  --stop           Stop all services"
    echo "  --restart        Restart all services"
    echo "  --logs           Follow validator logs"
    echo "  --full           Start with full monitoring stack"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start with default configuration"
    echo "  $0 --validate         # Validate config before starting"
    echo "  $0 --full             # Start with alerting and node-exporter"
    echo "  $0 --health-check     # Check health of running services"
    echo ""
}

# =============================================================================
# Validation Functions
# =============================================================================

check_docker() {
    log_info "Checking Docker installation..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available. Please install Docker Compose v2."
        exit 1
    fi

    log_success "Docker is installed and running"
}

check_env_file() {
    log_info "Checking environment file..."

    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.sepolia.example" ]]; then
            log_warning ".env file not found. Copying from .env.sepolia.example..."
            cp .env.sepolia.example .env
            log_warning "Please edit .env and set required values (ETH_PRIVATE_KEY, GRAFANA_ADMIN_PASSWORD)"
            exit 1
        else
            log_error ".env file not found and no example file available."
            exit 1
        fi
    fi

    log_success "Environment file exists"
}

validate_env_vars() {
    log_info "Validating environment variables..."

    # Source .env file
    set -a
    source .env
    set +a

    local errors=0

    # Check required variables
    if [[ -z "${VALIDATOR_NODE_ID:-}" ]]; then
        log_error "VALIDATOR_NODE_ID is not set"
        ((errors++))
    fi

    if [[ -z "${ETH_PRIVATE_KEY:-}" ]] || [[ "${ETH_PRIVATE_KEY}" == "0x_YOUR_PRIVATE_KEY_HERE" ]]; then
        log_error "ETH_PRIVATE_KEY is not set or is still the placeholder value"
        ((errors++))
    fi

    if [[ -z "${GRAFANA_ADMIN_PASSWORD:-}" ]] || [[ "${GRAFANA_ADMIN_PASSWORD}" == "CHANGE_THIS_PASSWORD" ]]; then
        log_error "GRAFANA_ADMIN_PASSWORD is not set or is still the placeholder value"
        ((errors++))
    fi

    # Validate contract addresses match expected
    local registry="${MYCELIX_REGISTRY_ADDRESS:-$EXPECTED_REGISTRY}"
    local reputation="${REPUTATION_ANCHOR_ADDRESS:-$EXPECTED_REPUTATION}"
    local payment="${PAYMENT_ROUTER_ADDRESS:-$EXPECTED_PAYMENT}"

    if [[ "$registry" != "$EXPECTED_REGISTRY" ]]; then
        log_warning "MYCELIX_REGISTRY_ADDRESS differs from expected Sepolia address"
    fi

    if [[ "$reputation" != "$EXPECTED_REPUTATION" ]]; then
        log_warning "REPUTATION_ANCHOR_ADDRESS differs from expected Sepolia address"
    fi

    if [[ "$payment" != "$EXPECTED_PAYMENT" ]]; then
        log_warning "PAYMENT_ROUTER_ADDRESS differs from expected Sepolia address"
    fi

    # Validate chain ID
    local chain_id="${ETH_CHAIN_ID:-$SEPOLIA_CHAIN_ID}"
    if [[ "$chain_id" != "$SEPOLIA_CHAIN_ID" ]]; then
        log_warning "ETH_CHAIN_ID ($chain_id) is not Sepolia ($SEPOLIA_CHAIN_ID)"
    fi

    if [[ $errors -gt 0 ]]; then
        log_error "Found $errors configuration error(s). Please fix before launching."
        exit 1
    fi

    log_success "Environment variables validated"
}

check_network_connectivity() {
    log_info "Checking network connectivity..."

    local rpc_url="${ETH_RPC_URL:-$DEFAULT_RPC}"

    # Test RPC endpoint
    local response
    response=$(curl -s -X POST "$rpc_url" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
        --max-time 10 2>/dev/null || echo "FAILED")

    if [[ "$response" == "FAILED" ]]; then
        log_error "Cannot connect to Sepolia RPC: $rpc_url"
        exit 1
    fi

    # Verify chain ID
    local chain_id_hex
    chain_id_hex=$(echo "$response" | grep -o '"result":"[^"]*"' | cut -d'"' -f4 || echo "")

    if [[ -z "$chain_id_hex" ]]; then
        log_error "Invalid response from RPC endpoint"
        exit 1
    fi

    # Convert hex to decimal
    local chain_id_dec
    chain_id_dec=$((chain_id_hex))

    if [[ "$chain_id_dec" != "$SEPOLIA_CHAIN_ID" ]]; then
        log_error "RPC endpoint returned wrong chain ID: $chain_id_dec (expected $SEPOLIA_CHAIN_ID)"
        exit 1
    fi

    log_success "Connected to Sepolia (Chain ID: $chain_id_dec)"
}

verify_contracts() {
    log_info "Verifying contract deployments on Sepolia..."

    local rpc_url="${ETH_RPC_URL:-$DEFAULT_RPC}"
    local all_verified=true

    for addr in "$EXPECTED_REGISTRY" "$EXPECTED_REPUTATION" "$EXPECTED_PAYMENT"; do
        local response
        response=$(curl -s -X POST "$rpc_url" \
            -H "Content-Type: application/json" \
            -d "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getCode\",\"params\":[\"$addr\",\"latest\"],\"id\":1}" \
            --max-time 10 2>/dev/null || echo "FAILED")

        if [[ "$response" == "FAILED" ]]; then
            log_warning "Could not verify contract at $addr"
            all_verified=false
            continue
        fi

        local code
        code=$(echo "$response" | grep -o '"result":"[^"]*"' | cut -d'"' -f4 || echo "0x")

        if [[ "$code" == "0x" ]] || [[ -z "$code" ]]; then
            log_warning "No contract found at $addr"
            all_verified=false
        else
            log_success "Contract verified: ${addr:0:10}...${addr: -8}"
        fi
    done

    if [[ "$all_verified" == true ]]; then
        log_success "All contracts verified on Sepolia"
    else
        log_warning "Some contracts could not be verified. This may be a network issue."
    fi
}

check_data_directories() {
    log_info "Checking data directories..."

    local data_path="${DATA_PATH:-./data}"
    local dirs=("validator" "keystore" "logs" "cache" "prometheus" "grafana")

    for dir in "${dirs[@]}"; do
        local full_path="$data_path/$dir"
        if [[ ! -d "$full_path" ]]; then
            log_info "Creating directory: $full_path"
            mkdir -p "$full_path"
        fi
    done

    # Check write permissions
    if ! touch "$data_path/.write_test" 2>/dev/null; then
        log_error "Cannot write to data directory: $data_path"
        exit 1
    fi
    rm -f "$data_path/.write_test"

    log_success "Data directories ready"
}

# =============================================================================
# Service Management Functions
# =============================================================================

start_services() {
    local full_stack="${1:-false}"

    log_info "Starting Mycelix validator services..."

    if [[ "$full_stack" == "true" ]]; then
        log_info "Starting with full monitoring stack..."
        docker compose --profile monitoring --profile alerting up -d
    else
        docker compose up -d
    fi

    log_success "Services started"

    # Wait for services to be ready
    log_info "Waiting for services to initialize (30 seconds)..."
    sleep 30

    run_health_checks
}

stop_services() {
    log_info "Stopping all services..."
    docker compose down
    log_success "Services stopped"
}

restart_services() {
    log_info "Restarting all services..."
    docker compose down
    docker compose up -d
    log_success "Services restarted"
}

follow_logs() {
    log_info "Following validator logs (Ctrl+C to exit)..."
    docker compose logs -f validator
}

# =============================================================================
# Health Check Functions
# =============================================================================

run_health_checks() {
    echo ""
    log_info "Running health checks..."
    echo ""

    local all_healthy=true

    # Check validator health
    log_info "Checking validator health..."
    local validator_health
    validator_health=$(curl -s http://localhost:8080/health --max-time 5 2>/dev/null || echo "FAILED")

    if [[ "$validator_health" == "FAILED" ]]; then
        log_warning "Validator health endpoint not responding"
        all_healthy=false
    else
        log_success "Validator health endpoint responding"
    fi

    # Check Prometheus
    log_info "Checking Prometheus..."
    local prometheus_health
    prometheus_health=$(curl -s http://localhost:9091/-/healthy --max-time 5 2>/dev/null || echo "FAILED")

    if [[ "$prometheus_health" == "FAILED" ]]; then
        log_warning "Prometheus not responding"
        all_healthy=false
    else
        log_success "Prometheus is healthy"
    fi

    # Check Grafana
    log_info "Checking Grafana..."
    local grafana_health
    grafana_health=$(curl -s http://localhost:3000/api/health --max-time 5 2>/dev/null || echo "FAILED")

    if [[ "$grafana_health" == "FAILED" ]]; then
        log_warning "Grafana not responding"
        all_healthy=false
    else
        log_success "Grafana is healthy"
    fi

    # Check container status
    echo ""
    log_info "Container status:"
    docker compose ps

    echo ""
    if [[ "$all_healthy" == true ]]; then
        log_success "All health checks passed!"
        echo ""
        echo "Access points:"
        echo "  - Health:     http://localhost:8080/health"
        echo "  - Metrics:    http://localhost:9090/metrics"
        echo "  - Prometheus: http://localhost:9091"
        echo "  - Grafana:    http://localhost:3000"
        echo ""
        echo "Contract Explorer Links (Sepolia):"
        echo "  - Registry:   https://sepolia.etherscan.io/address/$EXPECTED_REGISTRY"
        echo "  - Reputation: https://sepolia.etherscan.io/address/$EXPECTED_REPUTATION"
        echo "  - Payment:    https://sepolia.etherscan.io/address/$EXPECTED_PAYMENT"
    else
        log_warning "Some health checks failed. Check logs with: docker compose logs"
    fi
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    local validate_only=false
    local health_check_only=false
    local stop_only=false
    local restart_only=false
    local logs_only=false
    local full_stack=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --validate)
                validate_only=true
                shift
                ;;
            --health-check)
                health_check_only=true
                shift
                ;;
            --stop)
                stop_only=true
                shift
                ;;
            --restart)
                restart_only=true
                shift
                ;;
            --logs)
                logs_only=true
                shift
                ;;
            --full)
                full_stack=true
                shift
                ;;
            -h|--help)
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

    print_header

    # Handle special modes
    if [[ "$stop_only" == true ]]; then
        stop_services
        exit 0
    fi

    if [[ "$restart_only" == true ]]; then
        restart_services
        exit 0
    fi

    if [[ "$logs_only" == true ]]; then
        follow_logs
        exit 0
    fi

    if [[ "$health_check_only" == true ]]; then
        run_health_checks
        exit 0
    fi

    # Run validation
    check_docker
    check_env_file
    validate_env_vars
    check_network_connectivity
    verify_contracts
    check_data_directories

    echo ""
    log_success "All validations passed!"

    if [[ "$validate_only" == true ]]; then
        echo ""
        log_info "Validation complete. Run without --validate to start services."
        exit 0
    fi

    # Start services
    echo ""
    start_services "$full_stack"
}

main "$@"
