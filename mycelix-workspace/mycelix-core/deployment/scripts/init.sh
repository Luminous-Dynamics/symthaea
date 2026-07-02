#!/bin/bash
# =============================================================================
# MYCELIX VALIDATOR INITIALIZATION SCRIPT
# =============================================================================
#
# This script initializes the deployment environment and validates configuration.
#
# Usage:
#   ./scripts/init.sh [options]
#
# Options:
#   --check       Only validate configuration, don't create directories
#   --force       Overwrite existing .env file
#   --help        Show this help message
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

# Logging functions
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

# Parse arguments
CHECK_ONLY=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            head -30 "$0" | tail -20
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  Mycelix Validator Initialization"
echo "=============================================="
echo ""

# =============================================================================
# PREREQUISITE CHECKS
# =============================================================================

log_info "Checking prerequisites..."

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+' | head -1)
    log_success "Docker installed: v$DOCKER_VERSION"
else
    log_error "Docker is not installed"
    log_info "Install Docker: curl -fsSL https://get.docker.com | sh"
    exit 1
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version | grep -oP '\d+\.\d+' | head -1)
    log_success "Docker Compose installed: v$COMPOSE_VERSION"
else
    log_error "Docker Compose is not installed"
    log_info "Install: sudo apt install docker-compose-plugin"
    exit 1
fi

# Check Docker daemon is running
if docker info &> /dev/null; then
    log_success "Docker daemon is running"
else
    log_error "Docker daemon is not running"
    log_info "Start Docker: sudo systemctl start docker"
    exit 1
fi

# Check current user is in docker group
if groups | grep -q docker; then
    log_success "User is in docker group"
else
    log_warning "User is not in docker group (may need sudo for docker commands)"
fi

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

if [ "$CHECK_ONLY" = false ]; then
    log_info "Creating data directories..."

    DATA_DIRS=(
        "data/validator"
        "data/keystore"
        "data/logs"
        "data/cache"
        "data/prometheus"
        "data/grafana"
    )

    for dir in "${DATA_DIRS[@]}"; do
        mkdir -p "$DEPLOY_DIR/$dir"
        log_success "Created: $dir"
    done

    # Set permissions
    chmod -R 755 "$DEPLOY_DIR/data"
    log_success "Set directory permissions"
fi

# =============================================================================
# CONFIGURATION SETUP
# =============================================================================

log_info "Checking configuration..."

ENV_FILE="$DEPLOY_DIR/.env"
ENV_EXAMPLE="$DEPLOY_DIR/.env.example"

if [ -f "$ENV_FILE" ]; then
    if [ "$FORCE" = true ]; then
        log_warning "Overwriting existing .env file (--force)"
        cp "$ENV_EXAMPLE" "$ENV_FILE"
    else
        log_success ".env file exists"
    fi
else
    if [ "$CHECK_ONLY" = false ]; then
        log_info "Creating .env from example..."
        cp "$ENV_EXAMPLE" "$ENV_FILE"

        # Generate a unique node ID based on hostname
        HOSTNAME_LOWER=$(hostname | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
        RANDOM_SUFFIX=$(head /dev/urandom | tr -dc 'a-z0-9' | head -c 4)
        NODE_ID="validator-${HOSTNAME_LOWER}-${RANDOM_SUFFIX}"

        # Update .env with generated values
        sed -i "s/VALIDATOR_NODE_ID=validator-001/VALIDATOR_NODE_ID=$NODE_ID/" "$ENV_FILE"

        # Generate a random Grafana password
        GRAFANA_PASS=$(head /dev/urandom | tr -dc 'A-Za-z0-9' | head -c 16)
        sed -i "s/GRAFANA_ADMIN_PASSWORD=mycelix-admin/GRAFANA_ADMIN_PASSWORD=$GRAFANA_PASS/" "$ENV_FILE"

        log_success "Created .env with generated values"
        log_info "Node ID: $NODE_ID"
        log_info "Grafana password: $GRAFANA_PASS"
        echo ""
        log_warning "Please review and edit .env before starting!"
    else
        log_error ".env file does not exist"
    fi
fi

# =============================================================================
# VALIDATE CONFIGURATION
# =============================================================================

log_info "Validating docker-compose configuration..."

cd "$DEPLOY_DIR"
if docker compose config --quiet 2>/dev/null; then
    log_success "docker-compose.yml is valid"
else
    log_error "docker-compose.yml validation failed"
    docker compose config 2>&1 | head -10
    exit 1
fi

# Check required environment variables
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"

    if [ -z "$VALIDATOR_NODE_ID" ] || [ "$VALIDATOR_NODE_ID" = "validator-001" ]; then
        log_warning "VALIDATOR_NODE_ID should be customized"
    else
        log_success "VALIDATOR_NODE_ID: $VALIDATOR_NODE_ID"
    fi

    if [ -z "$VALIDATOR_OPERATOR" ] || [ "$VALIDATOR_OPERATOR" = "community" ]; then
        log_warning "VALIDATOR_OPERATOR should be customized"
    else
        log_success "VALIDATOR_OPERATOR: $VALIDATOR_OPERATOR"
    fi

    if [ "$GRAFANA_ADMIN_PASSWORD" = "mycelix-admin" ]; then
        log_warning "GRAFANA_ADMIN_PASSWORD is using default value!"
    fi
fi

# =============================================================================
# NETWORK CONNECTIVITY
# =============================================================================

log_info "Checking network connectivity..."

# Check bootstrap server
if curl -sf --max-time 5 "https://bootstrap.mycelix.net" > /dev/null 2>&1 || \
   curl -sf --max-time 5 "https://bootstrap-staging.holo.host" > /dev/null 2>&1; then
    log_success "Bootstrap servers reachable"
else
    log_warning "Could not reach bootstrap servers (may be firewall issue)"
fi

# Check signal server (WebSocket, so just check DNS)
if host signal.holo.host > /dev/null 2>&1; then
    log_success "Signal server DNS resolves"
else
    log_warning "Could not resolve signal.holo.host"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "  Initialization Complete"
echo "=============================================="
echo ""

if [ "$CHECK_ONLY" = true ]; then
    log_info "Check-only mode - no changes made"
else
    log_success "Environment is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Review configuration: nano .env"
    echo "  2. Start validator: docker compose up -d"
    echo "  3. View logs: docker compose logs -f validator"
    echo "  4. Check health: curl http://localhost:8080/health"
    echo ""
    echo "Dashboard will be available at: http://localhost:3000"
    if [ -n "$GRAFANA_PASS" ]; then
        echo "  Username: admin"
        echo "  Password: $GRAFANA_PASS"
    fi
fi
