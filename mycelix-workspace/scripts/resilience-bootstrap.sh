#!/usr/bin/env bash
# Mycelix Resilience Kit — Bootstrap Script
#
# Automated setup for community deployment:
#   1. Checks prerequisites (Docker, just, Nix)
#   2. Builds the 5-DNA resilience hApp
#   3. Starts conductor + Observatory
#   4. Prints access URLs
#
# Usage:
#   chmod +x scripts/resilience-bootstrap.sh
#   ./scripts/resilience-bootstrap.sh
#
# For Docker deployment:
#   ./scripts/resilience-bootstrap.sh --docker

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --------------------------------------------------------------------------
# Prerequisite checks
# --------------------------------------------------------------------------

check_command() {
    if ! command -v "$1" &>/dev/null; then
        error "$1 is required but not installed. $2"
    fi
    info "Found $1: $(command -v "$1")"
}

check_prerequisites() {
    info "Checking prerequisites..."
    echo ""

    if [[ "${1:-}" == "--docker" ]]; then
        check_command docker "Install: https://docs.docker.com/get-docker/"
        check_command docker-compose "Install: https://docs.docker.com/compose/install/"
    else
        check_command just "Install: cargo install just"
        check_command holochain "Install via nix develop"
        check_command hc "Install via nix develop"
        check_command lair-keystore "Install via nix develop"
        check_command pnpm "Install: npm install -g pnpm"
    fi

    echo ""
    info "All prerequisites met."
}

# --------------------------------------------------------------------------
# Local deployment (just + nix)
# --------------------------------------------------------------------------

deploy_local() {
    info "Starting LOCAL resilience deployment..."
    echo ""

    cd "$WORKSPACE_DIR"

    # Install Observatory dependencies if needed
    if [ -f "observatory/package.json" ] && [ ! -d "observatory/node_modules" ]; then
        info "Installing Observatory dependencies..."
        cd observatory && pnpm install && cd ..
    fi

    # Build and start
    info "Building and starting resilience kit..."
    just resilience-up

    echo ""
    info "Resilience Kit is running!"
    echo ""
    echo "  Dashboard:        http://localhost:5173/"
    echo "  TEND Exchange:    http://localhost:5173/tend"
    echo "  Food Tracking:    http://localhost:5173/food"
    echo "  Mutual Aid:       http://localhost:5173/mutual-aid"
    echo "  Emergency Comms:  http://localhost:5173/emergency"
    echo "  Value Anchor:     http://localhost:5173/value-anchor"
    echo ""
    echo "  Stop:  just stop"
    echo ""
}

# --------------------------------------------------------------------------
# Docker deployment
# --------------------------------------------------------------------------

deploy_docker() {
    info "Starting DOCKER resilience deployment..."
    echo ""

    cd "$WORKSPACE_DIR"

    # Build resilience hApp first
    info "Building resilience hApp..."
    just resilience-build

    # Generate LAIR passphrase if not set
    if [ -z "${LAIR_PASSPHRASE:-}" ]; then
        LAIR_PASSPHRASE="$(head -c 32 /dev/urandom | base64 | tr -d '=+/' | head -c 32)"
        warn "Generated LAIR_PASSPHRASE. Save this for future use:"
        echo "  export LAIR_PASSPHRASE=$LAIR_PASSPHRASE"
        echo ""
        export LAIR_PASSPHRASE
    fi

    # Create env file if needed
    if [ ! -f "deploy/.env" ]; then
        cat > deploy/.env <<EOF
LAIR_PASSPHRASE=$LAIR_PASSPHRASE
CONDUCTOR_ADMIN_PORT=4444
CONDUCTOR_APP_PORT=8888
OBSERVATORY_PORT=3000
VITE_CONDUCTOR_URL=ws://conductor:8888
VITE_FALLBACK_TO_SIMULATION=true
EOF
        info "Created deploy/.env"
    fi

    # Start containers
    cd deploy
    docker-compose -f docker-compose.prod.yml -f docker-compose.resilience.yml up -d --build

    echo ""
    info "Resilience Kit is running in Docker!"
    echo ""
    echo "  Observatory: http://localhost (via nginx)"
    echo ""
    echo "  Logs:  docker-compose -f docker-compose.prod.yml -f docker-compose.resilience.yml logs -f"
    echo "  Stop:  docker-compose -f docker-compose.prod.yml -f docker-compose.resilience.yml down"
    echo ""
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

echo ""
echo "============================================"
echo "  MYCELIX RESILIENCE KIT — BOOTSTRAP"
echo "============================================"
echo ""
echo "Community economic resilience infrastructure:"
echo "  - TEND mutual credit (1 TEND = 1 hour)"
echo "  - Food production tracking"
echo "  - Mutual aid timebank"
echo "  - Emergency communications"
echo ""

MODE="${1:-local}"

check_prerequisites "$MODE"

case "$MODE" in
    --docker)
        deploy_docker
        ;;
    *)
        deploy_local
        ;;
esac
