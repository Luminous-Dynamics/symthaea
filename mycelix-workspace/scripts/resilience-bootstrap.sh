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
#   ./scripts/resilience-bootstrap.sh            # local deployment
#   ./scripts/resilience-bootstrap.sh --docker   # Docker deployment
#   ./scripts/resilience-bootstrap.sh --dry-run  # validate prerequisites only

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
# Dry-run: validate prerequisites + environment without starting anything
# --------------------------------------------------------------------------

deploy_dry_run() {
    info "DRY RUN — validating deployment readiness..."
    echo ""
    local errors=0

    # Check all prerequisites (both local and Docker)
    info "--- Local prerequisites ---"
    for cmd in just holochain hc lair-keystore pnpm; do
        if command -v "$cmd" &>/dev/null; then
            info "  Found $cmd: $(command -v "$cmd")"
        else
            warn "  Missing: $cmd"
            errors=$((errors + 1))
        fi
    done

    echo ""
    info "--- Docker prerequisites ---"
    for cmd in docker docker-compose; do
        if command -v "$cmd" &>/dev/null; then
            info "  Found $cmd: $(command -v "$cmd")"
        else
            warn "  Missing: $cmd (Docker deployment unavailable)"
        fi
    done

    echo ""
    info "--- Port availability ---"
    for port in 5173 8888 4444 9100; do
        if ss -tlnp 2>/dev/null | grep -q ":${port} " || \
           lsof -i ":${port}" &>/dev/null; then
            warn "  Port $port: IN USE (may conflict)"
            errors=$((errors + 1))
        else
            info "  Port $port: available"
        fi
    done

    echo ""
    info "--- Disk space ---"
    local avail_mb
    avail_mb=$(df -m "$WORKSPACE_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -n "$avail_mb" ]; then
        if [ "$avail_mb" -lt 2048 ]; then
            warn "  Available: ${avail_mb}MB — recommend at least 2GB"
            errors=$((errors + 1))
        else
            info "  Available: ${avail_mb}MB"
        fi
    fi

    echo ""
    info "--- Key files ---"
    local key_files=(
        "happs/mycelix-resilience-happ.yaml"
        "observatory/package.json"
        "mesh-bridge/Cargo.toml"
        "scripts/resilience-bootstrap.sh"
    )
    cd "$WORKSPACE_DIR"
    for f in "${key_files[@]}"; do
        if [ -f "$f" ]; then
            info "  $f: present"
        else
            warn "  $f: MISSING"
            errors=$((errors + 1))
        fi
    done

    echo ""
    if [ "$errors" -eq 0 ]; then
        info "Dry run PASSED — system is ready for deployment."
    else
        warn "Dry run found $errors issue(s). Fix before deploying."
    fi
    echo ""
    exit "$errors"
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
# Status check: are services running?
# --------------------------------------------------------------------------

check_status() {
    info "Checking Resilience Kit status..."
    echo ""

    # Holochain conductor
    info "--- Holochain Conductor ---"
    if ss -tlnp 2>/dev/null | grep -q ":8888 "; then
        info "  Port 8888: LISTENING"
    elif lsof -i :8888 &>/dev/null; then
        info "  Port 8888: LISTENING"
    else
        warn "  Port 8888: NOT listening (conductor may be down)"
    fi

    # Observatory
    echo ""
    info "--- Observatory ---"
    if curl -sf -o /dev/null --connect-timeout 2 http://localhost:5173/ 2>/dev/null; then
        info "  http://localhost:5173/: REACHABLE"
    elif curl -sf -o /dev/null --connect-timeout 2 http://localhost/ 2>/dev/null; then
        info "  http://localhost/: REACHABLE (Docker/nginx)"
    else
        warn "  Observatory: NOT reachable"
    fi

    # Mesh bridge health
    echo ""
    info "--- Mesh Bridge ---"
    local health
    health=$(curl -sf --connect-timeout 2 http://localhost:9100/health 2>/dev/null)
    if [ -n "$health" ]; then
        info "  Health endpoint: REACHABLE"
        # Parse key metrics with basic tools
        local sent recv peers cycles
        sent=$(echo "$health" | grep -o '"messages_sent":[0-9]*' | cut -d: -f2)
        recv=$(echo "$health" | grep -o '"messages_received":[0-9]*' | cut -d: -f2)
        peers=$(echo "$health" | grep -o '"peers_seen":[0-9]*' | cut -d: -f2)
        cycles=$(echo "$health" | grep -o '"poll_cycles":[0-9]*' | cut -d: -f2)
        info "  Messages sent:     ${sent:-0}"
        info "  Messages received: ${recv:-0}"
        info "  Peers seen:        ${peers:-0}"
        info "  Poll cycles:       ${cycles:-0}"
    else
        warn "  Mesh bridge: NOT reachable (port 9100)"
    fi

    # Docker containers (if Docker is available)
    if command -v docker &>/dev/null; then
        echo ""
        info "--- Docker Containers ---"
        local containers
        containers=$(docker ps --filter "name=mycelix" --format "  {{.Names}}: {{.Status}}" 2>/dev/null)
        if [ -n "$containers" ]; then
            echo "$containers"
        else
            info "  No mycelix containers running"
        fi
    fi

    # Dedup cache
    echo ""
    info "--- Dedup Cache ---"
    local cache_dir="${MESH_CACHE_DIR:-/tmp/mycelix-mesh-bridge}"
    if [ -d "$cache_dir" ]; then
        for f in "$cache_dir"/*.cache; do
            if [ -f "$f" ]; then
                local size
                size=$(wc -c < "$f" 2>/dev/null)
                info "  $(basename "$f"): ${size} bytes"
            fi
        done
    else
        info "  No dedup cache found (fresh start)"
    fi

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

case "$MODE" in
    --dry-run)
        deploy_dry_run
        ;;
    --status)
        check_status
        ;;
    --docker)
        check_prerequisites "$MODE"
        deploy_docker
        ;;
    *)
        check_prerequisites "$MODE"
        deploy_local
        ;;
esac
