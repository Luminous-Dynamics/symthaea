#!/run/current-system/sw/bin/bash
# =============================================================================
# Mycelix-Core Launch Script
# =============================================================================
# One command to rule them all - spins up the complete Mycelix FL system
#
# Usage:
#   ./launch.sh              # Start the full stack
#   ./launch.sh --build      # Rebuild and start
#   ./launch.sh --holochain  # Include Holochain for decentralization
#   ./launch.sh --tools      # Include CLI tools container
#   ./launch.sh --down       # Stop all services
#   ./launch.sh --status     # Show status
#   ./launch.sh --logs       # Follow logs
#   ./launch.sh --clean      # Stop and remove all data
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}/docker"
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.ultimate.yml"

# Banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║   ███╗   ███╗██╗   ██╗ ██████╗███████╗██╗     ██╗██╗  ██╗       ║"
    echo "║   ████╗ ████║╚██╗ ██╔╝██╔════╝██╔════╝██║     ██║╚██╗██╔╝       ║"
    echo "║   ██╔████╔██║ ╚████╔╝ ██║     █████╗  ██║     ██║ ╚███╔╝        ║"
    echo "║   ██║╚██╔╝██║  ╚██╔╝  ██║     ██╔══╝  ██║     ██║ ██╔██╗        ║"
    echo "║   ██║ ╚═╝ ██║   ██║   ╚██████╗███████╗███████╗██║██╔╝ ██╗       ║"
    echo "║   ╚═╝     ╚═╝   ╚═╝    ╚═════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝       ║"
    echo "║                                                                  ║"
    echo "║            Federated Learning with Byzantine Tolerance           ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose (v2)
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose v2 is not available. Please update Docker."
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    log_success "All prerequisites met"
}

# Create .env file if it doesn't exist
setup_env() {
    if [ ! -f "${DOCKER_DIR}/.env" ]; then
        if [ -f "${DOCKER_DIR}/.env.example" ]; then
            log_info "Creating .env file from template..."
            cp "${DOCKER_DIR}/.env.example" "${DOCKER_DIR}/.env"
            log_warn "Please review ${DOCKER_DIR}/.env and update passwords for production use"
        fi
    fi
}

# Start services
start_services() {
    local BUILD_FLAG=""
    local PROFILES=""

    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --build)
                BUILD_FLAG="--build"
                ;;
            --holochain)
                PROFILES="${PROFILES} --profile holochain"
                ;;
            --tools)
                PROFILES="${PROFILES} --profile tools"
                ;;
        esac
    done

    log_info "Starting Mycelix-Core services..."

    cd "${DOCKER_DIR}"

    # Start services
    docker compose -f docker-compose.ultimate.yml ${PROFILES} up ${BUILD_FLAG} -d

    log_success "Services started!"
    echo ""
    show_access_info
}

# Stop services
stop_services() {
    log_info "Stopping Mycelix-Core services..."
    cd "${DOCKER_DIR}"
    docker compose -f docker-compose.ultimate.yml --profile holochain --profile tools down
    log_success "Services stopped"
}

# Clean up everything
clean_all() {
    log_warn "This will remove all containers and data volumes!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning up..."
        cd "${DOCKER_DIR}"
        docker compose -f docker-compose.ultimate.yml --profile holochain --profile tools down -v --remove-orphans
        log_success "Cleanup complete"
    else
        log_info "Cleanup cancelled"
    fi
}

# Show status
show_status() {
    log_info "Service Status:"
    echo ""
    cd "${DOCKER_DIR}"
    docker compose -f docker-compose.ultimate.yml ps

    echo ""
    log_info "Health Checks:"

    # Check coordinator
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "  Coordinator: ${GREEN}Healthy${NC}"
    else
        echo -e "  Coordinator: ${RED}Unhealthy${NC}"
    fi

    # Check Grafana
    if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
        echo -e "  Grafana:     ${GREEN}Healthy${NC}"
    else
        echo -e "  Grafana:     ${RED}Unhealthy${NC}"
    fi

    # Check Prometheus
    if curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
        echo -e "  Prometheus:  ${GREEN}Healthy${NC}"
    else
        echo -e "  Prometheus:  ${RED}Unhealthy${NC}"
    fi
}

# Follow logs
follow_logs() {
    local SERVICE=$1
    cd "${DOCKER_DIR}"
    if [ -n "$SERVICE" ]; then
        docker compose -f docker-compose.ultimate.yml logs -f "$SERVICE"
    else
        docker compose -f docker-compose.ultimate.yml logs -f
    fi
}

# Show access information
show_access_info() {
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                    Mycelix-Core is Starting!                       ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Service Endpoints:${NC}"
    echo ""
    echo -e "    FL Coordinator:    ${YELLOW}http://localhost:8080${NC}"
    echo -e "      - Health:        http://localhost:8080/health"
    echo -e "      - Status:        http://localhost:8080/status"
    echo -e "      - Metrics:       http://localhost:8080/metrics"
    echo ""
    echo -e "    Grafana Dashboard: ${YELLOW}http://localhost:3000${NC}"
    echo -e "      - User: admin    Password: admin"
    echo ""
    echo -e "    Prometheus:        ${YELLOW}http://localhost:9090${NC}"
    echo ""
    echo -e "    PostgreSQL:        ${YELLOW}localhost:5432${NC}"
    echo -e "      - Database: mycelix"
    echo ""
    echo -e "    Redis:             ${YELLOW}localhost:6379${NC}"
    echo ""
    echo -e "  ${CYAN}Quick Commands:${NC}"
    echo ""
    echo -e "    View logs:         ${YELLOW}./launch.sh --logs${NC}"
    echo -e "    Check status:      ${YELLOW}./launch.sh --status${NC}"
    echo -e "    Stop services:     ${YELLOW}./launch.sh --down${NC}"
    echo -e "    Use CLI:           ${YELLOW}docker exec -it mycelix-cli mycelix status${NC}"
    echo ""
    echo -e "  ${CYAN}Pre-configured Grafana Dashboards:${NC}"
    echo ""
    echo -e "    - Byzantine Detection"
    echo -e "    - Performance Metrics"
    echo -e "    - Node Health"
    echo -e "    - Model Convergence"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    log_info "Waiting for services to be ready..."
    echo ""

    # Wait for coordinator to be healthy
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
            log_success "Coordinator is ready!"
            break
        fi
        attempt=$((attempt + 1))
        echo -ne "\r  Waiting for coordinator... (${attempt}/${max_attempts})"
        sleep 2
    done
    echo ""

    if [ $attempt -eq $max_attempts ]; then
        log_warn "Coordinator may still be starting. Check logs with: ./launch.sh --logs coordinator"
    fi
}

# Show help
show_help() {
    echo "Mycelix-Core Launch Script"
    echo ""
    echo "Usage: ./launch.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (no args)      Start all services"
    echo "  --build        Rebuild containers before starting"
    echo "  --holochain    Include Holochain conductor"
    echo "  --tools        Include CLI tools container"
    echo "  --down         Stop all services"
    echo "  --status       Show service status"
    echo "  --logs [svc]   Follow logs (optionally for specific service)"
    echo "  --clean        Stop and remove all data"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh                    # Start basic stack"
    echo "  ./launch.sh --build            # Rebuild and start"
    echo "  ./launch.sh --holochain        # Start with Holochain"
    echo "  ./launch.sh --logs coordinator # Follow coordinator logs"
}

# Main
main() {
    print_banner

    case "${1:-}" in
        --help|-h)
            show_help
            ;;
        --down)
            stop_services
            ;;
        --status)
            show_status
            ;;
        --logs)
            follow_logs "$2"
            ;;
        --clean)
            clean_all
            ;;
        *)
            check_prerequisites
            setup_env
            start_services "$@"
            ;;
    esac
}

main "$@"
