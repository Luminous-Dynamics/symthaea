#!/bin/bash
# Conductor Integration Tests Runner
#
# This script manages the Holochain conductor for integration testing.
#
# Usage:
#   ./scripts/conductor-tests.sh         # Start conductor and run tests
#   ./scripts/conductor-tests.sh start   # Only start conductor
#   ./scripts/conductor-tests.sh stop    # Only stop conductor
#   ./scripts/conductor-tests.sh logs    # Show conductor logs
#   ./scripts/conductor-tests.sh clean   # Remove all test data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_DIR/docker"

# Configuration
CONDUCTOR_NAME="mycelix-conductor-test"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        echo "Please start Docker and try again"
        exit 1
    fi
}

# Start the conductor
start_conductor() {
    check_docker

    log_info "Building conductor image..."
    docker compose -f "$COMPOSE_FILE" build conductor

    log_info "Starting conductor..."
    docker compose -f "$COMPOSE_FILE" up -d conductor

    log_info "Waiting for conductor to be ready..."
    local max_retries=60
    local count=0

    while [ $count -lt $max_retries ]; do
        if curl -s http://localhost:8889/ > /dev/null 2>&1; then
            log_success "Conductor is ready!"
            echo ""
            echo "  Admin interface: ws://localhost:8889"
            echo "  App interface:   ws://localhost:8888"
            echo ""
            return 0
        fi
        count=$((count + 1))
        echo -n "."
        sleep 1
    done

    echo ""
    log_error "Conductor failed to start within $max_retries seconds"
    show_logs
    exit 1
}

# Stop the conductor
stop_conductor() {
    check_docker

    log_info "Stopping conductor..."
    docker compose -f "$COMPOSE_FILE" down

    log_success "Conductor stopped"
}

# Show conductor logs
show_logs() {
    check_docker

    log_info "Conductor logs:"
    docker compose -f "$COMPOSE_FILE" logs --tail=100 conductor
}

# Run integration tests
run_tests() {
    export CONDUCTOR_AVAILABLE=true
    export HOLOCHAIN_APP_URL=ws://localhost:8888
    export HOLOCHAIN_ADMIN_URL=ws://localhost:8889
    export MYCELIX_APP_ID=mycelix_ecosystem

    log_info "Running conductor integration tests..."
    cd "$PROJECT_DIR"

    # Run tests
    if npm run test:conductor; then
        log_success "All conductor tests passed!"
    else
        log_error "Some conductor tests failed"
        exit 1
    fi
}

# Clean up all test data
clean_all() {
    check_docker

    log_info "Stopping and removing all test containers and data..."
    docker compose -f "$COMPOSE_FILE" down -v --remove-orphans

    log_success "Cleanup complete"
}

# Check conductor status
check_status() {
    check_docker

    if docker compose -f "$COMPOSE_FILE" ps --services --filter "status=running" | grep -q conductor; then
        log_success "Conductor is running"
        echo ""
        docker compose -f "$COMPOSE_FILE" ps
    else
        log_warn "Conductor is not running"
    fi
}

# Main entry point
main() {
    local command="${1:-all}"

    case "$command" in
        start)
            start_conductor
            ;;
        stop)
            stop_conductor
            ;;
        logs)
            show_logs
            ;;
        clean)
            clean_all
            ;;
        status)
            check_status
            ;;
        test)
            run_tests
            ;;
        all)
            start_conductor
            run_tests
            stop_conductor
            ;;
        help|--help|-h)
            echo "Mycelix Conductor Integration Tests"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  start   Start the conductor"
            echo "  stop    Stop the conductor"
            echo "  test    Run integration tests (conductor must be running)"
            echo "  logs    Show conductor logs"
            echo "  status  Check conductor status"
            echo "  clean   Remove all test data and containers"
            echo "  all     Start conductor, run tests, stop conductor (default)"
            echo "  help    Show this help message"
            ;;
        *)
            log_error "Unknown command: $command"
            echo "Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
