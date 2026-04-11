#!/bin/bash
# Mycelix Mail - Package Publishing Script
# =========================================
# Publishes TypeScript client to npm

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
CLIENT_DIR="$PROJECT_ROOT/holochain/client"

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check npm
    if ! command -v npm &> /dev/null; then
        log_error "npm not found"
        exit 1
    fi

    # Check npm auth
    if ! npm whoami &> /dev/null; then
        log_error "Not logged in to npm. Run: npm login"
        exit 1
    fi

    log_success "Prerequisites OK"
}

# Validate package
validate_package() {
    log_info "Validating package..."

    cd "$CLIENT_DIR"

    # Type check
    log_info "Running type check..."
    npm run typecheck

    # Lint
    log_info "Running linter..."
    npm run lint

    # Tests
    log_info "Running tests..."
    npm run test:run

    # Build
    log_info "Building package..."
    npm run build

    log_success "Validation passed"
}

# Check version
check_version() {
    cd "$CLIENT_DIR"

    local local_version=$(node -p "require('./package.json').version")
    local pkg_name=$(node -p "require('./package.json').name")

    # Get published version
    local published_version=$(npm view "$pkg_name" version 2>/dev/null || echo "0.0.0")

    if [ "$local_version" == "$published_version" ]; then
        log_error "Version $local_version is already published"
        log_info "Run ./scripts/version.sh bump patch to increment version"
        exit 1
    fi

    log_success "Version $local_version is ready for publishing"
    echo "$local_version"
}

# Publish to npm
publish_npm() {
    local dry_run=${1:-false}

    cd "$CLIENT_DIR"

    log_info "Publishing to npm..."

    if [ "$dry_run" == "true" ]; then
        npm publish --dry-run
        log_success "Dry run complete"
    else
        npm publish --access public
        log_success "Published to npm"
    fi
}

# Create tarball
pack_tarball() {
    cd "$CLIENT_DIR"

    log_info "Creating tarball..."

    npm pack

    local tarball=$(ls -t *.tgz | head -1)
    log_success "Created $tarball"

    echo "$CLIENT_DIR/$tarball"
}

# Print usage
usage() {
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  check           Validate package and check version"
    echo "  dry-run         Run publish without actually publishing"
    echo "  publish         Publish to npm"
    echo "  pack            Create tarball without publishing"
    echo ""
    echo "Options:"
    echo "  --skip-tests    Skip test validation"
    echo ""
    echo "Examples:"
    echo "  $0 check"
    echo "  $0 dry-run"
    echo "  $0 publish"
    echo ""
}

# Main
main() {
    local skip_tests=false

    # Parse options
    for arg in "$@"; do
        case $arg in
            --skip-tests)
                skip_tests=true
                shift
                ;;
        esac
    done

    case "${1:-}" in
        check)
            check_prerequisites
            if [ "$skip_tests" != "true" ]; then
                validate_package
            fi
            check_version
            ;;

        dry-run)
            check_prerequisites
            if [ "$skip_tests" != "true" ]; then
                validate_package
            fi
            check_version
            publish_npm true
            ;;

        publish)
            check_prerequisites
            if [ "$skip_tests" != "true" ]; then
                validate_package
            fi
            check_version

            echo ""
            read -p "Are you sure you want to publish? [y/N] " confirm
            if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
                log_info "Cancelled"
                exit 0
            fi

            publish_npm false
            ;;

        pack)
            if [ "$skip_tests" != "true" ]; then
                validate_package
            fi
            pack_tarball
            ;;

        -h|--help|help)
            usage
            ;;

        *)
            log_error "Unknown command: ${1:-}"
            usage
            exit 1
            ;;
    esac
}

main "$@"
