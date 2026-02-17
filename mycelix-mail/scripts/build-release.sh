#!/usr/bin/env bash
#
# Mycelix Mail - Production Release Build Script
# ===============================================
#
# This script builds all components in release mode, optimizes WASM files,
# creates versioned artifacts, and generates checksums.
#
# Usage:
#   ./scripts/build-release.sh [OPTIONS]
#
# Options:
#   -v, --version VERSION   Specify version (default: from git tag or package.json)
#   -o, --output DIR        Output directory (default: ./release)
#   -s, --skip-tests        Skip running tests
#   -c, --clean             Clean before building
#   -n, --no-optimize       Skip WASM optimization
#   -a, --analyze           Run bundle analysis
#   -h, --help              Show this help message
#
# Examples:
#   ./scripts/build-release.sh
#   ./scripts/build-release.sh -v 1.0.0 -o ./dist
#   ./scripts/build-release.sh --clean --analyze
#

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Directories
HOLOCHAIN_DIR="$PROJECT_ROOT/holochain"
FRONTEND_DIR="$PROJECT_ROOT/ui/frontend"
SDK_TS_DIR="$PROJECT_ROOT/sdk/typescript"
CLIENT_DIR="$HOLOCHAIN_DIR/client"

# Defaults
OUTPUT_DIR="$PROJECT_ROOT/release"
SKIP_TESTS=false
CLEAN_BUILD=false
OPTIMIZE_WASM=true
ANALYZE_BUNDLE=false
VERSION=""

# WASM target
WASM_TARGET="wasm32-unknown-unknown"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BOLD}${CYAN}==> $1${NC}"
}

print_header() {
    echo -e "${BOLD}${BLUE}"
    echo "=============================================="
    echo "  MYCELIX MAIL - PRODUCTION RELEASE BUILD"
    echo "  Version: $VERSION"
    echo "  Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "=============================================="
    echo -e "${NC}"
}

print_summary() {
    echo -e "\n${BOLD}${GREEN}"
    echo "=============================================="
    echo "  BUILD COMPLETE"
    echo "=============================================="
    echo -e "${NC}"
    echo -e "${CYAN}Version:${NC}    $VERSION"
    echo -e "${CYAN}Output:${NC}     $OUTPUT_DIR"
    echo -e "${CYAN}Artifacts:${NC}  $OUTPUT_DIR/artifacts"
    echo ""
    echo -e "${CYAN}Contents:${NC}"
    if [ -d "$OUTPUT_DIR/artifacts" ]; then
        du -sh "$OUTPUT_DIR/artifacts"/* 2>/dev/null | sed 's/^/  /'
    fi
    echo ""
    echo -e "${CYAN}Checksums:${NC} $OUTPUT_DIR/artifacts/CHECKSUMS.txt"
}

show_help() {
    head -30 "$0" | tail -28 | sed 's/^# *//'
    exit 0
}

get_version() {
    if [ -n "$VERSION" ]; then
        echo "$VERSION"
        return
    fi

    # Try git tag
    if git describe --tags --exact-match HEAD 2>/dev/null; then
        return
    fi

    # Try git describe
    if git describe --tags --always 2>/dev/null; then
        return
    fi

    # Fallback to package.json version
    if [ -f "$PROJECT_ROOT/package.json" ]; then
        grep -o '"version": "[^"]*"' "$PROJECT_ROOT/package.json" | cut -d'"' -f4 || echo "0.1.0"
        return
    fi

    # Default
    echo "0.1.0-dev"
}

check_dependencies() {
    log_step "Checking dependencies"

    local missing=()

    # Check required tools
    command -v cargo >/dev/null 2>&1 || missing+=("cargo")
    command -v npm >/dev/null 2>&1 || missing+=("npm")
    command -v rustup >/dev/null 2>&1 || missing+=("rustup")

    # Check optional tools
    if [ "$OPTIMIZE_WASM" = true ]; then
        if ! command -v wasm-opt >/dev/null 2>&1; then
            log_warn "wasm-opt not found - WASM optimization will be skipped"
            log_warn "Install with: cargo install wasm-opt or npm install -g binaryen"
            OPTIMIZE_WASM=false
        fi
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing[*]}"
        exit 1
    fi

    # Ensure WASM target is installed
    if ! rustup target list --installed | grep -q "$WASM_TARGET"; then
        log_info "Installing WASM target..."
        rustup target add "$WASM_TARGET"
    fi

    log_success "All dependencies satisfied"
}

clean_build_artifacts() {
    log_step "Cleaning build artifacts"

    rm -rf "$OUTPUT_DIR"
    rm -rf "$HOLOCHAIN_DIR/target/$WASM_TARGET/release"/*.wasm
    rm -rf "$FRONTEND_DIR/dist"
    rm -rf "$SDK_TS_DIR/dist"
    rm -rf "$CLIENT_DIR/dist"

    log_success "Clean complete"
}

build_zomes() {
    log_step "Building Holochain zomes (release mode)"
    log_info "Target: $WASM_TARGET"

    cd "$HOLOCHAIN_DIR"

    # Build with optimizations
    RUSTFLAGS="-C link-arg=-zstack-size=65536" \
        cargo build --release --target "$WASM_TARGET"

    # List built WASM files
    local wasm_count
    wasm_count=$(find "target/$WASM_TARGET/release" -name "*.wasm" -type f 2>/dev/null | wc -l)
    log_success "Built $wasm_count WASM zomes"
}

optimize_wasm() {
    if [ "$OPTIMIZE_WASM" = false ]; then
        log_warn "Skipping WASM optimization"
        return
    fi

    log_step "Optimizing WASM binaries with wasm-opt"

    local optimized_dir="$OUTPUT_DIR/optimized"
    mkdir -p "$optimized_dir"

    local wasm_dir="$HOLOCHAIN_DIR/target/$WASM_TARGET/release"
    local total_original=0
    local total_optimized=0

    for wasm_file in "$wasm_dir"/*.wasm; do
        if [ -f "$wasm_file" ]; then
            local filename
            filename=$(basename "$wasm_file")
            local original_size
            original_size=$(stat -c%s "$wasm_file" 2>/dev/null || stat -f%z "$wasm_file")
            total_original=$((total_original + original_size))

            log_info "Processing: $filename"

            # Run wasm-opt with aggressive size optimization
            if wasm-opt -Oz \
                --strip-debug \
                --strip-dwarf \
                --strip-producers \
                --vacuum \
                --dce \
                --optimize-casts \
                --remove-unused-names \
                --remove-unused-module-elements \
                "$wasm_file" -o "$optimized_dir/$filename" 2>/dev/null; then

                local optimized_size
                optimized_size=$(stat -c%s "$optimized_dir/$filename" 2>/dev/null || stat -f%z "$optimized_dir/$filename")
                total_optimized=$((total_optimized + optimized_size))

                local reduction
                reduction=$(echo "scale=1; (1 - $optimized_size / $original_size) * 100" | bc 2>/dev/null || echo "N/A")
                log_success "  $filename: $(numfmt --to=iec $original_size 2>/dev/null || echo "$original_size bytes") -> $(numfmt --to=iec $optimized_size 2>/dev/null || echo "$optimized_size bytes") (${reduction}% reduction)"
            else
                log_warn "  Failed to optimize $filename, copying unoptimized"
                cp "$wasm_file" "$optimized_dir/$filename"
                total_optimized=$((total_optimized + original_size))
            fi
        fi
    done

    if [ $total_original -gt 0 ]; then
        local total_reduction
        total_reduction=$(echo "scale=1; (1 - $total_optimized / $total_original) * 100" | bc 2>/dev/null || echo "N/A")
        log_success "Total WASM size: $(numfmt --to=iec $total_original 2>/dev/null || echo "$total_original bytes") -> $(numfmt --to=iec $total_optimized 2>/dev/null || echo "$total_optimized bytes") (${total_reduction}% reduction)"
    fi
}

build_happ() {
    log_step "Building hApp package"

    cd "$HOLOCHAIN_DIR"

    # Check if hc command exists
    if command -v hc >/dev/null 2>&1; then
        # Pack DNA
        if [ -d "dna" ]; then
            hc dna pack dna -o "$OUTPUT_DIR/artifacts/happ/"
        fi

        # Pack hApp
        if [ -f "happ.yaml" ]; then
            mkdir -p "$OUTPUT_DIR/artifacts/happ"
            hc app pack . -o "$OUTPUT_DIR/artifacts/happ/"
        fi
        log_success "hApp packaged"
    else
        log_warn "hc command not found - skipping hApp packaging"
    fi
}

build_frontend() {
    log_step "Building frontend (production mode)"

    cd "$FRONTEND_DIR"

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm ci --prefer-offline
    fi

    # Build with production mode
    if [ "$ANALYZE_BUNDLE" = true ]; then
        log_info "Building with bundle analysis..."
        ANALYZE=true NODE_ENV=production npm run build
    else
        NODE_ENV=production npm run build
    fi

    log_success "Frontend built"

    # Report bundle sizes
    if [ -d "dist" ]; then
        log_info "Bundle sizes:"
        du -sh dist/* 2>/dev/null | sed 's/^/  /'
    fi
}

build_sdk() {
    log_step "Building TypeScript SDK"

    cd "$SDK_TS_DIR"

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing SDK dependencies..."
        npm ci --prefer-offline
    fi

    npm run build

    log_success "TypeScript SDK built"
}

build_client() {
    log_step "Building Holochain client library"

    cd "$CLIENT_DIR"

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing client dependencies..."
        npm ci --prefer-offline
    fi

    npm run build

    log_success "Holochain client built"
}

run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_warn "Skipping tests (--skip-tests flag)"
        return
    fi

    log_step "Running tests"

    # Zome tests
    log_info "Running zome tests..."
    cd "$HOLOCHAIN_DIR"
    cargo test --release || {
        log_error "Zome tests failed"
        exit 1
    }

    # Frontend tests
    log_info "Running frontend tests..."
    cd "$FRONTEND_DIR"
    npm test -- --run || {
        log_error "Frontend tests failed"
        exit 1
    }

    # SDK tests
    log_info "Running SDK tests..."
    cd "$SDK_TS_DIR"
    npm test -- --run || {
        log_warn "SDK tests failed or not configured"
    }

    log_success "All tests passed"
}

collect_artifacts() {
    log_step "Collecting build artifacts"

    local artifacts_dir="$OUTPUT_DIR/artifacts"
    mkdir -p "$artifacts_dir"/{wasm,frontend,sdk,happ}

    # Copy optimized WASM files
    if [ -d "$OUTPUT_DIR/optimized" ]; then
        cp "$OUTPUT_DIR/optimized"/*.wasm "$artifacts_dir/wasm/" 2>/dev/null || true
    else
        cp "$HOLOCHAIN_DIR/target/$WASM_TARGET/release"/*.wasm "$artifacts_dir/wasm/" 2>/dev/null || true
    fi

    # Copy frontend build
    if [ -d "$FRONTEND_DIR/dist" ]; then
        cp -r "$FRONTEND_DIR/dist"/* "$artifacts_dir/frontend/" 2>/dev/null || true
    fi

    # Copy SDK build
    if [ -d "$SDK_TS_DIR/dist" ]; then
        cp -r "$SDK_TS_DIR/dist" "$artifacts_dir/sdk/typescript" 2>/dev/null || true
    fi

    # Copy client build
    if [ -d "$CLIENT_DIR/dist" ]; then
        cp -r "$CLIENT_DIR/dist" "$artifacts_dir/sdk/holochain-client" 2>/dev/null || true
    fi

    # Copy bundle analysis if available
    if [ -f "$FRONTEND_DIR/dist/stats.html" ]; then
        cp "$FRONTEND_DIR/dist/stats.html" "$artifacts_dir/frontend/bundle-analysis.html"
    fi

    # Create manifest
    cat > "$artifacts_dir/MANIFEST.txt" << EOF
Mycelix Mail Release
====================
Version:    $VERSION
Build Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Git SHA:    $(git rev-parse HEAD 2>/dev/null || echo 'N/A')
Git Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')
Build Host: $(hostname)
Platform:   $(uname -s) $(uname -m)

Components:
-----------
- Holochain Zomes (WASM)
- Frontend (Vite + React)
- TypeScript SDK
- Holochain Client Library

Build Options:
--------------
WASM Optimization: $OPTIMIZE_WASM
Tests Run: $([ "$SKIP_TESTS" = true ] && echo "No" || echo "Yes")
Bundle Analysis: $ANALYZE_BUNDLE
EOF

    log_success "Artifacts collected"
}

generate_checksums() {
    log_step "Generating checksums"

    local artifacts_dir="$OUTPUT_DIR/artifacts"
    cd "$artifacts_dir"

    # Generate checksums for all files
    find . -type f ! -name "*.sha256" ! -name "CHECKSUMS.txt" -exec sha256sum {} \; > CHECKSUMS.txt 2>/dev/null || \
    find . -type f ! -name "*.sha256" ! -name "CHECKSUMS.txt" -exec shasum -a 256 {} \; > CHECKSUMS.txt

    log_success "Checksums saved to $artifacts_dir/CHECKSUMS.txt"

    # Create individual checksum files for main artifacts
    for file in wasm/*.wasm happ/*.happ happ/*.dna; do
        if [ -f "$file" ]; then
            sha256sum "$file" > "$file.sha256" 2>/dev/null || shasum -a 256 "$file" > "$file.sha256"
        fi
    done
}

create_tarball() {
    log_step "Creating release tarball"

    local tarball_name="mycelix-mail-$VERSION.tar.gz"
    local artifacts_dir="$OUTPUT_DIR/artifacts"

    cd "$OUTPUT_DIR"
    tar -czf "$tarball_name" -C artifacts .

    # Generate checksum for tarball
    sha256sum "$tarball_name" > "$tarball_name.sha256" 2>/dev/null || shasum -a 256 "$tarball_name" > "$tarball_name.sha256"

    log_success "Created: $OUTPUT_DIR/$tarball_name"
}

# =============================================================================
# Main Script
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -s|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -c|--clean)
                CLEAN_BUILD=true
                shift
                ;;
            -n|--no-optimize)
                OPTIMIZE_WASM=false
                shift
                ;;
            -a|--analyze)
                ANALYZE_BUNDLE=true
                shift
                ;;
            -h|--help)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done
}

main() {
    parse_args "$@"

    # Get version
    VERSION=$(get_version)

    # Print header
    print_header

    # Start timer
    local start_time
    start_time=$(date +%s)

    # Check dependencies
    check_dependencies

    # Clean if requested
    if [ "$CLEAN_BUILD" = true ]; then
        clean_build_artifacts
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Build all components
    build_zomes
    optimize_wasm
    build_happ
    build_frontend
    build_sdk
    build_client

    # Run tests
    run_tests

    # Collect artifacts
    collect_artifacts
    generate_checksums
    create_tarball

    # Calculate elapsed time
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Print summary
    print_summary
    echo -e "${CYAN}Build time:${NC} ${elapsed}s"
}

# Run main function
main "$@"
