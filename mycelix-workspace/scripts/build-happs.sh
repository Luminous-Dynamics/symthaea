#!/usr/bin/env bash
# Build all Mycelix hApp bundles
# Run from mycelix-workspace directory inside nix develop shell
#
# Usage:
#   nix develop
#   ./scripts/build-happs.sh [happ-name]
#
# If no happ-name is provided, builds all hApps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
HAPPS_DIR="$WORKSPACE_DIR/happs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for required tools
check_tools() {
    if ! command -v hc &> /dev/null; then
        log_error "hc not found. Run 'nix develop' first."
        exit 1
    fi
    if ! command -v cargo &> /dev/null; then
        log_error "cargo not found. Run 'nix develop' first."
        exit 1
    fi
    log_info "Tools found: hc $(hc --version 2>/dev/null || echo 'unknown'), cargo $(cargo --version)"
}

# Build a single hApp
build_happ() {
    local happ_name="$1"
    local happ_dir="$HAPPS_DIR/$happ_name"

    if [[ ! -d "$happ_dir" ]]; then
        log_error "hApp directory not found: $happ_dir"
        return 1
    fi

    # Follow symlink if needed
    if [[ -L "$happ_dir" ]]; then
        happ_dir="$(readlink -f "$happ_dir")"
    fi

    log_info "Building $happ_name from $happ_dir"
    cd "$happ_dir"

    # Determine structure - check multiple patterns
    local dna_dir=""
    local happ_yaml=""
    local pack_dir="."

    # Pattern 1: Standard structure (dna/ + root happ.yaml)
    if [[ -f "dna/dna.yaml" && -f "happ.yaml" ]]; then
        dna_dir="dna"
        happ_yaml="happ.yaml"
    # Pattern 2: happ.yaml at root with nested dna in dnas/[name]/workdir/
    elif [[ -f "happ.yaml" && -d "dnas" ]]; then
        if [[ -f "dnas/${happ_name}/workdir/dna.yaml" ]]; then
            dna_dir="dnas/${happ_name}/workdir"
            happ_yaml="happ.yaml"
        elif [[ -f "dnas/climate/workdir/dna.yaml" ]]; then
            dna_dir="dnas/climate/workdir"
            happ_yaml="happ.yaml"
        elif [[ -f "dnas/mutualaid/workdir/dna.yaml" ]]; then
            dna_dir="dnas/mutualaid/workdir"
            happ_yaml="happ.yaml"
        else
            log_warn "No dna.yaml found in dnas/ for $happ_name"
            return 1
        fi
    # Pattern 3: Native workspace (consensus style)
    elif [[ -f "happ.yaml" && -f "dna/dna.yaml" ]]; then
        dna_dir="dna"
        happ_yaml="happ.yaml"
    # Pattern 4: Music special case
    elif [[ -f "dnas/mycelix-music/dna.yaml" && -f "happ.yaml" ]]; then
        dna_dir="dnas/mycelix-music"
        happ_yaml="happ.yaml"
    else
        log_warn "No dna.yaml found for $happ_name, skipping"
        return 1
    fi

    if [[ ! -f "$happ_yaml" ]]; then
        log_warn "No happ.yaml found at $happ_yaml for $happ_name, skipping"
        return 1
    fi

    # Build WASM zomes
    log_info "  Compiling zomes to WASM..."
    cargo build --release --target wasm32-unknown-unknown 2>&1 | tail -5

    # Pack DNA
    log_info "  Packing DNA from $dna_dir..."
    hc dna pack "$dna_dir" 2>&1 | tail -3

    # Pack hApp from the directory containing happ.yaml
    log_info "  Packing hApp from $pack_dir..."
    hc app pack "$pack_dir" 2>&1 | tail -3

    # Check result
    local happ_file=$(find . -maxdepth 2 -name "*.happ" -newer "$dna_dir/dna.yaml" 2>/dev/null | head -1)
    if [[ -n "$happ_file" ]]; then
        log_info "  ✅ Built: $happ_file ($(du -h "$happ_file" | cut -f1))"
    else
        log_warn "  ⚠️ hApp file not found after build"
    fi
}

# Main
main() {
    check_tools

    if [[ $# -gt 0 ]]; then
        # Build specific hApp
        build_happ "$1"
    else
        # Build all hApps with proper configs
        log_info "Building all configured hApps..."

        # Standard structure hApps (dna/ + happ.yaml)
        for happ in energy health media property; do
            build_happ "$happ" || true
        done

        # Nested structure hApps (dnas/[name]/workdir/ or workdir/)
        for happ in climate mutualaid; do
            build_happ "$happ" || true
        done

        # Native workspace hApps
        for happ in consensus; do
            build_happ "$happ" || true
        done

        # Special structure hApps
        for happ in music; do
            build_happ "$happ" || true
        done

        log_info "Build complete!"

        # Summary
        echo ""
        log_info "=== hApp Bundle Summary ==="
        find "$HAPPS_DIR" -name "*.happ" -type f 2>/dev/null | while read f; do
            echo "  $(basename "$(dirname "$f")")/$(basename "$f"): $(du -h "$f" | cut -f1)"
        done
    fi
}

main "$@"
