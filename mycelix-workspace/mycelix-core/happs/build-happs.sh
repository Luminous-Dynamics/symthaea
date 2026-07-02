#!/usr/bin/env bash
#
# build-happs.sh - Build script for Mycelix-Core Holochain hApps
#
# This script compiles all zomes to WASM, packages DNAs, and creates hApp bundles.
#
# Usage:
#   ./build-happs.sh [OPTIONS]
#
# Options:
#   --clean         Clean build artifacts before building
#   --zomes-only    Only build zomes (skip DNA and hApp packaging)
#   --dnas-only     Only package DNAs (assumes zomes are built)
#   --happs-only    Only package hApps (assumes DNAs are packaged)
#   --release       Build with release optimizations (default)
#   --debug         Build with debug symbols
#   --parallel      Build zomes in parallel (faster, uses more resources)
#   --verbose       Enable verbose output
#   --help          Show this help message
#

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Directories
ZOMES_DIR="${PROJECT_ROOT}/zomes"
DNAS_DIR="${PROJECT_ROOT}/dnas"
DIST_DIR="${PROJECT_ROOT}/dist"
DIST_ZOMES="${DIST_DIR}/zomes"
DIST_DNAS="${DIST_DIR}/dnas"
DIST_HAPPS="${DIST_DIR}/happs"

# Build settings
BUILD_MODE="release"
WASM_TARGET="wasm32-unknown-unknown"
PARALLEL_BUILD=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Zome definitions
# =============================================================================

# All zomes in the ecosystem
declare -A ZOMES=(
    # Identity hApp
    ["identity_core"]="zerotrustml-identity"
    ["identity_core_coordinator"]="zerotrustml-identity"
    ["capability_tokens"]="capability-tokens"
    ["capability_tokens_coordinator"]="capability-tokens"

    # Reputation hApp
    ["reputation_ledger"]="zerotrustml-reputation"
    ["reputation_ledger_coordinator"]="zerotrustml-reputation"
    ["behavioral_proofs"]="behavioral-proofs"
    ["behavioral_proofs_coordinator"]="behavioral-proofs"
    ["stake_registry"]="stake-registry"
    ["stake_registry_coordinator"]="stake-registry"

    # Governance hApp
    ["governance_core"]="zerotrustml-governance"
    ["governance_core_coordinator"]="zerotrustml-governance"
    ["policy_registry"]="policy-registry"
    ["policy_registry_coordinator"]="policy-registry"
    ["prediction_markets"]="prediction-markets"
    ["prediction_markets_coordinator"]="prediction-markets"
    ["model_validation"]="model-validation"
    ["model_validation_coordinator"]="model-validation"

    # Mail hApp
    ["mail_core"]="mycelix-mail"
    ["mail_core_coordinator"]="mycelix-mail"
    ["channels"]="mail-channels"
    ["channels_coordinator"]="mail-channels"
    ["attachments"]="mail-attachments"
    ["attachments_coordinator"]="mail-attachments"
    ["notifications"]="mail-notifications"
    ["notifications_coordinator"]="mail-notifications"

    # Marketplace hApp
    ["marketplace_core"]="mycelix-marketplace"
    ["marketplace_core_coordinator"]="mycelix-marketplace"
    ["escrow"]="marketplace-escrow"
    ["escrow_coordinator"]="marketplace-escrow"
    ["model_registry"]="model-registry"
    ["model_registry_coordinator"]="model-registry"
    ["reviews"]="marketplace-reviews"
    ["reviews_coordinator"]="marketplace-reviews"
    ["analytics"]="marketplace-analytics"
    ["analytics_coordinator"]="marketplace-analytics"
)

# DNA definitions: dna_name -> "zome1,zome2,..."
declare -A DNAS=(
    ["zerotrustml-identity"]="identity_core,identity_core_coordinator"
    ["capability-tokens"]="capability_tokens,capability_tokens_coordinator"
    ["zerotrustml-reputation"]="reputation_ledger,reputation_ledger_coordinator"
    ["behavioral-proofs"]="behavioral_proofs,behavioral_proofs_coordinator"
    ["stake-registry"]="stake_registry,stake_registry_coordinator"
    ["zerotrustml-governance"]="governance_core,governance_core_coordinator"
    ["policy-registry"]="policy_registry,policy_registry_coordinator"
    ["prediction-markets"]="prediction_markets,prediction_markets_coordinator"
    ["model-validation"]="model_validation,model_validation_coordinator"
    ["mycelix-mail"]="mail_core,mail_core_coordinator"
    ["mail-channels"]="channels,channels_coordinator"
    ["mail-attachments"]="attachments,attachments_coordinator"
    ["mail-notifications"]="notifications,notifications_coordinator"
    ["mycelix-marketplace"]="marketplace_core,marketplace_core_coordinator"
    ["marketplace-escrow"]="escrow,escrow_coordinator"
    ["model-registry"]="model_registry,model_registry_coordinator"
    ["marketplace-reviews"]="reviews,reviews_coordinator"
    ["marketplace-analytics"]="analytics,analytics_coordinator"
)

# hApp definitions: happ_name -> "dna1,dna2,..."
declare -A HAPPS=(
    ["zerotrustml-identity"]="zerotrustml-identity,capability-tokens"
    ["zerotrustml-reputation"]="zerotrustml-reputation,behavioral-proofs,stake-registry"
    ["zerotrustml-governance"]="zerotrustml-governance,policy-registry,prediction-markets,model-validation"
    ["mycelix-mail"]="mycelix-mail,mail-channels,mail-attachments,mail-notifications"
    ["mycelix-marketplace"]="mycelix-marketplace,marketplace-escrow,model-registry,marketplace-reviews,marketplace-analytics"
)

# Build order (respecting dependencies)
BUILD_ORDER=(
    "zerotrustml-identity"
    "zerotrustml-reputation"
    "mycelix-mail"
    "zerotrustml-governance"
    "mycelix-marketplace"
)

# =============================================================================
# Helper Functions
# =============================================================================

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

log_step() {
    echo -e "${CYAN}==>${NC} $1"
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}[VERBOSE]${NC} $1"
    fi
}

show_help() {
    sed -n '/^# Usage:/,/^#$/p' "$0" | sed 's/^# //'
    exit 0
}

check_prerequisites() {
    log_step "Checking prerequisites..."

    local missing=()

    # Check Rust
    if ! command -v rustc &> /dev/null; then
        missing+=("rustc")
    else
        local rust_version=$(rustc --version | cut -d' ' -f2)
        log_verbose "Found Rust $rust_version"
    fi

    # Check Cargo
    if ! command -v cargo &> /dev/null; then
        missing+=("cargo")
    fi

    # Check WASM target
    if ! rustup target list --installed | grep -q "$WASM_TARGET"; then
        log_warn "WASM target not installed. Installing..."
        rustup target add "$WASM_TARGET"
    fi

    # Check Holochain CLI
    if ! command -v hc &> /dev/null; then
        missing+=("hc (Holochain CLI)")
    else
        local hc_version=$(hc --version 2>/dev/null || echo "unknown")
        log_verbose "Found Holochain CLI $hc_version"
    fi

    # Check wasm-opt (optional but recommended)
    if ! command -v wasm-opt &> /dev/null; then
        log_warn "wasm-opt not found. WASM optimization will be skipped."
        log_warn "Install binaryen for smaller WASM bundles."
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install the missing tools and try again."
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

create_directories() {
    log_step "Creating build directories..."

    mkdir -p "$DIST_ZOMES"
    mkdir -p "$DIST_DNAS"
    mkdir -p "$DIST_HAPPS"
    mkdir -p "$ZOMES_DIR"
    mkdir -p "$DNAS_DIR"

    log_success "Directories created"
}

clean_build() {
    log_step "Cleaning build artifacts..."

    rm -rf "$DIST_DIR"
    rm -rf "${PROJECT_ROOT}/target"

    log_success "Build artifacts cleaned"
}

# =============================================================================
# Build Functions
# =============================================================================

build_zome() {
    local zome_name=$1
    local zome_dir="${ZOMES_DIR}/${zome_name}"

    if [ ! -d "$zome_dir" ]; then
        log_warn "Zome directory not found: $zome_dir (will create scaffold)"
        create_zome_scaffold "$zome_name"
    fi

    log_verbose "Building zome: $zome_name"

    local cargo_flags=""
    if [ "$BUILD_MODE" = "release" ]; then
        cargo_flags="--release"
    fi

    (
        cd "$zome_dir"
        cargo build $cargo_flags --target "$WASM_TARGET" 2>&1 | while read line; do
            log_verbose "  $line"
        done
    )

    # Copy WASM to dist
    local wasm_source="${zome_dir}/target/${WASM_TARGET}/${BUILD_MODE}/${zome_name}.wasm"
    if [ -f "$wasm_source" ]; then
        cp "$wasm_source" "${DIST_ZOMES}/${zome_name}.wasm"

        # Optimize if wasm-opt is available
        if command -v wasm-opt &> /dev/null && [ "$BUILD_MODE" = "release" ]; then
            log_verbose "Optimizing WASM: $zome_name"
            wasm-opt -O3 "${DIST_ZOMES}/${zome_name}.wasm" \
                     -o "${DIST_ZOMES}/${zome_name}.wasm"
        fi

        local size=$(du -h "${DIST_ZOMES}/${zome_name}.wasm" | cut -f1)
        log_success "Built $zome_name ($size)"
    else
        log_error "WASM file not found: $wasm_source"
        return 1
    fi
}

create_zome_scaffold() {
    local zome_name=$1
    local zome_dir="${ZOMES_DIR}/${zome_name}"

    log_info "Creating scaffold for zome: $zome_name"

    mkdir -p "${zome_dir}/src"

    # Create Cargo.toml
    cat > "${zome_dir}/Cargo.toml" << EOF
[package]
name = "${zome_name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]
name = "${zome_name}"

[dependencies]
hdk = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
panic = "abort"
EOF

    # Create basic lib.rs
    cat > "${zome_dir}/src/lib.rs" << EOF
use hdk::prelude::*;

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

// TODO: Implement zome functions for ${zome_name}
EOF

    log_success "Created scaffold for $zome_name"
}

build_all_zomes() {
    log_step "Building all zomes..."

    local zome_names=("${!ZOMES[@]}")

    if [ "$PARALLEL_BUILD" = true ]; then
        log_info "Building in parallel..."
        local pids=()

        for zome_name in "${zome_names[@]}"; do
            build_zome "$zome_name" &
            pids+=($!)
        done

        # Wait for all builds
        local failed=0
        for pid in "${pids[@]}"; do
            if ! wait $pid; then
                ((failed++))
            fi
        done

        if [ $failed -gt 0 ]; then
            log_error "$failed zome(s) failed to build"
            return 1
        fi
    else
        for zome_name in "${zome_names[@]}"; do
            build_zome "$zome_name" || return 1
        done
    fi

    log_success "All zomes built successfully"
}

# =============================================================================
# DNA Packaging Functions
# =============================================================================

create_dna_manifest() {
    local dna_name=$1
    local manifest_file="${DNAS_DIR}/${dna_name}.yaml"

    local zomes_str="${DNAS[$dna_name]}"
    IFS=',' read -ra zome_list <<< "$zomes_str"

    cat > "$manifest_file" << EOF
---
manifest_version: "1"
name: ${dna_name}
integrity:
  origin_time: $(date -u +"%Y-%m-%dT%H:%M:%S.000000Z")
  network_seed: ~
  properties: ~
  zomes:
EOF

    # Add integrity zomes (non-coordinator)
    for zome in "${zome_list[@]}"; do
        if [[ ! "$zome" =~ _coordinator$ ]]; then
            cat >> "$manifest_file" << EOF
    - name: ${zome}
      bundled: ../dist/zomes/${zome}.wasm
EOF
        fi
    done

    cat >> "$manifest_file" << EOF
coordinator:
  zomes:
EOF

    # Add coordinator zomes
    for zome in "${zome_list[@]}"; do
        if [[ "$zome" =~ _coordinator$ ]]; then
            local integrity_zome="${zome%_coordinator}"
            cat >> "$manifest_file" << EOF
    - name: ${zome}
      bundled: ../dist/zomes/${zome}.wasm
      dependencies:
        - name: ${integrity_zome}
EOF
        fi
    done

    log_verbose "Created DNA manifest: $manifest_file"
}

package_dna() {
    local dna_name=$1
    local manifest_file="${DNAS_DIR}/${dna_name}.yaml"
    local output_file="${DIST_DNAS}/${dna_name}.dna"

    # Create manifest if it doesn't exist
    if [ ! -f "$manifest_file" ]; then
        create_dna_manifest "$dna_name"
    fi

    log_verbose "Packaging DNA: $dna_name"

    if command -v hc &> /dev/null; then
        hc dna pack "$manifest_file" -o "$output_file" 2>&1 | while read line; do
            log_verbose "  $line"
        done

        if [ -f "$output_file" ]; then
            local size=$(du -h "$output_file" | cut -f1)
            local hash=$(hc dna hash "$output_file" 2>/dev/null || echo "N/A")
            log_success "Packaged $dna_name ($size) - Hash: $hash"
        else
            log_error "Failed to package DNA: $dna_name"
            return 1
        fi
    else
        log_warn "hc CLI not available, creating placeholder for $dna_name"
        touch "$output_file"
    fi
}

package_all_dnas() {
    log_step "Packaging all DNAs..."

    for dna_name in "${!DNAS[@]}"; do
        package_dna "$dna_name" || return 1
    done

    log_success "All DNAs packaged successfully"
}

# =============================================================================
# hApp Packaging Functions
# =============================================================================

package_happ() {
    local happ_name=$1
    local manifest_file="${PROJECT_ROOT}/${happ_name}.happ.yaml"
    local output_file="${DIST_HAPPS}/${happ_name}.happ"

    if [ ! -f "$manifest_file" ]; then
        log_error "hApp manifest not found: $manifest_file"
        return 1
    fi

    log_verbose "Packaging hApp: $happ_name"

    if command -v hc &> /dev/null; then
        hc app pack "$manifest_file" -o "$output_file" 2>&1 | while read line; do
            log_verbose "  $line"
        done

        if [ -f "$output_file" ]; then
            local size=$(du -h "$output_file" | cut -f1)
            log_success "Packaged $happ_name ($size)"
        else
            log_error "Failed to package hApp: $happ_name"
            return 1
        fi
    else
        log_warn "hc CLI not available, creating placeholder for $happ_name"
        touch "$output_file"
    fi
}

package_all_happs() {
    log_step "Packaging all hApps..."

    for happ_name in "${BUILD_ORDER[@]}"; do
        package_happ "$happ_name" || return 1
    done

    log_success "All hApps packaged successfully"
}

# =============================================================================
# Build Summary
# =============================================================================

print_summary() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}           BUILD COMPLETE                  ${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""

    echo -e "${CYAN}Zomes:${NC}"
    if [ -d "$DIST_ZOMES" ]; then
        ls -lh "$DIST_ZOMES"/*.wasm 2>/dev/null | while read line; do
            echo "  $line"
        done
    fi

    echo ""
    echo -e "${CYAN}DNAs:${NC}"
    if [ -d "$DIST_DNAS" ]; then
        ls -lh "$DIST_DNAS"/*.dna 2>/dev/null | while read line; do
            echo "  $line"
        done
    fi

    echo ""
    echo -e "${CYAN}hApps:${NC}"
    if [ -d "$DIST_HAPPS" ]; then
        ls -lh "$DIST_HAPPS"/*.happ 2>/dev/null | while read line; do
            echo "  $line"
        done
    fi

    echo ""
    echo -e "${CYAN}Output Directory:${NC} $DIST_DIR"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local do_clean=false
    local build_zomes=true
    local build_dnas=true
    local build_happs=true

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                do_clean=true
                shift
                ;;
            --zomes-only)
                build_dnas=false
                build_happs=false
                shift
                ;;
            --dnas-only)
                build_zomes=false
                build_happs=false
                shift
                ;;
            --happs-only)
                build_zomes=false
                build_dnas=false
                shift
                ;;
            --release)
                BUILD_MODE="release"
                shift
                ;;
            --debug)
                BUILD_MODE="debug"
                shift
                ;;
            --parallel)
                PARALLEL_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done

    echo ""
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  Mycelix-Core hApp Build System           ${NC}"
    echo -e "${CYAN}============================================${NC}"
    echo ""
    echo -e "Build mode: ${YELLOW}${BUILD_MODE}${NC}"
    echo -e "Parallel:   ${YELLOW}${PARALLEL_BUILD}${NC}"
    echo ""

    # Execute build steps
    check_prerequisites

    if [ "$do_clean" = true ]; then
        clean_build
    fi

    create_directories

    if [ "$build_zomes" = true ]; then
        build_all_zomes || exit 1
    fi

    if [ "$build_dnas" = true ]; then
        package_all_dnas || exit 1
    fi

    if [ "$build_happs" = true ]; then
        package_all_happs || exit 1
    fi

    print_summary
}

# Run main with all arguments
main "$@"
