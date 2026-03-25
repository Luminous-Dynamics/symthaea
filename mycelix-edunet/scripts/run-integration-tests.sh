#!/usr/bin/env bash
# Run integration tests for Mycelix EduNet
#
# This script:
#   1. Builds all 20 WASM zomes (10 integrity + 10 coordinator)
#   2. Packages the DNA bundle
#   3. Packages the hApp bundle
#   4. Runs the ignored sweettest integration tests
#
# Prerequisites:
#   - Rust toolchain with wasm32-unknown-unknown target
#   - Holochain CLI tools (hc)
#   - holochain crate with sweettest feature (test dependency)
#
# Usage:
#   ./scripts/run-integration-tests.sh              # Run all ignored tests
#   ./scripts/run-integration-tests.sh --skip-build  # Skip WASM build (reuse existing)
#   ./scripts/run-integration-tests.sh --test e2e    # Run specific test binary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_BUILD=0
TEST_FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --test)
            TEST_FILTER="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--skip-build] [--test <name>]"
            echo ""
            echo "Options:"
            echo "  --skip-build   Skip WASM build, reuse existing artifacts"
            echo "  --test <name>  Run only a specific test binary (e.g., e2e_test)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$ROOT_DIR"

echo "Mycelix EduNet - Integration Test Runner"
echo "========================================="
echo ""

# ============================================================================
# Step 1: Build WASM zomes
# ============================================================================

if [[ $SKIP_BUILD -eq 0 ]]; then
    echo "[1/4] Building WASM zomes..."

    # All 10 integrity zomes
    INTEGRITY_ZOMES=(
        learning_integrity
        fl_integrity
        credential_integrity
        dao_integrity
        srs_integrity
        gamification_integrity
        adaptive_integrity
        integration_integrity
        pods_integrity
        knowledge_integrity
    )

    # All 10 coordinator zomes
    COORDINATOR_ZOMES=(
        learning_coordinator
        fl_coordinator
        credential_coordinator
        dao_coordinator
        srs_coordinator
        gamification_coordinator
        adaptive_coordinator
        integration_coordinator
        pods_coordinator
        knowledge_coordinator
    )

    # Build integrity zomes
    echo "  Building integrity zomes..."
    cargo build \
        --target wasm32-unknown-unknown \
        --release \
        $(printf -- '-p %s ' "${INTEGRITY_ZOMES[@]}")

    # Build coordinator zomes
    echo "  Building coordinator zomes..."
    cargo build \
        --target wasm32-unknown-unknown \
        --release \
        $(printf -- '-p %s ' "${COORDINATOR_ZOMES[@]}")

    echo "  All 20 WASM zomes built."
    echo ""

    # Verify WASM files exist
    WASM_DIR="target/wasm32-unknown-unknown/release"
    MISSING=0
    for zome in "${INTEGRITY_ZOMES[@]}" "${COORDINATOR_ZOMES[@]}"; do
        if [[ ! -f "$WASM_DIR/${zome}.wasm" ]]; then
            echo "  ERROR: Missing $WASM_DIR/${zome}.wasm"
            MISSING=1
        fi
    done
    if [[ $MISSING -eq 1 ]]; then
        echo "WASM build incomplete. Aborting."
        exit 1
    fi
else
    echo "[1/4] Skipping WASM build (--skip-build)"
    echo ""
fi

# ============================================================================
# Step 2: Package DNA
# ============================================================================

echo "[2/4] Packaging DNA bundle..."
pushd dna >/dev/null
hc dna pack . -o edunet.dna
popd >/dev/null

if [[ ! -f dna/edunet.dna ]]; then
    echo "  ERROR: DNA bundle not created at dna/edunet.dna"
    exit 1
fi
DNA_SIZE=$(du -h dna/edunet.dna | cut -f1)
echo "  DNA bundle: dna/edunet.dna ($DNA_SIZE)"
echo ""

# ============================================================================
# Step 3: Package hApp
# ============================================================================

echo "[3/4] Packaging hApp bundle..."

# Copy DNA into happ directory (happ.yaml references ./edunet.dna)
cp dna/edunet.dna happ/edunet.dna
hc app pack happ/ -o happ/mycelix-edunet.happ

if [[ ! -f happ/mycelix-edunet.happ ]]; then
    echo "  ERROR: hApp bundle not created at happ/mycelix-edunet.happ"
    exit 1
fi
HAPP_SIZE=$(du -h happ/mycelix-edunet.happ | cut -f1)
echo "  hApp bundle: happ/mycelix-edunet.happ ($HAPP_SIZE)"
echo ""

# ============================================================================
# Step 4: Run integration tests
# ============================================================================

echo "[4/4] Running integration tests (sweettest)..."
echo ""

# The sweettest tests use SweetConductor which spins up an in-process
# conductor. They load the DNA bundle from ../dna/edunet.dna (relative to
# the tests/ directory). No external conductor is needed.

TEST_ARGS=(
    --manifest-path tests/Cargo.toml
    -- --ignored
)

if [[ -n "$TEST_FILTER" ]]; then
    # Run specific test binary
    TEST_ARGS=(
        --manifest-path tests/Cargo.toml
        --test "$TEST_FILTER"
        -- --ignored
    )
fi

# Run from project root so relative paths in tests resolve correctly
cargo test "${TEST_ARGS[@]}"

echo ""
echo "========================================="
echo "Integration tests complete."
