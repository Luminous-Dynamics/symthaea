#!/usr/bin/env bash
#
# Build civic-happ WASM zomes and package DNA/hApp
#
# Prerequisites:
#   - Rust with wasm32-unknown-unknown target
#   - Holochain CLI (hc) - available in nix develop
#
# Usage:
#   ./scripts/build.sh        # Full build
#   ./scripts/build.sh wasm   # Only build WASM
#   ./scripts/build.sh pack   # Only package (assumes WASM built)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

build_wasm() {
    echo "Building WASM zomes..."
    cargo build --release --target wasm32-unknown-unknown
    echo "WASM build complete."
    ls -la target/wasm32-unknown-unknown/release/*.wasm
}

pack_dna() {
    echo "Packaging DNA..."
    if ! command -v hc &>/dev/null; then
        echo "Error: 'hc' not found. Run 'nix develop' first."
        exit 1
    fi
    hc dna pack dna/
    echo "DNA packaged: dna/civic.dna"
}

pack_happ() {
    echo "Packaging hApp..."
    if ! command -v hc &>/dev/null; then
        echo "Error: 'hc' not found. Run 'nix develop' first."
        exit 1
    fi
    hc app pack .
    echo "hApp packaged: civic.happ"
}

build_client() {
    echo "Building TypeScript client..."
    cd client
    npm install
    npm run build
    cd ..
    echo "Client build complete."
}

case "${1:-all}" in
    wasm)
        build_wasm
        ;;
    pack)
        pack_dna
        pack_happ
        ;;
    client)
        build_client
        ;;
    all)
        build_wasm
        pack_dna
        pack_happ
        build_client
        ;;
    *)
        echo "Usage: $0 [wasm|pack|client|all]"
        exit 1
        ;;
esac

echo "Build complete!"
