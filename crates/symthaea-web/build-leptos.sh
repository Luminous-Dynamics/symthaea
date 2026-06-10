#!/usr/bin/env bash
# build-leptos.sh — Build the Leptos CSR frontend with Trunk
#
# Usage:
#   bash build-leptos.sh          # Release build
#   bash build-leptos.sh --dev    # Dev build
#   bash build-leptos.sh --jobs 2 # Limit parallelism (useful under high system load)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

JOBS=""
MODE="--release"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)    MODE=""; shift ;;
        --jobs)   JOBS="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Export CARGO_BUILD_JOBS if specified (helps on memory-constrained systems)
if [[ -n "$JOBS" ]]; then
    export CARGO_BUILD_JOBS="$JOBS"
    echo "[leptos] Limiting cargo parallelism to $JOBS jobs"
fi

echo "[leptos] Building Leptos CSR frontend..."

# Use trunk from nixpkgs if not in PATH
if command -v trunk &>/dev/null; then
    TRUNK=trunk
else
    echo "[leptos] trunk not in PATH, using nix run nixpkgs#trunk"
    TRUNK="nix run nixpkgs#trunk --"
fi

# Build
if [[ -n "$MODE" ]]; then
    echo "[leptos] Building in release mode..."
    $TRUNK build $MODE
else
    echo "[leptos] Building in dev mode..."
    $TRUNK build
fi

echo ""
echo "[leptos] Output in dist/"
ls -la dist/ 2>/dev/null || echo "(dist/ not found — trunk may have failed)"

# Show WASM size if available
if ls dist/*.wasm &>/dev/null 2>&1; then
    echo "[leptos] WASM sizes:"
    du -h dist/*.wasm
fi
