#!/usr/bin/env bash
# GPU preflight and smoke test for Symthaea.
#
# This separates three failure classes:
# 1. Host driver/module missing
# 2. Device nodes not visible in the current environment
# 3. CUDA user-space path failing inside the Rust/Candle stack
#
# Usage:
#   ./scripts/gpu_smoke.sh
#   ./scripts/gpu_smoke.sh --with-broca-test

set -euo pipefail

cd "$(dirname "$0")/.."

run_broca_test=0
if [[ "${1:-}" == "--with-broca-test" ]]; then
    run_broca_test=1
fi

echo "=== Symthaea GPU Smoke ==="
echo

echo "[1/4] NVIDIA kernel module version"
if [[ -r /proc/driver/nvidia/version ]]; then
    cat /proc/driver/nvidia/version
else
    echo "  /proc/driver/nvidia/version not readable"
fi
echo

echo "[2/4] Device nodes visible in current environment"
if ls /dev/nvidia* >/dev/null 2>&1; then
    ls -l /dev/nvidia*
else
    echo "  No /dev/nvidia* nodes visible"
    echo "  This often means the current shell/container/sandbox cannot see the GPU."
fi
echo

echo "[3/4] nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi; then
        echo "  nvidia-smi OK"
    else
        echo "  nvidia-smi failed"
    fi
else
    echo "  nvidia-smi not found"
fi
echo

echo "[4/4] NixOS CUDA library path"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
if [[ -e /run/opengl-driver/lib/libcuda.so.1 ]]; then
    ls -l /run/opengl-driver/lib/libcuda.so.1
else
    echo "  /run/opengl-driver/lib/libcuda.so.1 missing"
fi
echo

if [[ "$run_broca_test" -eq 1 ]]; then
    echo "[extra] Broca CUDA smoke test"
    if [[ -z "${IN_NIX_SHELL:-}" ]]; then
        echo "  Broca CUDA smoke should run inside the GPU flake shell:"
        echo "    nix develop .#gpu -c ./scripts/gpu_smoke.sh --with-broca-test"
        exit 1
    fi

    cargo test -p symthaea-broca --features mamba --test cuda_smoke -- --ignored --nocapture
fi
