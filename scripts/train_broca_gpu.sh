#!/usr/bin/env bash
# train_broca_gpu.sh — run broca-train with GPU, handling NixOS CUDA linkage.
#
# The vendored `cudarc-0.13.9-cuda129` patch unblocks CUDA 12.9 at compile time
# (see symthaea/Cargo.toml ~line 1767). But at runtime the binary still needs
# to find libcuda.so from the NixOS OpenGL driver directory — that's what the
# LD_LIBRARY_PATH line below does. Without it you get:
#     CUDA_ERROR_STUB_LIBRARY ... using CPU
#
# Usage:
#   ./scripts/train_broca_gpu.sh                          # defaults: train-combined-v8 + eval-epistemic-v1, 20 epochs
#   ./scripts/train_broca_gpu.sh --epochs 5               # shorter run
#   ./scripts/train_broca_gpu.sh --data <path> --output <path>
#   ./scripts/train_broca_gpu.sh -- --help                # forward to broca-train

set -euo pipefail

cd "$(dirname "$0")/.."   # symthaea/

# NixOS: real libcuda.so lives in /run/opengl-driver/lib, not /usr/lib.
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"

# The candle-kernels build needs cuda_runtime.h at COMPILE time, which only
# exists inside the .#broca-gpu nix devshell (CUDA_COMPUTE_CAP + nvcc on
# PATH) -- plain `cargo build` outside it fails with
# "fatal error: cuda_runtime.h: No such file or directory". Only wrap if
# not already inside a nix shell (avoids nesting when this script itself
# is invoked from within `nix develop`).
run_in_gpu_shell() {
    if [[ -z "${IN_NIX_SHELL:-}" ]]; then
        nix develop .#broca-gpu --command "$@"
    else
        "$@"
    fi
}

export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-75}"

# Build if missing or stale (sccache makes repeat builds instant).
if ! test -x target/release/broca-train; then
    echo "Building broca-train with gpu feature..."
    run_in_gpu_shell cargo build --release -p symthaea-broca --bin broca-train \
        --features "simd,parallel,gpu"
fi

# Reasonable defaults — GPU on a laptop-class 2070-ish card measured at
# ~0.9 pairs/sec on this checkpoint architecture (2026-07-09 calibration
# run), so ~1.8h/epoch on the 5841-pair train-v8 corpus. The older
# "~1.5s/pair" estimate in this comment and CLAUDE.md's separate,
# self-flagged-unverified "10+ pairs/sec" claim were both off -- this is
# the real measured number. Adjust as needed.
DATA="${DATA:-crates/domains/symthaea-broca/data/train-combined-v8.jsonl}"
EVAL="${EVAL:-crates/domains/symthaea-broca/data/eval-creativity-v1.jsonl}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.001}"
OUTPUT="${OUTPUT:-crates/domains/symthaea-broca/data/broca-cfc-v9-gpu.bin}"
PATIENCE="${PATIENCE:-0}"

# Pass any args straight through to broca-train if the caller wants full control.
if [[ $# -gt 0 ]]; then
    exec ./target/release/broca-train "$@"
fi

echo "=== Broca GPU Training ==="
echo "  data:     $DATA"
echo "  eval:     $EVAL"
echo "  epochs:   $EPOCHS"
echo "  lr:       $LR"
echo "  patience: $PATIENCE"
echo "  output:   $OUTPUT"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo

RUST_LOG="${RUST_LOG:-info}" \
    ./target/release/broca-train \
        --data "$DATA" \
        --eval "$EVAL" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --patience "$PATIENCE" \
        --output "$OUTPUT"
