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

# Build if missing or stale (sccache makes repeat builds instant).
if ! test -x target/release/broca-train; then
    echo "Building broca-train with gpu feature..."
    cargo build --release -p symthaea-broca --bin broca-train \
        --features "simd,parallel,gpu"
fi

# Reasonable defaults — GPU on a laptop-class 2070-ish card is ~1.5s/pair on
# this checkpoint architecture, so ~2.4h/epoch on the 5841-pair train-v8
# corpus. Adjust as needed.
DATA="${DATA:-crates/symthaea-broca/data/train-combined-v8.jsonl}"
EVAL="${EVAL:-crates/symthaea-broca/data/eval-epistemic-v1.jsonl}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.001}"
OUTPUT="${OUTPUT:-crates/symthaea-broca/data/broca-cfc-v9-gpu.bin}"

# Pass any args straight through to broca-train if the caller wants full control.
if [[ $# -gt 0 ]]; then
    exec ./target/release/broca-train "$@"
fi

echo "=== Broca GPU Training ==="
echo "  data:    $DATA"
echo "  eval:    $EVAL"
echo "  epochs:  $EPOCHS"
echo "  lr:      $LR"
echo "  output:  $OUTPUT"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo

RUST_LOG="${RUST_LOG:-info}" \
    ./target/release/broca-train \
        --data "$DATA" \
        --eval "$EVAL" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --output "$OUTPUT"
