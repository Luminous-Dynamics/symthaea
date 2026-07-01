#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Compare direct Liquid-Mamba generation with the semantic attractor disabled vs
# enabled on the same projection checkpoint and canonical cases.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${BROCA_AB_CHECKPOINT:?BROCA_AB_CHECKPOINT must point to a projection checkpoint}"
OUT_DIR="${BROCA_AB_OUT_DIR:-/tmp/symthaea-broca-attractor-ab}"
CANONICAL="${BROCA_AB_CANONICAL:-crates/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl}"
EVAL_LIMIT="${BROCA_AB_EVAL_LIMIT:-8}"
MAX_GEN_TOKENS="${BROCA_AB_MAX_GEN_TOKENS:-8}"
MIN_NEW_TOKENS="${BROCA_AB_MIN_NEW_TOKENS:-1}"
USE_NIX="${BROCA_AB_USE_NIX:-1}"
NIX_ATTR="${BROCA_AB_NIX_ATTR:-.#broca-gpu}"
TARGET_DIR="${BROCA_AB_TARGET_DIR:-/tmp/symthaea-broca-attractor-ab-target}"
STRENGTH="${BROCA_AB_SEMANTIC_ATTRACTOR_STRENGTH:-0.5}"
TOP_K="${BROCA_AB_TOP_K:-8}"
ATTRACTOR_TOP_K="${BROCA_AB_SEMANTIC_ATTRACTOR_TOP_K:-$TOP_K}"
MAX_ADJUSTMENT="${BROCA_AB_SEMANTIC_ATTRACTOR_MAX_ADJUSTMENT:-1.5}"
NORMALIZE="${BROCA_AB_SEMANTIC_ATTRACTOR_NORMALIZE:-1}"
TEMPERATURE="${BROCA_AB_TEMPERATURE:-0.8}"
SAMPLING_SEED="${BROCA_AB_SAMPLING_SEED:-42}"

mkdir -p "$OUT_DIR"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$TARGET_DIR}"
export RUSTC_WRAPPER="${BROCA_AB_RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

run() {
    if [[ "$USE_NIX" == "1" && -z "${IN_NIX_SHELL:-}" ]]; then
        nix develop "$NIX_ATTR" -c "$@"
    else
        "$@"
    fi
}

run_eval() {
    local label="$1"
    local enabled="$2"
    local report="$OUT_DIR/${label}.json"
    local dump="$OUT_DIR/${label}.jsonl"

    echo "[broca-ab] running $label (semantic_attractor=$enabled)"
    run cargo run -p symthaea-broca --features mamba-cpu --bin broca-liquid-eval -- \
            --checkpoint "$CHECKPOINT" \
            --canonical-eval "$CANONICAL" \
            --eval-limit "$EVAL_LIMIT" \
            --max-gen-tokens "$MAX_GEN_TOKENS" \
            --min-new-tokens "$MIN_NEW_TOKENS" \
            --temperature "$TEMPERATURE" \
            --top-k "$TOP_K" \
            --sampling-seed "$SAMPLING_SEED" \
            --json-out "$report" \
            --dump-generations "$dump" \
            --allow-checkpoint-recovery \
            "$([[ "$enabled" == "1" ]] && printf '%s' "--semantic-attractor" || printf '%s' "--no-semantic-attractor")" \
            "$([[ "$NORMALIZE" == "1" ]] && printf '%s' "--semantic-attractor-normalize" || printf '%s' "--no-semantic-attractor-normalize")" \
            --semantic-attractor-strength "$STRENGTH" \
            --semantic-attractor-top-k "$ATTRACTOR_TOP_K" \
            --semantic-attractor-max-adjustment "$MAX_ADJUSTMENT"
}

run_eval "attractor-off" 0
run_eval "attractor-on" 1

echo "[broca-ab] reports:"
echo "  off: $OUT_DIR/attractor-off.json"
echo "  on:  $OUT_DIR/attractor-on.json"
echo "  dumps include per-token entropy, max probability, delta/B scales, attractor adjustments, and selected semantic alignment."
