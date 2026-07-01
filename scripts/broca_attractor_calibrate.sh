#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Measure whether the HDC semantic attractor is grounded: target tokens should
# rank above current Mamba distractors for the same thought state.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${BROCA_CAL_CHECKPOINT:?BROCA_CAL_CHECKPOINT must point to a projection checkpoint}"
OUT_DIR="${BROCA_CAL_OUT_DIR:-/tmp/symthaea-broca-attractor-calibrate}"
CANONICAL="${BROCA_CAL_CANONICAL:-crates/symthaea-broca/tests/fixtures/eval-canonical-v1.jsonl}"
EVAL_LIMIT="${BROCA_CAL_EVAL_LIMIT:-0}"
TOP_K="${BROCA_CAL_TOP_K:-32}"
MIN_NEW_TOKENS="${BROCA_CAL_MIN_NEW_TOKENS:-1}"
MIN_POSITIVE_MARGIN_RATE="${BROCA_CAL_MIN_POSITIVE_MARGIN_RATE:-}"
MIN_AVG_ALIGNMENT_MARGIN="${BROCA_CAL_MIN_AVG_ALIGNMENT_MARGIN:-}"
USE_NIX="${BROCA_CAL_USE_NIX:-1}"
NIX_ATTR="${BROCA_CAL_NIX_ATTR:-.#broca-gpu}"
FEATURES="${BROCA_CAL_FEATURES:-mamba}"
TARGET_DIR="${BROCA_CAL_TARGET_DIR:-/tmp/symthaea-broca-attractor-cal-target}"

mkdir -p "$OUT_DIR"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$TARGET_DIR}"
export RUSTC_WRAPPER="${BROCA_CAL_RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

run() {
    if [[ "$USE_NIX" == "1" && -z "${IN_NIX_SHELL:-}" ]]; then
        nix develop "$NIX_ATTR" -c "$@"
    else
        "$@"
    fi
}

echo "[broca-cal] checkpoint: $CHECKPOINT"
echo "[broca-cal] output:     $OUT_DIR"
echo "[broca-cal] features:   $FEATURES"

args=(
    --checkpoint "$CHECKPOINT"
    --canonical-eval "$CANONICAL"
    --eval-limit "$EVAL_LIMIT"
    --top-k "$TOP_K"
    --min-new-tokens "$MIN_NEW_TOKENS"
    --json-out "$OUT_DIR/calibration.json"
    --dump-cases "$OUT_DIR/calibration.jsonl"
    --allow-checkpoint-recovery
)

if [[ -n "$MIN_POSITIVE_MARGIN_RATE" ]]; then
    args+=(--min-positive-margin-rate "$MIN_POSITIVE_MARGIN_RATE")
fi
if [[ -n "$MIN_AVG_ALIGNMENT_MARGIN" ]]; then
    args+=(--min-avg-alignment-margin "$MIN_AVG_ALIGNMENT_MARGIN")
fi

run cargo run -p symthaea-broca --features "$FEATURES" --bin broca-attractor-calibrate -- "${args[@]}"

echo "[broca-cal] report: $OUT_DIR/calibration.json"
echo "[broca-cal] cases:  $OUT_DIR/calibration.jsonl"
