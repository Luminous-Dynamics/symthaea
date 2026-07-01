#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Deterministic epistemic calibration lane. This is local-first and
# ForecastBench-compatible: official/live questions can be converted to the same
# JSONL shape, while unresolved questions are excluded from proper scoring.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUSTC_WRAPPER="${FORECASTBENCH_RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

OUT_DIR="${FORECASTBENCH_OUT_DIR:-/tmp/symthaea-forecastbench}"
INPUT="${FORECASTBENCH_INPUT:-tests/fixtures/forecastbench_local.jsonl}"
BASELINE="${FORECASTBENCH_BASELINE:-tests/fixtures/forecastbench_baseline.json}"
REPORT="${FORECASTBENCH_REPORT:-$OUT_DIR/forecastbench-local.json}"
MAX_BRIER="${FORECASTBENCH_MAX_BRIER:-0.25}"
MAX_ECE="${FORECASTBENCH_MAX_ECE:-0.20}"
MIN_RESOLVED="${FORECASTBENCH_MIN_RESOLVED:-10}"
SOURCE="${FORECASTBENCH_SOURCE:-local_coding_forecastbench}"

mkdir -p "$OUT_DIR"

echo "[forecastbench] checking evaluator"
cargo check --example forecastbench_eval

echo "[forecastbench] running local calibration lane"
cargo run --example forecastbench_eval -- \
    --input "$INPUT" \
    --json \
    --source "$SOURCE" \
    --max-brier "$MAX_BRIER" \
    --max-ece "$MAX_ECE" \
    --min-resolved "$MIN_RESOLVED" \
    > "$REPORT"

echo "[forecastbench] checking baseline"
python3 scripts/check_forecastbench_baseline.py \
    --baseline "$BASELINE" \
    --report "$REPORT"

echo "[forecastbench] PASS"
echo "  report: $REPORT"
