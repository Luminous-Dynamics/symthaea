#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# One-command quality gate for the coding-agent stack:
#   1. core code-generation build
#   2. exact SMT proof-cache tests
#   3. registered proof-cache demo
#   4. deterministic coding backend regression
#   5. ForecastBench-style epistemic calibration regression

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUSTC_WRAPPER="${CODING_AGENT_RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

OUT_DIR="${CODING_AGENT_OUT_DIR:-/tmp/symthaea-coding-agent-quality}"
CODING_REPORT="${CODING_AGENT_CODING_REPORT:-$OUT_DIR/benchmark-coding-backends-smoke.json}"
FORECAST_REPORT="${CODING_AGENT_FORECAST_REPORT:-$OUT_DIR/forecastbench-local.json}"
POLICY_REPORT="${CODING_AGENT_POLICY_REPORT:-$OUT_DIR/coding-agent-policy.json}"
mkdir -p "$OUT_DIR"

echo "[coding-agent-quality] cargo check --lib --features code_generation"
cargo check --lib --features code_generation

echo "[coding-agent-quality] proof memory tests"
cargo test proof_memory --lib --features code_generation

echo "[coding-agent-quality] proof caching demo"
cargo check --example demo_proof_caching --features code_generation

if [[ "${CODING_AGENT_SKIP_BACKEND_REGRESSION:-0}" != "1" ]]; then
    echo "[coding-agent-quality] coding backend regression"
    CODING_BACKEND_REPORT="$CODING_REPORT" scripts/coding_backend_regression.sh
else
    echo "[coding-agent-quality] coding backend regression skipped"
fi

if [[ "${CODING_AGENT_SKIP_FORECASTBENCH:-0}" != "1" ]]; then
    echo "[coding-agent-quality] forecastbench regression"
    FORECASTBENCH_REPORT="$FORECAST_REPORT" scripts/forecastbench_regression.sh
else
    echo "[coding-agent-quality] forecastbench regression skipped"
fi

if [[ -f "$CODING_REPORT" && -f "$FORECAST_REPORT" ]]; then
    echo "[coding-agent-quality] deriving routing policy"
    python3 scripts/coding_agent_policy.py \
        --coding-report "$CODING_REPORT" \
        --forecast-report "$FORECAST_REPORT" \
        --output "$POLICY_REPORT"
fi

echo "[coding-agent-quality] PASS"
if [[ -f "$CODING_REPORT" ]]; then
    echo "  coding report: $CODING_REPORT"
fi
if [[ -f "$FORECAST_REPORT" ]]; then
    echo "  forecast report: $FORECAST_REPORT"
fi
if [[ -f "$POLICY_REPORT" ]]; then
    echo "  policy report: $POLICY_REPORT"
fi
