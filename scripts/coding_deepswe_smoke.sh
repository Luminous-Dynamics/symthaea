#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# DeepSWE integration smoke.
#
# Default mode is offline: it verifies the coding benchmark still compiles with
# the DeepSWE provider path present, but does not contact a model endpoint.
# Set DEEPSWE_LIVE=1 with DEEPSWE_BASE_URL/DEEPSWE_API_KEY to run a bounded live
# coding benchmark through the OpenAI-compatible DeepSWE adapter.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${DEEPSWE_SMOKE_OUT_DIR:-/tmp/symthaea-deepswe-smoke}"
REPORT="$OUT_DIR/deepswe-coding-smoke.json"
LANE="${DEEPSWE_SMOKE_LANE:-smoke}"
ENERGY_BUDGET="${DEEPSWE_SMOKE_ENERGY_BUDGET:-64}"
mkdir -p "$OUT_DIR"

export RUSTC_WRAPPER="${DEEPSWE_SMOKE_RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

echo "[deepswe-smoke] checking coding benchmark and provider route"
cargo check --example benchmark_coding_backends --features code_generation,geodesic_synthesis

if [[ "${DEEPSWE_LIVE:-0}" != "1" ]]; then
    cat > "$REPORT" <<JSON
{
  "schema_version": 1,
  "evidence_level": "offline-provider-smoke",
  "measured": false,
  "provider": "deepswe",
  "lane": "$LANE",
  "note": "Set DEEPSWE_LIVE=1 plus DEEPSWE_BASE_URL/DEEPSWE_API_KEY to run a live benchmark."
}
JSON
    echo "[deepswe-smoke] offline PASS"
    echo "  report: $REPORT"
    exit 0
fi

echo "[deepswe-smoke] running live DeepSWE coding benchmark"
SYMTHAEA_LLM_PROVIDER=deepswe \
CODING_BACKEND_LANE="$LANE" \
CODING_BACKEND_REPORT="$REPORT" \
CODING_BACKEND_ENERGY_BUDGET="$ENERGY_BUDGET" \
CODING_BACKEND_SKIP_BASELINE="${DEEPSWE_SMOKE_SKIP_BASELINE:-1}" \
scripts/coding_backend_regression.sh

echo "[deepswe-smoke] PASS"
echo "  report: $REPORT"
