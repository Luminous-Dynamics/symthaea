#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Deterministic coding-backend regression lane:
#   compile the benchmark example, run it with offline LLM fallback, and compare
#   the JSON report against the checked-in conservative baseline.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUSTC_WRAPPER="${CODING_BACKEND_RUSTC_WRAPPER:-}"
export SCCACHE_DISABLE="${SCCACHE_DISABLE:-1}"

OUT_DIR="${CODING_BACKEND_OUT_DIR:-/tmp/symthaea-coding-backend}"
LANE="${CODING_BACKEND_LANE:-smoke}"
REPORT="${CODING_BACKEND_REPORT:-$OUT_DIR/benchmark-coding-backends-$LANE.json}"
if [[ -n "${CODING_BACKEND_BASELINE:-}" ]]; then
    BASELINE="$CODING_BACKEND_BASELINE"
elif [[ "$LANE" == "hard" ]]; then
    BASELINE="tests/fixtures/coding_backends_hard_baseline.json"
elif [[ "$LANE" == "repair" ]]; then
    BASELINE="tests/fixtures/coding_backends_repair_baseline.json"
elif [[ "$LANE" == "frontier" ]]; then
    BASELINE="tests/fixtures/coding_backends_frontier_baseline.json"
else
    BASELINE="tests/fixtures/coding_backends_baseline.json"
fi
ENERGY_BUDGET="${CODING_BACKEND_ENERGY_BUDGET:-256}"
REPAIR_LESSONS="${CODING_BACKEND_REPAIR_LESSONS:-}"
BROCA_REPAIR_TRAINING="${CODING_BACKEND_BROCA_REPAIR_TRAINING:-}"
DISTILLATION_IN="${CODING_BACKEND_DISTILLATION_IN:-}"
DISTILLATION_OUT="${CODING_BACKEND_DISTILLATION_OUT:-}"
STRUCTURAL_PROTOTYPES_IN="${CODING_BACKEND_STRUCTURAL_PROTOTYPES_IN:-}"
STRUCTURAL_PROTOTYPES_OUT="${CODING_BACKEND_STRUCTURAL_PROTOTYPES_OUT:-}"

if [[ "$LANE" == "repair" || "$LANE" == "all" ]]; then
    export SYMTHAEA_ENABLE_FORCED_REPAIR_BENCH=1
    REPAIR_LESSONS="${REPAIR_LESSONS:-$OUT_DIR/repair-lessons-$LANE.jsonl}"
    BROCA_REPAIR_TRAINING="${BROCA_REPAIR_TRAINING:-$OUT_DIR/broca-repair-training-$LANE.jsonl}"
fi

mkdir -p "$OUT_DIR"

echo "[coding-backend] checking benchmark example"
cargo check --example benchmark_coding_backends --features code_generation,geodesic_synthesis

echo "[coding-backend] running deterministic benchmark"
benchmark_args=(--json --simulated-llm --energy-budget "$ENERGY_BUDGET" --lane "$LANE")
if [[ -n "$REPAIR_LESSONS" ]]; then
    benchmark_args+=(--repair-lessons-jsonl "$REPAIR_LESSONS")
fi
if [[ -n "$DISTILLATION_IN" ]]; then
    benchmark_args+=(--load-distillation-jsonl "$DISTILLATION_IN")
fi
if [[ -n "$DISTILLATION_OUT" ]]; then
    benchmark_args+=(--save-distillation-jsonl "$DISTILLATION_OUT")
fi
if [[ -n "$STRUCTURAL_PROTOTYPES_IN" ]]; then
    benchmark_args+=(--load-structural-prototypes "$STRUCTURAL_PROTOTYPES_IN")
fi
if [[ -n "$STRUCTURAL_PROTOTYPES_OUT" ]]; then
    benchmark_args+=(--save-structural-prototypes "$STRUCTURAL_PROTOTYPES_OUT")
fi
cargo run --example benchmark_coding_backends \
    --features code_generation,geodesic_synthesis \
    -- "${benchmark_args[@]}" \
    > "$REPORT"

echo "[coding-backend] checking baseline"
if [[ "${CODING_BACKEND_SKIP_BASELINE:-0}" == "1" ]]; then
    echo "[coding-backend] baseline check skipped"
else
    python3 scripts/check_coding_backend_baseline.py \
        --baseline "$BASELINE" \
        --report "$REPORT"
fi

if [[ -n "$REPAIR_LESSONS" && -n "$BROCA_REPAIR_TRAINING" ]]; then
    echo "[coding-backend] deriving Broca repair training data"
    python3 scripts/ingest_distillation_to_broca.py \
        --skip-input \
        --repair-lessons "$REPAIR_LESSONS" \
        --output "$BROCA_REPAIR_TRAINING"
fi

echo "[coding-backend] PASS"
echo "  report: $REPORT"
if [[ -n "$REPAIR_LESSONS" ]]; then
    echo "  repair lessons: $REPAIR_LESSONS"
fi
if [[ -n "$BROCA_REPAIR_TRAINING" ]]; then
    echo "  broca repair training: $BROCA_REPAIR_TRAINING"
fi
if [[ -n "$DISTILLATION_IN" ]]; then
    echo "  distillation in: $DISTILLATION_IN"
fi
if [[ -n "$DISTILLATION_OUT" ]]; then
    echo "  distillation out: $DISTILLATION_OUT"
fi
if [[ -n "$STRUCTURAL_PROTOTYPES_IN" ]]; then
    echo "  structural prototypes in: $STRUCTURAL_PROTOTYPES_IN"
fi
if [[ -n "$STRUCTURAL_PROTOTYPES_OUT" ]]; then
    echo "  structural prototypes out: $STRUCTURAL_PROTOTYPES_OUT"
fi
