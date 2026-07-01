#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Bounded offline epistemic foraging for Symthaea's coding loop.
#
# This does not commit or rewrite repository files. It runs the coding backend
# benchmark in a deterministic lane, exports repair/prototype memory, then runs
# a second pass with the exported prototypes imported so we can measure whether
# structural/semantic memory improved the loop.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${CODING_FORAGER_OUT_DIR:-$ROOT/target/coding-epistemic-forager}"
LANE="${CODING_FORAGER_LANE:-hard}"
ENERGY_BUDGET="${CODING_FORAGER_ENERGY_BUDGET:-200}"
FEATURES="${CODING_FORAGER_FEATURES:-code_generation,geodesic_synthesis}"

mkdir -p "$OUT_DIR"

BASELINE_REPORT="$OUT_DIR/baseline.json"
FORAGED_REPORT="$OUT_DIR/foraged.json"
DISTILLATION="$OUT_DIR/repair_records.jsonl"
PROTOTYPES="$OUT_DIR/semantic_structural_prototypes.json"

echo "== Symthaea coding epistemic forager =="
echo "lane=$LANE"
echo "energy_budget=$ENERGY_BUDGET"
echo "out_dir=$OUT_DIR"

(
  cd "$ROOT"
  cargo run --example benchmark_coding_backends \
    --features "$FEATURES" \
    -- \
    --json \
    --lane "$LANE" \
    --simulated-llm \
    --energy-budget "$ENERGY_BUDGET" \
    --save-distillation-jsonl "$DISTILLATION" \
    --save-structural-prototypes "$PROTOTYPES" \
    > "$BASELINE_REPORT"
)

(
  cd "$ROOT"
  SYMTHAEA_GEODESIC_REJECTION_SHADOW=1 \
  cargo run --example benchmark_coding_backends \
    --features "$FEATURES" \
    -- \
    --json \
    --lane "$LANE" \
    --simulated-llm \
    --energy-budget "$ENERGY_BUDGET" \
    --load-distillation-jsonl "$DISTILLATION" \
    --load-structural-prototypes "$PROTOTYPES" \
    > "$FORAGED_REPORT"
)

python3 - "$BASELINE_REPORT" "$FORAGED_REPORT" <<'PY'
import json
import sys

before = json.load(open(sys.argv[1]))
after = json.load(open(sys.argv[2]))

def number(doc, key):
    return doc.get(key)

summary = {
    "pass_rate_before": number(before, "pass_rate"),
    "pass_rate_after": number(after, "pass_rate"),
    "quality_pass_rate_before": number(before, "quality_pass_rate"),
    "quality_pass_rate_after": number(after, "quality_pass_rate"),
    "mean_attempts_before": number(before, "mean_attempts_per_task"),
    "mean_attempts_after": number(after, "mean_attempts_per_task"),
    "mean_structural_prior_before": number(before, "mean_structural_prior_score"),
    "mean_structural_prior_after": number(after, "mean_structural_prior_score"),
    "repair_records": number(after, "distillation_imported"),
    "structural_prototype_imported": number(after, "structural_prototype_imported"),
}

print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "reports:"
echo "  $BASELINE_REPORT"
echo "  $FORAGED_REPORT"
echo "  $DISTILLATION"
echo "  $PROTOTYPES"

# Graduation Step: Merge results into the main project memory for immediate lookup
if [[ -d "$ROOT/.symthaea" ]]; then
  echo "Graduating new discoveries to .symthaea/repair_records.jsonl..."
  cat "$DISTILLATION" >> "$ROOT/.symthaea/repair_records.jsonl" 2>/dev/null || true
  # Deduplicate while preserving order (using awk to keep first occurrence)
  if [[ -f "$ROOT/.symthaea/repair_records.jsonl" ]]; then
    awk '!seen[$0]++' "$ROOT/.symthaea/repair_records.jsonl" > "$ROOT/.symthaea/repair_records.jsonl.tmp"
    mv "$ROOT/.symthaea/repair_records.jsonl.tmp" "$ROOT/.symthaea/repair_records.jsonl"
  fi
  echo "Graduation complete."
fi
