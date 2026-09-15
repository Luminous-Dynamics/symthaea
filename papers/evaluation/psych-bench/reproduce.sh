#!/usr/bin/env bash
# Reproduce the executable surfaces used by the Psych-Bench paper workflow.
#
# What this runs:
#   1. Full core benchmark runner under its current default configuration,
#      writing the runner's supported JSON snapshot output.
#   2. Full paper CSV generator: profile, normative, ablation, SAT, reliability,
#      correlations, neuromodulator profiles, and neuromodulator curves.
#   3. Multi-seed robustness over its implemented fixed seed set
#      (42, 123, 456, 789, 1024), captured as Markdown.
#   4. Seven-benchmark Qualia Confidence Matrix at seed 42, captured as JSON.
#
# This script orchestrates the current executables. It does not itself assert
# that newly generated values match historical paper claims.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../" && pwd)"

echo "=== Psych-Bench reproduction ==="
echo "Repo root: $REPO"

cd "$REPO"

echo
echo "[1/4] Full core benchmark runner (default configuration)"
cargo run --release --example run_psych_benchmarks \
  --package symthaea-psych-bench -- --json-output "$HERE/out_default.json"

echo
echo "[2/4] Full paper CSV generator (includes reliability and neuromodulator outputs)"
cargo run --release --example psych_bench_paper_data \
  --package symthaea-psych-bench

echo
echo "[3/4] Multi-seed robustness (42, 123, 456, 789, 1024)"
cargo run --release --example multi_seed_robustness \
  --package symthaea-psych-bench > "$HERE/out_stability.md"

echo
echo "[4/4] Seven-benchmark Qualia Confidence Matrix (seed 42)"
cargo run --release --example qualia_confidence_report \
  --package symthaea-psych-bench -- --seed 42 --json > "$HERE/out_qualia_confidence.json"

echo
echo "=== Done ==="
echo "Paper CSV outputs: $REPO/papers/data/psych_bench/"
echo "Core runner JSON: $HERE/out_default.json"
echo "Multi-seed Markdown: $HERE/out_stability.md"
echo "Qualia Confidence JSON: $HERE/out_qualia_confidence.json"
