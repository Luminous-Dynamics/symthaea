#!/usr/bin/env bash
# Reproduce the numbers in psych_bench_paper.tex.
#
# What this runs:
#   1. Full 128-benchmark core harness → domain z-scores + grand mean
#   2. 17-benchmark neuromodulator evaluation framework
#   3. Multi-seed stability sweep (seeds 42, 123, 789)
#   4. Psychometric reliability (ICC) for the 19 benchmarks that support it
#
# Expected run time: ~15–30 minutes on a modern workstation.
# Expected result on a clean main: grand mean z = +1.261, SD = 0.019 across 3 seeds,
#   143-of-145 benchmarks finish (2 known-hang skipped), 5/19 ICC ≥ 0.50.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../" && pwd)"

echo "=== Psych-Bench reproduction ==="
echo "Repo root: $REPO"

cd "$REPO"

echo
echo "[1/4] Full 128-benchmark core harness (seed 42)"
cargo run --release --example run_psych_benchmarks \
  --package symthaea-psych-bench -- --seed 42 --out "$HERE/out_seed42.json"

echo
echo "[2/4] 17-benchmark neuromodulator framework"
cargo run --release --example psych_bench_paper_data \
  --package symthaea-psych-bench -- --neuromod-only --out "$HERE/out_neuromod.json"

echo
echo "[3/4] Multi-seed stability (42, 123, 789)"
cargo run --release --example multi_seed_robustness \
  --package symthaea-psych-bench -- --seeds 42,123,789 --out "$HERE/out_stability.json"

echo
echo "[4/4] Psychometric reliability (ICC)"
cargo run --release --example qualia_confidence_report \
  --package symthaea-psych-bench -- --out "$HERE/out_icc.json"

echo
echo "=== Done. Outputs in $HERE/out_*.json ==="
echo "Compare grand-mean z to paper Table (section 5.1): expected +1.261 ± 0.019"
