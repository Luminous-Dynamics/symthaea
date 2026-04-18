#!/usr/bin/env bash
# Reproduce the numbers in hai_paper.tex.
#
# What this runs:
#   1. Core HAI active-inference unit tests (prediction-error, precision dynamics)
#   2. T-Maze + Grid World benchmarks (§4.2 of the paper)
#   3. ETHICS moral reasoning (§4.1 — 92.9% accuracy claim)
#   4. Butlin 14-indicator runtime evaluation (§5.1)
#
# Expected run time: ~10–20 minutes.
# See papers/theory-foundations/hai-consciousness/hai_paper.tex §4 for expected values.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../" && pwd)"

echo "=== HAI reproduction ==="
cd "$REPO"

echo
echo "[1/4] Active-inference unit tests"
cargo test --release --lib test_fep_active_inference

echo
echo "[2/4] T-Maze + Grid World"
cargo test --release --lib t_maze
cargo test --release -p symthaea-core gridworld

echo
echo "[3/4] ETHICS moral reasoning (expect ~92.9%, in-distribution)"
cargo run --release --example benchmark_moral_unified

echo
echo "[4/4] Butlin 14-indicator runtime evaluation"
cargo test --release --lib butlin_runtime

echo
echo "=== Done. See paper §4 and §5 for expected values ==="
echo "Note: ETHICS 94.5% is IN-DISTRIBUTION on trained per-category classifiers,"
echo "not zero-shot moral reasoning (see paper §4.1 for caveat)."
