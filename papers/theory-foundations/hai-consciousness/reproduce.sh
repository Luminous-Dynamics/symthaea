#!/usr/bin/env bash
# Reproduce the numbers in hai_paper.tex.
#
# What this runs:
#   1. Core HAI active-inference unit tests (prediction-error, precision dynamics)
#   2. T-Maze + Grid World benchmarks (§4.2 of the paper)
#   3. Historical ETHICS path; previous aggregate claims are unqualified (see audit)
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
echo "[3/4] Task-matched ETHICS commonsense diagnostic (no full-suite score claimed)"
cargo run --release --example ethics_commonsense_holdout

echo
echo "[4/4] Butlin 14-indicator runtime evaluation"
cargo test --release --lib butlin_runtime

echo
echo "=== Done. See paper §4 and §5 for expected values ==="
echo "The prior 92.9%/94.5% aggregate headlines are not qualified evidence."
echo "See docs/ETHICS_TASK_SEMANTICS_AUDIT_2026-10-11.md for task-specific score requirements."
