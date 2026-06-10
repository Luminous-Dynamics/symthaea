#!/usr/bin/env bash
# Reproduce the numbers in epistemic_gating.tex.
#
# What this runs:
#   1. Broca pipeline tests (27,965 LOC, 403 tests)
#   2. EpistemicGate hallucination-prevention benchmark
#      (expected: 23.4% → 0.4% on 500 in-distribution generation episodes)
#   3. Epistemic Cube ablation (43 channels vs 20 legacy)
#
# Expected run time: ~20–40 minutes (Broca pipeline is heavy).
# NOT INCLUDED: adversarial red-team evaluation (jailbreak probes) — flagged
# as a revision task in the paper's Limitations section.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../" && pwd)"

echo "=== Epistemic Gating reproduction ==="
cd "$REPO"

echo
echo "[1/3] Broca pipeline tests (403 tests)"
cargo test --release -p symthaea-broca --features ssm_language

echo
echo "[2/3] Hallucination-prevention benchmark"
cargo run --release --example soak_broca_language -- --episodes 500 --seed 42

echo
echo "[3/3] Epistemic Cube ablation (43 vs 20 channels)"
cargo test --release -p symthaea-broca epistemic_cube_ablation

echo
echo "=== Done. Expected: 23.4% → 0.4% hallucination reduction in-distribution ==="
echo "See paper §4 Limitations for adversarial-robustness caveat."
