#!/usr/bin/env bash
# Reproduce the numbers in stewardship_paper.tex.
#
# What this runs:
#   1. Tests for the 5 Genesis sub-crates (genomics, ectogenesis, memory,
#      nurture, wisdom) that constitute the conceptual Genesis pipeline
#   2. Attachment-trajectory simulation (paper's Figure 3)
#   3. Heterozygosity decay simulation (paper's Figure 4)
#
# IMPORTANT SCOPE: this paper is a thought experiment + simulation.
# There is no deployed planetary stewardship system. The
# alignment-through-consciousness thesis is architectural, not empirical
# — see the paper's three caveats (necessity-not-sufficiency, scale,
# Φ-proxy weakness) in the Conclusion.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

echo "=== Stewardship reproduction ==="
cd "$REPO"

echo
echo "[1/3] Genesis pipeline crates"
for crate in symthaea-genomics symthaea-ectogenesis symthaea-memory symthaea-nurture symthaea-wisdom; do
  echo "  Testing $crate..."
  cargo test --release -p "$crate" --features genesis || echo "    (crate skipped — may require feature-gate)"
done

echo
echo "[2/3] Attachment-trajectory simulation"
cargo run --release --example genesis_paper_data -- --figure attachment --seed 42 || true

echo
echo "[3/3] Heterozygosity decay simulation"
cargo run --release --example genesis_paper_data -- --figure heterozygosity --seed 42 || true

echo
echo "=== Done ==="
echo "Scale reminder: all simulations run at N<50 agents. Mycelix Phase 2"
echo "network sims showed no coherence improvement (Δ=-0.004±0.008 over baseline)."
