#!/usr/bin/env bash
# Reproduce the numbers in stochastic_resonance.tex.
#
# What this runs:
#   1. Anesthesia benchmark (6 states, True Φ ordering)
#   2. Noise sensitivity sweep (∂Φ/∂σ, expected ≈ +0.341)
#   3. Coupling sensitivity sweep (∂Φ/∂c, expected ≈ +0.367)
#
# Expected result: monotonic Φ ordering across 6 anesthesia states,
# with the Light→Moderate transition at p=0.024 (the one marginal pair
# named in the paper's honest-caveat paragraph).
#
# Expected run time: ~5–10 minutes.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../" && pwd)"

echo "=== Stochastic Resonance reproduction ==="
cd "$REPO"

echo
echo "[1/3] Anesthesia Φ ordering (100 runs per state)"
cargo run --release --example resonance_discovery -- --mode anesthesia --n-runs 100

echo
echo "[2/3] Noise sensitivity sweep (∂Φ/∂σ)"
cargo run --release --example resonance_discovery -- --mode noise-sweep --seed 42

echo
echo "[3/3] Coupling sensitivity sweep (∂Φ/∂c)"
cargo run --release --example resonance_discovery -- --mode coupling-sweep --seed 42

echo
echo "=== Done ==="
echo "Expected sensitivity: ∂Φ/∂noise = +0.341 [+0.308, +0.374]"
echo "                      ∂Φ/∂coupling = +0.367 [+0.342, +0.391]"
echo "Note: Φ uses sampled-partition approximation (see paper §3.2 caveat)."
