#!/usr/bin/env bash
# Runs the S_p sweep at N=30 trials per point. Writes one log per S_p.
# Each point runs Adaptive + ISO + Φ-SprintFloor (opt-in).
# Cost: ~10-12 min per S_p × 6 S_p points ≈ 60-70 min total.

set -u

cd "$(dirname "$0")/../.."  # up to symtropy/papers/ (wait — let's use abs paths)

REPO=/srv/luminous-dynamics
CRATE="$REPO/symtropy/crates/symtropy-manipulator-demo"
BIN="$CRATE/target/release/examples/manipulator_benchmark"
OUTDIR=/srv/luminous-dynamics/symtropy/papers/phi-gated-safety/data/sp_sweep_n30

mkdir -p "$OUTDIR"

for sp in 0.5 1.0 2.0 2.25 2.5 3.0; do
    echo "=== starting S_p = $sp ==="
    MANIP_BENCH_ISO_SP="$sp" \
    MANIP_BENCH_PHI=1 \
    MANIP_BENCH_TRIALS=30 \
    MANIP_BENCH_PHI_TRIALS=10 \
    MANIP_BENCH_PHI_STEPS=50000 \
    MANIP_BENCH_PHI_SPRINT=1 \
        "$BIN" > "$OUTDIR/sp_${sp}.txt" 2>&1
    echo "done S_p = $sp (exit $?, $(wc -l < $OUTDIR/sp_${sp}.txt) lines)"
done

echo "=== all 6 points complete ==="
