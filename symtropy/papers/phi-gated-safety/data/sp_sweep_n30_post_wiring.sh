#!/usr/bin/env bash
# N=30 S_p sweep under POST-FEP-wiring code (commit `996750d12b`+).
#
# Same structure as sp_sweep_n30.sh (ISO_SP across {0.5, 1.0, 2.0, 2.25, 2.5, 3.0}
# with N=30 Adaptive/ISO trials + N=10 Φ-SprintFloor trials per point) but
# writes to a distinct dir so the pre-wiring sweep remains as a
# historical reference for the paper's §9.2 post-wiring update.
#
# Prereq: fresh `cargo build --release --example manipulator_benchmark`
# under current code — the binary at target/release/examples/
# manipulator_benchmark must be built from commit `996750d12b`+.
#
# Cost: ~10-12 min per S_p × 6 points ≈ 60-70 min total wall.

set -u

REPO=/srv/luminous-dynamics
CRATE="$REPO/symtropy/crates/symtropy-manipulator-demo"
BIN="$CRATE/target/release/examples/manipulator_benchmark"
OUTDIR=$REPO/symtropy/papers/phi-gated-safety/data/sp_sweep_n30_post_wiring

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
