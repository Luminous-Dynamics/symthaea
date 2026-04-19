#!/usr/bin/env bash
# 5-variant Φ→gain comparison under POST-FEP-wiring code (commit
# `996750d12b`+). Answers the §5 question: does SprintFloor still match
# Recalibrated to 3 decimal places, now that Φ is FEP-driven?
#
# Each variant runs at the same S_p = 1.0 m, same seeds. The Φ arm
# uses N=10 trials (matches paper's original setup).
#
# Cost: ~3-5 min per variant × 5 ≈ 15-25 min total.

set -u

REPO=/srv/luminous-dynamics
CRATE="$REPO/symtropy/crates/symtropy-manipulator-demo"
BIN="$CRATE/target/release/examples/manipulator_benchmark"
OUTDIR=$REPO/symtropy/papers/phi-gated-safety/data/five_variant_post_wiring

mkdir -p "$OUTDIR"

run_variant() {
    local name=$1
    local env_var=$2
    echo "=== starting variant: $name ==="
    eval "$env_var MANIP_BENCH_PHI=1 MANIP_BENCH_ISO_SP=1.0 \
          MANIP_BENCH_TRIALS=30 MANIP_BENCH_PHI_TRIALS=10 \
          MANIP_BENCH_PHI_STEPS=50000 \
          \"$BIN\" > \"$OUTDIR/${name}.txt\" 2>&1"
    echo "done $name (exit $?, $(wc -l < $OUTDIR/${name}.txt) lines)"
}

run_variant "default"      ""
run_variant "continuous"   "MANIP_BENCH_PHI_CONT=1"
run_variant "clamped"      "MANIP_BENCH_PHI_CLAMP=1"
run_variant "recalibrated" "MANIP_BENCH_PHI_RECAL=1"
run_variant "sprint_floor" "MANIP_BENCH_PHI_SPRINT=1"

echo "=== 5 variants complete ==="
echo ""
echo "─── summary (mean ± std, advantage vs ISO) ───"
for v in default continuous clamped recalibrated sprint_floor; do
    f="$OUTDIR/${v}.txt"
    [ -f "$f" ] || continue
    phi=$(grep "Φ-gated cycles" "$f" | head -1)
    adv=$(grep "Φ vs ISO" "$f" | head -1)
    echo "  $v: $phi"
    echo "    $adv"
done
