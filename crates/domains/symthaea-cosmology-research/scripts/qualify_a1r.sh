#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: qualify_a1r.sh A0_ARTIFACT_DIR EVIDENCE_DIR" >&2
  exit 2
fi

A0_DIR="$1"
OUT="$2"
ROOT="$(git rev-parse --show-toplevel)"
CRATE="$ROOT/crates/domains/symthaea-cosmology-research"
BIN_DIR="${DE001_BIN_DIR:-$ROOT/target/release}"
REF="$CRATE/references"

mkdir -p "$OUT"

for binary in \
  de001a-a0-verify \
  de001a-a1p-point-bind \
  de001a-a1r-oracle \
  de001a-a1n-convergence
do
  test -x "$BIN_DIR/$binary" || {
    echo "missing executable $BIN_DIR/$binary" >&2
    exit 2
  }
done

for role in \
  dataset-mean \
  dataset-covariance \
  likelihood-definition \
  reference-input-configuration \
  reference-expanded-configuration \
  reference-minimizer-configuration \
  reference-bestfit-text \
  reference-bestfit-getdist
do
  test -f "$A0_DIR/$role" || {
    echo "missing A0 artifact $A0_DIR/$role" >&2
    exit 2
  }
done

run_stage() {
  local label="$1"
  shift
  local json="$OUT/$label.json"
  local stderr="$OUT/$label.stderr"
  set +e
  "$@" >"$json" 2>"$stderr"
  local code=$?
  set -e
  printf '%s\n' "$code" >"$OUT/$label.exit"
  return 0
}

run_stage a0 \
  "$BIN_DIR/de001a-a0-verify" \
  "$REF/de001a_a0_artifacts_v1.json" \
  "dataset-mean=$A0_DIR/dataset-mean" \
  "dataset-covariance=$A0_DIR/dataset-covariance" \
  "likelihood-definition=$A0_DIR/likelihood-definition" \
  "reference-input-configuration=$A0_DIR/reference-input-configuration" \
  "reference-expanded-configuration=$A0_DIR/reference-expanded-configuration" \
  "reference-minimizer-configuration=$A0_DIR/reference-minimizer-configuration" \
  "reference-bestfit-text=$A0_DIR/reference-bestfit-text" \
  "reference-bestfit-getdist=$A0_DIR/reference-bestfit-getdist"

A0_EXIT="$(cat "$OUT/a0.exit")"
if [[ "$A0_EXIT" -ne 0 ]]; then
  echo "A0 integrity did not PASS (exit=$A0_EXIT)" >&2
  exit 2
fi

run_stage a1p \
  "$BIN_DIR/de001a-a1p-point-bind" \
  "$REF/de001a_a1r_oracle_v1.json" \
  "$OUT/a0.json" \
  "$A0_DIR/reference-bestfit-text"

A1P_EXIT="$(cat "$OUT/a1p.exit")"
if [[ "$A1P_EXIT" -ne 0 ]]; then
  echo "A1P parameter provenance did not PASS (exit=$A1P_EXIT)" >&2
  exit 2
fi

run_stage a1r-primary \
  "$BIN_DIR/de001a-a1r-oracle" \
  "$REF/de001a_a1r_oracle_v1.json" \
  "$OUT/a0.json" \
  "$A0_DIR/dataset-mean" \
  "$A0_DIR/dataset-covariance"

run_stage a1r-refined \
  "$BIN_DIR/de001a-a1r-oracle" \
  "$REF/de001a_a1r_oracle_refined_v1.json" \
  "$OUT/a0.json" \
  "$A0_DIR/dataset-mean" \
  "$A0_DIR/dataset-covariance"

PRIMARY_EXIT="$(cat "$OUT/a1r-primary.exit")"
REFINED_EXIT="$(cat "$OUT/a1r-refined.exit")"
if [[ "$PRIMARY_EXIT" -gt 1 || "$REFINED_EXIT" -gt 1 ]]; then
  echo "A1R execution was INVALID (primary=$PRIMARY_EXIT refined=$REFINED_EXIT)" >&2
  exit 2
fi

run_stage a1n \
  "$BIN_DIR/de001a-a1n-convergence" \
  "$REF/de001a_a1n_convergence_v1.json" \
  "$REF/de001a_a1r_oracle_v1.json" \
  "$REF/de001a_a1r_oracle_refined_v1.json" \
  "$OUT/a1r-primary.json" \
  "$OUT/a1r-refined.json"

A1N_EXIT="$(cat "$OUT/a1n.exit")"
if [[ "$A1N_EXIT" -ne 0 ]]; then
  echo "A1N numerical convergence did not PASS (exit=$A1N_EXIT)" >&2
  exit 2
fi

# A valid NEGATIVE reproduction remains evidence, so convergence is still
# retained above. The script returns 1 only after all integrity/provenance/
# convergence gates have succeeded.
if [[ "$PRIMARY_EXIT" -eq 1 || "$REFINED_EXIT" -eq 1 ]]; then
  echo "A1R fixed-point reproduction is valid but NEGATIVE" >&2
  exit 1
fi

exit 0
