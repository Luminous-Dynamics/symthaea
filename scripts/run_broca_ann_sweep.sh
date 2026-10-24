#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run a source-bound multi-size ANN sweep. This script intentionally does not
# modify production defaults or infer a pass from skipped CI workflows.
#
# Override knobs with environment variables, e.g.:
#   SIZES="1000 10000" QUERIES=400 ROUNDS=7 ./scripts/run_broca_ann_sweep.sh

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "refusing to benchmark a tracked dirty worktree" >&2
  exit 2
fi

SOURCE_SHA="$(git rev-parse HEAD)"
SIZES_TEXT="${SIZES:-1000 10000 50000}"
read -r -a SIZES_ARRAY <<< "$SIZES_TEXT"
QUERIES="${QUERIES:-200}"
ROUNDS="${ROUNDS:-5}"
TOP_K="${TOP_K:-10}"
CLUSTERS="${CLUSTERS:-64}"
CLUSTER_NOISE_BITS="${CLUSTER_NOISE_BITS:-1024}"
NOISE_BITS="${NOISE_BITS:-128}"
HARD_NOISE_BITS="${HARD_NOISE_BITS:-4096}"
SEED="${SEED:-42}"
LSH_CONFIGS="${LSH_CONFIGS:-32x8,16x16,64x4}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-4}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/broca-ann}"
mkdir -p "$OUT_DIR"

if [[ ${#SIZES_ARRAY[@]} -eq 0 ]]; then
  echo "SIZES must contain at least one positive store size" >&2
  exit 2
fi
for value in "$QUERIES" "$ROUNDS" "$TOP_K" "$CLUSTERS" "$CLUSTER_NOISE_BITS" "$NOISE_BITS" "$HARD_NOISE_BITS" "$CANDIDATE_MULTIPLIER"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "numeric sweep settings must be non-negative integers; got '$value'" >&2
    exit 2
  fi
done

for SIZE in "${SIZES_ARRAY[@]}"; do
  if [[ ! "$SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "each SIZES entry must be a positive integer; got '$SIZE'" >&2
    exit 2
  fi
  if (( CLUSTERS > SIZE )); then
    echo "CLUSTERS ($CLUSTERS) must not exceed SIZE ($SIZE)" >&2
    exit 2
  fi
done

# Build once; each run still pins and verifies the exact HEAD SHA.
cargo build --release -p symthaea-hdc-store --example ann_benchmark
BIN="$ROOT/target/release/examples/ann_benchmark"
if [[ ! -x "$BIN" ]]; then
  echo "expected benchmark binary was not produced: $BIN" >&2
  exit 3
fi

write_environment_receipt() {
  local path="$1"
  {
    echo "source_sha=$SOURCE_SHA"
    echo "git_branch=$(git branch --show-current)"
    echo "command=$*"
    echo "rustc:"
    rustc -Vv
    echo "cargo=$(cargo --version)"
    echo "os=$(uname -a)"
    if command -v lscpu >/dev/null 2>&1; then
      echo "cpu:"
      lscpu
    fi
  } > "$path"
}

for SIZE in "${SIZES_ARRAY[@]}"; do
  for FALLBACK in true false; do
    STEM="broca-ann-sha-${SOURCE_SHA}-n${SIZE}-q${QUERIES}-r${ROUNDS}-k${TOP_K}-seed${SEED}-empty-fallback-${FALLBACK}"
    OUTPUT="$OUT_DIR/$STEM.csv"
    RECEIPT="$OUT_DIR/$STEM.receipt.txt"
    ARGS=(
      "--source-sha=$SOURCE_SHA"
      "--size=$SIZE"
      "--clusters=$CLUSTERS"
      "--cluster-noise-bits=$CLUSTER_NOISE_BITS"
      "--noise-bits=$NOISE_BITS"
      "--hard-noise-bits=$HARD_NOISE_BITS"
      "--queries=$QUERIES"
      "--rounds=$ROUNDS"
      "--top-k=$TOP_K"
      "--seed=$SEED"
      "--candidate-multiplier=$CANDIDATE_MULTIPLIER"
      "--fallback-on-empty=$FALLBACK"
      "--lsh-configs=$LSH_CONFIGS"
    )
    printf 'Running n=%s fallback_on_empty=%s source=%s\n' "$SIZE" "$FALLBACK" "$SOURCE_SHA"
    write_environment_receipt "$RECEIPT" "$BIN" "${ARGS[@]}"
    set +e
    "$BIN" "${ARGS[@]}" > "$OUTPUT"
    STATUS=$?
    set -e
    {
      echo "exit_status=$STATUS"
      echo "output=$OUTPUT"
      if command -v sha256sum >/dev/null 2>&1 && [[ -f "$OUTPUT" ]]; then
        echo "output_sha256=$(sha256sum "$OUTPUT" | cut -d' ' -f1)"
      fi
    } >> "$RECEIPT"
    if (( STATUS != 0 )); then
      echo "benchmark failed (exit $STATUS); receipt retained at $RECEIPT" >&2
      exit "$STATUS"
    fi
    echo "wrote $OUTPUT"
  done
done

echo "Sweep complete. Preserve CSVs, receipts, exact source SHA, and full command output together."
