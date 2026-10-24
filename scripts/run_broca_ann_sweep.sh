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
SEEDS_TEXT="${SEEDS:-42 43 44}"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"
QUERIES="${QUERIES:-200}"
ROUNDS="${ROUNDS:-5}"
TOP_K="${TOP_K:-10}"
CLUSTERS="${CLUSTERS:-64}"
CLUSTER_NOISE_BITS="${CLUSTER_NOISE_BITS:-1024}"
NOISE_BITS="${NOISE_BITS:-128}"
HARD_NOISE_BITS="${HARD_NOISE_BITS:-4096}"
LSH_CONFIGS="${LSH_CONFIGS:-32x8,16x16,64x4}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-4}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/broca-ann}"
mkdir -p "$OUT_DIR"

if [[ ${#SIZES_ARRAY[@]} -eq 0 ]]; then
  echo "SIZES must contain at least one positive store size" >&2
  exit 2
fi
if [[ ${#SEEDS_ARRAY[@]} -eq 0 ]]; then
  echo "SEEDS must contain at least one non-negative integer" >&2
  exit 2
fi
declare -A SEEN_SEEDS=()
for seed_value in "${SEEDS_ARRAY[@]}"; do
  if [[ ! "$seed_value" =~ ^[0-9]+$ ]]; then
    echo "SEEDS entries must be non-negative integers; got '$seed_value'" >&2
    exit 2
  fi
  if [[ -n "${SEEN_SEEDS[$seed_value]+x}" ]]; then
    echo "SEEDS must not contain duplicates: '$seed_value'" >&2
    exit 2
  fi
  SEEN_SEEDS[$seed_value]=1
done
for value in "$QUERIES" "$ROUNDS" "$TOP_K" "$CLUSTERS" "$CLUSTER_NOISE_BITS" "$NOISE_BITS" "$HARD_NOISE_BITS" "$CANDIDATE_MULTIPLIER"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "numeric sweep settings must be non-negative integers; got '$value'" >&2
    exit 2
  fi
done

if (( QUERIES == 0 || ROUNDS == 0 || TOP_K == 0 || CLUSTERS == 0 )); then
  echo "QUERIES, ROUNDS, TOP_K and CLUSTERS must be positive" >&2
  exit 2
fi
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

# Build once; cargo run below respects Cargo/Nix target-directory settings.
cargo build --release -p symthaea-hdc-store --example ann_benchmark

write_environment_receipt() {
  local path="$1"
  shift
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
  for RUN_SEED in "${SEEDS_ARRAY[@]}"; do
  for FALLBACK in true false; do
    STEM="broca-ann-sha-${SOURCE_SHA}-n${SIZE}-q${QUERIES}-r${ROUNDS}-k${TOP_K}-seed${RUN_SEED}-empty-fallback-${FALLBACK}"
    OUTPUT="$OUT_DIR/$STEM.csv"
    STDERR="$OUT_DIR/$STEM.stderr.log"
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
      "--seed=$RUN_SEED"
      "--candidate-multiplier=$CANDIDATE_MULTIPLIER"
      "--fallback-on-empty=$FALLBACK"
      "--lsh-configs=$LSH_CONFIGS"
    )
    printf 'Running n=%s fallback_on_empty=%s source=%s\n' "$SIZE" "$FALLBACK" "$SOURCE_SHA"
    write_environment_receipt "$RECEIPT" cargo run --release -p symthaea-hdc-store --example ann_benchmark -- "${ARGS[@]}"
    set +e
    cargo run --release -p symthaea-hdc-store --example ann_benchmark -- "${ARGS[@]}" > "$OUTPUT" 2> "$STDERR"
    STATUS=$?
    set -e
    {
      echo "exit_status=$STATUS"
      echo "output=$OUTPUT"
      echo "stderr_log=$STDERR"
      if command -v sha256sum >/dev/null 2>&1 && [[ -f "$OUTPUT" ]]; then
        echo "output_sha256=$(sha256sum "$OUTPUT" | cut -d' ' -f1)"
      fi
      if command -v sha256sum >/dev/null 2>&1 && [[ -f "$STDERR" ]]; then
        echo "stderr_sha256=$(sha256sum "$STDERR" | cut -d' ' -f1)"
      fi
    } >> "$RECEIPT"
    if (( STATUS != 0 )); then
      echo "benchmark failed (exit $STATUS); receipt retained at $RECEIPT" >&2
      exit "$STATUS"
    fi
    echo "wrote $OUTPUT"
  done
  done
done

echo "Sweep complete. Preserve CSVs, stderr logs, receipts, exact source SHA, and full command output together."
