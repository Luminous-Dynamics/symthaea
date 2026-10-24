#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Source-bound runner for GMC-BUDGET-FRONTIER-V1.
# Uses tracked-source cleanliness; untracked output artifacts are allowed.

set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "FAIL: sha256sum is required for result receipts" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "FAIL: tracked working tree must be clean before benchmarking" >&2
  exit 2
fi

SOURCE_SHA="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
SEED="${SEED:-17}"
CANDIDATES="${CANDIDATES:-120}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/broca-consolidation-budget}"
mkdir -p "$OUT_DIR"

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "FAIL: SEED must be an unsigned integer" >&2
  exit 2
fi
if [[ ! "$CANDIDATES" =~ ^[0-9]+$ ]] || (( CANDIDATES < 30 || CANDIDATES > 1000 )); then
  echo "FAIL: CANDIDATES must be in 30..=1000" >&2
  exit 2
fi

STEM="budget-frontier-seed-${SEED}-n-${CANDIDATES}"
OUTPUT="$OUT_DIR/$STEM.jsonl"
STDERR_LOG="$OUT_DIR/$STEM.stderr.log"
RECEIPT="$OUT_DIR/$STEM.receipt.txt"
if [[ -e "$OUTPUT" || -e "$STDERR_LOG" || -e "$RECEIPT" ]]; then
  echo "FAIL: refusing to overwrite existing artifacts for $STEM" >&2
  exit 2
fi

COMMAND="cargo run --manifest-path research/broca-consolidation/Cargo.toml --release --bin budget_frontier -- --source-sha $SOURCE_SHA --seed $SEED --candidates $CANDIDATES"
set +e
cargo run --manifest-path research/broca-consolidation/Cargo.toml --release --bin budget_frontier -- \
  --source-sha "$SOURCE_SHA" --seed "$SEED" --candidates "$CANDIDATES" \
  > "$OUTPUT" 2> "$STDERR_LOG"
STATUS=$?
set -e

OUTPUT_SHA="$(sha256sum "$OUTPUT" | awk '{print $1}')"
STDERR_SHA="$(sha256sum "$STDERR_LOG" | awk '{print $1}')"
RUSTC_VERSION="$(rustc --version 2>&1 || printf 'unavailable')"
CARGO_VERSION="$(cargo --version 2>&1 || printf 'unavailable')"
CPU_MODEL="$(lscpu 2>/dev/null | awk -F: '/Model name/ { gsub(/^[ \t]+/, "", $2); print $2; exit }')"
CPU_MODEL="${CPU_MODEL:-unavailable}"

{
  printf 'protocol=GMC-BUDGET-FRONTIER-V1\n'
  printf 'source_sha=%s\n' "$SOURCE_SHA"
  printf 'branch=%s\n' "$BRANCH"
  printf 'seed=%s\n' "$SEED"
  printf 'candidates=%s\n' "$CANDIDATES"
  printf 'command=%s\n' "$COMMAND"
  printf 'rustc_version=%s\n' "$RUSTC_VERSION"
  printf 'cargo_version=%s\n' "$CARGO_VERSION"
  printf 'uname=%s\n' "$(uname -a)"
  printf 'cpu_model=%s\n' "$CPU_MODEL"
  printf 'output_sha256=%s\n' "$OUTPUT_SHA"
  printf 'stderr_sha256=%s\n' "$STDERR_SHA"
  printf 'exit_status=%s\n' "$STATUS"
} > "$RECEIPT"

if (( STATUS != 0 )); then
  echo "FAIL: benchmark exited $STATUS; artifacts preserved at $OUT_DIR" >&2
  exit "$STATUS"
fi

echo "PASS: source-bound benchmark exited 0"
echo "output=$OUTPUT"
echo "receipt=$RECEIPT"
