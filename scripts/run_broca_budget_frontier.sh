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
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "FAIL: SEED must be an unsigned integer" >&2
  exit 2
fi
if [[ ! "$CANDIDATES" =~ ^[0-9]+$ ]] || (( CANDIDATES < 30 || CANDIDATES > 200 )); then
  echo "FAIL: CANDIDATES must be in 30..=200 so raw evidence always fits the 8192-unit budget" >&2
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

OUTPUT_LINES="$(awk 'END { print NR + 0 }' "$OUTPUT")"
SUMMARY_RECORDS="$(awk '/"record":"budget_summary"/ { count++ } END { print count + 0 }' "$OUTPUT")"
RECORD_VALIDATION="not_checked"
if (( STATUS == 0 )); then
  if (( OUTPUT_LINES != 82 || SUMMARY_RECORDS != 81 )) || ! head -n 1 "$OUTPUT" | grep -q '^{"record":"run","protocol":"GMC-BUDGET-FRONTIER-V1",'; then
    STATUS=65
    RECORD_VALIDATION="failed"
    printf 'FAIL: expected one run record plus 81 budget summaries; got lines=%s summaries=%s\n' "$OUTPUT_LINES" "$SUMMARY_RECORDS" >> "$STDERR_LOG"
  else
    RECORD_VALIDATION="passed"
  fi
fi

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
  printf 'stderr_log=%s\n' "$STDERR_LOG"
  printf 'output_lines=%s\n' "$OUTPUT_LINES"
  printf 'summary_records=%s\n' "$SUMMARY_RECORDS"
  printf 'record_validation=%s\n' "$RECORD_VALIDATION"
  printf 'stderr_sha256=%s\n' "$STDERR_SHA"
  printf 'exit_status=%s\n' "$STATUS"
} > "$RECEIPT"

if (( STATUS != 0 )); then
  echo "FAIL: benchmark exited $STATUS; artifacts preserved at $OUT_DIR" >&2
  exit "$STATUS"
fi

if ! bash scripts/check_broca_budget_frontier.sh "$OUTPUT"; then
  sed -i 's/^exit_status=0$/exit_status=65/' "$RECEIPT"
  sed -i 's/^record_validation=passed$/record_validation=failed/' "$RECEIPT"
  echo "FAIL: independent frontier artifact verification failed; artifacts preserved at $OUT_DIR" >&2
  exit 65
fi

echo "PASS: source-bound benchmark and independent artifact verifier passed"
echo "output=$OUTPUT"
echo "receipt=$RECEIPT"
