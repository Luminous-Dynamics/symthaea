#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Offline protocol fixture test. No Cargo, model, dataset or network required.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKER="$ROOT/scripts/check_broca_task_eval.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SOURCE_SHA="0123456789abcdef0123456789abcdef01234567"
DATASET_SHA="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
CONFIG_SHA="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
HEADER="protocol_id,run_id,source_sha,dataset_id,dataset_sha256,config_sha256,split,case_id,group_id,stratum,arm,task_success,action_applicable,action_correct,abstention_applicable,abstention_correct,stale_memory_reuse,unauthorized_memory_exposure,deleted_memory_resurrection,query_latency_ms,write_latency_ms,input_tokens,output_tokens,storage_bytes,energy_j"

make_csv() {
  local output="$1"
  awk -v source="$SOURCE_SHA" -v dataset="$DATASET_SHA" -v config="$CONFIG_SHA" -v header="$HEADER" '
  BEGIN {
    print header
    arms[1] = "no_memory"
    arms[2] = "exact_hdc"
    arms[3] = "bm25"
    arms[4] = "approx_hdc"
    strata[1] = "explicit_fact"
    strata[2] = "compositional"
    strata[3] = "temporal_update"
    strata[4] = "tool_action"
    strata[5] = "governance"
    for (i = 1; i <= 100; i++) {
      case_id = sprintf("case-%04d", i)
      group_id = sprintf("group-%02d", int((i - 1) / 10) + 1)
      stratum = strata[((i - 1) % 5) + 1]
      for (j = 1; j <= 4; j++) {
        if (j == 1) success = (i % 3 == 0)
        else if (j == 2) success = (i % 3 == 0 || i % 5 == 0)
        else if (j == 3) success = (i % 3 == 0 || i % 4 == 0)
        else success = (i % 3 == 0 || i % 7 == 0)
        action_applicable = (i % 2)
        action_correct = (action_applicable && success) ? 1 : 0
        abstention_applicable = (i % 10 == 0)
        abstention_correct = (abstention_applicable && success) ? 1 : 0
        printf "BROCA-TASK-EVAL-V1,fixture-run-1,%s,fixture_dataset,%s,%s,heldout,%s,%s,%s,%s,%d,%d,%d,%d,%d,0,0,0,%.1f,%.1f,%d,%d,%d,-1\n", source, dataset, config, case_id, group_id, stratum, arms[j], success ? 1 : 0, action_applicable ? 1 : 0, action_correct, abstention_applicable ? 1 : 0, abstention_correct, 10 + j * 2, j + 1, 100 + j * 10, 20 + j * 5, 4096 * j
      }
    }
  }' > "$output"
}

make_receipt() {
  local csv="$1"
  local receipt
  receipt="$(dirname "$csv")/$(basename "$csv" .csv).receipt.txt"
  {
    printf 'exit_status=0\n'
    printf 'source_sha=%s\n' "$SOURCE_SHA"
    printf 'dataset_sha256=%s\n' "$DATASET_SHA"
    printf 'config_sha256=%s\n' "$CONFIG_SHA"
    printf 'output_sha256=%s\n' "$(sha256sum "$csv" | awk '{print $1}')"
  } > "$receipt"
}

GOOD="$TMP/good.csv"
make_csv "$GOOD"
make_receipt "$GOOD"
bash "$CHECKER" "$GOOD" > "$TMP/good.out"
grep -q 'PASS: structural/data-integrity gates passed' "$TMP/good.out"
grep -q 'PAIRED_TASK_SUCCESS_VS_NO_MEMORY' "$TMP/good.out"

BAD="$TMP/governance_violation.csv"
awk -F, 'BEGIN { OFS="," } NR == 5 { $19 = 1 } { print }' "$GOOD" > "$BAD"
make_receipt "$BAD"
if bash "$CHECKER" "$BAD" > "$TMP/bad.out" 2>&1; then
  echo "FAIL: checker accepted a deleted-memory-resurrection fixture" >&2
  exit 1
fi
grep -q 'GOVERNANCE FAIL:.*deleted-memory resurrection observed' "$TMP/bad.out"

DUPLICATE="$TMP/duplicate.csv"
awk 'NR == 2 { first = $0 } { print } NR == 2 { print first }' "$GOOD" > "$DUPLICATE"
make_receipt "$DUPLICATE"
if bash "$CHECKER" "$DUPLICATE" > "$TMP/duplicate.out" 2>&1; then
  echo "FAIL: checker accepted a duplicate case/arm record" >&2
  exit 1
fi
grep -q 'duplicate case_id/arm pair' "$TMP/duplicate.out"

UNMATCHED="$TMP/unmatched.csv"
awk -F, 'BEGIN { OFS="," } !($8 == "case-0001" && $11 == "approx_hdc") { print }' "$GOOD" > "$UNMATCHED"
make_receipt "$UNMATCHED"
if bash "$CHECKER" "$UNMATCHED" > "$TMP/unmatched.out" 2>&1; then
  echo "FAIL: checker accepted a missing case/arm pair" >&2
  exit 1
fi
grep -q 'is missing arm approx_hdc' "$TMP/unmatched.out"

TAMPERED="$TMP/tampered.csv"
cp "$GOOD" "$TAMPERED"
cp "$TMP/good.receipt.txt" "$TMP/tampered.receipt.txt"
printf '\n' >> "$TAMPERED"
if bash "$CHECKER" "$TAMPERED" > "$TMP/tampered.out" 2>&1; then
  echo "FAIL: checker accepted a CSV whose hash no longer matches its receipt" >&2
  exit 1
fi
grep -q 'CSV SHA-256 does not match receipt' "$TMP/tampered.out"

echo "PASS: good fixture accepted; governance violation, duplicate, missing-pair and tampered-receipt negative controls rejected"
