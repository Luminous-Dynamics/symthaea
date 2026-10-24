#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Fail-closed validator/descriptive scorecard for held-out Broca task records.
# It does not establish statistical significance or certify a true zero-risk rate.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: bash scripts/check_broca_task_eval.sh <heldout.csv>" >&2
  exit 2
fi

REPORT="$1"
if [[ ! -f "$REPORT" || "$REPORT" != *.csv ]]; then
  echo "FAIL: expected an existing .csv file: $REPORT" >&2
  exit 2
fi
if ! command -v awk >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1; then
  echo "FAIL: awk and sha256sum are required" >&2
  exit 2
fi

BASE="$(basename "$REPORT" .csv)"
RECEIPT="$(dirname "$REPORT")/$BASE.receipt.txt"
if [[ ! -f "$RECEIPT" ]]; then
  echo "FAIL: missing sidecar receipt: $RECEIPT" >&2
  exit 1
fi
if ! grep -qx 'exit_status=0' "$RECEIPT"; then
  echo "FAIL: receipt does not record exit_status=0" >&2
  exit 1
fi

receipt_value() {
  awk -F= -v key="$1" '
    $1 == key {
      count++
      value = substr($0, index($0, "=") + 1)
    }
    END {
      if (count != 1) exit 1
      print value
    }
  ' "$RECEIPT"
}

for key in source_sha dataset_sha256 config_sha256 output_sha256; do
  if ! receipt_value "$key" >/dev/null; then
    echo "FAIL: receipt must contain exactly one $key field" >&2
    exit 1
  fi
done

EXPECTED_OUTPUT_SHA="$(receipt_value output_sha256)"
ACTUAL_OUTPUT_SHA="$(sha256sum "$REPORT" | awk '{print $1}')"
if [[ "$EXPECTED_OUTPUT_SHA" != "$ACTUAL_OUTPUT_SHA" ]]; then
  echo "FAIL: CSV SHA-256 does not match receipt" >&2
  exit 1
fi

RECEIPT_SOURCE_SHA="$(receipt_value source_sha)"
RECEIPT_DATASET_SHA="$(receipt_value dataset_sha256)"
RECEIPT_CONFIG_SHA="$(receipt_value config_sha256)"
MIN_CASES="$(printenv MIN_CASES 2>/dev/null || printf '100')"
MIN_GROUPS="$(printenv MIN_GROUPS 2>/dev/null || printf '10')"
if [[ ! "$MIN_CASES" =~ ^[1-9][0-9]*$ || ! "$MIN_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "FAIL: MIN_CASES and MIN_GROUPS must be positive integers" >&2
  exit 2
fi

awk \
  -v receipt_source="$RECEIPT_SOURCE_SHA" \
  -v receipt_dataset="$RECEIPT_DATASET_SHA" \
  -v receipt_config="$RECEIPT_CONFIG_SHA" \
  -v min_cases="$MIN_CASES" \
  -v min_groups="$MIN_GROUPS" '
BEGIN {
  FS = ","
  OFS = ","
  expected_header = "protocol_id,run_id,source_sha,dataset_id,dataset_sha256,config_sha256,split,case_id,group_id,stratum,arm,task_success,action_applicable,action_correct,abstention_applicable,abstention_correct,stale_memory_reuse,unauthorized_memory_exposure,deleted_memory_resurrection,query_latency_ms,write_latency_ms,input_tokens,output_tokens,storage_bytes,energy_j"
}
function fail(message) {
  print "FAIL: " message > "/dev/stderr"
  bad = 1
}
function governance_fail(message) {
  print "GOVERNANCE FAIL: " message > "/dev/stderr"
  governance_bad = 1
}
function safe_id(value) {
  return value ~ /^[A-Za-z0-9._-]+$/
}
function nonnegative_number(value) {
  return value ~ /^[0-9]+([.][0-9]+)?$/
}
function rate(numerator, denominator) {
  if (denominator == 0) return "NA"
  return sprintf("%.6f", numerator / denominator)
}
FNR == 1 {
  if ($0 != expected_header) fail("header does not match scorecard schema v1")
  next
}
{
  line = FNR
  rows++
  if (index($0, "\"") != 0) fail("line " line ": quoted CSV fields are not allowed by protocol v1")
  if (NF != 25) {
    fail("line " line ": expected 25 columns, got " NF)
    next
  }
  if ($1 != "BROCA-TASK-EVAL-V1") fail("line " line ": wrong protocol_id")
  if (!safe_id($2) || !safe_id($4) || !safe_id($8) || !safe_id($9) || !safe_id($10) || !safe_id($11)) {
    fail("line " line ": invalid identifier; use letters, digits, dot, underscore or hyphen")
  }
  if ($3 !~ /^[0-9a-f]{40}$/ || $5 !~ /^[0-9a-f]{64}$/ || $6 !~ /^[0-9a-f]{64}$/) {
    fail("line " line ": malformed source/dataset/config digest")
  }
  if ($7 != "heldout") fail("line " line ": only heldout rows are accepted by this gate")
  if ($3 != receipt_source || $5 != receipt_dataset || $6 != receipt_config) {
    fail("line " line ": row hashes do not match sidecar receipt")
  }

  if (rows == 1) {
    run_id = $2
    source_sha = $3
    dataset_id = $4
    dataset_sha = $5
    config_sha = $6
  } else if ($2 != run_id || $3 != source_sha || $4 != dataset_id || $5 != dataset_sha || $6 != config_sha) {
    fail("line " line ": run/source/dataset/config metadata changes within one CSV")
  }

  for (i = 12; i <= 19; i++) {
    if ($i !~ /^[01]$/) fail("line " line ": fields 12-19 must be binary 0/1 values")
  }
  if ($13 == 0 && $14 != 0) fail("line " line ": action_correct must be zero when action_applicable is zero")
  if ($15 == 0 && $16 != 0) fail("line " line ": abstention_correct must be zero when abstention_applicable is zero")
  if (!nonnegative_number($20) || !nonnegative_number($21) ||
      $22 !~ /^[0-9]+$/ || $23 !~ /^[0-9]+$/ || $24 !~ /^[0-9]+$/) {
    fail("line " line ": latency/tokens/storage fields must be non-negative numbers")
  }
  if ($25 != "-1" && !nonnegative_number($25)) fail("line " line ": energy_j must be -1 (unmeasured) or non-negative")
  if (bad) next

  case_id = $8
  group_id = $9
  stratum = $10
  arm = $11
  pair_key = case_id SUBSEP arm
  if (pair_key in pair_seen) {
    fail("line " line ": duplicate case_id/arm pair: " case_id "/" arm)
    next
  }
  pair_seen[pair_key] = 1
  if (!(case_id in case_seen)) {
    case_seen[case_id] = 1
    case_count++
    case_group[case_id] = group_id
    case_stratum[case_id] = stratum
  } else if (case_group[case_id] != group_id || case_stratum[case_id] != stratum) {
    fail("line " line ": group_id or stratum differs between arms for case " case_id)
  }
  if (!(group_id in group_seen)) {
    group_seen[group_id] = 1
    group_count++
  }
  arms[arm] = 1
  strata[stratum] = 1
  arm_rows[arm]++
  successes[arm] += $12
  action_total[arm] += $13
  action_success[arm] += $14
  abstention_total[arm] += $15
  abstention_success[arm] += $16
  stale_events[arm] += $17
  scope_events[arm] += $18
  resurrection_events[arm] += $19
  query_latency_sum[arm] += $20
  write_latency_sum[arm] += $21
  input_tokens_sum[arm] += $22
  output_tokens_sum[arm] += $23
  storage_bytes_sum[arm] += $24
  if ($25 >= 0) {
    energy_sum[arm] += $25
    energy_count[arm]++
  }
  success_by_case[pair_key] = $12
  stratum_case_count[stratum SUBSEP arm]++
  stratum_success[stratum SUBSEP arm] += $12

  if ($17 > 0) governance_fail("line " line ": stale-memory reuse observed for case " case_id " arm " arm)
  if ($18 > 0) governance_fail("line " line ": unauthorized memory exposure observed for case " case_id " arm " arm)
  if ($19 > 0) governance_fail("line " line ": deleted-memory resurrection observed for case " case_id " arm " arm)
}
END {
  if (rows == 0) fail("CSV has no data rows")
  if (case_count < min_cases) fail("heldout case count " case_count " is below MIN_CASES=" min_cases)
  if (group_count < min_groups) fail("independent group count " group_count " is below MIN_GROUPS=" min_groups)
  if (!("no_memory" in arms)) fail("required baseline arm no_memory is missing")
  if (!("exact_hdc" in arms)) fail("required baseline arm exact_hdc is missing")
  competitor = 0
  if ("bm25" in arms || "dense_retriever" in arms || "strong_retriever" in arms) competitor = 1
  if (!competitor) fail("at least one external retrieval baseline is required: bm25, dense_retriever, or strong_retriever")
  for (case_id in case_seen) {
    for (arm in arms) {
      key = case_id SUBSEP arm
      if (!(key in pair_seen)) fail("case " case_id " is missing arm " arm)
    }
  }
  if (bad) exit 1

  print "BROCA-TASK-EVAL-V1 descriptive scorecard"
  print "run_id=" run_id
  print "dataset_id=" dataset_id
  print "source_sha=" source_sha
  print "dataset_sha256=" dataset_sha
  print "config_sha256=" config_sha
  print "heldout_cases=" case_count
  print "independent_groups=" group_count
  print ""
  print "ARM_METRICS"
  print "arm,cases,task_success_rate,action_accuracy,abstention_accuracy,stale_memory_reuse_rate,unauthorized_exposures,deleted_resurrections,mean_query_latency_ms,mean_write_latency_ms,mean_input_tokens,mean_output_tokens,mean_storage_bytes,mean_energy_j,energy_observed_cases"
  for (arm in arms) {
    printf "%s,%d,%s,%s,%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%d\n", arm, arm_rows[arm], rate(successes[arm], arm_rows[arm]), rate(action_success[arm], action_total[arm]), rate(abstention_success[arm], abstention_total[arm]), rate(stale_events[arm], arm_rows[arm]), scope_events[arm], resurrection_events[arm], rate(query_latency_sum[arm], arm_rows[arm]), rate(write_latency_sum[arm], arm_rows[arm]), rate(input_tokens_sum[arm], arm_rows[arm]), rate(output_tokens_sum[arm], arm_rows[arm]), rate(storage_bytes_sum[arm], arm_rows[arm]), rate(energy_sum[arm], energy_count[arm]), energy_count[arm]
  }
  print ""
  print "PAIRED_TASK_SUCCESS_VS_NO_MEMORY"
  print "arm,candidate_wins,no_memory_wins,ties,net_lift"
  for (arm in arms) {
    if (arm == "no_memory") continue
    wins = losses = ties = 0
    for (case_id in case_seen) {
      candidate = success_by_case[case_id SUBSEP arm]
      baseline = success_by_case[case_id SUBSEP "no_memory"]
      if (candidate > baseline) wins++
      else if (candidate < baseline) losses++
      else ties++
    }
    printf "%s,%d,%d,%d,%s\n", arm, wins, losses, ties, rate(wins - losses, case_count)
  }
  print ""
  print "STRATUM_TASK_SUCCESS"
  print "stratum,arm,cases,task_success_rate"
  for (stratum in strata) {
    for (arm in arms) {
      key = stratum SUBSEP arm
      if (stratum_case_count[key] > 0) {
        printf "%s,%s,%d,%s\n", stratum, arm, stratum_case_count[key], rate(stratum_success[key], stratum_case_count[key])
      }
    }
  }
  if (governance_bad) {
    print "FAIL: one or more non-compensatory governance gates were violated" > "/dev/stderr"
    exit 1
  }
  print "PASS: structural/data-integrity gates passed; this is not a statistical-significance or zero-risk certification"
}
' "$REPORT"
