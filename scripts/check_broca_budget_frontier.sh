#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Independent, fail-closed verifier for GMC-BUDGET-FRONTIER-V1 JSONL output.
# Validates hashes, run metadata, the complete policy x budget matrix, and basic
# oracle/budget invariants. It does not validate real-world task utility.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: bash scripts/check_broca_budget_frontier.sh <budget-frontier.jsonl>" >&2
  exit 2
fi

REPORT="$1"
if [[ ! -f "$REPORT" || "$REPORT" != *.jsonl ]]; then
  echo "FAIL: expected an existing .jsonl file: $REPORT" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1; then
  echo "FAIL: jq and sha256sum are required" >&2
  exit 2
fi

BASE="$(basename "$REPORT" .jsonl)"
RECEIPT="$(dirname "$REPORT")/$BASE.receipt.txt"
if [[ ! -f "$RECEIPT" ]]; then
  echo "FAIL: missing receipt: $RECEIPT" >&2
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

for key in protocol source_sha seed candidates output_sha256 stderr_log stderr_sha256 exit_status record_validation; do
  if ! receipt_value "$key" >/dev/null; then
    echo "FAIL: receipt must contain exactly one $key field" >&2
    exit 1
  fi
done

if [[ "$(receipt_value protocol)" != "GMC-BUDGET-FRONTIER-V1" ]]; then
  echo "FAIL: wrong receipt protocol" >&2
  exit 1
fi
if [[ "$(receipt_value exit_status)" != "0" || "$(receipt_value record_validation)" != "passed" ]]; then
  echo "FAIL: receipt does not certify successful process and record validation" >&2
  exit 1
fi

SOURCE_SHA="$(receipt_value source_sha)"
SEED="$(receipt_value seed)"
CANDIDATES="$(receipt_value candidates)"
EXPECTED_OUTPUT_SHA="$(receipt_value output_sha256)"
STDERR_PATH="$(receipt_value stderr_log)"
EXPECTED_STDERR_SHA="$(receipt_value stderr_sha256)"
if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ || ! "$SEED" =~ ^[0-9]+$ || ! "$CANDIDATES" =~ ^[0-9]+$ ]]; then
  echo "FAIL: malformed source SHA, seed, or candidate count in receipt" >&2
  exit 1
fi
if [[ ! -f "$STDERR_PATH" ]]; then
  echo "FAIL: stderr log from receipt is missing: $STDERR_PATH" >&2
  exit 1
fi
ACTUAL_OUTPUT_SHA="$(sha256sum "$REPORT" | awk '{print $1}')"
ACTUAL_STDERR_SHA="$(sha256sum "$STDERR_PATH" | awk '{print $1}')"
if [[ "$ACTUAL_OUTPUT_SHA" != "$EXPECTED_OUTPUT_SHA" ]]; then
  echo "FAIL: output SHA-256 does not match receipt" >&2
  exit 1
fi
if [[ "$ACTUAL_STDERR_SHA" != "$EXPECTED_STDERR_SHA" ]]; then
  echo "FAIL: stderr SHA-256 does not match receipt" >&2
  exit 1
fi

if ! jq -s -e \
  --arg source_sha "$SOURCE_SHA" \
  --arg seed "$SEED" \
  --arg candidates "$CANDIDATES" '
  def is_num: type == "number" and (tostring | test("^(NaN|Infinity|-Infinity)$") | not);
  def close_to_zero: (abs <= 0.000001);
  def budgets: [32,64,128,256,512,1024,2048,4096,8192];
  def policies: [
    "oracle_diagnostic","no_memory","always_raw","always_summary",
    "fixed_interval","dedup_merge","feature_threshold","budget_adaptive",
    "promote_everything"
  ];
  . as $all
  | if length != 82 then error("expected 82 JSONL records (1 run + 81 summaries)") else . end
  | .[0] as $run
  | if $run.record != "run" or $run.protocol != "GMC-BUDGET-FRONTIER-V1" then error("invalid run header") else . end
  | if $run.source_sha != $source_sha or ($run.source_sha | test("^[0-9a-f]{40}$") | not) then error("run source SHA differs from receipt") else . end
  | if ($run.seed | tostring) != $seed or ($run.candidate_count | tostring) != $candidates then error("run seed/candidate count differs from receipt") else . end
  | if $run.feature_contract != "observed-only-v1" or $run.utility_contract != "synthetic-v1" then error("unexpected feature or utility contract") else . end
  | if ($run.raw_token_demand | is_num | not) or $run.raw_token_demand <= 0 then error("invalid raw token demand") else . end
  | if ($run.storage_cost_per_token | is_num | not) or $run.storage_cost_per_token < 0 then error("invalid storage penalty") else . end
  | .[1:] as $rows
  | if any($rows[]; .record != "budget_summary" or .protocol != "GMC-BUDGET-FRONTIER-V1") then error("invalid summary row type/protocol") else . end
  | if any($rows[]; .source_sha != $source_sha or .seed != $run.seed or .candidates != $run.candidate_count or .raw_token_demand != $run.raw_token_demand) then error("summary metadata differs from run record") else . end
  | if any($rows[]; (.budget_tokens as $b | (budgets | index($b)) == null)) then error("unexpected budget value") else . end
  | if any($rows[]; (.policy as $p | (policies | index($p)) == null)) then error("unexpected policy value") else . end
  | if ([$rows[].budget_tokens] | unique | sort) != budgets then error("budget matrix incomplete") else . end
  | if ([$rows[].policy] | unique | sort) != (policies | sort) then error("policy matrix incomplete") else . end
  | if ([$rows[] | [.budget_tokens,.policy] | @json] | unique | length) != 81 then error("duplicate or missing policy/budget pair") else . end
  | if any($rows[]; (.used_tokens | is_num | not) or (.budget_tokens | is_num | not) or .used_tokens < 0 or .budget_tokens < 1 or .used_tokens > .budget_tokens) then error("budget exceeded or malformed used_tokens") else . end
  | if any($rows[]; (.regret | is_num | not) or (.net_utility | is_num | not) or (.oracle_net_utility | is_num | not) or .regret < -0.000001 or (.oracle_net_utility - .net_utility) < -0.000001) then error("invalid oracle regret/net-utility relation") else . end
  | if any($rows[]; (.synthetic_query_score | is_num | not) or .synthetic_query_score < 0 or .synthetic_query_score > 1 or (.load_bearing_coverage | is_num | not) or .load_bearing_coverage < 0 or .load_bearing_coverage > 1) then error("synthetic score/coverage outside [0,1]") else . end
  | if any($rows[]; (.stored_candidates | type) != "number" or .stored_candidates < 0 or .stored_candidates > $run.candidate_count or (.bad_promoted | type) != "number" or .bad_promoted < 0) then error("invalid candidate-count metric") else . end
  | if any($rows[] | select(.policy == "oracle_diagnostic"); ((.regret | close_to_zero) | not)) then error("oracle diagnostic regret is nonzero") else . end
  | if any($rows[] | select(.policy == "no_memory"); (.used_tokens != 0 or .stored_candidates != 0 or ((.net_utility | close_to_zero) | not) or ((.task_utility | close_to_zero) | not))) then error("no_memory arm is not an empty-memory zero-utility control") else . end
  | if any($rows[] | select(.policy == "always_raw" and .budget_tokens == 8192); (.used_tokens != $run.raw_token_demand or .stored_candidates != $run.candidate_count)) then error("raw-fit upper-budget control did not retain all raw candidates") else . end
  | if any($rows[] | select(.policy == "promote_everything" and .budget_tokens == 8192); .bad_promoted <= 0) then error("promote_everything anti-control did not expose unsafe promotion") else . end
  | true
' "$REPORT" >/dev/null; then
  echo "FAIL: JSONL frontier integrity or policy/budget invariants did not pass" >&2
  exit 1
fi

echo "PASS: receipt hashes, JSONL protocol, and all 81 policy/budget summaries verified"
