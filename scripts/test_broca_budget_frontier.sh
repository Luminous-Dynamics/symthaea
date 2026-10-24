#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Offline positive and negative-control fixtures for the frontier verifier.
# Produces synthetic protocol-shaped JSONL only; it is not benchmark evidence.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKER="$ROOT/scripts/check_broca_budget_frontier.sh"
if ! command -v jq >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1; then
  echo "FAIL: jq and sha256sum are required" >&2
  exit 2
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SOURCE_SHA="0123456789abcdef0123456789abcdef01234567"
SEED=17
CANDIDATES=120
RAW_DEMAND=3000
BUDGETS='[32,64,128,256,512,1024,2048,4096,8192]'
POLICIES='["oracle_diagnostic","no_memory","always_raw","always_summary","fixed_interval","dedup_merge","feature_threshold","budget_adaptive","promote_everything"]'
OUTPUT="$TMP/frontier.jsonl"
STDERR_LOG="$TMP/frontier.stderr.log"
RECEIPT="$TMP/frontier.receipt.txt"

make_fixture() {
  local output="$1"
  jq -cn --arg sha "$SOURCE_SHA" --argjson seed "$SEED" --argjson candidates "$CANDIDATES" --argjson raw "$RAW_DEMAND" \
    '{record:"run",protocol:"GMC-BUDGET-FRONTIER-V1",source_sha:$sha,seed:$seed,candidate_count:$candidates,raw_token_demand:$raw,feature_contract:"observed-only-v1",utility_contract:"synthetic-v1",storage_cost_per_token:0.02,budget_units:"abstract memory-token units"}' > "$output"
  jq -cn --arg sha "$SOURCE_SHA" --argjson seed "$SEED" --argjson candidates "$CANDIDATES" --argjson raw "$RAW_DEMAND" \
    --argjson budgets "$BUDGETS" --argjson policies "$POLICIES" '
    $budgets[] as $b | $policies[] as $p |
    {record:"budget_summary",protocol:"GMC-BUDGET-FRONTIER-V1",source_sha:$sha,seed:$seed,candidates:$candidates,raw_token_demand:$raw,
      policy:$p,budget_tokens:$b,
      used_tokens:(if $p=="no_memory" then 0 elif $p=="always_raw" then (if $b < $raw then $b else $raw end) elif $b < 10 then $b else 10 end),
      stored_candidates:(if $p=="no_memory" then 0 elif $p=="always_raw" and $b==8192 then $candidates elif $p=="always_raw" then 1 elif $b<10 then 1 else 10 end),
      future_queries:100,
      synthetic_query_score:0.5,
      task_utility:(if $p=="no_memory" then 0 elif $p=="oracle_diagnostic" then 100 else 60 end),
      net_utility:(if $p=="no_memory" then 0 elif $p=="oracle_diagnostic" then 100 else 60 end),
      oracle_net_utility:100,
      regret:(if $p=="oracle_diagnostic" then 0 else 40 end),
      load_bearing_coverage:(if $p=="no_memory" then 0 else 0.5 end),
      bad_retained:(if $p=="promote_everything" then 5 else 0 end),
      bad_promoted:(if $p=="promote_everything" and $b==8192 then 3 else 0 end),
      compression_ratio:1.5,net_utility_per_token:1.0,storage_cost_per_token:0.02}
  ' >> "$output"
}

make_receipt() {
  local report="$1"
  local receipt="$2"
  local stderr="$3"
  {
    printf 'protocol=GMC-BUDGET-FRONTIER-V1\n'
    printf 'source_sha=%s\n' "$SOURCE_SHA"
    printf 'seed=%s\n' "$SEED"
    printf 'candidates=%s\n' "$CANDIDATES"
    printf 'output_sha256=%s\n' "$(sha256sum "$report" | awk '{print $1}')"
    printf 'stderr_log=%s\n' "$stderr"
    printf 'stderr_sha256=%s\n' "$(sha256sum "$stderr" | awk '{print $1}')"
    printf 'exit_status=0\n'
    printf 'record_validation=passed\n'
  } > "$receipt"
}

: > "$STDERR_LOG"
make_fixture "$OUTPUT"
make_receipt "$OUTPUT" "$RECEIPT" "$STDERR_LOG"
bash "$CHECKER" "$OUTPUT" > "$TMP/good.out"
grep -q 'PASS: receipt hashes, JSONL protocol, and all 81 policy/budget summaries verified' "$TMP/good.out"

TAMPERED="$TMP/tampered.jsonl"
cp "$OUTPUT" "$TAMPERED"
printf '\n' >> "$TAMPERED"
make_receipt "$TAMPERED" "$TMP/tampered.receipt.txt" "$STDERR_LOG"
# Receipt path is inferred from the report stem, so copy into a matching bundle.
mkdir "$TMP/tampered"
cp "$TAMPERED" "$TMP/tampered/frontier.jsonl"
make_receipt "$TMP/tampered/frontier.jsonl" "$TMP/tampered/frontier.receipt.txt" "$STDERR_LOG"
printf '\n' >> "$TMP/tampered/frontier.jsonl"
if bash "$CHECKER" "$TMP/tampered/frontier.jsonl" > "$TMP/tampered.out" 2>&1; then
  echo "FAIL: verifier accepted a report modified after its receipt was written" >&2
  exit 1
fi
grep -q 'output SHA-256 does not match receipt' "$TMP/tampered.out"

mkdir "$TMP/missing"
jq -c 'select(.record=="run" or .policy!="no_memory" or .budget_tokens!=32)' "$OUTPUT" > "$TMP/missing/frontier.jsonl"
make_receipt "$TMP/missing/frontier.jsonl" "$TMP/missing/frontier.receipt.txt" "$STDERR_LOG"
if bash "$CHECKER" "$TMP/missing/frontier.jsonl" > "$TMP/missing.out" 2>&1; then
  echo "FAIL: verifier accepted an incomplete policy/budget matrix" >&2
  exit 1
fi
grep -q 'expected 82 JSONL records' "$TMP/missing.out"

mkdir "$TMP/oracle"
jq -c 'if .record=="budget_summary" and .policy=="oracle_diagnostic" and .budget_tokens==32 then .regret=1 else . end' "$OUTPUT" > "$TMP/oracle/frontier.jsonl"
make_receipt "$TMP/oracle/frontier.jsonl" "$TMP/oracle/frontier.receipt.txt" "$STDERR_LOG"
if bash "$CHECKER" "$TMP/oracle/frontier.jsonl" > "$TMP/oracle.out" 2>&1; then
  echo "FAIL: verifier accepted nonzero oracle regret" >&2
  exit 1
fi
grep -q 'oracle diagnostic regret is nonzero' "$TMP/oracle.out"

echo "PASS: valid fixture accepted; tampered digest, incomplete matrix, and oracle-regret negative controls rejected"
