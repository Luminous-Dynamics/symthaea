#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 RECEIPT.json" >&2
  exit 2
}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage
receipt_arg=$1
[[ -f "$receipt_arg" ]] || fail "receipt not found: $receipt_arg"
command -v jq >/dev/null 2>&1 || fail "jq is required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"
command -v realpath >/dev/null 2>&1 || fail "realpath is required"

receipt=$(realpath -e -- "$receipt_arg")
root=$(dirname -- "$receipt")

jq -e '
  def hex40: type == "string" and test("^[0-9a-f]{40}$");
  def hex64: type == "string" and test("^[0-9a-f]{64}$");
  .protocol_id == "BROCA-LME-V2-RECEIPT-V1" and
  .status == "completed" and
  .benchmark.name == "LongMemEval-V2" and
  .benchmark.repository == "https://github.com/xiaowu0162/LongMemEval-V2" and
  (.benchmark.upstream_commit | hex40) and
  (.benchmark.dataset_manifest_sha256 | hex64) and
  (.benchmark.dataset_manifest_file | type == "string" and length > 0) and
  (.benchmark.official_harness_manifest_sha256 | hex64) and
  (.benchmark.official_harness_manifest_file | type == "string" and length > 0) and
  (.benchmark.tier == "small" or .benchmark.tier == "medium") and
  .benchmark.question_count == 451 and
  (.experiment.symthaea_commit | hex40) and
  (.experiment.config_sha256 | hex64) and
  (.experiment.reader_model | type == "string" and length > 0) and
  (.experiment.reader_revision | type == "string" and length > 0) and
  (.experiment.judge_model | type == "string" and length > 0) and
  (.experiment.judge_revision | type == "string" and length > 0) and
  (.experiment.reader_prompt_sha256 | hex64) and
  (.experiment.reader_prompt_file | type == "string" and length > 0) and
  (.experiment.judge_prompt_sha256 | hex64) and
  (.experiment.judge_prompt_file | type == "string" and length > 0) and
  (.experiment.memory_context_max_tokens | type == "number" and . > 0) and
  (.benchmark.arms | type == "array" and length >= 3 and (unique | length) == length) and
  (.benchmark.arms | map(select(. == "broca" or . == "no_retrieval" or . == "rag_query_to_slice")) | unique | sort) == ["broca", "no_retrieval", "rag_query_to_slice"] and
  (.experiment.runs | type == "array" and length >= 6) and
  (.experiment.aggregates | type == "array" and length >= 3) and
  (.benchmark.reference_frontier_sha256 | hex64) and
  (.benchmark.reference_frontier_file | type == "string" and length > 0) and
  (.experiment.config_file | type == "string" and length > 0)
' "$receipt" >/dev/null || fail "receipt violates BROCA-LME-V2-RECEIPT-V1 required fields or pins"

# Do not allow unknown run arms to silently disappear from a summary.
if ! jq -e '
  [.benchmark.arms[]] as $declared |
  [.experiment.runs[].arm] as $observed |
  [.experiment.aggregates[].arm] as $aggregated |
  ($declared | unique | sort) == ($observed | unique | sort) and
  ($declared | unique | sort) == ($aggregated | unique | sort) and
  all($declared[]; . == "broca" or . == "no_retrieval" or . == "rag_query_to_slice" or . == "rag_query_to_slice_notes" or . == "agentrunbook_r" or . == "codex" or . == "agentrunbook_c")
' "$receipt" >/dev/null; then
  fail "declared arms must exactly match run and aggregate arms and use a known arm name"
fi

# One completed run for each declared arm × domain; the total is exactly the
# benchmark's 451 cases for every arm. Duplicate or skipped rows fail closed.
if ! jq -e '
  . as $root |
  $root.benchmark.arms as $arms |
  all($arms[]; . as $arm |
    ($root.experiment.runs | map(select(.arm == $arm))) as $runs |
    ($runs | length) == 2 and
    ([$runs[].domain] | sort) == ["enterprise", "web"] and
    all($runs[]; .status == "completed" and (.case_count | type == "number" and . > 0) and (.output_file | type == "string" and length > 0) and (.format == "canonical-question-jsonl") and (.sha256 | type == "string" and test("^[0-9a-f]{64}$")) and (.case_count == (.case_count | floor))) and
    ([$runs[].case_count] | add) == 451
  ) and
  all($root.experiment.aggregates[]; .status == "completed" and (.output_file | type == "string" and length > 0) and (.sha256 | type == "string" and test("^[0-9a-f]{64}$"))) and
  all($root.benchmark.arms[]; . as $arm | ([$root.experiment.runs[] | select(.arm == $arm)] | length) == 2) and
  all($root.benchmark.arms[]; . as $arm | ([$root.experiment.aggregates[] | select(.arm == $arm)] | length) == 1)
' "$receipt" >/dev/null; then
  fail "runs or aggregates are incomplete, duplicated, skipped, or have invalid case counts"
fi

verify_artifact() {
  local rel=$1 expected=$2 label=$3 full actual
  [[ "$rel" != /* ]] || fail "$label path must be relative to the receipt directory"
  [[ "$rel" != *".."* ]] || fail "$label path must not contain '..'"
  full=$(realpath -e -- "$root/$rel") || fail "$label artifact missing: $rel"
  [[ "$full" == "$root/"* ]] || fail "$label artifact resolves outside the receipt directory: $rel"
  [[ -f "$full" ]] || fail "$label artifact is not a regular file: $rel"
  actual=$(sha256sum -- "$full" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || fail "$label SHA-256 mismatch for $rel (expected $expected, got $actual)"
}

web_ids_digest=""
enterprise_ids_digest=""
while IFS=$'\t' read -r path digest arm domain expected_count; do
  verify_artifact "$path" "$digest" "run artifact ($arm/$domain)"
  full=$(realpath -e -- "$root/$path")
  if ! jq -s -e --argjson count "$expected_count" 'length == $count and all(.[]; type == "object" and (.case_id | type == "string" and length > 0)) and ([.[].case_id] | unique | length) == $count' "$full" >/dev/null; then
    fail "canonical JSONL must contain exactly $expected_count unique case_id records for $arm/$domain"
  fi
  ids_digest=$(jq -sr '[.[].case_id] | sort | join("\n")' "$full" | sha256sum | awk '{print $1}')
  if [[ "$domain" == "web" ]]; then
    [[ -z "$web_ids_digest" || "$web_ids_digest" == "$ids_digest" ]] || fail "paired case_id set differs across arms for domain $domain"
    web_ids_digest=$ids_digest
  else
    [[ -z "$enterprise_ids_digest" || "$enterprise_ids_digest" == "$ids_digest" ]] || fail "paired case_id set differs across arms for domain $domain"
    enterprise_ids_digest=$ids_digest
  fi
done < <(jq -r '.experiment.runs[] | [.output_file,.sha256,.arm,.domain,.case_count] | @tsv' "$receipt")

while IFS=$'\t' read -r path digest arm; do
  verify_artifact "$path" "$digest" "combined aggregate ($arm)"
  full=$(realpath -e -- "$root/$path")
  if ! jq -e '
    def metric: type == "number" and . >= 0 and . <= 1;
    has("overall_full_set") and has("gotchas_accuracy") and has("static_accuracy") and has("dynamic_accuracy") and has("procedure_accuracy") and has("memory_query_avg_seconds") and
    (.overall_full_set | metric) and (.gotchas_accuracy | metric) and (.static_accuracy | metric) and (.dynamic_accuracy | metric) and (.procedure_accuracy | metric) and
    (.memory_query_avg_seconds | type == "number" and . >= 0)
  ' "$full" >/dev/null; then
    fail "combined aggregate for $arm lacks valid official metric fields"
  fi
done < <(jq -r '.experiment.aggregates[] | [.output_file,.sha256,.arm] | @tsv' "$receipt")

while IFS=$'\t' read -r path digest label; do
  verify_artifact "$path" "$digest" "$label"
done < <(jq -r '[ [.benchmark.dataset_manifest_file,.benchmark.dataset_manifest_sha256,"dataset manifest"], [.benchmark.official_harness_manifest_file,.benchmark.official_harness_manifest_sha256,"official harness manifest"], [.benchmark.reference_frontier_file,.benchmark.reference_frontier_sha256,"official reference frontier"], [.experiment.config_file,.experiment.config_sha256,"experiment config"], [.experiment.reader_prompt_file,.experiment.reader_prompt_sha256,"reader prompt"], [.experiment.judge_prompt_file,.experiment.judge_prompt_sha256,"judge prompt"] ][] | @tsv' "$receipt")

echo "PASS: complete LongMemEval-V2 receipt; per-arm domain coverage, 451 unique paired case IDs, and all artifact hashes verified"
