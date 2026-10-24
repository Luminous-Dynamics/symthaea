#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
checker="$repo_root/scripts/check_broca_lme_v2_receipt.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

make_fixture() {
  local root=$1 arm domain n path digest arms runs aggregates dataset_digest harness_digest frontier_digest config_digest reader_prompt_digest judge_prompt_digest i
  mkdir -p "$root/outputs" "$root/aggregates" "$root/pins"
  printf 'dataset-id=longmemeval-v2\nsnapshot=test-fixture\n' > "$root/pins/dataset.manifest"
  printf 'repository=https://github.com/xiaowu0162/LongMemEval-V2\ncommit=%040d\n' 1 > "$root/pins/harness.manifest"
  printf '{"reference":"fixture-only","tier":"small"}\n' > "$root/pins/reference-frontier.json"
  printf '{"reader":"test-reader","judge":"test-judge","context_cap":4096,"arms":["broca","no_retrieval","rag_query_to_slice"]}\n' > "$root/pins/config.json"
  printf 'fixture reader prompt\n' > "$root/pins/reader.prompt"
  printf 'fixture judge prompt\n' > "$root/pins/judge.prompt"
  dataset_digest=$(sha256sum "$root/pins/dataset.manifest" | awk '{print $1}')
  harness_digest=$(sha256sum "$root/pins/harness.manifest" | awk '{print $1}')
  frontier_digest=$(sha256sum "$root/pins/reference-frontier.json" | awk '{print $1}')
  config_digest=$(sha256sum "$root/pins/config.json" | awk '{print $1}')
  reader_prompt_digest=$(sha256sum "$root/pins/reader.prompt" | awk '{print $1}')
  judge_prompt_digest=$(sha256sum "$root/pins/judge.prompt" | awk '{print $1}')
  arms='["broca","no_retrieval","rag_query_to_slice"]'
  runs='[]' aggregates='[]'
  for arm in broca no_retrieval rag_query_to_slice; do
    for domain in web enterprise; do
      if [[ "$domain" == web ]]; then n=200; else n=251; fi
      path="outputs/$arm"_"$domain.jsonl"
      : > "$root/$path"
      for i in $(seq 1 "$n"); do
        printf '{"case_id":"%s-%04d","task_success":0}\n' "$domain" "$i" >> "$root/$path"
      done
      digest=$(sha256sum "$root/$path" | awk '{print $1}')
      runs=$(jq -c --arg arm "$arm" --arg domain "$domain" --arg path "$path" --arg digest "$digest" --argjson n "$n" '. + [{arm:$arm,domain:$domain,status:"completed",case_count:$n,format:"canonical-question-jsonl",output_file:$path,sha256:$digest}]' <<<"$runs")
    done
    path="aggregates/$arm.json"
    printf '{"method":"%s","overall_full_set":0.5,"gotchas_accuracy":0.5,"static_accuracy":0.5,"dynamic_accuracy":0.5,"procedure_accuracy":0.5,"memory_query_avg_seconds":0.01}\n' "$arm" > "$root/$path"
    digest=$(sha256sum "$root/$path" | awk '{print $1}')
    aggregates=$(jq -c --arg arm "$arm" --arg path "$path" --arg digest "$digest" '. + [{arm:$arm,status:"completed",output_file:$path,sha256:$digest}]' <<<"$aggregates")
  done
  jq -n \
    --arg dataset_digest "$dataset_digest" --arg harness_digest "$harness_digest" \
    --arg frontier_digest "$frontier_digest" --arg config_digest "$config_digest" \
    --arg reader_prompt_digest "$reader_prompt_digest" --arg judge_prompt_digest "$judge_prompt_digest" \
    --argjson arms "$arms" --argjson runs "$runs" --argjson aggregates "$aggregates" \
    '{protocol_id:"BROCA-LME-V2-RECEIPT-V1",status:"completed",benchmark:{name:"LongMemEval-V2",repository:"https://github.com/xiaowu0162/LongMemEval-V2",upstream_commit:("1"*40),dataset_manifest_file:"pins/dataset.manifest",dataset_manifest_sha256:$dataset_digest,official_harness_manifest_file:"pins/harness.manifest",official_harness_manifest_sha256:$harness_digest,reference_frontier_file:"pins/reference-frontier.json",reference_frontier_sha256:$frontier_digest,tier:"small",question_count:451,arms:$arms},experiment:{symthaea_commit:("5"*40),config_file:"pins/config.json",config_sha256:$config_digest,reader_model:"test-reader",reader_revision:"test-reader-rev",judge_model:"test-judge",judge_revision:"test-judge-rev",reader_prompt_file:"pins/reader.prompt",reader_prompt_sha256:$reader_prompt_digest,judge_prompt_file:"pins/judge.prompt",judge_prompt_sha256:$judge_prompt_digest,memory_context_max_tokens:4096,runs:$runs,aggregates:$aggregates}}' > "$root/receipt.json"
}
expect_pass() {
  local receipt=$1
  if ! bash "$checker" "$receipt"; then echo "FAIL: expected receipt to pass" >&2; exit 1; fi
}
expect_fail() {
  local receipt=$1 label=$2
  if bash "$checker" "$receipt" >/dev/null 2>&1; then echo "FAIL: negative control was accepted: $label" >&2; exit 1; fi
}

make_fixture "$work/valid"
expect_pass "$work/valid/receipt.json"

# Negative control 1: remove one domain run.
jq ' .experiment.runs |= map(select(.arm != "broca" or .domain != "enterprise")) ' "$work/valid/receipt.json" > "$work/valid/missing-domain.json"
expect_fail "$work/valid/missing-domain.json" "missing enterprise run"

# Negative control 2: alter a recorded digest without changing the artifact.
jq ' .experiment.runs[0].sha256 = ("f"*64) ' "$work/valid/receipt.json" > "$work/valid/bad-digest.json"
expect_fail "$work/valid/bad-digest.json" "mismatched artifact digest"

# Negative control 3: mark a completed run skipped.
jq ' .experiment.runs[0].status = "skipped" ' "$work/valid/receipt.json" > "$work/valid/skipped-run.json"
expect_fail "$work/valid/skipped-run.json" "skipped run"

# Negative control 4: forge an aggregate arm that was never run.
jq ' .experiment.aggregates[0].arm = "codex" ' "$work/valid/receipt.json" > "$work/valid/mismatched-aggregate.json"
expect_fail "$work/valid/mismatched-aggregate.json" "arm/aggregate mismatch"

# Negative control 5: path traversal must be refused before hashing.
jq ' .experiment.runs[0].output_file = "../receipt.json" ' "$work/valid/receipt.json" > "$work/valid/path-traversal.json"
expect_fail "$work/valid/path-traversal.json" "path traversal"

# Negative control 6: matching digests cannot hide an unpaired question set.
cp "$work/valid/outputs/no_retrieval_web.jsonl" "$work/valid/outputs/no_retrieval_web.jsonl.bak"
sed -i '1s/web-0001/web-shifted/' "$work/valid/outputs/no_retrieval_web.jsonl"
new_digest=$(sha256sum "$work/valid/outputs/no_retrieval_web.jsonl" | awk '{print $1}')
jq --arg digest "$new_digest" ' .experiment.runs |= map(if .arm == "no_retrieval" and .domain == "web" then .sha256 = $digest else . end) ' "$work/valid/receipt.json" > "$work/valid/unpaired-cases.json"
expect_fail "$work/valid/unpaired-cases.json" "unpaired question IDs"

# Negative control 7: content hashes must bind the experiment config itself.
jq ' .experiment.config_sha256 = ("a"*64) ' "$work/valid/receipt.json" > "$work/valid/config-digest.json"
expect_fail "$work/valid/config-digest.json" "config digest mismatch"

echo "PASS: valid receipt accepted; missing domain, bad digest, skipped run, arm/aggregate mismatch, path traversal, unpaired IDs, and config digest tampering rejected"
