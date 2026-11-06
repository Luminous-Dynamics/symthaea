#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

manifest="tools/knowledge-benchmark-harness/Cargo.toml"
revision="${SUBJECT_REVISION:-$(git rev-parse HEAD)}"
reports_dir="${REPORTS_DIR:-target/knowledge-benchmark-reports}"
mkdir -p "$reports_dir"

run_validator() {
  local kind="$1"
  local schema="$2"
  local input="$3"
  local output="$4"

  echo "==> Validating kind=$kind input=$input"
  cargo run --locked --manifest-path "$manifest" --bin knowledge-benchmark-harness -- \
    --kind "$kind" \
    --schema "$schema" \
    --input "$input" \
    --subject-revision "$revision" \
    --out "$output"
}

cargo fmt --manifest-path "$manifest" -- --check
rustfmt --edition 2024 --check crates/domains/symthaea-epistemic-types/src/execution_receipt.rs
cargo test --locked --manifest-path "$manifest"

echo "==> Interpreting the pinned EPI-013R receipt corpus (candidate-only)"
cargo run --locked --manifest-path "$manifest" --bin interpret-execution-receipts -- \
  --receipt-corpus crates/domains/symthaea-epistemic-types/tests/fixtures/EPI_013R_TOOL_EXECUTION_RECEIPTS_V0_1.json \
  --profile-corpus crates/domains/symthaea-epistemic-types/tests/fixtures/EPI_013_OSINT_TOOL_CAPABILITY_PROFILES_V0_1.json \
  --expected-profile-sha256 d499b680d6461dbffed9b4ea053d3758ee4bd4e7d3e26658d26334acddc35cf7 \
  --out "$reports_dir/osint-execution-interpretations.json"

grep -Fq '"receipts_seen": 8' "$reports_dir/osint-execution-interpretations.json"
grep -Fq '"receipts_interpreted": 8' "$reports_dir/osint-execution-interpretations.json"
grep -Fq '"passed": true' "$reports_dir/osint-execution-interpretations.json"
grep -Fq '"authority_ceiling": "CandidateAnalysisOnly"' "$reports_dir/osint-execution-interpretations.json"

echo "==> Validating candidate-only retrieval-plan anchor contract"
cargo run --locked --manifest-path "$manifest" --bin validate-retrieval-plan -- \
  --schema benchmarks/retrieval-plan/v0/plan.schema.json \
  --input benchmarks/retrieval-plan/v0/valid-anchor-plan.json \
  > "$reports_dir/retrieval-plan-valid.json"

if cargo run --locked --manifest-path "$manifest" --bin validate-retrieval-plan -- \
  --schema benchmarks/retrieval-plan/v0/plan.schema.json \
  --input benchmarks/retrieval-plan/v0/invalid-dropped-anchor-plan.json \
  > "$reports_dir/retrieval-plan-invalid.json"; then
  echo "ERROR: dropped-anchor negative control was incorrectly accepted" >&2
  exit 1
fi
grep -Fq '"recommended_action": "reject_rewrite_and_use_original_query"' \
  "$reports_dir/retrieval-plan-invalid.json"
echo "Dropped-anchor negative control rejected as expected"

run_validator case \
  benchmarks/esoterica-bench/v0/case.schema.json \
  benchmarks/esoterica-bench/v0/cases.jsonl \
  "$reports_dir/esoterica-cases.json"

run_validator claim \
  benchmarks/esoterica-bench/v0/claim.schema.json \
  benchmarks/esoterica-bench/v0/claim-fixtures.jsonl \
  "$reports_dir/claim-fixtures.json"

run_validator case \
  benchmarks/mythology-bench/v0/case.schema.json \
  benchmarks/mythology-bench/v0/cases.jsonl \
  "$reports_dir/mythology-cases.json"

run_validator pair \
  benchmarks/mythology-bench/v0/metamorphic-pair.schema.json \
  benchmarks/mythology-bench/v0/metamorphic-pairs.jsonl \
  "$reports_dir/metamorphic-pairs.json"

run_validator source \
  benchmarks/source-catalog/v0/source.schema.json \
  benchmarks/source-catalog/v0/sources.jsonl \
  "$reports_dir/source-catalog.json"

run_validator source-pack \
  benchmarks/benchmark-runs/v0/source-pack.schema.json \
  benchmarks/benchmark-runs/v0/synthetic-source-pack.jsonl \
  "$reports_dir/source-pack.json"

run_validator annotation \
  benchmarks/knowledge-annotation/v0/annotation.schema.json \
  benchmarks/knowledge-annotation/v0/synthetic-fixtures.jsonl \
  "$reports_dir/annotation-fixtures.json"

cargo run --locked --manifest-path "$manifest" --bin score-annotation-agreement -- \
  --annotations benchmarks/knowledge-annotation/v0/synthetic-fixtures.jsonl \
  --out "$reports_dir/annotation-agreement.json"

run_validator run \
  benchmarks/benchmark-runs/v0/comparison-plan.schema.json \
  benchmarks/benchmark-runs/v0/synthetic-plans.jsonl \
  "$reports_dir/comparison-plans.json"

run_validator pair-result \
  benchmarks/benchmark-runs/v0/pair-result.schema.json \
  benchmarks/benchmark-runs/v0/synthetic-pair-results.jsonl \
  "$reports_dir/pair-results.json"

cargo run --locked --manifest-path "$manifest" --bin score-paired-results -- \
  --plan benchmarks/benchmark-runs/v0/synthetic-plans.jsonl \
  --pairs benchmarks/mythology-bench/v0/metamorphic-pairs.jsonl \
  --results benchmarks/benchmark-runs/v0/synthetic-pair-results.jsonl \
  --case-manifest benchmarks/mythology-bench/v0/metamorphic-pairs.jsonl \
  --source-catalog benchmarks/source-catalog/v0/sources.jsonl \
  --source-pack benchmarks/benchmark-runs/v0/synthetic-source-pack.jsonl \
  --out "$reports_dir/paired-intervention-score.json"

run_validator graph \
  benchmarks/mythology-bench/v0/narrative-graph.schema.json \
  benchmarks/mythology-bench/v0/narrative-graph-fixtures.jsonl \
  "$reports_dir/narrative-graphs.json"

echo "Reports written to $reports_dir"
