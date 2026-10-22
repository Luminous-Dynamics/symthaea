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
  cargo run --locked --manifest-path "$manifest" -- \
    --kind "$kind" \
    --schema "$schema" \
    --input "$input" \
    --subject-revision "$revision" \
    --out "$output"
}

cargo fmt --manifest-path "$manifest" -- --check
cargo test --locked --manifest-path "$manifest"

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
