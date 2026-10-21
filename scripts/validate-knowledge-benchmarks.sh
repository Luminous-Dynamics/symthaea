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

run_validator graph \
  benchmarks/mythology-bench/v0/narrative-graph.schema.json \
  benchmarks/mythology-bench/v0/narrative-graph-fixtures.jsonl \
  "$reports_dir/narrative-graphs.json"

echo "Reports written to $reports_dir"
