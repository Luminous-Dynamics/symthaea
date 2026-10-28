#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/crates/domains/symthaea-quantum-comp/src"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

modules=(
  trajectory_prior.rs
  trajectory_history.rs
  trajectory_budget.rs
  matched_evaluation.rs
)

for module in "${modules[@]}"; do
  input="$SRC/$module"
  output="$TMP/${module%.rs}-tests"
  echo "==> rustc --test $module"
  rustc --edition=2024 --test "$input" -o "$output"
  "$output" --test-threads=1
done
