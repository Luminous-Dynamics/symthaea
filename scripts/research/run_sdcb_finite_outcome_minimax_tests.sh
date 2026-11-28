#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

rustc --edition=2024 --test \
  "$repo_root/scripts/research/sdcb_finite_outcome_minimax.rs" \
  -o "$tmp_dir/sdcb-finite-outcome-tests"
"$tmp_dir/sdcb-finite-outcome-tests"

rustc --edition=2024 \
  "$repo_root/scripts/research/sdcb_finite_outcome_minimax.rs" \
  -o "$tmp_dir/sdcb-finite-outcome"
"$tmp_dir/sdcb-finite-outcome"
