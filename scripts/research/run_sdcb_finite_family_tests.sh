#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

rustc --edition=2024 --test \
  "$repo_root/scripts/research/sdcb_finite_family_minimax.rs" \
  -o "$tmp_dir/sdcb-finite-family-minimax-tests"

"$tmp_dir/sdcb-finite-family-minimax-tests"

rustc --edition=2024 \
  "$repo_root/scripts/research/sdcb_finite_family_minimax.rs" \
  -o "$tmp_dir/sdcb-finite-family-minimax"

"$tmp_dir/sdcb-finite-family-minimax"
