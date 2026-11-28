#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
rustc --edition=2024 --test \
  "$repo_root/scripts/research/sdcb_minimax_nuisance_induction.rs" \
  -o "$tmp_dir/sdcb-minimax-nuisance-induction-tests"
"$tmp_dir/sdcb-minimax-nuisance-induction-tests"
rustc --edition=2024 \
  "$repo_root/scripts/research/sdcb_minimax_nuisance_induction.rs" \
  -o "$tmp_dir/sdcb-minimax-nuisance-induction"
"$tmp_dir/sdcb-minimax-nuisance-induction"
