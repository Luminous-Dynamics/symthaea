#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
rustc --edition=2024 --test \
  "$repo_root/scripts/research/sdcb_nuisance_trajectory_semantics.rs" \
  -o "$tmp_dir/sdcb-nuisance-trajectory-tests"
"$tmp_dir/sdcb-nuisance-trajectory-tests"
rustc --edition=2024 \
  "$repo_root/scripts/research/sdcb_nuisance_trajectory_semantics.rs" \
  -o "$tmp_dir/sdcb-nuisance-trajectory"
"$tmp_dir/sdcb-nuisance-trajectory"
