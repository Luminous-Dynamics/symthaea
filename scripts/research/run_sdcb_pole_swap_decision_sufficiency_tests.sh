#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
rustc --edition=2024 --test \
  "$repo_root/scripts/research/sdcb_pole_swap_decision_sufficiency.rs" \
  -o "$tmp_dir/sdcb-pole-swap-decision-tests"
"$tmp_dir/sdcb-pole-swap-decision-tests"
rustc --edition=2024 \
  "$repo_root/scripts/research/sdcb_pole_swap_decision_sufficiency.rs" \
  -o "$tmp_dir/sdcb-pole-swap-decision"
"$tmp_dir/sdcb-pole-swap-decision"
