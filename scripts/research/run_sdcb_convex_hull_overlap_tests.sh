#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

rustc --edition=2024 --test \
  "$repo_root/scripts/research/sdcb_convex_hull_overlap.rs" \
  -o "$tmp_dir/sdcb-convex-hull-tests"
"$tmp_dir/sdcb-convex-hull-tests"

rustc --edition=2024 \
  "$repo_root/scripts/research/sdcb_convex_hull_overlap.rs" \
  -o "$tmp_dir/sdcb-convex-hull"
"$tmp_dir/sdcb-convex-hull"
