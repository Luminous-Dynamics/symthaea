#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

rustc --edition=2024 --test \
  "$repo_root/scripts/research/sdcb_gaussian_process_separation.rs" \
  -o "$tmp_dir/sdcb-gaussian-process-separation-tests"

"$tmp_dir/sdcb-gaussian-process-separation-tests"
