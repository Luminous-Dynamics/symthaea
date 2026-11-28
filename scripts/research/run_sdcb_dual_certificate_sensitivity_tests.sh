#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

rustc --edition=2024 --test \
  "$repo_root/scripts/research/sdcb_dual_certificate_sensitivity.rs" \
  -o "$tmp_dir/sdcb-dual-certificate-sensitivity-tests"
"$tmp_dir/sdcb-dual-certificate-sensitivity-tests"

rustc --edition=2024 \
  "$repo_root/scripts/research/sdcb_dual_certificate_sensitivity.rs" \
  -o "$tmp_dir/sdcb-dual-certificate-sensitivity"
"$tmp_dir/sdcb-dual-certificate-sensitivity"
