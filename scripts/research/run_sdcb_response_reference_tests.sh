#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SOURCES=(
  "scripts/research/sdcb_exponential_response.rs"
  "scripts/research/sdcb_response_interval_bridge.rs"
)

for source in "${SOURCES[@]}"; do
  binary="$TMP_DIR/$(basename "${source%.rs}")"
  printf '\n==> rustc --edition=2024 --test %s\n' "$source"
  rustc --edition=2024 --test "$ROOT/$source" -o "$binary"
  printf '\n==> running %s\n' "$source"
  "$binary"
done
