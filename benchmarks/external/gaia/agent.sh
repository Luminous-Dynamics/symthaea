#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bin_path="$repo_root/target/release/symthaea-gaia-agent"

if [[ ! -x "$bin_path" ]]; then
  echo "GAIA agent binary not found at $bin_path" >&2
  echo "Build it with: cargo build --bin symthaea-gaia-agent --release" >&2
  exit 2
fi

exec "$bin_path"
