#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"

if [[ ! -f "$script_dir/data/READY" ]]; then
  echo "OSWorld data missing. Run fetch.sh and set up NixOS VM." >&2
  exit 3
fi

cd "$script_dir"
./runner.py --output "$repo_root/benchmarks/external/results/osworld-verified.json"
