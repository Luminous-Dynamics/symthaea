#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"

data_file="$script_dir/data/mini.json"
if [[ ! -f "$data_file" ]]; then
  echo "SWE-bench data missing. Run fetch.sh and place mini.json in $script_dir/data" >&2
  exit 3
fi

touch "$script_dir/data/READY"

cd "$script_dir"
./runner.py --input "$data_file" --output "$repo_root/benchmarks/external/results/swe-bench-verified-mini.json"
