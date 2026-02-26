#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"

if [[ ! -d "$script_dir/data/training" || ! -d "$script_dir/data/evaluation" ]]; then
  echo "ARC-AGI-2 data missing. Run fetch.sh and place data in $script_dir/data" >&2
  exit 3
fi

touch "$script_dir/data/READY"

results_file="$repo_root/benchmarks/external/results/arc-agi-2.json"
mkdir -p "$(dirname "$results_file")"

export SYMTHAEA_ARC_DATA_DIR="$script_dir/data"
export SYMTHAEA_ARC_RESULTS_PATH="$results_file"

cd "$repo_root"
cargo run --example benchmark_arc_reasoning --release
