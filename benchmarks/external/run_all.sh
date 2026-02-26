#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

env_file="${SYMTHAEA_BENCH_ENV:-$script_dir/config/cyborg_mind.env}"
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$env_file"
  set +a
  echo "Loaded benchmark env: $env_file"
fi

skip_ready="${SYMTHAEA_BENCH_SKIP_READY:-0}"

if [[ "$skip_ready" == "1" ]]; then
  python "$script_dir/run_external.py" --all --skip-ready-check
else
  python "$script_dir/run_external.py" --check --all
  python "$script_dir/run_external.py" --all
fi

python "$script_dir/collect_results.py" --output "$script_dir/results/aggregate.json"

if [[ -f "$script_dir/results/aggregate.json" ]]; then
  echo "Wrote aggregate results to $script_dir/results/aggregate.json"
fi
