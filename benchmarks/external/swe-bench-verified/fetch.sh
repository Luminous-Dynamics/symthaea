#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/data"
mkdir -p "$data_dir"

cat <<'EOF'
SWE-bench Verified (Mini) must be downloaded manually.

Suggested approach:
1) Download the official SWE-bench Verified dataset.
2) Extract the Mini subset as:
   benchmarks/external/swe-bench-verified/data/mini.json
3) Create the READY marker:
   touch benchmarks/external/swe-bench-verified/data/READY
EOF

exit 2
