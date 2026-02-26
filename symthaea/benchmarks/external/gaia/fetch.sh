#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/data"
mkdir -p "$data_dir"

cat <<'EOF'
GAIA dev data must be downloaded manually.

Suggested approach (public dev split):
1) Use the official GAIA dataset on Hugging Face (public dev split).
2) Export the dev split to JSON and place it as:
   benchmarks/external/gaia/data/dev.json
3) Create the READY marker:
   touch benchmarks/external/gaia/data/READY
EOF

exit 2
