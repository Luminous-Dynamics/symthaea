#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/data"
mkdir -p "$data_dir"

cat <<'EOF'
ARC-AGI-2 data must be downloaded manually.

Steps:
1) Download the ARC-AGI-2 public training + evaluation data from the ARC Prize site.
2) Place the JSON task folders as:
   benchmarks/external/arc-agi-2/data/training/
   benchmarks/external/arc-agi-2/data/evaluation/
3) Create the READY marker:
   touch benchmarks/external/arc-agi-2/data/READY
EOF

exit 2
