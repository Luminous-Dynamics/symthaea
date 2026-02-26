#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/data"
mkdir -p "$data_dir"

cat <<'EOF'
OSWorld-Verified setup is manual.

Suggested approach:
1) Clone the official OSWorld repo and download the Verified task set.
2) Build a NixOS VM image for the benchmark environment.
3) Place task data and assets under:
   benchmarks/external/osworld-verified/data/
4) Create the READY marker:
   touch benchmarks/external/osworld-verified/data/READY
EOF

exit 2
