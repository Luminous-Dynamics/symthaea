#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/data"
mkdir -p "$data_dir"

cat <<'EOF'
HELM setup is manual.

Suggested approach:
1) Clone the official HELM repository.
2) Download HELM Capabilities and HELM Safety scenario data.
3) Place data under:
   benchmarks/external/helm/data/
4) Create the READY marker:
   touch benchmarks/external/helm/data/READY
EOF

exit 2
