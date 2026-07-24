#!/usr/bin/env sh
set -eu

snapshot=$(mktemp)
trap 'rm -f "$snapshot"' EXIT
cp validation/v0_7_reference_results.json "$snapshot"
python3 validation/generate_v0_7_references.py >/dev/null
cmp "$snapshot" validation/v0_7_reference_results.json
printf '%s\n' 'v0.7 independent reference values match the committed snapshot'
