#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
temporary=$(mktemp)
trap 'rm -f "$temporary"' EXIT
cp "$root/validation/v0_8_reference_results.json" "$temporary"
python3 "$root/validation/generate_v0_8_references.py" >/dev/null
cmp "$temporary" "$root/validation/v0_8_reference_results.json"
