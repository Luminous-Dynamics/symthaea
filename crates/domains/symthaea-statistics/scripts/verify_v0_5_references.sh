#!/usr/bin/env sh
set -eu

python3 validation/generate_v0_5_references.py
if ! git diff --exit-code -- validation/v0_5_reference_results.json; then
    echo "v0.5 reference results changed; inspect and commit intentionally" >&2
    exit 1
fi
