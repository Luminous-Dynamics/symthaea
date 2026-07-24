#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
python3 "$ROOT/scripts/reference_family_analysis.py" \
  "$ROOT/tests/fixtures/cognition-v82/family_statistics_input.json" \
  "$TMP"
cmp -s "$TMP" "$ROOT/tests/fixtures/cognition-v82/family_statistics_expected.json"
echo "V8.2 independent family-statistics reference passed"
