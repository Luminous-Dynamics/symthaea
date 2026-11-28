#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

rustc --edition=2021 --test "$ROOT/scripts/research/sdcb_false_unique_risk.rs" -o "$TMP/tests"
"$TMP/tests"

rustc --edition=2021 "$ROOT/scripts/research/sdcb_false_unique_risk.rs" -o "$TMP/report"
"$TMP/report"
