#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SOURCE="scripts/research/sdcb_decision_value_bounds.rs"
BINARY="$TMP_DIR/sdcb_decision_value_bounds"

printf '\n==> rustc --edition=2024 --test %s\n' "$SOURCE"
rustc --edition=2024 --test "$ROOT/$SOURCE" -o "$BINARY"

printf '\n==> running %s\n' "$SOURCE"
"$BINARY"
