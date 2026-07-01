#!/usr/bin/env bash
set -euo pipefail

CRATE="symthaea-visual-compression-probe"
FIXTURES="crates/domains/${CRATE}/fixtures"
WORKDIR="${1:-/tmp/svcp-smoke}"

echo "== doctor =="
cargo run -p "$CRATE" --bin svcp -- doctor

echo "== self-test =="
cargo run -p "$CRATE" --bin svcp -- self-test "$FIXTURES" --json

echo "== pipeline =="
rm -rf "$WORKDIR"
cargo run -p "$CRATE" --bin svcp -- pipeline "$FIXTURES" "$WORKDIR" --json

echo "== validate generated packets =="
for packet in "$WORKDIR"/packets/*.svmp; do
  cargo run -p "$CRATE" --bin svcp -- validate "$packet" --json
done

echo "ok: ${CRATE} smoke workflow completed"
echo "outputs: ${WORKDIR}"
