#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/crates/domains/symthaea-quantum-comp/src"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

modules=(
  trajectory_prior.rs
  trajectory_history.rs
  trajectory_budget.rs
  matched_evaluation.rs
)

for module in "${modules[@]}"; do
  input="$SRC/$module"
  output="$TMP/${module%.rs}-tests"
  echo "==> rustc --test $module"
  rustc --edition=2024 --test "$input" -o "$output"
  "$output" --test-threads=1
done

# Compile the QOBLIB manifest parser against the same dependency-free verifier
# source. This covers the manifest parser and verifier without Cargo workspace
# discovery; it does not exercise Git checkout/file-reading integration itself.
MANIFEST_TOOL="$ROOT/crates/domains/symthaea-quantum-comp/examples/qoblib_mis_manifest.rs"
MANIFEST_TESTS="$TMP/qoblib-mis-manifest-tests"
echo "==> rustc --test examples/qoblib_mis_manifest.rs"
rustc --edition=2024 --test "$MANIFEST_TOOL" -o "$MANIFEST_TESTS"
"$MANIFEST_TESTS" --test-threads=1
