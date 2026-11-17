#!/usr/bin/env bash
# Verify each isolated Symthaea Subterra reference package.
# Run from any directory inside a Symthaea checkout:
#   bash crates/domains/symthaea-subterranean/research/verify-subterra-reference-packages.sh
#
# These packages are nested Cargo workspaces, so root cargo test --workspace
# does not include them.

set -euo pipefail

for tool in cargo rustc rustfmt; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'ERROR: %s is not installed; no full Subterra Rust verification.\n' "$tool" >&2
    exit 127
  fi
done

ROOT="$(git rev-parse --show-toplevel)"
BASE="crates/domains/symthaea-subterranean/research"
CRATES=(
  "subterra-voi-kernel"
  "subterra-survival-planner"
  "subterra-robust-voi"
  "subterra-likelihood-calibration"
  "subterra-dependence-audit"
  "subterra-evidence-adapter"
)

printf 'Subterra reference verification\nRepository: %s\n' "$ROOT"
printf 'Rust: %s\nCargo: %s\n\n' "$(rustc --version)" "$(cargo --version)"

for crate in "${CRATES[@]}"; do
  manifest="$ROOT/$BASE/$crate/Cargo.toml"
  if [[ ! -f "$manifest" ]]; then
    printf 'ERROR: missing manifest: %s\n' "$manifest" >&2
    exit 2
  fi
done

printf '== Formatting ==\n'
for crate in "${CRATES[@]}"; do
  printf '\n-- %s --\n' "$crate"
  cargo fmt --manifest-path "$ROOT/$BASE/$crate/Cargo.toml" -- --check
done

printf '\n== Tests ==\n'
for crate in "${CRATES[@]}"; do
  printf '\n-- %s --\n' "$crate"
  cargo test --manifest-path "$ROOT/$BASE/$crate/Cargo.toml" --all-targets
done

printf '\n== Strict Clippy ==\n'
for crate in "${CRATES[@]}"; do
  printf '\n-- %s --\n' "$crate"
  cargo clippy --manifest-path "$ROOT/$BASE/$crate/Cargo.toml" --all-targets -- -D warnings
done

printf '\nPASS: six standalone Subterra packages passed formatting, tests and strict Clippy.\n'
printf 'This does not establish physical, analogue, flight or planetary qualification.\n'
