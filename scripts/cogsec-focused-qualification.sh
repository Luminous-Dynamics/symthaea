#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

PACKAGES=(
  -p symthaea-cogsec
  -p symthaea-cogsec-evidence
  -p symthaea-cogsec-qualification
)

printf 'CogSec focused qualification\n'
printf 'HEAD: %s\n' "$(git rev-parse HEAD)"
printf 'rustc: %s\n' "$(rustc --version)"
printf 'cargo: %s\n' "$(cargo --version)"

printf '\n[1/6] workspace metadata + lockfile consistency\n'
cargo metadata --locked --format-version=1 > /dev/null

git diff --exit-code -- Cargo.lock

printf '\n[2/6] rustfmt\n'
cargo fmt --check "${PACKAGES[@]}"

printf '\n[3/6] cargo check\n'
cargo check --locked "${PACKAGES[@]}"

printf '\n[4/6] package tests\n'
cargo test --locked "${PACKAGES[@]}"

printf '\n[5/6] documentation tests\n'
cargo test --locked --doc "${PACKAGES[@]}"

printf '\n[6/6] clippy -D warnings\n'
cargo clippy --locked --all-targets "${PACKAGES[@]}" -- -D warnings

printf '\nPASS: CogSec focused qualification\n'
