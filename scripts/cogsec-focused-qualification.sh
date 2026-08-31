#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

PACKAGES=(
  -p symthaea-cogsec
  -p symthaea-cogsec-evidence
  -p symthaea-cogsec-qualification
  -p symthaea-cogsec-shadow-runtime
)

RECEIPT_OUT="${COGSEC_RECEIPT_OUT:-}"
HEAD_SHA="$(git rev-parse HEAD)"
RUSTC_VERSION="$(rustc --version 2>/dev/null || printf 'unavailable')"
CARGO_VERSION="$(cargo --version 2>/dev/null || printf 'unavailable')"
STATUS="FAIL"
CURRENT_GATE="bootstrap"
LAST_COMPLETED_GATE="none"

write_receipt() {
  local exit_code="$1"
  [[ -n "$RECEIPT_OUT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT_OUT")" 2>/dev/null || return 0
  {
    printf 'schema_version\t1\n'
    printf 'status\t%s\n' "$STATUS"
    printf 'exit_code\t%s\n' "$exit_code"
    printf 'head\t%s\n' "$HEAD_SHA"
    printf 'rustc\t%s\n' "$RUSTC_VERSION"
    printf 'cargo\t%s\n' "$CARGO_VERSION"
    printf 'current_gate\t%s\n' "$CURRENT_GATE"
    printf 'last_completed_gate\t%s\n' "$LAST_COMPLETED_GATE"
  } > "$RECEIPT_OUT" || true
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    STATUS="PASS"
    CURRENT_GATE="complete"
  fi
  write_receipt "$exit_code"
}
trap on_exit EXIT

printf 'CogSec focused qualification\n'
printf 'HEAD: %s\n' "$HEAD_SHA"
printf 'rustc: %s\n' "$RUSTC_VERSION"
printf 'cargo: %s\n' "$CARGO_VERSION"

CURRENT_GATE="metadata-lockfile"
printf '\n[1/6] workspace metadata + lockfile consistency\n'
cargo metadata --locked --format-version=1 > /dev/null
git diff --exit-code -- Cargo.lock
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="rustfmt"
printf '\n[2/6] rustfmt\n'
cargo fmt --check "${PACKAGES[@]}"
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="cargo-check"
printf '\n[3/6] cargo check\n'
cargo check --locked "${PACKAGES[@]}"
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="package-tests"
printf '\n[4/6] package tests\n'
cargo test --locked "${PACKAGES[@]}"
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="doc-tests"
printf '\n[5/6] documentation tests\n'
cargo test --locked --doc "${PACKAGES[@]}"
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="clippy"
printf '\n[6/6] clippy -D warnings\n'
cargo clippy --locked --all-targets "${PACKAGES[@]}" -- -D warnings
LAST_COMPLETED_GATE="$CURRENT_GATE"

STATUS="PASS"
CURRENT_GATE="complete"
printf '\nPASS: CogSec focused qualification\n'
