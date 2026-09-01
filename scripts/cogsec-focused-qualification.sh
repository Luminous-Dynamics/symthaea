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
PACKAGE_SET="symthaea-cogsec,symthaea-cogsec-evidence,symthaea-cogsec-qualification,symthaea-cogsec-shadow-runtime"

RECEIPT_OUT="${COGSEC_RECEIPT_OUT:-}"
HEAD_SHA="$(git rev-parse HEAD)"
TREE_SHA="$(git rev-parse 'HEAD^{tree}')"
RUSTC_VERSION="$(rustc --version 2>/dev/null || printf 'unavailable')"
CARGO_VERSION="$(cargo --version 2>/dev/null || printf 'unavailable')"
LOCK_SHA256="$(sha256sum Cargo.lock | awk '{print $1}')"
MANIFEST_SHA256="$(sha256sum Cargo.toml | awk '{print $1}')"
SCRIPT_SHA256="$(sha256sum scripts/cogsec-focused-qualification.sh | awk '{print $1}')"
STATUS="FAIL"
CURRENT_GATE="bootstrap"
LAST_COMPLETED_GATE="none"

write_receipt() {
  local exit_code="$1"
  [[ -n "$RECEIPT_OUT" ]] || return 0
  mkdir -p "$(dirname "$RECEIPT_OUT")" 2>/dev/null || return 0
  {
    printf 'schema_version\t2\n'
    printf 'status\t%s\n' "$STATUS"
    printf 'exit_code\t%s\n' "$exit_code"
    printf 'head\t%s\n' "$HEAD_SHA"
    printf 'tree\t%s\n' "$TREE_SHA"
    printf 'cargo_lock_sha256\t%s\n' "$LOCK_SHA256"
    printf 'workspace_manifest_sha256\t%s\n' "$MANIFEST_SHA256"
    printf 'qualification_script_sha256\t%s\n' "$SCRIPT_SHA256"
    printf 'package_set\t%s\n' "$PACKAGE_SET"
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
printf 'tree: %s\n' "$TREE_SHA"
printf 'Cargo.lock SHA-256: %s\n' "$LOCK_SHA256"
printf 'rustc: %s\n' "$RUSTC_VERSION"
printf 'cargo: %s\n' "$CARGO_VERSION"

CURRENT_GATE="worktree-cleanliness"
printf '\n[pre] committed-worktree cleanliness\n'
DIRTY_STATE="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "$DIRTY_STATE" ]]; then
  printf '%s\n' "$DIRTY_STATE" >&2
  printf 'ERROR: focused qualification requires a clean committed worktree\n' >&2
  exit 1
fi
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="metadata-lockfile"
printf '\n[1/7] workspace metadata + lockfile consistency\n'
cargo metadata --locked --format-version=1 > /dev/null
git diff --exit-code -- Cargo.lock
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="rustfmt"
printf '\n[2/7] rustfmt\n'
cargo fmt --check -p symthaea "${PACKAGES[@]}"
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="cargo-check"
printf '\n[3/7] cargo check\n'
cargo check --locked "${PACKAGES[@]}"
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="package-tests"
printf '\n[4/7] CogSec package tests\n'
cargo test --locked "${PACKAGES[@]}"
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="legacy-control-determinism"
printf '\n[5/7] legacy S0/S1/S2 control determinism\n'
cargo test --locked -p symthaea --lib 'mind::tests::cogsec_shadow_control::'
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="doc-tests"
printf '\n[6/7] documentation tests\n'
cargo test --locked --doc "${PACKAGES[@]}"
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="clippy"
printf '\n[7/7] clippy -D warnings\n'
cargo clippy --locked --all-targets "${PACKAGES[@]}" -- -D warnings
LAST_COMPLETED_GATE="$CURRENT_GATE"

CURRENT_GATE="tracked-state-postcondition"
printf '\n[post] tracked repository state unchanged\n'
git diff --exit-code
git diff --cached --exit-code
[[ "$(git rev-parse HEAD)" == "$HEAD_SHA" ]] || {
  printf 'ERROR: HEAD changed during focused qualification\n' >&2
  exit 1
}
LAST_COMPLETED_GATE="$CURRENT_GATE"

STATUS="PASS"
CURRENT_GATE="complete"
printf '\nPASS: CogSec focused qualification\n'
