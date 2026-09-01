#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

EXPECTED_RUST="1.96.0"
COGSEC_PACKAGES=(
  symthaea-cogsec
  symthaea-cogsec-evidence
  symthaea-cogsec-qualification
  symthaea-cogsec-shadow-runtime
)

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

version_number() {
  # rustc/cargo version output is stable enough for the first whitespace-delimited
  # version token, e.g. `rustc 1.96.0 (...)` and `cargo 1.96.0 (...)`.
  "$1" --version | awk '{print $2}'
}

RUSTC_VERSION="$(version_number rustc)"
CARGO_VERSION="$(version_number cargo)"

[[ "$RUSTC_VERSION" == "$EXPECTED_RUST" ]] || \
  fail "rustc $EXPECTED_RUST required; found $RUSTC_VERSION"
[[ "$CARGO_VERSION" == "$EXPECTED_RUST" ]] || \
  fail "cargo $EXPECTED_RUST required; found $CARGO_VERSION"

[[ -f Cargo.lock ]] || fail "Cargo.lock is missing"
[[ -f scripts/cogsec-focused-qualification.sh ]] || \
  fail "scripts/cogsec-focused-qualification.sh is missing"

# Hydration is evidence-producing maintenance. Refuse any tracked or untracked
# contamination because workspace globs can make even an untracked crate affect
# Cargo metadata and therefore the lockfile.
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  fail "working tree must be completely clean before lock hydration"
fi

HEAD_BEFORE="$(git rev-parse HEAD)"
LOCK_BEFORE="$(sha256sum Cargo.lock | awk '{print $1}')"
BACKUP="$(mktemp "${TMPDIR:-/tmp}/cogsec-Cargo.lock.XXXXXX")"
METADATA="$(mktemp "${TMPDIR:-/tmp}/cogsec-metadata.XXXXXX.json")"
cp Cargo.lock "$BACKUP"

SUCCESS=0
restore_on_exit() {
  if [[ $SUCCESS -ne 1 ]]; then
    cp "$BACKUP" Cargo.lock
    printf '\nHydration failed; restored the original Cargo.lock.\n' >&2
  fi
  rm -f "$BACKUP" "$METADATA"
}
interrupt() {
  exit 130
}
trap restore_on_exit EXIT
trap interrupt INT TERM

printf 'CogSec Cargo.lock hydration + qualification\n'
printf 'HEAD:   %s\n' "$HEAD_BEFORE"
printf 'rustc:  %s\n' "$(rustc --version)"
printf 'cargo:  %s\n' "$(cargo --version)"
printf 'lock:   %s\n' "$LOCK_BEFORE"

all_present=1
for package in "${COGSEC_PACKAGES[@]}"; do
  if ! grep -Fqx "name = \"$package\"" Cargo.lock; then
    all_present=0
  fi
done

if [[ $all_present -eq 1 ]]; then
  printf '\nAll CogSec package entries are already present; verifying locked metadata.\n'
  cargo metadata --locked --format-version=1 > "$METADATA"
else
  printf '\nHydrating Cargo.lock with pinned Cargo %s...\n' "$EXPECTED_RUST"
  # Intentionally omit --locked exactly once. Cargo is the only authority allowed
  # to rewrite Cargo.lock; this script never synthesizes package/dependency entries.
  cargo metadata --format-version=1 > "$METADATA"
fi

# Cargo metadata must not have changed any tracked file except Cargo.lock.
mapfile -t CHANGED_TRACKED < <(git status --porcelain=v1 --untracked-files=all | sed -E 's/^...//')
for path in "${CHANGED_TRACKED[@]}"; do
  [[ "$path" == "Cargo.lock" ]] || fail "hydration changed unexpected path: $path"
done

for package in "${COGSEC_PACKAGES[@]}"; do
  grep -Fqx "name = \"$package\"" Cargo.lock || \
    fail "Cargo.lock still lacks package entry: $package"
done

# Re-enter the strict mode that CI uses. If Cargo would still mutate the lock,
# qualification stops here.
cargo metadata --locked --format-version=1 > /dev/null
git diff --check -- Cargo.lock

LOCK_AFTER="$(sha256sum Cargo.lock | awk '{print $1}')"
printf '\nLock hydration verified.\n'
printf 'before: %s\n' "$LOCK_BEFORE"
printf 'after:  %s\n' "$LOCK_AFTER"

printf '\nRunning the canonical seven-gate CogSec qualification...\n'
bash scripts/cogsec-focused-qualification.sh

# Qualification must not mutate tracked state beyond the intentional lock update.
mapfile -t FINAL_CHANGED < <(git status --porcelain=v1 --untracked-files=all | sed -E 's/^...//')
for path in "${FINAL_CHANGED[@]}"; do
  [[ "$path" == "Cargo.lock" ]] || fail "qualification changed unexpected path: $path"
done

[[ "$(git rev-parse HEAD)" == "$HEAD_BEFORE" ]] || fail "HEAD changed during qualification"

SUCCESS=1
trap - EXIT INT TERM
rm -f "$BACKUP" "$METADATA"

printf '\nPASS: CogSec lock hydration and focused qualification completed.\n'
if [[ "$LOCK_BEFORE" == "$LOCK_AFTER" ]]; then
  printf 'Cargo.lock was already qualified; no lockfile change is required.\n'
else
  printf 'Review and commit only the Cargo.lock diff produced by pinned Cargo %s.\n' "$EXPECTED_RUST"
fi
