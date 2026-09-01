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
  fail "working tree must be completely clean before lock hydration/qualification"
fi

HEAD_BEFORE="$(git rev-parse HEAD)"
LOCK_BEFORE="$(sha256sum Cargo.lock | awk '{print $1}')"
BACKUP="$(mktemp "${TMPDIR:-/tmp}/cogsec-Cargo.lock.XXXXXX")"
METADATA="$(mktemp "${TMPDIR:-/tmp}/cogsec-metadata.XXXXXX.json")"
cp Cargo.lock "$BACKUP"

KEEP_LOCK=0
cleanup() {
  if [[ $KEEP_LOCK -ne 1 ]]; then
    cp "$BACKUP" Cargo.lock
  fi
  rm -f "$BACKUP" "$METADATA"
}
interrupt() {
  exit 130
}
trap cleanup EXIT
trap interrupt INT TERM

printf 'CogSec Cargo.lock hydration / qualification\n'
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
  printf '\nAll CogSec package entries are present; verifying committed lock state.\n'
  cargo metadata --locked --format-version=1 > "$METADATA"
else
  printf '\nCogSec entries are missing; hydrating Cargo.lock with pinned Cargo %s...\n' "$EXPECTED_RUST"
  # Intentionally omit --locked exactly once. Cargo is the only authority allowed
  # to rewrite Cargo.lock; this script never synthesizes package/dependency entries.
  cargo metadata --format-version=1 > "$METADATA"
fi

# Cargo metadata must not have changed any tracked file except Cargo.lock.
mapfile -t CHANGED_PATHS < <(git status --porcelain=v1 --untracked-files=all | sed -E 's/^...//')
for path in "${CHANGED_PATHS[@]}"; do
  [[ "$path" == "Cargo.lock" ]] || fail "Cargo changed unexpected path: $path"
done

# CogSec hydration is deliberately narrower than general dependency maintenance.
# Existing lockfile material may not disappear or be rewritten under cover of this
# repair. A deletion in the diff means the repository needs a separately reviewed
# broader lock update, so restore the original lock and fail closed.
LOCK_DELETIONS="$(git diff --unified=0 -- Cargo.lock | grep -E '^-[^-]' || true)"
if [[ -n "$LOCK_DELETIONS" ]]; then
  printf '%s\n' "$LOCK_DELETIONS" >&2
  fail "hydration is not additive-only; refusing unrelated Cargo.lock churn"
fi

for package in "${COGSEC_PACKAGES[@]}"; do
  grep -Fqx "name = \"$package\"" Cargo.lock || \
    fail "Cargo.lock still lacks package entry: $package"
done

# Re-enter the strict mode CI uses. If Cargo would still mutate the hydrated lock,
# stop before presenting it as a candidate for commit.
cargo metadata --locked --format-version=1 > /dev/null
git diff --check -- Cargo.lock

LOCK_AFTER="$(sha256sum Cargo.lock | awk '{print $1}')"
printf '\nLock consistency verified.\n'
printf 'before: %s\n' "$LOCK_BEFORE"
printf 'after:  %s\n' "$LOCK_AFTER"

if [[ "$LOCK_BEFORE" != "$LOCK_AFTER" ]]; then
  # Do not run focused qualification yet: its first gate intentionally requires
  # `git diff --exit-code -- Cargo.lock`, so qualifying an uncommitted lock would
  # weaken the evidence boundary. Preserve only the Cargo-generated lock diff.
  KEEP_LOCK=1
  trap - EXIT INT TERM
  rm -f "$BACKUP" "$METADATA"

  printf '\nHYDRATED: Cargo.lock changed additively and passes locked metadata validation.\n'
  printf 'Review the Cargo.lock diff and commit it without editing generated entries.\n'
  printf 'Then rerun this same command from the clean committed head; the second pass\n'
  printf 'will execute the canonical seven-gate CogSec qualification.\n'
  exit 0
fi

printf '\nCargo.lock is already committed and stable. Running focused qualification...\n'
bash scripts/cogsec-focused-qualification.sh

[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || \
  fail "qualification mutated the clean worktree"
[[ "$(git rev-parse HEAD)" == "$HEAD_BEFORE" ]] || \
  fail "HEAD changed during qualification"

KEEP_LOCK=1
trap - EXIT INT TERM
rm -f "$BACKUP" "$METADATA"

printf '\nPASS: committed CogSec lock state passed focused qualification.\n'
