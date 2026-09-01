#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

EXPECTED_RUST="1.96.0"
OUT_DIR="${COGSEC_LOCK_DIAGNOSTIC_OUT:-target/cogsec-lock-diagnostic}"
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
  "$1" --version | awk '{print $2}'
}

RUSTC_VERSION="$(version_number rustc)"
CARGO_VERSION="$(version_number cargo)"
[[ "$RUSTC_VERSION" == "$EXPECTED_RUST" ]] || \
  fail "rustc $EXPECTED_RUST required; found $RUSTC_VERSION"
[[ "$CARGO_VERSION" == "$EXPECTED_RUST" ]] || \
  fail "cargo $EXPECTED_RUST required; found $CARGO_VERSION"

[[ -f Cargo.lock ]] || fail "Cargo.lock is missing"

# Diagnostic runs must start from committed repository state. The output directory
# is created only after this check so its own artifacts cannot make the tree dirty.
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  fail "working tree must be completely clean before lock diagnosis"
fi

mkdir -p "$OUT_DIR"
RECEIPT="$OUT_DIR/receipt.tsv"
LOCKED_STDERR="$OUT_DIR/locked-metadata.stderr"
CANDIDATE_LOCK="$OUT_DIR/Cargo.lock.candidate"
CANDIDATE_PATCH="$OUT_DIR/Cargo.lock.candidate.patch"

HEAD_SHA="$(git rev-parse HEAD)"
LOCK_BEFORE="$(sha256sum Cargo.lock | awk '{print $1}')"
STATUS="FAIL"
LOCK_AFTER="$LOCK_BEFORE"

write_receipt() {
  local exit_code="$1"
  {
    printf 'schema_version\t1\n'
    printf 'status\t%s\n' "$STATUS"
    printf 'exit_code\t%s\n' "$exit_code"
    printf 'head\t%s\n' "$HEAD_SHA"
    printf 'rustc\t%s\n' "$(rustc --version)"
    printf 'cargo\t%s\n' "$(cargo --version)"
    printf 'lock_before_sha256\t%s\n' "$LOCK_BEFORE"
    printf 'lock_candidate_sha256\t%s\n' "$LOCK_AFTER"
  } > "$RECEIPT"
}

BACKUP="$(mktemp "${TMPDIR:-/tmp}/cogsec-lock-diagnostic.XXXXXX")"
cp Cargo.lock "$BACKUP"
RESTORED=0
cleanup() {
  local exit_code=$?
  if [[ $RESTORED -ne 1 ]]; then
    cp "$BACKUP" Cargo.lock
  fi
  rm -f "$BACKUP"
  write_receipt "$exit_code" || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if cargo metadata --locked --format-version=1 > /dev/null 2> "$LOCKED_STDERR"; then
  STATUS="LOCK_READY"
  RESTORED=1
  rm -f "$BACKUP"
  trap - EXIT INT TERM
  write_receipt 0
  printf 'LOCK_READY: committed Cargo.lock passes locked workspace metadata.\n'
  exit 0
fi

printf 'LOCK_STALE: committed Cargo.lock failed locked metadata; generating a diagnostic candidate.\n'

# This unlocked metadata run is diagnostic only. The resulting lock is never used
# by the qualification step in this checkout.
cargo metadata --format-version=1 > /dev/null

mapfile -t CHANGED_TRACKED < <(git diff --name-only)
for path in "${CHANGED_TRACKED[@]}"; do
  [[ "$path" == "Cargo.lock" ]] || fail "diagnostic changed unexpected tracked path: $path"
done

LOCK_DELETIONS="$(git diff --unified=0 -- Cargo.lock | grep -E '^-[^-]' || true)"
if [[ -n "$LOCK_DELETIONS" ]]; then
  printf '%s\n' "$LOCK_DELETIONS" >&2
  fail "diagnostic candidate is not additive-only"
fi

for package in "${COGSEC_PACKAGES[@]}"; do
  grep -Fqx "name = \"$package\"" Cargo.lock || \
    fail "diagnostic candidate still lacks package entry: $package"
done

cargo metadata --locked --format-version=1 > /dev/null
git diff --check -- Cargo.lock

LOCK_AFTER="$(sha256sum Cargo.lock | awk '{print $1}')"
cp Cargo.lock "$CANDIDATE_LOCK"
git diff -- Cargo.lock > "$CANDIDATE_PATCH"

# Qualification must see the exact committed lock, never the generated candidate.
cp "$BACKUP" Cargo.lock
git diff --exit-code -- Cargo.lock > /dev/null
RESTORED=1
rm -f "$BACKUP"
trap - EXIT INT TERM

STATUS="CANDIDATE_GENERATED"
write_receipt 0
printf 'CANDIDATE_GENERATED: review the uploaded lock candidate/patch; qualification will still run against the committed lock.\n'
