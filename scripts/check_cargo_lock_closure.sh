#!/usr/bin/env bash
set -euo pipefail

ROOT="${CARGO_LOCK_CLOSURE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR_INPUT="${1:-target/cargo-lock-closure}"
CARGO_BIN="${CARGO_BIN:-cargo}"
if [[ "$OUT_DIR_INPUT" = /* ]]; then
  OUT_DIR="$OUT_DIR_INPUT"
else
  OUT_DIR="$ROOT/$OUT_DIR_INPUT"
fi
mkdir -p "$OUT_DIR"
cd "$ROOT"
REPORT="$OUT_DIR/report.json"

{
  printf 'cargo: '
  "$CARGO_BIN" --version
  printf 'rustc: '
  rustc --version
  rustc -Vv
} > "$OUT_DIR/toolchain.txt" 2>&1

write_report() {
  local classification="$1"
  local message="$2"
  local locked_status="$3"
  local projection_status="$4"
  local proposed_locked_status="$5"
  local committed_sha="$6"
  local proposed_sha="$7"
  local locked_metadata_sha="$8"
  local projection_metadata_sha="$9"
  local proposed_locked_metadata_sha="${10}"
  cat > "$REPORT" <<JSON
{
  "schema": "cargo-lock-closure-report-v1",
  "classification": "$classification",
  "message": "$message",
  "locked_metadata_exit_code": $locked_status,
  "projection_metadata_exit_code": $projection_status,
  "proposed_locked_metadata_exit_code": $proposed_locked_status,
  "committed_lock_sha256": "$committed_sha",
  "proposed_lock_sha256": "$proposed_sha",
  "locked_metadata_sha256": "$locked_metadata_sha",
  "projection_metadata_sha256": "$projection_metadata_sha",
  "proposed_locked_metadata_sha256": "$proposed_locked_metadata_sha",
  "root": "$ROOT",
  "lockfile": "Cargo.lock",
  "metadata_command": "cargo metadata --format-version 1 (full dependency graph; no --no-deps)"
}
JSON
}

if [[ ! -f Cargo.lock ]]; then
  write_report "FAIL_MISSING_LOCKFILE" "Cargo.lock is absent" null null null "none" "none" "none" "none" "none"
  echo "FAIL: Cargo.lock is absent" >&2
  exit 2
fi

dirty="$(git status --porcelain --untracked-files=no)"
if [[ -n "$dirty" ]]; then
  write_report "FAIL_DIRTY_CHECKOUT" "tracked worktree was already dirty; refusing to run lock projection" null null null "unknown" "unknown" "none" "none" "none"
  echo "FAIL: tracked worktree was already dirty; refusing lock projection" >&2
  exit 2
fi

committed_sha="$(sha256sum Cargo.lock | awk '{print $1}')"
backup="$(mktemp)"
cp Cargo.lock "$backup"
restore_lock() {
  cp "$backup" "$ROOT/Cargo.lock"
  rm -f "$backup"
}
trap restore_lock EXIT HUP INT TERM
cp Cargo.lock "$OUT_DIR/committed-Cargo.lock"

if "$CARGO_BIN" metadata --locked --format-version 1 > "$OUT_DIR/metadata-locked.json" 2> "$OUT_DIR/metadata-locked.stderr"; then
  metadata_sha="$(sha256sum "$OUT_DIR/metadata-locked.json" | awk '{print $1}')"
  write_report "LOCK_CLEAN" "full Cargo metadata resolved with the committed lockfile" 0 null null "$committed_sha" "none" "$metadata_sha" "none" "none"
  printf 'LOCK_CLEAN committed_lock_sha256=%s metadata_sha256=%s\n' "$committed_sha" "$metadata_sha"
  exit 0
else
  locked_status=$?
fi
locked_metadata_sha="$(sha256sum "$OUT_DIR/metadata-locked.json" | awk '{print $1}')"

locked_after_sha="$(sha256sum Cargo.lock | awk '{print $1}')"
if [[ "$locked_after_sha" != "$committed_sha" ]]; then
  cp "$backup" Cargo.lock
  write_report "FAIL_LOCKED_MUTATION" "Cargo metadata changed Cargo.lock despite --locked; original lock restored" "$locked_status" null null "$committed_sha" "$locked_after_sha" "$locked_metadata_sha" "none" "none"
  echo "FAIL_LOCKED_MUTATION: restored the committed Cargo.lock" >&2
  exit 2
fi

set +e
"$CARGO_BIN" metadata --format-version 1 > "$OUT_DIR/metadata-projection.json" 2> "$OUT_DIR/metadata-projection.stderr"
projection_status=$?
set -e
projection_metadata_sha="$(sha256sum "$OUT_DIR/metadata-projection.json" | awk '{print $1}')"
if [[ "$projection_status" -ne 0 ]]; then
  cp "$backup" Cargo.lock
  write_report "FAIL_METADATA_OR_RESOLUTION" "unlocked full Cargo metadata could not resolve the workspace; no proposed lock is qualified" "$locked_status" "$projection_status" null "$committed_sha" "none" "$locked_metadata_sha" "$projection_metadata_sha" "none"
  echo "FAIL_METADATA_OR_RESOLUTION: see metadata-locked.stderr and metadata-projection.stderr" >&2
  exit 2
fi

changed_paths="$(git diff --name-only)"
if [[ "$changed_paths" != "Cargo.lock" ]]; then
  proposed_sha="$(sha256sum Cargo.lock | awk '{print $1}')"
  cp "$backup" Cargo.lock
  write_report "FAIL_UNEXPECTED_WORKTREE_MUTATION" "lock projection changed tracked paths other than Cargo.lock" "$locked_status" "$projection_status" null "$committed_sha" "$proposed_sha" "$locked_metadata_sha" "$projection_metadata_sha" "none"
  echo "FAIL_UNEXPECTED_WORKTREE_MUTATION: projection changed tracked paths: $changed_paths" >&2
  exit 2
fi

proposed_sha="$(sha256sum Cargo.lock | awk '{print $1}')"
if [[ "$proposed_sha" == "$committed_sha" ]]; then
  cp "$backup" Cargo.lock
  write_report "FAIL_LOCKED_METADATA_WITHOUT_LOCK_DIFF" "locked metadata failed but unlocked metadata produced no lockfile difference" "$locked_status" "$projection_status" null "$committed_sha" "$proposed_sha" "$locked_metadata_sha" "$projection_metadata_sha" "none"
  echo "FAIL_LOCKED_METADATA_WITHOUT_LOCK_DIFF: inspect retained Cargo logs" >&2
  exit 2
fi

cp Cargo.lock "$OUT_DIR/proposed-Cargo.lock"
git diff --binary -- Cargo.lock > "$OUT_DIR/Cargo.lock.patch"

if "$CARGO_BIN" metadata --locked --format-version 1 > "$OUT_DIR/metadata-proposed-locked.json" 2> "$OUT_DIR/metadata-proposed-locked.stderr"; then
  proposed_locked_metadata_sha="$(sha256sum "$OUT_DIR/metadata-proposed-locked.json" | awk '{print $1}')"
  write_report "FAIL_LOCK_DRIFT" "Cargo-generated proposed lock resolves under --locked, but the committed lock was stale; review proposed-Cargo.lock and Cargo.lock.patch before committing" "$locked_status" "$projection_status" 0 "$committed_sha" "$proposed_sha" "$locked_metadata_sha" "$projection_metadata_sha" "$proposed_locked_metadata_sha"
  printf 'FAIL_LOCK_DRIFT committed_lock_sha256=%s proposed_lock_sha256=%s\n' "$committed_sha" "$proposed_sha" >&2
  exit 1
else
  proposed_locked_status=$?
  proposed_locked_metadata_sha="$(sha256sum "$OUT_DIR/metadata-proposed-locked.json" | awk '{print $1}')"
  write_report "FAIL_PROPOSED_LOCK_NOT_CLOSED" "Cargo-generated proposed lock still fails full metadata under --locked" "$locked_status" "$projection_status" "$proposed_locked_status" "$committed_sha" "$proposed_sha" "$locked_metadata_sha" "$projection_metadata_sha" "$proposed_locked_metadata_sha"
  echo "FAIL_PROPOSED_LOCK_NOT_CLOSED: see proposed metadata stderr" >&2
  exit 2
fi
