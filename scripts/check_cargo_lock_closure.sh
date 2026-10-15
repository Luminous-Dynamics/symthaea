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

write_report() {
  local classification="$1"
  local message="$2"
  local locked_status="$3"
  local projection_status="$4"
  local committed_sha="$5"
  local proposed_sha="$6"
  cat > "$REPORT" <<JSON
{
  "schema": "cargo-lock-closure-report-v1",
  "classification": "$classification",
  "message": "$message",
  "locked_metadata_exit_code": $locked_status,
  "projection_metadata_exit_code": $projection_status,
  "committed_lock_sha256": "$committed_sha",
  "proposed_lock_sha256": "$proposed_sha",
  "root": "$ROOT",
  "lockfile": "Cargo.lock",
  "metadata_command": "cargo metadata --format-version 1 (full dependency graph; no --no-deps)"
}
JSON
}

if [[ ! -f Cargo.lock ]]; then
  write_report "FAIL_MISSING_LOCKFILE" "Cargo.lock is absent" 2 2 "none" "none"
  echo "FAIL: Cargo.lock is absent" >&2
  exit 2
fi

dirty="$(git status --porcelain --untracked-files=no)"
if [[ -n "$dirty" ]]; then
  write_report "FAIL_DIRTY_CHECKOUT" "tracked worktree was already dirty; refusing to run lock projection" 2 2 "unknown" "unknown"
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
  write_report "LOCK_CLEAN" "full Cargo metadata resolved with the committed lockfile" 0 0 "$committed_sha" "none"
  printf 'LOCK_CLEAN committed_lock_sha256=%s metadata_sha256=%s\n' "$committed_sha" "$metadata_sha"
  exit 0
else
  locked_status=$?
fi

locked_after_sha="$(sha256sum Cargo.lock | awk '{print $1}')"
if [[ "$locked_after_sha" != "$committed_sha" ]]; then
  cp "$backup" Cargo.lock
  write_report "FAIL_LOCKED_MUTATION" "Cargo metadata changed Cargo.lock despite --locked; original lock restored" "$locked_status" 2 "$committed_sha" "$locked_after_sha"
  echo "FAIL_LOCKED_MUTATION: restored the committed Cargo.lock" >&2
  exit 2
fi

set +e
"$CARGO_BIN" metadata --format-version 1 > "$OUT_DIR/metadata-projection.json" 2> "$OUT_DIR/metadata-projection.stderr"
projection_status=$?
set -e
if [[ "$projection_status" -ne 0 ]]; then
  cp "$backup" Cargo.lock
  write_report "FAIL_METADATA_OR_RESOLUTION" "unlocked full Cargo metadata could not resolve the workspace; no proposed lock is qualified" "$locked_status" "$projection_status" "$committed_sha" "none"
  echo "FAIL_METADATA_OR_RESOLUTION: see metadata-locked.stderr and metadata-projection.stderr" >&2
  exit 2
fi

changed_paths="$(git diff --name-only)"
if [[ "$changed_paths" != "Cargo.lock" ]]; then
  proposed_sha="$(sha256sum Cargo.lock | awk '{print $1}')"
  cp "$backup" Cargo.lock
  write_report "FAIL_UNEXPECTED_WORKTREE_MUTATION" "lock projection changed tracked paths other than Cargo.lock" "$locked_status" "$projection_status" "$committed_sha" "$proposed_sha"
  echo "FAIL_UNEXPECTED_WORKTREE_MUTATION: projection changed tracked paths: $changed_paths" >&2
  exit 2
fi

proposed_sha="$(sha256sum Cargo.lock | awk '{print $1}')"
if [[ "$proposed_sha" == "$committed_sha" ]]; then
  cp "$backup" Cargo.lock
  write_report "FAIL_LOCKED_METADATA_WITHOUT_LOCK_DIFF" "locked metadata failed but unlocked metadata produced no lockfile difference" "$locked_status" "$projection_status" "$committed_sha" "$proposed_sha"
  echo "FAIL_LOCKED_METADATA_WITHOUT_LOCK_DIFF: inspect retained Cargo logs" >&2
  exit 2
fi

cp Cargo.lock "$OUT_DIR/proposed-Cargo.lock"
git diff --binary -- Cargo.lock > "$OUT_DIR/Cargo.lock.patch"

if "$CARGO_BIN" metadata --locked --format-version 1 > "$OUT_DIR/metadata-proposed-locked.json" 2> "$OUT_DIR/metadata-proposed-locked.stderr"; then
  write_report "FAIL_LOCK_DRIFT" "Cargo-generated proposed lock resolves under --locked, but the committed lock was stale; review proposed-Cargo.lock and Cargo.lock.patch before committing" "$locked_status" "$projection_status" "$committed_sha" "$proposed_sha"
  printf 'FAIL_LOCK_DRIFT committed_lock_sha256=%s proposed_lock_sha256=%s\n' "$committed_sha" "$proposed_sha" >&2
  exit 1
else
  proposed_locked_status=$?
  write_report "FAIL_PROPOSED_LOCK_NOT_CLOSED" "Cargo-generated proposed lock still fails full metadata under --locked" "$locked_status" "$proposed_locked_status" "$committed_sha" "$proposed_sha"
  echo "FAIL_PROPOSED_LOCK_NOT_CLOSED: see proposed metadata stderr" >&2
  exit 2
fi
