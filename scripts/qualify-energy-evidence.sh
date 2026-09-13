#!/usr/bin/env bash
# Exact-subject package qualification for the Tier-1 energy-material evidence lineage.
#
# This is the single implementation used by CI and by local/pinned-toolchain runs.
# A stale Cargo.lock permits an unlocked *diagnostic* pass so compiler/test/lint
# failures remain observable, but the final result still fails reproducibility.

set -uo pipefail

PACKAGES=(
  symthaea-discovery
  symthaea-energy-material-screening
  symthaea-energy-material-dossier
  symthaea-energy-material-candidate-version
  symthaea-energy-material-campaign
  symthaea-energy-evidence-envelope
  symthaea-energy-native-dossier
  symthaea-energy-native-campaign-admission
)

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "ERROR: must run inside a Git worktree" >&2
  exit 2
fi
cd "$ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked worktree/index must be clean before qualification" >&2
  exit 2
fi

ACTUAL_HEAD="$(git rev-parse HEAD)"
EXPECTED_HEAD="${QUALIFICATION_HEAD_SHA:-$ACTUAL_HEAD}"
BASE_SHA="${QUALIFICATION_BASE_SHA:-}"
EVENT_SHA="${EVENT_SHA:-$ACTUAL_HEAD}"

if [[ "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "ERROR: actual HEAD $ACTUAL_HEAD != declared subject $EXPECTED_HEAD" >&2
  exit 2
fi

RUSTC_VERSION="$(rustc --version 2>/dev/null || true)"
CARGO_VERSION="$(cargo --version 2>/dev/null || true)"
if [[ "$RUSTC_VERSION" != rustc\ 1.96.0* ]]; then
  echo "ERROR: exact Rust 1.96.0 required; got: ${RUSTC_VERSION:-unavailable}" >&2
  exit 2
fi
if [[ "$CARGO_VERSION" != cargo\ 1.96.0* ]]; then
  echo "ERROR: exact Cargo 1.96.0 required; got: ${CARGO_VERSION:-unavailable}" >&2
  exit 2
fi

EVIDENCE_DIR="${ENERGY_EVIDENCE_DIR:-${RUNNER_TEMP:-$ROOT/target}/energy-evidence-fast}"
mkdir -p "$EVIDENCE_DIR"
rm -f \
  "$EVIDENCE_DIR/fmt.tsv" \
  "$EVIDENCE_DIR/tests.tsv" \
  "$EVIDENCE_DIR/clippy.tsv" \
  "$EVIDENCE_DIR/Cargo.lock.patch" \
  "$EVIDENCE_DIR/locked.err" \
  "$EVIDENCE_DIR/qualification-receipt.json" \
  "$EVIDENCE_DIR/qualification-receipt.sha256"

LOCK_BACKUP="$EVIDENCE_DIR/Cargo.lock.original"
LOCKED_OK=false
LOCK_BACKED_UP=false
RESTORED=false

restore_lock() {
  if [[ "$LOCK_BACKED_UP" == true && "$RESTORED" != true ]]; then
    cp "$LOCK_BACKUP" Cargo.lock
    RESTORED=true
  fi
}
trap restore_lock EXIT

printf 'Qualification head: %s\n' "$ACTUAL_HEAD"
printf 'Qualification base: %s\n' "${BASE_SHA:-n/a}"
printf 'Rust: %s\nCargo: %s\n' "$RUSTC_VERSION" "$CARGO_VERSION"

if cargo metadata --locked --format-version 1 \
    >"$EVIDENCE_DIR/metadata.json" \
    2>"$EVIDENCE_DIR/locked.err"; then
  LOCKED_OK=true
  echo "Cargo.lock is fresh under --locked."
else
  cp Cargo.lock "$LOCK_BACKUP"
  LOCK_BACKED_UP=true
  echo "Cargo.lock is stale; continuing with diagnostic unlocked resolution." >&2
  cat "$EVIDENCE_DIR/locked.err" >&2
fi

run_matrix() {
  local kind="$1"
  shift
  local outfile="$EVIDENCE_DIR/${kind}.tsv"
  local overall=0
  : > "$outfile"

  for package in "${PACKAGES[@]}"; do
    echo "== $kind: $package =="
    if "$@" "$package"; then
      printf '%s\tpass\n' "$package" >> "$outfile"
    else
      printf '%s\tfail\n' "$package" >> "$outfile"
      overall=1
    fi
  done
  return "$overall"
}

fmt_one() {
  cargo fmt -p "$1" -- --check
}

test_one_locked() {
  cargo test --locked -p "$1" --all-targets
}

test_one_unlocked() {
  cargo test -p "$1" --all-targets
}

clippy_one_locked() {
  cargo clippy --locked -p "$1" --all-targets -- -D warnings
}

clippy_one_unlocked() {
  cargo clippy -p "$1" --all-targets -- -D warnings
}

FMT_OK=true
TEST_OK=true
CLIPPY_OK=true

run_matrix fmt fmt_one || FMT_OK=false
if [[ "$LOCKED_OK" == true ]]; then
  run_matrix tests test_one_locked || TEST_OK=false
  run_matrix clippy clippy_one_locked || CLIPPY_OK=false
else
  run_matrix tests test_one_unlocked || TEST_OK=false
  run_matrix clippy clippy_one_unlocked || CLIPPY_OK=false
  git diff -- Cargo.lock > "$EVIDENCE_DIR/Cargo.lock.patch" || true
fi

restore_lock
trap - EXIT

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  echo "ERROR: qualification subject moved during execution" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked checkout differs after qualification/restoration" >&2
  git status --short >&2
  exit 2
fi

export ACTUAL_HEAD BASE_SHA EVENT_SHA RUSTC_VERSION CARGO_VERSION LOCKED_OK FMT_OK TEST_OK CLIPPY_OK EVIDENCE_DIR
export PACKAGES_JOINED="${PACKAGES[*]}"
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["EVIDENCE_DIR"])

def read_matrix(name):
    path = root / f"{name}.tsv"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            package, status = line.split("\t", 1)
            out[package] = status
    return out

patch = root / "Cargo.lock.patch"
patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest() if patch.exists() else None
receipt = {
    "schema": "symthaea.energy-evidence-fast-lane.receipt.v1",
    "head_sha": os.environ["ACTUAL_HEAD"],
    "base_sha": os.environ.get("BASE_SHA") or None,
    "event_sha": os.environ["EVENT_SHA"],
    "subject_class": "raw_git_head_package_focused",
    "full_workspace_qualification_implied": False,
    "rustc_version": os.environ["RUSTC_VERSION"],
    "cargo_version": os.environ["CARGO_VERSION"],
    "cargo_lock_fresh": os.environ["LOCKED_OK"] == "true",
    "diagnostic_unlocked_execution": os.environ["LOCKED_OK"] != "true",
    "packages": os.environ["PACKAGES_JOINED"].split(),
    "fmt": read_matrix("fmt"),
    "tests": read_matrix("tests"),
    "clippy": read_matrix("clippy"),
    "all_fmt_passed": os.environ["FMT_OK"] == "true",
    "all_tests_passed": os.environ["TEST_OK"] == "true",
    "all_clippy_passed": os.environ["CLIPPY_OK"] == "true",
    "diagnostic_lock_patch_sha256": patch_sha,
    "qualification_eligible": all([
        os.environ["LOCKED_OK"] == "true",
        os.environ["FMT_OK"] == "true",
        os.environ["TEST_OK"] == "true",
        os.environ["CLIPPY_OK"] == "true",
    ]),
    "authority_boundary": (
        "Package-focused reproducibility evidence only; not whole-workspace integration, "
        "scientific validity, candidate promotion, synthesis authority, safety certification, "
        "investment approval, or deployment authority."
    ),
}
body = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
out = root / "qualification-receipt.json"
out.write_text(body)
digest = hashlib.sha256(body.encode()).hexdigest()
(root / "qualification-receipt.sha256").write_text(digest + "  qualification-receipt.json\n")
print(body, end="")
print(f"qualification receipt sha256: {digest}")
PY

if [[ "$LOCKED_OK" != true ]]; then
  echo "FAIL: Cargo.lock was stale. Unlocked results are diagnostic only." >&2
  exit 1
fi
if [[ "$FMT_OK" != true || "$TEST_OK" != true || "$CLIPPY_OK" != true ]]; then
  echo "FAIL: one or more focused qualification checks failed." >&2
  exit 1
fi

echo "PASS: exact-head Tier-1 energy evidence package qualification."
