#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: run_capsule_nix.sh TARGET_WORKTREE PROFILE EXPECTED_HEAD OUTPUT_DIR

Runs the reviewed qualification capsule runner inside TARGET_WORKTREE's own
Nix .#default development shell. OUTPUT_DIR must be outside TARGET_WORKTREE.
USAGE
  exit 64
}

[[ $# -eq 4 ]] || usage
TARGET_INPUT=$1
PROFILE_INPUT=$2
EXPECTED_HEAD=$3
OUTPUT_INPUT=$4

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RUNNER="$SCRIPT_DIR/run_capsule.py"
MANIFEST="$SCRIPT_DIR/TOOLING_V1.sha256"

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 69; }
command -v nix >/dev/null 2>&1 || { echo "nix is required" >&2; exit 69; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 69; }
command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum is required" >&2; exit 69; }

[[ -f "$RUNNER" && -f "$MANIFEST" ]] || { echo "qualification tooling is incomplete" >&2; exit 66; }

TARGET=$(git -C "$TARGET_INPUT" rev-parse --show-toplevel 2>/dev/null) || {
  echo "target is not a Git worktree: $TARGET_INPUT" >&2
  exit 66
}
TARGET=$(CDPATH= cd -- "$TARGET" && pwd -P)
PROFILE=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$PROFILE_INPUT")
OUTPUT=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$OUTPUT_INPUT")
PYTHON_BIN=$(command -v python3)

[[ "$EXPECTED_HEAD" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "EXPECTED_HEAD must be exactly 40 hex characters" >&2; exit 64; }
EXPECTED_HEAD=${EXPECTED_HEAD,,}
ACTUAL_HEAD=$(git -C "$TARGET" rev-parse HEAD)
[[ "$ACTUAL_HEAD" == "$EXPECTED_HEAD" ]] || {
  echo "target HEAD mismatch: expected $EXPECTED_HEAD got $ACTUAL_HEAD" >&2
  exit 65
}
[[ -z "$(git -C "$TARGET" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "target worktree is dirty" >&2
  exit 65
}
[[ -f "$TARGET/flake.nix" && -f "$TARGET/flake.lock" ]] || {
  echo "target must contain flake.nix and flake.lock" >&2
  exit 66
}
[[ -f "$PROFILE" ]] || { echo "profile not found: $PROFILE" >&2; exit 66; }

case "$OUTPUT/" in
  "$TARGET"/*) echo "OUTPUT_DIR must be outside TARGET_WORKTREE" >&2; exit 64 ;;
esac
[[ ! -e "$OUTPUT" ]] || { echo "OUTPUT_DIR already exists: $OUTPUT" >&2; exit 73; }

(
  cd "$SCRIPT_DIR"
  sha256sum --check --strict TOOLING_V1.sha256
)

export QUALIFICATION_TOOLING_MANIFEST_SHA256
QUALIFICATION_TOOLING_MANIFEST_SHA256=$(sha256sum "$MANIFEST" | awk '{print $1}')
export QUALIFICATION_LAUNCHER_SHA256
QUALIFICATION_LAUNCHER_SHA256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')

cd "$TARGET"
exec nix develop --no-write-lock-file .#default --command \
  "$PYTHON_BIN" "$RUNNER" \
  --profile "$PROFILE" \
  --expected-head "$EXPECTED_HEAD" \
  --executor LOCAL_NIX \
  --output "$OUTPUT"
