#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Reproduce the focused Spore boot qualification without depending on a GitHub
# runner. The committed HEAD is checked in a detached temporary git worktree so
# uncommitted developer state cannot accidentally enter the result.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/qualify-spore-boot-local.sh [options]

Options:
  --apply-lock       Copy Cargo's resolver-produced lock back to this checkout.
                     Refuses to overwrite an already-modified Cargo.lock.
  --vm               Also run the NixOS quicken VM gate.
  --keep-worktree    Preserve the detached qualification worktree for inspection.
  --output-dir PATH  Receipt/lock export directory (default: /tmp/spore-boot-qualification-<sha>).
  -h, --help         Show this help.

The script prefers `nix develop .#rust`, whose Rust toolchain is derived from
rust-toolchain.toml. If Nix is unavailable it accepts a directly installed Cargo
only when rustc exactly matches the pinned channel.
EOF
}

APPLY_LOCK=0
RUN_VM=0
KEEP_WORKTREE=0
OUTPUT_DIR=""

while (($#)); do
  case "$1" in
    --apply-lock)
      APPLY_LOCK=1
      shift
      ;;
    --vm)
      RUN_VM=1
      shift
      ;;
    --keep-worktree)
      KEEP_WORKTREE=1
      shift
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --output-dir requires a path" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v git >/dev/null 2>&1 || { echo "ERROR: git is required" >&2; exit 2; }

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ERROR: run this script from a Symthaea git checkout" >&2
  exit 2
}
cd "$ROOT"

HEAD_SHA="$(git rev-parse HEAD)"
SHORT_SHA="${HEAD_SHA:0:12}"
PINNED_RUST="$(sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' rust-toolchain.toml | head -n1)"
[[ -n "$PINNED_RUST" ]] || { echo "ERROR: could not read rust-toolchain.toml channel" >&2; exit 2; }

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${TMPDIR:-/tmp}/spore-boot-qualification-${SHORT_SHA}"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

WORK_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/spore-boot-worktree.XXXXXX")"
WORKTREE="$WORK_PARENT/tree"

cleanup() {
  local rc=$?
  if [[ "$KEEP_WORKTREE" == 1 ]]; then
    echo "Preserved qualification worktree: $WORKTREE"
  else
    git -C "$ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
    rm -rf "$WORK_PARENT"
  fi
  return "$rc"
}
trap cleanup EXIT

git worktree add --detach "$WORKTREE" "$HEAD_SHA" >/dev/null

QUALIFY_COMMAND='set -euo pipefail
printf "Rust:  "; rustc --version
printf "Cargo: "; cargo --version
cargo metadata --format-version 1 --no-deps >/dev/null
touch .spore-lock-resolved
bash scripts/check-spore-boot-stack.sh'
if [[ "$RUN_VM" == 1 ]]; then
  QUALIFY_COMMAND+=$' --vm'
fi

RUNNER_KIND=""
set +e
if command -v nix >/dev/null 2>&1; then
  RUNNER_KIND="nix-devshell-rust"
  (
    cd "$WORKTREE"
    nix develop .#rust --command bash -lc "$QUALIFY_COMMAND"
  )
  QUALIFICATION_RC=$?
elif command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
  ACTUAL_RUST="$(rustc --version | awk '{print $2}')"
  if [[ "$ACTUAL_RUST" != "$PINNED_RUST" ]]; then
    echo "ERROR: rustc $ACTUAL_RUST is installed, but repository pin is $PINNED_RUST" >&2
    echo "Install/use the pinned toolchain or run from NixOS with nix develop .#rust." >&2
    QUALIFICATION_RC=2
  else
    RUNNER_KIND="direct-pinned-rust"
    (
      cd "$WORKTREE"
      bash -lc "$QUALIFY_COMMAND"
    )
    QUALIFICATION_RC=$?
  fi
else
  echo "ERROR: neither Nix nor a directly installed Cargo/rustc toolchain is available" >&2
  QUALIFICATION_RC=2
fi
set -e

LOCK_RESOLVED=0
LOCK_CHANGED=0
EXPORTED_LOCK=""
if [[ -f "$WORKTREE/.spore-lock-resolved" && -f "$WORKTREE/Cargo.lock" ]]; then
  LOCK_RESOLVED=1
  if ! cmp -s "$ROOT/Cargo.lock" "$WORKTREE/Cargo.lock"; then
    LOCK_CHANGED=1
    EXPORTED_LOCK="$OUTPUT_DIR/Cargo.lock.resolved"
    cp "$WORKTREE/Cargo.lock" "$EXPORTED_LOCK"
    echo
    echo "Cargo.lock differs from committed HEAD."
    echo "Resolver-produced lock exported to: $EXPORTED_LOCK"
    git diff --no-index --stat "$ROOT/Cargo.lock" "$EXPORTED_LOCK" || true

    if [[ "$APPLY_LOCK" == 1 ]]; then
      if ! git diff --quiet -- Cargo.lock || ! git diff --cached --quiet -- Cargo.lock; then
        echo "ERROR: refusing --apply-lock because this checkout already modifies Cargo.lock" >&2
        QUALIFICATION_RC=2
      else
        cp "$EXPORTED_LOCK" "$ROOT/Cargo.lock"
        echo "Applied resolver-produced Cargo.lock to current checkout."
        echo "Review and commit it unchanged before claiming locked qualification."
      fi
    fi
  else
    echo "Cargo.lock already matches Cargo's resolver output."
  fi
fi

STATUS="failed"
if [[ "$QUALIFICATION_RC" == 0 ]]; then
  STATUS="passed"
fi

RECEIPT="$OUTPUT_DIR/LOCAL_QUALIFICATION.json"
python3 - "$RECEIPT" "$HEAD_SHA" "$PINNED_RUST" "$RUNNER_KIND" "$STATUS" \
  "$QUALIFICATION_RC" "$LOCK_RESOLVED" "$LOCK_CHANGED" "$EXPORTED_LOCK" "$RUN_VM" <<'PY'
import datetime
import json
import pathlib
import sys

(
    receipt,
    head,
    rust,
    runner,
    status,
    rc,
    lock_resolved,
    lock_changed,
    exported_lock,
    vm,
) = sys.argv[1:]

payload = {
    "schema": "spore-boot-local-qualification-v1",
    "qualified_head": head,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "rust_channel": rust,
    "runner": runner or "unavailable",
    "status": status,
    "exit_code": int(rc),
    "cargo_lock_resolved": bool(int(lock_resolved)),
    "cargo_lock_changed": bool(int(lock_changed)),
    "resolver_lock_export": exported_lock or None,
    "vm_requested": bool(int(vm)),
}
pathlib.Path(receipt).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo
echo "Local qualification receipt: $RECEIPT"
if [[ "$QUALIFICATION_RC" == 0 ]]; then
  echo "PASS: committed HEAD $SHORT_SHA passed the focused Spore boot lane locally."
  if [[ "$LOCK_CHANGED" == 1 && "$APPLY_LOCK" == 0 ]]; then
    echo "NOTE: qualification used the exported resolver lock; commit it unchanged before a --locked PR gate can pass."
  fi
else
  echo "FAIL: focused local qualification exited with code $QUALIFICATION_RC" >&2
fi

exit "$QUALIFICATION_RC"
