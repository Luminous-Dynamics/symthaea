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

Exit status:
  0  exact committed HEAD passed the focused lane and Cargo.lock was unchanged
  1  focused qualification failed
  2  local/tooling/precondition error
  3  focused lane passed using a resolver-updated Cargo.lock; commit that exact
     lock and rerun on the new HEAD before claiming locked qualification
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
# Match hosted qualification: permit Cargo to produce the exact minimal lock
# candidate first, then require every substantive gate to run with --locked.
cargo metadata --format-version 1 --no-deps >/dev/null
touch .spore-lock-resolved
bash scripts/check-spore-boot-stack.sh'
if [[ "$RUN_VM" == 1 ]]; then
  QUALIFY_COMMAND+=$' --vm'
fi

RUNNER_KIND=""
LANE_RC=2
set +e
if command -v nix >/dev/null 2>&1; then
  RUNNER_KIND="nix-devshell-rust"
  (
    cd "$WORKTREE"
    nix develop .#rust --command bash -c "$QUALIFY_COMMAND"
  )
  LANE_RC=$?
elif command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
  ACTUAL_RUST="$(rustc --version | awk '{print $2}')"
  if [[ "$ACTUAL_RUST" != "$PINNED_RUST" ]]; then
    echo "ERROR: rustc $ACTUAL_RUST is installed, but repository pin is $PINNED_RUST" >&2
    echo "Install/use the pinned toolchain or run from NixOS with nix develop .#rust." >&2
    LANE_RC=2
  else
    RUNNER_KIND="direct-pinned-rust"
    (
      cd "$WORKTREE"
      bash -c "$QUALIFY_COMMAND"
    )
    LANE_RC=$?
  fi
else
  echo "ERROR: neither Nix nor a directly installed Cargo/rustc toolchain is available" >&2
  LANE_RC=2
fi
set -e

LOCK_RESOLVED=0
LOCK_CHANGED=0
EXPORTED_LOCK=""
APPLY_LOCK_FAILED=0
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
        APPLY_LOCK_FAILED=1
      else
        cp "$EXPORTED_LOCK" "$ROOT/Cargo.lock"
        echo "Applied resolver-produced Cargo.lock to current checkout."
        echo "Review and commit it unchanged, then rerun qualification on the new HEAD."
      fi
    fi
  else
    echo "Cargo.lock already matches Cargo's resolver output."
  fi
fi

# Overall qualification is intentionally stricter than the inner lane. A lane
# that passed only after mutating Cargo.lock did NOT qualify the committed HEAD.
OVERALL_RC="$LANE_RC"
STATUS="failed"
if [[ "$APPLY_LOCK_FAILED" == 1 ]]; then
  OVERALL_RC=2
  STATUS="precondition-error"
elif [[ "$LANE_RC" == 0 && "$LOCK_CHANGED" == 1 ]]; then
  OVERALL_RC=3
  STATUS="lock-update-required"
elif [[ "$LANE_RC" == 0 ]]; then
  OVERALL_RC=0
  STATUS="passed"
elif [[ "$LANE_RC" == 2 ]]; then
  STATUS="precondition-error"
fi

json_escape() {
  local value="$1"
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '%s' "$value"
}

json_bool() {
  [[ "$1" == 1 ]] && printf 'true' || printf 'false'
}

RECEIPT="$OUTPUT_DIR/LOCAL_QUALIFICATION.json"
CREATED_AT_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
{
  printf '{\n'
  printf '  "schema": "spore-boot-local-qualification-v1",\n'
  printf '  "qualified_head": "%s",\n' "$(json_escape "$HEAD_SHA")"
  printf '  "created_at_utc": "%s",\n' "$(json_escape "$CREATED_AT_UTC")"
  printf '  "rust_channel": "%s",\n' "$(json_escape "$PINNED_RUST")"
  printf '  "runner": "%s",\n' "$(json_escape "${RUNNER_KIND:-unavailable}")"
  printf '  "status": "%s",\n' "$(json_escape "$STATUS")"
  printf '  "lane_exit_code": %d,\n' "$LANE_RC"
  printf '  "exit_code": %d,\n' "$OVERALL_RC"
  printf '  "cargo_lock_resolved": %s,\n' "$(json_bool "$LOCK_RESOLVED")"
  printf '  "cargo_lock_changed": %s,\n' "$(json_bool "$LOCK_CHANGED")"
  if [[ -n "$EXPORTED_LOCK" ]]; then
    printf '  "resolver_lock_export": "%s",\n' "$(json_escape "$EXPORTED_LOCK")"
  else
    printf '  "resolver_lock_export": null,\n'
  fi
  printf '  "vm_requested": %s\n' "$(json_bool "$RUN_VM")"
  printf '}\n'
} >"$RECEIPT"

echo
echo "Local qualification receipt: $RECEIPT"
case "$OVERALL_RC" in
  0)
    echo "PASS: exact committed HEAD $SHORT_SHA passed the focused Spore boot lane with an unchanged Cargo.lock."
    ;;
  3)
    echo "LOCK UPDATE REQUIRED: the semantic lane passed, but committed HEAD $SHORT_SHA is not locked-qualified." >&2
    echo "Commit the exact resolver-produced Cargo.lock unchanged, then rerun on that new HEAD." >&2
    ;;
  2)
    echo "ERROR: local qualification could not establish a valid exact-head result." >&2
    ;;
  *)
    echo "FAIL: focused local qualification exited with code $LANE_RC" >&2
    ;;
esac

exit "$OVERALL_RC"
