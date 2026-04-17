#!/usr/bin/env bash
# session-cargo-target.sh — Automatically isolate cargo target dirs per Claude session.
#
# Called by SessionStart hook. Sets CARGO_TARGET_DIR to a session-specific
# directory so concurrent Claude sessions don't contend on the cargo lock.
# sccache (configured in ~/.cargo/config.toml) shares compiled artifacts
# across all sessions, so the first build is fast for cached crates.
#
# Target dirs: /var/cache/cargo-targets/<session-id>/ (on Intel ext4 — fast writes, ephemeral)
# Cleanup: old targets are removed automatically (>48h, no active processes)

set -euo pipefail

PROJECT_DIR="/srv/luminous-dynamics"
# Cargo targets on Intel root drive (write-heavy, ephemeral — no need for Samsung btrfs)
TARGETS_DIR="/var/cache/cargo-targets"

# Read session_id from hook stdin (JSON)
SESSION_ID=$(cat | jq -r '.session_id // empty' 2>/dev/null || echo "")

if [[ -z "$SESSION_ID" ]]; then
    # Fallback: use PID-based ID
    SESSION_ID="pid-$$-$(date +%s)"
fi

# Create session-specific target directory
SESSION_TARGET="${TARGETS_DIR}/${SESSION_ID}"
mkdir -p "$SESSION_TARGET"

# Write env var to persist for the session
if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
    echo "CARGO_TARGET_DIR=${SESSION_TARGET}" >> "$CLAUDE_ENV_FILE"
fi

# Cleanup stale targets (>48h old, no active cargo processes referencing them)
if [[ -d "$TARGETS_DIR" ]]; then
    for target_dir in "$TARGETS_DIR"/*/; do
        [[ -d "$target_dir" ]] || continue
        [[ "$target_dir" == "${SESSION_TARGET}/" ]] && continue

        # Check age
        local_mod=$(stat -c %Y "$target_dir" 2>/dev/null || echo 0)
        local_now=$(date +%s)
        local_age_hours=$(( (local_now - local_mod) / 3600 ))

        if (( local_age_hours > 48 )); then
            # Check no active cargo processes reference this dir
            local_procs=$(ps aux 2>/dev/null | grep -c "$target_dir" || true)
            if (( local_procs <= 1 )); then
                rm -rf "$target_dir" 2>/dev/null || true
            fi
        fi
    done
fi

# Ensure the tracked .githooks/ are active (cross-project staging guard
# + cargo fmt check). `core.hooksPath` is a local-only git-config
# setting; re-assert per session so fresh clones / new worktree dirs
# pick it up. Idempotent.
if [[ -d "$PROJECT_DIR/.githooks" ]]; then
    current_hooks_path=$(git -C "$PROJECT_DIR" config --local --get core.hooksPath 2>/dev/null || echo "")
    if [[ "$current_hooks_path" != ".githooks" ]]; then
        git -C "$PROJECT_DIR" config --local core.hooksPath .githooks 2>/dev/null || true
    fi
fi

exit 0
