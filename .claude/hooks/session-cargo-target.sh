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

# Write env var to persist for the session. Must be `export`ed, not a bare
# assignment — verified 2026-07-03: a bare `CARGO_TARGET_DIR=...` line here
# results in a shell variable visible via $CARGO_TARGET_DIR expansion within
# the session shell, but NOT propagated into the environment of child
# processes (cargo, timeout, backgrounded subshells) — i.e. it silently did
# nothing for every cargo invocation all session, causing exactly the
# heavy shared-target-dir lock contention this hook exists to prevent.
if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
    echo "export CARGO_TARGET_DIR=${SESSION_TARGET}" >> "$CLAUDE_ENV_FILE"
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

# Educational nudge: warn ONCE per session if this session is opening
# in the main tree (not a worktree) while other Claude sessions are
# active. The cross-project guard catches cross-project scoops; this
# nudge suggests the isolation-level fix (worktree) for same-project
# scoops. Non-blocking — users may have good reasons to stay in main.
# Writes to stdout (SessionStart convention: Claude Code surfaces this
# to the model as initial-turn context).
session_cwd="${CLAUDE_PROJECT_DIR:-$PROJECT_DIR}"
case "$session_cwd" in
    "$PROJECT_DIR"/.claude/worktrees/*) ;;  # in a worktree — skip
    "$PROJECT_DIR"|"$PROJECT_DIR"/)
        # In main tree. Count other active Claude sessions.
        claude_sessions=$(pgrep -f claude-unwrapped 2>/dev/null | wc -l)
        if (( claude_sessions > 1 )); then
            cat <<NOTICE
────────────────────────────────────────────────────────────────
 Concurrent-session notice — opened in main tree, ${claude_sessions} sessions active
────────────────────────────────────────────────────────────────
 Cross-project staging guard is armed (.githooks/pre-commit).
 For source-level isolation (stops same-project scoops + silent
 reverts), consider a worktree BEFORE starting edits:

     ./scripts/session-worktree.sh create <name>
     cd .claude/worktrees/session-<name>

 Playbook: .claude/rules/CONCURRENT_SESSIONS.md
────────────────────────────────────────────────────────────────
NOTICE
        fi
        ;;
esac

exit 0
