# Concurrent Session Safety

## Problem
Multiple Claude Code sessions (7+ observed April 2026) run on the same monorepo
simultaneously. Sessions editing the same files cause silent reverts — changes
made by one session are overwritten by another session's `cargo fmt` hooks,
stash operations, or direct edits.

## Rules

### Before starting a coding session
```bash
# Check for other active sessions
ps aux | grep claude-unwrapped | grep -v grep | wc -l
```

If >1 session is running:
1. **DO NOT edit shared crates** (`crates/mycelix-bridge-common/`, `crates/mycelix-zkp-core/`, etc.)
   without coordinating with other sessions
2. **Prefer new files** over editing existing files (new test files, new modules)
3. **Stage + commit immediately** after each logical change — don't batch
4. **Never use `git stash`** when other sessions are active (stash pop merges destroy work)

### Safe zones (low collision risk)
- New test files in cluster-specific `tests/` directories
- Submodule work (each submodule has its own git state)
- Frontend-only changes (Trunk.toml, CSS, index.html)
- Documentation files

### Danger zones (high collision risk)
- `crates/mycelix-bridge-common/src/` — shared by ALL clusters
- `crates/mycelix-zkp-core/src/` — shared by governance, health, identity
- Any file with `M` status in `git status` at session start — another session is editing it

### Recovery if changes are lost
1. Check `git reflog` for your commits
2. Check `git stash list` for stashed changes
3. Use `git diff HEAD~N` to find the commit with your changes
4. Cherry-pick or re-apply from the reflog
5. **Deleted branches are not gone immediately** — pruned-branch commits live in
   `.git`'s object store until GC runs. `git cat-file -t <sha>` confirms a
   commit is still reachable. See Phase I.B recovery
   (`d1df1216d1`, 2026-04-17): 11 commits cherry-picked back onto main
   weeks after the worktree branch was deleted.

## Cross-project commit guard

`.githooks/pre-commit` rejects commits that stage files across more than one
top-level project directory. This prevents the canonical scoop pattern: session
A leaves uncommitted edits in `symthaea/`; session B runs `git add -A &&
git commit` for their unrelated `symtropy/` work and sweeps up A's edits into
B's commit. Happened 2026-04-17 (Phase I.C scrcpy edits absorbed into
symtropy-bevy publish-prep commit).

Enabled via `core.hooksPath=.githooks` (re-asserted per session by
`.claude/hooks/session-cargo-target.sh`). Merges, cherry-picks, and reverts
skip the guard automatically (`MERGE_HEAD`, `CHERRY_PICK_HEAD`, `REVERT_HEAD`
detection). Intentional cross-project commits override with
`CLAUDE_ALLOW_CROSS_PROJECT=1 git commit ...`.

## What still doesn't protect you

The pre-commit guard catches cross-project scoops, but **same-project** scoops
(session A edits `symthaea/src/foo.rs`, session B edits `symthaea/src/bar.rs`
and broadly stages) still pass. For those, the defense is:

1. **Stage files specifically, not broadly.** `git add <paths>`, never
   `git add -A` / `git add .` / `git commit -a`. Rule 8 of the project
   CLAUDE.md makes this explicit.
2. **Commit quickly after editing.** Gap between `git add` and `git commit`
   is the vulnerability window.
3. **Use worktrees for source-level isolation** when editing shared files
   concurrently with other sessions (`./scripts/session-worktree.sh create
   <name>` per Rule #6).
