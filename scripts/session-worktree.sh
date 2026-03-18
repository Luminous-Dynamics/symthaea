#!/usr/bin/env bash
# session-worktree.sh — Manage git worktrees for concurrent Claude sessions
#
# Each session gets an isolated worktree with its own target/ directory.
# sccache (configured in .cargo/config.toml) shares compilation cache globally,
# so the first build in a new worktree is fast for unchanged crates.
#
# Usage:
#   ./scripts/session-worktree.sh create <name>    — Create a new session worktree
#   ./scripts/session-worktree.sh enter <name>     — Print the path (for cd)
#   ./scripts/session-worktree.sh list              — List active worktrees
#   ./scripts/session-worktree.sh cleanup           — Remove stale worktrees
#   ./scripts/session-worktree.sh cleanup-all       — Remove ALL session worktrees
#   ./scripts/session-worktree.sh status            — Show CPU/process health

set -euo pipefail

REPO_ROOT="/srv/luminous-dynamics"
WORKTREE_DIR="${REPO_ROOT}/.claude/worktrees"
SESSION_PREFIX="session"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cmd_create() {
    local name="${1:?Usage: session-worktree.sh create <name>}"
    local wt_path="${WORKTREE_DIR}/${SESSION_PREFIX}-${name}"
    local branch="worktree-${SESSION_PREFIX}-${name}"

    if [[ -d "$wt_path" ]]; then
        echo -e "${YELLOW}Worktree already exists:${NC} $wt_path"
        echo "Use: cd $wt_path"
        return 0
    fi

    echo -e "${BLUE}Creating worktree:${NC} ${SESSION_PREFIX}-${name}"

    cd "$REPO_ROOT"

    # Strategy: git worktree for tracked files, then overlay untracked files.
    # This handles repos with significant uncommitted work (new crates, new modules).

    # Step 1: Create worktree from a stash snapshot (captures tracked file changes)
    local base_ref="HEAD"
    if [[ -n "$(git diff --name-only HEAD 2>/dev/null)" ]]; then
        echo -e "  ${BLUE}Snapshotting modified tracked files...${NC}"
        local stash_commit
        stash_commit=$(git stash create "worktree-snapshot-${name}" 2>/dev/null || true)
        if [[ -n "$stash_commit" ]]; then
            base_ref="$stash_commit"
        fi
    fi

    git worktree add -b "$branch" "$wt_path" "$base_ref" 2>/dev/null || \
        git worktree add "$wt_path" "$base_ref" 2>/dev/null || \
        git worktree add -b "$branch" "$wt_path" HEAD

    # Step 2: Rsync untracked files into worktree (new crates, modules, etc.)
    # Excludes target/, .git, data dirs (symlinked separately), and large artifacts
    echo -e "  ${BLUE}Syncing untracked files...${NC}"
    local untracked_count=0
    while IFS= read -r -d '' file; do
        local rel="${file#${REPO_ROOT}/}"
        local dst="${wt_path}/${rel}"

        # Skip files in directories we'll symlink
        case "$rel" in
            symthaea/crates/symthaea-broca/data/*|symthaea/crates/symthaea-broca/models/*) continue ;;
            */target/*|*/.git/*|*.bin|*.safetensors) continue ;;
        esac

        mkdir -p "$(dirname "$dst")"
        cp "$file" "$dst" 2>/dev/null && untracked_count=$((untracked_count + 1))
    done < <(git ls-files --others --exclude-standard -z 2>/dev/null)

    if (( untracked_count > 0 )); then
        echo -e "  ${GREEN}Copied:${NC} ${untracked_count} untracked files"
    fi

    # Symlink large data directories to avoid duplication
    # (training data, model checkpoints — read-only shared resources)
    local data_dirs=(
        "symthaea/crates/symthaea-broca/data"
        "symthaea/crates/symthaea-broca/models"
    )
    for rel_dir in "${data_dirs[@]}"; do
        local src="${REPO_ROOT}/${rel_dir}"
        local dst="${wt_path}/${rel_dir}"
        if [[ -d "$src" && ! -L "$dst" ]]; then
            rm -rf "$dst" 2>/dev/null || true
            mkdir -p "$(dirname "$dst")"
            ln -s "$src" "$dst"
            echo -e "  ${GREEN}Symlinked:${NC} ${rel_dir} → shared data"
        fi
    done

    # Symlink untracked workspace crates (referenced in Cargo.toml but not in git)
    # Without these, cargo fails in the worktree
    for crate_dir in "${REPO_ROOT}"/symthaea/crates/*/; do
        [[ -d "$crate_dir" ]] || continue
        local crate_name=$(basename "$crate_dir")
        local cargo_toml="${crate_dir}Cargo.toml"
        [[ -f "$cargo_toml" ]] || continue

        # Check if git-tracked
        if ! git -C "$REPO_ROOT" ls-files --error-unmatch "${cargo_toml#${REPO_ROOT}/}" &>/dev/null; then
            local dst="${wt_path}/symthaea/crates/${crate_name}"
            if [[ ! -e "$dst" ]]; then
                ln -s "$crate_dir" "$dst"
                echo -e "  ${GREEN}Symlinked:${NC} untracked crate ${crate_name}"
            fi
        fi
    done

    # Also symlink untracked test directories
    for test_dir in "${REPO_ROOT}"/symthaea/crates/*/tests/; do
        [[ -d "$test_dir" ]] || continue
        local crate_name=$(basename "$(dirname "$test_dir")")
        local dst="${wt_path}/symthaea/crates/${crate_name}/tests"
        local src_parent="${wt_path}/symthaea/crates/${crate_name}"

        # Only if the crate dir exists (tracked) but tests/ is untracked
        if [[ -d "$src_parent" && ! -L "$src_parent" && ! -e "$dst" ]]; then
            if ! git -C "$REPO_ROOT" ls-files --error-unmatch "symthaea/crates/${crate_name}/tests/" &>/dev/null 2>&1; then
                ln -s "$test_dir" "$dst"
                echo -e "  ${GREEN}Symlinked:${NC} untracked tests for ${crate_name}"
            fi
        fi
    done

    # Symlink untracked doc directories
    for doc_dir in "${REPO_ROOT}"/symthaea/docs/ "${REPO_ROOT}"/docs/; do
        [[ -d "$doc_dir" ]] || continue
        local rel="${doc_dir#${REPO_ROOT}/}"
        local dst="${wt_path}/${rel}"
        if [[ ! -e "$dst" ]]; then
            ln -s "$doc_dir" "$dst" 2>/dev/null || true
        fi
    done

    # Verify sccache is wired (global ~/.cargo/config.toml or project-level)
    if grep -q "sccache" ~/.cargo/config.toml 2>/dev/null; then
        echo -e "  ${GREEN}sccache:${NC} wired via ~/.cargo/config.toml (global)"
    elif [[ -f "${wt_path}/.cargo/config.toml" ]] && grep -q "sccache" "${wt_path}/.cargo/config.toml" 2>/dev/null; then
        echo -e "  ${GREEN}sccache:${NC} wired via project .cargo/config.toml"
    else
        echo -e "  ${YELLOW}Warning:${NC} sccache not configured — add rustc-wrapper to ~/.cargo/config.toml"
    fi

    echo -e "${GREEN}Ready:${NC} cd $wt_path"
    echo ""
    echo "First build will populate sccache — subsequent builds are instant for cached crates."
    echo "Each worktree has its own target/ directory (no lock contention)."
}

cmd_enter() {
    local name="${1:?Usage: session-worktree.sh enter <name>}"
    local wt_path="${WORKTREE_DIR}/${SESSION_PREFIX}-${name}"

    if [[ ! -d "$wt_path" ]]; then
        echo -e "${RED}No worktree found:${NC} ${SESSION_PREFIX}-${name}"
        echo "Available:"
        cmd_list
        return 1
    fi

    echo "$wt_path"
}

cmd_list() {
    cd "$REPO_ROOT"
    echo -e "${BLUE}Active worktrees:${NC}"
    echo ""

    git worktree list | while read -r path commit branch; do
        local name=$(basename "$path")
        local target_size="no target/"

        if [[ -d "${path}/symthaea/target" ]]; then
            target_size=$(du -sh "${path}/symthaea/target" 2>/dev/null | cut -f1)
        elif [[ -d "${path}/target" ]]; then
            target_size=$(du -sh "${path}/target" 2>/dev/null | cut -f1)
        fi

        # Check if any cargo/rustc processes reference this worktree
        local procs=$(ps aux 2>/dev/null | grep -c "$path" || true)
        local status="${GREEN}idle${NC}"
        if (( procs > 2 )); then
            status="${YELLOW}active (${procs} procs)${NC}"
        fi

        printf "  %-40s %s  target: %s  %b\n" "$name" "$commit" "$target_size" "$status"
    done
}

cmd_cleanup() {
    cd "$REPO_ROOT"
    local cleaned=0

    echo -e "${BLUE}Checking for stale worktrees...${NC}"

    for wt_dir in "${WORKTREE_DIR}"/*/; do
        [[ -d "$wt_dir" ]] || continue
        local name=$(basename "$wt_dir")
        local wt_path="${wt_dir%/}"

        # Check if any processes are using this worktree
        local procs=$(ps aux 2>/dev/null | grep -c "$wt_path" || true)

        if (( procs <= 1 )); then
            # Check age — stale if older than 24 hours with no activity
            local age_hours=0
            if [[ -f "${wt_path}/.git" ]]; then
                local mod_time=$(stat -c %Y "$wt_path" 2>/dev/null || echo 0)
                local now=$(date +%s)
                age_hours=$(( (now - mod_time) / 3600 ))
            fi

            if (( age_hours > 24 )); then
                echo -e "  ${YELLOW}Removing stale:${NC} $name (${age_hours}h old, no active processes)"

                # Clean up target dir first (can be huge)
                rm -rf "${wt_path}/target" "${wt_path}/symthaea/target" 2>/dev/null || true

                # Remove worktree
                local branch=$(git -C "$wt_path" branch --show-current 2>/dev/null || true)
                git worktree remove --force "$wt_path" 2>/dev/null || rm -rf "$wt_path"

                # Clean up branch
                if [[ -n "$branch" && "$branch" == worktree-* ]]; then
                    git branch -D "$branch" 2>/dev/null || true
                fi

                cleaned=$((cleaned + 1))
            else
                echo -e "  ${GREEN}Keeping:${NC} $name (${age_hours}h old)"
            fi
        else
            echo -e "  ${GREEN}Active:${NC} $name ($procs processes)"
        fi
    done

    # Also prune any worktrees git knows about but whose dirs are gone
    git worktree prune 2>/dev/null

    echo -e "${GREEN}Cleaned ${cleaned} stale worktrees.${NC}"
}

cmd_cleanup_all() {
    cd "$REPO_ROOT"
    echo -e "${RED}Removing ALL session worktrees...${NC}"

    for wt_dir in "${WORKTREE_DIR}"/*/; do
        [[ -d "$wt_dir" ]] || continue
        local name=$(basename "$wt_dir")
        local wt_path="${wt_dir%/}"

        echo -e "  Removing: $name"
        rm -rf "${wt_path}/target" "${wt_path}/symthaea/target" 2>/dev/null || true

        local branch=$(git -C "$wt_path" branch --show-current 2>/dev/null || true)
        git worktree remove --force "$wt_path" 2>/dev/null || rm -rf "$wt_path"

        if [[ -n "$branch" && "$branch" == worktree-* ]]; then
            git branch -D "$branch" 2>/dev/null || true
        fi
    done

    git worktree prune 2>/dev/null
    echo -e "${GREEN}Done.${NC}"
}

cmd_status() {
    echo -e "${BLUE}=== System Health ===${NC}"
    echo ""

    # CPU load
    echo -e "${BLUE}Load average:${NC}"
    uptime
    echo ""

    # Top cargo/rustc processes
    echo -e "${BLUE}Cargo/Rust processes:${NC}"
    ps aux --sort=-%cpu 2>/dev/null | grep -E '(rustc|cargo|broca|symthaea)' | grep -v grep | head -10 | \
        awk '{printf "  PID %-8s CPU %-6s MEM %-6s %s\n", $2, $3, $4, substr($0, index($0,$11))}' | \
        cut -c1-120
    echo ""

    # sccache stats
    echo -e "${BLUE}sccache:${NC}"
    sccache --show-stats 2>&1 | grep -E "Cache hits rate|Cache size|Cache location" | sed 's/^/  /'
    echo ""

    # Claude sessions
    local claude_count=$(ps aux 2>/dev/null | grep -c "claude.*unwrapped" || true)
    echo -e "${BLUE}Claude sessions:${NC} $((claude_count - 1))"
    echo ""

    # Worktree summary
    echo -e "${BLUE}Worktrees:${NC}"
    cd "$REPO_ROOT"
    git worktree list 2>/dev/null | wc -l | xargs -I{} echo "  {} total"
}

# Dispatch
case "${1:-help}" in
    create)     cmd_create "${2:-}" ;;
    enter)      cmd_enter "${2:-}" ;;
    list)       cmd_list ;;
    cleanup)    cmd_cleanup ;;
    cleanup-all) cmd_cleanup_all ;;
    status)     cmd_status ;;
    *)
        echo "Usage: session-worktree.sh {create|enter|list|cleanup|cleanup-all|status} [name]"
        echo ""
        echo "Manages git worktrees for concurrent Claude Code sessions."
        echo "Each worktree gets its own target/ directory; sccache shares compiled artifacts."
        ;;
esac
