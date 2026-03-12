#!/usr/bin/env bash
# Sync to Standalone — Push mycelix clusters to Luminous-Dynamics/mycelix repo
#
# Syncs all mycelix clusters, shared crates, and workspace from the monorepo
# to the standalone public GitHub repository.
#
# Usage:
#   bash mycelix-workspace/scripts/sync-to-standalone.sh [--dry-run] [--skip-check] [--force]
#
# Options:
#   --dry-run      Show what would change without modifying the standalone repo
#   --skip-check   Skip the post-sync `cargo check` verification
#   --force        Commit and push without interactive confirmation

set -euo pipefail

# --- Configuration -----------------------------------------------------------

MONOREPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STANDALONE_REPO="/tmp/mycelix-standalone-sync"
STANDALONE_REMOTE="git@github.com:Luminous-Dynamics/mycelix.git"

DRY_RUN=false
SKIP_CHECK=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=true ;;
        --skip-check) SKIP_CHECK=true ;;
        --force)      FORCE=true ;;
    esac
done

# --- Colors -------------------------------------------------------------------

GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
CYAN="\033[36m"
RESET="\033[0m"
BOLD="\033[1m"

info()  { printf "${CYAN}[info]${RESET}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[warn]${RESET}  %s\n" "$*"; }
ok()    { printf "${GREEN}[ok]${RESET}    %s\n" "$*"; }
error() { printf "${RED}[error]${RESET} %s\n" "$*"; exit 1; }

# --- Validate monorepo --------------------------------------------------------

if [ ! -d "${MONOREPO_ROOT}/mycelix-finance" ]; then
    error "Cannot find ${MONOREPO_ROOT}/mycelix-finance — run from monorepo root"
fi

info "Monorepo root: ${MONOREPO_ROOT}"
info "Standalone:    ${STANDALONE_REPO}"
if $DRY_RUN; then
    warn "DRY RUN — no commits or pushes will be made"
fi
echo

# --- Clone or update standalone repo ------------------------------------------

if [ -d "${STANDALONE_REPO}/.git" ]; then
    info "Updating existing standalone clone..."
    git -C "${STANDALONE_REPO}" fetch origin
    git -C "${STANDALONE_REPO}" checkout main
    git -C "${STANDALONE_REPO}" reset --hard origin/main
else
    info "Cloning standalone repo..."
    git clone "${STANDALONE_REMOTE}" "${STANDALONE_REPO}"
fi
echo

# --- Rsync options ------------------------------------------------------------

RSYNC_EXCLUDE=(
    --exclude='target/'
    --exclude='.DS_Store'
    --exclude='__pycache__/'
    --exclude='node_modules/'
    --exclude='corpus/'
    --exclude='artifacts/'
    --exclude='coverage/'
    --exclude='.conductor/'
)
RSYNC_OPTS=(-a --delete "${RSYNC_EXCLUDE[@]}")
if $DRY_RUN; then
    RSYNC_OPTS+=(--dry-run -v)
else
    RSYNC_OPTS+=(-v)
fi

# --- Sync cluster directories ------------------------------------------------
#
# Each cluster is its own workspace in the standalone repo.
# Path mapping is 1:1 between monorepo and standalone.

CLUSTERS=(
    mycelix-commons
    mycelix-civic
    mycelix-hearth
    mycelix-finance
    mycelix-governance
    mycelix-identity
    mycelix-personal
    mycelix-attribution
)

sync_dir() {
    local src="$1"
    local dst="$2"
    if [ ! -d "$src" ]; then
        warn "Skipping $(basename "$src")/ (not found in monorepo)"
        return
    fi
    info "Syncing $(basename "$dst")/"
    rsync "${RSYNC_OPTS[@]}" "$src/" "$dst/"
}

info "=== Syncing cluster directories ==="
for cluster in "${CLUSTERS[@]}"; do
    sync_dir "${MONOREPO_ROOT}/${cluster}" "${STANDALONE_REPO}/${cluster}"
done
echo

# --- Sync shared crates ------------------------------------------------------
#
# Shared crates live in different monorepo locations but map to crates/ in standalone.
#
# Monorepo                                          → Standalone
# crates/mycelix-bridge-common/                     → crates/mycelix-bridge-common/
# crates/mycelix-bridge-entry-types/                → crates/mycelix-bridge-entry-types/
# mycelix-core/libs/mycelix-core-types/             → crates/mycelix-core-types/
# mycelix-core/libs/feldman-dkg/                    → crates/feldman-dkg/

info "=== Syncing shared crates ==="

# Direct path crates (same location in both repos)
for crate_name in mycelix-bridge-common mycelix-bridge-entry-types; do
    sync_dir \
        "${MONOREPO_ROOT}/crates/${crate_name}" \
        "${STANDALONE_REPO}/crates/${crate_name}"
done

# Remapped crates (different paths in monorepo vs standalone)
sync_dir \
    "${MONOREPO_ROOT}/mycelix-core/libs/mycelix-core-types" \
    "${STANDALONE_REPO}/crates/mycelix-core-types"

sync_dir \
    "${MONOREPO_ROOT}/mycelix-core/libs/feldman-dkg" \
    "${STANDALONE_REPO}/crates/feldman-dkg"

echo

# --- Sync mycelix-workspace (SDKs, happs, scripts, etc.) --------------------

info "=== Syncing mycelix-workspace ==="
sync_dir "${MONOREPO_ROOT}/mycelix-workspace" "${STANDALONE_REPO}/mycelix-workspace"
echo

# --- Rewrite path dependencies for standalone layout -------------------------
#
# In the monorepo, some shared crates live at mycelix-core/libs/*.
# In the standalone repo, they live at crates/*.
# Clusters that depend on these need their Cargo.toml paths rewritten.

if ! $DRY_RUN; then
    info "=== Rewriting path dependencies for standalone layout ==="

    # Map: monorepo path → standalone path (relative from cluster root)
    # ../mycelix-core/libs/mycelix-core-types → ../crates/mycelix-core-types
    # ../mycelix-core/libs/feldman-dkg        → ../crates/feldman-dkg
    while IFS= read -r toml_file; do
        if grep -q 'mycelix-core/libs/' "$toml_file" 2>/dev/null; then
            sed -i 's|mycelix-core/libs/mycelix-core-types|crates/mycelix-core-types|g' "$toml_file"
            sed -i 's|mycelix-core/libs/feldman-dkg|crates/feldman-dkg|g' "$toml_file"
            ok "Rewrote paths in $(basename "$(dirname "$(dirname "$toml_file")")")/...Cargo.toml"
        fi
    done < <(find "${STANDALONE_REPO}" -name "Cargo.toml" -not -path "*/target/*" 2>/dev/null)
fi
echo

# --- Copy individual files that live at the standalone root ------------------
# Note: Do NOT overwrite CLAUDE.md, README.md, .github/, .gitmodules — these
# are standalone-specific and maintained there.

info "=== Preserving standalone-specific files ==="
ok "CLAUDE.md, README.md, LICENSE, .github/, .gitmodules are standalone-only (not synced)"
echo

# --- Post-sync cargo check (finance cluster as sanity check) -----------------

if ! $DRY_RUN && ! $SKIP_CHECK; then
    info "Running cargo check on mycelix-finance (sanity check)..."
    if (cd "${STANDALONE_REPO}/mycelix-finance" && cargo check 2>&1 | tail -5); then
        ok "cargo check passed"
    else
        warn "cargo check failed — the sync may have issues"
        warn "Run with --skip-check to bypass, or investigate manually:"
        echo "  cd ${STANDALONE_REPO}/mycelix-finance && cargo check"
    fi
    echo
elif $SKIP_CHECK; then
    info "Skipping cargo check (--skip-check)"
    echo
fi

# --- Show diff summary --------------------------------------------------------

info "Changes in standalone repo:"
echo
git -C "${STANDALONE_REPO}" add -A
git -C "${STANDALONE_REPO}" diff --cached --stat
echo

CHANGES=$(git -C "${STANDALONE_REPO}" diff --cached --name-only | wc -l)
if [ "$CHANGES" -eq 0 ]; then
    info "No changes to sync — standalone is up to date."
    exit 0
fi

printf "${BOLD}%s files changed${RESET}\n" "$CHANGES"
echo

# --- Commit -------------------------------------------------------------------

if $DRY_RUN; then
    warn "DRY RUN — skipping commit and push"
    git -C "${STANDALONE_REPO}" reset HEAD -- . >/dev/null 2>&1
    exit 0
fi

MONOREPO_SHA=$(git -C "${MONOREPO_ROOT}" rev-parse --short HEAD)
COMMIT_MSG="sync: update from monorepo @ ${MONOREPO_SHA} ($(date +%Y-%m-%d))"

if $FORCE; then
    info "Committing: ${COMMIT_MSG}"
    git -C "${STANDALONE_REPO}" commit -m "${COMMIT_MSG}"
else
    printf "${YELLOW}Commit with message:${RESET} %s\n" "${COMMIT_MSG}"
    printf "Proceed? [y/N] "
    read -r confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        warn "Aborted — changes staged but not committed"
        git -C "${STANDALONE_REPO}" reset HEAD -- . >/dev/null 2>&1
        exit 1
    fi
    git -C "${STANDALONE_REPO}" commit -m "${COMMIT_MSG}"
fi

printf "${GREEN}Committed.${RESET}\n"
echo

# --- Push ---------------------------------------------------------------------

if $FORCE; then
    info "Pushing to origin/main..."
    git -C "${STANDALONE_REPO}" push origin main
    ok "Pushed to origin/main"
else
    printf "${YELLOW}Push to origin/main?${RESET} [y/N] "
    read -r push_confirm
    if [[ "$push_confirm" =~ ^[Yy]$ ]]; then
        git -C "${STANDALONE_REPO}" push origin main
        ok "Pushed to origin/main"
    else
        info "Not pushed. You can push manually:"
        echo "  git -C ${STANDALONE_REPO} push origin main"
    fi
fi
