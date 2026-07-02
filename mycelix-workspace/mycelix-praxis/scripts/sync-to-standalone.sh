#!/usr/bin/env bash
# Sync to Standalone — Push mycelix-edunet to Luminous-Dynamics/mycelix-edunet repo
#
# Syncs the mycelix-edunet hApp from the monorepo to the standalone public
# GitHub repository, applying exclusions and sanity checks.
#
# Usage:
#   bash mycelix-edunet/scripts/sync-to-standalone.sh [--dry-run] [--skip-check] [--force]
#
# Options:
#   --dry-run      Show what would change without modifying the standalone repo
#   --skip-check   Skip the post-sync `cargo check` verification
#   --force        Commit and push without interactive confirmation

set -euo pipefail

# --- Configuration -----------------------------------------------------------

MONOREPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EDUNET_DIR="${MONOREPO_ROOT}/mycelix-edunet"
STANDALONE_REPO="/tmp/mycelix-edunet-standalone-sync"
STANDALONE_REMOTE="git@github.com:Luminous-Dynamics/mycelix-edunet.git"

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

if [ ! -f "${EDUNET_DIR}/Cargo.toml" ]; then
    error "Cannot find ${EDUNET_DIR}/Cargo.toml — run from monorepo root"
fi

info "Monorepo root: ${MONOREPO_ROOT}"
info "EduNet dir:    ${EDUNET_DIR}"
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
    --exclude='.claude/'
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

# --- Sync directories ---------------------------------------------------------
#
# Sync all project directories from monorepo to standalone.
# Missing directories are silently skipped.

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

SYNC_DIRS=(
    crates
    zomes
    apps
    conductor
    dna
    happ
    tests
    schemas
    examples
    scripts
    docs
    .github
)

info "=== Syncing directories ==="
for dir in "${SYNC_DIRS[@]}"; do
    sync_dir "${EDUNET_DIR}/${dir}" "${STANDALONE_REPO}/${dir}"
done
echo

# --- Copy root files ----------------------------------------------------------
#
# Individual files at the project root that should be synced.

ROOT_FILES=(
    Cargo.toml
    Cargo.lock
    rust-toolchain.toml
    flake.nix
    flake.lock
    deny.toml
    codecov.yml
    Makefile
    docker-compose.yml
    docker-compose.prod.yml
    README.md
    CONTRIBUTING.md
    ROADMAP.md
    GOVERNANCE.md
    SECURITY.md
    CHANGELOG.md
    LICENSE
)

info "=== Syncing root files ==="
for f in "${ROOT_FILES[@]}"; do
    if [ -f "${EDUNET_DIR}/${f}" ]; then
        if $DRY_RUN; then
            info "Would copy ${f}"
        else
            cp "${EDUNET_DIR}/${f}" "${STANDALONE_REPO}/${f}"
            ok "Copied ${f}"
        fi
    else
        warn "Skipping ${f} (not found)"
    fi
done
echo

# --- Strip workspace lints (CI compatibility) ---------------------------------
#
# If workspace lints exist, they can conflict with CI's feature-gated clippy
# flags. Strip them from the root Cargo.toml in the standalone repo.

if ! $DRY_RUN; then
    STANDALONE_TOML="${STANDALONE_REPO}/Cargo.toml"
    if grep -q '\[workspace\.lints' "$STANDALONE_TOML" 2>/dev/null; then
        info "Stripping [workspace.lints] from standalone Cargo.toml..."
        # Remove [workspace.lints.*] sections (everything from [workspace.lints
        # until the next top-level section or EOF)
        sed -i '/^\[workspace\.lints/,/^\[/{/^\[workspace\.lints/d;/^\[/!d}' "$STANDALONE_TOML"
        # Also remove `lints.workspace = true` from member crates
        while IFS= read -r toml_file; do
            if grep -q 'lints\.workspace' "$toml_file" 2>/dev/null; then
                sed -i '/^\[lints\]/,/^\[/{/^\[lints\]/d;/^\[/!d}' "$toml_file"
                sed -i '/^lints\.workspace/d' "$toml_file"
            fi
        done < <(find "${STANDALONE_REPO}" -name "Cargo.toml" -not -path "*/target/*" 2>/dev/null)
        ok "Stripped workspace lints for CI compatibility"
    else
        ok "No workspace lints to strip"
    fi
fi
echo

# --- Post-sync cargo check (sanity check) ------------------------------------

if ! $DRY_RUN && ! $SKIP_CHECK; then
    info "Running cargo check --workspace (sanity check)..."
    if (cd "${STANDALONE_REPO}" && cargo check --workspace 2>&1 | tail -10); then
        ok "cargo check passed"
    else
        warn "cargo check failed — the sync may have issues"
        warn "Run with --skip-check to bypass, or investigate manually:"
        echo "  cd ${STANDALONE_REPO} && cargo check --workspace"
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
