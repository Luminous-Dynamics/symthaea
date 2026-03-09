#!/usr/bin/env bash
# Sync to Standalone — Push symthaea crate to Luminous-Dynamics/symthaea repo
#
# Syncs the symthaea crate from the monorepo to the standalone public GitHub
# repository, applying Cargo.toml fixups for standalone builds (stub paths
# for mycelix dependencies).
#
# Usage:
#   bash symthaea/scripts/sync-to-standalone.sh [--dry-run]

set -euo pipefail

# --- Configuration -----------------------------------------------------------

MONOREPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SYMTHAEA_DIR="${MONOREPO_ROOT}/symthaea"
STANDALONE_REPO="/tmp/symthaea-standalone-sync"
STANDALONE_REMOTE="git@github.com:Luminous-Dynamics/symthaea.git"

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
fi

# --- Colors -------------------------------------------------------------------

GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
CYAN="\033[36m"
RESET="\033[0m"
BOLD="\033[1m"

info()  { printf "${CYAN}[info]${RESET}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[warn]${RESET}  %s\n" "$*"; }
error() { printf "${RED}[error]${RESET} %s\n" "$*"; exit 1; }

# --- Validate monorepo --------------------------------------------------------

if [ ! -f "${SYMTHAEA_DIR}/Cargo.toml" ]; then
    error "Cannot find ${SYMTHAEA_DIR}/Cargo.toml — run from monorepo root"
fi

info "Monorepo root: ${MONOREPO_ROOT}"
info "Symthaea dir:  ${SYMTHAEA_DIR}"
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

# --- Rsync directories -------------------------------------------------------

RSYNC_OPTS=(-av --delete --exclude='target/')
if $DRY_RUN; then
    RSYNC_OPTS+=(--dry-run)
fi

sync_dir() {
    local dir="$1"
    info "Syncing ${dir}/"
    rsync "${RSYNC_OPTS[@]}" \
        "${SYMTHAEA_DIR}/${dir}/" \
        "${STANDALONE_REPO}/${dir}/"
}

sync_dir "src"
sync_dir "crates"
sync_dir "tests"
sync_dir "examples"
sync_dir "papers"
sync_dir "scripts"

# .github lives at monorepo level for standalone, sync from symthaea's own
if [ -d "${SYMTHAEA_DIR}/.github" ]; then
    info "Syncing .github/"
    rsync "${RSYNC_OPTS[@]}" \
        "${SYMTHAEA_DIR}/.github/" \
        "${STANDALONE_REPO}/.github/"
fi

echo

# --- Copy individual files ----------------------------------------------------

INDIVIDUAL_FILES=(
    "rust-toolchain.toml"
    "clippy.toml"
    "deny.toml"
    "rustfmt.toml"
)

for f in "${INDIVIDUAL_FILES[@]}"; do
    if [ -f "${SYMTHAEA_DIR}/${f}" ]; then
        info "Copying ${f}"
        if ! $DRY_RUN; then
            cp "${SYMTHAEA_DIR}/${f}" "${STANDALONE_REPO}/${f}"
        fi
    else
        warn "Skipping ${f} (not found in monorepo)"
    fi
done

echo

# --- Cargo.toml with standalone fixups ----------------------------------------

info "Copying Cargo.toml with standalone fixups..."

if ! $DRY_RUN; then
    cp "${SYMTHAEA_DIR}/Cargo.toml" "${STANDALONE_REPO}/Cargo.toml"

    # Fix external path deps → stub crates (match any quoted path)
    sed -i 's|^mycelix-fl-core = { path = "[^"]*"|mycelix-fl-core = { path = "stubs/mycelix-fl-core"|' \
        "${STANDALONE_REPO}/Cargo.toml"

    sed -i 's|^mycelix-sdk = { path = "[^"]*"|mycelix-sdk = { path = "stubs/mycelix-sdk"|' \
        "${STANDALONE_REPO}/Cargo.toml"

    info "Cargo.toml fixups applied"
else
    info "(dry-run) Would copy and patch Cargo.toml"
fi

echo

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

COMMIT_MSG="sync: update from monorepo ($(date +%Y-%m-%d))"

printf "${YELLOW}Commit with message:${RESET} %s\n" "$COMMIT_MSG"
printf "Proceed? [y/N] "
read -r confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    warn "Aborted — changes staged but not committed"
    git -C "${STANDALONE_REPO}" reset HEAD -- . >/dev/null 2>&1
    exit 1
fi

git -C "${STANDALONE_REPO}" commit -m "${COMMIT_MSG}"
printf "${GREEN}Committed.${RESET}\n"
echo

# --- Push ---------------------------------------------------------------------

printf "${YELLOW}Push to origin/main?${RESET} [y/N] "
read -r push_confirm
if [[ "$push_confirm" =~ ^[Yy]$ ]]; then
    git -C "${STANDALONE_REPO}" push origin main
    printf "${GREEN}Pushed to origin/main.${RESET}\n"
else
    info "Not pushed. You can push manually:"
    echo "  git -C ${STANDALONE_REPO} push origin main"
fi
