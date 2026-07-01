#!/usr/bin/env bash
# Sync to Standalone v2 — git subtree split + TOML-aware rewriting
#
# Extracts symthaea/ history via git subtree split, rewrites external
# Cargo.toml path deps using a Python script (not sed), and pushes
# to the standalone Luminous-Dynamics/symthaea repo.
#
# Advantages over v1 (rsync + sed):
#   - Preserves full git commit history
#   - TOML-aware path rewriting (no regex breakage)
#   - Deterministic: same monorepo state always produces same standalone state
#
# Usage:
#   bash symthaea/scripts/sync-to-standalone-v2.sh [--dry-run] [--force]
#
# Prerequisites:
#   - Run from monorepo root (or anywhere with MONOREPO_ROOT set)
#   - SSH access to git@github.com:Luminous-Dynamics/symthaea.git
#   - Python 3 available

set -euo pipefail

# --- Configuration -----------------------------------------------------------

MONOREPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SYMTHAEA_DIR="${MONOREPO_ROOT}/symthaea"
SCRIPT_DIR="${SYMTHAEA_DIR}/scripts"
STANDALONE_REMOTE="git@github.com:Luminous-Dynamics/symthaea.git"
WORK_DIR="/tmp/symthaea-subtree-sync"
SPLIT_BRANCH="symthaea-standalone-split"

DRY_RUN=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --force)   FORCE=true ;;
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

# --- Validate ----------------------------------------------------------------

[ -f "${SYMTHAEA_DIR}/Cargo.toml" ] || error "Not in monorepo root"
command -v python3 >/dev/null || error "python3 required"
command -v git >/dev/null || error "git required"

info "Monorepo: ${MONOREPO_ROOT}"
info "Remote:   ${STANDALONE_REMOTE}"
$DRY_RUN && warn "DRY RUN mode"
echo

# --- Step 1: git subtree split -----------------------------------------------

info "Running git subtree split --prefix=symthaea..."
SPLIT_START=$(date +%s)

cd "${MONOREPO_ROOT}"
SPLIT_SHA=$(git subtree split --prefix=symthaea -b "${SPLIT_BRANCH}" 2>&1 | tail -1)

SPLIT_END=$(date +%s)
SPLIT_SECS=$((SPLIT_END - SPLIT_START))
SPLIT_COMMITS=$(git rev-list --count "${SPLIT_BRANCH}")
ok "Split complete: ${SPLIT_COMMITS} commits in ${SPLIT_SECS}s (HEAD: ${SPLIT_SHA:0:12})"
echo

# --- Step 2: Clone split into work directory ---------------------------------

info "Preparing work directory..."
rm -rf "${WORK_DIR}"
git clone --branch "${SPLIT_BRANCH}" "${MONOREPO_ROOT}" "${WORK_DIR}" 2>/dev/null
cd "${WORK_DIR}"

# Add standalone as remote
git remote add standalone "${STANDALONE_REMOTE}" 2>/dev/null || true
git fetch standalone main 2>/dev/null || info "First push — standalone/main doesn't exist yet"
echo

# --- Step 3: Ensure stubs exist ----------------------------------------------

info "Ensuring stub crates exist..."

ensure_stub() {
    local name="$1"
    local stub_dir="stubs/${name}"

    if [ -d "${stub_dir}" ]; then
        info "  ${stub_dir}/ already exists"
        return
    fi

    info "  Creating ${stub_dir}/"
    mkdir -p "${stub_dir}/src"

    # Create minimal Cargo.toml
    cat > "${stub_dir}/Cargo.toml" <<TOML
[package]
name = "${name}"
version = "0.1.0"
edition = "2021"
description = "Stub crate for standalone symthaea builds"

[dependencies]
sha3 = "0.10"
TOML

    # Create minimal lib.rs
    echo "// Stub for standalone builds" > "${stub_dir}/src/lib.rs"
}

ensure_stub "mycelix-fl-core"
ensure_stub "mycelix-sdk"
echo

# --- Step 4: Rewrite Cargo.toml path deps ------------------------------------

info "Rewriting Cargo.toml external path deps..."

if [ -f "${SCRIPT_DIR}/rewrite-cargo-toml.py" ]; then
    if $DRY_RUN; then
        python3 "${SCRIPT_DIR}/rewrite-cargo-toml.py" "${WORK_DIR}/Cargo.toml" --dry-run
    else
        python3 "${SCRIPT_DIR}/rewrite-cargo-toml.py" "${WORK_DIR}/Cargo.toml"
    fi
else
    # Fallback: inline sed (matches any quoted path for known crates)
    warn "Python rewriter not found, falling back to sed"
    if ! $DRY_RUN; then
        sed -i 's|^mycelix-fl-core = { path = "[^"]*"|mycelix-fl-core = { path = "stubs/mycelix-fl-core"|' Cargo.toml
        sed -i 's|^mycelix-sdk = { path = "[^"]*"|mycelix-sdk = { path = "stubs/mycelix-sdk"|' Cargo.toml
    fi
fi
echo

# --- Step 5: Copy standalone-only files from standalone repo -----------------

# Preserve files that only exist in standalone (CI, stubs, README overrides)
if git rev-parse standalone/main >/dev/null 2>&1; then
    info "Preserving standalone-only files..."
    for f in .github/workflows/ci.yml .github/workflows/benchmark.yml CI_TRIAGE.md; do
        if git show standalone/main:"${f}" >/dev/null 2>&1; then
            mkdir -p "$(dirname "${f}")"
            git show standalone/main:"${f}" > "${f}" 2>/dev/null && \
                info "  Preserved ${f}" || true
        fi
    done

    # Preserve stubs from standalone if they exist and we didn't just create them
    for stub in stubs/mycelix-fl-core stubs/mycelix-sdk; do
        if git show standalone/main:"${stub}/Cargo.toml" >/dev/null 2>&1; then
            mkdir -p "${stub}/src"
            git show standalone/main:"${stub}/Cargo.toml" > "${stub}/Cargo.toml" 2>/dev/null
            git show standalone/main:"${stub}/src/lib.rs" > "${stub}/src/lib.rs" 2>/dev/null
            info "  Preserved ${stub}/ from standalone"
        fi
    done
    echo
fi

# --- Step 6: Stage and show diff ---------------------------------------------

git add -A
CHANGES=$(git diff --cached --name-only | wc -l)

if [ "$CHANGES" -eq 0 ]; then
    ok "No changes — standalone is up to date with monorepo."
    # Cleanup
    cd "${MONOREPO_ROOT}"
    rm -rf "${WORK_DIR}"
    exit 0
fi

info "Changes:"
git diff --cached --stat
printf "\n${BOLD}%s files changed${RESET}\n\n" "$CHANGES"

# --- Step 7: Commit and push -------------------------------------------------

if $DRY_RUN; then
    warn "DRY RUN — skipping commit and push"
    cd "${MONOREPO_ROOT}"
    rm -rf "${WORK_DIR}"
    exit 0
fi

MONOREPO_SHA=$(git -C "${MONOREPO_ROOT}" rev-parse --short HEAD)
COMMIT_MSG="sync: update from monorepo @ ${MONOREPO_SHA} ($(date +%Y-%m-%d))"

git commit -m "${COMMIT_MSG}" --allow-empty

if $FORCE; then
    info "Force pushing to standalone/main..."
    git push standalone HEAD:main --force
else
    printf "${YELLOW}Push to standalone/main?${RESET} [y/N] "
    read -r confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        git push standalone HEAD:main --force
    else
        info "Not pushed. Manual push:"
        echo "  cd ${WORK_DIR} && git push standalone HEAD:main --force"
        exit 0
    fi
fi

ok "Pushed to standalone/main"

# --- Cleanup -----------------------------------------------------------------

cd "${MONOREPO_ROOT}"
rm -rf "${WORK_DIR}"
git branch -D "${SPLIT_BRANCH}" 2>/dev/null || true

ok "Sync complete!"
