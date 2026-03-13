#!/usr/bin/env bash
# Sync to Standalone — Push symthaea crate to Luminous-Dynamics/symthaea repo
#
# Syncs the symthaea crate from the monorepo to the standalone public GitHub
# repository, applying Cargo.toml fixups for standalone builds (stub paths
# for mycelix dependencies).
#
# Usage:
#   bash symthaea/scripts/sync-to-standalone.sh [--dry-run] [--skip-check] [--force]
#
# Options:
#   --dry-run      Show what would change without modifying the standalone repo
#   --skip-check   Skip the post-sync `cargo check` verification
#   --force        Commit and push without interactive confirmation

set -euo pipefail

# --- Configuration -----------------------------------------------------------

MONOREPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SYMTHAEA_DIR="${MONOREPO_ROOT}/symthaea"
STANDALONE_REPO="/tmp/symthaea-standalone-sync"
STANDALONE_REMOTE="git@github.com:Luminous-Dynamics/symthaea.git"

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

# --- Verify stubs exist in standalone ----------------------------------------

STUBS_OK=true
for stub in mycelix-fl-core mycelix-sdk; do
    if [ ! -f "${STANDALONE_REPO}/stubs/${stub}/Cargo.toml" ]; then
        warn "Stub missing: stubs/${stub}/Cargo.toml"
        STUBS_OK=false
    elif [ ! -f "${STANDALONE_REPO}/stubs/${stub}/src/lib.rs" ]; then
        warn "Stub missing: stubs/${stub}/src/lib.rs"
        STUBS_OK=false
    else
        ok "Stub verified: stubs/${stub}/"
    fi
done

if ! $STUBS_OK; then
    error "Stubs are missing in standalone repo. The stubs/ directory must exist with mycelix-fl-core and mycelix-sdk stub crates. These are maintained in the standalone repo, not synced from the monorepo."
fi
echo

# --- Rsync directories -------------------------------------------------------
#
# All directories that need to be synced to the standalone repo.
# Exclusions:
#   - target/        (build artifacts)
#   - __pyphi_cache__/ (python cache)
#   - logs/          (runtime output)
#   - output/        (runtime output)
#   - result/        (runtime output)
#   - results/       (runtime output)
#   - benchmark_output/ (runtime output)
#   - video_output/  (runtime output)
#   - datasets/      (large data files)
#   - zenodo-dataset/ (large data files)
#   - stubs/         (maintained in standalone, not monorepo)
#   - archive/       (old/archived code)
#   - book/          (mdbook output)
#   - data/          (large data)
#   - studies/       (local studies output)
#   - symthaea/      (nested dir, not needed)

RSYNC_EXCLUDE=(
    --exclude='target/'
    --exclude='__pyphi_cache__/'
    --exclude='.DS_Store'
)
RSYNC_OPTS=(-a --delete "${RSYNC_EXCLUDE[@]}")
if $DRY_RUN; then
    RSYNC_OPTS+=(--dry-run -v)
else
    RSYNC_OPTS+=(-v)
fi

# Directories that contain source code and are needed for compilation/CI
SOURCE_DIRS=(
    src
    crates
    symthaea-core
    tests
    examples
    benches
)

# Directories that contain supporting files needed by CI or the project
SUPPORT_DIRS=(
    .github
    papers
    scripts
    docs
    api
    completions
    dashboard
    demos
    figures
    # models — excluded: 21GB of ONNX files, too large for standalone
    nix
    proptest-regressions
    rust-sentinels
    static
    systemd
    tla
    tools
    validation
)

sync_dir() {
    local dir="$1"
    if [ ! -d "${SYMTHAEA_DIR}/${dir}" ]; then
        warn "Skipping ${dir}/ (not found in monorepo)"
        return
    fi
    info "Syncing ${dir}/"
    rsync "${RSYNC_OPTS[@]}" \
        "${SYMTHAEA_DIR}/${dir}/" \
        "${STANDALONE_REPO}/${dir}/"
}

info "=== Syncing source directories ==="
for dir in "${SOURCE_DIRS[@]}"; do
    sync_dir "$dir"
done

echo
info "=== Syncing support directories ==="
for dir in "${SUPPORT_DIRS[@]}"; do
    sync_dir "$dir"
done

# Directories NOT synced (with explanation):
#   archive/          - historical, not needed in standalone
#   audio_output/     - runtime artifacts (empty OK if present in standalone)
#   benchmark_output/ - runtime artifacts
#   book/             - mdbook build output
#   data/             - large local data files
#   datasets/         - large datasets
#   logs/             - runtime logs
#   output/           - runtime output
#   result/           - runtime output
#   results/          - runtime output
#   studies/          - local analysis output
#   symthaea/         - nested duplicate (artifact of subtree split)
#   video_output/     - runtime output
#   zenodo-dataset/   - large archival data

echo

# --- Copy individual files ----------------------------------------------------

INDIVIDUAL_FILES=(
    "Cargo.lock"
    "rust-toolchain.toml"
    "clippy.toml"
    "deny.toml"
    "rustfmt.toml"
    "build.rs"
    "LICENSE"
    "CHANGELOG.md"
    "CONTRIBUTING.md"
    "SECURITY.md"
    "SAFETY.md"
)

for f in "${INDIVIDUAL_FILES[@]}"; do
    if [ -f "${SYMTHAEA_DIR}/${f}" ]; then
        info "Copying ${f}"
        if ! $DRY_RUN; then
            cp "${SYMTHAEA_DIR}/${f}" "${STANDALONE_REPO}/${f}"
        fi
    fi
done

echo

# --- Cargo.toml with standalone fixups ----------------------------------------

info "Copying Cargo.toml with standalone fixups..."

if ! $DRY_RUN; then
    cp "${SYMTHAEA_DIR}/Cargo.toml" "${STANDALONE_REPO}/Cargo.toml"

    # Rewrite external path deps to point to stubs/.
    # These sed patterns match the dependency name at the start of the line,
    # then replace only the path = "..." value, preserving all other keys
    # (optional, default-features, features, etc.).
    #
    # Known external deps (escaping the symthaea/ tree via ../):
    #   mycelix-fl-core = { path = "../mycelix-workspace/crates/mycelix-fl-core", ... }
    #   mycelix-sdk     = { path = "../mycelix-workspace/sdk-ts/../sdk", ... }

    sed -i 's|^\(mycelix-fl-core\s*=\s*{\s*path\s*=\s*\)"[^"]*"|\1"stubs/mycelix-fl-core"|' \
        "${STANDALONE_REPO}/Cargo.toml"

    sed -i 's|^\(mycelix-sdk\s*=\s*{\s*path\s*=\s*\)"[^"]*"|\1"stubs/mycelix-sdk"|' \
        "${STANDALONE_REPO}/Cargo.toml"

    # Verify the rewrites actually happened
    REWRITE_OK=true
    if ! grep -q 'mycelix-fl-core.*stubs/mycelix-fl-core' "${STANDALONE_REPO}/Cargo.toml"; then
        warn "Failed to rewrite mycelix-fl-core path"
        REWRITE_OK=false
    fi
    if ! grep -q 'mycelix-sdk.*stubs/mycelix-sdk' "${STANDALONE_REPO}/Cargo.toml"; then
        warn "Failed to rewrite mycelix-sdk path"
        REWRITE_OK=false
    fi

    # Scan for any remaining external path deps (paths containing ../ that
    # escape the workspace). Sub-crate internal paths like "../symthaea-fep"
    # or "../../symthaea-core" are fine since they stay within the tree.
    ESCAPED_PATHS=$(grep -n 'path\s*=\s*"[^"]*\.\./\.\.' "${STANDALONE_REPO}/Cargo.toml" | \
                    grep -v '^#' | \
                    grep -v 'stubs/' || true)
    if [ -n "$ESCAPED_PATHS" ]; then
        warn "Cargo.toml still has external path deps:"
        echo "$ESCAPED_PATHS"
        REWRITE_OK=false
    fi

    # Strip workspace lints — they conflict with CI's feature-gated clippy -A flags.
    # Crate-level #![deny(...)] in lib.rs provides local lint enforcement instead.
    sed -i '/^\[workspace\.lints\.clippy\]/,/^$/d' "${STANDALONE_REPO}/Cargo.toml"
    sed -i '/^# Workspace-wide lint configuration/d' "${STANDALONE_REPO}/Cargo.toml"
    sed -i '/^# These lints apply on top/d' "${STANDALONE_REPO}/Cargo.toml"
    # Remove [lints] workspace = true from main crate
    sed -i '/^\[lints\]/{N;/workspace = true/d;}' "${STANDALONE_REPO}/Cargo.toml"
    # Remove [lints] workspace = true from symthaea-core
    if [ -f "${STANDALONE_REPO}/symthaea-core/Cargo.toml" ]; then
        sed -i '/^\[lints\]/{N;/workspace = true/d;}' "${STANDALONE_REPO}/symthaea-core/Cargo.toml"
    fi
    # Remove [lints] workspace = true from all sub-crates
    for subcrate_toml in "${STANDALONE_REPO}"/crates/*/Cargo.toml; do
        if [ -f "$subcrate_toml" ]; then
            sed -i '/^\[lints\]/{N;/workspace = true/d;}' "$subcrate_toml"
        fi
    done
    ok "Stripped workspace lints (incompatible with CI clippy config)"

    # Remove workspace members that don't exist in the standalone repo
    # (e.g., symthaea-crucible exists in monorepo but isn't synced)
    TOML="${STANDALONE_REPO}/Cargo.toml"
    REMOVED_MEMBERS=0
    while IFS= read -r line; do
        crate_path=$(echo "$line" | sed -n 's/.*"\(crates\/[^"]*\)".*/\1/p')
        if [ -n "$crate_path" ] && [ ! -d "${STANDALONE_REPO}/${crate_path}" ]; then
            sed -i "\|\"${crate_path}\"|d" "$TOML"
            REMOVED_MEMBERS=$((REMOVED_MEMBERS + 1))
            warn "Removed missing workspace member: ${crate_path}"
        fi
    done < <(grep '"crates/' "$TOML")
    if [ $REMOVED_MEMBERS -gt 0 ]; then
        ok "Removed $REMOVED_MEMBERS missing workspace members"
    fi

    if $REWRITE_OK; then
        ok "Cargo.toml fixups verified"
    else
        error "Cargo.toml rewrite verification failed — check the sed patterns"
    fi
else
    info "(dry-run) Would copy and patch Cargo.toml"
fi

echo

# --- Scan sub-crate Cargo.tomls for escaping paths ---------------------------
#
# All sub-crates should only reference paths within the symthaea tree.
# This check catches any new deps that might have been added pointing to
# monorepo-only locations.

info "Scanning sub-crate Cargo.tomls for external path deps..."
SUBCRATE_ISSUES=false
while IFS= read -r toml_file; do
    # Get path relative to standalone root
    rel_path="${toml_file#${STANDALONE_REPO}/}"
    # Skip the root Cargo.toml (already handled) and stubs
    if [ "$rel_path" = "Cargo.toml" ] || [[ "$rel_path" == stubs/* ]]; then
        continue
    fi
    # Look for paths that escape the workspace (contain ../ going above crates/)
    # We need to check if any path dep resolves outside the standalone tree.
    # A path like "../../symthaea-core" from crates/X/ resolves to symthaea-core/ — fine.
    # A path like "../../../mycelix-workspace" would be bad.
    BAD_PATHS=$(grep -n 'path\s*=\s*"[^"]*\.\./\.\./\.\.' "$toml_file" 2>/dev/null | \
                grep -v '^#' || true)
    if [ -n "$BAD_PATHS" ]; then
        warn "External path dep in ${rel_path}:"
        echo "  $BAD_PATHS"
        SUBCRATE_ISSUES=true
    fi
done < <(find "${STANDALONE_REPO}" -name "Cargo.toml" -not -path "*/target/*" 2>/dev/null)

if $SUBCRATE_ISSUES; then
    warn "Some sub-crates have path deps escaping the workspace tree"
else
    ok "All sub-crate Cargo.tomls OK"
fi
echo

# --- Post-sync cargo check ---------------------------------------------------

if ! $DRY_RUN && ! $SKIP_CHECK; then
    info "Running cargo check in standalone repo (default features)..."
    if (cd "${STANDALONE_REPO}" && cargo check 2>&1 | tail -5); then
        ok "cargo check passed"
    else
        warn "cargo check failed — the sync may have issues"
        warn "Run with --skip-check to bypass, or investigate manually:"
        echo "  cd ${STANDALONE_REPO} && cargo check"
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
    printf "${YELLOW}Commit with message:${RESET} %s\n" "$COMMIT_MSG"
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
