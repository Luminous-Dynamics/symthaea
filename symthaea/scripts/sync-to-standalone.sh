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
    GIT_LFS_SKIP_SMUDGE=1 git clone "${STANDALONE_REMOTE}" "${STANDALONE_REPO}"
fi
echo

# --- Verify stubs exist in standalone ----------------------------------------

STUBS_OK=true
for stub in mycelix-fl-core mycelix-sdk mycelix-crypto positioning; do
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
    --exclude='broca-pipeline.bin'
    --exclude='*.bin'
    --exclude='checkpoints/'
    --exclude='android/demo/build/'
    --exclude='build/'
    # node_modules is gitignored in the monorepo parent (/srv/luminous-dynamics/.gitignore)
    # but not in symthaea/.gitignore — exclude here so every npm install doesn't push
    # ~52MB to the standalone public repo on each sync.
    --exclude='node_modules/'
    --exclude='.next/'
    --exclude='dist/'
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
    book
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
    # tla — excluded: 1.3GB of TLA+ model checker states, not needed for standalone CI
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
#   book/build/       - mdbook build output (source in book/src/ IS synced)
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
    ".gitleaks.toml"
    ".gitignore"
)

# Legal files from monorepo root (AGPL dual-license framework)
MONOREPO_LEGAL_FILES=(
    "COMMERCIAL_LICENSE.md"
    "CLA.md"
    "LICENSING_FAQ.md"
)

for f in "${INDIVIDUAL_FILES[@]}"; do
    if [ -f "${SYMTHAEA_DIR}/${f}" ]; then
        info "Copying ${f}"
        if ! $DRY_RUN; then
            cp "${SYMTHAEA_DIR}/${f}" "${STANDALONE_REPO}/${f}"
        fi
    fi
done

# Copy legal files from monorepo root (these don't live in symthaea/)
for f in "${MONOREPO_LEGAL_FILES[@]}"; do
    if [ -f "${MONOREPO_ROOT}/${f}" ]; then
        info "Copying ${f} (from monorepo root)"
        if ! $DRY_RUN; then
            cp "${MONOREPO_ROOT}/${f}" "${STANDALONE_REPO}/${f}"
        fi
    else
        warn "Missing ${f} in monorepo root"
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

    sed -i 's|^\(mycelix-crypto\s*=\s*{\s*path\s*=\s*\)"[^"]*"|\1"stubs/mycelix-crypto"|' \
        "${STANDALONE_REPO}/Cargo.toml"

    # Strip external deps that escape the workspace (prism, positioning)
    # and their feature flags. These are optional and not needed for CI.
    sed -i '/^prism-common\s*=.*path.*\.\.\//s/^/# [standalone-stripped] /' \
        "${STANDALONE_REPO}/Cargo.toml"
    sed -i '/^prism-search\s*=.*path.*\.\.\//s/^/# [standalone-stripped] /' \
        "${STANDALONE_REPO}/Cargo.toml"
    sed -i '/^plexus-common\s*=.*path.*\.\.\//s/^/# [standalone-stripped] /' \
        "${STANDALONE_REPO}/Cargo.toml"
    sed -i '/^plexus-search\s*=.*path.*\.\.\//s/^/# [standalone-stripped] /' \
        "${STANDALONE_REPO}/Cargo.toml"
    # Rewrite positioning to stub (required dep, not optional)
    sed -i 's|^\(positioning\s*=\s*{\s*path\s*=\s*\)"[^"]*"|\1"stubs/positioning"|' \
        "${STANDALONE_REPO}/Cargo.toml"
    # Strip feature flags that reference stripped deps
    sed -i '/^prism_search\s*=/s/^/# [standalone-stripped] /' \
        "${STANDALONE_REPO}/Cargo.toml"

    # Also strip from sub-crate Cargo.tomls (paths escaping workspace with ../../..)
    find "${STANDALONE_REPO}/crates" -name "Cargo.toml" -exec \
        sed -i '/path.*\.\.\/\.\.\/\.\./s/^/# [standalone-stripped] /' {} \;
    # Strip feature flags referencing stripped deps in sub-crates
    # Match both prism-search and prism_search (Cargo normalizes hyphens)
    find "${STANDALONE_REPO}/crates" -name "Cargo.toml" -exec \
        sed -i '/^prism[-_]search\s*=\s*\[/s/^/# [standalone-stripped] /' {} \;
    # Also strip positioning feature if it references the dep
    find "${STANDALONE_REPO}/crates" -name "Cargo.toml" -exec \
        sed -i '/^positioning\s*=\s*\[.*dep:positioning/s/^/# [standalone-stripped] /' {} \;

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
    if ! grep -q 'mycelix-crypto.*stubs/mycelix-crypto' "${STANDALONE_REPO}/Cargo.toml"; then
        warn "Failed to rewrite mycelix-crypto path"
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
    # Remove [lints] workspace = true from ALL Cargo.toml files in the repo
    while IFS= read -r subcrate_toml; do
        if grep -q '\[lints\]' "$subcrate_toml" 2>/dev/null; then
            sed -i '/^\[lints\]/{N;/workspace = true/d;}' "$subcrate_toml"
        fi
    done < <(find "${STANDALONE_REPO}" -name Cargo.toml -not -path '*/target/*' 2>/dev/null)
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

    # Remove monorepo-only dependencies that escape the standalone tree.
    # These are dev-dependencies that point to ../crates/ (monorepo root)
    # and don't have stubs in the standalone repo.
    sed -i '/^mycelix-bridge-common.*path.*"\.\./d' "$TOML"
    # Strip any remaining path deps that point outside the standalone tree
    sed -i '/path\s*=\s*"\.\.\//d' "$TOML"
    ok "Stripped monorepo-only path dependencies"

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

# --- Ensure broca defaults don't include GPU (requires nvcc/CUDA) -----------
BROCA_TOML="${STANDALONE_REPO}/crates/symthaea-broca/Cargo.toml"
if [ -f "$BROCA_TOML" ] && grep -q '^default.*"gpu"' "$BROCA_TOML"; then
    sed -i 's/^default = \["\(.*\)", "gpu"\]/default = ["\1"]/' "$BROCA_TOML"
    sed -i 's/^default = \["gpu", \(.*\)\]/default = [\1]/' "$BROCA_TOML"
    # Handle middle position
    sed -i 's/, "gpu"//' "$BROCA_TOML"
    ok "Removed 'gpu' from symthaea-broca defaults (CUDA not available in CI)"
fi

echo

# --- Post-sync cargo check ---------------------------------------------------

if ! $DRY_RUN && ! $SKIP_CHECK; then
    SYNC_CHECK_FAILED=false

    info "Running cargo fmt --check in standalone repo..."
    if (cd "${STANDALONE_REPO}" && cargo fmt --check 2>&1 | head -20); then
        ok "cargo fmt passed"
    else
        warn "cargo fmt --check failed — unformatted code detected"
        SYNC_CHECK_FAILED=true
    fi

    info "Running cargo check in standalone repo (default features)..."
    if (cd "${STANDALONE_REPO}" && cargo check 2>&1 | tail -5); then
        ok "cargo check passed"
    else
        warn "cargo check failed — the sync may have issues"
        SYNC_CHECK_FAILED=true
    fi

    # Run the same CI feature set to catch compilation errors before pushing.
    # This mirrors the clippy job's CI_FEATURES in ci.yml.
    info "Running cargo check --all CI features (catches cross-feature compilation errors)..."
    CI_FEATURES="parallel,service,shell,demo,api_module,\
voice-tts,voice-stt,audio,vocal-tract,neural-vocoder,\
embeddings,vision,perception,vision-manifold,foveation,\
integrity,semantic-encoder,neural-bridge,webcam,\
mesh,mesh-encryption,mesh-key-exchange,swarm,notifications,\
nix-mind,identity,physics,physics-bridge,\
flight,humanoid,hal,ssm-power,ssm_language,\
lancedb-backend,multi_agent,full_consciousness,full_perception,\
full_language,magi_loop,reasoning_engine,code_generation,\
wasm-sandbox,school_learning,benchmarks,all_benchmarks,\
integration_module,observability_module,support,web_research_module,\
genomics,cell-foundry,ectogenesis,nurture,population,genesis,\
genesis-missions,fusion-twin,safety-agents,lab-controller,\
materials,nuclear-forensics,water-prediction,physics-unification,\
grid-scaling,fission-reactor,accelerator,threat-assessment,\
datacenter,experiment-planner,strategic-materials,critical-minerals,\
advanced-manufacturing,building-systems,design-production,\
mycelix,unstable-examples"
    if (cd "${STANDALONE_REPO}" && cargo check -p symthaea --features "$CI_FEATURES" 2>&1 | tail -10); then
        ok "cargo check (CI features) passed"
    else
        warn "cargo check (CI features) failed — compilation errors will break CI"
        SYNC_CHECK_FAILED=true
    fi

    if $SYNC_CHECK_FAILED; then
        warn "Pre-push checks failed. Run with --skip-check to bypass, or investigate:"
        echo "  cd ${STANDALONE_REPO} && cargo fmt --check"
        echo "  cd ${STANDALONE_REPO} && cargo check"
    fi
    echo
elif $SKIP_CHECK; then
    info "Skipping cargo check (--skip-check)"
    echo
fi

# --- Post-sync license fixups -------------------------------------------------
# All crates are now AGPL-3.0-or-later (unified March 2026).
# No per-crate license overrides needed.

# --- Format with CI toolchain -------------------------------------------------
# CI uses dtolnay/rust-toolchain@1.93.0, so format with the same version
# to avoid spurious diffs from different rustfmt versions.

if command -v rustup >/dev/null 2>&1 && rustup run 1.93.0 rustfmt --version >/dev/null 2>&1; then
    info "Formatting with rustfmt 1.93.0 (CI toolchain)..."
    (cd "${STANDALONE_REPO}" && rustup run 1.93.0 cargo fmt 2>/dev/null) && ok "Formatted" || warn "cargo fmt failed (non-fatal)"
else
    warn "rustup 1.93.0 not available — skipping CI-toolchain format"
fi
echo

# --- Show diff summary --------------------------------------------------------

info "Changes in standalone repo:"
echo
# Stage all changes EXCEPT LFS-tracked binaries (avoid clean filter failures on /tmp)
git -C "${STANDALONE_REPO}" add -A -- ':!*.bin'
# Re-add .gitattributes in case it changed (LFS tracking config)
git -C "${STANDALONE_REPO}" add .gitattributes 2>/dev/null || true
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
