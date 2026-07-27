#!/usr/bin/env bash
# Export to Standalone — publish an explicit, clean commit of the symthaea/
# crate to Luminous-Dynamics/symthaea via a reviewable PR.
#
# Replaces sync-to-standalone.sh (disabled 2026-07-26, see that file's guard
# comment and https://github.com/Luminous-Dynamics/luminous-dynamics/issues/25).
# The old script ran `rsync -a --delete` against the monorepo's LIVE working
# tree, then labeled the resulting commit with whatever HEAD SHA happened to
# be current -- so a sync initiated while any concurrent session (this
# monorepo routinely runs 10+) had uncommitted or differently-branched work
# sitting in the shared tree could publish that content to the public repo's
# main, with no real link to the commit it claimed to be from. Confirmed:
# standalone main commit 997746177b ("sync: update from monorepo @
# 431d2cafdf") does not match monorepo commit 431d2cafdf's actual content for
# at least 8 files, and ships a broken Cargo.toml (a `dep:symthaea` feature
# reference the sync's own path-stripping rule commented out) that fails
# `cargo metadata` for the entire workspace -- something the cited commit
# never had.
#
# This script instead:
#   - requires an explicit source commit (defaults to HEAD, but refuses to
#     run if the working tree is dirty unless a commit is given explicitly --
#     never silently exports uncommitted state)
#   - exports via `git archive` from that exact commit object -- never reads
#     the live working tree, so concurrent sessions' uncommitted edits
#     cannot leak into the export regardless of what's sitting on disk
#   - applies the same narrowly-scoped Cargo.toml/path transformations as
#     the old script, kept verbatim where unchanged (they're independent of
#     the provenance bug and still correct)
#   - adds flake.nix/flake.lock to the copied file set (disclosed gap, issue
#     #25 comment 3: standalone has never had a flake.nix in 71 commits,
#     so "Hardened Nix Regressions" can never have passed there)
#   - pushes to a NEW branch and opens a PR against standalone main -- never
#     pushes main directly, so a bad export is a closeable PR, not a
#     public-main incident
#   - records the exact source commit and a content-addressed transformed-
#     tree hash in the PR body, so provenance is verifiable after the fact
#
# Usage:
#   bash symthaea/scripts/export-to-standalone.sh [<commit-ish>] [options]
#
# Arguments:
#   <commit-ish>  Defaults to HEAD. If omitted and the working tree is dirty,
#                 the script refuses to run -- commit first, or pass an
#                 explicit ref/SHA to export regardless of working-tree state.
#
# Options:
#   --dry-run              Export and show the diff without pushing or opening a PR
#   --skip-check            Skip the post-export cargo check verification entirely
#   --allow-check-failure   Open the PR even if post-export verification fails
#                           (default: a failed check aborts before push, same
#                           policy as the old script -- a broken PR is still
#                           better than a broken main, but shouldn't be silent)

set -euo pipefail

# --- Configuration -----------------------------------------------------------

MONOREPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SYMTHAEA_SUBDIR="symthaea"
STANDALONE_REPO="/tmp/symthaea-standalone-export"
STANDALONE_REMOTE="git@github.com:Luminous-Dynamics/symthaea.git"
LOCK_FILE="/tmp/symthaea-standalone-export.lock"

DRY_RUN=false
SKIP_CHECK=false
ALLOW_CHECK_FAILURE=false
COMMIT_ISH=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)              DRY_RUN=true ;;
        --skip-check)           SKIP_CHECK=true ;;
        --allow-check-failure)  ALLOW_CHECK_FAILURE=true ;;
        -*)                     echo "Unknown option: $arg" >&2; exit 1 ;;
        *)                      COMMIT_ISH="$arg" ;;
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

cd "${MONOREPO_ROOT}"

# --- Resolve and validate the source commit -----------------------------------
#
# The entire point of this script: pin an exact commit object up front, and
# never touch the live working tree again after this point.

if [ -z "$COMMIT_ISH" ]; then
    if [ -n "$(git status --porcelain -- symthaea)" ]; then
        error "Working tree has uncommitted changes under symthaea/. Commit first, or pass an explicit commit-ish to export regardless (e.g. 'HEAD', a tag, or a SHA) if you understand you may be exporting content not on that ref."
    fi
    COMMIT_ISH="HEAD"
fi

SOURCE_SHA="$(git rev-parse --verify "${COMMIT_ISH}^{commit}" 2>/dev/null)" \
    || error "Could not resolve '${COMMIT_ISH}' to a commit object"
SOURCE_SHA_SHORT="$(git rev-parse --short "${SOURCE_SHA}")"

info "Monorepo root:  ${MONOREPO_ROOT}"
info "Source commit:  ${SOURCE_SHA} (${SOURCE_SHA_SHORT})"
info "Standalone:     ${STANDALONE_REPO}"
if $DRY_RUN; then
    warn "DRY RUN — no branch push or PR will be created"
fi
echo

# --- Acquire lock --------------------------------------------------------------

exec 200>"${LOCK_FILE}"
info "Acquiring export lock (${LOCK_FILE})..."
if ! flock -w 600 200; then
    error "Could not acquire ${LOCK_FILE} within 10 minutes -- another export-to-standalone.sh is likely running. Wait for it to finish, or if you're sure none is running, 'rm -f ${LOCK_FILE}' and retry."
fi
ok "Lock acquired"
echo

# --- Export a clean tree of symthaea/ at the pinned commit --------------------
#
# git archive reads only committed objects -- it is structurally incapable of
# picking up uncommitted working-tree state, which is the entire class of bug
# this script exists to close off.

EXPORT_TMPDIR="$(mktemp -d /tmp/symthaea-export-XXXXXX)"
trap 'rm -rf "${EXPORT_TMPDIR}"' EXIT

info "Exporting ${SYMTHAEA_SUBDIR}/ from ${SOURCE_SHA_SHORT} via git archive..."
git archive --format=tar "${SOURCE_SHA}" -- "${SYMTHAEA_SUBDIR}" | tar -x -C "${EXPORT_TMPDIR}"
if [ ! -d "${EXPORT_TMPDIR}/${SYMTHAEA_SUBDIR}" ]; then
    error "git archive produced no ${SYMTHAEA_SUBDIR}/ directory -- does that path exist at ${SOURCE_SHA_SHORT}?"
fi

# Also export the monorepo-root legal files this crate's Cargo.toml/license
# framework depends on (same set the old script copies from monorepo root).
MONOREPO_LEGAL_FILES=(
    "COMMERCIAL_LICENSE.md"
    "CLA.md"
    "LICENSING_FAQ.md"
)
mkdir -p "${EXPORT_TMPDIR}/_root"
for f in "${MONOREPO_LEGAL_FILES[@]}"; do
    git show "${SOURCE_SHA}:${f}" > "${EXPORT_TMPDIR}/_root/${f}" 2>/dev/null \
        && ok "Exported ${f} (from monorepo root @ ${SOURCE_SHA_SHORT})" \
        || warn "Missing ${f} in monorepo root at ${SOURCE_SHA_SHORT}"
done

SYMTHAEA_DIR="${EXPORT_TMPDIR}/${SYMTHAEA_SUBDIR}"
MONOREPO_ROOT_EXPORT="${EXPORT_TMPDIR}/_root"
ok "Clean export ready at ${SYMTHAEA_DIR} (source: ${SOURCE_SHA})"
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

# --- Copy directories from the clean export (never the live working tree) ----
#
# Same directory selection as the old script -- this list is independent of
# the provenance bug and still correct. The only change is the source: a
# clean git-archive export of a pinned commit, not live disk state.

SOURCE_DIRS=(
    src
    crates
    symthaea-core
    tests
    examples
    benches
    xtask
    vendor
)

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
    nix
    proptest-regressions
    rust-sentinels
    static
    systemd
    tools
    validation
)

RSYNC_EXCLUDE=(
    --exclude='target/'
    --exclude='__pyphi_cache__/'
    --exclude='.DS_Store'
    --exclude='broca-pipeline.bin'
    --exclude='*.bin'
    --exclude='checkpoints/'
    --exclude='android/demo/build/'
    --exclude='build/'
    --exclude='node_modules/'
    --exclude='.next/'
    --exclude='dist/'
)
# NOTE: --dry-run does NOT skip writing to STANDALONE_REPO. That directory is
# a disposable /tmp scratch clone, reset from origin/main every run -- writing
# to it (and running cargo checks against it) carries zero risk to the real
# public repo. The only step --dry-run actually gates is the final
# branch/commit/push/PR at the bottom of this script. This lets --dry-run
# mean what you actually want it to mean: "verify everything, publish nothing."
RSYNC_OPTS=(-a --delete -v "${RSYNC_EXCLUDE[@]}")

sync_dir() {
    local dir="$1"
    if [ ! -d "${SYMTHAEA_DIR}/${dir}" ]; then
        warn "Skipping ${dir}/ (not found in exported tree)"
        return
    fi
    info "Copying ${dir}/"
    rsync "${RSYNC_OPTS[@]}" \
        "${SYMTHAEA_DIR}/${dir}/" \
        "${STANDALONE_REPO}/${dir}/"
}

info "=== Copying source directories ==="
for dir in "${SOURCE_DIRS[@]}"; do
    sync_dir "$dir"
done

echo
info "=== Copying referenced patches/ subdirectories ==="
PATCH_SUBDIRS=(
    "ed25519-dalek/curve25519-dalek"
    "ed25519-dalek/curve25519-dalek-derive"
    "ed25519-dalek/ed25519-dalek"
    "iroh/iroh"
    "iroh/iroh-base"
    "iroh/iroh-relay"
)
for subdir in "${PATCH_SUBDIRS[@]}"; do
    if [ ! -d "${SYMTHAEA_DIR}/patches/${subdir}" ]; then
        warn "Skipping patches/${subdir}/ (not found in exported tree)"
        continue
    fi
    info "Copying patches/${subdir}/"
    mkdir -p "$(dirname "${STANDALONE_REPO}/patches/${subdir}")"
    rsync "${RSYNC_OPTS[@]}" --exclude='.git/' \
        "${SYMTHAEA_DIR}/patches/${subdir}/" \
        "${STANDALONE_REPO}/patches/${subdir}/"
done

echo
info "=== Copying support directories ==="
for dir in "${SUPPORT_DIRS[@]}"; do
    sync_dir "$dir"
done
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
    # Disclosed gap (GitHub issue #25 comment 3): standalone has never had a
    # flake.nix in its entire 71-commit history because the old script's
    # INDIVIDUAL_FILES list never included it. "Hardened Nix Regressions"
    # can never have passed on standalone -- a day-one gap, not a regression.
    "flake.nix"
    "flake.lock"
)

for f in "${INDIVIDUAL_FILES[@]}"; do
    if [ -f "${SYMTHAEA_DIR}/${f}" ]; then
        info "Copying ${f}"
        cp "${SYMTHAEA_DIR}/${f}" "${STANDALONE_REPO}/${f}"
    fi
done

for f in "${MONOREPO_LEGAL_FILES[@]}"; do
    if [ -f "${MONOREPO_ROOT_EXPORT}/${f}" ]; then
        info "Copying ${f} (from monorepo root)"
        cp "${MONOREPO_ROOT_EXPORT}/${f}" "${STANDALONE_REPO}/${f}"
    else
        warn "Missing ${f} in monorepo root"
    fi
done
echo

# --- Cargo.toml with standalone fixups ----------------------------------------
# Unchanged from the old script -- these transformations are independent of
# the provenance bug (they're generic path/feature rewrites for standalone
# builds, not related to what source content gets fed in).

info "Copying Cargo.toml with standalone fixups..."

cp "${SYMTHAEA_DIR}/Cargo.toml" "${STANDALONE_REPO}/Cargo.toml"

    sed -i 's|^\(mycelix-fl-core\s*=\s*{\s*path\s*=\s*\)"[^"]*"|\1"stubs/mycelix-fl-core"|' \
        "${STANDALONE_REPO}/Cargo.toml"
    sed -i 's|^\(mycelix-sdk\s*=\s*{\s*path\s*=\s*\)"[^"]*"|\1"stubs/mycelix-sdk"|' \
        "${STANDALONE_REPO}/Cargo.toml"
    sed -i 's|^\(mycelix-crypto\s*=\s*{\s*path\s*=\s*\)"[^"]*"|\1"stubs/mycelix-crypto"|' \
        "${STANDALONE_REPO}/Cargo.toml"

    # mycelix-zkp-core and symtropy-robotics-bridge-core: stub-path substitution
    # for [workspace.dependencies]. Originally assumed these keys were only
    # ever referenced via `workspace = true` inheritance and never directly
    # declared with a real path, so a blind `sed -i '/[workspace.dependencies]/a ...'`
    # insert was safe. That assumption drifted: the monorepo root Cargo.toml
    # now ALSO declares them directly with real external paths (e.g.
    # `mycelix-zkp-core = { path = "../mycelix-workspace/...", features =
    # ["dilithium"] }`), so the blind insert produced a second entry for the
    # same key -- a duplicate-key TOML parse error that broke `cargo
    # metadata` for the whole workspace (found verifying this script).
    # Fix: rewrite the existing line's path if the key is already present;
    # only fall back to inserting a new line if it's genuinely absent.
    for wsdep in mycelix-zkp-core symtropy-robotics-bridge-core; do
        if grep -qE "^${wsdep}\s*=\s*\{" "${STANDALONE_REPO}/Cargo.toml"; then
            sed -i "s|^\(${wsdep}\s*=\s*{[^}]*path\s*=\s*\)\"[^\"]*\"|\1\"stubs/${wsdep}\"|" \
                "${STANDALONE_REPO}/Cargo.toml"
        else
            sed -i "/^\[workspace\.dependencies\]/a ${wsdep} = { path = \"stubs/${wsdep}\" }" \
                "${STANDALONE_REPO}/Cargo.toml"
        fi
        if ! grep -qE "^${wsdep}\s*=\s*\{[^}]*\"stubs/${wsdep}\"" "${STANDALONE_REPO}/Cargo.toml"; then
            warn "Failed to point ${wsdep} at its stub — check [workspace.dependencies]"
        fi
    done

    find "${STANDALONE_REPO}/crates" -name "Cargo.toml" -exec \
        sed -i 's|^symtropy-robotics-bridge-core\s*=\s*{\s*path\s*=\s*"[^"]*"\s*}|symtropy-robotics-bridge-core = { workspace = true }|' {} \;

    declare -A symtropy_dep_versions=(
        [symtropy-math]="0.2.1"
        [symtropy-physics]="0.2.1"
        [symtropy-consciousness-physics]="0.2.0"
    )
    for symtropy_dep in "${!symtropy_dep_versions[@]}"; do
        dep_version="${symtropy_dep_versions[$symtropy_dep]}"
        find "${STANDALONE_REPO}/crates" -name "Cargo.toml" -exec \
            sed -i "s|^${symtropy_dep}\\s*=\\s*{\\s*path\\s*=\\s*\"[^\"]*\"\\s*}|${symtropy_dep} = \"${dep_version}\"|" {} \;
        find "${STANDALONE_REPO}/crates" -name "Cargo.toml" -exec \
            sed -i "s|^path = \"\\.\\./\\.\\./\\.\\./\\.\\./symtropy/crates/[a-z]*/${symtropy_dep}\"|version = \"${dep_version}\"|" {} \;
    done

    sed -i '/^prism-common\s*=.*path.*\.\.\//s/^/# [standalone-stripped] /' \
        "${STANDALONE_REPO}/Cargo.toml"
    sed -i '/^prism-search\s*=.*path.*\.\.\//s/^/# [standalone-stripped] /' \
        "${STANDALONE_REPO}/Cargo.toml"
    sed -i 's|^\(positioning\s*=\s*{\s*path\s*=\s*\)"[^"]*"|\1"stubs/positioning"|' \
        "${STANDALONE_REPO}/Cargo.toml"
    sed -i 's|^prism_search\s*=.*|prism_search = []  # [standalone-neutered] deps stripped, kept declared for check-cfg|' \
        "${STANDALONE_REPO}/Cargo.toml"

    # --- Strip genuinely-external path deps, by RESOLUTION not dot-counting --
    #
    # The old script's rule here was `/path.*\.\.\/\.\.\/\.\./` -- a plain
    # substring match for three "../" segments. That's indistinguishable from
    # a legitimate workspace-internal self-reference: symthaea-psych-bench
    # and symthaea-pulse both live at crates/domains/<name>/ and declare
    # `symthaea = { path = "../../.." }` to depend on the crate root itself
    # (exactly 3 levels up) -- the same three-dot substring a 4-level escape
    # like `../../../../mycelix-workspace/sdk` also contains. The old rule
    # stripped both alike, which is the exact bug that broke standalone
    # main's `cargo metadata` for the whole workspace (issue #25): the
    # feature `symthaea-backend = [..., "dep:symthaea"]` survived, but the
    # `symthaea` dependency it requires got commented out from under it.
    #
    # Fix: resolve each path= value relative to its own file's location and
    # check whether it lands inside or outside STANDALONE_REPO. Depth-of-
    # nesting varies across sub-crates (crates/<name>/ vs
    # crates/<tier>/<name>/), so a fixed dot-count can never be correct here
    # -- only actual path resolution can.
    strip_external_path_deps() {
        local toml_file="$1"
        local crate_dir
        crate_dir="$(dirname "$toml_file")"
        local line_no rel_path resolved
        while IFS=: read -r line_no line_content; do
            [[ "$line_content" =~ ^[[:space:]]*# ]] && continue
            rel_path="$(echo "$line_content" | sed -n 's/.*path[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p')"
            [ -z "$rel_path" ] && continue
            resolved="$(realpath -m "${crate_dir}/${rel_path}")"
            case "$resolved" in
                "${STANDALONE_REPO}"/*|"${STANDALONE_REPO}")
                    : # resolves inside the standalone tree -- internal, leave alone
                    ;;
                *)
                    sed -i "${line_no}s/^/# [standalone-stripped] /" "$toml_file"
                    warn "Stripped external path dep (${toml_file#${STANDALONE_REPO}/}:${line_no}): ${rel_path} -> ${resolved}"
                    ;;
            esac
        done < <(grep -n 'path[[:space:]]*=[[:space:]]*"' "$toml_file")
    }
    while IFS= read -r crate_toml; do
        strip_external_path_deps "$crate_toml"
    done < <(find "${STANDALONE_REPO}/crates" -name "Cargo.toml")

    find "${STANDALONE_REPO}/crates" -name "Cargo.toml" -exec \
        sed -i '/^prism[-_]search\s*=\s*\[/s/^/# [standalone-stripped] /' {} \;
    find "${STANDALONE_REPO}/crates" -name "Cargo.toml" -exec \
        sed -i '/^positioning\s*=\s*\[.*dep:positioning/s/^/# [standalone-stripped] /' {} \;

    # --- Fix the exact bug that broke `cargo metadata` workspace-wide -------
    #
    # symthaea-psych-bench's Cargo.toml declares a `symthaea-backend` feature
    # requiring `dep:symthaea`. If a generic external-path-stripping rule
    # above ever removes the `symthaea` path dependency from a sub-crate
    # while leaving a feature that references `dep:symthaea`, cargo refuses
    # to parse the manifest at all (invalid feature/dependency combination) --
    # this is exactly what broke standalone main's `cargo metadata` for an
    # unknown period (GitHub issue #25). `symthaea` is a workspace-internal
    # path (`../../..`, i.e. this crate itself), not an external escape, so
    # the generic `../../..`-stripping rule above must not touch it. Verify
    # that invariant explicitly rather than relying on the sed patterns above
    # never matching it by accident.
    PSYCH_BENCH_TOML="${STANDALONE_REPO}/crates/domains/symthaea-psych-bench/Cargo.toml"
    if [ -f "$PSYCH_BENCH_TOML" ]; then
        if grep -q 'dep:symthaea' "$PSYCH_BENCH_TOML" && \
           ! grep -qE '^symthaea\s*=\s*\{' "$PSYCH_BENCH_TOML"; then
            error "symthaea-psych-bench/Cargo.toml references dep:symthaea but the symthaea dependency line was stripped or is missing -- this is the exact bug from issue #25. Aborting before push."
        fi
    fi

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

    ESCAPED_PATHS=$(grep -n 'path\s*=\s*"[^"]*\.\./\.\.' "${STANDALONE_REPO}/Cargo.toml" | \
                    grep -v '^#' | \
                    grep -v 'stubs/' || true)
    if [ -n "$ESCAPED_PATHS" ]; then
        warn "Cargo.toml still has external path deps:"
        echo "$ESCAPED_PATHS"
        REWRITE_OK=false
    fi

    sed -i '/^\[workspace\.lints\.clippy\]/,/^$/d' "${STANDALONE_REPO}/Cargo.toml"
    sed -i '/^# Workspace-wide lint configuration/d' "${STANDALONE_REPO}/Cargo.toml"
    sed -i '/^# These lints apply on top/d' "${STANDALONE_REPO}/Cargo.toml"
    sed -i '/^\[lints\]/{N;/workspace = true/d;}' "${STANDALONE_REPO}/Cargo.toml"
    if [ -f "${STANDALONE_REPO}/symthaea-core/Cargo.toml" ]; then
        sed -i '/^\[lints\]/{N;/workspace = true/d;}' "${STANDALONE_REPO}/symthaea-core/Cargo.toml"
    fi
    while IFS= read -r subcrate_toml; do
        if grep -q '\[lints\]' "$subcrate_toml" 2>/dev/null; then
            sed -i '/^\[lints\]/{N;/workspace = true/d;}' "$subcrate_toml"
        fi
    done < <(find "${STANDALONE_REPO}" -name Cargo.toml -not -path '*/target/*' 2>/dev/null)
    ok "Stripped workspace lints (incompatible with CI clippy config)"

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

    sed -i '/^mycelix-bridge-common.*path.*"\.\./d' "$TOML"
    sed -i '/path\s*=\s*"\.\.\/\//d' "$TOML"
    ok "Stripped monorepo-only path dependencies"

if $REWRITE_OK; then
    ok "Cargo.toml fixups verified"
else
    error "Cargo.toml rewrite verification failed — check the sed patterns"
fi
echo

# --- Scan sub-crate Cargo.tomls for escaping paths ---------------------------

info "Scanning sub-crate Cargo.tomls for external path deps still standing (resolution-based, not dot-counting -- see strip_external_path_deps above)..."
SUBCRATE_ISSUES=false
while IFS= read -r toml_file; do
    rel_path="${toml_file#${STANDALONE_REPO}/}"
    if [ "$rel_path" = "Cargo.toml" ] || [[ "$rel_path" == stubs/* ]]; then
        continue
    fi
    crate_dir="$(dirname "$toml_file")"
    while IFS=: read -r line_no line_content; do
        [[ "$line_content" =~ ^[[:space:]]*# ]] && continue
        found_rel_path="$(echo "$line_content" | sed -n 's/.*path[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p')"
        [ -z "$found_rel_path" ] && continue
        resolved="$(realpath -m "${crate_dir}/${found_rel_path}")"
        case "$resolved" in
            "${STANDALONE_REPO}"/*|"${STANDALONE_REPO}")
                ;;
            *)
                warn "External path dep still standing in ${rel_path}:${line_no}: ${found_rel_path} -> ${resolved}"
                SUBCRATE_ISSUES=true
                ;;
        esac
    done < <(grep -n 'path[[:space:]]*=[[:space:]]*"' "$toml_file" 2>/dev/null)
done < <(find "${STANDALONE_REPO}" -name "Cargo.toml" -not -path "*/target/*" 2>/dev/null)

if $SUBCRATE_ISSUES; then
    warn "Some sub-crates have path deps genuinely escaping the workspace tree (uncaught by the strip pass above -- investigate before pushing)"
else
    ok "All sub-crate Cargo.tomls OK"
fi

# --- Ensure broca defaults don't include GPU (requires nvcc/CUDA) -----------
BROCA_TOML="${STANDALONE_REPO}/crates/symthaea-broca/Cargo.toml"
if [ -f "$BROCA_TOML" ] && grep -q '^default.*"gpu"' "$BROCA_TOML"; then
    sed -i 's/^default = \["\(.*\)", "gpu"\]/default = ["\1"]/' "$BROCA_TOML"
    sed -i 's/^default = \["gpu", \(.*\)\]/default = [\1]/' "$BROCA_TOML"
    sed -i 's/, "gpu"//' "$BROCA_TOML"
    ok "Removed 'gpu' from symthaea-broca defaults (CUDA not available in CI)"
fi
echo

# --- Post-export cargo check ---------------------------------------------------

SYNC_CHECK_FAILED=false
if ! $SKIP_CHECK; then
    # set -o pipefail is required here: without it, `cargo check 2>&1 | tail
    # -40` reports tail's exit code, not cargo's -- since tail almost always
    # exits 0, every piped check below was structurally unable to fail
    # (found while investigating an unrelated false-positive: real cargo
    # failures would have been silently swallowed here since this function
    # was written).
    # CARGO_BUILD_JOBS capped per .claude/rules/CONCURRENT_SESSIONS.md --
    # this monorepo routinely runs many concurrent sessions; don't compete
    # for every core on a shared box.
    run_standalone_cargo() {
        nix develop "${MONOREPO_ROOT}" --command bash -c \
            "set -o pipefail; export CARGO_BUILD_JOBS=2; cd '${STANDALONE_REPO}' && $1"
    }

    info "Running cargo metadata sanity check first (this is the exact failure mode issue #25 found)..."
    if run_standalone_cargo "cargo metadata --no-deps --format-version=1 >/dev/null"; then
        ok "cargo metadata parses cleanly"
    else
        warn "cargo metadata failed to parse the workspace — this is a hard blocker, not a soft check"
        SYNC_CHECK_FAILED=true
    fi

    info "Running cargo fmt --check in standalone repo..."
    if run_standalone_cargo "cargo fmt --check 2>&1 | head -20"; then
        ok "cargo fmt passed"
    else
        warn "cargo fmt --check failed — unformatted code detected"
        SYNC_CHECK_FAILED=true
    fi

    info "Running cargo check in standalone repo (default features)..."
    if run_standalone_cargo "cargo check 2>&1 | tail -40"; then
        ok "cargo check passed"
    else
        warn "cargo check failed — the export may have issues"
        SYNC_CHECK_FAILED=true
    fi

    info "Running cargo check --all CI features (catches cross-feature compilation errors)..."
    # vision / perception / full_perception REMOVED here: the byte-hash
    # perception stratum was deleted and symthaea-perception archived
    # (commit 219c1c5c7c, "delete the byte-hash perception stratum; archive
    # symthaea-perception (P2.2/P2.3)") -- these three feature names no
    # longer exist in Cargo.toml. Found via a real `cargo check` failure
    # ("the package 'symthaea' does not contain these features") verifying
    # this script, not by inspection -- this list drifts with Cargo.toml and
    # isn't otherwise tested anywhere. vision-manifold/foveation (the
    # replacement/surviving vision-adjacent features) were already present
    # below and still are.
    CI_FEATURES="parallel,service,shell,demo,api_module,\
voice-tts,voice-stt,audio,vocal-tract,neural-vocoder,\
embeddings,vision-manifold,foveation,\
integrity,semantic-encoder,neural-bridge,webcam,\
mesh,mesh-encryption,mesh-key-exchange,swarm,notifications,\
nixward,identity,physics,physics-bridge,\
flight,humanoid,hal,ssm-power,ssm_language,\
lancedb-backend,multi_agent,full_consciousness,\
full_language,magi_loop,reasoning_engine,code_generation,\
wasm-sandbox,school_learning,benchmarks,all_benchmarks,\
integration_module,observability_module,support,web_research_module,\
genomics,cell-foundry,ectogenesis,nurture,population,genesis,\
fusion-twin,safety-agents,\
materials,nuclear-forensics,water-prediction,\
grid-scaling,fission-reactor,accelerator,threat-assessment,\
datacenter,experiment-planner,strategic-materials,critical-minerals,\
advanced-manufacturing,\
mycelix,unstable-examples"
    if run_standalone_cargo "cargo check -p symthaea --features '$CI_FEATURES' 2>&1 | tail -40"; then
        ok "cargo check (CI features) passed"
    else
        warn "cargo check (CI features) failed — compilation errors will break CI"
        SYNC_CHECK_FAILED=true
    fi

    info "Running cargo check on crates/hdc-zkp-bench (separate workspace)..."
    if run_standalone_cargo "cargo check --manifest-path crates/hdc-zkp-bench/Cargo.toml 2>&1 | tail -40"; then
        ok "cargo check (hdc-zkp-bench) passed"
    else
        warn "cargo check (hdc-zkp-bench) failed"
        SYNC_CHECK_FAILED=true
    fi

    if $SYNC_CHECK_FAILED; then
        warn "Pre-push checks failed:"
        echo "  cd ${STANDALONE_REPO} && nix develop ${MONOREPO_ROOT} --command cargo metadata --no-deps"
        echo "  cd ${STANDALONE_REPO} && nix develop ${MONOREPO_ROOT} --command cargo fmt --check"
        echo "  cd ${STANDALONE_REPO} && nix develop ${MONOREPO_ROOT} --command cargo check"
        echo "  cd ${STANDALONE_REPO} && nix develop ${MONOREPO_ROOT} --command cargo check --manifest-path crates/hdc-zkp-bench/Cargo.toml"
        if $ALLOW_CHECK_FAILURE; then
            warn "--allow-check-failure set — proceeding to open a PR anyway"
        else
            git -C "${STANDALONE_REPO}" reset HEAD -- . >/dev/null 2>&1
            error "Aborting before branch/PR. Fix the above or re-run with --allow-check-failure to open a PR anyway (a PR, not main, so this is reviewable rather than a public-main incident)."
        fi
    fi
    echo
elif $SKIP_CHECK; then
    info "Skipping cargo check (--skip-check)"
    echo
fi

# --- Format with CI toolchain -------------------------------------------------
#
# CI_TOOLCHAIN is read from rust-toolchain.toml, not hardcoded, because a
# hardcoded version drifts silently and actively HURTS here: this host has
# multiple rustup toolchains installed (including a stale 1.93.0 this step
# used to hardcode, and the correct 1.96.0 rust-toolchain.toml/ci.yml
# actually pin). Reformatting with the WRONG version doesn't just fail to
# help -- it can positively corrupt formatting that already matched the
# correct version, actively causing the `cargo fmt --check` failure this
# step exists to prevent (found 2026-07-27 verifying PR #31: Format Check
# failed in real CI after this step "helpfully" reformatted with 1.93.0).
CI_TOOLCHAIN="$(grep -oP '^channel\s*=\s*"\K[^"]+' "${SYMTHAEA_DIR}/rust-toolchain.toml" 2>/dev/null || true)"
if [ -z "$CI_TOOLCHAIN" ]; then
    warn "Could not read channel from rust-toolchain.toml — skipping CI-toolchain format"
elif command -v rustup >/dev/null 2>&1 && rustup run "$CI_TOOLCHAIN" rustfmt --version >/dev/null 2>&1; then
    info "Formatting with rustfmt ${CI_TOOLCHAIN} (matches rust-toolchain.toml / ci.yml's actual pin)..."
    (cd "${STANDALONE_REPO}" && rustup run "$CI_TOOLCHAIN" cargo fmt 2>/dev/null) && ok "Formatted" || warn "cargo fmt failed (non-fatal)"
else
    warn "rustup toolchain ${CI_TOOLCHAIN} not available — skipping CI-toolchain format"
fi
echo

# --- Show diff summary --------------------------------------------------------

info "Changes in standalone repo:"
echo
git -C "${STANDALONE_REPO}" add -A -- ':!*.bin'
git -C "${STANDALONE_REPO}" add .gitattributes 2>/dev/null || true
git -C "${STANDALONE_REPO}" diff --cached --stat
echo

CHANGES=$(git -C "${STANDALONE_REPO}" diff --cached --name-only | wc -l)
if [ "$CHANGES" -eq 0 ]; then
    info "No changes to export — standalone already matches ${SOURCE_SHA_SHORT}."
    exit 0
fi

printf "${BOLD}%s files changed${RESET}\n" "$CHANGES"
echo

if $DRY_RUN; then
    warn "DRY RUN — skipping branch, commit, and PR"
    git -C "${STANDALONE_REPO}" reset HEAD -- . >/dev/null 2>&1
    exit 0
fi

# --- Content-addressed provenance ---------------------------------------------
#
# git write-tree hashes the currently-staged index -- a content-addressed
# fingerprint of exactly what's about to be published, independent of any
# commit message claim. Recorded in the PR body so provenance is verifiable
# after the fact without trusting the commit message alone.

TRANSFORMED_TREE_SHA="$(git -C "${STANDALONE_REPO}" write-tree)"
ok "Transformed tree hash: ${TRANSFORMED_TREE_SHA}"

# --- Branch, commit, push, open PR --------------------------------------------

EXPORT_BRANCH="export/${SOURCE_SHA_SHORT}-$(date +%Y%m%d-%H%M%S)"
COMMIT_MSG="export: from monorepo commit ${SOURCE_SHA}

Source commit:          ${SOURCE_SHA}
Transformed tree hash:  ${TRANSFORMED_TREE_SHA}

Exported via git archive from the exact pinned commit above -- not from a
live working tree. See export-to-standalone.sh and
https://github.com/Luminous-Dynamics/luminous-dynamics/issues/25 for why
this matters."

info "Creating branch ${EXPORT_BRANCH}..."
git -C "${STANDALONE_REPO}" checkout -b "${EXPORT_BRANCH}"
git -C "${STANDALONE_REPO}" commit -m "${COMMIT_MSG}"
ok "Committed on ${EXPORT_BRANCH}"

info "Pushing ${EXPORT_BRANCH}..."
git -C "${STANDALONE_REPO}" push origin "${EXPORT_BRANCH}"
ok "Pushed"

PR_BODY="Automated export from monorepo commit \`${SOURCE_SHA}\`.

**Source commit:** \`${SOURCE_SHA}\`
**Transformed tree hash:** \`${TRANSFORMED_TREE_SHA}\`

Generated by \`export-to-standalone.sh\`, which exports via \`git archive\`
from the exact commit above (never the live working tree) — see
[issue #25](https://github.com/Luminous-Dynamics/luminous-dynamics/issues/25)
for why the old \`sync-to-standalone.sh\` could not guarantee this.

Review before merging. Do not squash away the source-commit/tree-hash
provenance in the commit message."

info "Opening PR..."
if command -v gh >/dev/null 2>&1; then
    PR_URL=$(gh pr create --repo Luminous-Dynamics/symthaea \
        --base main --head "${EXPORT_BRANCH}" \
        --title "export: from monorepo @ ${SOURCE_SHA_SHORT}" \
        --body "${PR_BODY}")
    ok "PR opened: ${PR_URL}"
else
    warn "gh CLI not found — branch pushed but no PR opened. Open one manually:"
    echo "  https://github.com/Luminous-Dynamics/symthaea/compare/main...${EXPORT_BRANCH}"
fi
