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

if [ ! -d "${MONOREPO_ROOT}/mycelix-workspace/mycelix-finance" ]; then
    error "Cannot find ${MONOREPO_ROOT}/mycelix-workspace/mycelix-finance — run from monorepo root"
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
    # ML training data — mycelix-core/0TML/datasets/ is 7.6 GB of femnist/mnist/
    # cifar/emnist/shakespeare training corpora. Regenerable from standard
    # sources; never belongs in a public repo. Same rule for any `datasets/`
    # or `checkpoints/` directory that sneaks in via other clusters.
    --exclude='datasets'
    --exclude='checkpoints'
    --exclude='train_data'
    # Python virtualenvs — never belong in git. Includes nested `*/venv/` etc.
    --exclude='.venv'
    --exclude='venv'
    --exclude='env/'
    # Build outputs that regenerate on every trunk/cargo/tsc build — exclude so
    # content-hashed artifacts (e.g. dist/main-<hash>.css) don't accumulate
    # in the public standalone with every sync.
    #
    # Patterns use no leading slash + no trailing slash so they match any
    # directory-or-file by that basename, anywhere in the tree (the earlier
    # trailing-slash form silently failed to match nested observatory dirs).
    --exclude='dist'
    --exclude='.svelte-kit'
    --exclude='.stage'
    --exclude='.next'
    --exclude='.turbo'
    # TypeDoc / Typedoc / auto-generated API HTML (e.g. sdk-ts/docs/).
    # Dominates the repo language breakdown if included (4,487 HTML files →
    # 58% HTML on a Rust project). Regenerable from source via `npx typedoc`.
    --exclude='docs/classes'
    --exclude='docs/interfaces'
    --exclude='docs/enums'
    --exclude='docs/functions'
    --exclude='docs/variables'
    --exclude='docs/types'
    --exclude='docs/modules'
    --exclude='docs/assets'
    --exclude='docs/media'
)
RSYNC_OPTS=(-a --delete -v "${RSYNC_EXCLUDE[@]}")
# Note: no --dry-run here even when $DRY_RUN is set. ${STANDALONE_REPO} is
# always a disposable local clone (reset --hard to origin/main at the top of
# every run — see above), so writing into it is harmless either way; the
# real dry-run safety gate is the commit/push skip near the bottom of this
# script. Passing --dry-run to rsync used to mean it never actually wrote
# anything, which made the later "Changes in standalone repo" diff always
# trivially empty — a --dry-run invocation could never show what it would
# actually change. Letting rsync really write here fixes that.

# --- Sync cluster directories ------------------------------------------------
#
# Each cluster is its own workspace in the standalone repo.
# Path mapping is 1:1 between monorepo and standalone.

CLUSTERS=(
    # Core civic + resource clusters (Tier 1-2)
    mycelix-commons
    mycelix-civic
    mycelix-hearth
    mycelix-finance
    mycelix-governance
    mycelix-identity
    mycelix-personal
    mycelix-attribution
    # Learning + reputation pipeline
    mycelix-praxis
    mycelix-craft
    # Knowledge + media
    mycelix-knowledge
    mycelix-music
    # Physical + energy
    mycelix-energy
    mycelix-climate
    mycelix-manufacturing
    # Foundation (FL + mycelix-core-types + feldman-dkg live as shared crates
    # already, but the top-level cluster carries its README + docs + 0TML research)
    mycelix-core
    # State interop (dual-DID airlock for state-facing credentials)
    mycelix-lawful-identity
    # Frozen — maintenance-only, see mycelix-marketplace/STATUS.md. Still
    # synced (no active dedicated standalone repo/sync script found for it,
    # unlike desci/supplychain/space/observatory below — the "own repo"
    # note here was stale, same as the now-deleted mycelix-praxis
    # sync-to-standalone.sh that referenced a nonexistent repo).
    mycelix-marketplace
    # Note: clusters WITH their own standalone public repos intentionally NOT
    # synced here (to avoid duplicating code):
    #   mycelix-desci       → Luminous-Dynamics/mycelix-desci
    #   mycelix-supplychain → Luminous-Dynamics/mycelix-supplychain
    #   mycelix-space       → Luminous-Dynamics/mycelix-space
    #   mycelix-observatory → Luminous-Dynamics/mycelix-observatory
    # Submodules (own standalone):
    #   mycelix-health      → Luminous-Dynamics/mycelix-health (submodule)
    # Renamed-to-pulse:
    #   mycelix-workspace/mycelix-pulse (synced via the mycelix-workspace block below)
)

# All Mycelix cluster work is being consolidated into mycelix-workspace/
# (see MYCELIX_REVIEW.md P0 #1 and the July 2 migration commits). Clusters
# already moved read from their new mycelix-workspace/<cluster> location;
# clusters not yet moved still read from top-level until they migrate too.
# Remaining as of 2026-07-02: desci (blocked on mycelix-multiworld-sim
# settling). praxis, core, health, and marketplace have since moved
# (health via submodule surgery, has its own standalone repo — see the
# wholesale sync exclude below; marketplace frozen but kept, see its
# STATUS.md). Once every cluster in CLUSTERS has moved, this whole loop
# collapses into the wholesale mycelix-workspace sync below and can be
# deleted.
MOVED_TO_WORKSPACE=(
    mycelix-commons mycelix-civic mycelix-hearth mycelix-finance
    mycelix-governance mycelix-identity mycelix-personal mycelix-attribution
    mycelix-craft mycelix-knowledge mycelix-music mycelix-energy
    mycelix-climate mycelix-manufacturing mycelix-lawful-identity mycelix-core
    mycelix-praxis mycelix-marketplace
)

cluster_source_dir() {
    local cluster="$1"
    for moved in "${MOVED_TO_WORKSPACE[@]}"; do
        if [ "$cluster" = "$moved" ]; then
            echo "${MONOREPO_ROOT}/mycelix-workspace/${cluster}"
            return
        fi
    done
    echo "${MONOREPO_ROOT}/${cluster}"
}

sync_dir() {
    local src="$1"
    local dst="$2"
    shift 2
    local extra_excludes=("$@")
    if [ ! -d "$src" ]; then
        warn "Skipping $(basename "$src")/ (not found in monorepo)"
        return
    fi
    info "Syncing $(basename "$dst")/"

    # Export from git HEAD rather than rsyncing the live working tree.
    #
    # This monorepo routinely runs 12+ concurrent Claude sessions (see
    # .claude/rules/CONCURRENT_SESSIONS.md), each of which can leave
    # uncommitted, unrelated, in-progress work sitting in files anywhere
    # under the sync source paths. Rsyncing the working tree directly (the
    # old behavior) means that state — half-finished, unreviewed, possibly
    # from a completely different task — gets shipped to the public
    # standalone repo. Discovered 2026-07-02 (see MYCELIX_REVIEW.md
    # addendum): a routine sync attempt would have carried another
    # session's in-progress Symtropy refactor and several other clusters'
    # uncommitted edits, mixed into what was meant to be one targeted fix.
    #
    # Exporting `HEAD` only ever includes committed content, which also
    # matches what the eventual commit message already claims below
    # ("update from monorepo @ <SHA>") — the sync was always meant to
    # represent a specific commit, not scratch working-tree state.
    local rel_path="${src#"${MONOREPO_ROOT}"/}"
    local clean_tmp
    clean_tmp="$(mktemp -d)"
    if ! git -C "${MONOREPO_ROOT}" archive HEAD -- "${rel_path}" 2>/dev/null \
        | tar -x -C "${clean_tmp}" 2>/dev/null; then
        warn "Skipping $(basename "$dst")/ (${rel_path} has no committed content at HEAD)"
        rm -rf "${clean_tmp}"
        return
    fi
    if [ ! -d "${clean_tmp}/${rel_path}" ]; then
        warn "Skipping $(basename "$dst")/ (${rel_path} has no committed content at HEAD)"
        rm -rf "${clean_tmp}"
        return
    fi
    rsync "${RSYNC_OPTS[@]}" "${extra_excludes[@]}" "${clean_tmp}/${rel_path}/" "${dst}/"
    rm -rf "${clean_tmp}"
}

info "=== Syncing cluster directories ==="
for cluster in "${CLUSTERS[@]}"; do
    sync_dir "$(cluster_source_dir "$cluster")" "${STANDALONE_REPO}/${cluster}"
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
#
# This list was found incomplete 2026-07-02: mycelix-zkp-core and
# mycelix-bridge-proc were referenced by moved clusters via a direct
# "../crates/<name>" path but never actually synced here — meaning every
# cluster depending on them would fail to build in the standalone repo.
# Found by testing `cargo check` against a real synced copy, not by
# inspection — if a future crate hits the same failure mode, grep for it:
#   grep -rhoE '[a-z0-9_-]+ = \{ path = "\.\./crates/[a-z0-9_-]+"' mycelix-workspace --include="Cargo.toml"
for crate_name in mycelix-bridge-common mycelix-bridge-entry-types mycelix-leptos-core mycelix-leptos-client \
                   mycelix-zkp-core mycelix-bridge-proc; do
    sync_dir \
        "${MONOREPO_ROOT}/crates/${crate_name}" \
        "${STANDALONE_REPO}/crates/${crate_name}"
done

# Direct path crates that only ever lived inside mycelix-workspace/crates/
# (never had a top-level ancestor, unlike the ones above). mycelix-crypto
# belongs here too but is skipped for now — it has zero git-tracked files
# (see task #34), so there's nothing for git-archive-based sync_dir() to
# export yet; add it once it's committed.
sync_dir \
    "${MONOREPO_ROOT}/mycelix-workspace/crates/mycelix-zome-helpers" \
    "${STANDALONE_REPO}/crates/mycelix-zome-helpers"

# Remapped crates (different paths in monorepo vs standalone)
# mycelix-core moved into mycelix-workspace/ 2026-07-02 — source paths updated.
sync_dir \
    "${MONOREPO_ROOT}/mycelix-workspace/mycelix-core/libs/mycelix-core-types" \
    "${STANDALONE_REPO}/crates/mycelix-core-types"

sync_dir \
    "${MONOREPO_ROOT}/mycelix-workspace/mycelix-core/libs/feldman-dkg" \
    "${STANDALONE_REPO}/crates/feldman-dkg"

echo

# --- Sync mycelix-workspace (SDKs, happs, scripts, etc.) --------------------
#
# mycelix-workspace/ also contains mycelix-workspace/mycelix-<cluster>/
# subdirectories left over from an earlier consolidation attempt — stale,
# partial (sometimes empty) duplicates of the real top-level cluster
# directories already synced above. Excluded here so this wholesale sync
# doesn't ship dead duplicate code to the public repo alongside the real
# copies. mycelix-pulse is the one cluster that genuinely lives inside
# mycelix-workspace and is NOT excluded.
info "=== Syncing mycelix-workspace ==="
sync_dir "${MONOREPO_ROOT}/mycelix-workspace" "${STANDALONE_REPO}/mycelix-workspace" \
    --exclude='/mycelix-supplychain/' \
    --exclude='/mycelix-desci/' \
    --exclude='/mycelix-health/'
    # mycelix-core's and mycelix-praxis's excludes removed 2026-07-02 — both moved here with real
    # tracked content (was previously excluded to hide an untracked stub).
    # mycelix-marketplace's exclude also removed 2026-07-02 — migrated,
    # frozen but kept (STATUS.md), no active dedicated standalone repo
    # found for it, so it's treated like core/praxis rather than like
    # desci/supplychain/health.
    # Note: mycelix-prism is NOT excluded — it has no top-level counterpart
    # and no dedicated sync script; this wholesale sync is its only current
    # path to the public repo. mycelix-health IS now excluded (added
    # 2026-07-02 when it moved here via submodule surgery) — it has its own
    # standalone repo (Luminous-Dynamics/mycelix-health) and git archive
    # doesn't recurse into submodule content anyway, so shipping it here
    # would be redundant at best.
echo

# --- Rewrite path dependencies for standalone layout -------------------------
#
# In the monorepo, some shared crates live at mycelix-core/libs/*.
# In the standalone repo, they live at crates/*.
# Clusters that depend on these need their Cargo.toml paths rewritten.
#
# sovereign-profile is different: it's a published crate (crates.io, v0.1.2 —
# see root CLAUDE.md's published-crates table) that several clusters depend on
# via a local `path` dependency, at a `../` depth tuned for wherever that
# cluster happens to sit in the monorepo. That depth is wrong as soon as the
# cluster's position relative to sovereign-profile's source changes — which it
# does, every time, in the standalone repo's flatter layout (e.g.
# mycelix-finance/crates/finance-wire-types sits one directory level shallower
# in standalone than in the monorepo, where it's nested inside
# mycelix-workspace/). Rather than chase a moving relative-path target,
# rewrite it to depend on the published crate directly — the standalone repo
# doesn't need or want a local copy of sovereign-profile's source at all.

if ! $DRY_RUN; then
    info "=== Rewriting path dependencies for standalone layout ==="

    # Map: monorepo path → standalone path (relative from cluster root)
    # ../mycelix-core/libs/mycelix-core-types → ../crates/mycelix-core-types
    # ../mycelix-core/libs/feldman-dkg        → ../crates/feldman-dkg
    # sovereign-profile path dep              → sovereign-profile = "0.1.2" (crates.io —
    #   0.1.2 specifically: 0.1.1's published feature set lacked
    #   `default = ["serde"]`, which the local source has, so path deps that
    #   omit an explicit `features = ["serde"]` (like finance-wire-types')
    #   would silently lose the Deserialize impl if 0.1.1 got resolved)
    # mycelix-zome-helpers's own mycelix-bridge-common ref → ../mycelix-bridge-common
    #   (mycelix-zome-helpers is synced from mycelix-workspace/crates/ where it
    #   needs 3 "../" to reach the top-level crates/mycelix-bridge-common; in
    #   the standalone it lands as a sibling under crates/ where 1 "../"
    #   is correct — same depth-mismatch shape as sovereign-profile above, but
    #   this dependency isn't a published crate, so it gets a path rewrite
    #   instead of a version rewrite.)
    while IFS= read -r toml_file; do
        if grep -q 'mycelix-core/libs/' "$toml_file" 2>/dev/null; then
            sed -i 's|mycelix-core/libs/mycelix-core-types|crates/mycelix-core-types|g' "$toml_file"
            sed -i 's|mycelix-core/libs/feldman-dkg|crates/feldman-dkg|g' "$toml_file"
            ok "Rewrote paths in $(basename "$(dirname "$(dirname "$toml_file")")")/...Cargo.toml"
        fi
        if grep -q 'sovereign-profile = { path = "[^"]*"' "$toml_file" 2>/dev/null; then
            # Broad match on purpose: sovereign-profile's relative path varies
            # by how deep the referencing crate sits (e.g. "../sovereign-profile"
            # from crates/mycelix-bridge-common itself vs.
            # "../../../crates/sovereign-profile" from a nested cluster crate) —
            # every variant needs the same published-crate rewrite, not just
            # ones that happen to spell out "crates/" in the path text.
            sed -i 's|sovereign-profile = { path = "[^"]*"|sovereign-profile = { version = "0.1.2"|' "$toml_file"
            ok "Rewrote sovereign-profile to published crate in $(basename "$(dirname "$toml_file")")/Cargo.toml"
        fi
        if [[ "$toml_file" == "${STANDALONE_REPO}/crates/mycelix-zome-helpers/Cargo.toml" ]] \
            && grep -q 'mycelix-bridge-common = { path = "[^"]*"' "$toml_file" 2>/dev/null; then
            sed -i 's|mycelix-bridge-common = { path = "[^"]*"|mycelix-bridge-common = { path = "../mycelix-bridge-common"|' "$toml_file"
            ok "Rewrote mycelix-zome-helpers's mycelix-bridge-common path for standalone layout"
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
