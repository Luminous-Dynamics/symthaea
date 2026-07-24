#!/usr/bin/env bash
# Sync the Muse showcase site to its standalone public GitHub Pages repo.
#
# Standalone: github.com/Luminous-Dynamics/muse-site
#   Serves muse.luminousdynamics.io via GitHub Pages (DNS CNAME to
#   luminous-dynamics.github.io — the Sol Atlas pattern, NOT the tunnel).
#   Pushing to main deploys the public site.
#
# DELIBERATELY a separate repo from the symthaea source standalone: the
# site is presentation (HTML + encoded audio), the source repo is 139
# crates with CI — coupling them would rebuild/redeploy the site on every
# source sync and bloat the source history with audio assets.
#
# Layout: monorepo crates/domains/symthaea-muse/site/ -> standalone / (root)
# The site is exported from git HEAD, never the working tree.
#
# Usage:
#   bash crates/domains/symthaea-muse/scripts/sync-site-to-standalone.sh [--dry-run]
#
# ONLY run once the content is REVIEWED — pushing publishes the site.
# First run: create the repo first with
#   gh repo create Luminous-Dynamics/muse-site --public \
#     --description "Symthaea Muse — music with provenance (muse.luminousdynamics.io)"
# and enable Pages (Settings → Pages → deploy from branch, main, / root),
# then add the Cloudflare DNS record:
#   CNAME  muse  luminous-dynamics.github.io  (DNS only, no proxy)

set -euo pipefail

STANDALONE_REMOTE="git@github.com:Luminous-Dynamics/muse-site.git"
SYMTHAEA_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SITE_DIR_REL="crates/domains/symthaea-muse/site"
STANDALONE_REPO="/tmp/muse-site-standalone-sync"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { printf "${CYAN}[info]${RESET}  %s\n" "$*"; }
ok()    { printf "${GREEN}[ok]${RESET}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[warn]${RESET}  %s\n" "$*"; }
error() { printf "${RED}[error]${RESET} %s\n" "$*"; exit 1; }

[ -f "${SYMTHAEA_ROOT}/${SITE_DIR_REL}/index.html" ] || error "site not found at ${SITE_DIR_REL}"
$DRY_RUN && warn "DRY RUN — no commits or pushes"

# Export from HEAD, never the working tree (the Sol Atlas lesson).
EXPORT_DIR="$(mktemp -d /tmp/muse-site-export-XXXX)"
trap 'rm -rf "${EXPORT_DIR}"' EXIT
git -C "${SYMTHAEA_ROOT}" archive HEAD -- "${SITE_DIR_REL}" | tar -x -C "${EXPORT_DIR}"
SRC="${EXPORT_DIR}/${SITE_DIR_REL}"
[ -f "${SRC}/index.html" ] || error "index.html missing from HEAD export — commit the site first"

if [ -d "${STANDALONE_REPO}/.git" ]; then
    info "Updating standalone clone..."
    git -C "${STANDALONE_REPO}" fetch origin && git -C "${STANDALONE_REPO}" reset --hard origin/main
else
    info "Cloning standalone..."
    git clone "${STANDALONE_REMOTE}" "${STANDALONE_REPO}"
fi

rsync -a --delete --exclude '.git' "${SRC}/" "${STANDALONE_REPO}/"

cd "${STANDALONE_REPO}"
if git status --porcelain | grep -q .; then
    git add -A
    HEAD_SHA="$(git -C "${SYMTHAEA_ROOT}" rev-parse --short HEAD)"
    if $DRY_RUN; then
        warn "DRY RUN — would commit and push:"
        git status --short
    else
        git commit -m "sync from symthaea monorepo @ ${HEAD_SHA}"
        git push origin main
        ok "Pushed — GitHub Pages will deploy muse.luminousdynamics.io shortly."
    fi
else
    ok "Standalone already up to date."
fi
