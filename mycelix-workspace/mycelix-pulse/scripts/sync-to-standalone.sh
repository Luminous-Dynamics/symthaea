#!/usr/bin/env bash
# Sync mycelix-mail from monorepo to standalone public repo
# Usage: ./scripts/sync-to-standalone.sh [--dry-run] [--skip-check] [--force]
#
# This script syncs the mycelix-mail directory to the standalone
# Luminous-Dynamics/mycelix-mail repository for CI testing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MONOREPO_ROOT="$(cd "$MAIL_DIR/.." && pwd)"

# Configuration
STANDALONE_REMOTE="git@github.com:Luminous-Dynamics/mycelix-mail.git"
STANDALONE_DIR="${STANDALONE_DIR:-/tmp/mycelix-mail-standalone}"

# Parse flags
DRY_RUN=false
SKIP_CHECK=false
FORCE=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --skip-check) SKIP_CHECK=true ;;
        --force) FORCE=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

echo "=== Mycelix-Mail Standalone Sync ==="
echo "Source: $MAIL_DIR"
echo "Target: $STANDALONE_DIR"
echo ""

# Pre-flight checks
if [ "$SKIP_CHECK" = false ]; then
    echo "Running pre-flight checks..."

    # Check for uncommitted changes
    cd "$MAIL_DIR"
    if [ -n "$(git status --porcelain holochain/ happ/ tests/ scripts/ 2>/dev/null)" ]; then
        if [ "$FORCE" = false ]; then
            echo "WARNING: Uncommitted changes in mycelix-mail/"
            echo "Use --force to sync anyway, or commit first."
            exit 1
        fi
    fi

    # Quick cargo check
    echo "Running cargo check..."
    cd "$MAIL_DIR/holochain"
    if ! cargo check --quiet 2>/dev/null; then
        if [ "$FORCE" = false ]; then
            echo "ERROR: cargo check failed. Fix errors before syncing."
            exit 1
        fi
    fi
    echo "Pre-flight checks passed."
fi

# Prepare standalone directory
if [ -d "$STANDALONE_DIR" ]; then
    echo "Updating existing standalone repo..."
    cd "$STANDALONE_DIR"
    git pull --rebase origin main 2>/dev/null || true
else
    echo "Cloning standalone repo..."
    git clone "$STANDALONE_REMOTE" "$STANDALONE_DIR" 2>/dev/null || {
        echo "Creating fresh standalone directory (no remote yet)..."
        mkdir -p "$STANDALONE_DIR"
        cd "$STANDALONE_DIR"
        git init
    }
fi

# Sync files
echo ""
echo "Syncing files..."

RSYNC_FLAGS="-av --delete --exclude=target/ --exclude=node_modules/ --exclude=.git/ --exclude=fuzz/corpus/ --exclude=fuzz/artifacts/"

if [ "$DRY_RUN" = true ]; then
    RSYNC_FLAGS="$RSYNC_FLAGS --dry-run"
fi

# Sync holochain workspace
rsync $RSYNC_FLAGS "$MAIL_DIR/holochain/" "$STANDALONE_DIR/holochain/"

# Sync backend
rsync $RSYNC_FLAGS "$MAIL_DIR/happ/" "$STANDALONE_DIR/happ/"

# Sync tests
rsync $RSYNC_FLAGS "$MAIL_DIR/tests/" "$STANDALONE_DIR/tests/" 2>/dev/null || true

# Sync scripts
rsync $RSYNC_FLAGS "$MAIL_DIR/scripts/" "$STANDALONE_DIR/scripts/"

# Sync config files
for f in Cargo.toml flake.nix README.md LICENSE .gitignore; do
    if [ -f "$MAIL_DIR/$f" ]; then
        cp "$MAIL_DIR/$f" "$STANDALONE_DIR/$f"
    fi
done

# Sync CI workflow
mkdir -p "$STANDALONE_DIR/.github/workflows"
if [ -f "$MAIL_DIR/.github/workflows/ci.yml" ]; then
    cp "$MAIL_DIR/.github/workflows/ci.yml" "$STANDALONE_DIR/.github/workflows/ci.yml"
fi

# Fix license if needed (AGPL for standalone)
if [ -f "$STANDALONE_DIR/LICENSE" ]; then
    if ! grep -q "AGPL" "$STANDALONE_DIR/LICENSE"; then
        echo "WARNING: LICENSE file doesn't mention AGPL. Verify license is correct."
    fi
fi

echo ""
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN complete. No files were changed."
else
    echo "Sync complete!"
    echo ""
    echo "Next steps:"
    echo "  cd $STANDALONE_DIR"
    echo "  git add -A"
    echo "  git commit -m 'sync: update from monorepo'"
    echo "  git push origin main"
fi
