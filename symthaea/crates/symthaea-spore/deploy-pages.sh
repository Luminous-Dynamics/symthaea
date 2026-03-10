#!/usr/bin/env bash
# Deploy Spore demo to GitHub Pages on the public symthaea repo.
#
# This script:
# 1. Builds the WASM binary (if not already built)
# 2. Copies www/ + www/pkg/ to a temporary directory
# 3. Pushes to the gh-pages branch of Luminous-Dynamics/symthaea
#
# Prerequisites:
#   - gh CLI authenticated
#   - WASM binary built (run ./build-wasm.sh first)
#
# Usage:
#   ./deploy-pages.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WWW_DIR="$SCRIPT_DIR/www"
PKG_DIR="$WWW_DIR/pkg"
REPO="Luminous-Dynamics/symthaea"
BRANCH="gh-pages"

# Check WASM is built
if [[ ! -f "$PKG_DIR/symthaea_spore_bg.wasm" ]]; then
    echo "[deploy] WASM not found. Building..."
    "$SCRIPT_DIR/build-wasm.sh"
fi

echo "[deploy] WASM binary: $(du -h "$PKG_DIR/symthaea_spore_bg.wasm" | cut -f1)"

# Create temp deploy directory
DEPLOY_DIR=$(mktemp -d)
trap "rm -rf $DEPLOY_DIR" EXIT

# Copy demo files
cp "$WWW_DIR/index.html" "$DEPLOY_DIR/"
cp "$WWW_DIR/spore-worker.js" "$DEPLOY_DIR/"
cp "$WWW_DIR/smoke-test.mjs" "$DEPLOY_DIR/"
mkdir -p "$DEPLOY_DIR/pkg"
cp "$PKG_DIR/symthaea_spore_bg.wasm" "$DEPLOY_DIR/pkg/"
cp "$PKG_DIR/symthaea_spore.js" "$DEPLOY_DIR/pkg/"
cp "$PKG_DIR/symthaea_spore.d.ts" "$DEPLOY_DIR/pkg/"
cp "$PKG_DIR/symthaea_spore_bg.wasm.d.ts" "$DEPLOY_DIR/pkg/"

# Add ETHICS.md if it exists
if [[ -f "$SCRIPT_DIR/ETHICS.md" ]]; then
    cp "$SCRIPT_DIR/ETHICS.md" "$DEPLOY_DIR/"
fi

# Add a .nojekyll file (GitHub Pages)
touch "$DEPLOY_DIR/.nojekyll"

# Check if gh-pages branch exists
cd "$DEPLOY_DIR"
git init
git checkout -b "$BRANCH"
git add -A
git commit -m "Deploy Spore consciousness demo

WASM: $(du -h "$PKG_DIR/symthaea_spore_bg.wasm" | cut -f1) raw
Features: consciousness engine, 4 validation experiments, multi-substrate comparison
Ethical safeguards: epistemic status, Eight Harmonies, instance counter"

echo ""
echo "[deploy] Ready to push. Files:"
ls -lhR "$DEPLOY_DIR/"
echo ""
echo "[deploy] To deploy, run:"
echo "  cd $DEPLOY_DIR"
echo "  git remote add origin https://github.com/$REPO.git"
echo "  git push -f origin $BRANCH"
echo ""
echo "Then enable GitHub Pages in repo settings: Source = gh-pages branch."
echo "Demo will be at: https://luminous-dynamics.github.io/symthaea/"
echo ""
echo "NOTE: This will force-push to $BRANCH. The deploy dir is at: $DEPLOY_DIR"
echo "To auto-deploy, run: cd $DEPLOY_DIR && git remote add origin https://github.com/$REPO.git && git push -f origin $BRANCH"
