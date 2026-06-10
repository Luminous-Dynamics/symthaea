#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Deploy Spore demo to /spore/ subdirectory on gh-pages.
# Preserves existing root content (Sovereign Inoculation installer).
#
# Usage:
#   ./deploy-spore-subdir.sh
#
# Result:
#   Spore available at https://install.nixforhumanity.org/spore/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WWW_DIR="$SCRIPT_DIR/www"
PKG_DIR="$WWW_DIR/pkg"
REPO="Luminous-Dynamics/symthaea"
BRANCH="gh-pages"

# Check WASM
if [[ ! -f "$PKG_DIR/symthaea_spore_bg.wasm" ]]; then
    echo "[deploy] WASM not found. Run ./build-wasm.sh first."
    exit 1
fi

echo "[deploy] WASM binary: $(du -h "$PKG_DIR/symthaea_spore_bg.wasm" | cut -f1)"

# Clone gh-pages (shallow)
DEPLOY_DIR=$(mktemp -d)
trap "rm -rf $DEPLOY_DIR" EXIT

echo "[deploy] Cloning gh-pages..."
git clone --depth 1 --branch "$BRANCH" "https://github.com/$REPO.git" "$DEPLOY_DIR"

# Create /spore/ subdirectory
SPORE_DIR="$DEPLOY_DIR/spore"
rm -rf "$SPORE_DIR"
mkdir -p "$SPORE_DIR/pkg" "$SPORE_DIR/css" "$SPORE_DIR/js"

# Copy Spore files
cp "$WWW_DIR/index.html" "$SPORE_DIR/"
cp "$WWW_DIR/portal.html" "$SPORE_DIR/" 2>/dev/null || true
cp "$PKG_DIR/symthaea_spore_bg.wasm" "$SPORE_DIR/pkg/"
cp "$PKG_DIR/symthaea_spore.js" "$SPORE_DIR/pkg/"
cp "$PKG_DIR/symthaea_spore.d.ts" "$SPORE_DIR/pkg/" 2>/dev/null || true
cp -r "$WWW_DIR/css/"* "$SPORE_DIR/css/" 2>/dev/null || true
cp -r "$WWW_DIR/js/"* "$SPORE_DIR/js/" 2>/dev/null || true
cp "$WWW_DIR/glyphs.js" "$SPORE_DIR/" 2>/dev/null || true
cp "$WWW_DIR/spore-worker.js" "$SPORE_DIR/" 2>/dev/null || true
cp "$WWW_DIR/sw.js" "$SPORE_DIR/" 2>/dev/null || true

# Commit and push
cd "$DEPLOY_DIR"
git add spore/
git commit -m "deploy: Spore consciousness demo at /spore/" --allow-empty
git push origin "$BRANCH"

echo ""
echo "[deploy] Spore deployed to: https://install.nixforhumanity.org/spore/"
echo "[deploy] Root installer unchanged."
