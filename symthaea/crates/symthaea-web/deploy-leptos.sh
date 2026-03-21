#!/usr/bin/env bash
# deploy-leptos.sh — Deploy Leptos frontend to GitHub Pages
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Build
bash build-leptos.sh "$@"

# Ensure SporeEngine WASM is in dist (Trunk copies assets/ but may miss large files)
if [[ -f assets/pkg/symthaea_spore_bg.wasm ]] && [[ ! -f dist/assets/pkg/symthaea_spore_bg.wasm ]]; then
    cp assets/pkg/symthaea_spore_bg.wasm dist/assets/pkg/
    echo "[deploy] Copied SporeEngine WASM to dist"
fi

# Copy Broca checkpoint to assets dir (worker fetches ./assets/broca-spore-v1.bin)
mkdir -p dist/assets
cp assets/broca-spore-v1.bin dist/assets/ 2>/dev/null && echo "[deploy] Copied broca checkpoint to dist/assets/" || echo "[deploy] broca-spore-v1.bin not found (not critical)"

# Add .nojekyll for GitHub Pages (prevents underscore-prefixed files being ignored)
touch dist/.nojekyll

# Custom domain
echo "symthaea.luminousdynamics.io" > dist/CNAME

echo ""
echo "[deploy] dist/ contents:"
ls -la dist/
echo ""
echo "[deploy] WASM sizes:"
du -h dist/*.wasm dist/assets/pkg/*.wasm 2>/dev/null || true

# Deploy to gh-pages branch
cd dist
git init -q
git checkout -b gh-pages
git add -A
git commit -q -m "deploy: Leptos CSR portal $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git remote add origin git@github.com:Luminous-Dynamics/symthaea.git
git push -f origin gh-pages
cd ..
rm -rf dist/.git

echo ""
echo "[deploy] Deployed to https://luminous-dynamics.github.io/symthaea/"
