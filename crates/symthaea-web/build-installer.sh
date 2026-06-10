#!/usr/bin/env bash
# Build the Sovereign Inoculation installer (not the consciousness portal).
# Deploys to install.nixforhumanity.org via GitHub Pages.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CARGO_BUILD_FEATURES="installer"

echo "Building Sovereign Inoculation installer..."
~/.cargo/bin/trunk build --release --features installer

echo ""
echo "Built! Deploy with:"
echo "  cd dist && git init && git checkout -b gh-pages && git add -A"
echo "  echo 'install.nixforhumanity.org' > CNAME && touch .nojekyll"
echo "  git commit -m 'deploy' && git remote add origin git@github.com:Luminous-Dynamics/symthaea.git"
echo "  git push -f origin gh-pages && rm -rf .git"
