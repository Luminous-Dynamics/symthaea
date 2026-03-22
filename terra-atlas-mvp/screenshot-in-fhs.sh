#!/usr/bin/env bash
# Script to take screenshot inside FHS environment

echo "🎭 Entering Playwright FHS environment..."
nix develop .#playwright --command playwright-fhs -c "
  cd /srv/luminous-dynamics/terra-atlas-mvp
  echo '📸 Taking screenshot...'
  node screenshot-firefox.js
"