#!/usr/bin/env bash
# Launch Symtropy Nexus (engine launcher)
# Handles NixOS library paths for X11 + Vulkan rendering
set -euo pipefail

cd "$(dirname "$0")"

echo "[symtropy] Starting Nexus..."

exec nix-shell -p \
  xorg.libX11 \
  xorg.libXcursor \
  xorg.libXi \
  xorg.libXrandr \
  vulkan-loader \
  libxkbcommon \
  --run "
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:\$LD_LIBRARY_PATH
    exec ./target/release/symtropy \"\$@\"
  " -- "$@"
