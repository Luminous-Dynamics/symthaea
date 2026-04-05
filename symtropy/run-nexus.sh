#!/usr/bin/env bash
# Launch Symtropy Nexus (engine launcher)
# Handles NixOS library paths for X11 + Vulkan rendering
set -euo pipefail

cd "$(dirname "$0")"

echo "[symtropy] Starting Nexus..."

exec nix-shell -p \
  libx11 \
  libxcursor \
  libxi \
  libxrandr \
  vulkan-loader \
  libxkbcommon \
  --run "
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:\$LD_LIBRARY_PATH
    unset WAYLAND_DISPLAY
    export WINIT_UNIX_BACKEND=x11
    exec ./target/release/symtropy \"\$@\"
  " -- "$@"
