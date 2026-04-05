#!/usr/bin/env bash
# Launch Sol Atlas via Symtropy Engine
# Handles NixOS library paths for X11 + Vulkan rendering
set -euo pipefail

cd "$(dirname "$0")"

EXTRA_ARGS="${*}"

echo "[symtropy] Starting Sol Atlas... (args: --globe $EXTRA_ARGS)"

exec nix-shell -p \
  libx11 \
  libxcursor \
  libxi \
  libxrandr \
  vulkan-loader \
  libxkbcommon \
  --run "
    # Extract library paths from NIX_LDFLAGS
    NIXLIBS=\$(echo \$NIX_LDFLAGS | grep -oP '(?<=-L)[^ ]+' | tr '\n' ':')
    export LD_LIBRARY_PATH=\"/run/opengl-driver/lib:\${NIXLIBS}\${LD_LIBRARY_PATH:-}\"
    unset WAYLAND_DISPLAY
    export WINIT_UNIX_BACKEND=x11
    exec ./target/release/symtropy --globe $EXTRA_ARGS
  "
