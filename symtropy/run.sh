#!/usr/bin/env bash
# Symtropy launcher — sets up NixOS environment and runs the game.
# Usage: ./run.sh [--release] [--x11]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# NixOS runtime library paths (both X11 and Wayland)
ALSA_LIB=$(nix-build '<nixpkgs>' -A alsa-lib --no-out-link 2>/dev/null)/lib
XKBCOMMON=$(nix-build '<nixpkgs>' -A libxkbcommon --no-out-link 2>/dev/null)/lib
VULKAN=$(nix-build '<nixpkgs>' -A vulkan-loader --no-out-link 2>/dev/null)/lib
X11=$(nix-build '<nixpkgs>' -A xorg.libX11 --no-out-link 2>/dev/null)/lib
XCURSOR=$(nix-build '<nixpkgs>' -A xorg.libXcursor --no-out-link 2>/dev/null)/lib
XI=$(nix-build '<nixpkgs>' -A xorg.libXi --no-out-link 2>/dev/null)/lib
XRANDR=$(nix-build '<nixpkgs>' -A xorg.libXrandr --no-out-link 2>/dev/null)/lib
WAYLAND=$(nix-build '<nixpkgs>' -A wayland --no-out-link 2>/dev/null)/lib
LIBDECOR=$(nix-build '<nixpkgs>' -A libdecor --no-out-link 2>/dev/null)/lib

export LD_LIBRARY_PATH="$XKBCOMMON:$VULKAN:$X11:$XCURSOR:$XI:$XRANDR:$WAYLAND:$LIBDECOR:$ALSA_LIB:${LD_LIBRARY_PATH:-}"

# Backend selection
if [[ "${1:-}" == "--x11" ]] || [[ "${2:-}" == "--x11" ]]; then
    export WINIT_UNIX_BACKEND=x11
    echo "[symtropy] Forcing X11 backend"
elif [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    export WINIT_UNIX_BACKEND=wayland
    echo "[symtropy] Using native Wayland"
else
    echo "[symtropy] Using default backend"
fi

# Choose debug or release binary
if [[ "${1:-}" == "--release" ]] || [[ "${2:-}" == "--release" ]]; then
    BIN="target/release/symtropy"
else
    BIN="target/debug/symtropy"
fi

if [[ ! -f "$BIN" ]]; then
    echo "Binary not found: $BIN"
    echo "Build with: cargo build [--release]"
    exit 1
fi

echo "[symtropy] Launching $BIN..."
export RUST_LOG="${RUST_LOG:-warn,symtropy=info}"
export RUST_BACKTRACE=1
exec "$BIN" "$@"
