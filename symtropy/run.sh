#!/usr/bin/env bash
# Symtropy launcher — runs the game via the project's nix devShell (flake.nix).
# Usage: ./run.sh [--release] [--atlas] [--mycelix] [--swarm] [--x11] [--autostart] [--globe]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Backend selection
for arg in "$@"; do
    case "$arg" in
        --x11) export WINIT_UNIX_BACKEND=x11 ;;
    esac
done
if [[ -z "${WINIT_UNIX_BACKEND:-}" ]]; then
    if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
        export WINIT_UNIX_BACKEND=wayland
        echo "[symtropy] Using native Wayland"
    else
        echo "[symtropy] Using default backend"
    fi
else
    echo "[symtropy] Forcing X11 backend"
fi

# Feature flags and game args
FEATURES=""
GAME_ARGS=""
PROFILE="debug"
for arg in "$@"; do
    case "$arg" in
        --release) PROFILE="release" ;;
        --atlas) FEATURES="${FEATURES:+$FEATURES,}atlas" ;;
        --mycelix) FEATURES="${FEATURES:+$FEATURES,}mycelix,vision-manifold,swarm,api_module" ;;
        --swarm) FEATURES="${FEATURES:+$FEATURES,}vision-manifold,swarm,api_module" ;;
        --autostart) GAME_ARGS="$GAME_ARGS --autostart" ;;
        --globe) GAME_ARGS="$GAME_ARGS --globe" ;;
    esac
done

if [[ "$PROFILE" == "release" ]]; then
    BIN="target/release/symtropy-launcher"
    CARGO_PROFILE_ARGS="--release"
else
    BIN="target/debug/symtropy-launcher"
    CARGO_PROFILE_ARGS=""
fi

if [[ ! -f "$BIN" ]]; then
    echo "[symtropy] Building $PROFILE binary (first run for this profile/feature set — can take a while)..."
    if [[ -n "$FEATURES" ]]; then
        nix develop --command cargo build $CARGO_PROFILE_ARGS --features "$FEATURES"
    else
        nix develop --command cargo build $CARGO_PROFILE_ARGS
    fi
fi

# Bevy resolves assets relative to the binary's own directory, not cwd —
# without this symlink the game can't find its shaders/textures at all
# (previously had to be recreated by hand after every fresh build).
mkdir -p "target/$PROFILE"
ln -sfn ../../assets "target/$PROFILE/assets"

echo "[symtropy] Launching $BIN..."
if [[ -n "$FEATURES" ]]; then
    echo "[symtropy] Features active: $FEATURES"
fi
export RUST_LOG="${RUST_LOG:-warn,symtropy=info}"
export RUST_BACKTRACE=1
exec nix develop --command "$BIN" $GAME_ARGS
