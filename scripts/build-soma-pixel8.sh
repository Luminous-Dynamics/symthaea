#!/usr/bin/env bash
# Build Soma for Pixel 8 Pro and deploy via adb.
#
# Prerequisites: Run inside `nix develop .#mobile` shell.
#
# Usage:
#   nix develop .#mobile
#   ./scripts/build-soma-pixel8.sh [--deploy]
#
# Options:
#   --deploy    Install APK to connected Pixel 8 Pro via adb
#   --release   Build in release mode (default)
#   --debug     Build in debug mode
#   --holon IP  Set desktop Holon IP for the demo app (default: auto-detect LAN IP)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
SOMA_CRATE="$WORKSPACE/crates/symthaea-soma"
ANDROID_DIR="$SOMA_CRATE/android"
TARGET="aarch64-linux-android"
PROFILE="release"
DEPLOY=false
HOLON_IP=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --deploy) DEPLOY=true; shift ;;
        --debug) PROFILE="debug"; shift ;;
        --release) PROFILE="release"; shift ;;
        --holon) HOLON_IP="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Verify we're in the mobile nix shell
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    echo "ERROR: Not in mobile nix shell. Run: nix develop .#mobile"
    exit 1
fi

# Auto-detect LAN IP if not specified
if [[ -z "$HOLON_IP" ]]; then
    HOLON_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' || echo "192.168.1.100")
fi

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Building Soma for Pixel 8 Pro (Tensor G3, arm64-v8a)       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Profile:  $PROFILE"
echo "  Target:   $TARGET"
echo "  Holon IP: $HOLON_IP:7778"
echo ""

# Step 1: Build Rust native library
echo ">>> Step 1/3: Building Rust native library..."
cd "$WORKSPACE"
if [[ "$PROFILE" == "release" ]]; then
    cargo build --target "$TARGET" --release -p symthaea-soma --features native-ffi
else
    cargo build --target "$TARGET" -p symthaea-soma --features native-ffi
fi
echo "    Done: target/$TARGET/$PROFILE/libsymthaea_soma.so"

# Step 2: Build JNI glue
echo ">>> Step 2/3: Building JNI layer..."
cd "$ANDROID_DIR"
./build-jni.sh
echo "    Done: jniLibs/arm64-v8a/libsoma_jni.so"

# Step 3: Build APK
echo ">>> Step 3/3: Building Android APK..."
if [[ "$PROFILE" == "release" ]]; then
    ./gradlew assembleRelease --quiet
    APK="demo/build/outputs/apk/release/demo-release-unsigned.apk"
else
    ./gradlew assembleDebug --quiet
    APK="demo/build/outputs/apk/debug/demo-debug.apk"
fi
echo "    Done: $APK"

# Deploy if requested
if $DEPLOY; then
    echo ""
    echo ">>> Deploying to Pixel 8 Pro..."
    adb install -r "$APK"
    echo "    Installed. Launch: io.symthaea.soma.demo"
    echo ""
    echo "  Configure Holon host in the app: $HOLON_IP"
    echo "  Desktop daemon: $WORKSPACE/target/debug/symthaea-holon"
    echo "  Dashboard: http://$HOLON_IP:7778/holon/dashboard"
fi

echo ""
echo "Build complete."
