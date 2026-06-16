#!/usr/bin/env bash
# Build Symthaea Spore for Android (aarch64-linux-android).
#
# Prerequisites:
#   - Android NDK installed (ANDROID_NDK_HOME set)
#   - Rust target: rustup target add aarch64-linux-android
#
# Usage:
#   ./build-android.sh          # Build release .so + .a
#   ./build-android.sh --check  # Type-check only (no NDK needed)
#
# Output:
#   target/aarch64-linux-android/release/libsymthaea_spore.so   (cdylib)
#   target/aarch64-linux-android/release/libsymthaea_spore.a    (staticlib)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="aarch64-linux-android"
PROFILE="release"

cd "$WORKSPACE_ROOT"

# Ensure target is installed
if ! rustup target list --installed | grep -q "$TARGET"; then
    echo "Installing Rust target $TARGET..."
    rustup target add "$TARGET"
fi

# Check-only mode (no NDK required)
if [[ "${1:-}" == "--check" ]]; then
    echo "=== Type-checking symthaea-spore for $TARGET ==="
    # CARGO_FEATURE_PURE=1 tells blake3 to skip C assembly (no NDK C compiler needed)
    RUSTC_WRAPPER="" CARGO_FEATURE_PURE=1 \
        cargo check --target "$TARGET" -p symthaea-spore
    echo "=== Check passed ==="
    exit 0
fi

# Full build requires NDK
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    echo "ERROR: ANDROID_NDK_HOME not set."
    echo "Install Android NDK and set ANDROID_NDK_HOME to the NDK root."
    echo "  NixOS: nix-shell -p androidenv.androidPkgs.ndk-bundle"
    echo "  Manual: https://developer.android.com/ndk/downloads"
    echo ""
    echo "For type-checking only (no NDK needed): $0 --check"
    exit 1
fi

# Find the NDK clang for aarch64
NDK_TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64"
if [[ ! -d "$NDK_TOOLCHAIN" ]]; then
    echo "ERROR: NDK toolchain not found at $NDK_TOOLCHAIN"
    exit 1
fi

# Use API level 24 (Android 7.0) as minimum — covers 99%+ of devices
API_LEVEL="${ANDROID_API_LEVEL:-24}"
CC_ANDROID="$NDK_TOOLCHAIN/bin/aarch64-linux-android${API_LEVEL}-clang"
AR_ANDROID="$NDK_TOOLCHAIN/bin/llvm-ar"

if [[ ! -f "$CC_ANDROID" ]]; then
    echo "ERROR: Android clang not found at $CC_ANDROID"
    echo "Available API levels:"
    ls "$NDK_TOOLCHAIN/bin"/aarch64-linux-android*-clang 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "=== Building symthaea-spore for $TARGET (API $API_LEVEL) ==="
echo "  NDK: $ANDROID_NDK_HOME"
echo "  CC:  $CC_ANDROID"

# Configure cross-compilation
export CC_aarch64_linux_android="$CC_ANDROID"
export AR_aarch64_linux_android="$AR_ANDROID"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_ANDROID"

# Build release
cargo build --target "$TARGET" --release -p symthaea-spore

# Report outputs
echo ""
echo "=== Build complete ==="
SO_PATH="target/$TARGET/$PROFILE/libsymthaea_spore.so"
A_PATH="target/$TARGET/$PROFILE/libsymthaea_spore.a"

if [[ -f "$SO_PATH" ]]; then
    SIZE=$(du -h "$SO_PATH" | cut -f1)
    echo "  .so (cdylib):   $SO_PATH  ($SIZE)"
fi
if [[ -f "$A_PATH" ]]; then
    SIZE=$(du -h "$A_PATH" | cut -f1)
    echo "  .a  (staticlib): $A_PATH  ($SIZE)"
fi
