#!/usr/bin/env bash
# Build the JNI C glue layer (spore_jni.c) into libspore_jni.so.
#
# Prerequisites: ANDROID_NDK_HOME set (nix develop provides this).
# This script also builds the Rust .so if not already present.
#
# Output: src/main/jniLibs/arm64-v8a/libspore_jni.so
#         src/main/jniLibs/arm64-v8a/libsymthaea_spore.so (copied from Rust build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPORE_CRATE="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$SPORE_CRATE/../.." && pwd)"
JNILIBS="$SCRIPT_DIR/src/main/jniLibs/arm64-v8a"
TARGET="aarch64-linux-android"
API="${ANDROID_API_LEVEL:-24}"

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    echo "ERROR: ANDROID_NDK_HOME not set. Run: nix develop --impure"
    exit 1
fi

NDK_TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64"
CC="$NDK_TOOLCHAIN/bin/aarch64-linux-android${API}-clang"
AR="$NDK_TOOLCHAIN/bin/llvm-ar"

mkdir -p "$JNILIBS"

# Step 1: Build Rust .so if missing
RUST_SO="$WORKSPACE/target/$TARGET/release/libsymthaea_spore.so"
if [[ ! -f "$RUST_SO" ]]; then
    echo "=== Building Rust libsymthaea_spore.so ==="
    cd "$WORKSPACE"
    export CC_aarch64_linux_android="$CC"
    export AR_aarch64_linux_android="$AR"
    export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC"
    cargo build --target "$TARGET" --release -p symthaea-spore --features native-ffi
fi

# Copy Rust .so
cp "$RUST_SO" "$JNILIBS/libsymthaea_spore.so"
echo "  Copied: libsymthaea_spore.so ($(du -h "$JNILIBS/libsymthaea_spore.so" | cut -f1))"

# Step 2: Compile JNI C glue
echo "=== Building JNI glue (spore_jni.c) ==="
"$CC" -shared -o "$JNILIBS/libspore_jni.so" \
    -I"$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include" \
    "$SCRIPT_DIR/src/main/cpp/spore_jni.c" \
    -L"$JNILIBS" -lsymthaea_spore \
    -llog \
    -fPIC -O2 -Wall -Werror

echo "  Built:  libspore_jni.so ($(du -h "$JNILIBS/libspore_jni.so" | cut -f1))"

echo ""
echo "=== JNI build complete ==="
ls -lh "$JNILIBS/"
