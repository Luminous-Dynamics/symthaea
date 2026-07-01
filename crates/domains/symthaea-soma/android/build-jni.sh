#!/usr/bin/env bash
# Build the JNI C glue layer (soma_jni.c) into libsoma_jni.so.
#
# Prerequisites: ANDROID_NDK_HOME set (nix develop provides this).
# This script also builds the Rust .so if not already present.
#
# Output: src/main/jniLibs/arm64-v8a/libsoma_jni.so
#         src/main/jniLibs/arm64-v8a/libsymthaea_soma.so (copied from Rust build)
#
# NOTE: JNI symbol names are soma_engine_* (renamed from spore_engine_* in Phase 4).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOMA_CRATE="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$SOMA_CRATE/../.." && pwd)"
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

# Step 0: Build LiteRT shim FIRST (Rust .so links against it)
echo "=== Building LiteRT shim (litert_shim.cpp) ==="
LITERT_SRC="$SOMA_CRATE/native"
LITERT_FLAGS=""

if [[ -d "${LITERT_LM_SDK:-}" ]]; then
    # SDK provides c/engine.h and prebuilt libs
    LITERT_FLAGS="-DLITERT_LM_AVAILABLE -I$LITERT_LM_SDK -L$LITERT_LM_SDK/prebuilt/android/arm64-v8a -llitert_lm"
    echo "  LiteRT-LM SDK found at $LITERT_LM_SDK — building with full inference"
else
    echo "  LiteRT-LM SDK not found — building stub (set LITERT_LM_SDK to enable)"
fi

"${NDK_TOOLCHAIN}/bin/aarch64-linux-android${API}-clang++" -shared -o "$JNILIBS/liblitert_shim.so" \
    "$LITERT_SRC/litert_shim.cpp" \
    -std=c++17 -fPIC -O2 -Wall \
    -static-libstdc++ \
    -Wl,-soname,liblitert_shim.so \
    -Wl,-z,max-page-size=16384 \
    $LITERT_FLAGS

echo "  Built:  liblitert_shim.so ($(du -h "$JNILIBS/liblitert_shim.so" | cut -f1))"

# Step 1: Build Rust .so (links against liblitert_shim.so)
RUST_SO="$WORKSPACE/target/$TARGET/release/libsymthaea_soma.so"
echo "=== Building Rust libsymthaea_soma.so ==="
cd "$WORKSPACE"
export CC_aarch64_linux_android="$CC"
export AR_aarch64_linux_android="$AR"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC"
# 16KB page alignment + link against liblitert_shim.so for LiteRT FFI symbols
export RUSTFLAGS="-C link-arg=-z -C link-arg=max-page-size=16384 -C link-arg=-L$JNILIBS -C link-arg=-llitert_shim"
# Note: broca-full excluded for Android (pulls candle-cuda which needs desktop CUDA).
# On-device language uses LiteRT (gemma4:e2b) instead of native Broca.
cargo build --target "$TARGET" --release -p symthaea-soma --features native-ffi,screen-vision,litert,prism-search

# Copy Rust .so (soma_jni.c links against -lsymthaea_soma)
cp "$RUST_SO" "$JNILIBS/libsymthaea_soma.so"
echo "  Copied: libsymthaea_soma.so ($(du -h "$JNILIBS/libsymthaea_soma.so" | cut -f1))"

# Step 2: Compile JNI C glue
echo "=== Building JNI glue (soma_jni.c) ==="
"$CC" -shared -o "$JNILIBS/libsoma_jni.so" \
    -I"$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include" \
    "$SCRIPT_DIR/src/main/cpp/soma_jni.c" \
    -L"$JNILIBS" -lsymthaea_soma \
    -llog \
    -fPIC -O2 -Wall -Werror \
    -Wl,-z,max-page-size=16384

echo "  Built:  libsoma_jni.so ($(du -h "$JNILIBS/libsoma_jni.so" | cut -f1))"

echo ""
echo "=== JNI build complete ==="
ls -lh "$JNILIBS/"
