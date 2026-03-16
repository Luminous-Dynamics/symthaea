#!/usr/bin/env bash
# Build Symthaea Spore for iOS (aarch64-apple-ios).
#
# Prerequisites:
#   - macOS with Xcode Command Line Tools
#   - Rust targets: rustup target add aarch64-apple-ios aarch64-apple-ios-sim
#
# Usage:
#   ./build-ios.sh          # Build release .a (staticlib)
#   ./build-ios.sh --check  # Type-check only (works on Linux too)
#
# Output:
#   target/aarch64-apple-ios/release/libsymthaea_spore.a

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="aarch64-apple-ios"
PROFILE="release"

cd "$WORKSPACE_ROOT"

# Ensure target is installed
if ! rustup target list --installed | grep -q "$TARGET"; then
    echo "Installing Rust target $TARGET..."
    rustup target add "$TARGET"
fi

# Check-only mode (works on Linux — no Xcode needed)
if [[ "${1:-}" == "--check" ]]; then
    echo "=== Type-checking symthaea-spore for $TARGET ==="
    RUSTC_WRAPPER="" CARGO_FEATURE_PURE=1 \
        cargo check --target "$TARGET" -p symthaea-spore
    echo "=== Check passed ==="
    exit 0
fi

echo "=== Building symthaea-spore for $TARGET ==="

cargo build --target "$TARGET" --release -p symthaea-spore

# Report outputs
echo ""
echo "=== Build complete ==="
A_PATH="target/$TARGET/$PROFILE/libsymthaea_spore.a"

if [[ -f "$A_PATH" ]]; then
    SIZE=$(du -h "$A_PATH" | cut -f1)
    echo "  .a (staticlib): $A_PATH  ($SIZE)"
fi
