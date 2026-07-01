#!/usr/bin/env bash
# Build sovereign-scan for all platforms
# Requires: cross (cargo install cross)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VERSION="0.1.0"
OUT="releases"
mkdir -p "$OUT"

echo "Building sovereign-scan v${VERSION} for all platforms..."

# Linux x86_64 (native, fastest)
echo "[1/4] Linux x86_64..."
cargo build --release
cp target/release/sovereign-scan "$OUT/sovereign-scan-x86_64-unknown-linux-gnu"

# Linux aarch64
if command -v cross &>/dev/null; then
    echo "[2/4] Linux aarch64..."
    cross build --release --target aarch64-unknown-linux-gnu
    cp target/aarch64-unknown-linux-gnu/release/sovereign-scan "$OUT/sovereign-scan-aarch64-unknown-linux-gnu"
else
    echo "[2/4] SKIP Linux aarch64 (install cross: cargo install cross)"
fi

# macOS x86_64
if command -v cross &>/dev/null; then
    echo "[3/4] macOS x86_64..."
    cross build --release --target x86_64-apple-darwin 2>/dev/null || echo "  SKIP (needs macOS SDK)"
fi

# Windows x86_64
if rustup target list --installed | grep -q x86_64-pc-windows-gnu; then
    echo "[4/4] Windows x86_64..."
    cargo build --release --target x86_64-pc-windows-gnu
    cp target/x86_64-pc-windows-gnu/release/sovereign-scan.exe "$OUT/sovereign-scan-x86_64-pc-windows-gnu.exe"
elif command -v cross &>/dev/null; then
    echo "[4/4] Windows x86_64 (via cross)..."
    cross build --release --target x86_64-pc-windows-gnu
    cp target/x86_64-pc-windows-gnu/release/sovereign-scan.exe "$OUT/sovereign-scan-x86_64-pc-windows-gnu.exe"
else
    echo "[4/4] SKIP Windows (add target: rustup target add x86_64-pc-windows-gnu)"
fi

echo ""
echo "Built releases:"
ls -lh "$OUT/"
echo ""
echo "Upload to: gh release create v${VERSION} $OUT/* --title 'Sovereign Scan v${VERSION}'"
