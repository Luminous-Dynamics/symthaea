#!/usr/bin/env bash
# Mycelix Mail - Build with rustup toolchain (not Nix)
set -euo pipefail

echo "🔨 Building Mycelix Mail DNA with rustup toolchain..."
echo ""

cd "$(dirname "$0")/../dna"

RUST_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
echo "Using rustup toolchain: ${RUST_TOOLCHAIN}"
echo "rustc: $(rustup which --toolchain "${RUST_TOOLCHAIN}" rustc)"
echo "Version: $(rustup run "${RUST_TOOLCHAIN}" rustc --version)"
echo "Sysroot: $(rustup run "${RUST_TOOLCHAIN}" rustc --print sysroot)"
echo ""

# Verify wasm32 target exists
if ! rustup target list --installed --toolchain "${RUST_TOOLCHAIN}" | grep -q wasm32-unknown-unknown; then
    echo "❌ ERROR: wasm32-unknown-unknown target not found!"
    echo "   Install with: rustup target add --toolchain ${RUST_TOOLCHAIN} wasm32-unknown-unknown"
    exit 1
fi

echo "✅ wasm32-unknown-unknown target found"
echo ""

# Build integrity zome first (dependency)
echo "📦 Building integrity zome..."
cd integrity
rustup run "${RUST_TOOLCHAIN}" cargo build --release --target wasm32-unknown-unknown
cd ..
echo ""

# Build mail_messages zome
echo "📦 Building mail_messages zome..."
cd zomes/mail_messages
rustup run "${RUST_TOOLCHAIN}" cargo build --release --target wasm32-unknown-unknown
cd ../..
echo ""

# Build trust_filter zome
echo "📦 Building trust_filter zome..."
cd zomes/trust_filter
rustup run "${RUST_TOOLCHAIN}" cargo build --release --target wasm32-unknown-unknown
cd ../..
echo ""

# Copy WASM files
echo "📋 Copying WASM files..."
cp integrity/target/wasm32-unknown-unknown/release/mycelix_mail_integrity.wasm integrity.wasm
cp zomes/mail_messages/target/wasm32-unknown-unknown/release/mail_messages.wasm mail_messages.wasm
cp zomes/trust_filter/target/wasm32-unknown-unknown/release/trust_filter.wasm trust_filter.wasm

# Check files exist and show sizes
echo ""
echo "✅ Build complete!"
echo ""
echo "WASM files:"
ls -lh *.wasm
echo ""

# Pack DNA if hc is available
if command -v hc &> /dev/null; then
    echo "📦 Packing DNA..."
    hc dna pack .
    echo ""
    echo "✅ DNA packed: mycelix-mail.dna"
    ls -lh *.dna
else
    echo "⚠️  hc command not found, skipping DNA packing"
    echo "   Install with: nix-shell -p holochain"
fi

echo ""
echo "🎉 Success! Ready to test."
