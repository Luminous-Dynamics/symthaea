#!/usr/bin/env bash

# Mycelix Mail - Build Script
# Compiles all zomes and packs the DNA

set -e

echo "🔨 Building Mycelix Mail DNA..."

cd "$(dirname "$0")/.."

# Navigate to DNA directory
cd dna

echo "📦 Building Rust zomes for WASM..."

# Build all zomes
cargo build --release --target wasm32-unknown-unknown

echo "📋 Copying WASM files..."

# Copy compiled WASM files to DNA directory
cp target/wasm32-unknown-unknown/release/mycelix_mail_integrity.wasm integrity.wasm
cp target/wasm32-unknown-unknown/release/mail_messages.wasm mail_messages.wasm
cp target/wasm32-unknown-unknown/release/trust_filter.wasm trust_filter.wasm

echo "📦 Packing DNA..."

# Pack the DNA
hc dna pack .

echo "✅ Build complete!"
echo ""
echo "DNA file: dna/mycelix-mail.dna"
echo ""
echo "Next steps:"
echo "  1. Test in sandbox: hc sandbox create -d dna/mycelix-mail.dna"
echo "  2. Run tests: cargo test"
echo "  3. Start development: See README.md"
