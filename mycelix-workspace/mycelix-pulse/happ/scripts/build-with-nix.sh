#!/usr/bin/env bash
# Mycelix Mail - Build with Nix development shell
set -euo pipefail

echo "🔨 Building Mycelix Mail DNA with Nix development shell..."
echo ""

# Change to repository root
cd "$(dirname "$0")/.."

echo "📦 Entering Nix development shell..."
nix develop --command bash -c '
    set -euo pipefail
    cd dna

    echo "✅ In Nix shell, Rust version: $(rustc --version)"
    echo ""

    # Check wasm32 target
    if ! rustc --print target-list | grep -q wasm32-unknown-unknown; then
        echo "❌ ERROR: wasm32-unknown-unknown target not available"
        exit 1
    fi

    # Add wasm32 target if not installed
    if [ ! -d "$(rustc --print sysroot)/lib/rustlib/wasm32-unknown-unknown" ]; then
        echo "📥 Adding wasm32-unknown-unknown target..."
        rustup target add wasm32-unknown-unknown
    fi

    echo "✅ wasm32-unknown-unknown target ready"
    echo ""

    # Build integrity zome first (dependency)
    echo "📦 Building integrity zome..."
    cd integrity
    cargo build --release --target wasm32-unknown-unknown 2>&1 | head -50
    cd ..
    echo ""

    # Build mail_messages zome
    echo "📦 Building mail_messages zome..."
    cd zomes/mail_messages
    cargo build --release --target wasm32-unknown-unknown 2>&1 | head -50
    cd ../..
    echo ""

    # Build trust_filter zome
    echo "📦 Building trust_filter zome..."
    cd zomes/trust_filter
    cargo build --release --target wasm32-unknown-unknown 2>&1 | head -50
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
'
