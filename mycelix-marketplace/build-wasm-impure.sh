#!/usr/bin/env bash
# WASM Build Script - Impure Nix Version
# Works even if backend/ directory is not tracked by git

set -e

echo "🚀 Mycelix Marketplace - WASM Build (Impure Nix)"
echo "================================================"
echo ""

cd "$(dirname "$0")"

echo "📍 Project root: $(pwd)"
echo ""

# Use --impure flag to bypass git tracking requirement
echo "🔧 Entering Nix development environment (--impure mode)..."
nix develop ./backend --impure --command bash -c '
    set -e

    cd backend

    echo "📍 Working directory: $(pwd)"
    echo ""
    echo "🛠️  Build environment:"
    echo "  Rust: $(rustc --version)"
    echo "  Cargo: $(cargo --version)"
    echo "  Holochain: $(holochain --version 2>/dev/null || echo "N/A")"
    echo ""

    # Ensure wasm32 target is installed
    if command -v rustup &>/dev/null; then
        if ! rustup target list | grep -q "wasm32-unknown-unknown (installed)"; then
            echo "📦 Installing wasm32-unknown-unknown target..."
            rustup target add wasm32-unknown-unknown
        else
            echo "✅ wasm32-unknown-unknown target already installed"
        fi
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1️⃣  Building Integrity Zomes..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for zome in listings reputation transactions arbitration messaging; do
        echo "  📦 Building ${zome}_integrity..."
        cargo build --release --target wasm32-unknown-unknown -p "${zome}_integrity"
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "2️⃣  Building Coordinator Zomes..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for zome in listings reputation transactions arbitration messaging; do
        echo "  📦 Building ${zome}..."
        cargo build --release --target wasm32-unknown-unknown -p "${zome}"
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "3️⃣  Copying WASM Files..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Create zome directories
    for zome in listings reputation transactions arbitration messaging; do
        mkdir -p "zomes/${zome}"
    done

    # Copy integrity zomes
    for zome in listings reputation transactions arbitration messaging; do
        echo "  📁 Copying ${zome}_integrity.wasm..."
        cp "target/wasm32-unknown-unknown/release/${zome}_integrity.wasm" \
           "zomes/${zome}/integrity.wasm"
    done

    # Copy coordinator zomes
    for zome in listings reputation transactions arbitration messaging; do
        echo "  📁 Copying ${zome}.wasm..."
        cp "target/wasm32-unknown-unknown/release/${zome}.wasm" \
           "zomes/${zome}/coordinator.wasm"
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "4️⃣  Packaging DNA..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if command -v hc &>/dev/null; then
        hc dna pack .
    else
        echo "⚠️  hc command not found, skipping DNA packaging"
        echo "   (Install via nix develop to package DNA and hApp)"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "5️⃣  Packaging hApp..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if command -v hc &>/dev/null; then
        hc app pack .
    else
        echo "⚠️  hc command not found, skipping hApp packaging"
        echo "   (Install via nix develop to package DNA and hApp)"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Build Complete!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📦 WASM Files Built:"
    find zomes -name "*.wasm" -type f -exec ls -lh {} \; | awk "{print \"  - \"\$9\" (\"\$5\")\"}"
    echo ""

    if [ -f "dna.dna" ]; then
        echo "📦 DNA Package:"
        ls -lh dna.dna | awk "{print \"  - dna.dna (\"\$5\")\"}"
    fi

    if [ -f "mycelix_marketplace.happ" ]; then
        echo "📦 hApp Package:"
        ls -lh mycelix_marketplace.happ | awk "{print \"  - mycelix_marketplace.happ (\"\$5\")\"}"
    fi

    echo ""
    echo "🎉 All 10 zomes successfully built!"
    echo ""
'

echo ""
echo "✅ Build script completed!"
echo ""
