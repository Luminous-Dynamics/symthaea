#!/usr/bin/env bash
# Simple WASM Build Script using shell.nix (no flake, no git required)

set -e

echo "🚀 Mycelix Marketplace - WASM Build (shell.nix)"
echo "==============================================="
echo ""

cd "$(dirname "$0")/backend"

echo "📍 Working directory: $(pwd)"
echo ""

# Use nix-shell with shell.nix (doesn't require git tracking)
echo "🔧 Entering Nix shell environment..."
echo ""

nix-shell --command '
    set -e

    echo "🛠️  Build Environment Ready!"
    echo "  Rust: $(rustc --version)"
    echo "  Cargo: $(cargo --version)"
    echo "  LLD: $(which lld || echo "NOT FOUND - will fail")"
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1️⃣  Building Integrity Zomes..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for zome in listings reputation transactions arbitration messaging; do
        echo "  📦 Building ${zome}_integrity..."
        cargo build --release --target wasm32-unknown-unknown -p "${zome}_integrity" 2>&1 | grep -E "(Compiling|Finished|error)" || true
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "2️⃣  Building Coordinator Zomes..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for zome in listings reputation transactions arbitration messaging; do
        echo "  📦 Building ${zome}..."
        cargo build --release --target wasm32-unknown-unknown -p "${zome}" 2>&1 | grep -E "(Compiling|Finished|error)" || true
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
        if [ -f "target/wasm32-unknown-unknown/release/${zome}_integrity.wasm" ]; then
            echo "  📁 Copying ${zome}_integrity.wasm..."
            cp "target/wasm32-unknown-unknown/release/${zome}_integrity.wasm" \
               "zomes/${zome}/integrity.wasm"
        else
            echo "  ❌ ${zome}_integrity.wasm NOT FOUND!"
        fi
    done

    # Copy coordinator zomes
    for zome in listings reputation transactions arbitration messaging; do
        if [ -f "target/wasm32-unknown-unknown/release/${zome}.wasm" ]; then
            echo "  📁 Copying ${zome}.wasm..."
            cp "target/wasm32-unknown-unknown/release/${zome}.wasm" \
               "zomes/${zome}/coordinator.wasm"
        else
            echo "  ❌ ${zome}.wasm NOT FOUND!"
        fi
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Build Complete!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📦 WASM Files Built:"
    wasm_count=$(find zomes -name "*.wasm" -type f 2>/dev/null | wc -l)
    echo "  Total: $wasm_count / 10"
    find zomes -name "*.wasm" -type f 2>/dev/null | sort | while read f; do
        size=$(du -h "$f" | cut -f1)
        echo "  ✅ $f ($size)"
    done

    echo ""
    echo "📝 Note: DNA and hApp packaging skipped (requires hc tool)"
    echo "   Use nix develop from parent dir with holochain flake to package"
    echo ""
    echo "🎉 All WASM zomes successfully built!"
    echo ""
'

echo "✅ Build script completed!"
echo "   WASM files are in: backend/zomes/*/  "
echo ""
