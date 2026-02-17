#!/usr/bin/env bash
# Robust WASM Build Script - Continues even if some builds fail

set +e  # Don't exit on error, we want to see all results

echo "🚀 Mycelix Marketplace - Robust WASM Build"
echo "=========================================="
echo ""

cd "$(dirname "$0")/backend"

echo "📍 Working directory: $(pwd)"
echo ""

# Use nix-shell with shell.nix
nix-shell --command '
    set +e  # Continue on errors

    echo "🛠️  Build Environment:"
    echo "  Rust: $(rustc --version)"
    echo "  Cargo: $(cargo --version)"
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1️⃣  Building All Zomes..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    success_count=0
    fail_count=0

    # Build all zome packages
    for pkg in listings_integrity reputation_integrity transactions_integrity arbitration_integrity messaging_integrity listings_coordinator reputation_coordinator transactions_coordinator arbitration_coordinator messaging_coordinator; do
        echo "📦 Building $pkg..."
        if cargo build --release --target wasm32-unknown-unknown -p "$pkg" 2>&1 | tail -3; then
            echo "✅ $pkg built successfully"
            ((success_count++))
        else
            echo "❌ $pkg build failed"
            ((fail_count++))
        fi
        echo ""
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "2️⃣  Copying WASM Files..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Create zome directories
    for zome in listings reputation transactions arbitration messaging; do
        mkdir -p "zomes/${zome}"
    done

    copied_count=0

    # Copy integrity zomes (check actual lib names in target)
    for zome in listings reputation transactions arbitration messaging; do
        wasm_file="target/wasm32-unknown-unknown/release/${zome}_integrity.wasm"
        if [ -f "$wasm_file" ]; then
            echo "  ✅ Copying ${zome}_integrity.wasm"
            cp "$wasm_file" "zomes/${zome}/integrity.wasm"
            ((copied_count++))
        else
            echo "  ⚠️  ${zome}_integrity.wasm not found"
        fi
    done

    # Copy coordinator zomes (lib name is just zome name)
    for zome in listings reputation transactions arbitration messaging; do
        wasm_file="target/wasm32-unknown-unknown/release/${zome}.wasm"
        if [ -f "$wasm_file" ]; then
            echo "  ✅ Copying ${zome}.wasm"
            cp "$wasm_file" "zomes/${zome}/coordinator.wasm"
            ((copied_count++))
        else
            echo "  ⚠️  ${zome}.wasm not found"
        fi
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Build Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Packages built: $success_count / 10"
    echo "WASM files copied: $copied_count / 10"
    echo ""

    if [ $copied_count -eq 10 ]; then
        echo "✅ SUCCESS - All 10 WASM zomes built!"
    elif [ $copied_count -gt 0 ]; then
        echo "⚠️  PARTIAL SUCCESS - $copied_count / 10 zomes built"
    else
        echo "❌ BUILD FAILED - No WASM files created"
    fi

    echo ""
    echo "📦 WASM Files:"
    find zomes -name "*.wasm" -type f 2>/dev/null | sort | while read f; do
        size=$(du -h "$f" | cut -f1)
        echo "  - $f ($size)"
    done

    echo ""
'

echo ""
echo "✅ Build attempt completed!"
echo ""
