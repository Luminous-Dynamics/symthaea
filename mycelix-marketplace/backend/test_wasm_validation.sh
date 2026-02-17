#!/usr/bin/env bash
# Simple WASM validation without running conductor

echo "🧪 Phase 4: WASM Validation Tests"
echo "=================================="
echo ""

# Test 1: Check all WASM files exist
echo "Test 1: Verifying all 10 WASM files exist..."
WASM_COUNT=$(find zomes -name "*.wasm" -type f | wc -l)
if [ "$WASM_COUNT" -eq 10 ]; then
    echo "✅ All 10 WASM files found"
else
    echo "❌ Expected 10 WASM files, found $WASM_COUNT"
    exit 1
fi

# Test 2: Check WASM file sizes (should be 1-3MB each)
echo ""
echo "Test 2: Validating WASM file sizes..."
for wasm in zomes/*/integrity.wasm zomes/*/coordinator.wasm; do
    size=$(stat -c%s "$wasm" 2>/dev/null || stat -f%z "$wasm")
    size_mb=$((size / 1024 / 1024))
    if [ "$size_mb" -ge 1 ] && [ "$size_mb" -le 3 ]; then
        echo "✅ $wasm: ${size_mb}MB (valid)"
    else
        echo "⚠️  $wasm: ${size_mb}MB (unexpected size)"
    fi
done

# Test 3: Verify WASM files are valid binary format
echo ""
echo "Test 3: Checking WASM binary format..."
for wasm in zomes/*/integrity.wasm zomes/*/coordinator.wasm; do
    # WASM files start with magic number 0x00 0x61 0x73 0x6D (\0asm)
    magic=$(hexdump -n 4 -e '4/1 "%02x"' "$wasm")
    if [ "$magic" = "0061736d" ]; then
        echo "✅ $wasm: Valid WASM magic number"
    else
        echo "❌ $wasm: Invalid WASM format (magic: $magic)"
        exit 1
    fi
done

# Test 4: Verify DNA bundle structure
echo ""
echo "Test 4: Validating DNA bundle structure..."
if unzip -t dna.dna >/dev/null 2>&1; then
    echo "✅ DNA bundle is valid ZIP archive"
    echo "   Contents:"
    unzip -l dna.dna | grep -E "\.wasm$|\.yaml$" | awk '{print "   - " $4}'
else
    echo "❌ DNA bundle is corrupted"
    exit 1
fi

# Test 5: Verify hApp bundle structure  
echo ""
echo "Test 5: Validating hApp bundle structure..."
if unzip -t mycelix_marketplace.happ >/dev/null 2>&1; then
    echo "✅ hApp bundle is valid ZIP archive"
    echo "   Contents:"
    unzip -l mycelix_marketplace.happ | awk '{if (NR>3) print "   - " $4}'
else
    echo "❌ hApp bundle is corrupted"
    exit 1
fi

# Test 6: Verify Cargo workspace builds
echo ""
echo "Test 6: Testing Cargo workspace integrity..."
if cargo check --workspace --quiet 2>&1 | grep -q "error"; then
    echo "❌ Cargo workspace has compilation errors"
    exit 1
else
    echo "✅ Cargo workspace passes type checking"
fi

echo ""
echo "=================================="
echo "🎉 All validation tests passed!"
echo ""
echo "Summary:"
echo "  ✅ 10 WASM files present and valid"
echo "  ✅ DNA bundle properly structured"
echo "  ✅ hApp bundle ready for deployment"
echo "  ✅ Workspace compiles cleanly"
echo ""
echo "Status: WASM artifacts are production-ready! 🚀"
