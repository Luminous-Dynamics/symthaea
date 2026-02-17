#!/usr/bin/env bash
# Monitor the WASM build progress

echo "🔍 Mycelix Marketplace WASM Build Monitor"
echo "=========================================="
echo ""

# Check if build process is running
if ps aux | grep "nix develop --command ./build-wasm-complete.sh" | grep -v grep > /dev/null; then
    echo "✅ Build process is RUNNING"
    ps aux | grep "nix develop --command ./build-wasm-complete.sh" | grep -v grep | awk '{print "   PID: "$2", CPU: "$3"%, MEM: "$4"%, TIME: "$10}'
    echo ""
else
    echo "⚠️  Build process NOT running (may have completed or failed)"
    echo ""
fi

# Show last 30 lines of build log
echo "📋 Last 30 lines of build log:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -30 /tmp/mycelix_wasm_build_final.log 2>/dev/null || echo "Log file not found"
echo ""

# Check for WASM files
echo "📦 WASM Build Artifacts:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "backend/zomes" ]; then
    find backend/zomes -name "*.wasm" 2>/dev/null | while read f; do
        echo "  ✅ $(basename $f) ($(du -h "$f" | cut -f1))"
    done
else
    echo "  ⏳ No WASM files yet (build in progress)"
fi
echo ""

# Check for final artifacts
echo "🎯 Final Artifacts:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "backend/dna.dna" ]; then
    echo "  ✅ dna.dna ($(du -h backend/dna.dna | cut -f1))"
else
    echo "  ⏳ dna.dna not yet created"
fi

if [ -f "backend/mycelix_marketplace.happ" ]; then
    echo "  ✅ mycelix_marketplace.happ ($(du -h backend/mycelix_marketplace.happ | cut -f1))"
else
    echo "  ⏳ mycelix_marketplace.happ not yet created"
fi
echo ""

echo "📊 Current Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
wasm_count=$(find backend/zomes -name "*.wasm" 2>/dev/null | wc -l)
echo "  WASM files built: $wasm_count / 10"

if [ -f "backend/mycelix_marketplace.happ" ]; then
    echo "  🎉 BUILD COMPLETE!"
elif [ "$wasm_count" = "10" ]; then
    echo "  ⏳ All WASM built, packaging in progress..."
elif [ "$wasm_count" -gt "0" ]; then
    echo "  ⏳ Build in progress ($wasm_count/10 zomes)"
else
    echo "  ⏳ Initializing build environment..."
fi
echo ""
