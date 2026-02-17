#!/usr/bin/env bash
# Check if WASM build is complete and show final results

set -e

echo "🏁 Mycelix Marketplace WASM Build - Completion Check"
echo "======================================================"
echo ""

cd "$(dirname "$0")"

# Check if build process is still running
if ps aux | grep "nix develop --command ./build-wasm-complete.sh" | grep -v grep > /dev/null; then
    echo "⏳ Build is STILL RUNNING"
    echo ""
    ps aux | grep "nix develop --command ./build-wasm-complete.sh" | grep -v grep | awk '{print "   PID: "$2", CPU: "$3"%, TIME: "$10}'
    echo ""
    echo "Run ./monitor-build.sh to see detailed progress"
    echo ""
    exit 0
fi

echo "✅ Build process has FINISHED"
echo ""

# Check for WASM files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 WASM Build Artifacts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

wasm_files=$(find backend/zomes -name "*.wasm" 2>/dev/null | wc -l)

if [ "$wasm_files" -eq 10 ]; then
    echo "✅ All 10 WASM files built successfully!"
    echo ""
    find backend/zomes -name "*.wasm" 2>/dev/null | sort | while read f; do
        size=$(du -h "$f" | cut -f1)
        name=$(basename $(dirname $f))/$(basename $f)
        echo "  ✅ $name ($size)"
    done
else
    echo "❌ Expected 10 WASM files, found: $wasm_files"
    echo ""
    if [ "$wasm_files" -gt 0 ]; then
        echo "Built files:"
        find backend/zomes -name "*.wasm" | while read f; do
            echo "  - $f"
        done
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Final Artifacts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check DNA
if [ -f "backend/dna.dna" ]; then
    size=$(du -h backend/dna.dna | cut -f1)
    echo "✅ dna.dna ($size)"
else
    echo "❌ dna.dna NOT FOUND"
fi

# Check hApp
if [ -f "backend/mycelix_marketplace.happ" ]; then
    size=$(du -h backend/mycelix_marketplace.happ | cut -f1)
    echo "✅ mycelix_marketplace.happ ($size)"
else
    echo "❌ mycelix_marketplace.happ NOT FOUND"
fi

echo ""

# Overall status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Overall Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$wasm_files" -eq 10 ] && [ -f "backend/dna.dna" ] && [ -f "backend/mycelix_marketplace.happ" ]; then
    echo "🎉 BUILD COMPLETE - ALL ARTIFACTS CREATED!"
    echo ""
    echo "✅ 10/10 WASM zomes built"
    echo "✅ DNA bundle packaged"
    echo "✅ hApp bundle created"
    echo ""
    echo "🚀 Next Steps:"
    echo "  1. Review build log: tail /tmp/mycelix_wasm_build_final.log"
    echo "  2. Test with conductor: holochain -c backend/conductor-config.yaml"
    echo "  3. Deploy to network"
    echo ""
elif [ "$wasm_files" -eq 10 ]; then
    echo "⚠️  PARTIAL SUCCESS - WASM files built but packaging failed"
    echo ""
    echo "✅ 10/10 WASM zomes built"
    echo "❌ DNA or hApp packaging incomplete"
    echo ""
    echo "Check build log: tail -100 /tmp/mycelix_wasm_build_final.log"
else
    echo "❌ BUILD FAILED OR INCOMPLETE"
    echo ""
    echo "WASM files: $wasm_files / 10"
    echo "DNA: $([ -f backend/dna.dna ] && echo '✅' || echo '❌')"
    echo "hApp: $([ -f backend/mycelix_marketplace.happ ] && echo '✅' || echo '❌')"
    echo ""
    echo "📋 Check build log for errors:"
    echo "  tail -200 /tmp/mycelix_wasm_build_final.log"
fi

echo ""
