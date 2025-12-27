#!/usr/bin/env bash
# Consciousness Framework Performance Benchmarking
# Date: 2025-12-26

set -e

cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb

echo "🧠 Consciousness Framework Benchmarking"
echo "======================================"
echo ""
echo "Starting benchmarks at $(date)"
echo ""

# Run benchmarks in release mode for accurate performance metrics
echo "📊 Running consciousness benchmarks (release mode for accurate metrics)..."
nix develop --command bash -c "
    cargo bench --bench consciousness_benchmarks -- \
        --warm-up-time 3 \
        --measurement-time 10 \
        --sample-size 50 \
        2>&1 | tee /tmp/consciousness-bench-results.txt
"

echo ""
echo "✅ Benchmarking complete at $(date)"
echo ""
echo "📈 Results summary:"
grep -E "(test |Benchmarking|time:)" /tmp/consciousness-bench-results.txt || echo "See full results in /tmp/consciousness-bench-results.txt"
