#!/usr/bin/env bash
# Symthaea HLB Benchmark Runner
#
# Runs all performance benchmarks and compares against README claims
#
# Usage:
#   ./run_benchmarks.sh        # Run all benchmarks
#   ./run_benchmarks.sh quick  # Quick mode (fewer iterations)
#   ./run_benchmarks.sh hdc    # HDC benchmarks only
#   ./run_benchmarks.sh ltc    # LTC benchmarks only

set -e

QUICK_FLAG=""
BENCH_FILTER=""

case "${1:-all}" in
    quick)
        QUICK_FLAG="-- --quick"
        echo "Running benchmarks in quick mode..."
        ;;
    hdc)
        BENCH_FILTER="--bench hdc_benchmark"
        echo "Running HDC benchmarks only..."
        ;;
    ltc)
        BENCH_FILTER="--bench ltc_benchmark"
        echo "Running LTC benchmarks only..."
        ;;
    all)
        echo "Running all benchmarks..."
        ;;
    *)
        echo "Usage: $0 [quick|hdc|ltc|all]"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "Symthaea HLB Performance Benchmarks"
echo "======================================"
echo ""
echo "README.md Claims to Validate:"
echo "  - HDC Encoding:       ~0.05ms (20,000 ops/sec)"
echo "  - HDC Recall:         ~0.10ms (10,000 ops/sec)"
echo "  - LTC Step:           ~0.02ms (50,000 steps/sec)"
echo "  - Consciousness Check: ~0.01ms (100,000 ops/sec)"
echo "  - Full Query:         ~0.50ms (2,000 queries/sec)"
echo ""
echo "======================================"
echo ""

cargo bench $BENCH_FILTER $QUICK_FLAG

echo ""
echo "======================================"
echo "Benchmark complete!"
echo "Results saved to: target/criterion/"
echo "======================================"
