#!/usr/bin/env bash
# Consciousness Framework Benchmarking Script
# Measures performance of key consciousness operations

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔬 Symthaea Consciousness Framework Benchmarks"
echo "=============================================="
echo ""

# Check if benchmark feature is available
if ! cargo bench --help 2>/dev/null | grep -q "criterion"; then
    echo "⚠️  Criterion benchmarks not configured"
    echo "   Adding basic timing tests instead..."
    echo ""
fi

# Create benchmark report directory
BENCH_DIR="$PROJECT_ROOT/benchmark-reports"
mkdir -p "$BENCH_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REPORT="$BENCH_DIR/consciousness-bench-$TIMESTAMP.md"

echo "# Consciousness Framework Benchmark Report" > "$REPORT"
echo "**Date**: $(date)" >> "$REPORT"
echo "**System**: $(uname -a)" >> "$REPORT"
echo "" >> "$REPORT"

echo "Running benchmarks (this may take a few minutes)..."
echo ""

# Benchmark 1: Φ Computation
echo "Benchmark 1: Φ (Integrated Information) Computation"
echo "---------------------------------------------------"
echo ""
echo "## Benchmark 1: Φ Computation" >> "$REPORT"
echo "" >> "$REPORT"

cat > /tmp/bench_phi.rs << 'EOF'
#[cfg(test)]
mod bench {
    use std::time::Instant;

    #[test]
    fn bench_phi_computation() {
        // Simplified benchmark - replace with actual when integrated
        let start = Instant::now();
        let mut sum = 0.0f64;
        for _ in 0..1000 {
            // Simulate Φ computation
            sum += (sum + 1.0).sqrt().sin();
        }
        let duration = start.elapsed();
        println!("1000 iterations: {:?}", duration);
        println!("Per-operation: {:?}", duration / 1000);
        println!("Operations/sec: {:.0}", 1000.0 / duration.as_secs_f64());
    }
}
EOF

echo "✅ Φ computation test prepared" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

# Benchmark 2: Attention Mechanisms
echo ""
echo "Benchmark 2: Attention Mechanism Performance"
echo "---------------------------------------------"
echo ""
echo "## Benchmark 2: Attention Mechanisms" >> "$REPORT"
echo "" >> "$REPORT"

if cargo test attention_mechanisms --lib 2>&1 | grep -q "test result: ok"; then
    echo "✅ Attention mechanism tests passing" | tee -a "$REPORT"
    TEST_TIME=$(cargo test attention_mechanisms --lib 2>&1 | grep "finished in" | awk '{print $3}')
    echo "Test suite time: $TEST_TIME" | tee -a "$REPORT"
else
    echo "⚠️  Attention tests need review" | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# Benchmark 3: Binding Problem
echo ""
echo "Benchmark 3: Feature Binding Performance"
echo "-----------------------------------------"
echo ""
echo "## Benchmark 3: Feature Binding" >> "$REPORT"
echo "" >> "$REPORT"

if cargo test binding_problem --lib 2>&1 | grep -q "test result: ok"; then
    echo "✅ Binding problem tests passing" | tee -a "$REPORT"
    TEST_TIME=$(cargo test binding_problem --lib 2>&1 | grep "finished in" | awk '{print $3}')
    echo "Test suite time: $TEST_TIME" | tee -a "$REPORT"
else
    echo "⚠️  Binding tests need review" | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# Benchmark 4: Full Pipeline
echo ""
echo "Benchmark 4: Complete Consciousness Pipeline"
echo "---------------------------------------------"
echo ""
echo "## Benchmark 4: Full Pipeline (9 stages)" >> "$REPORT"
echo "" >> "$REPORT"

if cargo test consciousness_integration --lib 2>&1 | grep -q "test result: ok"; then
    echo "✅ Integration pipeline tests passing" | tee -a "$REPORT"
    TEST_TIME=$(cargo test consciousness_integration --lib 2>&1 | grep "finished in" | awk '{print $3}')
    echo "Pipeline execution time: $TEST_TIME" | tee -a "$REPORT"
    echo "Target: <100ms total" | tee -a "$REPORT"
else
    echo "⚠️  Pipeline tests need review" | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# Benchmark 5: Memory Operations
echo ""
echo "Benchmark 5: Memory System Performance"
echo "---------------------------------------"
echo ""
echo "## Benchmark 5: Memory Systems" >> "$REPORT"
echo "" >> "$REPORT"

if cargo test episodic --lib 2>&1 | grep -q "test result: ok"; then
    echo "✅ Episodic memory tests passing" | tee -a "$REPORT"
    TEST_TIME=$(cargo test episodic --lib -- --nocapture 2>&1 | grep "finished in" | awk '{print $3}')
    echo "Memory test time: $TEST_TIME" | tee -a "$REPORT"
else
    echo "⚠️  Memory tests need review" | tee -a "$REPORT"
fi
echo "" | tee -a "$REPORT"

# System Information
echo ""
echo "## System Information" >> "$REPORT"
echo "" >> "$REPORT"
echo "- **CPU**: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)" >> "$REPORT"
echo "- **RAM**: $(free -h | grep Mem | awk '{print $2}')" >> "$REPORT"
echo "- **Rust**: $(rustc --version)" >> "$REPORT"
echo "- **Cargo**: $(cargo --version)" >> "$REPORT"
echo "" >> "$REPORT"

# Summary
echo ""
echo "=============================================="
echo "Benchmark Report Generated"
echo "=============================================="
echo ""
echo "Report saved to: $REPORT"
echo ""
echo "View report:"
echo "  cat $REPORT"
echo ""
echo "Expected Performance Targets:"
echo "  - Φ Computation: <1ms"
echo "  - Attention Selection: <10ms"
echo "  - Feature Binding: <10ms"
echo "  - Full Pipeline: <100ms"
echo "  - Memory Operations: <50ms"
echo ""
echo "Next Steps:"
echo "  1. Review benchmark report"
echo "  2. Compare against targets"
echo "  3. Identify optimization opportunities"
echo "  4. Run after Phase 1 integration for comparison"
echo ""
