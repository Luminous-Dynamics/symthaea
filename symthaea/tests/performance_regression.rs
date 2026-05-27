// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Performance Regression Tests
//!
//! These tests validate that critical operations maintain expected performance
//! characteristics. Based on documented baselines in benches/BENCHMARK_GUIDE.md:
//!
//! | Operation | Expected Time (Release) |
//! |-----------|-------------------------|
//! | BinaryHV bind | ~50 ns |
//! | BinaryHV similarity | ~100 ns |
//!
//! **NOTE**: Debug builds are ~1000x slower than release builds!
//! These tests use conservative thresholds for debug mode compatibility.
//! For accurate performance measurement, use:
//!   `cargo bench` (Criterion benchmarks in release mode)
//!
//! Run with: cargo test --test performance_regression -- --nocapture
//! For release mode: cargo test --test performance_regression --release -- --nocapture

use std::hint::black_box;
use std::time::Instant;
use symthaea::hdc::binary_hv::BinaryHV;

/// Number of iterations for timing (lower for debug mode)
const ITERATIONS: usize = 1_000;

/// Warm-up iterations
const WARMUP: usize = 100;

// ============================================================================
// BinaryHV OPERATIONS - Debug-mode compatible thresholds (sanity checks)
// For accurate perf measurement, use `cargo bench`
// ============================================================================

/// BinaryHV bind: release ~50ns, debug ~500µs, threshold: 10ms (catastrophic regression check)
#[test]
fn regression_hv16_bind() {
    println!("\n=== BinaryHV Bind Performance Regression ===\n");

    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);

    // Warm-up
    for _ in 0..WARMUP {
        black_box(black_box(&a).bind(black_box(&b)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(black_box(&a).bind(black_box(&b)));
    }
    let elapsed = start.elapsed();
    let us_per_op = elapsed.as_micros() as f64 / ITERATIONS as f64;

    println!(
        "  BinaryHV.bind(): {:.1} µs/op (release: ~0.05µs, debug: ~500µs)",
        us_per_op
    );

    // 10ms threshold - only fails for catastrophic regression
    assert!(
        us_per_op < 10_000.0,
        "CATASTROPHIC REGRESSION: BinaryHV.bind() took {:.1} µs (threshold: 10000 µs)",
        us_per_op
    );
}

/// BinaryHV similarity: release ~100ns, debug ~500µs, threshold: 10ms
#[test]
fn regression_hv16_similarity() {
    println!("\n=== BinaryHV Similarity Performance Regression ===\n");

    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);

    // Warm-up
    for _ in 0..WARMUP {
        black_box(black_box(&a).similarity(black_box(&b)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(black_box(&a).similarity(black_box(&b)));
    }
    let elapsed = start.elapsed();
    let us_per_op = elapsed.as_micros() as f64 / ITERATIONS as f64;

    println!(
        "  BinaryHV.similarity(): {:.1} µs/op (release: ~0.1µs, debug: ~500µs)",
        us_per_op
    );

    // 10ms threshold - only fails for catastrophic regression
    assert!(
        us_per_op < 10_000.0,
        "CATASTROPHIC REGRESSION: BinaryHV.similarity() took {:.1} µs (threshold: 10000 µs)",
        us_per_op
    );
}

// ============================================================================
// SIMD OPERATIONS - Debug-mode compatible thresholds
// ============================================================================

/// BinaryHV bind: threshold 10ms (catastrophic regression check)
#[test]
fn regression_simd_bind() {
    println!("\n=== BinaryHV Bind Performance Regression ===\n");

    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);

    // Warm-up
    for _ in 0..WARMUP {
        black_box(black_box(&a).bind(black_box(&b)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(black_box(&a).bind(black_box(&b)));
    }
    let elapsed = start.elapsed();
    let us_per_op = elapsed.as_micros() as f64 / ITERATIONS as f64;

    println!("  BinaryHV.bind(): {:.1} µs/op", us_per_op);

    assert!(
        us_per_op < 10_000.0,
        "CATASTROPHIC REGRESSION: BinaryHV.bind() took {:.1} µs (threshold: 10000 µs)",
        us_per_op
    );
}

/// BinaryHV similarity: threshold 10ms (catastrophic regression check)
#[test]
fn regression_simd_similarity() {
    println!("\n=== BinaryHV Similarity Performance Regression ===\n");

    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);

    // Warm-up
    for _ in 0..WARMUP {
        black_box(black_box(&a).similarity(black_box(&b)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(black_box(&a).similarity(black_box(&b)));
    }
    let elapsed = start.elapsed();
    let us_per_op = elapsed.as_micros() as f64 / ITERATIONS as f64;

    println!("  BinaryHV.similarity(): {:.1} µs/op", us_per_op);

    assert!(
        us_per_op < 10_000.0,
        "CATASTROPHIC REGRESSION: BinaryHV.similarity() took {:.1} µs (threshold: 10000 µs)",
        us_per_op
    );
}

// ============================================================================
// BUNDLE OPERATIONS - Multi-vector aggregation (debug-mode compatible)
// ============================================================================

/// BinaryHV bundle of 10 vectors: threshold 1 second (catastrophic regression check)
#[test]
fn regression_hv16_bundle() {
    println!("\n=== BinaryHV Bundle Performance Regression ===\n");

    let vectors: Vec<BinaryHV> = (0..10).map(|i| BinaryHV::random(i + 100)).collect();

    // Warm-up
    for _ in 0..10 {
        black_box(BinaryHV::bundle(black_box(&vectors)));
    }

    // Benchmark (fewer iterations for debug mode)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(BinaryHV::bundle(black_box(&vectors)));
    }
    let elapsed = start.elapsed();
    let ms_per_op = elapsed.as_millis() as f64 / iterations as f64;

    println!("  BinaryHV::bundle(10): {:.2} ms/op", ms_per_op);

    // 1 second threshold - only fails for catastrophic regression
    assert!(
        ms_per_op < 1000.0,
        "CATASTROPHIC REGRESSION: BinaryHV::bundle(10) took {:.2} ms (threshold: 1000 ms)",
        ms_per_op
    );
}

// ============================================================================
// LSH SEARCH - Debug-mode compatible thresholds
// ============================================================================

/// LSH index creation: threshold 5 seconds (catastrophic regression check)
#[test]
fn regression_lsh_index_creation() {
    use symthaea::hdc::lsh_simhash::{SimHashConfig, SimHashIndex};

    println!("\n=== LSH Index Creation Performance Regression ===\n");

    // Create 500 vectors (above LSH threshold)
    let vectors: Vec<BinaryHV> = (0..500).map(|i| BinaryHV::random(i as u64)).collect();

    // Warm-up
    let mut index = SimHashIndex::new(SimHashConfig::fast());
    index.insert_batch(&vectors);

    // Benchmark
    let iterations = 3;
    let start = Instant::now();
    for _ in 0..iterations {
        let mut index = SimHashIndex::new(SimHashConfig::fast());
        index.insert_batch(black_box(&vectors));
        black_box(&index);
    }
    let elapsed = start.elapsed();
    let ms_per_op = elapsed.as_millis() as f64 / iterations as f64;

    println!("  SimHashIndex::insert_batch(500): {:.1} ms/op", ms_per_op);

    // 5 second threshold - only fails for catastrophic regression
    assert!(
        ms_per_op < 5000.0,
        "CATASTROPHIC REGRESSION: SimHashIndex::insert_batch(500) took {:.1} ms (threshold: 5000 ms)",
        ms_per_op
    );
}

/// LSH query: threshold 1 second (catastrophic regression check)
#[test]
fn regression_lsh_query_performance() {
    use symthaea::hdc::lsh_simhash::{SimHashConfig, SimHashIndex};

    println!("\n=== LSH Query Performance Regression ===\n");

    // Create 1000 vectors
    let vectors: Vec<BinaryHV> = (0..1000).map(|i| BinaryHV::random(i as u64)).collect();
    let query = BinaryHV::random(9999);

    let mut index = SimHashIndex::new(SimHashConfig::balanced());
    index.insert_batch(&vectors);

    // Warm-up
    for _ in 0..3 {
        black_box(index.query_approximate(black_box(&query), 5, black_box(&vectors)));
    }

    // Benchmark LSH query (fewer iterations for debug mode)
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(index.query_approximate(black_box(&query), 5, black_box(&vectors)));
    }
    let elapsed = start.elapsed();
    let ms_per_op = elapsed.as_millis() as f64 / iterations as f64;

    println!("  LSH query (1000 vectors, top-5): {:.1} ms/op", ms_per_op);

    // 1 second threshold - only fails for catastrophic regression
    assert!(
        ms_per_op < 1000.0,
        "CATASTROPHIC REGRESSION: LSH query took {:.1} ms (threshold: 1000 ms)",
        ms_per_op
    );
}

// ============================================================================
// MEMORY CHARACTERISTICS
// ============================================================================

/// BinaryHV should be exactly 2048 bytes (16,384 bits / 8)
#[test]
fn regression_hv16_size() {
    println!("\n=== BinaryHV Memory Size Regression ===\n");

    let size = std::mem::size_of::<BinaryHV>();
    println!("  BinaryHV size: {} bytes (expected: 2048)", size);

    assert_eq!(
        size, 2048,
        "BinaryHV size changed: {} bytes (expected 2048)",
        size
    );
}

/// BinaryHV should be same size as BinaryHV (aligned for SIMD)
#[test]
fn regression_simd_hv16_size() {
    println!("\n=== BinaryHV Memory Size Regression ===\n");

    let size = std::mem::size_of::<BinaryHV>();
    println!("  BinaryHV size: {} bytes (expected: 2048)", size);

    assert_eq!(
        size, 2048,
        "BinaryHV size changed: {} bytes (expected 2048)",
        size
    );
}

// ============================================================================
// COGNITIVE CYCLE TIMING - Regression guards for the main cognitive loop
// ============================================================================

/// Maximum acceptable cycle time (microseconds).
/// Release baseline (2026-02-27): ~32ms warm steady-state.
/// Debug builds are ~10x slower (~300-400ms), so we use a generous threshold
/// that catches catastrophic regressions while passing in both modes.
/// For accurate perf measurement, use `cargo bench`.
const MAX_CYCLE_US: u64 = if cfg!(debug_assertions) {
    2_000_000 // 2s debug mode — catastrophic regression guard only
} else {
    50_000 // 50ms release mode — meaningful regression guard
};

/// Cognitive cycle steady-state: release ~32ms, debug ~300-400ms.
///
/// Baseline (2026-02-27): ~32ms warm steady-state (measured via criterion).
/// Debug builds are ~10x slower. Thresholds are mode-adaptive.
#[test]
fn regression_cognitive_cycle_steady_state() {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .expect("Failed to create CognitiveLoopService");

    // Warm up to reach steady state
    for _ in 0..20 {
        let _ = service.cycle("warm up input for regression guard");
    }

    let inputs = [
        "the sun rises in the east",
        "cause and effect are linked",
        "patterns emerge from chaos",
        "consciousness arises from integration",
        "temporal prediction drives learning",
    ];

    let measure_cycles: usize = if cfg!(debug_assertions) { 20 } else { 100 };
    let start = Instant::now();
    for i in 0..measure_cycles {
        let _ = service.cycle(inputs[i % inputs.len()]);
    }
    let total_us = start.elapsed().as_micros() as u64;
    let avg_us = total_us / measure_cycles as u64;

    println!(
        "\n=== Cognitive Cycle Regression ===\n  avg cycle = {}us ({:.1}Hz), budget = {}us ({:.1}Hz)\n",
        avg_us,
        1_000_000.0 / avg_us as f64,
        MAX_CYCLE_US,
        1_000_000.0 / MAX_CYCLE_US as f64,
    );

    assert!(
        avg_us < MAX_CYCLE_US,
        "PERFORMANCE REGRESSION: avg cycle time {}us exceeds budget {}us ({:.1}Hz < {:.1}Hz)",
        avg_us,
        MAX_CYCLE_US,
        1_000_000.0 / avg_us as f64,
        1_000_000.0 / MAX_CYCLE_US as f64,
    );
}

/// Cold start should complete within 2x budget (release: 100ms, debug: 4s).
#[test]
fn regression_cognitive_cycle_cold_start() {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .expect("Failed to create CognitiveLoopService");

    let start = Instant::now();
    let _ = service.cycle("cold start performance check");
    let cold_us = start.elapsed().as_micros() as u64;

    let cold_budget = MAX_CYCLE_US * 2;

    println!(
        "\n=== Cold Start Regression ===\n  cold start: {}us (budget: {}us)\n",
        cold_us, cold_budget,
    );

    assert!(
        cold_us < cold_budget,
        "COLD START REGRESSION: {}us exceeds {}us budget",
        cold_us,
        cold_budget,
    );
}

/// Sustained 500-cycle p95 should stay below 1.5x budget (release: 75ms, debug: 3s).
#[test]
fn regression_cognitive_cycle_sustained_p95() {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .expect("Failed to create CognitiveLoopService");

    for _ in 0..20 {
        let _ = service.cycle("warmup");
    }

    let inputs = [
        "alpha cognition",
        "beta dynamics",
        "gamma binding",
        "delta sleep",
        "epsilon exploration",
    ];

    let n_cycles: usize = if cfg!(debug_assertions) { 50 } else { 500 };
    let mut times_us = Vec::with_capacity(n_cycles);
    for i in 0..n_cycles {
        let start = Instant::now();
        let _ = service.cycle(inputs[i % inputs.len()]);
        times_us.push(start.elapsed().as_micros() as u64);
    }

    times_us.sort();
    let p50 = times_us[times_us.len() / 2];
    let p95 = times_us[times_us.len() * 95 / 100];
    let p99 = times_us[times_us.len() * 99 / 100];

    println!(
        "\n=== Sustained Throughput Regression ===\n  p50={}us p95={}us p99={}us\n",
        p50, p95, p99,
    );

    let p95_budget = MAX_CYCLE_US * 3 / 2;
    assert!(
        p95 < p95_budget,
        "SUSTAINED REGRESSION: p95 {}us exceeds {}us budget",
        p95,
        p95_budget,
    );
}

/// Module timings should be populated after warm-up.
#[test]
fn regression_module_timings_populated() {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .expect("Failed to create CognitiveLoopService");

    for _ in 0..50 {
        let _ = service.cycle("timing check");
    }

    let result = service.cycle("final timing check");
    let timings = &result.metadata.module_timings_us;

    assert!(timings.core_hdc_encode > 0, "core_hdc_encode should be > 0");
    assert!(timings.core_cfc_step > 0, "core_cfc_step should be > 0");
    assert!(timings.core_predict > 0, "core_predict should be > 0");
}