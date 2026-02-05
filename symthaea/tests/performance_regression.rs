#![cfg(feature = "benchmarks_module")]
//! Performance Regression Tests
//!
//! These tests validate that critical operations maintain expected performance
//! characteristics. Based on documented baselines in benches/BENCHMARK_GUIDE.md:
//!
//! | Operation | Expected Time (Release) |
//! |-----------|-------------------------|
//! | HV16 bind | ~50 ns |
//! | HV16 similarity | ~100 ns |
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
use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::simd_hv16::SimdHV16;

/// Number of iterations for timing (lower for debug mode)
const ITERATIONS: usize = 1_000;

/// Warm-up iterations
const WARMUP: usize = 100;

// ============================================================================
// HV16 OPERATIONS - Debug-mode compatible thresholds (sanity checks)
// For accurate perf measurement, use `cargo bench`
// ============================================================================

/// HV16 bind: release ~50ns, debug ~500µs, threshold: 10ms (catastrophic regression check)
#[test]
fn regression_hv16_bind() {
    println!("\n=== HV16 Bind Performance Regression ===\n");

    let a = HV16::random(42);
    let b = HV16::random(43);

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

    println!("  HV16.bind(): {:.1} µs/op (release: ~0.05µs, debug: ~500µs)", us_per_op);

    // 10ms threshold - only fails for catastrophic regression
    assert!(
        us_per_op < 10_000.0,
        "CATASTROPHIC REGRESSION: HV16.bind() took {:.1} µs (threshold: 10000 µs)",
        us_per_op
    );
}

/// HV16 similarity: release ~100ns, debug ~500µs, threshold: 10ms
#[test]
fn regression_hv16_similarity() {
    println!("\n=== HV16 Similarity Performance Regression ===\n");

    let a = HV16::random(42);
    let b = HV16::random(43);

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

    println!("  HV16.similarity(): {:.1} µs/op (release: ~0.1µs, debug: ~500µs)", us_per_op);

    // 10ms threshold - only fails for catastrophic regression
    assert!(
        us_per_op < 10_000.0,
        "CATASTROPHIC REGRESSION: HV16.similarity() took {:.1} µs (threshold: 10000 µs)",
        us_per_op
    );
}

// ============================================================================
// SIMD OPERATIONS - Debug-mode compatible thresholds
// ============================================================================

/// SimdHV16 bind: threshold 10ms (catastrophic regression check)
#[test]
fn regression_simd_bind() {
    println!("\n=== SimdHV16 Bind Performance Regression ===\n");

    let a = SimdHV16::random(42);
    let b = SimdHV16::random(43);

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

    println!("  SimdHV16.bind(): {:.1} µs/op", us_per_op);

    assert!(
        us_per_op < 10_000.0,
        "CATASTROPHIC REGRESSION: SimdHV16.bind() took {:.1} µs (threshold: 10000 µs)",
        us_per_op
    );
}

/// SimdHV16 similarity: threshold 10ms (catastrophic regression check)
#[test]
fn regression_simd_similarity() {
    println!("\n=== SimdHV16 Similarity Performance Regression ===\n");

    let a = SimdHV16::random(42);
    let b = SimdHV16::random(43);

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

    println!("  SimdHV16.similarity(): {:.1} µs/op", us_per_op);

    assert!(
        us_per_op < 10_000.0,
        "CATASTROPHIC REGRESSION: SimdHV16.similarity() took {:.1} µs (threshold: 10000 µs)",
        us_per_op
    );
}

// ============================================================================
// BUNDLE OPERATIONS - Multi-vector aggregation (debug-mode compatible)
// ============================================================================

/// HV16 bundle of 10 vectors: threshold 1 second (catastrophic regression check)
#[test]
fn regression_hv16_bundle() {
    println!("\n=== HV16 Bundle Performance Regression ===\n");

    let vectors: Vec<HV16> = (0..10).map(|i| HV16::random(i + 100)).collect();

    // Warm-up
    for _ in 0..10 {
        black_box(HV16::bundle(black_box(&vectors)));
    }

    // Benchmark (fewer iterations for debug mode)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(HV16::bundle(black_box(&vectors)));
    }
    let elapsed = start.elapsed();
    let ms_per_op = elapsed.as_millis() as f64 / iterations as f64;

    println!("  HV16::bundle(10): {:.2} ms/op", ms_per_op);

    // 1 second threshold - only fails for catastrophic regression
    assert!(
        ms_per_op < 1000.0,
        "CATASTROPHIC REGRESSION: HV16::bundle(10) took {:.2} ms (threshold: 1000 ms)",
        ms_per_op
    );
}

// ============================================================================
// LSH SEARCH - Debug-mode compatible thresholds
// ============================================================================

/// LSH index creation: threshold 5 seconds (catastrophic regression check)
#[test]
fn regression_lsh_index_creation() {
    use symthaea::hdc::lsh_simhash::{SimHashIndex, SimHashConfig};

    println!("\n=== LSH Index Creation Performance Regression ===\n");

    // Create 500 vectors (above LSH threshold)
    let vectors: Vec<HV16> = (0..500).map(|i| HV16::random(i as u64)).collect();

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
    use symthaea::hdc::lsh_simhash::{SimHashIndex, SimHashConfig};

    println!("\n=== LSH Query Performance Regression ===\n");

    // Create 1000 vectors
    let vectors: Vec<HV16> = (0..1000).map(|i| HV16::random(i as u64)).collect();
    let query = HV16::random(9999);

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

/// HV16 should be exactly 2048 bytes (16,384 bits / 8)
#[test]
fn regression_hv16_size() {
    println!("\n=== HV16 Memory Size Regression ===\n");

    let size = std::mem::size_of::<HV16>();
    println!("  HV16 size: {} bytes (expected: 2048)", size);

    assert_eq!(
        size, 2048,
        "HV16 size changed: {} bytes (expected 2048)",
        size
    );
}

/// SimdHV16 should be same size as HV16 (aligned for SIMD)
#[test]
fn regression_simd_hv16_size() {
    println!("\n=== SimdHV16 Memory Size Regression ===\n");

    let size = std::mem::size_of::<SimdHV16>();
    println!("  SimdHV16 size: {} bytes (expected: 2048)", size);

    assert_eq!(
        size, 2048,
        "SimdHV16 size changed: {} bytes (expected 2048)",
        size
    );
}
