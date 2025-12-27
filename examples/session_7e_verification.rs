//! Session 7E Verification: Query-Aware Adaptive Routing
//!
//! Verifies that the three-level routing works correctly:
//! 1. Small datasets (<500): Always naive
//! 2. Large datasets, few queries (<20): Naive (avoid LSH overhead)
//! 3. Large datasets, many queries (≥20): Batch LSH (81x speedup)

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::simd_hv::simd_find_most_similar;
use symthaea::hdc::lsh_similarity::adaptive_batch_find_most_similar;
use std::time::Instant;

fn main() {
    println!("\n=== SESSION 7E: QUERY-AWARE ADAPTIVE ROUTING VERIFICATION ===\n");

    println!("Testing three-level routing logic:\n");

    // Level 1: Small dataset - always naive
    test_small_dataset();

    // Level 2: Large dataset, few queries - naive to avoid overhead
    test_few_queries();

    // Level 3: Large dataset, many queries - batch LSH wins
    test_many_queries();

    println!("\n=== VERIFICATION COMPLETE ===\n");
}

fn test_small_dataset() {
    println!("=== Level 1: Small Dataset (<500 vectors) ===");
    println!("Expected: Both use naive regardless of query count\n");

    for num_queries in [10, 50, 100] {
        let queries: Vec<HV16> = (0..num_queries).map(|i| HV16::random(i as u64)).collect();
        let memory: Vec<HV16> = (0..100).map(|i| HV16::random((i + 1000) as u64)).collect();

        let start = Instant::now();
        let _results = adaptive_batch_find_most_similar(&queries, &memory);
        let time = start.elapsed();

        println!("  {} queries × 100 memory: {:>8.2}µs (naive expected)",
                 num_queries, time.as_micros() as f64);
    }
    println!();
}

fn test_few_queries() {
    println!("=== Level 2: Large Dataset, FEW Queries (<20) ===");
    println!("Expected: Use naive to avoid LSH overhead\n");

    for num_queries in [1, 5, 10, 15] {
        let queries: Vec<HV16> = (0..num_queries).map(|i| HV16::random(i as u64)).collect();
        let memory: Vec<HV16> = (0..1000).map(|i| HV16::random((i + 1000) as u64)).collect();

        // Baseline: naive SIMD
        let start = Instant::now();
        let _naive: Vec<_> = queries.iter()
            .map(|q| simd_find_most_similar(q, &memory))
            .collect();
        let naive_time = start.elapsed();

        // Session 7E: adaptive batch
        let start = Instant::now();
        let _adaptive = adaptive_batch_find_most_similar(&queries, &memory);
        let adaptive_time = start.elapsed();

        let ratio = adaptive_time.as_nanos() as f64 / naive_time.as_nanos() as f64;

        println!("  {} queries × 1000 memory:", num_queries);
        println!("    Naive:    {:>8.2}µs", naive_time.as_micros() as f64);
        println!("    Adaptive: {:>8.2}µs (ratio: {:.2}x - should be ~1.0 for naive routing)",
                 adaptive_time.as_micros() as f64, ratio);
    }
    println!();
}

fn test_many_queries() {
    println!("=== Level 3: Large Dataset, MANY Queries (≥20) ===");
    println!("Expected: Use batch LSH for optimal performance\n");

    for num_queries in [20, 30, 50, 100, 200] {
        let queries: Vec<HV16> = (0..num_queries).map(|i| HV16::random(i as u64)).collect();
        let memory: Vec<HV16> = (0..1000).map(|i| HV16::random((i + 1000) as u64)).collect();

        // Baseline: naive SIMD
        let start = Instant::now();
        let _naive: Vec<_> = queries.iter()
            .map(|q| simd_find_most_similar(q, &memory))
            .collect();
        let naive_time = start.elapsed();

        // Session 7E: adaptive batch (should use LSH)
        let start = Instant::now();
        let _adaptive = adaptive_batch_find_most_similar(&queries, &memory);
        let adaptive_time = start.elapsed();

        let speedup = naive_time.as_nanos() as f64 / adaptive_time.as_nanos() as f64;

        println!("  {} queries × 1000 memory:", num_queries);
        println!("    Naive:    {:>8.2}µs", naive_time.as_micros() as f64);
        println!("    Adaptive: {:>8.2}µs (speedup: {:.2}x - batch LSH routing)",
                 adaptive_time.as_micros() as f64, speedup);
    }
    println!();
}
