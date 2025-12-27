//! Direct Comparison: Naive vs Batch-Aware LSH
//!
//! Tests the SAME scenario with both approaches to see actual speedup

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::simd_hv::simd_find_most_similar;
use symthaea::hdc::lsh_similarity::adaptive_batch_find_most_similar;
use std::time::Instant;

fn main() {
    println!("\n=== NAIVE VS BATCH-AWARE LSH: DIRECT COMPARISON ===\n");

    test_scenario(10, 100);   // Small memory (below LSH threshold)
    test_scenario(10, 500);   // At LSH threshold
    test_scenario(10, 1000);  // Above LSH threshold
    test_scenario(100, 1000); // Many queries, large memory

    println!("\n=== ANALYSIS COMPLETE ===\n");
}

fn test_scenario(num_queries: usize, memory_size: usize) {
    println!("=== Scenario: {} queries, {} memory vectors ===", num_queries, memory_size);

    // Create test data
    let queries: Vec<HV16> = (0..num_queries)
        .map(|i| HV16::random(i as u64))
        .collect();

    let memory: Vec<HV16> = (0..memory_size)
        .map(|j| HV16::random((j + 1000) as u64))
        .collect();

    // Approach 1: Naive (individual queries)
    let start = Instant::now();
    let _naive_results: Vec<_> = queries.iter()
        .map(|q| simd_find_most_similar(q, &memory))
        .collect();
    let naive_time = start.elapsed();

    // Approach 2: Batch-aware adaptive
    let start = Instant::now();
    let _batch_results = adaptive_batch_find_most_similar(&queries, &memory);
    let batch_time = start.elapsed();

    let speedup = naive_time.as_nanos() as f64 / batch_time.as_nanos() as f64;

    println!("  Naive (individual):  {:>10.2}µs", naive_time.as_micros() as f64);
    println!("  Batch-aware:         {:>10.2}µs", batch_time.as_micros() as f64);
    println!("  Speedup:             {:>10.2}x", speedup);

    if memory_size < 500 {
        println!("  Note: Below LSH threshold - both use naive");
    } else if speedup < 1.0 {
        println!("  ⚠️  SLOWER! LSH overhead > benefit for {} queries", num_queries);
    } else if speedup < 2.0 {
        println!("  ⚠️  Marginal improvement - overhead high");
    } else {
        println!("  ✅ Clear win - LSH beneficial!");
    }
    println!();
}
