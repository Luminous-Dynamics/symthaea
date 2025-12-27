//! Session 8 Parallel Verification: Measure actual parallel speedup
//!
//! Tests parallel vs sequential processing with varying query counts

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::simd_hv::simd_find_most_similar;
use symthaea::hdc::lsh_similarity::adaptive_batch_find_most_similar;
use std::time::Instant;

fn main() {
    println!("\n=== SESSION 8: PARALLEL QUERY PROCESSING VERIFICATION ===");
    println!("CPU Cores: {}\n", rayon::current_num_threads());

    // Test with varying query counts
    test_scenario(10, 1000);    // Current production workload
    test_scenario(20, 1000);    // Double queries
    test_scenario(50, 1000);    // More queries
    test_scenario(100, 1000);   // Many queries

    println!("\n=== VERIFICATION COMPLETE ===\n");
}

fn test_scenario(num_queries: usize, memory_size: usize) {
    println!("=== Scenario: {} queries × {} memory vectors ===", num_queries, memory_size);

    let queries: Vec<HV16> = (0..num_queries)
        .map(|i| HV16::random(i as u64))
        .collect();

    let memory: Vec<HV16> = (0..memory_size)
        .map(|j| HV16::random((j + 1000) as u64))
        .collect();

    // Sequential baseline (manual iteration)
    let start = Instant::now();
    let _sequential: Vec<_> = queries.iter()
        .map(|q| simd_find_most_similar(q, &memory))
        .collect();
    let seq_time = start.elapsed();

    // Parallel version (using adaptive_batch which now has par_iter)
    let start = Instant::now();
    let _parallel = adaptive_batch_find_most_similar(&queries, &memory);
    let par_time = start.elapsed();

    let speedup = seq_time.as_nanos() as f64 / par_time.as_nanos() as f64;

    println!("  Sequential: {:>10.2}µs", seq_time.as_micros() as f64);
    println!("  Parallel:   {:>10.2}µs", par_time.as_micros() as f64);
    println!("  Speedup:    {:>10.2}x", speedup);

    if speedup > 1.5 {
        println!("  ✅ Good parallel speedup!");
    } else if speedup > 1.0 {
        println!("  ⚠️  Marginal speedup - overhead high for {} queries", num_queries);
    } else {
        println!("  ❌ Parallel slower - overhead > benefit");
    }
    println!();
}
