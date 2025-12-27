//! Realistic Consciousness Cycle Profiling
//!
//! This benchmark tests ACTUAL production patterns:
//! - Multiple queries per cycle (batch operations)
//! - Realistic memory sizes (1000+ vectors)
//! - Uses parallel_batch_find_most_similar (the real API)

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::simd_hv::{simd_bind, simd_bundle, simd_permute};
use symthaea::hdc::parallel_hv::parallel_batch_find_most_similar;
use std::time::Instant;

fn main() {
    println!("\n=== REALISTIC CONSCIOUSNESS CYCLE PROFILING ===\n");
    println!("This benchmark matches ACTUAL production usage:");
    println!("- Multiple queries per cycle (10-100 parallel searches)");
    println!("- Realistic memory (1000 episodic memory vectors)");
    println!("- Batch-aware operations (Session 7C optimization active)\n");

    // Realistic scenario: Process consciousness cycle with multiple inputs
    let iterations = 100;
    let queries_per_cycle = 10;  // Multiple sensory inputs, thoughts, etc.
    let memory_size = 1000;      // Realistic episodic memory size

    let mut time_total = 0u128;
    let mut time_encoding = 0u128;
    let mut time_bind = 0u128;
    let mut time_bundle = 0u128;
    let mut time_similarity = 0u128;
    let mut time_permute = 0u128;

    // Create persistent memory (doesn't change every cycle)
    let memory_hvs: Vec<HV16> = (0..memory_size)
        .map(|j| HV16::random(j as u64 + 10000))
        .collect();

    println!("Running {} consciousness cycles...", iterations);
    println!("Each cycle: {} queries × {} memory vectors = {} comparisons\n",
             queries_per_cycle, memory_size, queries_per_cycle * memory_size);

    for i in 0..iterations {
        let start_cycle = Instant::now();

        // 1. ENCODING: Create multiple query vectors (sensory inputs, thoughts)
        let start = Instant::now();
        let query_hvs: Vec<HV16> = (0..queries_per_cycle)
            .map(|q| HV16::random((i * queries_per_cycle + q) as u64))
            .collect();
        time_encoding += start.elapsed().as_nanos();

        // 2. BIND: Bind with context (typical operation)
        let start = Instant::now();
        let key = HV16::random(i as u64 + 5000);
        let bound_hvs: Vec<HV16> = query_hvs.iter()
            .map(|q| simd_bind(q, &key))
            .collect();
        time_bind += start.elapsed().as_nanos();

        // 3. BUNDLE: Combine contexts (typical operation)
        let start = Instant::now();
        let context_hvs: Vec<HV16> = (0..10)
            .map(|j| HV16::random((i * 10 + j) as u64 + 1000))
            .collect();
        let _bundled = simd_bundle(&context_hvs);
        time_bundle += start.elapsed().as_nanos();

        // 4. SIMILARITY: THE CRITICAL OPERATION - Batch search!
        // This is where Session 7C's batch-aware LSH makes the difference!
        let start = Instant::now();
        let _results = parallel_batch_find_most_similar(&bound_hvs, &memory_hvs);
        time_similarity += start.elapsed().as_nanos();

        // 5. PERMUTE: Transform representations
        let start = Instant::now();
        for hv in &bound_hvs {
            let _permuted = simd_permute(hv, 1);
        }
        time_permute += start.elapsed().as_nanos();

        time_total += start_cycle.elapsed().as_nanos();
    }

    // Calculate averages
    let iterations_u128 = iterations as u128;
    let avg_encoding = time_encoding / iterations_u128;
    let avg_bind = time_bind / iterations_u128;
    let avg_bundle = time_bundle / iterations_u128;
    let avg_similarity = time_similarity / iterations_u128;
    let avg_permute = time_permute / iterations_u128;
    let avg_total = time_total / iterations_u128;

    println!("=== RESULTS ===\n");
    println!("Average times per consciousness cycle ({} iterations):", iterations);
    println!();
    println!("  1. Encoding ({} vectors):  {:>8} ns  ({:>5.1}%)",
             queries_per_cycle, avg_encoding,
             100.0 * avg_encoding as f64 / avg_total as f64);
    println!("  2. Bind ({}x):             {:>8} ns  ({:>5.1}%)",
             queries_per_cycle, avg_bind,
             100.0 * avg_bind as f64 / avg_total as f64);
    println!("  3. Bundle:                  {:>8} ns  ({:>5.1}%)", avg_bundle,
             100.0 * avg_bundle as f64 / avg_total as f64);
    println!("  4. Similarity (BATCH):      {:>8} ns  ({:>5.1}%) ← Session 7C Optimized",
             avg_similarity,
             100.0 * avg_similarity as f64 / avg_total as f64);
    println!("  5. Permute ({}x):          {:>8} ns  ({:>5.1}%)",
             queries_per_cycle, avg_permute,
             100.0 * avg_permute as f64 / avg_total as f64);
    println!();
    println!("  TOTAL CYCLE TIME:           {:>8} ns  (100.0%)", avg_total);
    println!();

    // Identify bottleneck
    let times = [avg_encoding, avg_bind, avg_bundle, avg_similarity, avg_permute];
    let max_time = times.iter().max().unwrap();

    let bottleneck = if max_time == &avg_encoding {
        "Encoding"
    } else if max_time == &avg_bind {
        "Bind operations"
    } else if max_time == &avg_bundle {
        "Bundle"
    } else if max_time == &avg_similarity {
        "Batch similarity search"
    } else {
        "Permute"
    };

    println!("🎯 PRIMARY BOTTLENECK: {} ({} ns, {:.1}%)",
             bottleneck, max_time, 100.0 * *max_time as f64 / avg_total as f64);

    println!("\n=== BATCH-AWARE LSH IMPACT ===\n");
    println!("With {} queries × {} memory = {} comparisons per cycle:",
             queries_per_cycle, memory_size, queries_per_cycle * memory_size);
    println!();
    println!("Session 7C batch-aware LSH optimization:");
    println!("- Builds LSH index ONCE (amortized across {} queries)", queries_per_cycle);
    println!("- Each query searches candidates (~10.9% of {}) = ~{} comparisons",
             memory_size, (memory_size as f32 * 0.109) as usize);
    println!("- Total: Build overhead + {} efficient queries", queries_per_cycle);
    println!();
    println!("Comparison with naive approach:");
    println!("- Naive: {} full scans × {} vectors = {} comparisons",
             queries_per_cycle, memory_size, queries_per_cycle * memory_size);
    println!("- LSH Batch: 1 index build + {} queries on ~10.9% candidates", queries_per_cycle);
    println!();

    println!("=== PROFILING COMPLETE ===\n");
}
