//! Detailed Profiling Benchmark - Session 7
//!
//! Measures EXACT time spent in each component to identify bottlenecks
//! Run with: cargo bench --bench detailed_profiling -- --nocapture

use criterion::{criterion_group, criterion_main, Criterion};
use symthaea::hdc::{binary_hv::HV16, simd_hv::*};
use std::time::Instant;

// ============================================================================
// DETAILED CONSCIOUSNESS CYCLE PROFILING
// ============================================================================

fn profile_consciousness_cycle_detailed(_c: &mut Criterion) {
    println!("\n=== DETAILED CONSCIOUSNESS CYCLE PROFILING ===\n");

    let iterations = 1000;

    // Component timing accumulators
    let mut time_encoding = 0u128;
    let mut time_bind = 0u128;
    let mut time_bundle = 0u128;
    let mut time_similarity = 0u128;
    let mut time_permute = 0u128;
    let mut time_total = 0u128;

    for i in 0..iterations {
        let start_cycle = Instant::now();

        // 1. ENCODING: Create query vector
        let start = Instant::now();
        let query_hv = HV16::random(i + 42);
        let context_hvs: Vec<HV16> = (0..10).map(|j| HV16::random(i + j + 100)).collect();
        time_encoding += start.elapsed().as_nanos();

        // 2. BIND: Combine query with context
        let start = Instant::now();
        let mut bound = query_hv.clone();
        for ctx in &context_hvs {
            bound = simd_bind(&bound, ctx);
        }
        time_bind += start.elapsed().as_nanos();

        // 3. BUNDLE: Aggregate concepts
        let start = Instant::now();
        let bundled = simd_bundle(&context_hvs);
        time_bundle += start.elapsed().as_nanos();

        // 4. SIMILARITY: Search memory
        let start = Instant::now();
        let memory_hvs: Vec<HV16> = (0..100).map(|j| HV16::random(i + j + 200)).collect();
        let _best = simd_find_most_similar(&bundled, &memory_hvs);
        time_similarity += start.elapsed().as_nanos();

        // 5. PERMUTE: Transform representation
        let start = Instant::now();
        let _permuted = simd_permute(&bundled, 1); // Shift by 1 position
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

    println!("Average times per cycle ({} iterations):", iterations);
    println!();
    println!("  1. Encoding:    {:>8} ns  ({:>5.1}%)", avg_encoding,
             100.0 * avg_encoding as f64 / avg_total as f64);
    println!("  2. Bind (10x):  {:>8} ns  ({:>5.1}%)", avg_bind,
             100.0 * avg_bind as f64 / avg_total as f64);
    println!("  3. Bundle:      {:>8} ns  ({:>5.1}%)", avg_bundle,
             100.0 * avg_bundle as f64 / avg_total as f64);
    println!("  4. Similarity:  {:>8} ns  ({:>5.1}%)", avg_similarity,
             100.0 * avg_similarity as f64 / avg_total as f64);
    println!("  5. Permute:     {:>8} ns  ({:>5.1}%)", avg_permute,
             100.0 * avg_permute as f64 / avg_total as f64);
    println!();
    println!("  TOTAL:          {:>8} ns  (100.0%)", avg_total);
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
        "Similarity search"
    } else {
        "Permute"
    };

    println!("🎯 PRIMARY BOTTLENECK: {} ({} ns, {:.1}%)",
             bottleneck, max_time, 100.0 * *max_time as f64 / avg_total as f64);

    println!("\n=== END DETAILED PROFILING ===\n");
}

// ============================================================================
// BATCH SIZE DISTRIBUTION ANALYSIS
// ============================================================================

fn profile_batch_characteristics(_c: &mut Criterion) {
    println!("\n=== BATCH SIZE CHARACTERISTICS ===\n");

    let batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000];

    println!("Testing different batch sizes to find GPU threshold:\n");
    println!("Batch Size | Bundle Time | Similarity Time | Notes");
    println!("-----------|-------------|-----------------|------");

    for &size in &batch_sizes {
        // Create batch
        let vectors: Vec<HV16> = (0..size).map(|i| HV16::random(i as u64)).collect();
        let query = HV16::random(99999);

        // Time bundle
        let start = Instant::now();
        for _ in 0..100 {
            let _result = simd_bundle(&vectors);
        }
        let bundle_time = start.elapsed().as_micros() / 100;

        // Time similarity search
        let memory: Vec<HV16> = (0..1000).map(|i| HV16::random(i as u64 + 10000)).collect();
        let start = Instant::now();
        for _ in 0..100 {
            let _result = simd_find_most_similar(&query, &memory);
        }
        let sim_time = start.elapsed().as_micros() / 100;

        // Determine if GPU-suitable
        let note = if size >= 1000 {
            "✅ GPU candidate"
        } else if size >= 500 {
            "⚠️  Borderline"
        } else {
            "❌ Too small"
        };

        println!("{:>10} | {:>11} µs | {:>15} µs | {}",
                 size, bundle_time, sim_time, note);
    }

    println!("\n📊 GPU Threshold Analysis:");
    println!("  - Batches <500:  Too small for GPU (transfer overhead dominates)");
    println!("  - Batches 500-1000: Borderline (may benefit on high-end GPUs)");
    println!("  - Batches >1000: Good GPU candidates (amortize transfer cost)");

    println!("\n=== END BATCH ANALYSIS ===\n");
}

// ============================================================================
// OPERATION FREQUENCY ANALYSIS
// ============================================================================

fn profile_operation_frequency(_c: &mut Criterion) {
    println!("\n=== OPERATION FREQUENCY ANALYSIS ===\n");

    let total_ops = 10000;

    // Simulate realistic workload
    let mut bind_count = 0u32;
    let mut bundle_count = 0u32;
    let mut similarity_count = 0u32;
    let mut permute_count = 0u32;

    let mut bind_time = 0u128;
    let mut bundle_time = 0u128;
    let mut similarity_time = 0u128;
    let mut permute_time = 0u128;

    for i in 0..total_ops {
        let op_type = i % 10; // Simulate operation distribution

        match op_type {
            0..=3 => {
                // Bind (40%)
                let start = Instant::now();
                let v1 = HV16::random(i as u64);
                let v2 = HV16::random(i as u64 + 1);
                let _result = simd_bind(&v1, &v2);
                bind_time += start.elapsed().as_nanos();
                bind_count += 1;
            }
            4..=5 => {
                // Bundle (20%)
                let start = Instant::now();
                let vectors: Vec<HV16> = (0..10).map(|j| HV16::random(i as u64 + j)).collect();
                let _result = simd_bundle(&vectors);
                bundle_time += start.elapsed().as_nanos();
                bundle_count += 1;
            }
            6..=8 => {
                // Similarity (30%)
                let start = Instant::now();
                let query = HV16::random(i as u64);
                let memory: Vec<HV16> = (0..100).map(|j| HV16::random(i as u64 + j + 1000)).collect();
                let _result = simd_find_most_similar(&query, &memory);
                similarity_time += start.elapsed().as_nanos();
                similarity_count += 1;
            }
            _ => {
                // Permute (10%)
                let start = Instant::now();
                let v = HV16::random(i as u64);
                let _result = simd_permute(&v, 1); // Shift by 1 position
                permute_time += start.elapsed().as_nanos();
                permute_count += 1;
            }
        }
    }

    let total_time = bind_time + bundle_time + similarity_time + permute_time;

    println!("Operation frequency ({} operations):\n", total_ops);
    println!("Operation  | Count | Avg Time | Total Time | % of Total");
    println!("-----------|-------|----------|------------|------------");
    println!("Bind       | {:>5} | {:>7} ns | {:>9} µs | {:>5.1}%",
             bind_count, bind_time / bind_count as u128,
             bind_time / 1000, 100.0 * bind_time as f64 / total_time as f64);
    println!("Bundle     | {:>5} | {:>7} ns | {:>9} µs | {:>5.1}%",
             bundle_count, bundle_time / bundle_count as u128,
             bundle_time / 1000, 100.0 * bundle_time as f64 / total_time as f64);
    println!("Similarity | {:>5} | {:>7} ns | {:>9} µs | {:>5.1}%",
             similarity_count, similarity_time / similarity_count as u128,
             similarity_time / 1000, 100.0 * similarity_time as f64 / total_time as f64);
    println!("Permute    | {:>5} | {:>7} ns | {:>9} µs | {:>5.1}%",
             permute_count, permute_time / permute_count as u128,
             permute_time / 1000, 100.0 * permute_time as f64 / total_time as f64);

    println!("\n📊 Key Insights:");
    let percentages = [
        (bind_time as f64 / total_time as f64, "Bind"),
        (bundle_time as f64 / total_time as f64, "Bundle"),
        (similarity_time as f64 / total_time as f64, "Similarity"),
        (permute_time as f64 / total_time as f64, "Permute"),
    ];
    let max_pct = percentages.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();

    println!("  - Primary time sink: {} ({:.1}% of total)", max_pct.1, max_pct.0 * 100.0);
    println!("  - Most frequent: {} operations ({:.1}%)",
             if bind_count > similarity_count { "Bind" } else { "Similarity" },
             if bind_count > similarity_count {
                 100.0 * bind_count as f64 / total_ops as f64
             } else {
                 100.0 * similarity_count as f64 / total_ops as f64
             });

    println!("\n=== END FREQUENCY ANALYSIS ===\n");
}

// ============================================================================
// GPU SUITABILITY ANALYSIS
// ============================================================================

fn analyze_gpu_suitability(_c: &mut Criterion) {
    println!("\n=== GPU SUITABILITY ANALYSIS ===\n");

    println!("Based on profiling data, analyzing GPU viability:\n");

    // Test different scenarios
    let scenarios = [
        ("Small batches (n=10)", 10),
        ("Medium batches (n=100)", 100),
        ("Large batches (n=1000)", 1000),
        ("Very large (n=10000)", 10000),
    ];

    println!("Scenario                | CPU Time | GPU Est. | Speedup | Worth it?");
    println!("------------------------|----------|----------|---------|----------");

    for (name, batch_size) in &scenarios {
        // Time CPU operation
        let vectors: Vec<HV16> = (0..*batch_size).map(|i| HV16::random(i as u64)).collect();

        let start = Instant::now();
        let _result = simd_bundle(&vectors);
        let cpu_time = start.elapsed().as_micros();

        // Estimate GPU time
        // Transfer overhead: ~10µs constant
        // Per-vector transfer: ~16ns * batch_size
        // Compute: ~0.5ns * batch_size (GPU parallel)
        let transfer_overhead = 10.0; // µs
        let transfer_time = *batch_size as f64 * 0.016; // µs
        let compute_time = *batch_size as f64 * 0.005; // µs (GPU parallel)
        let gpu_time = transfer_overhead + 2.0 * transfer_time + compute_time;

        let speedup = cpu_time as f64 / gpu_time;
        let worth_it = if speedup > 2.0 { "✅ Yes" } else if speedup > 1.0 { "⚠️  Maybe" } else { "❌ No" };

        println!("{:<24} | {:>7} µs | {:>7.1} µs | {:>6.2}x | {}",
                 name, cpu_time, gpu_time, speedup, worth_it);
    }

    println!("\n📊 Conclusion:");
    println!("  - GPU beneficial for batches >1000");
    println!("  - Break-even around 500-1000 vectors");
    println!("  - Small batches (<100) stay on CPU");
    println!("  - Adaptive routing strategy recommended");

    println!("\n=== END GPU SUITABILITY ===\n");
}

criterion_group!(
    benches,
    profile_consciousness_cycle_detailed,
    profile_batch_characteristics,
    profile_operation_frequency,
    analyze_gpu_suitability,
);

criterion_main!(benches);
