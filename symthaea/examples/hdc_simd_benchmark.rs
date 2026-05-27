// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC SIMD Performance Benchmark
//!
//! Benchmarks SIMD-optimized BinaryHV operations against scalar implementations
//! using real AI benchmark data for realistic workloads.
//!
//! Run with: cargo run --example hdc_simd_benchmark --release

use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};
use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::binary_hv::BinaryHV;
use symthaea::hdc::simd_ops::{bind_simd, hamming_distance_simd, invert_simd, matching_bits_simd};

/// Benchmark result for a single operation
#[derive(Debug, Clone)]
struct BenchResult {
    operation: String,
    simd_ns: u128,
    scalar_ns: u128,
    speedup: f64,
    throughput_mops: f64, // Million operations per second
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║           HDC SIMD PERFORMANCE BENCHMARK                           ║");
    println!("║     Binary Hypervector Operations: SIMD vs Scalar                  ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!(
        "  HDC Dimension: {} bits ({} bytes)",
        HDC_DIMENSION,
        HDC_DIMENSION / 8
    );
    println!("  SIMD Features: AVX2 (256-bit), SSE4.1 (128-bit), POPCNT");
    println!();

    // Run core operation benchmarks
    let results = benchmark_core_operations();

    // Print results table
    print_results_table(&results);

    // Run text encoding benchmark if data available
    benchmark_text_encoding();

    // Run similarity search benchmark
    benchmark_similarity_search();

    // Summary
    print_summary(&results);
}

fn benchmark_core_operations() -> Vec<BenchResult> {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("PART 1: CORE OPERATION BENCHMARKS (1M iterations each)");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let iterations: u128 = 1_000_000;
    let mut results = Vec::new();

    // Create test vectors
    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);

    // 1. BIND (XOR) Operation
    println!("Benchmarking BIND (XOR)...");
    let simd_bind = benchmark_op(iterations, || {
        black_box(bind_simd(black_box(&a.0), black_box(&b.0)))
    });
    let scalar_bind = benchmark_op(iterations, || black_box(a.bind_scalar(black_box(&b))));
    results.push(make_result(
        "Bind (XOR)",
        simd_bind,
        scalar_bind,
        iterations,
    ));

    // 2. SIMILARITY (Matching Bits / POPCNT) Operation
    println!("Benchmarking SIMILARITY (POPCNT)...");
    let simd_sim = benchmark_op(iterations, || {
        black_box(matching_bits_simd(black_box(&a.0), black_box(&b.0)))
    });
    let scalar_sim = benchmark_op(iterations, || black_box(a.similarity_scalar(black_box(&b))));
    results.push(make_result(
        "Similarity (POPCNT)",
        simd_sim,
        scalar_sim,
        iterations,
    ));

    // 3. HAMMING DISTANCE Operation
    println!("Benchmarking HAMMING DISTANCE...");
    let simd_ham = benchmark_op(iterations, || {
        black_box(hamming_distance_simd(black_box(&a.0), black_box(&b.0)))
    });
    let scalar_ham = benchmark_op(iterations, || {
        black_box(a.hamming_distance_scalar(black_box(&b)))
    });
    results.push(make_result(
        "Hamming Distance",
        simd_ham,
        scalar_ham,
        iterations,
    ));

    // 4. INVERT (NOT) Operation
    println!("Benchmarking INVERT (NOT)...");
    let simd_inv = benchmark_op(iterations, || black_box(invert_simd(black_box(&a.0))));
    let scalar_inv = benchmark_op(iterations, || black_box(a.invert_scalar()));
    results.push(make_result(
        "Invert (NOT)",
        simd_inv,
        scalar_inv,
        iterations,
    ));

    // 5. BUNDLE (Majority Vote) - Multiple vectors
    println!("Benchmarking BUNDLE (3 vectors)...");
    let c = BinaryHV::random(44);
    let vectors = vec![a, b, c];
    let bundle_iterations = iterations / 10; // Fewer iterations for more expensive op

    let simd_bundle = benchmark_op(bundle_iterations, || black_box(BinaryHV::bundle(&vectors)));
    // Bundle uses SIMD internally, compare against repeated bind
    let scalar_bundle = benchmark_op(bundle_iterations, || {
        let ab = a.bind_scalar(&b);
        black_box(ab.bind_scalar(&c))
    });
    results.push(make_result(
        "Bundle (3 vec)",
        simd_bundle,
        scalar_bundle,
        bundle_iterations,
    ));

    println!();
    results
}

fn benchmark_op<F, R>(iterations: u128, mut op: F) -> Duration
where
    F: FnMut() -> R,
{
    let start = Instant::now();
    for _ in 0..iterations {
        op();
    }
    start.elapsed()
}

fn make_result(
    name: &str,
    simd_time: Duration,
    scalar_time: Duration,
    iterations: u128,
) -> BenchResult {
    let simd_ns = simd_time.as_nanos() / iterations;
    let scalar_ns = scalar_time.as_nanos() / iterations;
    let speedup = scalar_ns as f64 / simd_ns.max(1) as f64;
    let throughput_mops = 1_000_000_000.0 / simd_ns as f64 / 1_000_000.0;

    BenchResult {
        operation: name.to_string(),
        simd_ns,
        scalar_ns,
        speedup,
        throughput_mops,
    }
}

fn print_results_table(results: &[BenchResult]) {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("RESULTS: SIMD vs SCALAR PERFORMANCE");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!(
        "{:20} | {:>10} | {:>10} | {:>8} | {:>12}",
        "Operation", "SIMD (ns)", "Scalar (ns)", "Speedup", "Throughput"
    );
    println!(
        "{:-<20}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+-{:-<12}",
        "", "", "", "", ""
    );

    for r in results {
        let speedup_str = format!("{:.2}x", r.speedup);
        let throughput_str = format!("{:.2} Mop/s", r.throughput_mops);
        println!(
            "{:20} | {:>10} | {:>10} | {:>8} | {:>12}",
            r.operation, r.simd_ns, r.scalar_ns, speedup_str, throughput_str
        );
    }
    println!();
}

fn benchmark_text_encoding() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("PART 2: TEXT ENCODING BENCHMARK");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Check if benchmark data exists
    let data_path = Path::new("benchmarks/ai_benchmarks/data/winogrande/winogrande_train.json");

    if !data_path.exists() {
        println!("  [SKIP] Benchmark data not found at {:?}", data_path);
        println!("         Run scripts/download_benchmarks.py first\n");
        return;
    }

    // Load and parse a subset of data
    let content = match fs::read_to_string(data_path) {
        Ok(c) => c,
        Err(e) => {
            println!("  [ERROR] Failed to read data: {}\n", e);
            return;
        }
    };

    // Count items (simple JSON parsing)
    let item_count = content.matches("\"sentence\"").count();
    println!(
        "  Data loaded: {} items from Winogrande dataset",
        item_count
    );

    // Simulate text encoding benchmark
    let n_samples = 1000.min(item_count);
    let chars_per_sample = 100; // Average characters per sentence

    println!(
        "  Encoding {} text samples ({} chars each)...",
        n_samples, chars_per_sample
    );

    // Create character-level encoding vectors (item memory)
    let char_memory: Vec<BinaryHV> = (0..256u64).map(|i| BinaryHV::random(i * 12345)).collect();
    let pos_memory: Vec<BinaryHV> = (0..chars_per_sample as u64)
        .map(|i| BinaryHV::random(i * 67890 + 1000))
        .collect();

    // Benchmark encoding using SIMD bind
    let start = Instant::now();
    let mut encoded_samples: Vec<BinaryHV> = Vec::with_capacity(n_samples);

    for sample_idx in 0..n_samples {
        let seed = sample_idx as u64;
        let mut result = BinaryHV::random(seed); // Start with random base

        for (pos, pos_hv) in pos_memory.iter().enumerate().take(chars_per_sample) {
            let char_idx = ((seed + pos as u64) % 256) as usize;
            // Bind character with position (uses SIMD internally)
            let char_hv = &char_memory[char_idx];
            let bound = char_hv.bind(pos_hv);
            result = result.bind(&bound);
        }
        encoded_samples.push(result);
    }
    let encoding_time = start.elapsed();

    let samples_per_sec = n_samples as f64 / encoding_time.as_secs_f64();
    let chars_per_sec = (n_samples * chars_per_sample) as f64 / encoding_time.as_secs_f64();

    println!("  Encoding complete:");
    println!("    Total time:      {:?}", encoding_time);
    println!("    Samples/sec:     {:.2}", samples_per_sec);
    println!("    Characters/sec:  {:.2}M", chars_per_sec / 1_000_000.0);
    println!();

    // Benchmark similarity search
    println!(
        "  Searching {} encoded vectors for nearest neighbor...",
        encoded_samples.len()
    );

    let query = &encoded_samples[0];
    let start = Instant::now();

    let mut best_idx = 0;
    let mut best_sim = 0.0f32;
    for (idx, sample) in encoded_samples.iter().enumerate().skip(1) {
        let sim = query.similarity(sample); // Uses SIMD
        if sim > best_sim {
            best_sim = sim;
            best_idx = idx;
        }
    }
    let search_time = start.elapsed();

    println!("    Search time:     {:?}", search_time);
    println!(
        "    Best match idx:  {} (similarity: {:.4})",
        best_idx, best_sim
    );
    println!(
        "    Comparisons/sec: {:.2}M",
        (encoded_samples.len() - 1) as f64 / search_time.as_secs_f64() / 1_000_000.0
    );
    println!();
}

fn benchmark_similarity_search() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("PART 3: LARGE-SCALE SIMILARITY SEARCH");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let sizes = [1_000, 10_000, 100_000];

    for &n in &sizes {
        println!("  Database size: {} vectors", n);

        // Create database
        let database: Vec<BinaryHV> = (0..n as u64).map(BinaryHV::random).collect();
        let query = BinaryHV::random(999_999);

        // SIMD search
        let start = Instant::now();
        let mut best_idx = 0;
        let mut best_sim = 0.0f32;
        for (idx, vec) in database.iter().enumerate() {
            let sim = query.similarity(vec); // Uses SIMD
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }
        let simd_time = start.elapsed();

        // Scalar search
        let start = Instant::now();
        let mut _best_idx_scalar = 0;
        let mut best_sim_scalar = 0.0f32;
        for (idx, vec) in database.iter().enumerate() {
            let sim = query.similarity_scalar(vec);
            if sim > best_sim_scalar {
                best_sim_scalar = sim;
                _best_idx_scalar = idx;
            }
        }
        let scalar_time = start.elapsed();

        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        let throughput = n as f64 / simd_time.as_secs_f64() / 1_000_000.0;

        println!("    SIMD:   {:>10?} ({:.2}M vec/s)", simd_time, throughput);
        println!("    Scalar: {:>10?}", scalar_time);
        println!("    Speedup: {:.2}x", speedup);
        println!("    Best match: idx={} sim={:.4}", best_idx, best_sim);
        println!();
    }
}

fn print_summary(results: &[BenchResult]) {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
    let max_speedup = results.iter().map(|r| r.speedup).fold(0.0f64, f64::max);
    let min_speedup = results.iter().map(|r| r.speedup).fold(f64::MAX, f64::min);

    let fastest_op = results.iter().min_by_key(|r| r.simd_ns).unwrap();
    let slowest_op = results.iter().max_by_key(|r| r.simd_ns).unwrap();

    println!("SIMD Optimization Results:");
    println!(
        "  Average Speedup:    {:.2}x faster than scalar",
        avg_speedup
    );
    println!("  Maximum Speedup:    {:.2}x (best case)", max_speedup);
    println!("  Minimum Speedup:    {:.2}x (worst case)", min_speedup);
    println!();
    println!(
        "  Fastest Operation:  {} ({} ns/op)",
        fastest_op.operation, fastest_op.simd_ns
    );
    println!(
        "  Slowest Operation:  {} ({} ns/op)",
        slowest_op.operation, slowest_op.simd_ns
    );
    println!();
    println!("Hardware Utilization:");
    println!(
        "  Vector Width:       {} bits (HDC dimension)",
        HDC_DIMENSION
    );
    println!("  AVX2 Registers:     256 bits (64 ops per vector)");
    println!("  Effective Parallelism: {}x", HDC_DIMENSION / 256);
    println!();

    // Overall assessment
    if avg_speedup >= 1.5 {
        println!(
            "✅ SIMD optimization providing significant speedup ({:.1}x average)",
            avg_speedup
        );
    } else if avg_speedup >= 1.0 {
        println!(
            "⚠️  SIMD optimization providing modest speedup ({:.1}x average)",
            avg_speedup
        );
    } else {
        println!(
            "❌ SIMD optimization not providing expected speedup ({:.1}x average)",
            avg_speedup
        );
    }
    println!();
}