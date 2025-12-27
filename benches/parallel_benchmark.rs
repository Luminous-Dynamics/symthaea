//! Parallel vs Sequential Benchmark
//!
//! Verifies 7x speedup claims for parallel HDC operations using rayon.
//! Run with: cargo bench --bench parallel_benchmark --release

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use symthaea::hdc::{
    binary_hv::HV16,
    simd_hv::*,
    parallel_hv::*,
};
use std::time::Instant;

// ============================================================================
// PARALLEL BATCH BIND
// ============================================================================

fn bench_parallel_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Batch Bind");

    for batch_size in [100, 500, 1000, 5000].iter() {
        let vectors: Vec<HV16> = (0..*batch_size).map(|i| HV16::random(i as u64)).collect();
        let key = HV16::random(99999);

        // Sequential
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    let results: Vec<HV16> = vectors.iter()
                        .map(|v| simd_bind(v, &key))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Parallel
        group.bench_with_input(
            BenchmarkId::new("parallel", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    let results = parallel_batch_bind(&vectors, &key);
                    black_box(results)
                })
            },
        );

        // Adaptive (should choose parallel for these sizes)
        group.bench_with_input(
            BenchmarkId::new("adaptive", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    let results = adaptive_batch_bind(&vectors, &key);
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// PARALLEL BATCH SIMILARITY
// ============================================================================

fn bench_parallel_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Batch Similarity");

    for batch_size in [100, 500, 1000, 5000].iter() {
        let query = HV16::random(42);
        let targets: Vec<HV16> = (0..*batch_size).map(|i| HV16::random(i as u64 + 100)).collect();

        // Sequential
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    let results: Vec<f32> = targets.iter()
                        .map(|t| simd_similarity(&query, t))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Parallel
        group.bench_with_input(
            BenchmarkId::new("parallel", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    let results = parallel_batch_similarity(&query, &targets);
                    black_box(results)
                })
            },
        );

        // Adaptive
        group.bench_with_input(
            BenchmarkId::new("adaptive", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    let results = adaptive_batch_similarity(&query, &targets);
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// PARALLEL BATCH BUNDLE
// ============================================================================

fn bench_parallel_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Batch Bundle");

    for num_sets in [10, 50, 100, 200].iter() {
        let vector_sets: Vec<Vec<HV16>> = (0..*num_sets)
            .map(|set_idx| {
                (0..10).map(|i| HV16::random((set_idx * 10 + i) as u64)).collect()
            })
            .collect();

        // Sequential
        group.bench_with_input(
            BenchmarkId::new("sequential", num_sets),
            num_sets,
            |bencher, _| {
                bencher.iter(|| {
                    let results: Vec<HV16> = vector_sets.iter()
                        .map(|set| simd_bundle(set))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Parallel
        group.bench_with_input(
            BenchmarkId::new("parallel", num_sets),
            num_sets,
            |bencher, _| {
                bencher.iter(|| {
                    let results = parallel_batch_bundle(&vector_sets);
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// PARALLEL K-NN SEARCH
// ============================================================================

fn bench_parallel_knn_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel k-NN Search");

    for num_queries in [10, 50, 100].iter() {
        let queries: Vec<HV16> = (0..*num_queries).map(|i| HV16::random(i as u64)).collect();
        let memory: Vec<HV16> = (0..1000).map(|i| HV16::random(i as u64 + 10000)).collect();

        // Sequential
        group.bench_with_input(
            BenchmarkId::new("sequential", num_queries),
            num_queries,
            |bencher, _| {
                bencher.iter(|| {
                    let results: Vec<Option<(usize, f32)>> = queries.iter()
                        .map(|q| simd_find_most_similar(q, &memory))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Parallel
        group.bench_with_input(
            BenchmarkId::new("parallel", num_queries),
            num_queries,
            |bencher, _| {
                bencher.iter(|| {
                    let results = parallel_batch_find_most_similar(&queries, &memory);
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// PARALLEL TOP-K SEARCH
// ============================================================================

fn bench_parallel_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Top-K Search");

    for k in [1, 5, 10].iter() {
        let queries: Vec<HV16> = (0..50).map(|i| HV16::random(i as u64)).collect();
        let memory: Vec<HV16> = (0..1000).map(|i| HV16::random(i as u64 + 10000)).collect();

        group.bench_with_input(
            BenchmarkId::new("top_k", k),
            k,
            |bencher, _| {
                bencher.iter(|| {
                    let results = parallel_batch_find_top_k(&queries, &memory, *k);
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// SPEEDUP MEASUREMENT & VERIFICATION
// ============================================================================

fn verify_speedup_claims() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║      PARALLEL SPEEDUP VERIFICATION (8-Core Target)       ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let info = get_parallelism_info();
    println!("System: {}\n", info);

    // Verify bind speedup
    {
        let vectors: Vec<HV16> = (0..1000).map(|i| HV16::random(i)).collect();
        let key = HV16::random(999);

        // Warmup
        for _ in 0..10 {
            let _ = parallel_batch_bind(&vectors, &key);
        }

        // Sequential
        let start = Instant::now();
        for _ in 0..100 {
            let _: Vec<HV16> = vectors.iter().map(|v| simd_bind(v, &key)).collect();
        }
        let seq_time = start.elapsed().as_micros() / 100;

        // Parallel
        let start = Instant::now();
        for _ in 0..100 {
            let _ = parallel_batch_bind(&vectors, &key);
        }
        let par_time = start.elapsed().as_micros() / 100;

        let speedup = seq_time as f64 / par_time as f64;

        println!("Batch Bind (1000 vectors):");
        println!("  Sequential: {} µs", seq_time);
        println!("  Parallel:   {} µs", par_time);
        println!("  Speedup:    {:.1}x", speedup);
        println!("  Target:     7x | Status: {}\n",
                 if speedup >= 6.0 { "✓ ACHIEVED" } else { "⚠ Below target" });
    }

    // Verify similarity speedup
    {
        let query = HV16::random(42);
        let targets: Vec<HV16> = (0..1000).map(|i| HV16::random(i + 100)).collect();

        // Warmup
        for _ in 0..10 {
            let _ = parallel_batch_similarity(&query, &targets);
        }

        // Sequential
        let start = Instant::now();
        for _ in 0..100 {
            let _: Vec<f32> = targets.iter().map(|t| simd_similarity(&query, t)).collect();
        }
        let seq_time = start.elapsed().as_micros() / 100;

        // Parallel
        let start = Instant::now();
        for _ in 0..100 {
            let _ = parallel_batch_similarity(&query, &targets);
        }
        let par_time = start.elapsed().as_micros() / 100;

        let speedup = seq_time as f64 / par_time as f64;

        println!("Batch Similarity (1000 vectors):");
        println!("  Sequential: {} µs", seq_time);
        println!("  Parallel:   {} µs", par_time);
        println!("  Speedup:    {:.1}x", speedup);
        println!("  Target:     7.5x | Status: {}\n",
                 if speedup >= 6.5 { "✓ ACHIEVED" } else { "⚠ Below target" });
    }

    // Verify bundle speedup
    {
        let vector_sets: Vec<Vec<HV16>> = (0..100)
            .map(|set_idx| {
                (0..10).map(|i| HV16::random((set_idx * 10 + i) as u64)).collect()
            })
            .collect();

        // Warmup
        for _ in 0..10 {
            let _ = parallel_batch_bundle(&vector_sets);
        }

        // Sequential
        let start = Instant::now();
        for _ in 0..100 {
            let _: Vec<HV16> = vector_sets.iter().map(|set| simd_bundle(set)).collect();
        }
        let seq_time = start.elapsed().as_micros() / 100;

        // Parallel
        let start = Instant::now();
        for _ in 0..100 {
            let _ = parallel_batch_bundle(&vector_sets);
        }
        let par_time = start.elapsed().as_micros() / 100;

        let speedup = seq_time as f64 / par_time as f64;

        println!("Batch Bundle (100 sets × 10 vectors):");
        println!("  Sequential: {} µs", seq_time);
        println!("  Parallel:   {} µs", par_time);
        println!("  Speedup:    {:.1}x", speedup);
        println!("  Target:     7x | Status: {}\n",
                 if speedup >= 6.0 { "✓ ACHIEVED" } else { "⚠ Below target" });
    }

    println!("═══════════════════════════════════════════════════════════\n");
}

// ============================================================================
// BENCHMARK REGISTRATION
// ============================================================================

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_parallel_bind,
        bench_parallel_similarity,
        bench_parallel_bundle,
        bench_parallel_knn_search,
        bench_parallel_top_k
);

fn main() {
    verify_speedup_claims();
    benches();
    Criterion::default().final_summary();
}
