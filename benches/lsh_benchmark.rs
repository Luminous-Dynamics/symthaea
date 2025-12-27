// Session 6: LSH vs Brute Force Similarity Search Benchmarks
// Verify the 100-1000x speedup claim with rigorous measurements

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use symthaea::hdc::{HV16, lsh_index::{LshIndex, LshConfig}};
use std::collections::HashSet;

/// Create test dataset of random vectors
fn create_test_dataset(size: usize, seed: u64) -> Vec<HV16> {
    (0..size)
        .map(|i| HV16::random(seed + i as u64))
        .collect()
}

/// Brute force top-k similarity search (current approach)
fn brute_force_search(query: &HV16, vectors: &[HV16], k: usize) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, query.similarity(v)))
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(k);
    results
}

/// LSH approximate top-k similarity search (new approach)
fn lsh_search(
    index: &LshIndex,
    query: &HV16,
    vectors: &[HV16],
    k: usize,
) -> Vec<(usize, f32)> {
    index.query_approximate(query, k, vectors)
}

/// Measure recall: What percentage of true top-k did LSH find?
fn measure_recall(
    lsh_results: &[(usize, f32)],
    brute_force_results: &[(usize, f32)],
) -> f64 {
    let bf_ids: HashSet<usize> = brute_force_results.iter().map(|(id, _)| *id).collect();
    let lsh_ids: HashSet<usize> = lsh_results.iter().map(|(id, _)| *id).collect();

    let intersection = bf_ids.intersection(&lsh_ids).count();
    (intersection as f64 / brute_force_results.len() as f64) * 100.0
}

/// Benchmark brute force vs LSH across different dataset sizes
fn bench_lsh_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_vs_brute_force");

    // Test different dataset sizes: 1K, 10K, 100K, 1M
    let sizes = vec![1_000, 10_000, 100_000];
    let k = 10; // Find top-10 similar vectors

    for size in sizes.iter() {
        let vectors = create_test_dataset(*size, 42);
        let query = HV16::random(99999);

        // Benchmark brute force
        group.bench_with_input(
            BenchmarkId::new("brute_force", size),
            size,
            |b, _| {
                b.iter(|| {
                    let results = brute_force_search(
                        black_box(&query),
                        black_box(&vectors),
                        black_box(k),
                    );
                    black_box(results)
                })
            },
        );

        // Build LSH index
        let config = LshConfig::balanced(); // 10 tables, ~95% recall
        let mut index = LshIndex::new(config);
        index.insert_batch(&vectors);

        // Benchmark LSH search
        group.bench_with_input(
            BenchmarkId::new("lsh_approximate", size),
            size,
            |b, _| {
                b.iter(|| {
                    let results = lsh_search(
                        black_box(&index),
                        black_box(&query),
                        black_box(&vectors),
                        black_box(k),
                    );
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark LSH index build time
fn bench_lsh_build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_index_build");

    let sizes = vec![1_000, 10_000, 100_000];

    for size in sizes.iter() {
        let vectors = create_test_dataset(*size, 42);

        group.bench_with_input(
            BenchmarkId::new("build_index", size),
            size,
            |b, _| {
                b.iter(|| {
                    let config = LshConfig::balanced();
                    let mut index = LshIndex::new(config);
                    index.insert_batch(black_box(&vectors));
                    black_box(index)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark accuracy vs speed trade-off for different LSH configs
fn bench_lsh_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_config_tradeoff");

    let vectors = create_test_dataset(10_000, 42);
    let query = HV16::random(99999);
    let k = 10;

    // Test different configurations
    let configs = vec![
        ("fast_5_tables", LshConfig::fast()),
        ("balanced_10_tables", LshConfig::balanced()),
        ("accurate_20_tables", LshConfig::accurate()),
    ];

    for (name, config) in configs.iter() {
        let mut index = LshIndex::new(config.clone());
        index.insert_batch(&vectors);

        group.bench_with_input(
            BenchmarkId::new("query", name),
            name,
            |b, _| {
                b.iter(|| {
                    let results = lsh_search(
                        black_box(&index),
                        black_box(&query),
                        black_box(&vectors),
                        black_box(k),
                    );
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

/// Comprehensive accuracy test: Measure recall for different LSH configs
fn bench_lsh_accuracy(c: &mut Criterion) {
    println!("\n=== LSH ACCURACY ANALYSIS ===\n");

    let vectors = create_test_dataset(10_000, 42);
    let query = HV16::random(99999);
    let k = 10;

    // Ground truth: Brute force top-10
    let ground_truth = brute_force_search(&query, &vectors, k);

    // Test different configurations
    let configs = vec![
        ("Fast (5 tables)", LshConfig::fast()),
        ("Balanced (10 tables)", LshConfig::balanced()),
        ("Accurate (20 tables)", LshConfig::accurate()),
    ];

    for (name, config) in configs.iter() {
        let mut index = LshIndex::new(config.clone());
        index.insert_batch(&vectors);

        let lsh_results = lsh_search(&index, &query, &vectors, k);
        let recall = measure_recall(&lsh_results, &ground_truth);

        println!("{}: {:.1}% recall", name, recall);
    }

    println!("\n=== END ACCURACY ANALYSIS ===\n");
}

/// Measure LSH performance scaling: Does it stay constant as dataset grows?
fn bench_lsh_constant_scaling(c: &mut Criterion) {
    println!("\n=== LSH SCALING VERIFICATION ===");
    println!("Testing hypothesis: LSH query time stays constant as dataset grows\n");

    let sizes = vec![1_000, 10_000, 100_000];
    let k = 10;

    for size in sizes.iter() {
        let vectors = create_test_dataset(*size, 42);
        let query = HV16::random(99999);

        let config = LshConfig::balanced();
        let mut index = LshIndex::new(config);
        index.insert_batch(&vectors);

        // Measure average query time
        let mut total_time = std::time::Duration::ZERO;
        let samples = 10;

        for _ in 0..samples {
            let start = std::time::Instant::now();
            let _results = lsh_search(&index, &query, &vectors, k);
            total_time += start.elapsed();
        }

        let avg_time_ms = total_time.as_micros() as f64 / samples as f64 / 1000.0;
        println!("{:>6} vectors: {:.3}ms per query", size, avg_time_ms);
    }

    println!("\nExpected: Time should stay relatively constant (LSH advantage)\n");
    println!("=== END SCALING VERIFICATION ===\n");
}

/// Benchmark candidate collection efficiency
fn bench_lsh_candidates(c: &mut Criterion) {
    println!("\n=== LSH CANDIDATE ANALYSIS ===\n");

    let vectors = create_test_dataset(100_000, 42);
    let query = HV16::random(99999);

    let configs = vec![
        ("Fast (5 tables)", LshConfig::fast()),
        ("Balanced (10 tables)", LshConfig::balanced()),
        ("Accurate (20 tables)", LshConfig::accurate()),
    ];

    for (name, config) in configs.iter() {
        let mut index = LshIndex::new(config.clone());
        index.insert_batch(&vectors);

        // Count candidates (we'll need to add a method for this)
        // For now, just report the configuration
        println!("{}: {}", name, config.num_tables);
    }

    println!("\n=== END CANDIDATE ANALYSIS ===\n");
}

criterion_group!(
    benches,
    bench_lsh_scaling,
    bench_lsh_build_time,
    bench_lsh_configs,
    bench_lsh_accuracy,
    bench_lsh_constant_scaling,
    bench_lsh_candidates,
);

criterion_main!(benches);
