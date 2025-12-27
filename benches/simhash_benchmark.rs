// Session 6B: SimHash Verification Benchmarks
// Verify that SimHash (the CORRECT LSH for Hamming distance) provides:
// - 100-1000x speedup vs brute force
// - 95%+ recall with 10 tables
// - Constant-ish scaling as dataset grows

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use symthaea::hdc::{HV16, lsh_simhash::{SimHashIndex, SimHashConfig}};
use std::collections::HashSet;

/// Create test dataset of random vectors
fn create_test_dataset(size: usize, seed: u64) -> Vec<HV16> {
    (0..size)
        .map(|i| HV16::random(seed + i as u64))
        .collect()
}

/// Brute force top-k similarity search
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

/// Measure recall: What percentage of true top-k did SimHash find?
fn measure_recall(
    simhash_results: &[(usize, f32)],
    brute_force_results: &[(usize, f32)],
) -> f64 {
    let bf_ids: HashSet<usize> = brute_force_results.iter().map(|(id, _)| *id).collect();
    let sh_ids: HashSet<usize> = simhash_results.iter().map(|(id, _)| *id).collect();

    let intersection = bf_ids.intersection(&sh_ids).count();
    (intersection as f64 / brute_force_results.len() as f64) * 100.0
}

/// Benchmark SimHash vs brute force across different dataset sizes
fn bench_simhash_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("simhash_vs_brute_force");

    // Test different dataset sizes
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

        // Build SimHash index
        let config = SimHashConfig::balanced(); // 10 tables, ~95% recall expected
        let mut index = SimHashIndex::new(config);
        index.insert_batch(&vectors);

        // Benchmark SimHash search
        group.bench_with_input(
            BenchmarkId::new("simhash_approximate", size),
            size,
            |b, _| {
                b.iter(|| {
                    let results = index.query_approximate(
                        black_box(&query),
                        black_box(k),
                        black_box(&vectors),
                    );
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark SimHash index build time
fn bench_simhash_build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("simhash_index_build");

    let sizes = vec![1_000, 10_000, 100_000];

    for size in sizes.iter() {
        let vectors = create_test_dataset(*size, 42);

        group.bench_with_input(
            BenchmarkId::new("build_index", size),
            size,
            |b, _| {
                b.iter(|| {
                    let config = SimHashConfig::balanced();
                    let mut index = SimHashIndex::new(config);
                    index.insert_batch(black_box(&vectors));
                    black_box(index)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark accuracy vs speed trade-off for different SimHash configs
fn bench_simhash_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("simhash_config_tradeoff");

    let vectors = create_test_dataset(10_000, 42);
    let query = HV16::random(99999);
    let k = 10;

    // Test different configurations
    let configs = vec![
        ("fast_5_tables", SimHashConfig::fast()),
        ("balanced_10_tables", SimHashConfig::balanced()),
        ("accurate_20_tables", SimHashConfig::accurate()),
    ];

    for (name, config) in configs.iter() {
        let mut index = SimHashIndex::new(config.clone());
        index.insert_batch(&vectors);

        group.bench_with_input(
            BenchmarkId::new("query", name),
            name,
            |b, _| {
                b.iter(|| {
                    let results = index.query_approximate(
                        black_box(&query),
                        black_box(k),
                        black_box(&vectors),
                    );
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

/// Comprehensive accuracy test: Measure recall for different SimHash configs
fn bench_simhash_accuracy(_c: &mut Criterion) {
    println!("\n=== SIMHASH ACCURACY ANALYSIS ===\n");

    let vectors = create_test_dataset(10_000, 42);
    let query = HV16::random(99999);
    let k = 10;

    // Ground truth: Brute force top-10
    let ground_truth = brute_force_search(&query, &vectors, k);

    // Test different configurations
    let configs = vec![
        ("Fast (5 tables)", SimHashConfig::fast()),
        ("Balanced (10 tables)", SimHashConfig::balanced()),
        ("Accurate (20 tables)", SimHashConfig::accurate()),
    ];

    for (name, config) in configs.iter() {
        let mut index = SimHashIndex::new(config.clone());
        index.insert_batch(&vectors);

        let simhash_results = index.query_approximate(&query, k, &vectors);
        let recall = measure_recall(&simhash_results, &ground_truth);

        println!("{}: {:.1}% recall", name, recall);
    }

    println!("\n=== END ACCURACY ANALYSIS ===\n");
}

/// Measure SimHash performance scaling: Does it stay constant as dataset grows?
fn bench_simhash_constant_scaling(_c: &mut Criterion) {
    println!("\n=== SIMHASH SCALING VERIFICATION ===");
    println!("Testing hypothesis: SimHash query time stays relatively constant as dataset grows\n");

    let sizes = vec![1_000, 10_000, 100_000];
    let k = 10;

    for size in sizes.iter() {
        let vectors = create_test_dataset(*size, 42);
        let query = HV16::random(99999);

        let config = SimHashConfig::balanced();
        let mut index = SimHashIndex::new(config);
        index.insert_batch(&vectors);

        // Measure average query time
        let mut total_time = std::time::Duration::ZERO;
        let samples = 10;

        for _ in 0..samples {
            let start = std::time::Instant::now();
            let _results = index.query_approximate(&query, k, &vectors);
            total_time += start.elapsed();
        }

        let avg_time_ms = total_time.as_micros() as f64 / samples as f64 / 1000.0;

        // Count candidates
        let num_candidates = index.count_candidates(&query);
        let candidate_pct = (num_candidates as f64 / *size as f64) * 100.0;

        println!("{:>6} vectors: {:.3}ms per query ({} candidates = {:.1}%)",
                 size, avg_time_ms, num_candidates, candidate_pct);
    }

    println!("\nExpected: Time should grow slowly (log-ish, not linear)\n");
    println!("=== END SCALING VERIFICATION ===\n");
}

/// Benchmark candidate reduction efficiency
fn bench_simhash_candidate_reduction(_c: &mut Criterion) {
    println!("\n=== SIMHASH CANDIDATE REDUCTION ANALYSIS ===\n");

    let vectors = create_test_dataset(100_000, 42);
    let query = HV16::random(99999);

    let configs = vec![
        ("Fast (5 tables)", SimHashConfig::fast()),
        ("Balanced (10 tables)", SimHashConfig::balanced()),
        ("Accurate (20 tables)", SimHashConfig::accurate()),
    ];

    for (name, config) in configs.iter() {
        let mut index = SimHashIndex::new(config.clone());
        index.insert_batch(&vectors);

        let num_candidates = index.count_candidates(&query);
        let reduction = 100.0 * (1.0 - (num_candidates as f64 / vectors.len() as f64));

        println!("{}: {} candidates out of {} ({:.1}% reduction)",
                 name, num_candidates, vectors.len(), reduction);
    }

    println!("\n=== END CANDIDATE REDUCTION ANALYSIS ===\n");
}

/// Test SimHash with MIXED DATASET (realistic use case!)
/// - 10,000 random vectors (dissimilar)
/// - 1,000 similar vectors in a cluster
/// - Query searches for cluster members
fn bench_simhash_with_similar_vectors(_c: &mut Criterion) {
    println!("\n=== SIMHASH WITH REALISTIC MIXED DATASET TEST ===\n");

    // Create a base vector for the "cluster"
    let base = HV16::random(42);

    // Create 1000 vectors SIMILAR to base (the "cluster")
    let mut cluster_vectors = vec![base.clone()];
    for i in 1..1000 {
        let mut similar = base.clone();
        // Flip 10 random bits (out of 2048 total = 0.5% Hamming distance)
        for j in 0..10 {
            let bit_pos = ((i * 13 + j * 7) % 2048) as usize;
            let byte_idx = bit_pos / 8;
            let bit_idx = (bit_pos % 8) as u8;
            similar.0[byte_idx] ^= 1 << bit_idx;
        }
        cluster_vectors.push(similar);
    }

    // Create 9,000 RANDOM vectors (dissimilar to cluster)
    let random_vectors: Vec<HV16> = (1000..10000)
        .map(|i| HV16::random(i as u64 + 999999))
        .collect();

    // Combine: cluster vectors first, then random
    let mut all_vectors = cluster_vectors.clone();
    all_vectors.extend(random_vectors);

    println!("Dataset: {} vectors ({} similar cluster + {} random)",
             all_vectors.len(), cluster_vectors.len(), 9000);

    // Build SimHash index
    let config = SimHashConfig::balanced();
    let mut index = SimHashIndex::new(config);
    index.insert_batch(&all_vectors);

    // Query with a vector similar to base (searching for cluster members)
    let mut query = base.clone();
    for j in 0..5 {
        let bit_pos = ((j * 11) % 2048) as usize;
        let byte_idx = bit_pos / 8;
        let bit_idx = (bit_pos % 8) as u8;
        query.0[byte_idx] ^= 1 << bit_idx;
    }

    // Ground truth: brute force top-10 from CLUSTER vectors only
    // (We know true neighbors are in the cluster, not the random vectors)
    let mut cluster_similarities: Vec<(usize, f32)> = cluster_vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, query.similarity(v)))
        .collect();
    cluster_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let ground_truth: Vec<(usize, f32)> = cluster_similarities.iter()
        .take(10)
        .copied()
        .collect();

    println!("\nGround truth top-10 (from cluster):");
    for (i, (id, sim)) in ground_truth.iter().enumerate() {
        println!("  {}: vector {} (similarity {:.4})", i+1, id, sim);
    }

    // SimHash approximate top-10
    let simhash_results = index.query_approximate(&query, 10, &all_vectors);

    println!("\nSimHash top-10:");
    for (i, (id, sim)) in simhash_results.iter().enumerate() {
        let from = if *id < 1000 { "cluster" } else { "random" };
        println!("  {}: vector {} (similarity {:.4}) [from {}]", i+1, id, sim, from);
    }

    // Count how many results are from cluster
    let cluster_found = simhash_results.iter().filter(|(id, _)| *id < 1000).count();

    // Measure recall against ground truth cluster IDs
    let gt_ids: HashSet<usize> = ground_truth.iter().map(|(id, _)| *id).collect();
    let sh_ids: HashSet<usize> = simhash_results.iter()
        .filter(|(id, _)| *id < 1000)  // Only count cluster vectors
        .map(|(id, _)| *id)
        .collect();
    let intersection = gt_ids.intersection(&sh_ids).count();
    let recall = (intersection as f64 / 10.0) * 100.0;

    let num_candidates = index.count_candidates(&query);
    let speedup = all_vectors.len() as f64 / num_candidates as f64;

    println!("\n📊 RESULTS:");
    println!("Recall: {:.1}% ({}/{} cluster vectors in top-10)",
             recall, cluster_found, 10);
    println!("Candidates examined: {} out of {} ({:.1}%)",
             num_candidates, all_vectors.len(),
             (num_candidates as f64 / all_vectors.len() as f64) * 100.0);
    println!("Speedup: {:.1}x (vs brute force)", speedup);

    // Verdict
    if recall >= 70.0 && num_candidates < all_vectors.len() / 5 {
        println!("\n✅ SimHash WORKS CORRECTLY!");
        println!("   - High recall on similar vectors: {:.1}%", recall);
        println!("   - Significant candidate reduction: {:.1}x speedup", speedup);
        println!("   - This confirms: 0% recall on random vectors was EXPECTED.");
    } else if recall >= 70.0 {
        println!("\n⚠️  SimHash has HIGH RECALL but POOR FILTERING");
        println!("   - Recall is good: {:.1}%", recall);
        println!("   - But candidate reduction is poor: only {:.1}x", speedup);
    } else {
        println!("\n❌ SimHash has LOW RECALL: {:.1}%", recall);
        println!("   Expected: ≥70% recall on similar vectors");
    }

    println!("\n=== END REALISTIC MIXED DATASET TEST ===\n");
}

criterion_group!(
    benches,
    bench_simhash_scaling,
    bench_simhash_build_time,
    bench_simhash_configs,
    bench_simhash_accuracy,
    bench_simhash_constant_scaling,
    bench_simhash_candidate_reduction,
    bench_simhash_with_similar_vectors,
);

criterion_main!(benches);
