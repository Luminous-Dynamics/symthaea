//! SIMD Performance Benchmark Suite
//!
//! Measures ACTUAL performance of SIMD-accelerated HV16 operations
//! and compares against original scalar implementations.
//!
//! Run with: cargo bench --bench simd_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::simd_hv::{
    simd_bind, simd_similarity, simd_bundle, simd_hamming_distance,
    simd_find_most_similar, has_avx2, has_sse2, simd_capabilities,
};
use symthaea::hdc::optimized_hv::{
    bundle_optimized, similarity_optimized, bind_optimized,
};

// ============================================================================
// BIND BENCHMARKS - Target: <10ns with AVX2
// ============================================================================

fn bench_bind_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("HV16 Bind Operations");

    // Create test vectors
    let a = HV16::random();
    let b = HV16::random();

    // Original HV16::bind method
    group.bench_function("original_bind", |bencher| {
        bencher.iter(|| {
            black_box(a.bind(black_box(&b)))
        })
    });

    // Optimized (unrolled) bind
    group.bench_function("optimized_bind", |bencher| {
        bencher.iter(|| {
            black_box(bind_optimized(black_box(&a), black_box(&b)))
        })
    });

    // SIMD bind (auto-dispatched)
    group.bench_function("simd_bind", |bencher| {
        bencher.iter(|| {
            black_box(simd_bind(black_box(&a), black_box(&b)))
        })
    });

    group.finish();
}

/// Batch bind benchmark - measures throughput
fn bench_bind_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("HV16 Bind Batch");

    for batch_size in [10, 100, 1000].iter() {
        let vectors: Vec<HV16> = (0..*batch_size).map(|_| HV16::random()).collect();
        let key = HV16::random();

        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("original", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    for v in &vectors {
                        black_box(v.bind(&key));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd", batch_size),
            batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    for v in &vectors {
                        black_box(simd_bind(v, &key));
                    }
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// SIMILARITY BENCHMARKS - Target: <25ns with AVX2
// ============================================================================

fn bench_similarity_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("HV16 Similarity Operations");

    let a = HV16::random();
    let b = HV16::random();

    // Original HV16::similarity method
    group.bench_function("original_similarity", |bencher| {
        bencher.iter(|| {
            black_box(a.similarity(black_box(&b)))
        })
    });

    // Optimized similarity
    group.bench_function("optimized_similarity", |bencher| {
        bencher.iter(|| {
            black_box(similarity_optimized(black_box(&a), black_box(&b)))
        })
    });

    // SIMD similarity
    group.bench_function("simd_similarity", |bencher| {
        bencher.iter(|| {
            black_box(simd_similarity(black_box(&a), black_box(&b)))
        })
    });

    // SIMD Hamming distance (raw bit count)
    group.bench_function("simd_hamming", |bencher| {
        bencher.iter(|| {
            black_box(simd_hamming_distance(black_box(&a), black_box(&b)))
        })
    });

    group.finish();
}

/// Similarity search benchmark - find most similar in memory
fn bench_similarity_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("HV16 Similarity Search");

    for memory_size in [10, 100, 1000].iter() {
        let memory: Vec<HV16> = (0..*memory_size).map(|_| HV16::random()).collect();
        let query = HV16::random();

        group.throughput(Throughput::Elements(*memory_size as u64));

        // Linear search (original)
        group.bench_with_input(
            BenchmarkId::new("linear_search", memory_size),
            memory_size,
            |bencher, _| {
                bencher.iter(|| {
                    let mut best_idx = 0;
                    let mut best_sim = f32::MIN;
                    for (idx, v) in memory.iter().enumerate() {
                        let sim = v.similarity(&query);
                        if sim > best_sim {
                            best_sim = sim;
                            best_idx = idx;
                        }
                    }
                    black_box((best_idx, best_sim))
                })
            },
        );

        // SIMD optimized search
        group.bench_with_input(
            BenchmarkId::new("simd_search", memory_size),
            memory_size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(simd_find_most_similar(&query, &memory))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// BUNDLE BENCHMARKS - Target: <5µs for 10 vectors with AVX2
// ============================================================================

fn bench_bundle_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("HV16 Bundle Operations");

    for num_vectors in [3, 5, 10, 50, 100].iter() {
        let vectors: Vec<HV16> = (0..*num_vectors).map(|_| HV16::random()).collect();

        group.throughput(Throughput::Elements(*num_vectors as u64));

        // Original HV16::bundle method
        group.bench_with_input(
            BenchmarkId::new("original_bundle", num_vectors),
            &vectors,
            |bencher, vecs| {
                bencher.iter(|| {
                    black_box(HV16::bundle(black_box(vecs)))
                })
            },
        );

        // Optimized bundle
        group.bench_with_input(
            BenchmarkId::new("optimized_bundle", num_vectors),
            &vectors,
            |bencher, vecs| {
                bencher.iter(|| {
                    black_box(bundle_optimized(black_box(vecs)))
                })
            },
        );

        // SIMD bundle
        group.bench_with_input(
            BenchmarkId::new("simd_bundle", num_vectors),
            &vectors,
            |bencher, vecs| {
                bencher.iter(|| {
                    black_box(simd_bundle(black_box(vecs)))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// CORRECTNESS VERIFICATION (ensures SIMD produces same results)
// ============================================================================

fn verify_correctness() {
    println!("\n=== SIMD Correctness Verification ===\n");

    let a = HV16::random();
    let b = HV16::random();

    // Test bind
    let original_bind = a.bind(&b);
    let simd_bind_result = simd_bind(&a, &b);
    assert_eq!(original_bind.0, simd_bind_result.0, "SIMD bind mismatch!");
    println!("✓ Bind: SIMD matches original");

    // Test similarity
    let original_sim = a.similarity(&b);
    let simd_sim = simd_similarity(&a, &b);
    let diff = (original_sim - simd_sim).abs();
    assert!(diff < 0.0001, "SIMD similarity mismatch: {} vs {}", original_sim, simd_sim);
    println!("✓ Similarity: SIMD matches original (diff: {:.6})", diff);

    // Test bundle
    let vectors: Vec<HV16> = (0..10).map(|_| HV16::random()).collect();
    let original_bundle = HV16::bundle(&vectors);
    let simd_bundle_result = simd_bundle(&vectors);
    assert_eq!(original_bundle.0, simd_bundle_result.0, "SIMD bundle mismatch!");
    println!("✓ Bundle: SIMD matches original");

    println!("\n✅ All SIMD operations produce correct results!\n");
}

// ============================================================================
// CAPABILITY REPORT
// ============================================================================

fn print_capabilities() {
    println!("\n=== SIMD Capability Report ===\n");
    println!("{}", simd_capabilities());
    println!();
}

// ============================================================================
// MAIN BENCHMARK GROUPS
// ============================================================================

criterion_group!(
    name = simd_benches;
    config = Criterion::default()
        .sample_size(1000)  // More samples for accurate nanosecond measurements
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_bind_comparison,
        bench_bind_batch,
        bench_similarity_comparison,
        bench_similarity_search,
        bench_bundle_comparison
);

fn main() {
    // Print capabilities before benchmarks
    print_capabilities();

    // Verify correctness before benchmarking
    verify_correctness();

    // Run benchmarks
    simd_benches();

    // Final summary
    println!("\n=== Performance Summary ===");
    println!("AVX2 available: {}", has_avx2());
    println!("SSE2 available: {}", has_sse2());
    println!("\nTarget Performance (with AVX2):");
    println!("  bind:       <10ns");
    println!("  similarity: <25ns");
    println!("  bundle(10): <5µs");

    Criterion::default()
        .configure_from_args()
        .final_summary();
}
