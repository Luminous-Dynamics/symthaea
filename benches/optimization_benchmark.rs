//! Optimization Benchmark - Compare Original vs Optimized Implementations
//!
//! This benchmark validates the performance improvements from:
//! - optimized_hv: 10-100x faster HV16 operations
//! - sparse_ltc: 10-100x faster LTC networks
//!
//! Run with: cargo bench --bench optimization_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::optimized_hv::{bundle_optimized, permute_optimized, similarity_optimized, bind_optimized};
use symthaea::ltc::LiquidNetwork;
use symthaea::sparse_ltc::SparseLiquidNetwork;

/// Benchmark bundle: Original vs Optimized
fn bench_bundle_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bundle Comparison");

    for num_vectors in [10, 50, 100] {
        let vectors: Vec<HV16> = (0..num_vectors).map(|i| HV16::random(i)).collect();

        group.bench_with_input(
            BenchmarkId::new("Original", num_vectors),
            &vectors,
            |b, vecs| b.iter(|| black_box(HV16::bundle(vecs)))
        );

        group.bench_with_input(
            BenchmarkId::new("Optimized", num_vectors),
            &vectors,
            |b, vecs| b.iter(|| black_box(bundle_optimized(vecs)))
        );
    }

    group.finish();
}

/// Benchmark permute: Original vs Optimized
fn bench_permute_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Permute Comparison");
    let hv = HV16::random(42);

    // Test aligned shifts (multiples of 8 - should be fastest)
    for shift in [8, 16, 64, 256] {
        group.bench_with_input(
            BenchmarkId::new("Original/aligned", shift),
            &shift,
            |b, &s| b.iter(|| black_box(hv.permute(s)))
        );

        group.bench_with_input(
            BenchmarkId::new("Optimized/aligned", shift),
            &shift,
            |b, &s| b.iter(|| black_box(permute_optimized(&hv, s)))
        );
    }

    // Test unaligned shifts
    for shift in [1, 7, 13, 127] {
        group.bench_with_input(
            BenchmarkId::new("Original/unaligned", shift),
            &shift,
            |b, &s| b.iter(|| black_box(hv.permute(s)))
        );

        group.bench_with_input(
            BenchmarkId::new("Optimized/unaligned", shift),
            &shift,
            |b, &s| b.iter(|| black_box(permute_optimized(&hv, s)))
        );
    }

    group.finish();
}

/// Benchmark similarity: Original vs Optimized
fn bench_similarity_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Similarity Comparison");
    let a = HV16::random(42);
    let b = HV16::random(43);

    group.bench_function("Original", |bencher| {
        bencher.iter(|| black_box(a.similarity(&b)))
    });

    group.bench_function("Optimized", |bencher| {
        bencher.iter(|| black_box(similarity_optimized(&a, &b)))
    });

    group.finish();
}

/// Benchmark bind: Original vs Optimized
fn bench_bind_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bind Comparison");
    let a = HV16::random(42);
    let b = HV16::random(43);

    group.bench_function("Original", |bencher| {
        bencher.iter(|| black_box(a.bind(&b)))
    });

    group.bench_function("Optimized", |bencher| {
        bencher.iter(|| black_box(bind_optimized(&a, &b)))
    });

    group.finish();
}

/// Benchmark LTC step: Dense vs Sparse
fn bench_ltc_step_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("LTC Step Comparison");

    for neurons in [100, 250, 500, 1000] {
        let mut dense = LiquidNetwork::new(neurons).unwrap();
        let mut sparse = SparseLiquidNetwork::new(neurons).unwrap();

        group.bench_with_input(
            BenchmarkId::new("Dense", neurons),
            &neurons,
            |b, _| b.iter(|| black_box(dense.step().unwrap()))
        );

        group.bench_with_input(
            BenchmarkId::new("Sparse", neurons),
            &neurons,
            |b, _| b.iter(|| black_box(sparse.step().unwrap()))
        );
    }

    group.finish();
}

/// Benchmark LTC full cycle: Dense vs Sparse
fn bench_ltc_cycle_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("LTC Full Cycle (10 steps)");

    for neurons in [100, 500, 1000] {
        let mut dense = LiquidNetwork::new(neurons).unwrap();
        let mut sparse = SparseLiquidNetwork::new(neurons).unwrap();

        group.bench_with_input(
            BenchmarkId::new("Dense", neurons),
            &neurons,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        dense.step().unwrap();
                    }
                    black_box(dense.consciousness_level())
                })
            }
        );

        group.bench_with_input(
            BenchmarkId::new("Sparse", neurons),
            &neurons,
            |b, _| {
                b.iter(|| {
                    for _ in 0..10 {
                        sparse.step().unwrap();
                    }
                    black_box(sparse.consciousness_level())
                })
            }
        );
    }

    group.finish();
}

/// Benchmark consciousness level: Dense vs Sparse
fn bench_consciousness_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Consciousness Level");

    for neurons in [100, 500, 1000] {
        let mut dense = LiquidNetwork::new(neurons).unwrap();
        let mut sparse = SparseLiquidNetwork::new(neurons).unwrap();

        // Run some steps first
        for _ in 0..10 {
            let _ = dense.step();
            let _ = sparse.step();
        }

        group.bench_with_input(
            BenchmarkId::new("Dense", neurons),
            &neurons,
            |b, _| b.iter(|| black_box(dense.consciousness_level()))
        );

        group.bench_with_input(
            BenchmarkId::new("Sparse", neurons),
            &neurons,
            |b, _| b.iter(|| black_box(sparse.consciousness_level()))
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bundle_comparison,
    bench_permute_comparison,
    bench_similarity_comparison,
    bench_bind_comparison,
    bench_ltc_step_comparison,
    bench_ltc_cycle_comparison,
    bench_consciousness_comparison,
);

criterion_main!(benches);
