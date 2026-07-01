// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC Improvements Benchmark Suite
//!
//! Benchmarks for all Phase 1-4 improvements:
//! 1. bundle_safe vs bundle (stack-safe version)
//! 2. bundle_normalized (with density correction)
//! 3. permute vs permute_legacy (word-level rotation is now default)
//! 4. SparseHV operations
//!
//! Run with: cargo bench --bench hdc_improvements

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::sparse_hv::SparseHV;

// ============================================================================
// BinaryHV BUNDLE BENCHMARKS
// ============================================================================

fn bench_hv16_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinaryHV_bundle");

    for n in [10, 50, 100, 500] {
        let vectors: Vec<BinaryHV> = (0..n).map(|i| BinaryHV::random(i)).collect();

        group.bench_with_input(BenchmarkId::new("original", n), &vectors, |b, vecs| {
            b.iter(|| black_box(BinaryHV::bundle(vecs)))
        });

        group.bench_with_input(BenchmarkId::new("safe", n), &vectors, |b, vecs| {
            b.iter(|| black_box(BinaryHV::bundle_safe(vecs)))
        });

        group.bench_with_input(BenchmarkId::new("normalized", n), &vectors, |b, vecs| {
            b.iter(|| black_box(BinaryHV::bundle_normalized(vecs)))
        });
    }

    group.finish();
}

// ============================================================================
// BinaryHV PERMUTE BENCHMARKS
// ============================================================================

fn bench_hv16_permute(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinaryHV_permute");

    let v = BinaryHV::random(42);

    for shift in [1, 8, 64, 128, 1000] {
        // permute_legacy: original bit-by-bit implementation
        group.bench_with_input(BenchmarkId::new("legacy", shift), &shift, |b, &s| {
            b.iter(|| black_box(v.permute_legacy(s)))
        });

        // permute: now uses fast word-level rotation (13-22x faster)
        group.bench_with_input(BenchmarkId::new("fast", shift), &shift, |b, &s| {
            b.iter(|| black_box(v.permute(s)))
        });
    }

    group.finish();
}

// ============================================================================
// BinaryHV DENSITY BENCHMARKS
// ============================================================================

fn bench_hv16_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinaryHV_density");

    let random = BinaryHV::random(42);
    let ones = BinaryHV::ones();

    group.bench_function("density_random", |b| b.iter(|| black_box(random.density())));

    group.bench_function("ensure_density_balanced", |b| {
        b.iter(|| black_box(random.ensure_density(0.4, 0.6)))
    });

    group.bench_function("ensure_density_saturated", |b| {
        b.iter(|| black_box(ones.ensure_density(0.4, 0.6)))
    });

    group.finish();
}

// ============================================================================
// SPARSE HV BENCHMARKS
// ============================================================================

fn bench_sparse_hv(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparseHV");

    // Test at different densities
    for density in [0.01, 0.1, 0.3, 0.5] {
        let sparse_a = SparseHV::random(1, density);
        let sparse_b = SparseHV::random(2, density);

        group.bench_with_input(
            BenchmarkId::new("jaccard_similarity", format!("{:.0}%", density * 100.0)),
            &(&sparse_a, &sparse_b),
            |b, (a, bb)| b.iter(|| black_box(a.similarity_jaccard(bb))),
        );

        group.bench_with_input(
            BenchmarkId::new("lsh_similarity", format!("{:.0}%", density * 100.0)),
            &(&sparse_a, &sparse_b),
            |b, (a, bb)| b.iter(|| black_box(a.similarity_lsh(bb))),
        );

        group.bench_with_input(
            BenchmarkId::new("bind", format!("{:.0}%", density * 100.0)),
            &(&sparse_a, &sparse_b),
            |b, (a, bb)| b.iter(|| black_box(a.bind(bb))),
        );

        group.bench_with_input(
            BenchmarkId::new("permute", format!("{:.0}%", density * 100.0)),
            &sparse_a,
            |b, a| b.iter(|| black_box(a.permute(100))),
        );
    }

    group.finish();
}

fn bench_sparse_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparseHV_conversion");

    let dense = BinaryHV::random(42);
    let sparse = SparseHV::from_hv16(&dense);

    group.bench_function("from_hv16", |b| {
        b.iter(|| black_box(SparseHV::from_hv16(&dense)))
    });

    group.bench_function("to_hv16", |b| b.iter(|| black_box(sparse.to_hv16())));

    group.finish();
}

fn bench_sparse_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparseHV_bundle");

    for n in [5, 10, 20] {
        let vectors: Vec<SparseHV> = (0..n).map(|i| SparseHV::random(i, 0.3)).collect();

        group.bench_with_input(BenchmarkId::new("bundle", n), &vectors, |b, vecs| {
            b.iter(|| black_box(SparseHV::bundle(vecs)))
        });
    }

    group.finish();
}

// ============================================================================
// COMPARISON BENCHMARKS
// ============================================================================

fn bench_bind_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind_comparison");

    let a = BinaryHV::random(1);
    let b = BinaryHV::random(2);
    let sparse_a = SparseHV::from_hv16(&a);
    let sparse_b = SparseHV::from_hv16(&b);

    group.bench_function("BinaryHV", |bench| bench.iter(|| black_box(a.bind(&b))));

    group.bench_function("SparseHV_50%", |bench| {
        bench.iter(|| black_box(sparse_a.bind(&sparse_b)))
    });

    // Low density sparse
    let low_sparse_a = SparseHV::random(1, 0.05);
    let low_sparse_b = SparseHV::random(2, 0.05);

    group.bench_function("SparseHV_5%", |bench| {
        bench.iter(|| black_box(low_sparse_a.bind(&low_sparse_b)))
    });

    group.finish();
}

fn bench_similarity_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_comparison");

    let a = BinaryHV::random(1);
    let b = BinaryHV::random(2);
    let sparse_a = SparseHV::from_hv16(&a);
    let sparse_b = SparseHV::from_hv16(&b);

    group.bench_function("BinaryHV_hamming", |bench| {
        bench.iter(|| black_box(a.similarity(&b)))
    });

    group.bench_function("SparseHV_jaccard", |bench| {
        bench.iter(|| black_box(sparse_a.similarity_jaccard(&sparse_b)))
    });

    group.bench_function("SparseHV_lsh", |bench| {
        bench.iter(|| black_box(sparse_a.similarity_lsh(&sparse_b)))
    });

    group.finish();
}

// ============================================================================
// CRITERION MAIN
// ============================================================================

criterion_group!(
    benches,
    bench_hv16_bundle,
    bench_hv16_permute,
    bench_hv16_density,
    bench_sparse_hv,
    bench_sparse_conversion,
    bench_sparse_bundle,
    bench_bind_comparison,
    bench_similarity_comparison,
);

criterion_main!(benches);
