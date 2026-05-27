// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SIMD Continuous HV Benchmark Suite
//!
//! Benchmarks for SIMD-accelerated f32 hypervector operations:
//! - dot_product_simd vs scalar
//! - bind_simd vs scalar
//! - bundle_simd vs scalar
//! - similarity_simd vs scalar
//!
//! Target: 4x+ speedup for 16,384-dim vectors
//!
//! Run with:
//!   CARGO_TARGET_DIR=/tmp/symthaea-target cargo bench --bench simd_continuous --features simd

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

const HDC_DIM: usize = 16_384;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// Scalar baseline implementations for comparison
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn scalar_bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

fn scalar_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_ab = 0.0f32;
    let mut dot_aa = 0.0f32;
    let mut dot_bb = 0.0f32;

    for (&av, &bv) in a.iter().zip(b.iter()) {
        dot_ab += av * bv;
        dot_aa += av * av;
        dot_bb += bv * bv;
    }

    let denom = (dot_aa * dot_bb).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (dot_ab / denom).clamp(-1.0, 1.0)
    }
}

fn scalar_bundle(hvs: &[&[f32]], weights: &[f32]) -> Vec<f32> {
    if hvs.is_empty() || weights.is_empty() {
        return Vec::new();
    }

    let dim = hvs[0].len();
    let weight_sum: f32 = weights.iter().sum();
    let inv_weight_sum = if weight_sum.abs() > 1e-10 {
        1.0 / weight_sum
    } else {
        0.0
    };

    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        for (hv, &weight) in hvs.iter().zip(weights.iter()) {
            sum += hv[i] * weight;
        }
        result[i] = sum * inv_weight_sum;
    }
    result
}

fn scalar_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

// =============================================================================
// DOT PRODUCT BENCHMARKS
// =============================================================================

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for dim in [1024, 4096, HDC_DIM, 32768] {
        let a = random_vec(dim, 42);
        let b = random_vec(dim, 43);

        group.bench_with_input(
            BenchmarkId::new("scalar", dim),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(scalar_dot_product(a, b))),
        );

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", dim), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(symthaea_core::hdc::simd_continuous::dot_product_simd(a, b)))
        });
    }

    group.finish();
}

// =============================================================================
// BIND BENCHMARKS
// =============================================================================

fn bench_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind");

    for dim in [1024, 4096, HDC_DIM, 32768] {
        let a = random_vec(dim, 42);
        let b = random_vec(dim, 43);

        group.bench_with_input(
            BenchmarkId::new("scalar", dim),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(scalar_bind(a, b))),
        );

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", dim), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(symthaea_core::hdc::simd_continuous::bind_simd(a, b)))
        });
    }

    group.finish();
}

// =============================================================================
// SIMILARITY BENCHMARKS
// =============================================================================

fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    for dim in [1024, 4096, HDC_DIM, 32768] {
        let a = random_vec(dim, 42);
        let b = random_vec(dim, 43);

        group.bench_with_input(
            BenchmarkId::new("scalar", dim),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(scalar_similarity(a, b))),
        );

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", dim), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(symthaea_core::hdc::simd_continuous::similarity_simd(a, b)))
        });
    }

    group.finish();
}

// =============================================================================
// BUNDLE BENCHMARKS
// =============================================================================

fn bench_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle");

    // Test with varying number of vectors to bundle
    for n_vecs in [3, 10, 50, 100] {
        let vecs: Vec<Vec<f32>> = (0..n_vecs).map(|i| random_vec(HDC_DIM, i + 100)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let weights: Vec<f32> = (0..n_vecs).map(|i| 1.0 + (i as f32) * 0.1).collect();

        group.bench_with_input(
            BenchmarkId::new("scalar", n_vecs),
            &(&refs, &weights),
            |bench, (refs, weights)| bench.iter(|| black_box(scalar_bundle(refs, weights))),
        );

        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", n_vecs),
            &(&refs, &weights),
            |bench, (refs, weights)| {
                bench.iter(|| {
                    black_box(symthaea_core::hdc::simd_continuous::bundle_simd(
                        refs, weights,
                    ))
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// NORM BENCHMARKS
// =============================================================================

fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm");

    for dim in [1024, 4096, HDC_DIM, 32768] {
        let a = random_vec(dim, 42);

        group.bench_with_input(BenchmarkId::new("scalar", dim), &a, |bench, a| {
            bench.iter(|| black_box(scalar_norm(a)))
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", dim), &a, |bench, a| {
            bench.iter(|| black_box(symthaea_core::hdc::simd_continuous::norm_simd(a)))
        });
    }

    group.finish();
}

// =============================================================================
// ContinuousHV INTEGRATION BENCHMARKS (with simd feature)
// =============================================================================

#[cfg(feature = "simd")]
fn bench_continuous_hv_integration(c: &mut Criterion) {
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    let mut group = c.benchmark_group("ContinuousHV_integration");

    let a = ContinuousHV::random(HDC_DIM, 42);
    let b = ContinuousHV::random(HDC_DIM, 43);

    group.bench_function("bind", |bench| bench.iter(|| black_box(a.bind(&b))));

    group.bench_function("similarity", |bench| {
        bench.iter(|| black_box(a.similarity(&b)))
    });

    group.bench_function("norm", |bench| bench.iter(|| black_box(a.norm())));

    // Weighted bundle
    let vecs: Vec<ContinuousHV> = (0..10)
        .map(|i| ContinuousHV::random(HDC_DIM, i + 100))
        .collect();
    let refs: Vec<&ContinuousHV> = vecs.iter().collect();
    let weights: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.1).collect();

    group.bench_function("weighted_bundle_10", |bench| {
        bench.iter(|| black_box(ContinuousHV::weighted_bundle(&refs, &weights)))
    });

    group.finish();
}

// =============================================================================
// COMPREHENSIVE SPEEDUP REPORT
// =============================================================================

fn bench_speedup_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_summary_16K");
    group.sample_size(100);

    let a = random_vec(HDC_DIM, 42);
    let b = random_vec(HDC_DIM, 43);

    // Dot product
    group.bench_function("dot_scalar", |bench| {
        bench.iter(|| black_box(scalar_dot_product(&a, &b)))
    });

    #[cfg(feature = "simd")]
    group.bench_function("dot_simd", |bench| {
        bench.iter(|| {
            black_box(symthaea_core::hdc::simd_continuous::dot_product_simd(
                &a, &b,
            ))
        })
    });

    // Similarity
    group.bench_function("sim_scalar", |bench| {
        bench.iter(|| black_box(scalar_similarity(&a, &b)))
    });

    #[cfg(feature = "simd")]
    group.bench_function("sim_simd", |bench| {
        bench.iter(|| black_box(symthaea_core::hdc::simd_continuous::similarity_simd(&a, &b)))
    });

    // Bind
    group.bench_function("bind_scalar", |bench| {
        bench.iter(|| black_box(scalar_bind(&a, &b)))
    });

    #[cfg(feature = "simd")]
    group.bench_function("bind_simd", |bench| {
        bench.iter(|| black_box(symthaea_core::hdc::simd_continuous::bind_simd(&a, &b)))
    });

    group.finish();
}

// =============================================================================
// CRITERION MAIN
// =============================================================================

#[cfg(feature = "simd")]
criterion_group!(
    benches,
    bench_dot_product,
    bench_bind,
    bench_similarity,
    bench_bundle,
    bench_norm,
    bench_continuous_hv_integration,
    bench_speedup_summary,
);

#[cfg(not(feature = "simd"))]
criterion_group!(
    benches,
    bench_dot_product,
    bench_bind,
    bench_similarity,
    bench_bundle,
    bench_norm,
    bench_speedup_summary,
);

criterion_main!(benches);
