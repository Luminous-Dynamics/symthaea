// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Entropy Methods Benchmark Suite
//!
//! Benchmarks for various entropy estimation methods:
//! 1. Histogram-based entropy (baseline)
//! 2. k-NN entropy (Kozachenko-Leonenko)
//! 3. k-NN fast (sorted optimization)
//! 4. KDE entropy
//! 5. KDE fast (truncated Gaussian)
//! 6. Adaptive binning
//! 7. Mutual information methods
//! 8. True Φ computation
//! 9. Parallel entropy computation
//!
//! Run with: cargo bench --bench entropy_methods

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::consciousness_metrics::{
    ContinuousEntropyEstimator, EntropyMethod, IIT4Calculator, ParallelEntropyCalculator,
    QuantumEntropyCalculator, TruePhiCalculator,
};
use symthaea_core::hdc::unified_hv::ContinuousHV;

const HDC_DIMENSION: usize = 16384;

// ============================================================================
// ENTROPY ESTIMATION BENCHMARKS
// ============================================================================

fn bench_entropy_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_estimation");

    // Create test vector
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    // Histogram (baseline)
    let hist_estimator = ContinuousEntropyEstimator::fast();
    group.bench_function("histogram_16bins", |b| {
        b.iter(|| black_box(hist_estimator.entropy(&hv)))
    });

    // k-NN (accurate but slow)
    let knn_estimator = ContinuousEntropyEstimator::knn(3);
    group.bench_function("knn_k3", |b| {
        b.iter(|| black_box(knn_estimator.entropy(&hv)))
    });

    // k-NN fast (sorted optimization)
    let knn_fast_estimator = ContinuousEntropyEstimator::knn_fast(3);
    group.bench_function("knn_fast_k3", |b| {
        b.iter(|| black_box(knn_fast_estimator.entropy(&hv)))
    });

    // KDE (accurate but slow)
    let kde_estimator = ContinuousEntropyEstimator::kde();
    group.bench_function("kde", |b| b.iter(|| black_box(kde_estimator.entropy(&hv))));

    // KDE fast (truncated Gaussian)
    let kde_fast_estimator = ContinuousEntropyEstimator::kde_fast();
    group.bench_function("kde_fast", |b| {
        b.iter(|| black_box(kde_fast_estimator.entropy(&hv)))
    });

    // Adaptive binning
    let adaptive_estimator = ContinuousEntropyEstimator::adaptive(32);
    group.bench_function("adaptive_32bins", |b| {
        b.iter(|| black_box(adaptive_estimator.entropy(&hv)))
    });

    group.finish();
}

// ============================================================================
// MUTUAL INFORMATION BENCHMARKS
// ============================================================================

fn bench_mutual_information(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutual_information");

    let hv1 = ContinuousHV::random(HDC_DIMENSION, 1);
    let hv2 = ContinuousHV::random(HDC_DIMENSION, 2);

    let estimator = ContinuousEntropyEstimator::fast();

    // Fast MI (histogram-based)
    group.bench_function("mi_fast", |b| {
        b.iter(|| black_box(estimator.mutual_information_fast(&hv1, &hv2)))
    });

    // k-NN MI (Kraskov estimator)
    group.bench_function("mi_knn_k3", |b| {
        b.iter(|| black_box(estimator.mutual_information_knn(&hv1, &hv2)))
    });

    // True Phi calculator MI
    let calc = TruePhiCalculator::new();
    group.bench_function("mi_true_phi", |b| {
        b.iter(|| black_box(calc.mutual_information(&hv1, &hv2)))
    });

    group.finish();
}

// ============================================================================
// TRUE Φ COMPUTATION BENCHMARKS
// ============================================================================

fn bench_true_phi(c: &mut Criterion) {
    let mut group = c.benchmark_group("true_phi");

    // Test with different system sizes
    for n in [2, 3, 4, 5, 6] {
        let components: Vec<ContinuousHV> = (0..n)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        let calc = TruePhiCalculator::new();

        group.bench_with_input(
            BenchmarkId::new("compute_phi", n),
            &components,
            |b, comps| b.iter(|| black_box(calc.compute_true_phi(comps))),
        );
    }

    group.finish();
}

// ============================================================================
// IIT 4.0 BENCHMARKS
// ============================================================================

fn bench_iit4(c: &mut Criterion) {
    let mut group = c.benchmark_group("iit4");

    for n in [2, 3, 4, 5] {
        let components: Vec<ContinuousHV> = (0..n)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        let calc = IIT4Calculator::new();

        group.bench_with_input(BenchmarkId::new("analyze", n), &components, |b, comps| {
            b.iter(|| black_box(calc.analyze(comps)))
        });
    }

    group.finish();
}

// ============================================================================
// QUANTUM ENTROPY BENCHMARKS
// ============================================================================

fn bench_quantum_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_entropy");

    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    // Default dimension (64)
    let calc_64 = QuantumEntropyCalculator::new();
    group.bench_function("von_neumann_dim64", |b| {
        b.iter(|| black_box(calc_64.von_neumann_entropy(&hv)))
    });

    group.bench_function("purity_dim64", |b| {
        b.iter(|| black_box(calc_64.purity(&hv)))
    });

    group.bench_function("analyze_dim64", |b| {
        b.iter(|| black_box(calc_64.analyze(&hv)))
    });

    // Smaller dimension (32) for speed
    let calc_32 = QuantumEntropyCalculator::with_dimension(32);
    group.bench_function("analyze_dim32", |b| {
        b.iter(|| black_box(calc_32.analyze(&hv)))
    });

    group.finish();
}

// ============================================================================
// PARALLEL ENTROPY BENCHMARKS
// ============================================================================

fn bench_parallel_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_entropy");

    // Create multiple vectors
    let vectors: Vec<ContinuousHV> = (0..16)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    let parallel_calc = ParallelEntropyCalculator::new();
    let serial_estimator = ContinuousEntropyEstimator::fast();

    // Serial entropy computation
    group.bench_function("serial_16_vectors", |b| {
        b.iter(|| {
            vectors
                .iter()
                .map(|v| serial_estimator.entropy(v))
                .collect::<Vec<_>>()
        })
    });

    // Parallel entropy computation
    group.bench_function("parallel_16_vectors", |b| {
        b.iter(|| black_box(parallel_calc.entropy_batch(&vectors)))
    });

    // Parallel MI matrix
    group.bench_function("parallel_mi_matrix_8", |b| {
        let subset: Vec<ContinuousHV> = vectors[..8].to_vec();
        b.iter(|| black_box(parallel_calc.mutual_information_matrix(&subset)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_entropy_methods,
    bench_mutual_information,
    bench_true_phi,
    bench_iit4,
    bench_quantum_entropy,
    bench_parallel_entropy,
);

criterion_main!(benches);
