// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spectral MIP Finder Benchmark Suite
//!
//! Measures SpectralMIPFinder latency at various dimension scales:
//! n=64, 128, 256, 512 — verifying the 50Hz (20ms) budget.
//!
//! Run with: cargo bench --bench spectral_mip

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::consciousness_metrics::{SpectralMIPConfig, SpectralMIPFinder};
use symthaea_core::hdc::unified_hv::ContinuousHV;

const HDC_DIMENSION: usize = 16384;

/// Build a block-diagonal covariance for benchmarking.
fn bench_covariance(n: usize) -> Vec<f64> {
    let half = n / 2;
    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0 + 1e-6; // regularization baked in
        for j in (i + 1)..n {
            let same_block = (i < half && j < half) || (i >= half && j >= half);
            let corr = if same_block { 0.3 } else { 0.02 };
            cov[i * n + j] = corr;
            cov[j * n + i] = corr;
        }
    }
    cov
}

fn bench_compute_from_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_mip_compute_from_cov");

    for &n in &[64, 128, 256, 512] {
        let cov = bench_covariance(n);
        let finder = SpectralMIPFinder::with_defaults();

        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, &n| {
            b.iter(|| black_box(finder.compute_from_covariance(&cov, n, 50)))
        });
    }

    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_mip_full_pipeline");

    for &n in &[64, 128] {
        let config = SpectralMIPConfig {
            num_components: n,
            window_size: 50,
            min_samples: 10,
            regularization: 1e-6,
            normalize_variance: true,
            temporal_decorrelation: true,
        };

        // Pre-fill the window
        let mut finder = SpectralMIPFinder::new(config);
        for i in 0..50 {
            finder.push(&ContinuousHV::random(HDC_DIMENSION, i));
        }

        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| black_box(finder.compute()))
        });
    }

    group.finish();
}

fn bench_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_mip_push");
    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    for &n in &[64, 128, 256] {
        let config = SpectralMIPConfig {
            num_components: n,
            window_size: 50,
            min_samples: 10,
            regularization: 1e-6,
            normalize_variance: true,
            temporal_decorrelation: true,
        };
        let mut finder = SpectralMIPFinder::new(config);

        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| finder.push(black_box(&hv)))
        });
    }

    group.finish();
}

fn bench_hierarchical(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_mip_hierarchical");

    for &n in &[64, 128] {
        let config = SpectralMIPConfig {
            num_components: n,
            window_size: 50,
            min_samples: 10,
            regularization: 1e-6,
            normalize_variance: true,
            temporal_decorrelation: true,
        };

        let mut finder = SpectralMIPFinder::new(config);
        for i in 0..50 {
            finder.push(&ContinuousHV::random(HDC_DIMENSION, i));
        }

        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            b.iter(|| black_box(finder.compute_hierarchical()))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compute_from_covariance,
    bench_full_pipeline,
    bench_push,
    bench_hierarchical,
);
criterion_main!(benches);
