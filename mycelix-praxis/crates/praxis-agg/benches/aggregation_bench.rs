// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Benchmarks for aggregation methods
//
// Run with: cargo bench
//
// This benchmark suite tests the performance of different aggregation methods
// under various conditions to establish baselines and identify optimization opportunities.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use praxis_agg::{
    methods::{clip_l2_norm, median, trimmed_mean, weighted_mean},
    AggregationConfig,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Generate random gradient vectors with Gaussian distribution
fn generate_gradients(participant_count: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    (0..participant_count)
        .map(|_| (0..dims).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

/// Generate weights for weighted mean
fn generate_weights(participant_count: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let weights: Vec<f32> = (0..participant_count)
        .map(|_| rng.gen_range(1.0..10.0))
        .collect();

    // Normalize to sum to 1.0
    let sum: f32 = weights.iter().sum();
    weights.iter().map(|w| w / sum).collect()
}

/// Benchmark trimmed mean aggregation with varying participant counts
fn bench_trimmed_mean_participants(c: &mut Criterion) {
    let mut group = c.benchmark_group("trimmed_mean_by_participants");

    let dims = 512;
    let participant_counts = vec![10, 50, 100, 500, 1000];

    for count in participant_counts {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, &count| {
                let gradients = generate_gradients(count, dims, 42);
                let config = AggregationConfig::default();

                b.iter(|| {
                    let _ = trimmed_mean(black_box(&gradients), black_box(&config));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark trimmed mean aggregation with varying gradient dimensions
fn bench_trimmed_mean_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("trimmed_mean_by_dimensions");

    let participants = 100;
    let dimensions = vec![128, 512, 1024, 2048, 4096];

    for dims in dimensions {
        group.throughput(Throughput::Bytes((participants * dims * 4) as u64)); // 4 bytes per f32

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, &dims| {
            let gradients = generate_gradients(participants, dims, 42);
            let config = AggregationConfig::default();

            b.iter(|| {
                let _ = trimmed_mean(black_box(&gradients), black_box(&config));
            });
        });
    }

    group.finish();
}

/// Benchmark median aggregation
fn bench_median(c: &mut Criterion) {
    let mut group = c.benchmark_group("median");

    let dims = 512;
    let participant_counts = vec![10, 50, 100, 500];

    for count in participant_counts {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, &count| {
                let gradients = generate_gradients(count, dims, 42);
                let config = AggregationConfig::default();

                b.iter(|| {
                    let _ = median(black_box(&gradients), black_box(&config));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark weighted mean aggregation
fn bench_weighted_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_mean");

    let dims = 512;
    let participant_counts = vec![10, 50, 100, 500];

    for count in participant_counts {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, &count| {
                let gradients = generate_gradients(count, dims, 42);
                let weights = generate_weights(count, 43);
                let mut config = AggregationConfig::default();
                config.method = "weighted_mean".to_string();

                b.iter(|| {
                    let _ = weighted_mean(
                        black_box(&gradients),
                        black_box(&weights),
                        black_box(&config),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark L2 norm clipping
fn bench_clipping(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_clipping");

    let dimensions = vec![128, 512, 1024, 2048];
    let clip_norm = 1.0;

    for dims in dimensions {
        group.throughput(Throughput::Bytes((dims * 4) as u64)); // 4 bytes per f32

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, &dims| {
            let mut gradient = generate_gradients(1, dims, 42)[0].clone();

            b.iter(|| {
                clip_l2_norm(black_box(&mut gradient), black_box(clip_norm));
            });
        });
    }

    group.finish();
}

/// Benchmark comparison: all methods with same parameters
fn bench_method_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("method_comparison");

    let participants = 100;
    let dims = 512;
    let gradients = generate_gradients(participants, dims, 42);
    let weights = generate_weights(participants, 43);
    let config = AggregationConfig::default();

    group.bench_function("trimmed_mean", |b| {
        b.iter(|| {
            let _ = trimmed_mean(black_box(&gradients), black_box(&config));
        });
    });

    group.bench_function("median", |b| {
        b.iter(|| {
            let _ = median(black_box(&gradients), black_box(&config));
        });
    });

    group.bench_function("weighted_mean", |b| {
        b.iter(|| {
            let _ = weighted_mean(black_box(&gradients), black_box(&weights), black_box(&config));
        });
    });

    group.finish();
}

/// Benchmark trimmed mean with different trim percentages
fn bench_trim_percentages(c: &mut Criterion) {
    let mut group = c.benchmark_group("trim_percentages");

    let participants = 100;
    let dims = 512;
    let gradients = generate_gradients(participants, dims, 42);
    let trim_percentages = vec![0.0, 0.1, 0.2, 0.3, 0.4];

    for trim_pct in trim_percentages {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.0}%", trim_pct * 100.0)),
            &trim_pct,
            |b, &trim_pct| {
                let mut config = AggregationConfig::default();
                config.trim_percent = trim_pct;

                b.iter(|| {
                    let _ = trimmed_mean(black_box(&gradients), black_box(&config));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_trimmed_mean_participants,
    bench_trimmed_mean_dimensions,
    bench_median,
    bench_weighted_mean,
    bench_clipping,
    bench_method_comparison,
    bench_trim_percentages
);

criterion_main!(benches);
