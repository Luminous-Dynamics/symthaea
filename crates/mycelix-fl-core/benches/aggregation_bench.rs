// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mycelix_fl_core::*;
use std::collections::HashMap;

fn generate_updates(n: usize, dim: usize) -> Vec<GradientUpdate> {
    // Generate n GradientUpdate with `dim` dimensions, random f32 values using simple deterministic formula
    (0..n)
        .map(|i| {
            let gradients: Vec<f32> = (0..dim)
                .map(|j| ((i * 7 + j * 13) % 1000) as f32 / 1000.0 - 0.5)
                .collect();
            GradientUpdate::new(format!("node-{}", i), 1, gradients, 100, 0.5)
        })
        .collect()
}

fn generate_reputations(n: usize) -> HashMap<String, f32> {
    (0..n).map(|i| (format!("node-{}", i), 0.8)).collect()
}

fn bench_aggregation_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregation");

    for &n in &[10, 50, 100] {
        let dim = 10_000;
        let updates = generate_updates(n, dim);

        group.bench_with_input(BenchmarkId::new("fedavg", n), &updates, |b, updates| {
            b.iter(|| fedavg(black_box(updates)))
        });

        group.bench_with_input(
            BenchmarkId::new("coordinate_median", n),
            &updates,
            |b, updates| b.iter(|| coordinate_median(black_box(updates))),
        );

        group.bench_with_input(
            BenchmarkId::new("trimmed_mean", n),
            &updates,
            |b, updates| b.iter(|| trimmed_mean(black_box(updates), 0.1)),
        );

        if n <= 50 {
            // Krum is O(n^2 * d), expensive for 100
            group.bench_with_input(BenchmarkId::new("krum", n), &updates, |b, updates| {
                b.iter(|| krum(black_box(updates), 1))
            });
        }
    }
    group.finish();
}

fn bench_gradient_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimensions");
    let n = 10;

    for &dim in &[1_000, 10_000, 100_000] {
        let updates = generate_updates(n, dim);

        group.bench_with_input(BenchmarkId::new("fedavg", dim), &updates, |b, updates| {
            b.iter(|| fedavg(black_box(updates)))
        });
    }
    group.finish();
}

fn bench_unified_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline");

    for &n in &[10, 50] {
        let dim = 10_000;
        let updates = generate_updates(n, dim);
        let reputations = generate_reputations(n);

        group.bench_with_input(
            BenchmarkId::new("full_pipeline", n),
            &(updates, reputations),
            |b, (updates, reps)| {
                b.iter(|| {
                    let mut pipeline = UnifiedPipeline::new(PipelineConfig::default());
                    pipeline.aggregate(black_box(updates), black_box(reps))
                })
            },
        );
    }
    group.finish();
}

#[cfg(feature = "shapley")]
fn bench_shapley(c: &mut Criterion) {
    use mycelix_fl_core::shapley::{ShapleyCalculator, ShapleyConfig};

    let mut group = c.benchmark_group("shapley");

    for &n in &[5, 10, 20] {
        let dim = 1_000;
        let updates = generate_updates(n, dim);
        let gradient_map: HashMap<String, Vec<f32>> = updates
            .iter()
            .map(|u| (u.participant_id.clone(), u.gradients.clone()))
            .collect();
        let aggregated: Vec<f32> = {
            let mut avg = vec![0.0f32; dim];
            let count = gradient_map.len() as f32;
            for g in gradient_map.values() {
                for (i, v) in g.iter().enumerate() {
                    avg[i] += v / count;
                }
            }
            avg
        };

        group.bench_with_input(
            BenchmarkId::new("monte_carlo_100", n),
            &(&gradient_map, &aggregated),
            |b, (gmap, agg)| {
                b.iter(|| {
                    let config = ShapleyConfig::monte_carlo(100).with_seed(42);
                    let mut calc = ShapleyCalculator::new(config);
                    calc.calculate_values(black_box(gmap), black_box(agg))
                })
            },
        );

        if n <= 10 {
            group.bench_with_input(
                BenchmarkId::new("monte_carlo_1000", n),
                &(&gradient_map, &aggregated),
                |b, (gmap, agg)| {
                    b.iter(|| {
                        let config = ShapleyConfig::monte_carlo(1000).with_seed(42);
                        let mut calc = ShapleyCalculator::new(config);
                        calc.calculate_values(black_box(gmap), black_box(agg))
                    })
                },
            );
        }
    }
    group.finish();
}

#[cfg(feature = "shapley")]
criterion_group!(
    benches,
    bench_aggregation_algorithms,
    bench_gradient_dimensions,
    bench_unified_pipeline,
    bench_shapley
);

#[cfg(not(feature = "shapley"))]
criterion_group!(
    benches,
    bench_aggregation_algorithms,
    bench_gradient_dimensions,
    bench_unified_pipeline
);

criterion_main!(benches);
