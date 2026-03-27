// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Criterion benchmarks for PoGQ v4.1 Enhanced aggregation.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mycelix_fl::pogq::config::PoGQv41Config;
use mycelix_fl::pogq::v41_enhanced::PoGQv41Enhanced;
use mycelix_fl::types::Gradient;

fn generate_honest_gradient(dim: usize, node_idx: usize, round: usize) -> Vec<f32> {
    (0..dim)
        .map(|d| {
            let base = (d as f32) * 0.01;
            let noise = ((node_idx * 1000 + round * 100 + d) as f32 * 0.0001).sin() * 0.05;
            base + noise
        })
        .collect()
}

fn generate_byzantine_gradient(dim: usize) -> Vec<f32> {
    (0..dim).map(|d| -(d as f32) * 0.02).collect()
}

fn bench_pogq_single_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("PoGQ_single_round");

    for &n_nodes in &[10, 50, 100] {
        for &dim in &[1_000, 10_000] {
            let config = PoGQv41Config::default();
            let n_byzantine = n_nodes / 5;

            group.bench_with_input(
                BenchmarkId::new(format!("{}n_{}d_{}byz", n_nodes, dim, n_byzantine), ""),
                &(),
                |b, _| {
                    b.iter_batched(
                        || {
                            let pogq = PoGQv41Enhanced::new(config.clone());
                            let mut gradients: Vec<Gradient> = (0..n_nodes - n_byzantine)
                                .map(|i| {
                                    Gradient::new(
                                        format!("honest-{}", i),
                                        generate_honest_gradient(dim, i, 0),
                                        0,
                                    )
                                })
                                .collect();
                            for i in 0..n_byzantine {
                                gradients.push(Gradient::new(
                                    format!("byz-{}", i),
                                    generate_byzantine_gradient(dim),
                                    0,
                                ));
                            }
                            (pogq, gradients)
                        },
                        |(mut pogq, gradients)| pogq.aggregate(&gradients).unwrap(),
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn bench_pogq_multi_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("PoGQ_20_rounds");
    group.sample_size(10); // Fewer samples for multi-round

    for &n_nodes in &[10, 50] {
        let dim = 1_000;
        let config = PoGQv41Config::default();
        let n_byzantine = n_nodes / 5;

        group.bench_with_input(
            BenchmarkId::new(format!("{}n_{}d", n_nodes, dim), ""),
            &(),
            |b, _| {
                b.iter(|| {
                    let mut pogq = PoGQv41Enhanced::new(config.clone());
                    for round in 0..20 {
                        let mut gradients: Vec<Gradient> = (0..n_nodes - n_byzantine)
                            .map(|i| {
                                Gradient::new(
                                    format!("h-{}", i),
                                    generate_honest_gradient(dim, i, round),
                                    round as u64,
                                )
                            })
                            .collect();
                        for i in 0..n_byzantine {
                            gradients.push(Gradient::new(
                                format!("b-{}", i),
                                generate_byzantine_gradient(dim),
                                round as u64,
                            ));
                        }
                        let _ = pogq.aggregate(&gradients);
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_pogq_single_round, bench_pogq_multi_round);
criterion_main!(benches);
