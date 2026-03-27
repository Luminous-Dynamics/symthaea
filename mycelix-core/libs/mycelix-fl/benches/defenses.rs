// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Criterion benchmarks for stateless defense algorithms.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mycelix_fl::defenses::krum::Krum;
use mycelix_fl::defenses::{CoordinateMedian, Defense, FedAvg, Rfa, TrimmedMean};
use mycelix_fl::types::{DefenseConfig, Gradient};

/// Deterministic pseudo-random gradient generator (no external deps).
fn generate_gradients(n_nodes: usize, dim: usize, seed: u64) -> Vec<Gradient> {
    let mut gradients = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        let values: Vec<f32> = (0..dim)
            .map(|d| {
                let x = (seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(i as u64 * 1000 + d as u64))
                    as f32
                    / u64::MAX as f32;
                x * 2.0 - 1.0
            })
            .collect();
        gradients.push(Gradient::new(format!("node-{}", i), values, 0));
    }
    gradients
}

fn bench_defense(c: &mut Criterion, name: &str, defense: &dyn Defense) {
    let config = DefenseConfig::default();
    let mut group = c.benchmark_group(name);

    for &n_nodes in &[10, 50, 100] {
        for &dim in &[1_000, 10_000, 100_000] {
            let gradients = generate_gradients(n_nodes, dim, 42);
            group.bench_with_input(
                BenchmarkId::new(format!("{}n_{}d", n_nodes, dim), ""),
                &gradients,
                |b, grads| {
                    b.iter(|| defense.aggregate(grads, &config).unwrap());
                },
            );
        }
    }
    group.finish();
}

fn bench_fedavg(c: &mut Criterion) {
    bench_defense(c, "FedAvg", &FedAvg);
}

fn bench_trimmed_mean(c: &mut Criterion) {
    bench_defense(c, "TrimmedMean", &TrimmedMean);
}

fn bench_coordinate_median(c: &mut Criterion) {
    bench_defense(c, "CoordinateMedian", &CoordinateMedian);
}

fn bench_rfa(c: &mut Criterion) {
    bench_defense(c, "RFA", &Rfa);
}

fn bench_krum(c: &mut Criterion) {
    let mut group = c.benchmark_group("Krum");

    for &n_nodes in &[10, 50, 100] {
        let dim = 10_000;
        let gradients = generate_gradients(n_nodes, dim, 42);
        let num_byzantine = n_nodes / 5; // assume 20% Byzantine
        group.bench_with_input(
            BenchmarkId::new(format!("{}n_{}d", n_nodes, dim), ""),
            &(),
            |b, _| {
                b.iter(|| Krum::aggregate(&gradients, num_byzantine).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fedavg,
    bench_trimmed_mean,
    bench_coordinate_median,
    bench_rfa,
    bench_krum
);
criterion_main!(benches);
