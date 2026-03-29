// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmarks for Differential Privacy primitives

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use mycelix_differential_privacy::{
    GaussianMechanism, LaplaceMechanism, Mechanism,
    GradientClipper, MomentsAccountant,
};

fn bench_gaussian_noise(c: &mut Criterion) {
    let mechanism = GaussianMechanism::new(1.0, 1.0, 1e-5).unwrap();

    c.bench_function("gaussian_single_value", |b| {
        b.iter(|| mechanism.add_noise(black_box(1.0)))
    });

    let mut group = c.benchmark_group("gaussian_vector");
    for size in [100, 1000, 10000, 100000].iter() {
        let values: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| b.iter(|| mechanism.apply(black_box(&values))),
        );
    }
    group.finish();
}

fn bench_laplace_noise(c: &mut Criterion) {
    let mechanism = LaplaceMechanism::new(1.0, 1.0).unwrap();

    c.bench_function("laplace_single_value", |b| {
        b.iter(|| mechanism.add_noise(black_box(1.0)))
    });

    let mut group = c.benchmark_group("laplace_vector");
    for size in [100, 1000, 10000, 100000].iter() {
        let values: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| b.iter(|| mechanism.apply(black_box(&values))),
        );
    }
    group.finish();
}

fn bench_gradient_clipping(c: &mut Criterion) {
    let mut clipper = GradientClipper::l2(1.0).unwrap();

    let mut group = c.benchmark_group("gradient_clipping");
    for size in [100, 1000, 10000].iter() {
        let gradient: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.001).collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| b.iter(|| clipper.clip(black_box(&gradient))),
        );
    }
    group.finish();
}

fn bench_clip_and_aggregate(c: &mut Criterion) {
    let mut clipper = GradientClipper::l2(1.0).unwrap();

    let mut group = c.benchmark_group("clip_and_aggregate");
    for (n_clients, dim) in [(10, 1000), (100, 1000), (1000, 100)].iter() {
        let gradients: Vec<Vec<f64>> = (0..*n_clients)
            .map(|_| (0..*dim).map(|i| (i as f64) * 0.001).collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("clients_x_dim", format!("{}x{}", n_clients, dim)),
            &gradients,
            |b, grads| b.iter(|| clipper.clip_and_aggregate(black_box(grads))),
        );
    }
    group.finish();
}

fn bench_moments_accountant(c: &mut Criterion) {
    c.bench_function("moments_accountant_step", |b| {
        let mut accountant = MomentsAccountant::new(0.01, 1.0, 1e-5).unwrap();
        b.iter(|| accountant.record_step())
    });

    c.bench_function("moments_accountant_epsilon", |b| {
        let mut accountant = MomentsAccountant::new(0.01, 1.0, 1e-5).unwrap();
        accountant.record_steps(100);
        b.iter(|| accountant.current_epsilon())
    });
}

criterion_group!(
    benches,
    bench_gaussian_noise,
    bench_laplace_noise,
    bench_gradient_clipping,
    bench_clip_and_aggregate,
    bench_moments_accountant,
);

criterion_main!(benches);
