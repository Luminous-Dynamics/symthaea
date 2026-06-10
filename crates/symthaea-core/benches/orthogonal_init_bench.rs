// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Orthogonal Initialization Benchmark
//!
//! Compares random vs orthogonal HV initialization:
//! 1. Generation speed at different dimensions
//! 2. Pairwise orthogonality quality
//! 3. Convergence speed of neurons with different init strategies
//!
//! Run with: cargo bench -p symthaea-core --bench orthogonal_init_bench

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNeuron, UnifiedConfig};

// ============================================================================
// GENERATION SPEED BENCHMARKS
// ============================================================================

fn bench_generation_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("orthogonal_init/generation");

    for &dim in &[512, 1024, 4096, 16384] {
        group.bench_with_input(BenchmarkId::new("random_5", dim), &dim, |b, &dim| {
            b.iter(|| {
                let hvs: Vec<ContinuousHV> =
                    (0..5).map(|i| ContinuousHV::random(dim, 42 + i)).collect();
                black_box(hvs)
            })
        });

        group.bench_with_input(BenchmarkId::new("orthogonal_5", dim), &dim, |b, &dim| {
            b.iter(|| black_box(ContinuousHV::orthogonal_set(dim, 5, 42)))
        });
    }

    group.finish();
}

// ============================================================================
// INIT QUALITY BENCHMARKS
// ============================================================================

fn bench_init_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("orthogonal_init/quality");

    for &dim in &[512, 1024, 4096, 16384] {
        group.bench_with_input(
            BenchmarkId::new("random_similarity", dim),
            &dim,
            |b, &dim| {
                let hvs: Vec<ContinuousHV> =
                    (0..5).map(|i| ContinuousHV::random(dim, 42 + i)).collect();
                b.iter(|| {
                    let mut sum = 0.0f64;
                    let mut count = 0;
                    for i in 0..hvs.len() {
                        for j in (i + 1)..hvs.len() {
                            sum += hvs[i].similarity(&hvs[j]).abs() as f64;
                            count += 1;
                        }
                    }
                    black_box(sum / count as f64)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("orthogonal_similarity", dim),
            &dim,
            |b, &dim| {
                let hvs = ContinuousHV::orthogonal_set(dim, 5, 42);
                b.iter(|| {
                    let mut sum = 0.0f64;
                    let mut count = 0;
                    for i in 0..hvs.len() {
                        for j in (i + 1)..hvs.len() {
                            sum += hvs[i].similarity(&hvs[j]).abs() as f64;
                            count += 1;
                        }
                    }
                    black_box(sum / count as f64)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// CONVERGENCE BENCHMARKS
// ============================================================================

fn bench_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("orthogonal_init/convergence");
    group.sample_size(20); // Fewer samples since each iteration runs 100 evolve steps

    let dim = 4096;
    let target = ContinuousHV::random(dim, 999);

    let config = UnifiedConfig {
        dimension: dim,
        ..UnifiedConfig::default()
    };

    group.bench_function("random_init_100steps", |b| {
        b.iter(|| {
            let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
            for _ in 0..100 {
                neuron.evolve_closed_form(0.01, &target);
            }
            black_box(neuron.state().similarity(&target))
        })
    });

    group.bench_function("orthogonal_init_100steps", |b| {
        b.iter(|| {
            let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 73);
            for _ in 0..100 {
                neuron.evolve_closed_form(0.01, &target);
            }
            black_box(neuron.state().similarity(&target))
        })
    });

    group.finish();
}

// ============================================================================
// CRITERION MAIN
// ============================================================================

criterion_group!(
    benches,
    bench_generation_speed,
    bench_init_quality,
    bench_convergence,
);

criterion_main!(benches);
