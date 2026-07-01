// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Benchmarks for HdcLtcUnifiedNeuron Learning Dynamics
//!
//! Run with: `cargo bench --bench hdc_ltc_learning`
//!
//! Benchmarks:
//! - Hebbian learning throughput
//! - Contrastive learning throughput
//! - STDP update performance
//! - Pattern completion accuracy vs speed
//! - Sequence learning convergence

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig};
use symthaea_core::hdc::unified_hv::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// TEST CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Standard dimension for benchmarks
const BENCH_DIM: usize = 1024;

/// Full dimension for realistic benchmarks
const FULL_DIM: usize = 16384;

fn bench_config(dim: usize) -> UnifiedConfig {
    UnifiedConfig {
        dimension: dim,
        tau_base: 0.1,
        backbone_tau: 0.5,
        activation: UnifiedActivation::Tanh,
        learning_rate: 0.01,
        momentum: 0.9,
        weight_decay: 0.0001,
        gating_steepness: 1.0,
        interp_bias: 0.0,
        fourier_frequencies: Vec::new(),
        fourier_amplitude: 0.1,
    }
}

fn random_pattern(dim: usize, seed: u64) -> ContinuousHV {
    ContinuousHV::random(dim, seed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEBBIAN LEARNING BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_hebbian_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hebbian_Update");

    for dim in [BENCH_DIM, FULL_DIM] {
        let config = bench_config(dim);
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
        let pattern = random_pattern(dim, 100);

        // Pre-evolve
        for _ in 0..20 {
            neuron.evolve(0.01, &pattern);
        }

        group.bench_with_input(
            BenchmarkId::new("single_update", dim),
            &(neuron.clone(), pattern.clone()),
            |b, (n, p)| {
                let mut neuron = n.clone();
                b.iter(|| {
                    neuron.hebbian_update(black_box(p), Some(0.01));
                });
            },
        );
    }

    group.finish();
}

fn bench_hebbian_training_epoch(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hebbian_Training_Epoch");

    for n_patterns in [5, 10, 20] {
        let config = bench_config(BENCH_DIM);
        let patterns: Vec<ContinuousHV> = (0..n_patterns)
            .map(|i| random_pattern(BENCH_DIM, 100 + i as u64))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("epoch", n_patterns),
            &patterns,
            |b, pats| {
                let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
                b.iter(|| {
                    for pattern in pats {
                        neuron.reset();
                        for _ in 0..10 {
                            neuron.evolve(0.01, pattern);
                        }
                        neuron.hebbian_update(black_box(pattern), Some(0.02));
                    }
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTRASTIVE LEARNING BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_contrastive_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("Contrastive_Update");

    for dim in [BENCH_DIM, FULL_DIM] {
        let config = bench_config(dim);
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);
        let positive = random_pattern(dim, 100);
        let negative = random_pattern(dim, 200);

        // Pre-evolve
        for _ in 0..20 {
            neuron.evolve(0.01, &positive);
        }

        group.bench_with_input(
            BenchmarkId::new("single_update", dim),
            &(neuron.clone(), positive.clone(), negative.clone()),
            |b, (n, pos, neg)| {
                let mut neuron = n.clone();
                b.iter(|| {
                    neuron.contrastive_update(black_box(pos), black_box(neg), 0.01);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// STDP LEARNING BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_stdp_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("STDP_Update");

    for dim in [BENCH_DIM, FULL_DIM] {
        let config = bench_config(dim);
        let neuron = HdcLtcUnifiedNeuron::new(config, 42);
        let pre = random_pattern(dim, 100);
        let post = random_pattern(dim, 200);

        group.bench_with_input(
            BenchmarkId::new("positive_timing", dim),
            &(neuron.clone(), pre.clone(), post.clone()),
            |b, (n, pre, post)| {
                let mut neuron = n.clone();
                b.iter(|| {
                    neuron.stdp_update(black_box(pre), black_box(post), 0.01, 0.01);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("negative_timing", dim),
            &(neuron.clone(), pre.clone(), post.clone()),
            |b, (n, pre, post)| {
                let mut neuron = n.clone();
                b.iter(|| {
                    neuron.stdp_update(black_box(pre), black_box(post), -0.01, 0.01);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRIPLET LEARNING BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_triplet_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("Triplet_Update");

    for dim in [BENCH_DIM, FULL_DIM] {
        let config = bench_config(dim);
        let neuron = HdcLtcUnifiedNeuron::new(config, 42);
        let anchor = random_pattern(dim, 100);
        let positive = random_pattern(dim, 101);
        let negative = random_pattern(dim, 999);

        group.bench_with_input(
            BenchmarkId::new("single_update", dim),
            &(
                neuron.clone(),
                anchor.clone(),
                positive.clone(),
                negative.clone(),
            ),
            |b, (n, anc, pos, neg)| {
                let mut neuron = n.clone();
                b.iter(|| {
                    neuron.triplet_update(
                        black_box(anc),
                        black_box(pos),
                        black_box(neg),
                        0.1,
                        0.01,
                    );
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING CONVERGENCE BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_learning_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("Learning_Convergence");
    group.sample_size(20); // Reduce samples for longer benchmarks

    let config = bench_config(BENCH_DIM);
    let patterns: Vec<ContinuousHV> = (0..10)
        .map(|i| random_pattern(BENCH_DIM, 100 + i))
        .collect();
    let negative = random_pattern(BENCH_DIM, 999);

    // Hebbian convergence
    group.bench_function("hebbian_50_epochs", |b| {
        b.iter(|| {
            let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
            for _epoch in 0..50 {
                for pattern in &patterns {
                    neuron.reset();
                    for _ in 0..10 {
                        neuron.evolve(0.01, pattern);
                    }
                    neuron.hebbian_update(black_box(pattern), Some(0.02));
                }
            }
            neuron.state().norm()
        });
    });

    // Contrastive convergence
    group.bench_function("contrastive_50_epochs", |b| {
        b.iter(|| {
            let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
            for _epoch in 0..50 {
                for pattern in &patterns {
                    neuron.reset();
                    for _ in 0..10 {
                        neuron.evolve(0.01, pattern);
                    }
                    neuron.contrastive_update(black_box(pattern), black_box(&negative), 0.02);
                }
            }
            neuron.state().norm()
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPARISON BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_learning_method_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Learning_Method_Comparison");

    let dim = BENCH_DIM;
    let config = bench_config(dim);
    let pattern = random_pattern(dim, 100);
    let positive = random_pattern(dim, 101);
    let negative = random_pattern(dim, 999);

    // Hebbian
    group.bench_function("hebbian", |b| {
        let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        for _ in 0..20 {
            neuron.evolve(0.01, &pattern);
        }
        b.iter(|| {
            neuron.hebbian_update(black_box(&pattern), Some(0.01));
        });
    });

    // Contrastive
    group.bench_function("contrastive", |b| {
        let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        for _ in 0..20 {
            neuron.evolve(0.01, &pattern);
        }
        b.iter(|| {
            neuron.contrastive_update(black_box(&pattern), black_box(&negative), 0.01);
        });
    });

    // STDP
    group.bench_function("stdp", |b| {
        let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        b.iter(|| {
            neuron.stdp_update(black_box(&pattern), black_box(&positive), 0.01, 0.01);
        });
    });

    // Regularized Hebbian
    group.bench_function("regularized_hebbian", |b| {
        let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        for _ in 0..20 {
            neuron.evolve(0.01, &pattern);
        }
        b.iter(|| {
            neuron.regularized_hebbian_update(black_box(&pattern), 0.01, 1.0);
        });
    });

    // Triplet
    group.bench_function("triplet", |b| {
        let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        b.iter(|| {
            neuron.triplet_update(
                black_box(&pattern),
                black_box(&positive),
                black_box(&negative),
                0.1,
                0.01,
            );
        });
    });

    // Adaptive
    group.bench_function("adaptive", |b| {
        let mut neuron = HdcLtcUnifiedNeuron::new(config.clone(), 42);
        for _ in 0..20 {
            neuron.evolve(0.01, &pattern);
        }
        let gradient = pattern.bind(&neuron.state().clone());
        b.iter(|| {
            neuron.adaptive_update(black_box(&gradient), 0.01, 0.9, 0.999);
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// CRITERION MAIN
// ═══════════════════════════════════════════════════════════════════════════════

criterion_group!(
    benches,
    bench_hebbian_update,
    bench_hebbian_training_epoch,
    bench_contrastive_update,
    bench_stdp_update,
    bench_triplet_update,
    bench_learning_convergence,
    bench_learning_method_comparison,
);

criterion_main!(benches);
