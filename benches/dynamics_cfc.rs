// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC Network Benchmark Suite
//!
//! Measures CfC forward pass, sequence processing, and training step latency
//! at varying hidden dimensions. These are the core temporal computations
//! executed every cognitive cycle at ~50Hz.
//!
//! Run with: cargo bench --bench dynamics_cfc

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use symthaea::dynamics::{CfCConfig, CfCNetwork, CfCNetworkConfig};

// ---------------------------------------------------------------------------
// Helper: build a CfC network with specified dimensions
// ---------------------------------------------------------------------------

fn make_network(input_dim: usize, hidden_dim: usize, num_layers: usize) -> CfCNetwork {
    let cell_config = CfCConfig {
        input_dim,
        hidden_dim,
        ..Default::default()
    };

    let config = CfCNetworkConfig {
        input_dim,
        hidden_dim,
        num_layers,
        output_dim: hidden_dim / 2,
        cell_config,
        residual: true,
        bidirectional: false,
        enable_online_learning: false,
        ..Default::default()
    };

    CfCNetwork::new(config)
}

/// Deterministic input vector from seed.
fn make_input(dim: usize, seed: u64) -> Array1<f32> {
    let mut state = seed ^ 0x9E3779B97F4A7C15;
    Array1::from_shape_fn(dim, |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32 / u64::MAX as f32) * 2.0 - 1.0
    })
}

// ---------------------------------------------------------------------------
// Benchmark: Single forward pass
// ---------------------------------------------------------------------------

fn bench_forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_forward");

    for &hidden_dim in &[64, 128, 256, 512] {
        let input_dim = hidden_dim / 2;
        let mut net = make_network(input_dim, hidden_dim, 2);
        let input = make_input(input_dim, 42);

        group.bench_with_input(
            BenchmarkId::new("hidden", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    black_box(net.forward(black_box(&input), 0.02));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Sequence of 10 forward passes (simulating 200ms at 50Hz)
// ---------------------------------------------------------------------------

fn bench_forward_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_forward_seq10");

    for &hidden_dim in &[64, 128, 256] {
        let input_dim = hidden_dim / 2;
        let mut net = make_network(input_dim, hidden_dim, 2);
        let inputs: Vec<Array1<f32>> = (0..10).map(|i| make_input(input_dim, i)).collect();
        let dts = vec![0.02f32; 10];

        group.bench_with_input(
            BenchmarkId::new("hidden", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    black_box(net.forward_sequence(black_box(&inputs), black_box(&dts)));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: BPTT training step (single sample)
// ---------------------------------------------------------------------------

fn bench_train_step_bptt(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_train_bptt");

    for &hidden_dim in &[64, 128, 256] {
        let input_dim = hidden_dim / 2;
        let output_dim = hidden_dim / 2;
        let mut net = make_network(input_dim, hidden_dim, 2);
        let input = make_input(input_dim, 42);
        let target = make_input(output_dim, 99);

        group.bench_with_input(
            BenchmarkId::new("hidden", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(net.train_step_bptt(
                        black_box(std::slice::from_ref(&input)),
                        black_box(std::slice::from_ref(&target)),
                        black_box(&[0.02]),
                        0.001,
                    ));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Optimized BPTT training step (cached forward)
// ---------------------------------------------------------------------------

fn bench_train_step_bptt_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_train_bptt_opt");

    for &hidden_dim in &[64, 128, 256] {
        let input_dim = hidden_dim / 2;
        let output_dim = hidden_dim / 2;
        let mut net = make_network(input_dim, hidden_dim, 2);
        let input = make_input(input_dim, 42);
        let target = make_input(output_dim, 99);

        group.bench_with_input(
            BenchmarkId::new("hidden", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(net.train_step_bptt_optimized(
                        black_box(std::slice::from_ref(&input)),
                        black_box(std::slice::from_ref(&target)),
                        black_box(&[0.02]),
                        0.001,
                    ));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Diagnose dynamics (Jacobian perturbation analysis)
// ---------------------------------------------------------------------------

fn bench_diagnose_dynamics(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_diagnose_dynamics");

    for &hidden_dim in &[64, 128] {
        let input_dim = hidden_dim / 2;
        let mut net = make_network(input_dim, hidden_dim, 2);
        let input = make_input(input_dim, 42);

        // Warm up network state
        for i in 0..5 {
            net.forward(&make_input(input_dim, i), 0.02);
        }

        group.bench_with_input(
            BenchmarkId::new("hidden", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    black_box(net.diagnose_dynamics(black_box(&input), 0.02));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Layer scaling (1 vs 2 vs 4 layers)
// ---------------------------------------------------------------------------

fn bench_layer_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_layer_scaling");
    let hidden_dim = 128;
    let input_dim = 64;

    for &num_layers in &[1, 2, 4] {
        let mut net = make_network(input_dim, hidden_dim, num_layers);
        let input = make_input(input_dim, 42);

        group.bench_with_input(
            BenchmarkId::new("layers", num_layers),
            &num_layers,
            |b, _| {
                b.iter(|| {
                    black_box(net.forward(black_box(&input), 0.02));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: State diversity computation
// ---------------------------------------------------------------------------

fn bench_state_diversity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_state_diversity");

    for &hidden_dim in &[64, 128, 256, 512] {
        let input_dim = hidden_dim / 2;
        let mut net = make_network(input_dim, hidden_dim, 2);

        // Warm up
        for i in 0..5 {
            net.forward(&make_input(input_dim, i), 0.02);
        }

        group.bench_with_input(
            BenchmarkId::new("hidden", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    black_box(net.state_diversity());
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_forward_pass,
    bench_forward_sequence,
    bench_train_step_bptt,
    bench_train_step_bptt_optimized,
    bench_diagnose_dynamics,
    bench_layer_scaling,
    bench_state_diversity,
);
criterion_main!(benches);
