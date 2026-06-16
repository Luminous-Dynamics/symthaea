// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stability Analysis Benchmark Suite
//!
//! Measures Jacobian computation, eigenvalue analysis, Lyapunov exponent
//! estimation, fixed-point detection, and contraction rate at varying
//! system dimensions. These are called every few cognitive cycles.
//!
//! Run with: cargo bench --bench dynamics_stability

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea::dynamics::{StabilityAnalyzer, StabilityConfig};

// ---------------------------------------------------------------------------
// Benchmark ODE systems (closures returning Vec<f64>)
// ---------------------------------------------------------------------------

/// N-dimensional coupled linear system: dy_i/dt = -alpha_i * y_i + coupling * y_{i-1}
/// This models a chain of CfC neurons with nearest-neighbor coupling.
#[allow(clippy::type_complexity)]
fn coupled_linear_system(dim: usize) -> Box<dyn Fn(&[f64]) -> Vec<f64>> {
    Box::new(move |x: &[f64]| {
        let mut dx = vec![0.0; dim];
        let coupling = 0.1;
        for i in 0..dim {
            let alpha = 0.5 + 1.5 * (i as f64) / dim.max(1) as f64;
            dx[i] = -alpha * x[i];
            if i > 0 {
                dx[i] += coupling * x[i - 1];
            }
        }
        dx
    })
}

/// N-dimensional CfC-like nonlinear system:
///   dy_i/dt = -y_i / tau_i + sigmoid(sum_j w_ij * y_j)
#[allow(clippy::type_complexity)]
fn cfc_nonlinear_system(dim: usize, seed: u64) -> Box<dyn Fn(&[f64]) -> Vec<f64>> {
    // Pre-compute weights and tau deterministically
    let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
    let mut next_f64 = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
    };

    let scale = 1.0 / (dim as f64).sqrt();
    let weights: Vec<f64> = (0..dim * dim).map(|_| next_f64() * scale).collect();
    let tau: Vec<f64> = (0..dim)
        .map(|i| 0.1 + 2.0 * ((i as f64 + 1.0) / dim as f64))
        .collect();

    Box::new(move |x: &[f64]| {
        let mut dx = vec![0.0; dim];
        for i in 0..dim {
            let decay = -x[i] / tau[i];
            let mut pre_act = 0.0;
            for j in 0..dim {
                pre_act += weights[i * dim + j] * x[j];
            }
            let activation = 1.0 / (1.0 + (-pre_act).exp());
            dx[i] = decay + activation;
        }
        dx
    })
}

/// Deterministic state vector.
fn state_at(dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|i| 0.5 * (1.0 + (i as f64 * 0.3).sin()))
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark: Jacobian + eigenvalue computation
// ---------------------------------------------------------------------------

fn bench_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability_jacobian");

    for &dim in &[8, 16, 32, 64] {
        let analyzer = StabilityAnalyzer::with_defaults(dim);
        let f = coupled_linear_system(dim);
        let x = state_at(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(analyzer.compute_jacobian(f.as_ref(), black_box(&x)));
            })
        });
    }

    group.finish();
}

fn bench_jacobian_nonlinear(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability_jacobian_nonlinear");

    for &dim in &[8, 16, 32, 64] {
        let analyzer = StabilityAnalyzer::with_defaults(dim);
        let f = cfc_nonlinear_system(dim, 42);
        let x = state_at(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(analyzer.compute_jacobian(f.as_ref(), black_box(&x)));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Lyapunov exponents
// ---------------------------------------------------------------------------

fn bench_lyapunov_short(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability_lyapunov_500steps");
    // Use short step count for benchmarking (500 steps instead of default 10K)
    group.sample_size(10); // Lyapunov is relatively slow

    for &dim in &[4, 8, 16] {
        let config = StabilityConfig {
            lyapunov_steps: 500,
            ..Default::default()
        };
        let analyzer = StabilityAnalyzer::new(dim, config);
        let f = coupled_linear_system(dim);
        let x0 = state_at(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(analyzer.lyapunov_exponents(f.as_ref(), black_box(&x0), 0.01, 500));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Fixed point detection (Newton-Raphson)
// ---------------------------------------------------------------------------

fn bench_find_fixed_points(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability_fixed_points");

    for &dim in &[4, 8, 16, 32] {
        let config = StabilityConfig {
            max_iterations: 50,
            ..Default::default()
        };
        let analyzer = StabilityAnalyzer::new(dim, config);

        // Linear system: fixed point at origin
        let f = coupled_linear_system(dim);
        let guesses: Vec<Vec<f64>> = vec![state_at(dim)];

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(analyzer.find_fixed_points(f.as_ref(), black_box(&guesses)));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Contraction rate
// ---------------------------------------------------------------------------

fn bench_contraction_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability_contraction_rate");

    for &dim in &[8, 16, 32, 64] {
        let analyzer = StabilityAnalyzer::with_defaults(dim);

        // Build a known Jacobian matrix (diagonally dominant, stable)
        let jacobian: Vec<Vec<f64>> = (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        if i == j {
                            -(1.0 + i as f64 * 0.1) // diagonal: negative
                        } else {
                            0.05 / dim as f64 // weak coupling
                        }
                    })
                    .collect()
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(analyzer.contraction_rate(black_box(&jacobian)));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Bifurcation detection (parameter sweep)
// ---------------------------------------------------------------------------

fn bench_bifurcation_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability_bifurcation");
    group.sample_size(10); // Each iteration does 50 Jacobian evaluations

    for &dim in &[4, 8, 16] {
        let analyzer = StabilityAnalyzer::with_defaults(dim);
        let x_fixed = vec![0.0; dim];

        #[allow(clippy::type_complexity)]
        let f_param = move |mu: f64| -> Box<dyn Fn(&[f64]) -> Vec<f64>> {
            let d = dim;
            Box::new(move |x: &[f64]| {
                let mut dx = vec![0.0; d];
                for i in 0..d {
                    dx[i] = mu * x[i] - x[i].powi(3);
                }
                dx
            })
        };

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(analyzer.detect_bifurcations(
                    &f_param,
                    black_box(&x_fixed),
                    (-1.0, 1.0),
                    50,
                ));
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_jacobian,
    bench_jacobian_nonlinear,
    bench_lyapunov_short,
    bench_find_fixed_points,
    bench_contraction_rate,
    bench_bifurcation_detection,
);
criterion_main!(benches);
