// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ODE Solver Benchmark Suite
//!
//! Measures ODE solver latency for Euler, RK4, Dormand-Prince, Implicit Midpoint,
//! Backward Euler, and Exponential integrators across varying dimensions.
//!
//! Run with: cargo bench --bench dynamics_ode

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea::dynamics::{OdeConfig, OdeSolver, OdeSolverEngine, OdeSystem};

// ---------------------------------------------------------------------------
// Benchmark ODE systems
// ---------------------------------------------------------------------------

/// N-dimensional coupled exponential decay: dy_i/dt = -alpha_i * y_i
/// with linearly spaced decay rates alpha_i in [0.5, 2.0].
/// This represents a typical CfC-like linear relaxation system.
struct CoupledDecay {
    dim: usize,
    rates: Vec<f64>,
}

impl CoupledDecay {
    fn new(dim: usize) -> Self {
        let rates: Vec<f64> = (0..dim)
            .map(|i| 0.5 + 1.5 * (i as f64) / (dim.max(1) as f64))
            .collect();
        Self { dim, rates }
    }
}

impl OdeSystem for CoupledDecay {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn evaluate(&self, _t: f64, state: &[f64], derivative: &mut [f64]) {
        for i in 0..self.dim {
            derivative[i] = -self.rates[i] * state[i];
        }
    }
}

/// N-dimensional CfC-like nonlinear system:
///   dy_i/dt = -y_i / tau_i + sigmoid(sum_j w_ij * y_j + b_i)
/// This models the actual CfC dynamics more faithfully.
struct CfCLikeSystem {
    dim: usize,
    tau: Vec<f64>,
    weights: Vec<f64>, // dim x dim, row-major
    bias: Vec<f64>,
}

impl CfCLikeSystem {
    fn new(dim: usize, seed: u64) -> Self {
        let tau: Vec<f64> = (0..dim)
            .map(|i| 0.1 + 2.0 * ((i as f64 + 1.0) / dim as f64))
            .collect();

        // Deterministic pseudo-random weights using xorshift
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let mut next_f64 = || -> f64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        let scale = 1.0 / (dim as f64).sqrt();
        let weights: Vec<f64> = (0..dim * dim).map(|_| next_f64() * scale).collect();
        let bias: Vec<f64> = (0..dim).map(|_| next_f64() * 0.1).collect();

        Self {
            dim,
            tau,
            weights,
            bias,
        }
    }
}

impl OdeSystem for CfCLikeSystem {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn evaluate(&self, _t: f64, state: &[f64], derivative: &mut [f64]) {
        for i in 0..self.dim {
            // Linear decay term
            let decay = -state[i] / self.tau[i];

            // Nonlinear activation: sigmoid(W*y + b)
            let mut pre_activation = self.bias[i];
            for (j, &s_j) in state.iter().enumerate().take(self.dim) {
                pre_activation += self.weights[i * self.dim + j] * s_j;
            }
            let activation = 1.0 / (1.0 + (-pre_activation).exp());

            derivative[i] = decay + activation;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: deterministic initial state
// ---------------------------------------------------------------------------

fn initial_state(dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|i| 0.5 * (1.0 + (i as f64 * 0.1).sin()))
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmarks: Single step comparison (Euler vs RK4 vs DormandPrince)
// ---------------------------------------------------------------------------

fn bench_euler_single_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode_euler_step");

    for &dim in &[64, 128, 256, 512] {
        let system = CfCLikeSystem::new(dim, 42);
        let config = OdeConfig {
            solver: OdeSolver::Euler,
            dt: 0.01,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, dim);
        let y0 = initial_state(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(engine.solve(&system, &y0, (0.0, 0.01)));
            })
        });
    }

    group.finish();
}

fn bench_rk4_single_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode_rk4_step");

    for &dim in &[64, 128, 256, 512] {
        let system = CfCLikeSystem::new(dim, 42);
        let config = OdeConfig {
            solver: OdeSolver::RK4,
            dt: 0.01,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, dim);
        let y0 = initial_state(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(engine.solve(&system, &y0, (0.0, 0.01)));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: Full integration (10 steps)
// ---------------------------------------------------------------------------

fn bench_euler_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode_euler_10steps");

    for &dim in &[64, 128, 256, 512] {
        let system = CfCLikeSystem::new(dim, 42);
        let config = OdeConfig {
            solver: OdeSolver::Euler,
            dt: 0.01,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, dim);
        let y0 = initial_state(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(engine.solve(&system, &y0, (0.0, 0.1)));
            })
        });
    }

    group.finish();
}

fn bench_rk4_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode_rk4_10steps");

    for &dim in &[64, 128, 256, 512] {
        let system = CfCLikeSystem::new(dim, 42);
        let config = OdeConfig {
            solver: OdeSolver::RK4,
            dt: 0.01,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, dim);
        let y0 = initial_state(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(engine.solve(&system, &y0, (0.0, 0.1)));
            })
        });
    }

    group.finish();
}

fn bench_dormand_prince_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode_dormand_prince_10steps");

    for &dim in &[64, 128, 256] {
        let system = CfCLikeSystem::new(dim, 42);
        let config = OdeConfig {
            solver: OdeSolver::DormandPrince,
            dt: 0.01,
            tolerance: 1e-6,
            max_step: 0.1,
            min_step: 1e-10,
            max_iterations: 50,
        };
        let engine = OdeSolverEngine::new(config, dim);
        let y0 = initial_state(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(engine.solve(&system, &y0, (0.0, 0.1)));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: Implicit solvers on stiff system
// ---------------------------------------------------------------------------

fn bench_implicit_midpoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode_implicit_midpoint_10steps");

    for &dim in &[64, 128, 256] {
        let system = CoupledDecay::new(dim);
        let config = OdeConfig {
            solver: OdeSolver::ImplicitMidpoint,
            dt: 0.01,
            tolerance: 1e-8,
            max_iterations: 20,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, dim);
        let y0 = initial_state(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(engine.solve(&system, &y0, (0.0, 0.1)));
            })
        });
    }

    group.finish();
}

fn bench_exponential_integrator(c: &mut Criterion) {
    let mut group = c.benchmark_group("ode_exponential_10steps");

    // Exponential integrator does per-component finite differences,
    // so cost is O(dim^2) per step -- keep dimensions modest.
    for &dim in &[64, 128] {
        let system = CfCLikeSystem::new(dim, 42);
        let config = OdeConfig {
            solver: OdeSolver::Exponential,
            dt: 0.01,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, dim);
        let y0 = initial_state(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(engine.solve(&system, &y0, (0.0, 0.1)));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Standalone rk4_step_fn (the closure-based API used in LTC)
// ---------------------------------------------------------------------------

fn bench_rk4_step_fn_standalone(c: &mut Criterion) {
    use symthaea::dynamics::ode_solvers::rk4_step_fn;

    let mut group = c.benchmark_group("ode_rk4_step_fn");

    for &dim in &[64, 128, 256, 512] {
        let y = initial_state(dim);
        let mut y_out = vec![0.0; dim];

        // Simple decay: dy/dt = -y
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                rk4_step_fn(
                    0.0,
                    black_box(&y),
                    0.01,
                    |_t, state, deriv| {
                        for i in 0..state.len() {
                            deriv[i] = -state[i];
                        }
                    },
                    &mut y_out,
                );
                black_box(&y_out);
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Newton solver (standalone)
// ---------------------------------------------------------------------------

fn bench_newton_solve(c: &mut Criterion) {
    use symthaea::dynamics::newton_solve;

    let mut group = c.benchmark_group("ode_newton_solve");

    for &dim in &[4, 16, 32, 64] {
        // Solve x_i^2 - (i+1) = 0 for each component
        let x0: Vec<f64> = (0..dim).map(|i| 0.5 + i as f64 * 0.1).collect();

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(newton_solve(
                    |x: &[f64], out: &mut [f64]| {
                        for i in 0..x.len() {
                            out[i] = x[i] * x[i] - (i + 1) as f64;
                        }
                    },
                    &x0,
                    1e-10,
                    50,
                ));
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_euler_single_step,
    bench_rk4_single_step,
    bench_euler_integration,
    bench_rk4_integration,
    bench_dormand_prince_integration,
    bench_implicit_midpoint,
    bench_exponential_integrator,
    bench_rk4_step_fn_standalone,
    bench_newton_solve,
);
criterion_main!(benches);
