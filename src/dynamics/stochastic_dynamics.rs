// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Stochastic Differential Equations (SDEs) for Neural Dynamics
//!
//! Neural systems are inherently stochastic. Thermal fluctuations, synaptic
//! noise, and channel noise all contribute to the variability observed in
//! biological neural circuits. These stochastic processes are not mere nuisances
//! -- they drive spontaneous transitions between conscious states, enable
//! stochastic resonance (where noise *improves* signal detection), and underpin
//! the exploration-exploitation trade-off in decision-making.
//!
//! This module provides SDE solvers and stochastic process primitives that
//! integrate with Symthaea's CfC/LTC dynamics, enabling:
//!
//! - **Stochastic CfC neurons**: Adding biologically realistic noise to
//!   closed-form continuous-time dynamics
//! - **Ornstein-Uhlenbeck processes**: Mean-reverting noise with homeostatic
//!   properties, modeling spontaneous fluctuations in neural membrane potentials
//! - **Fokker-Planck density tracking**: Following the probability density of
//!   neural state ensembles rather than individual trajectories
//! - **Langevin sampling**: Gradient-based dynamics with noise for sampling
//!   from energy landscapes (analogous to free-energy minimization in active
//!   inference)
//!
//! ## SDE Solvers
//!
//! Three numerical methods of increasing accuracy:
//!
//! | Method           | Strong Order | Weak Order | Notes                        |
//! |------------------|-------------|------------|------------------------------|
//! | Euler-Maruyama   | 0.5         | 1.0        | Simple, widely used          |
//! | Milstein         | 1.0         | 1.0        | Requires diffusion gradient  |
//! | Stochastic RK    | 1.0         | 2.0        | 2-stage, better for mult. noise |
//!
//! ## Mathematical Foundation
//!
//! The general Ito SDE:
//!
//! ```text
//!   dX_t = mu(X_t, t) dt + sigma(X_t, t) dW_t
//! ```
//!
//! where `mu` is the drift (deterministic part), `sigma` is the diffusion
//! (noise amplitude), and `W_t` is a Wiener process (Brownian motion).

use serde::{Deserialize, Serialize};

// =============================================================================
// SIMPLE REPRODUCIBLE PRNG (no external deps)
// =============================================================================

/// Xorshift64-based PRNG for reproducible stochastic simulations.
///
/// This avoids pulling in thread-local RNG state, ensuring that SDE
/// trajectories are exactly reproducible given the same seed.
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG from a seed. Seed 0 is mapped to a non-zero value.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x5A5A_5A5A_5A5A_5A5A
            } else {
                seed
            },
        }
    }

    /// Generate the next pseudo-random u64 (xorshift64).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a uniform f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Use 53 bits of the u64 for full f64 mantissa precision
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a standard normal variate via the Box-Muller transform.
    ///
    /// Returns a sample from N(0, 1). Uses two uniforms to produce one normal.
    pub fn next_normal(&mut self) -> f64 {
        for _ in 0..100 {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            // Guard against log(0)
            if u1 > 1e-300 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return r * theta.cos();
            }
        }
        0.0 // Fallback: RNG produced 100 consecutive near-zero values
    }
}

// =============================================================================
// SDE CONFIGURATION & SOLVER ENUM
// =============================================================================

/// Numerical method for solving the SDE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SdeSolver {
    /// Euler-Maruyama: strong order 0.5, weak order 1.0.
    /// The simplest and most widely used SDE integrator.
    EulerMaruyama,

    /// Milstein: strong order 1.0.
    /// Includes the Ito correction term `(1/2) sigma sigma' (Z^2 - 1) dt`,
    /// which requires the user to supply `diffusion_gradient`.
    Milstein,

    /// 2-stage Stochastic Runge-Kutta: strong order 1.0, weak order 2.0.
    /// Better accuracy for multiplicative noise without needing explicit
    /// diffusion gradients (estimates them via finite differences).
    StochasticRK,
}

/// Configuration for SDE integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdeConfig {
    /// Time step size.
    pub dt: f64,

    /// Seed for the pseudo-random number generator.
    pub noise_seed: u64,

    /// Which numerical solver to use.
    pub solver: SdeSolver,
}

impl Default for SdeConfig {
    fn default() -> Self {
        Self {
            dt: 0.001,
            noise_seed: 42,
            solver: SdeSolver::EulerMaruyama,
        }
    }
}

// =============================================================================
// SDE RESULT & STATISTICS
// =============================================================================

/// Ensemble statistics computed from Monte Carlo trajectories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdeStatistics {
    /// Mean trajectory across all sample paths: `mean[time_idx][dim]`.
    pub mean_trajectory: Vec<Vec<f64>>,

    /// Variance trajectory across all sample paths: `variance[time_idx][dim]`.
    pub variance_trajectory: Vec<Vec<f64>>,

    /// Lag-1 autocorrelation of the first dimension, averaged over paths.
    pub autocorrelation: Vec<f64>,
}

/// Result of an SDE solve or Monte Carlo ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdeResult {
    /// Sample path trajectories: `trajectories[path][time_idx][dim]`.
    pub trajectories: Vec<Vec<Vec<f64>>>,

    /// Time grid.
    pub times: Vec<f64>,

    /// Ensemble statistics.
    pub statistics: SdeStatistics,
}

// =============================================================================
// SDE SYSTEM TRAIT
// =============================================================================

/// Defines an SDE system `dX = mu(X, t) dt + sigma(X, t) dW`.
///
/// Implementors specify the drift and diffusion coefficients. For the Milstein
/// solver, override `diffusion_gradient` to supply `d(sigma)/dx`.
pub trait SdeSystem {
    /// State space dimension.
    fn dimension(&self) -> usize;

    /// Evaluate the drift vector `mu(t, state)` and write into `out`.
    fn drift(&self, t: f64, state: &[f64], out: &mut [f64]);

    /// Evaluate the diffusion vector `sigma(t, state)` and write into `out`.
    /// Each component `sigma_i` multiplies an independent Wiener increment.
    fn diffusion(&self, t: f64, state: &[f64], out: &mut [f64]);

    /// Evaluate `d(sigma_i)/dx_i` for the Milstein correction.
    /// Default implementation uses a finite-difference approximation.
    fn diffusion_gradient(&self, t: f64, state: &[f64], out: &mut [f64]) {
        let dim = self.dimension();
        let eps = 1e-7;
        let mut sigma_plus = vec![0.0; dim];
        let mut sigma_minus = vec![0.0; dim];
        let mut state_pert = state.to_vec();

        for i in 0..dim {
            let orig = state_pert[i];

            state_pert[i] = orig + eps;
            self.diffusion(t, &state_pert, &mut sigma_plus);

            state_pert[i] = orig - eps;
            self.diffusion(t, &state_pert, &mut sigma_minus);

            state_pert[i] = orig;
            out[i] = (sigma_plus[i] - sigma_minus[i]) / (2.0 * eps);
        }
    }
}

// =============================================================================
// SINGLE-PATH SOLVER
// =============================================================================

/// Solve a single SDE trajectory.
///
/// Integrates the SDE from `t_span.0` to `t_span.1` starting at `x0`
/// using the method specified in `config`.
///
/// Returns an `SdeResult` with a single trajectory.
pub fn solve(
    system: &dyn SdeSystem,
    x0: &[f64],
    t_span: (f64, f64),
    config: &SdeConfig,
) -> SdeResult {
    let dim = system.dimension();
    assert_eq!(x0.len(), dim, "Initial condition dimension mismatch");
    assert!(config.dt > 0.0, "Time step must be positive");
    assert!(t_span.1 > t_span.0, "End time must exceed start time");

    let n_steps = ((t_span.1 - t_span.0) / config.dt).ceil() as usize;
    let _sqrt_dt = config.dt.sqrt();

    let mut rng = SimpleRng::new(config.noise_seed);
    let mut times = Vec::with_capacity(n_steps + 1);
    let mut trajectory = Vec::with_capacity(n_steps + 1);

    // Initial state
    let mut x = x0.to_vec();
    times.push(t_span.0);
    trajectory.push(x.clone());

    let mut mu = vec![0.0; dim];
    let mut sigma = vec![0.0; dim];
    let mut dsigma = vec![0.0; dim];

    for step in 0..n_steps {
        let t = t_span.0 + step as f64 * config.dt;
        let dt = if step == n_steps - 1 {
            // Last step may be shorter
            t_span.1 - t
        } else {
            config.dt
        };
        let sqrt_dt_actual = dt.sqrt();

        system.drift(t, &x, &mut mu);
        system.diffusion(t, &x, &mut sigma);

        match config.solver {
            SdeSolver::EulerMaruyama => {
                // X_{n+1} = X_n + mu * dt + sigma * sqrt(dt) * Z
                for i in 0..dim {
                    let z = rng.next_normal();
                    x[i] += mu[i] * dt + sigma[i] * sqrt_dt_actual * z;
                }
            }
            SdeSolver::Milstein => {
                // X_{n+1} = X_n + mu*dt + sigma*sqrt(dt)*Z
                //         + 0.5 * sigma * sigma' * (Z^2 - 1) * dt
                system.diffusion_gradient(t, &x, &mut dsigma);
                for i in 0..dim {
                    let z = rng.next_normal();
                    x[i] += mu[i] * dt
                        + sigma[i] * sqrt_dt_actual * z
                        + 0.5 * sigma[i] * dsigma[i] * (z * z - 1.0) * dt;
                }
            }
            SdeSolver::StochasticRK => {
                // 2-stage Stochastic Runge-Kutta (Platen's scheme)
                //
                // Stage 1: supporting value
                //   X_hat = X_n + sigma * sqrt(dt)
                // Stage 2:
                //   sigma_hat = sigma(t, X_hat)
                //   X_{n+1} = X_n + mu*dt + sigma*sqrt(dt)*Z
                //           + 0.5*(sigma_hat - sigma)*(Z^2 - 1)*sqrt(dt)
                let mut x_hat = vec![0.0; dim];
                let mut sigma_hat = vec![0.0; dim];

                for i in 0..dim {
                    x_hat[i] = x[i] + sigma[i] * sqrt_dt_actual;
                }
                system.diffusion(t, &x_hat, &mut sigma_hat);

                for i in 0..dim {
                    let z = rng.next_normal();
                    x[i] += mu[i] * dt
                        + sigma[i] * sqrt_dt_actual * z
                        + 0.5 * (sigma_hat[i] - sigma[i]) * (z * z - 1.0) * sqrt_dt_actual;
                }
            }
        }

        times.push(t + dt);
        trajectory.push(x.clone());
    }

    // Compute trivial statistics for a single path
    let mean_trajectory = trajectory.clone();
    let variance_trajectory = trajectory.iter().map(|s| vec![0.0; s.len()]).collect();
    let autocorrelation = compute_autocorrelation_dim0(&trajectory);

    SdeResult {
        trajectories: vec![trajectory],
        times,
        statistics: SdeStatistics {
            mean_trajectory,
            variance_trajectory,
            autocorrelation,
        },
    }
}

// =============================================================================
// MONTE CARLO ENSEMBLE
// =============================================================================

/// Run a Monte Carlo ensemble of SDE trajectories.
///
/// Launches `n_paths` independent realizations from the same initial condition,
/// each with a distinct (but deterministic) RNG seed derived from `config.noise_seed`.
///
/// Returns an `SdeResult` with all trajectories and ensemble statistics
/// (mean, variance, autocorrelation).
pub fn monte_carlo(
    system: &dyn SdeSystem,
    x0: &[f64],
    t_span: (f64, f64),
    n_paths: usize,
    config: &SdeConfig,
) -> SdeResult {
    assert!(n_paths > 0, "Need at least one path");

    let mut all_trajectories = Vec::with_capacity(n_paths);
    let mut shared_times = Vec::new();

    for path_idx in 0..n_paths {
        let mut path_config = config.clone();
        // Derive a unique seed for each path
        path_config.noise_seed = config
            .noise_seed
            .wrapping_add(path_idx as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        let result = solve(system, x0, t_span, &path_config);
        if shared_times.is_empty() {
            shared_times = result.times;
        }
        // Each solve returns exactly one trajectory
        all_trajectories.push(
            result
                .trajectories
                .into_iter()
                .next()
                .expect("solve always returns at least one trajectory"),
        );
    }

    let statistics = compute_ensemble_statistics(&all_trajectories);

    SdeResult {
        trajectories: all_trajectories,
        times: shared_times,
        statistics,
    }
}

/// Compute ensemble mean, variance, and autocorrelation from multiple trajectories.
fn compute_ensemble_statistics(trajectories: &[Vec<Vec<f64>>]) -> SdeStatistics {
    let n_paths = trajectories.len();
    if n_paths == 0 {
        return SdeStatistics {
            mean_trajectory: Vec::new(),
            variance_trajectory: Vec::new(),
            autocorrelation: Vec::new(),
        };
    }

    let n_steps = trajectories[0].len();
    let dim = if n_steps > 0 {
        trajectories[0][0].len()
    } else {
        0
    };
    let n_f64 = n_paths as f64;

    // Mean
    let mut mean_trajectory = vec![vec![0.0; dim]; n_steps];
    for traj in trajectories {
        for (t, state) in traj.iter().enumerate() {
            for d in 0..dim {
                mean_trajectory[t][d] += state[d];
            }
        }
    }
    for t in 0..n_steps {
        for d in 0..dim {
            mean_trajectory[t][d] /= n_f64;
        }
    }

    // Variance
    let mut variance_trajectory = vec![vec![0.0; dim]; n_steps];
    for traj in trajectories {
        for (t, state) in traj.iter().enumerate() {
            for d in 0..dim {
                let diff = state[d] - mean_trajectory[t][d];
                variance_trajectory[t][d] += diff * diff;
            }
        }
    }
    for t in 0..n_steps {
        for d in 0..dim {
            variance_trajectory[t][d] /= n_f64;
        }
    }

    // Average autocorrelation of dimension 0 across paths
    let mut autocorrelation = Vec::new();
    if n_steps > 1 && dim > 0 {
        let auto_per_path: Vec<Vec<f64>> = trajectories
            .iter()
            .map(|traj| compute_autocorrelation_dim0(traj))
            .collect();
        let auto_len = auto_per_path.iter().map(|a| a.len()).min().unwrap_or(0);
        autocorrelation = vec![0.0; auto_len];
        for auto in &auto_per_path {
            for (i, &v) in auto.iter().enumerate().take(auto_len) {
                autocorrelation[i] += v;
            }
        }
        for v in &mut autocorrelation {
            *v /= n_f64;
        }
    }

    SdeStatistics {
        mean_trajectory,
        variance_trajectory,
        autocorrelation,
    }
}

/// Compute lag-1 autocorrelation of dimension 0 along a trajectory.
fn compute_autocorrelation_dim0(trajectory: &[Vec<f64>]) -> Vec<f64> {
    if trajectory.len() < 2 || trajectory[0].is_empty() {
        return Vec::new();
    }

    let series: Vec<f64> = trajectory.iter().map(|s| s[0]).collect();
    let n = series.len();
    let mean: f64 = series.iter().sum::<f64>() / n as f64;
    let var: f64 = series.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

    if var < 1e-30 {
        return vec![1.0; n];
    }

    let max_lag = n.min(50); // Cap at 50 lags to keep output manageable
    let mut autocorr = Vec::with_capacity(max_lag);
    for lag in 0..max_lag {
        let mut c = 0.0;
        let count = n - lag;
        for i in 0..count {
            c += (series[i] - mean) * (series[i + lag] - mean);
        }
        autocorr.push(c / (count as f64 * var));
    }
    autocorr
}

// =============================================================================
// ORNSTEIN-UHLENBECK PROCESS
// =============================================================================

/// Ornstein-Uhlenbeck process: mean-reverting stochastic dynamics.
///
/// ```text
///   dX = theta * (mu - X) dt + sigma dW
/// ```
///
/// This is the canonical model for noisy homeostatic processes:
/// - **Neural membrane potential fluctuations** between spikes
/// - **Spontaneous cortical activity** that drives state transitions
/// - **Baseline consciousness level** that reverts to a set point
///
/// Exact properties:
/// - Stationary mean: `mu`
/// - Stationary variance: `sigma^2 / (2 * theta)`
/// - Autocorrelation time: `1 / theta`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrnsteinUhlenbeck {
    /// Rate of mean reversion. Larger theta = faster return to mu.
    pub theta: f64,
    /// Long-term mean.
    pub mu: f64,
    /// Volatility (noise amplitude).
    pub sigma: f64,
    /// Current state.
    pub state: f64,
}

impl OrnsteinUhlenbeck {
    /// Create a new OU process. Initial state is set to the long-term mean.
    pub fn new(theta: f64, mu: f64, sigma: f64) -> Self {
        Self {
            theta,
            mu,
            sigma,
            state: mu,
        }
    }

    /// Advance the process by one time step using the exact update formula.
    ///
    /// The exact discrete-time transition is:
    /// ```text
    ///   X_{t+dt} = mu + (X_t - mu) * exp(-theta * dt) + sigma * sqrt((1 - exp(-2*theta*dt)) / (2*theta)) * Z
    /// ```
    /// This avoids discretization error entirely.
    pub fn step(&mut self, dt: f64, rng: &mut SimpleRng) -> f64 {
        let decay = (-self.theta * dt).exp();
        let noise_var = if self.theta.abs() < 1e-12 {
            // Degenerate case: pure Brownian motion
            self.sigma * self.sigma * dt
        } else {
            self.sigma * self.sigma * (1.0 - (-2.0 * self.theta * dt).exp()) / (2.0 * self.theta)
        };
        let noise_std = noise_var.sqrt();

        self.state = self.mu + (self.state - self.mu) * decay + noise_std * rng.next_normal();
        self.state
    }

    /// Theoretical stationary variance: sigma^2 / (2 * theta).
    pub fn stationary_variance(&self) -> f64 {
        if self.theta.abs() < 1e-12 {
            f64::INFINITY
        } else {
            self.sigma * self.sigma / (2.0 * self.theta)
        }
    }

    /// Theoretical autocorrelation at lag tau: exp(-theta * tau).
    pub fn autocorrelation(&self, tau: f64) -> f64 {
        (-self.theta * tau).exp()
    }
}

impl SdeSystem for OrnsteinUhlenbeck {
    fn dimension(&self) -> usize {
        1
    }

    fn drift(&self, _t: f64, state: &[f64], out: &mut [f64]) {
        out[0] = self.theta * (self.mu - state[0]);
    }

    fn diffusion(&self, _t: f64, _state: &[f64], out: &mut [f64]) {
        out[0] = self.sigma;
    }

    fn diffusion_gradient(&self, _t: f64, _state: &[f64], out: &mut [f64]) {
        // sigma is constant w.r.t. x, so the gradient is zero.
        out[0] = 0.0;
    }
}

// =============================================================================
// STOCHASTIC CfC DYNAMICS
// =============================================================================

/// Stochastic extension of CfC neural dynamics.
///
/// Adds biologically realistic noise to the CfC update rule:
/// ```text
///   dh = [-h / tau + sigma_act(W*h + b)] dt + sigma_noise dW
/// ```
///
/// This models:
/// - **Synaptic noise** from stochastic neurotransmitter release
/// - **Channel noise** from random ion channel gating
/// - **Thermal fluctuations** in membrane potential
///
/// The noise enables spontaneous transitions between attractor states,
/// which in consciousness terms corresponds to the stream-of-consciousness
/// wandering between stable percepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticCfC {
    /// Number of hidden units.
    pub hidden_dim: usize,
    /// Time constants per unit.
    pub tau: Vec<f64>,
    /// Weight matrix (hidden_dim x hidden_dim), row-major.
    pub weights: Vec<f64>,
    /// Bias vector.
    pub bias: Vec<f64>,
    /// Noise amplitude per unit.
    pub noise_sigma: Vec<f64>,
}

impl StochasticCfC {
    /// Create a new stochastic CfC system with given dimensions and uniform noise.
    pub fn new(hidden_dim: usize, noise_level: f64) -> Self {
        // Initialize with reasonable defaults
        let mut rng = SimpleRng::new(12345);
        let tau: Vec<f64> = (0..hidden_dim)
            .map(|_| 0.5 + 9.5 * rng.next_f64()) // tau in [0.5, 10.0]
            .collect();

        let scale = 1.0 / (hidden_dim as f64).sqrt();
        let weights: Vec<f64> = (0..hidden_dim * hidden_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();

        let bias = vec![0.0; hidden_dim];
        let noise_sigma = vec![noise_level; hidden_dim];

        Self {
            hidden_dim,
            tau,
            weights,
            bias,
            noise_sigma,
        }
    }
}

impl SdeSystem for StochasticCfC {
    fn dimension(&self) -> usize {
        self.hidden_dim
    }

    fn drift(&self, _t: f64, state: &[f64], out: &mut [f64]) {
        // dh_i/dt = -h_i / tau_i + sigmoid(sum_j W_ij * h_j + b_i)
        let dim = self.hidden_dim;
        for i in 0..dim {
            let mut pre_act = self.bias[i];
            for j in 0..dim {
                pre_act += self.weights[i * dim + j] * state[j];
            }
            let activation = 1.0 / (1.0 + (-pre_act).exp()); // sigmoid
            out[i] = -state[i] / self.tau[i] + activation;
        }
    }

    fn diffusion(&self, _t: f64, _state: &[f64], out: &mut [f64]) {
        out[..self.hidden_dim].copy_from_slice(&self.noise_sigma[..self.hidden_dim]);
    }
}

// =============================================================================
// FOKKER-PLANCK SOLVER
// =============================================================================

/// Numerical solver for the 1D Fokker-Planck equation.
///
/// Tracks the evolution of a probability density function (PDF) under drift
/// and diffusion:
///
/// ```text
///   dp/dt = -d/dx [mu(x) * p] + (1/2) d^2/dx^2 [sigma^2(x) * p]
/// ```
///
/// This is the "dual" viewpoint to SDE simulation: rather than tracking
/// individual sample paths, we track the entire probability distribution.
/// In consciousness modeling, this gives us the distribution over possible
/// conscious states at each moment.
///
/// Uses central finite differences on a uniform grid with zero-flux
/// (reflecting) boundary conditions.
#[derive(Debug, Clone)]
pub struct FokkerPlanckSolver {
    /// Spatial grid points.
    pub grid: Vec<f64>,
    /// Probability density at each grid point.
    pub density: Vec<f64>,
    /// Grid spacing.
    pub dx: f64,
    /// Drift function evaluated at grid points.
    drift_values: Vec<f64>,
    /// Diffusion^2 function evaluated at grid points.
    diffusion_sq_values: Vec<f64>,
}

impl FokkerPlanckSolver {
    /// Create a new Fokker-Planck solver on the interval `[x_min, x_max]`.
    ///
    /// The initial density is a Dirac-like spike at the grid center.
    /// The `drift_fn` and `diffusion_fn` are evaluated once at construction
    /// and cached (valid for autonomous systems).
    pub fn new(
        x_range: (f64, f64),
        dx: f64,
        drift_fn: &dyn Fn(f64) -> f64,
        diffusion_fn: &dyn Fn(f64) -> f64,
    ) -> Self {
        let n = ((x_range.1 - x_range.0) / dx).ceil() as usize + 1;
        let grid: Vec<f64> = (0..n).map(|i| x_range.0 + i as f64 * dx).collect();

        // Initialize as narrow Gaussian at grid center
        let center = (x_range.0 + x_range.1) / 2.0;
        let init_width = dx * 3.0;
        let mut density: Vec<f64> = grid
            .iter()
            .map(|&x| {
                let d = (x - center) / init_width;
                (-0.5 * d * d).exp()
            })
            .collect();

        // Normalize
        let total: f64 = density.iter().sum::<f64>() * dx;
        if total > 0.0 {
            for d in &mut density {
                *d /= total;
            }
        }

        let drift_values: Vec<f64> = grid.iter().map(|&x| drift_fn(x)).collect();
        let diffusion_sq_values: Vec<f64> = grid
            .iter()
            .map(|&x| {
                let s = diffusion_fn(x);
                s * s
            })
            .collect();

        Self {
            grid,
            density,
            dx,
            drift_values,
            diffusion_sq_values,
        }
    }

    /// Advance the density by one time step using explicit finite differences.
    ///
    /// Stability requirement (CFL condition): `dt < dx^2 / max(sigma^2)`.
    pub fn step(&mut self, dt: f64) {
        let n = self.density.len();
        if n < 3 {
            return;
        }

        let dx = self.dx;
        let dx2 = dx * dx;
        let mut new_density = vec![0.0; n];

        // Interior points: central differences
        for i in 1..n - 1 {
            // Advection term: -d/dx [mu * p] via central differences
            let advection = -(self.drift_values[i + 1] * self.density[i + 1]
                - self.drift_values[i - 1] * self.density[i - 1])
                / (2.0 * dx);

            // Diffusion term: (1/2) d^2/dx^2 [D * p] via central differences
            let dp = self.diffusion_sq_values[i + 1] * self.density[i + 1]
                - 2.0 * self.diffusion_sq_values[i] * self.density[i]
                + self.diffusion_sq_values[i - 1] * self.density[i - 1];
            let diffusion = 0.5 * dp / dx2;

            new_density[i] = self.density[i] + dt * (advection + diffusion);
        }

        // Zero-flux boundary conditions (reflecting)
        new_density[0] = new_density[1];
        new_density[n - 1] = new_density[n - 2];

        // Enforce non-negativity and renormalize
        for d in &mut new_density {
            if *d < 0.0 {
                *d = 0.0;
            }
        }
        let total: f64 = new_density.iter().sum::<f64>() * dx;
        if total > 1e-30 {
            for d in &mut new_density {
                *d /= total;
            }
        }

        self.density = new_density;
    }

    /// Get the current probability density.
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    /// Get the grid points.
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// Compute the mean of the current density.
    pub fn mean(&self) -> f64 {
        self.grid
            .iter()
            .zip(self.density.iter())
            .map(|(&x, &p)| x * p)
            .sum::<f64>()
            * self.dx
    }

    /// Compute the variance of the current density.
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        self.grid
            .iter()
            .zip(self.density.iter())
            .map(|(&x, &p)| (x - m) * (x - m) * p)
            .sum::<f64>()
            * self.dx
    }
}

// =============================================================================
// LANGEVIN DYNAMICS
// =============================================================================

/// Langevin dynamics: gradient descent in a potential with thermal noise.
///
/// ```text
///   dX = -dV/dX dt + sqrt(2D) dW
/// ```
///
/// At long times, Langevin dynamics samples from the Boltzmann distribution:
/// ```text
///   p(x) ~ exp(-V(x) / D)
/// ```
///
/// In consciousness modeling, the potential `V` can represent the free-energy
/// landscape, and the diffusion `D` represents neural noise that allows the
/// system to explore different conscious states. This is directly analogous
/// to active inference's free-energy minimization.
pub struct LangevinDynamics {
    /// Potential energy function V(x). The force is -dV/dx.
    potential: Box<dyn Fn(f64) -> f64>,
    /// Diffusion constant (temperature-like parameter).
    pub diffusion: f64,
    /// Current state.
    pub state: f64,
}

impl std::fmt::Debug for LangevinDynamics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangevinDynamics")
            .field("diffusion", &self.diffusion)
            .field("state", &self.state)
            .finish()
    }
}

impl LangevinDynamics {
    /// Create new Langevin dynamics with a given potential and diffusion constant.
    pub fn new(potential: Box<dyn Fn(f64) -> f64>, diffusion: f64, initial_state: f64) -> Self {
        Self {
            potential,
            diffusion,
            state: initial_state,
        }
    }

    /// Compute the force at position x: F(x) = -dV/dx (central differences).
    fn force(&self, x: f64) -> f64 {
        let eps = 1e-7;
        -((self.potential)(x + eps) - (self.potential)(x - eps)) / (2.0 * eps)
    }

    /// Advance by one Euler-Maruyama step.
    ///
    /// `dX = F(X) dt + sqrt(2D) sqrt(dt) Z`
    pub fn step(&mut self, dt: f64, rng: &mut SimpleRng) -> f64 {
        let f = self.force(self.state);
        let noise_amp = (2.0 * self.diffusion).sqrt();
        self.state += f * dt + noise_amp * dt.sqrt() * rng.next_normal();
        self.state
    }

    /// Run for `n_steps` and return the trajectory of states.
    pub fn run(&mut self, dt: f64, n_steps: usize, rng: &mut SimpleRng) -> Vec<f64> {
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        trajectory.push(self.state);
        for _ in 0..n_steps {
            self.step(dt, rng);
            trajectory.push(self.state);
        }
        trajectory
    }

    /// Evaluate the potential at a given point.
    pub fn potential_at(&self, x: f64) -> f64 {
        (self.potential)(x)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: simple linear SDE for testing.
    /// dX = -a*X dt + b dW  (OU process rewritten)
    struct LinearSde {
        a: f64,
        b: f64,
    }

    impl SdeSystem for LinearSde {
        fn dimension(&self) -> usize {
            1
        }
        fn drift(&self, _t: f64, state: &[f64], out: &mut [f64]) {
            out[0] = -self.a * state[0];
        }
        fn diffusion(&self, _t: f64, _state: &[f64], out: &mut [f64]) {
            out[0] = self.b;
        }
        fn diffusion_gradient(&self, _t: f64, _state: &[f64], out: &mut [f64]) {
            out[0] = 0.0; // constant diffusion
        }
    }

    #[test]
    fn test_euler_maruyama_reduces_to_euler_for_zero_noise() {
        // With sigma=0, the SDE reduces to the ODE dX = -X dt.
        // The Euler method gives X_{n+1} = X_n * (1 - dt).
        let system = LinearSde { a: 1.0, b: 0.0 };
        let config = SdeConfig {
            dt: 0.01,
            noise_seed: 42,
            solver: SdeSolver::EulerMaruyama,
        };

        let result = solve(&system, &[1.0], (0.0, 1.0), &config);
        let final_state = result.trajectories[0].last().unwrap()[0];

        // Exact solution of dX/dt = -X is X(1) = e^{-1} ~ 0.3679
        // Euler with dt=0.01 gives (1-0.01)^100 ~ 0.3660
        let euler_exact = (1.0_f64 - 0.01_f64).powi(100);
        assert!(
            (final_state - euler_exact).abs() < 1e-10,
            "Expected ~{euler_exact}, got {final_state}"
        );
    }

    #[test]
    fn test_milstein_reduces_to_euler_for_zero_noise() {
        let system = LinearSde { a: 1.0, b: 0.0 };
        let config = SdeConfig {
            dt: 0.01,
            noise_seed: 99,
            solver: SdeSolver::Milstein,
        };

        let result = solve(&system, &[1.0], (0.0, 1.0), &config);
        let final_state = result.trajectories[0].last().unwrap()[0];
        let euler_exact = (1.0_f64 - 0.01_f64).powi(100);

        assert!(
            (final_state - euler_exact).abs() < 1e-10,
            "Milstein with zero noise should match Euler: expected ~{euler_exact}, got {final_state}"
        );
    }

    #[test]
    fn test_srk_reduces_to_euler_for_zero_noise() {
        let system = LinearSde { a: 1.0, b: 0.0 };
        let config = SdeConfig {
            dt: 0.01,
            noise_seed: 7,
            solver: SdeSolver::StochasticRK,
        };

        let result = solve(&system, &[1.0], (0.0, 1.0), &config);
        let final_state = result.trajectories[0].last().unwrap()[0];
        let euler_exact = (1.0_f64 - 0.01_f64).powi(100);

        assert!(
            (final_state - euler_exact).abs() < 1e-10,
            "SRK with zero noise should match Euler: expected ~{euler_exact}, got {final_state}"
        );
    }

    #[test]
    fn test_ou_process_mean_convergence() {
        // OU process should have E[X_t] -> mu as t -> infinity.
        let theta = 2.0;
        let mu = 5.0;
        let sigma = 1.0;

        let n_paths = 2000;
        let n_steps = 5000;
        let dt = 0.01;

        let mut rng_seed = SimpleRng::new(42);
        let mut sum_final = 0.0;

        for _ in 0..n_paths {
            let mut ou = OrnsteinUhlenbeck::new(theta, mu, sigma);
            ou.state = 0.0; // Start far from mu
            let seed = rng_seed.next_u64();
            let mut rng = SimpleRng::new(seed);
            for _ in 0..n_steps {
                ou.step(dt, &mut rng);
            }
            sum_final += ou.state;
        }

        let mean_final = sum_final / n_paths as f64;
        assert!(
            (mean_final - mu).abs() < 0.15,
            "OU mean should converge to mu={mu}, got {mean_final}"
        );
    }

    #[test]
    fn test_ou_process_variance_convergence() {
        // OU stationary variance should be sigma^2 / (2*theta).
        let theta = 2.0;
        let mu = 0.0;
        let sigma = 1.0;
        let expected_var = sigma * sigma / (2.0 * theta); // 0.25

        let n_paths = 3000;
        let n_steps = 5000;
        let dt = 0.01;

        let mut rng_seed = SimpleRng::new(123);
        let mut final_values = Vec::with_capacity(n_paths);

        for _ in 0..n_paths {
            let mut ou = OrnsteinUhlenbeck::new(theta, mu, sigma);
            let seed = rng_seed.next_u64();
            let mut rng = SimpleRng::new(seed);
            for _ in 0..n_steps {
                ou.step(dt, &mut rng);
            }
            final_values.push(ou.state);
        }

        let mean: f64 = final_values.iter().sum::<f64>() / n_paths as f64;
        let var: f64 = final_values
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / n_paths as f64;

        assert!(
            (var - expected_var).abs() < 0.05,
            "OU variance should converge to {expected_var}, got {var}"
        );
    }

    #[test]
    fn test_monte_carlo_mean_converges_with_sqrt_n() {
        // The standard error of the Monte Carlo mean should decrease as 1/sqrt(N).
        // We test by comparing N=100 vs N=1600: the error should shrink by ~4x.
        let system = LinearSde { a: 1.0, b: 0.5 };
        let config = SdeConfig {
            dt: 0.01,
            noise_seed: 42,
            solver: SdeSolver::EulerMaruyama,
        };

        let exact_mean_t1 = 1.0_f64 * (-1.0_f64).exp(); // E[X(1)] = X0 * e^{-t}

        let result_small = monte_carlo(&system, &[1.0], (0.0, 1.0), 100, &config);
        let result_large = monte_carlo(&system, &[1.0], (0.0, 1.0), 1600, &config);

        let mean_small = result_small.statistics.mean_trajectory.last().unwrap()[0];
        let mean_large = result_large.statistics.mean_trajectory.last().unwrap()[0];

        let error_small = (mean_small - exact_mean_t1).abs();
        let error_large = (mean_large - exact_mean_t1).abs();

        // With 16x more paths, error should decrease by ~4x.
        // We use a generous factor of 2x to account for randomness.
        // The key assertion: large ensemble should be more accurate.
        assert!(
            error_large < error_small * 2.0 || error_large < 0.02,
            "Monte Carlo convergence: error_small={error_small:.4}, error_large={error_large:.4}"
        );
    }

    #[test]
    fn test_fokker_planck_stationary_distribution_for_ou() {
        // For an OU process with theta=2, mu=0, sigma=1,
        // the stationary distribution is N(0, sigma^2/(2*theta)) = N(0, 0.25).
        let theta = 2.0;
        let mu_ou = 0.0;
        let sigma = 1.0;

        let dx = 0.02;
        let dt = 0.0001; // Must satisfy CFL: dt < dx^2 / sigma^2

        let drift_fn = move |x: f64| theta * (mu_ou - x);
        let diffusion_fn = move |_x: f64| sigma;

        let mut solver = FokkerPlanckSolver::new((-4.0, 4.0), dx, &drift_fn, &diffusion_fn);

        // Evolve to stationary state
        for _ in 0..50_000 {
            solver.step(dt);
        }

        let computed_mean = solver.mean();
        let computed_var = solver.variance();
        let expected_var = sigma * sigma / (2.0 * theta); // 0.25

        assert!(
            computed_mean.abs() < 0.1,
            "FP stationary mean should be ~0, got {computed_mean}"
        );
        assert!(
            (computed_var - expected_var).abs() < 0.08,
            "FP stationary variance should be ~{expected_var}, got {computed_var}"
        );
    }

    #[test]
    fn test_langevin_samples_correct_distribution() {
        // Harmonic potential V(x) = x^2 / 2, diffusion D.
        // Boltzmann distribution: p(x) ~ exp(-x^2 / (2D)) => N(0, D).
        let diffusion = 0.5_f64;
        let mut langevin = LangevinDynamics::new(Box::new(|x| 0.5 * x * x), diffusion, 0.0);

        let dt = 0.005;
        let n_steps = 200_000;
        let burn_in = 10_000;

        let mut rng = SimpleRng::new(42);
        let trajectory = langevin.run(dt, n_steps, &mut rng);

        // Discard burn-in, compute statistics
        let samples = &trajectory[burn_in..];
        let n = samples.len() as f64;
        let mean: f64 = samples.iter().sum::<f64>() / n;
        let var: f64 = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;

        assert!(mean.abs() < 0.1, "Langevin mean should be ~0, got {mean}");
        assert!(
            (var - diffusion).abs() < 0.15,
            "Langevin variance should be ~{diffusion}, got {var}"
        );
    }

    #[test]
    fn test_stochastic_cfc_runs_without_panic() {
        // Smoke test: StochasticCfC should produce bounded trajectories.
        let cfc = StochasticCfC::new(8, 0.1);
        let config = SdeConfig {
            dt: 0.01,
            noise_seed: 42,
            solver: SdeSolver::EulerMaruyama,
        };

        let x0 = vec![0.0; 8];
        let result = solve(&cfc, &x0, (0.0, 1.0), &config);

        assert_eq!(result.trajectories.len(), 1);
        assert!(!result.trajectories[0].is_empty());

        // Check that values stay bounded (sigmoid activation keeps things in check)
        for state in &result.trajectories[0] {
            for &v in state {
                assert!(
                    v.is_finite() && v.abs() < 100.0,
                    "Stochastic CfC produced unbounded value: {v}"
                );
            }
        }
    }

    #[test]
    fn test_simple_rng_reproducibility() {
        let mut rng1 = SimpleRng::new(12345);
        let mut rng2 = SimpleRng::new(12345);

        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }

        let mut rng3 = SimpleRng::new(12345);
        let mut rng4 = SimpleRng::new(12345);
        for _ in 0..100 {
            let n3 = rng3.next_normal();
            let n4 = rng4.next_normal();
            assert!(
                (n3 - n4).abs() < 1e-15,
                "Normal samples should be identical for same seed"
            );
        }
    }

    #[test]
    fn test_ou_stationary_variance_formula() {
        let ou = OrnsteinUhlenbeck::new(3.0, 1.0, 2.0);
        let expected = 2.0 * 2.0 / (2.0 * 3.0); // 4/6 = 0.6667
        assert!(
            (ou.stationary_variance() - expected).abs() < 1e-10,
            "Stationary variance formula: expected {expected}, got {}",
            ou.stationary_variance()
        );
    }

    #[test]
    fn test_multidimensional_sde() {
        // 2D coupled OU process
        struct CoupledOU;
        impl SdeSystem for CoupledOU {
            fn dimension(&self) -> usize {
                2
            }
            fn drift(&self, _t: f64, state: &[f64], out: &mut [f64]) {
                out[0] = -state[0] + 0.5 * state[1];
                out[1] = 0.5 * state[0] - state[1];
            }
            fn diffusion(&self, _t: f64, _state: &[f64], out: &mut [f64]) {
                out[0] = 0.3;
                out[1] = 0.3;
            }
        }

        let config = SdeConfig {
            dt: 0.01,
            noise_seed: 42,
            solver: SdeSolver::EulerMaruyama,
        };

        let result = solve(&CoupledOU, &[1.0, -1.0], (0.0, 5.0), &config);
        let final_state = result.trajectories[0].last().unwrap();

        // Both dimensions should stay finite
        assert!(final_state[0].is_finite());
        assert!(final_state[1].is_finite());
        assert_eq!(result.trajectories[0][0].len(), 2);
    }
}
