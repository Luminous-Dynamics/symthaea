// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # CfC/LTC Stability Analysis
//!
//! Dynamical systems analysis for Closed-form Continuous-time (CfC) and
//! Liquid Time-Constant (LTC) neural networks. Stability analysis reveals
//! how neural dynamics create stable conscious states -- attractors in the
//! state space correspond to consolidated percepts, while bifurcations mark
//! transitions between qualitatively different modes of awareness.
//!
//! ## Mathematical Foundation
//!
//! For CfC: h(t) = h_inf + (h_0 - h_inf) * exp(-dt / tau)
//! For LTC: dx/dt = -x/tau + sigma(W*x + b)
//!
//! We analyze these systems through:
//! - **Jacobian computation**: Local linearization of the dynamics
//! - **Eigenvalue analysis**: Stability classification at fixed points
//! - **Lyapunov exponents**: Global trajectory divergence/convergence rates
//! - **Fixed point detection**: Finding attractors via Newton-Raphson
//! - **Bifurcation detection**: Identifying qualitative regime changes
//! - **Contraction analysis**: Uniform convergence guarantees
//!
//! In the consciousness context, a negative maximum Lyapunov exponent indicates
//! that nearby neural trajectories converge -- the hallmark of a stable percept.
//! Positive exponents signal chaotic sensitivity, which may correspond to
//! exploratory or creative cognitive states.

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for stability analysis computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityConfig {
    /// Perturbation size for numerical Jacobian (central differences).
    /// Smaller values increase accuracy but risk floating-point noise.
    /// Default: 1e-7
    pub epsilon: f64,

    /// Maximum iterations for Newton-Raphson fixed-point search.
    /// Default: 100
    pub max_iterations: usize,

    /// Number of time steps for Lyapunov exponent estimation via QR method.
    /// More steps yield more accurate exponents but cost proportionally more.
    /// Default: 10_000
    pub lyapunov_steps: usize,

    /// Convergence tolerance for QR eigenvalue iteration.
    /// Default: 1e-10
    pub qr_tolerance: f64,

    /// Convergence tolerance for Newton-Raphson (norm of f(x)).
    /// Default: 1e-8
    pub newton_tolerance: f64,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-7,
            max_iterations: 100,
            lyapunov_steps: 10_000,
            qr_tolerance: 1e-10,
            newton_tolerance: 1e-8,
        }
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of computing the Jacobian matrix at a point, including eigenvalue
/// decomposition and stability classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JacobianResult {
    /// The Jacobian matrix J_ij = df_i/dx_j, stored row-major.
    pub matrix: Vec<Vec<f64>>,

    /// Eigenvalues as (real_part, imaginary_part) pairs.
    pub eigenvalues: Vec<(f64, f64)>,

    /// Spectral radius: max |lambda_i|.
    pub spectral_radius: f64,

    /// Whether the system is locally stable (all Re(lambda) < 0).
    pub is_stable: bool,
}

/// Result of Lyapunov exponent computation along a trajectory.
///
/// Lyapunov exponents measure the average exponential rate of divergence
/// of nearby trajectories. In consciousness dynamics, the spectrum reveals
/// the effective dimensionality and predictability of neural state evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovResult {
    /// Full Lyapunov spectrum, sorted descending (largest first).
    pub exponents: Vec<f64>,

    /// Maximum (leading) Lyapunov exponent.
    pub max_exponent: f64,

    /// Whether the dynamics are chaotic (max exponent > 0).
    pub is_chaotic: bool,

    /// Kaplan-Yorke dimension: a fractal dimension estimate from the spectrum.
    /// D_KY = j + sum_{i=1}^{j} lambda_i / |lambda_{j+1}|
    /// where j is the largest index such that the partial sum of exponents >= 0.
    pub kaplan_yorke_dimension: f64,
}

/// A fixed point of the dynamical system with stability classification.
///
/// Fixed points are the "resting states" of consciousness dynamics --
/// attractors that the system settles into given sufficient time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPoint {
    /// State vector at the fixed point.
    pub state: Vec<f64>,

    /// Stability type determined by eigenvalue analysis.
    pub stability: FixedPointType,

    /// Eigenvalues of the Jacobian at this fixed point.
    pub eigenvalues: Vec<(f64, f64)>,
}

/// Classification of a fixed point based on the eigenvalue spectrum
/// of the Jacobian evaluated at that point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixedPointType {
    /// All eigenvalues have negative real parts, no imaginary components.
    /// Trajectories converge monotonically -- a stable resting state.
    StableNode,

    /// All eigenvalues have negative real parts, some have imaginary components.
    /// Trajectories spiral inward -- an oscillatory approach to equilibrium.
    StableSpiral,

    /// At least one eigenvalue has positive real part, none imaginary.
    /// Trajectories diverge monotonically from this point.
    UnstableNode,

    /// At least one eigenvalue has positive real part, some imaginary.
    /// Trajectories spiral outward.
    UnstableSpiral,

    /// Mixed: some eigenvalues have positive and some negative real parts.
    /// The system is attracted along some directions and repelled along others.
    SaddlePoint,

    /// All eigenvalues are purely imaginary (Re = 0, Im != 0).
    /// Neutral stability: closed orbits surround this point.
    Center,
}

/// A detected bifurcation point where the qualitative behavior changes.
///
/// In consciousness terms, bifurcations correspond to phase transitions:
/// the moment when a neural system shifts between qualitatively distinct
/// modes of operation (e.g., from a stable percept to an oscillatory search).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BifurcationPoint {
    /// The parameter value at which the bifurcation occurs.
    pub parameter_value: f64,

    /// Type of bifurcation detected.
    pub bifurcation_type: BifurcationType,

    /// Eigenvalues of the Jacobian at the bifurcation point.
    pub eigenvalues_at: Vec<(f64, f64)>,
}

/// Classification of bifurcation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BifurcationType {
    /// A real eigenvalue crosses zero. A stable and unstable fixed point
    /// collide and annihilate (or are born).
    SaddleNode,

    /// A complex conjugate pair crosses the imaginary axis.
    /// A limit cycle is born from (or absorbed into) a fixed point.
    Hopf,

    /// A real eigenvalue crosses zero with symmetric structure.
    /// The fixed point exchanges stability with two new branches.
    PitchFork,

    /// Two fixed points exchange stability as they cross.
    Transcritical,
}

// ============================================================================
// Core Analyzer
// ============================================================================

/// Stability analyzer for continuous-time neural dynamics.
///
/// Provides Jacobian computation, eigenvalue analysis, Lyapunov exponent
/// estimation, fixed-point detection, bifurcation tracking, and contraction
/// analysis for CfC/LTC systems of arbitrary dimension.
#[derive(Debug, Clone)]
pub struct StabilityAnalyzer {
    /// State space dimension.
    dim: usize,

    /// Analysis configuration.
    config: StabilityConfig,
}

impl StabilityAnalyzer {
    /// Create a new stability analyzer for a system of dimension `dim`.
    pub fn new(dim: usize, config: StabilityConfig) -> Self {
        assert!(dim > 0, "Dimension must be positive");
        Self { dim, config }
    }

    /// Create with default configuration.
    pub fn with_defaults(dim: usize) -> Self {
        Self::new(dim, StabilityConfig::default())
    }

    /// Return the state-space dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    // ========================================================================
    // Jacobian Computation
    // ========================================================================

    /// Compute the Jacobian matrix of `f` at point `x` using central finite
    /// differences, then perform eigenvalue analysis.
    ///
    /// J_ij ~ (f_i(x + eps*e_j) - f_i(x - eps*e_j)) / (2*eps)
    ///
    /// The dynamical system `f` maps state vectors to their time derivatives.
    pub fn compute_jacobian(&self, f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> JacobianResult {
        assert_eq!(x.len(), self.dim, "State dimension mismatch");

        let eps = self.config.epsilon;
        let n = self.dim;
        let mut jacobian = vec![vec![0.0; n]; n];

        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();

        for j in 0..n {
            // Perturb dimension j
            x_plus[j] = x[j] + eps;
            x_minus[j] = x[j] - eps;

            let f_plus = f(&x_plus);
            let f_minus = f(&x_minus);

            for i in 0..n {
                jacobian[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * eps);
            }

            // Restore
            x_plus[j] = x[j];
            x_minus[j] = x[j];
        }

        // Eigenvalue analysis
        let eigenvalues = self.compute_eigenvalues(&jacobian);

        let spectral_radius = eigenvalues
            .iter()
            .map(|(re, im)| (re * re + im * im).sqrt())
            .fold(0.0_f64, f64::max);

        let is_stable = eigenvalues.iter().all(|(re, _)| *re < 0.0);

        JacobianResult {
            matrix: jacobian,
            eigenvalues,
            spectral_radius,
            is_stable,
        }
    }

    // ========================================================================
    // Eigenvalue Computation (QR Algorithm)
    // ========================================================================

    /// Compute eigenvalues of a square matrix using the QR algorithm with
    /// implicit Wilkinson shifts. Returns (real, imaginary) pairs.
    ///
    /// Steps:
    /// 1. Reduce to upper Hessenberg form (similarity transformation).
    /// 2. Iterate QR decomposition with shifts until convergence.
    /// 3. Extract eigenvalues from the quasi-upper-triangular result.
    fn compute_eigenvalues(&self, matrix: &[Vec<f64>]) -> Vec<(f64, f64)> {
        let n = matrix.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![(matrix[0][0], 0.0)];
        }

        // Work on a mutable copy
        let mut h = matrix.to_vec();

        // Step 1: Reduce to upper Hessenberg form
        self.hessenberg_reduce(&mut h);

        // Step 2: QR iteration with implicit shifts
        self.qr_iteration(&mut h);

        // Step 3: Extract eigenvalues from quasi-upper-triangular form
        self.extract_eigenvalues(&h)
    }

    /// Reduce a matrix to upper Hessenberg form using Householder reflections.
    /// H is upper Hessenberg if h_{ij} = 0 for i > j + 1.
    fn hessenberg_reduce(&self, h: &mut [Vec<f64>]) {
        let n = h.len();
        if n <= 2 {
            return;
        }

        for k in 0..n - 2 {
            // Build Householder vector for column k, rows k+1..n
            let m = n - k - 1;
            let mut v = vec![0.0; m];
            for i in 0..m {
                v[i] = h[k + 1 + i][k];
            }

            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                continue;
            }

            let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
            v[0] += sign * norm;

            let v_norm_sq = v.iter().map(|x| x * x).sum::<f64>();
            if v_norm_sq < 1e-30 {
                continue;
            }

            // Apply Householder from left: H = (I - 2*v*v'/||v||^2) * H
            // Affects rows k+1..n, columns k..n
            for j in k..n {
                let dot: f64 = (0..m).map(|i| v[i] * h[k + 1 + i][j]).sum();
                let coeff = 2.0 * dot / v_norm_sq;
                for i in 0..m {
                    h[k + 1 + i][j] -= coeff * v[i];
                }
            }

            // Apply Householder from right: H = H * (I - 2*v*v'/||v||^2)
            // Affects rows 0..n, columns k+1..n
            for i in 0..n {
                let dot: f64 = (0..m).map(|j| v[j] * h[i][k + 1 + j]).sum();
                let coeff = 2.0 * dot / v_norm_sq;
                for j in 0..m {
                    h[i][k + 1 + j] -= coeff * v[j];
                }
            }
        }
    }

    /// Implicit QR iteration with Wilkinson shift on an upper Hessenberg matrix.
    fn qr_iteration(&self, h: &mut [Vec<f64>]) {
        let n = h.len();
        let tol = self.config.qr_tolerance;
        let max_iter = 200 * n; // Generous iteration budget

        let mut p = n; // Active matrix size

        for _iter in 0..max_iter {
            if p <= 1 {
                break;
            }

            // Check for convergence: deflate if subdiagonal element is small
            let mut converged = false;
            for q in (1..p).rev() {
                let scale = h[q - 1][q - 1].abs() + h[q][q].abs();
                let threshold = if scale > 0.0 { tol * scale } else { tol };
                if h[q][q - 1].abs() <= threshold {
                    h[q][q - 1] = 0.0;
                    if q == p - 1 {
                        p -= 1;
                        converged = true;
                        break;
                    }
                }
            }
            if converged {
                continue;
            }
            if p <= 1 {
                break;
            }

            // Wilkinson shift: eigenvalue of trailing 2x2 block closest to h[p-1][p-1]
            let a = h[p - 2][p - 2];
            let b = h[p - 2][p - 1];
            let c = h[p - 1][p - 2];
            let d = h[p - 1][p - 1];

            let trace = a + d;
            let det = a * d - b * c;
            let disc = trace * trace - 4.0 * det;

            let shift = if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let lambda1 = (trace + sqrt_disc) / 2.0;
                let lambda2 = (trace - sqrt_disc) / 2.0;
                // Pick the eigenvalue closest to d
                if (lambda1 - d).abs() < (lambda2 - d).abs() {
                    lambda1
                } else {
                    lambda2
                }
            } else {
                // Complex eigenvalues: use real part as shift
                d
            };

            // Shifted QR step via Givens rotations on h[0..p][0..p]
            self.qr_step_givens(h, p, shift);
        }
    }

    /// Single implicit QR step using Givens rotations with shift `mu`.
    fn qr_step_givens(&self, h: &mut [Vec<f64>], p: usize, mu: f64) {
        let n = h.len();

        for k in 0..p - 1 {
            // Compute Givens rotation to zero out h[k+1][k] of (H - mu*I)
            let _a = h[k][k] - mu * if k == 0 { 1.0 } else { 0.0 };
            let _b = h[k + 1][k];

            // For k > 0, we use h[k][k-1] and h[k+1][k-1] from the bulge
            let (a_val, b_val) = if k == 0 {
                (h[0][0] - mu, h[1][0])
            } else {
                (h[k][k - 1], h[k + 1][k - 1])
            };

            let r = (a_val * a_val + b_val * b_val).sqrt();
            if r < 1e-30 {
                continue;
            }
            let cs = a_val / r;
            let sn = b_val / r;

            // Apply Givens rotation from the left: rows k, k+1
            for j in (if k > 0 { k - 1 } else { 0 })..n {
                let t1 = h[k][j];
                let t2 = h[k + 1][j];
                h[k][j] = cs * t1 + sn * t2;
                h[k + 1][j] = -sn * t1 + cs * t2;
            }

            // Apply Givens rotation from the right: columns k, k+1
            let row_limit = (k + 3).min(p).min(n);
            for i in 0..row_limit {
                let t1 = h[i][k];
                let t2 = h[i][k + 1];
                h[i][k] = cs * t1 + sn * t2;
                h[i][k + 1] = -sn * t1 + cs * t2;
            }
        }
    }

    /// Extract eigenvalues from a quasi-upper-triangular (Schur) matrix.
    /// Diagonal 1x1 blocks give real eigenvalues; 2x2 blocks give
    /// complex conjugate pairs.
    fn extract_eigenvalues(&self, h: &[Vec<f64>]) -> Vec<(f64, f64)> {
        let n = h.len();
        let mut eigenvalues = Vec::with_capacity(n);
        let mut i = 0;

        while i < n {
            if i + 1 < n && h[i + 1][i].abs() > self.config.qr_tolerance * 100.0 {
                // 2x2 block: complex conjugate pair
                let a = h[i][i];
                let b = h[i][i + 1];
                let c = h[i + 1][i];
                let d = h[i + 1][i + 1];

                let trace = a + d;
                let det = a * d - b * c;
                let disc = trace * trace - 4.0 * det;

                if disc < 0.0 {
                    let real = trace / 2.0;
                    let imag = (-disc).sqrt() / 2.0;
                    eigenvalues.push((real, imag));
                    eigenvalues.push((real, -imag));
                } else {
                    let sqrt_disc = disc.sqrt();
                    eigenvalues.push(((trace + sqrt_disc) / 2.0, 0.0));
                    eigenvalues.push(((trace - sqrt_disc) / 2.0, 0.0));
                }
                i += 2;
            } else {
                // 1x1 block: real eigenvalue
                eigenvalues.push((h[i][i], 0.0));
                i += 1;
            }
        }

        eigenvalues
    }

    // ========================================================================
    // Lyapunov Exponents
    // ========================================================================

    /// Compute the Lyapunov exponent spectrum of the dynamical system `f`
    /// starting from initial condition `x0`, integrating with time step `dt`.
    ///
    /// Uses the QR decomposition method:
    /// 1. Evolve the state and a set of perturbation vectors.
    /// 2. Periodically QR-decompose the perturbation matrix.
    /// 3. Accumulate log(|R_ii|) to estimate each exponent.
    ///
    /// The maximum Lyapunov exponent (MLE) determines overall stability:
    /// - MLE < 0: stable attractor (convergent consciousness)
    /// - MLE ~ 0: limit cycle / neutral (rhythmic states)
    /// - MLE > 0: chaotic (exploratory / creative dynamics)
    pub fn lyapunov_exponents(
        &self,
        f: &dyn Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        dt: f64,
        steps: usize,
    ) -> LyapunovResult {
        let n = self.dim;
        assert_eq!(x0.len(), n, "Initial state dimension mismatch");

        let total_steps = if steps > 0 {
            steps
        } else {
            self.config.lyapunov_steps
        };

        // Initialize state
        let mut x = x0.to_vec();

        // Initialize perturbation matrix as identity (columns are perturbation vectors)
        let mut q_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            q_mat[i][i] = 1.0;
        }

        let mut lyap_sums = vec![0.0_f64; n];
        let reorth_interval = 10; // Re-orthogonalize every 10 steps
        let mut count = 0u64;

        for step in 0..total_steps {
            // Evolve the state: forward Euler x_{k+1} = x_k + dt * f(x_k)
            let fx = f(&x);
            for i in 0..n {
                x[i] += dt * fx[i];
            }

            // Evolve each perturbation vector using the linearized dynamics
            // delta_{k+1} = delta_k + dt * J(x_k) * delta_k
            // We compute J * delta_j by finite differences for each column j
            let eps = self.config.epsilon;
            for j in 0..n {
                // Perturbation vector j
                let mut x_pert = x.clone();
                for i in 0..n {
                    x_pert[i] += eps * q_mat[i][j];
                }
                let f_pert = f(&x_pert);

                // Linearized evolution: q_j += dt * (f(x + eps*q_j) - f(x)) / eps
                let fx_now = f(&x);
                for i in 0..n {
                    q_mat[i][j] += dt * (f_pert[i] - fx_now[i]) / eps;
                }
            }

            // Periodically QR-decompose to prevent collapse of perturbation vectors
            if (step + 1) % reorth_interval == 0 {
                let r_diag = self.gram_schmidt_qr(&mut q_mat);
                for i in 0..n {
                    if r_diag[i].abs() > 1e-300 {
                        lyap_sums[i] += r_diag[i].abs().ln();
                    }
                }
                count += 1;
            }
        }

        // Compute exponents: lambda_i = (1/T) * sum ln |R_ii|
        let total_time = count as f64 * reorth_interval as f64 * dt;
        let mut exponents: Vec<f64> = if total_time > 0.0 {
            lyap_sums.iter().map(|s| s / total_time).collect()
        } else {
            vec![0.0; n]
        };

        // Sort descending
        exponents.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let max_exponent = exponents.first().copied().unwrap_or(0.0);
        let is_chaotic = max_exponent > 1e-6;

        // Kaplan-Yorke dimension
        let kaplan_yorke_dimension = self.compute_kaplan_yorke(&exponents);

        LyapunovResult {
            exponents,
            max_exponent,
            is_chaotic,
            kaplan_yorke_dimension,
        }
    }

    /// Modified Gram-Schmidt QR decomposition in-place.
    /// Orthogonalizes the columns of `q` and returns the diagonal of R.
    fn gram_schmidt_qr(&self, q: &mut [Vec<f64>]) -> Vec<f64> {
        let n = q[0].len(); // number of columns
        let m = q.len(); // number of rows
        let mut r_diag = vec![0.0; n];

        for j in 0..n {
            // Orthogonalize column j against previous columns
            for k in 0..j {
                // dot = q_k . q_j
                let dot: f64 = (0..m).map(|i| q[i][k] * q[i][j]).sum();
                for i in 0..m {
                    q[i][j] -= dot * q[i][k];
                }
            }

            // Compute norm of column j
            let norm: f64 = (0..m).map(|i| q[i][j] * q[i][j]).sum::<f64>().sqrt();
            r_diag[j] = norm;

            if norm > 1e-300 {
                for i in 0..m {
                    q[i][j] /= norm;
                }
            }
        }

        r_diag
    }

    /// Compute the Kaplan-Yorke dimension from sorted Lyapunov exponents.
    ///
    /// D_KY = j + (sum_{i=1}^{j} lambda_i) / |lambda_{j+1}|
    /// where j is the largest index such that sum_{i=1}^{j} lambda_i >= 0.
    fn compute_kaplan_yorke(&self, exponents: &[f64]) -> f64 {
        if exponents.is_empty() {
            return 0.0;
        }

        let n = exponents.len();
        let mut partial_sum = 0.0;
        let mut j = 0;

        for i in 0..n {
            partial_sum += exponents[i];
            if partial_sum < 0.0 {
                break;
            }
            j = i + 1;
        }

        if j == 0 {
            // All exponents negative or zero
            return 0.0;
        }

        if j >= n {
            // All exponents non-negative
            return n as f64;
        }

        // Sum of first j exponents
        let sum_j: f64 = exponents[..j].iter().sum();
        let next_exp_abs = exponents[j].abs();

        if next_exp_abs > 1e-300 {
            j as f64 + sum_j / next_exp_abs
        } else {
            j as f64
        }
    }

    // ========================================================================
    // Fixed Point Detection
    // ========================================================================

    /// Find fixed points of the dynamical system `f` (where f(x*) = 0)
    /// using Newton-Raphson iteration from multiple initial guesses.
    ///
    /// Fixed points of CfC/LTC dynamics correspond to the stable conscious
    /// states that the brain can settle into. Multiple fixed points represent
    /// the repertoire of stable percepts available to the system.
    pub fn find_fixed_points(
        &self,
        f: &dyn Fn(&[f64]) -> Vec<f64>,
        initial_guesses: &[Vec<f64>],
    ) -> Vec<FixedPoint> {
        let mut fixed_points = Vec::new();

        for guess in initial_guesses {
            if let Some(fp) = self.newton_raphson(f, guess) {
                // Check for duplicates (within tolerance)
                let is_duplicate = fixed_points.iter().any(|existing: &FixedPoint| {
                    let dist: f64 = existing
                        .state
                        .iter()
                        .zip(fp.state.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt();
                    dist < self.config.newton_tolerance * 100.0
                });

                if !is_duplicate {
                    fixed_points.push(fp);
                }
            }
        }

        fixed_points
    }

    /// Newton-Raphson iteration to find a fixed point starting from `x0`.
    ///
    /// x_{n+1} = x_n - J^{-1}(x_n) * f(x_n)
    fn newton_raphson(&self, f: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64]) -> Option<FixedPoint> {
        let n = self.dim;
        let mut x = x0.to_vec();

        for _iter in 0..self.config.max_iterations {
            let fx = f(&x);

            // Check convergence: ||f(x)|| < tolerance
            let norm: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm < self.config.newton_tolerance {
                // Found a fixed point; classify it
                let jac = self.compute_jacobian(f, &x);
                let stability = Self::classify_fixed_point(&jac.eigenvalues);
                return Some(FixedPoint {
                    state: x,
                    stability,
                    eigenvalues: jac.eigenvalues,
                });
            }

            // Compute Jacobian
            let jac_matrix = self.numerical_jacobian(f, &x);

            // Solve J * delta = -f(x) for delta
            let neg_fx: Vec<f64> = fx.iter().map(|v| -v).collect();
            let delta = self.solve_linear_system(&jac_matrix, &neg_fx)?;

            // Update with damping to improve convergence
            let step_size = 1.0_f64;
            for i in 0..n {
                x[i] += step_size * delta[i];
            }

            // Divergence check
            let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !x_norm.is_finite() || x_norm > 1e15 {
                return None;
            }
        }

        None // Did not converge
    }

    /// Compute a numerical Jacobian matrix (without eigenvalue analysis).
    fn numerical_jacobian(&self, f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> Vec<Vec<f64>> {
        let n = self.dim;
        let eps = self.config.epsilon;
        let mut jac = vec![vec![0.0; n]; n];

        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();

        for j in 0..n {
            x_plus[j] = x[j] + eps;
            x_minus[j] = x[j] - eps;

            let f_plus = f(&x_plus);
            let f_minus = f(&x_minus);

            for i in 0..n {
                jac[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * eps);
            }

            x_plus[j] = x[j];
            x_minus[j] = x[j];
        }

        jac
    }

    /// Solve a linear system A * x = b using Gaussian elimination with
    /// partial pivoting. Returns None if the system is singular.
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let n = a.len();
        // Build augmented matrix
        let mut aug: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                r.push(b[i]);
                r
            })
            .collect();

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in col + 1..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }

            if max_val < 1e-14 {
                return None; // Singular
            }

            aug.swap(col, max_row);

            let pivot = aug[col][col];
            for row in col + 1..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in i + 1..n {
                sum -= aug[i][j] * x[j];
            }
            if aug[i][i].abs() < 1e-14 {
                return None;
            }
            x[i] = sum / aug[i][i];
        }

        Some(x)
    }

    /// Classify a fixed point based on its eigenvalue spectrum.
    fn classify_fixed_point(eigenvalues: &[(f64, f64)]) -> FixedPointType {
        if eigenvalues.is_empty() {
            return FixedPointType::StableNode;
        }

        let tol = 1e-8;

        let has_positive_real = eigenvalues.iter().any(|(re, _)| *re > tol);
        let has_negative_real = eigenvalues.iter().any(|(re, _)| *re < -tol);
        let all_zero_real = eigenvalues.iter().all(|(re, _)| re.abs() <= tol);
        let has_imaginary = eigenvalues.iter().any(|(_, im)| im.abs() > tol);

        if has_positive_real && has_negative_real {
            FixedPointType::SaddlePoint
        } else if all_zero_real && has_imaginary {
            FixedPointType::Center
        } else if has_positive_real {
            if has_imaginary {
                FixedPointType::UnstableSpiral
            } else {
                FixedPointType::UnstableNode
            }
        } else {
            // All real parts <= 0 (and at least some < 0 since not all zero)
            if has_imaginary {
                FixedPointType::StableSpiral
            } else {
                FixedPointType::StableNode
            }
        }
    }

    // ========================================================================
    // Bifurcation Detection
    // ========================================================================

    /// Sweep a parameter and detect bifurcations by tracking eigenvalue
    /// crossings of the imaginary axis.
    ///
    /// The callable `f_param` takes a parameter value and returns a closure
    /// representing the dynamical system at that parameter setting.
    /// `x_fixed` provides the approximate fixed point location for Jacobian evaluation;
    /// for more accuracy, combine with `find_fixed_points`.
    ///
    /// In consciousness dynamics, bifurcations mark the boundaries between
    /// qualitatively different cognitive regimes -- e.g., the transition from
    /// wakeful stability to dream-like oscillation.
    pub fn detect_bifurcations(
        &self,
        f_param: &dyn Fn(f64) -> Box<dyn Fn(&[f64]) -> Vec<f64>>,
        x_fixed: &[f64],
        param_range: (f64, f64),
        steps: usize,
    ) -> Vec<BifurcationPoint> {
        let mut bifurcations = Vec::new();
        let step_size = (param_range.1 - param_range.0) / steps as f64;

        let mut prev_eigenvalues: Option<Vec<(f64, f64)>> = None;
        let mut prev_param = param_range.0;

        for s in 0..=steps {
            let param = param_range.0 + s as f64 * step_size;
            let f = f_param(param);
            let jac = self.compute_jacobian(f.as_ref(), x_fixed);

            if let Some(ref prev_eigs) = prev_eigenvalues {
                // Check for eigenvalue crossings of the imaginary axis
                if let Some(bif) = self.detect_crossing(
                    prev_eigs,
                    &jac.eigenvalues,
                    prev_param,
                    param,
                    &jac.eigenvalues,
                ) {
                    bifurcations.push(bif);
                }
            }

            prev_eigenvalues = Some(jac.eigenvalues);
            prev_param = param;
        }

        bifurcations
    }

    /// Detect whether eigenvalues cross the imaginary axis between two
    /// parameter samples. Returns a `BifurcationPoint` if a crossing is found.
    fn detect_crossing(
        &self,
        prev_eigs: &[(f64, f64)],
        curr_eigs: &[(f64, f64)],
        prev_param: f64,
        curr_param: f64,
        eigenvalues_at: &[(f64, f64)],
    ) -> Option<BifurcationPoint> {
        let n = prev_eigs.len().min(curr_eigs.len());
        let tol = 1e-6;

        for i in 0..n {
            let prev_re = prev_eigs[i].0;
            let curr_re = curr_eigs[i].0;

            // Sign change in real part: crossing the imaginary axis
            if prev_re * curr_re < 0.0
                || (prev_re.abs() < tol && curr_re.abs() > tol)
                || (prev_re.abs() > tol && curr_re.abs() < tol)
            {
                // Interpolate parameter value at crossing
                let frac = if (curr_re - prev_re).abs() > 1e-15 {
                    prev_re.abs() / (prev_re.abs() + curr_re.abs())
                } else {
                    0.5
                };
                let crossing_param = prev_param + frac * (curr_param - prev_param);

                let has_imaginary = curr_eigs[i].1.abs() > tol;

                let bifurcation_type = if has_imaginary {
                    BifurcationType::Hopf
                } else {
                    // Real eigenvalue crossing: could be saddle-node, pitchfork,
                    // or transcritical. Without symmetry information we default
                    // to saddle-node, the generic codimension-1 bifurcation.
                    BifurcationType::SaddleNode
                };

                return Some(BifurcationPoint {
                    parameter_value: crossing_param,
                    bifurcation_type,
                    eigenvalues_at: eigenvalues_at.to_vec(),
                });
            }
        }

        None
    }

    // ========================================================================
    // Contraction Analysis
    // ========================================================================

    /// Compute the contraction rate of the system at a point, defined as
    /// the maximum eigenvalue of the symmetric part of the Jacobian:
    ///
    ///   J_s = (J + J^T) / 2
    ///   beta = -lambda_max(J_s)
    ///
    /// If beta > 0, the system is contracting at this point with rate beta,
    /// meaning all trajectories converge exponentially. In consciousness terms,
    /// a contracting system will always settle into its unique attractor
    /// regardless of initial conditions -- a strong form of perceptual stability.
    pub fn contraction_rate(&self, jacobian: &[Vec<f64>]) -> f64 {
        let n = jacobian.len();
        if n == 0 {
            return 0.0;
        }

        // Compute symmetric part J_s = (J + J^T) / 2
        let mut j_sym = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                j_sym[i][j] = (jacobian[i][j] + jacobian[j][i]) / 2.0;
            }
        }

        // Find maximum eigenvalue of J_s
        let eigenvalues = self.compute_eigenvalues(&j_sym);

        // Symmetric matrix => all eigenvalues are real
        let max_eigenvalue = eigenvalues
            .iter()
            .map(|(re, _)| *re)
            .fold(f64::NEG_INFINITY, f64::max);

        // Contraction rate is the negation: beta = -lambda_max
        // Positive beta means contracting
        -max_eigenvalue
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: 1D linear system dx/dt = -alpha * x.
    /// Eigenvalue: lambda = -alpha. Stable for alpha > 0.
    fn linear_1d(alpha: f64) -> impl Fn(&[f64]) -> Vec<f64> {
        move |x: &[f64]| vec![-alpha * x[0]]
    }

    /// Helper: 2D damped oscillator (spiral).
    /// dx/dt = -sigma*x - omega*y
    /// dy/dt =  omega*x - sigma*y
    /// Eigenvalues: -sigma +/- i*omega
    fn damped_oscillator(sigma: f64, omega: f64) -> impl Fn(&[f64]) -> Vec<f64> {
        move |x: &[f64]| vec![-sigma * x[0] - omega * x[1], omega * x[0] - sigma * x[1]]
    }

    #[test]
    fn test_jacobian_linear_1d() {
        // dx/dt = -2x => J = [-2], eigenvalue = -2
        let analyzer = StabilityAnalyzer::with_defaults(1);
        let f = linear_1d(2.0);
        let result = analyzer.compute_jacobian(&f, &[1.0]);

        assert!((result.matrix[0][0] - (-2.0)).abs() < 1e-5);
        assert_eq!(result.eigenvalues.len(), 1);
        assert!((result.eigenvalues[0].0 - (-2.0)).abs() < 1e-5);
        assert!(result.eigenvalues[0].1.abs() < 1e-5);
        assert!(result.is_stable);
    }

    #[test]
    fn test_jacobian_2d_spiral() {
        // Damped oscillator with sigma=0.5, omega=3.0
        // Eigenvalues: -0.5 +/- 3i
        let analyzer = StabilityAnalyzer::with_defaults(2);
        let f = damped_oscillator(0.5, 3.0);
        let result = analyzer.compute_jacobian(&f, &[0.0, 0.0]);

        assert!(result.is_stable, "Damped oscillator should be stable");
        assert_eq!(result.eigenvalues.len(), 2);

        // Both eigenvalues should have real part ~ -0.5
        for (re, _im) in &result.eigenvalues {
            assert!(
                (re - (-0.5)).abs() < 1e-4,
                "Real part should be -0.5, got {}",
                re
            );
        }

        // Should have nonzero imaginary parts
        let has_imaginary = result.eigenvalues.iter().any(|(_, im)| im.abs() > 0.1);
        assert!(has_imaginary, "Should have imaginary eigenvalue components");
    }

    #[test]
    fn test_fixed_point_linear() {
        // dx/dt = -x => fixed point at x* = 0
        let analyzer = StabilityAnalyzer::with_defaults(1);
        let f = linear_1d(1.0);
        let fps = analyzer.find_fixed_points(&f, &[vec![5.0], vec![-3.0]]);

        assert!(!fps.is_empty(), "Should find at least one fixed point");
        let fp = &fps[0];
        assert!(
            fp.state[0].abs() < 1e-6,
            "Fixed point should be near 0, got {}",
            fp.state[0]
        );
        assert_eq!(fp.stability, FixedPointType::StableNode);
    }

    #[test]
    fn test_fixed_point_2d() {
        // dx/dt = -x + y, dy/dt = -x - y => fixed point at origin
        // J = [[-1, 1], [-1, -1]], eigenvalues: -1 +/- i => stable spiral
        let analyzer = StabilityAnalyzer::with_defaults(2);
        let f = |x: &[f64]| -> Vec<f64> { vec![-x[0] + x[1], -x[0] - x[1]] };
        let fps = analyzer.find_fixed_points(&f, &[vec![1.0, 1.0]]);

        assert!(!fps.is_empty(), "Should find a fixed point");
        let fp = &fps[0];
        assert!(fp.state[0].abs() < 1e-6);
        assert!(fp.state[1].abs() < 1e-6);
        assert_eq!(
            fp.stability,
            FixedPointType::StableSpiral,
            "Should classify as stable spiral"
        );
    }

    #[test]
    fn test_lyapunov_dissipative() {
        // dx/dt = -x has Lyapunov exponent lambda = -1
        let analyzer = StabilityAnalyzer::new(
            1,
            StabilityConfig {
                lyapunov_steps: 5000,
                ..Default::default()
            },
        );
        let f = linear_1d(1.0);
        let result = analyzer.lyapunov_exponents(&f, &[1.0], 0.01, 5000);

        assert!(
            result.max_exponent < 0.0,
            "Dissipative system should have negative MLE, got {}",
            result.max_exponent
        );
        assert!(
            !result.is_chaotic,
            "Dissipative linear system is not chaotic"
        );

        // The exponent should be close to -1.0 (within tolerance for numerical method)
        assert!(
            (result.max_exponent - (-1.0)).abs() < 0.5,
            "Lyapunov exponent should be near -1.0, got {}",
            result.max_exponent
        );
    }

    #[test]
    fn test_lyapunov_2d_stable() {
        // 2D damped oscillator: exponents should both be negative
        let analyzer = StabilityAnalyzer::new(
            2,
            StabilityConfig {
                lyapunov_steps: 3000,
                ..Default::default()
            },
        );
        let f = damped_oscillator(0.5, 2.0);
        let result = analyzer.lyapunov_exponents(&f, &[1.0, 0.0], 0.005, 3000);

        assert!(
            result.max_exponent < 0.0,
            "Damped oscillator should have negative max Lyapunov exponent, got {}",
            result.max_exponent
        );
        assert!(!result.is_chaotic);
    }

    #[test]
    fn test_contraction_rate_simple() {
        // For J = [[-2]], the symmetric part is [[-2]],
        // lambda_max = -2, so contraction_rate = 2.0
        let analyzer = StabilityAnalyzer::with_defaults(1);
        let jacobian = vec![vec![-2.0]];
        let rate = analyzer.contraction_rate(&jacobian);

        assert!(
            (rate - 2.0).abs() < 1e-6,
            "Contraction rate should be 2.0, got {}",
            rate
        );
    }

    #[test]
    fn test_contraction_rate_2d() {
        // J = [[-3, 1], [-1, -3]]
        // J_s = (J + J^T)/2 = [[-3, 0], [0, -3]]
        // lambda_max = -3, contraction_rate = 3.0
        let analyzer = StabilityAnalyzer::with_defaults(2);
        let jacobian = vec![vec![-3.0, 1.0], vec![-1.0, -3.0]];
        let rate = analyzer.contraction_rate(&jacobian);

        assert!(
            (rate - 3.0).abs() < 1e-4,
            "Contraction rate should be 3.0, got {}",
            rate
        );
    }

    #[test]
    fn test_contraction_matches_spectral() {
        // Verify that the contraction rate equals -lambda_max of J_s
        // for an arbitrary matrix.
        // J = [[-1, 2], [0, -4]]
        // J_s = [[-1, 1], [1, -4]]
        // Eigenvalues of J_s: trace = -5, det = 4 - 1 = 3
        // lambda = (-5 +/- sqrt(25 - 12))/2 = (-5 +/- sqrt(13))/2
        // lambda_max = (-5 + sqrt(13))/2 ~ (-5 + 3.606)/2 ~ -0.697
        // contraction_rate ~ 0.697
        let analyzer = StabilityAnalyzer::with_defaults(2);
        let jacobian = vec![vec![-1.0, 2.0], vec![0.0, -4.0]];
        let rate = analyzer.contraction_rate(&jacobian);

        let expected = (5.0 - 13.0_f64.sqrt()) / 2.0; // ~ 0.697
        assert!(
            (rate - expected).abs() < 1e-3,
            "Contraction rate should be ~{:.4}, got {:.4}",
            expected,
            rate
        );
    }

    #[test]
    fn test_unstable_system() {
        // dx/dt = +x => eigenvalue = +1, unstable
        let analyzer = StabilityAnalyzer::with_defaults(1);
        let f = |x: &[f64]| -> Vec<f64> { vec![x[0]] };
        let result = analyzer.compute_jacobian(&f, &[0.0]);

        assert!(
            !result.is_stable,
            "Positive eigenvalue system should be unstable"
        );
        assert!(
            (result.eigenvalues[0].0 - 1.0).abs() < 1e-5,
            "Eigenvalue should be +1"
        );
    }

    #[test]
    fn test_saddle_point() {
        // dx/dt = x, dy/dt = -y => eigenvalues +1, -1 => saddle
        let analyzer = StabilityAnalyzer::with_defaults(2);
        let f = |x: &[f64]| -> Vec<f64> { vec![x[0], -x[1]] };
        let fps = analyzer.find_fixed_points(&f, &[vec![0.1, 0.1]]);

        assert!(!fps.is_empty());
        assert_eq!(fps[0].stability, FixedPointType::SaddlePoint);
    }

    #[test]
    fn test_bifurcation_detection() {
        // dx/dt = mu*x - x^3
        // For mu < 0: stable at x=0 (eigenvalue = mu < 0)
        // For mu > 0: unstable at x=0 (eigenvalue = mu > 0) -- pitchfork
        // Bifurcation at mu = 0
        let analyzer = StabilityAnalyzer::with_defaults(1);

        let f_param = |mu: f64| -> Box<dyn Fn(&[f64]) -> Vec<f64>> {
            Box::new(move |x: &[f64]| vec![mu * x[0] - x[0].powi(3)])
        };

        let bifs = analyzer.detect_bifurcations(&f_param, &[0.0], (-1.0, 1.0), 100);

        assert!(!bifs.is_empty(), "Should detect a bifurcation near mu = 0");

        // The bifurcation parameter should be near 0
        let bif = &bifs[0];
        assert!(
            bif.parameter_value.abs() < 0.15,
            "Bifurcation should be near mu=0, got {}",
            bif.parameter_value
        );
    }

    #[test]
    fn test_kaplan_yorke_dimension() {
        let analyzer = StabilityAnalyzer::with_defaults(3);

        // All negative: D_KY = 0
        assert_eq!(analyzer.compute_kaplan_yorke(&[-1.0, -2.0, -3.0]), 0.0);

        // [0.5, -0.2, -1.0]: partial sums 0.5, 0.3, -0.7
        // j = 2, sum_j = 0.3, |lambda_3| = 1.0
        // D_KY = 2 + 0.3/1.0 = 2.3
        let dky = analyzer.compute_kaplan_yorke(&[0.5, -0.2, -1.0]);
        assert!((dky - 2.3).abs() < 1e-10, "D_KY should be 2.3, got {}", dky);
    }

    #[test]
    fn test_eigenvalue_3x3_real() {
        // Diagonal matrix with known eigenvalues -1, -2, -3
        let analyzer = StabilityAnalyzer::with_defaults(3);
        let matrix = vec![
            vec![-1.0, 0.0, 0.0],
            vec![0.0, -2.0, 0.0],
            vec![0.0, 0.0, -3.0],
        ];
        let eigs = analyzer.compute_eigenvalues(&matrix);
        assert_eq!(eigs.len(), 3);

        let mut reals: Vec<f64> = eigs.iter().map(|(re, _)| *re).collect();
        reals.sort_by(|a, b| a.total_cmp(b));

        assert!((reals[0] - (-3.0)).abs() < 1e-6);
        assert!((reals[1] - (-2.0)).abs() < 1e-6);
        assert!((reals[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_linear_solver() {
        // Solve [[2, 1], [1, 3]] * x = [5, 10]
        // x = [0.5, 4.0] approximately: 2*0.5 + 4 = 5, 0.5 + 12 = 12.5 no...
        // Actually: x = [1, 3] gives [2+3=5, 1+9=10]. Yes.
        let analyzer = StabilityAnalyzer::with_defaults(2);
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let x = analyzer.solve_linear_system(&a, &b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }
}
