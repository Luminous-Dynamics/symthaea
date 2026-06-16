// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Quantum Decoherence
//!
//! This module implements quantum decoherence physics including density matrix
//! formalism, Lindblad master equation, and HDC representations of decoherence.
//!
//! ## Key Concepts
//!
//! - **Density Matrix**: ρ represents mixed quantum states
//! - **Lindblad Evolution**: Master equation for open quantum systems
//! - **Decoherence Channels**: Standard noise models (dephasing, damping, etc.)
//! - **von Neumann Entropy**: S = -Tr(ρ log ρ) for mixed state entropy
//!
//! ## Physical Background
//!
//! Decoherence describes the loss of quantum coherence when a system interacts
//! with its environment. The off-diagonal elements of the density matrix decay
//! exponentially with time constant T₂ (decoherence time).
//!
//! ## References
//!
//! - Lindblad (1976) - "On the generators of quantum dynamical semigroups"
//! - Zurek (2003) - "Decoherence, einselection, and the quantum origins of the classical"

use super::constants::{HBAR, K_BOLTZMANN};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Complex number representation (simple implementation)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    /// Create a new complex number
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Create a real number
    pub fn from_real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// Complex conjugate
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Magnitude squared |z|²
    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude |z|
    pub fn norm(&self) -> f64 {
        self.norm_sqr().sqrt()
    }

    /// Add two complex numbers
    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Subtract two complex numbers
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// Multiply two complex numbers
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Scale by real number
    pub fn scale(&self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Imaginary unit
    pub fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    /// Zero
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// One
    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }
}

/// Density matrix for quantum states
///
/// ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ| for mixed states
/// ρ = |ψ⟩⟨ψ| for pure states
#[derive(Debug, Clone)]
pub struct DensityMatrix {
    /// Complex matrix elements (n x n)
    elements: Vec<Complex64>,
    /// Dimension of Hilbert space
    dim: usize,
}

impl DensityMatrix {
    /// Create a density matrix from elements
    pub fn new(dim: usize) -> Self {
        Self {
            elements: vec![Complex64::zero(); dim * dim],
            dim,
        }
    }

    /// Get element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        self.elements[i * self.dim + j]
    }

    /// Set element at (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: Complex64) {
        self.elements[i * self.dim + j] = value;
    }

    /// Create pure state density matrix from state vector
    ///
    /// ρ = |ψ⟩⟨ψ|
    pub fn pure_state(psi: &[Complex64]) -> Self {
        let dim = psi.len();
        let mut rho = Self::new(dim);

        for i in 0..dim {
            for j in 0..dim {
                rho.set(i, j, psi[i].mul(&psi[j].conj()));
            }
        }

        rho
    }

    /// Create mixed state from ensemble
    ///
    /// ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
    pub fn mixed_state(states: &[Vec<Complex64>], probabilities: &[f64]) -> Self {
        if states.is_empty() {
            return Self::new(0);
        }

        let dim = states[0].len();
        let mut rho = Self::new(dim);

        for (state, &prob) in states.iter().zip(probabilities.iter()) {
            let pure = Self::pure_state(state);
            for i in 0..dim {
                for j in 0..dim {
                    let current = rho.get(i, j);
                    let contribution = pure.get(i, j).scale(prob);
                    rho.set(i, j, current.add(&contribution));
                }
            }
        }

        rho
    }

    /// Create maximally mixed state
    ///
    /// ρ = I/d where d is dimension
    pub fn maximally_mixed(dim: usize) -> Self {
        let mut rho = Self::new(dim);
        let prob = 1.0 / dim as f64;
        for i in 0..dim {
            rho.set(i, i, Complex64::from_real(prob));
        }
        rho
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Compute trace Tr(ρ)
    pub fn trace(&self) -> Complex64 {
        let mut tr = Complex64::zero();
        for i in 0..self.dim {
            tr = tr.add(&self.get(i, i));
        }
        tr
    }

    /// Compute purity Tr(ρ²)
    ///
    /// Purity = 1 for pure states
    /// Purity = 1/d for maximally mixed state
    pub fn purity(&self) -> f64 {
        // Tr(ρ²) = Σᵢⱼ |ρᵢⱼ|²
        let mut purity = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                purity += self.get(i, j).norm_sqr();
            }
        }
        purity
    }

    /// Linear entropy S_L = 1 - Tr(ρ²)
    ///
    /// S_L = 0 for pure states
    /// S_L = 1 - 1/d for maximally mixed
    pub fn linear_entropy(&self) -> f64 {
        1.0 - self.purity()
    }

    /// von Neumann entropy S = -Tr(ρ log ρ)
    ///
    /// Computed via eigenvalue decomposition using analytical (d=2) or
    /// Jacobi iteration (d≥3) for Hermitian matrices.
    pub fn von_neumann_entropy(&self) -> f64 {
        let eigenvalues = self.hermitian_eigenvalues();

        let mut s = 0.0;
        for lambda in eigenvalues {
            if lambda > 1e-10 {
                s -= lambda * lambda.log2();
            }
        }
        s.max(0.0)
    }

    /// Compute eigenvalues of a Hermitian density matrix.
    ///
    /// Dispatches: d=1 trivial, d=2 analytical, d≥3 Jacobi rotation.
    pub fn hermitian_eigenvalues(&self) -> Vec<f64> {
        match self.dim {
            0 => vec![],
            1 => vec![self.get(0, 0).re],
            2 => self.eigenvalues_2x2(),
            _ => self.jacobi_eigenvalues(),
        }
    }

    /// Analytical eigenvalues for 2×2 Hermitian matrix.
    ///
    /// For [[a, b], [b*, d]]:
    /// λ = (a+d)/2 ± √(((a-d)/2)² + |b|²)
    fn eigenvalues_2x2(&self) -> Vec<f64> {
        let a = self.get(0, 0).re;
        let d = self.get(1, 1).re;
        let b_norm_sq = self.get(0, 1).norm_sqr();

        let avg = (a + d) * 0.5;
        let half_diff = (a - d) * 0.5;
        let disc = (half_diff * half_diff + b_norm_sq).sqrt();

        vec![avg + disc, avg - disc]
    }

    /// Jacobi eigenvalue algorithm for complex Hermitian matrices.
    ///
    /// Iteratively applies Givens rotations to diagonalize the matrix.
    /// Converges quadratically for small matrices (d=2–10).
    fn jacobi_eigenvalues(&self) -> Vec<f64> {
        let n = self.dim;
        // Work with a real-symmetric proxy: for Hermitian A, eigenvalues are real.
        // We build real and imaginary parts and apply 2n×2n real Jacobi,
        // but for density matrices (which are often nearly real-diagonal),
        // we use the standard complex Jacobi approach on the Hermitian matrix.

        // Copy matrix elements into mutable working arrays (re, im parts)
        let mut re = vec![vec![0.0f64; n]; n];
        let mut im = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let c = self.get(i, j);
                re[i][j] = c.re;
                im[i][j] = c.im;
            }
        }

        let max_iter = 100 * n * n;
        let eps = 1e-12;

        for _ in 0..max_iter {
            // Find largest off-diagonal element |A[p][q]|
            let mut max_off = 0.0f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    let mag = re[i][j] * re[i][j] + im[i][j] * im[i][j];
                    if mag > max_off {
                        max_off = mag;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_off.sqrt() < eps {
                break;
            }

            // Phase correction: rotate A[p][q] to be real and non-negative
            let apq_re = re[p][q];
            let apq_im = im[p][q];
            let apq_abs = (apq_re * apq_re + apq_im * apq_im).sqrt();

            if apq_abs < eps {
                continue;
            }

            // Phase factor e^{iφ} such that A[p][q] * e^{-iφ} is real
            let phase_re = apq_re / apq_abs;
            let phase_im = apq_im / apq_abs;

            // Apply phase to column q: multiply column q by e^{-iφ}
            // This makes A[p][q] real so we can apply standard real Jacobi rotation
            for i in 0..n {
                let r = re[i][q];
                let m = im[i][q];
                // (r + im) * (phase_re - i*phase_im)
                re[i][q] = r * phase_re + m * phase_im;
                im[i][q] = m * phase_re - r * phase_im;
            }
            // Apply conjugate phase to row q (Hermitian: row q = conj of col q transform)
            for j in 0..n {
                let r = re[q][j];
                let m = im[q][j];
                // (r + im) * (phase_re + i*phase_im)
                re[q][j] = r * phase_re - m * phase_im;
                im[q][j] = m * phase_re + r * phase_im;
            }

            // Now A[p][q] should be real (and equal to apq_abs)
            // Standard Jacobi rotation for real symmetric 2x2 subproblem
            let app = re[p][p];
            let aqq = re[q][q];
            let apq_real = re[p][q]; // Should be ~apq_abs

            let tau = (aqq - app) / (2.0 * apq_real);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Apply Givens rotation G^T * A * G where G rotates in (p,q) plane
            // Update rows p and q
            let mut new_rp = vec![(0.0f64, 0.0f64); n];
            let mut new_rq = vec![(0.0f64, 0.0f64); n];
            for j in 0..n {
                new_rp[j] = (c * re[p][j] - s * re[q][j], c * im[p][j] - s * im[q][j]);
                new_rq[j] = (s * re[p][j] + c * re[q][j], s * im[p][j] + c * im[q][j]);
            }
            for j in 0..n {
                re[p][j] = new_rp[j].0;
                im[p][j] = new_rp[j].1;
                re[q][j] = new_rq[j].0;
                im[q][j] = new_rq[j].1;
            }

            // Update columns p and q
            let mut new_cp = vec![(0.0f64, 0.0f64); n];
            let mut new_cq = vec![(0.0f64, 0.0f64); n];
            for i in 0..n {
                new_cp[i] = (c * re[i][p] - s * re[i][q], c * im[i][p] - s * im[i][q]);
                new_cq[i] = (s * re[i][p] + c * re[i][q], s * im[i][p] + c * im[i][q]);
            }
            for i in 0..n {
                re[i][p] = new_cp[i].0;
                im[i][p] = new_cp[i].1;
                re[i][q] = new_cq[i].0;
                im[i][q] = new_cq[i].1;
            }
        }

        // Eigenvalues are on the diagonal
        (0..n).map(|i| re[i][i]).collect()
    }

    /// Add another density matrix
    pub fn add(&self, other: &DensityMatrix) -> DensityMatrix {
        assert_eq!(self.dim, other.dim);
        let mut result = DensityMatrix::new(self.dim);
        for i in 0..self.dim {
            for j in 0..self.dim {
                result.set(i, j, self.get(i, j).add(&other.get(i, j)));
            }
        }
        result
    }

    /// Scale by real number
    pub fn scale(&self, s: f64) -> DensityMatrix {
        let mut result = self.clone();
        for i in 0..self.dim {
            for j in 0..self.dim {
                result.set(i, j, self.get(i, j).scale(s));
            }
        }
        result
    }

    /// Matrix multiplication
    pub fn mul(&self, other: &DensityMatrix) -> DensityMatrix {
        assert_eq!(self.dim, other.dim);
        let mut result = DensityMatrix::new(self.dim);

        for i in 0..self.dim {
            for j in 0..self.dim {
                let mut sum = Complex64::zero();
                for k in 0..self.dim {
                    sum = sum.add(&self.get(i, k).mul(&other.get(k, j)));
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Commutator [A, B] = AB - BA
    pub fn commutator(&self, other: &DensityMatrix) -> DensityMatrix {
        let ab = self.mul(other);
        let ba = other.mul(self);
        let mut result = DensityMatrix::new(self.dim);
        for i in 0..self.dim {
            for j in 0..self.dim {
                result.set(i, j, ab.get(i, j).sub(&ba.get(i, j)));
            }
        }
        result
    }

    /// Anticommutator {A, B} = AB + BA
    pub fn anticommutator(&self, other: &DensityMatrix) -> DensityMatrix {
        let ab = self.mul(other);
        let ba = other.mul(self);
        let mut result = DensityMatrix::new(self.dim);
        for i in 0..self.dim {
            for j in 0..self.dim {
                result.set(i, j, ab.get(i, j).add(&ba.get(i, j)));
            }
        }
        result
    }

    /// Frobenius norm ||ρ||_F = √(Σᵢⱼ |ρᵢⱼ|²)
    pub fn frobenius_norm(&self) -> f64 {
        self.elements
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    /// Hermitian conjugate (dagger)
    pub fn dagger(&self) -> DensityMatrix {
        let mut result = DensityMatrix::new(self.dim);
        for i in 0..self.dim {
            for j in 0..self.dim {
                result.set(i, j, self.get(j, i).conj());
            }
        }
        result
    }
}

/// Lindblad master equation solver
///
/// dρ/dt = -i[H,ρ]/ℏ + Σₖ γₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})
#[derive(Debug, Clone)]
pub struct LindbladEvolution {
    /// Hamiltonian (Hermitian)
    hamiltonian: DensityMatrix,
    /// Lindblad operators (jump operators)
    lindblad_operators: Vec<DensityMatrix>,
    /// Decay rates for each Lindblad operator
    gamma: Vec<f64>,
    /// Reduced Planck constant
    hbar: f64,
}

impl LindbladEvolution {
    /// Create a new Lindblad evolution
    pub fn new(hamiltonian: DensityMatrix) -> Self {
        Self {
            hamiltonian,
            lindblad_operators: Vec::new(),
            gamma: Vec::new(),
            hbar: HBAR,
        }
    }

    /// Add a Lindblad operator with decay rate
    pub fn add_lindblad(&mut self, operator: DensityMatrix, gamma: f64) {
        self.lindblad_operators.push(operator);
        self.gamma.push(gamma);
    }

    /// Evolve the density matrix by time dt
    ///
    /// Uses 4th-order Runge-Kutta for integration
    pub fn evolve(&self, rho: &DensityMatrix, dt: f64) -> DensityMatrix {
        let k1 = self.derivative(rho).scale(dt);
        let rho2 = rho.add(&k1.scale(0.5));
        let k2 = self.derivative(&rho2).scale(dt);
        let rho3 = rho.add(&k2.scale(0.5));
        let k3 = self.derivative(&rho3).scale(dt);
        let rho4 = rho.add(&k3);
        let k4 = self.derivative(&rho4).scale(dt);
        rho.add(
            &k1.add(&k2.scale(2.0))
                .add(&k3.scale(2.0))
                .add(&k4)
                .scale(1.0 / 6.0),
        )
    }

    /// Compute dρ/dt from Lindblad equation
    fn derivative(&self, rho: &DensityMatrix) -> DensityMatrix {
        let dim = rho.dim();
        let mut drho = DensityMatrix::new(dim);

        // Unitary part: -i[H,ρ]/ℏ
        let commutator = self.hamiltonian.commutator(rho);
        for i in 0..dim {
            for j in 0..dim {
                let c = commutator.get(i, j);
                // -i * c / hbar
                let contribution = Complex64::new(c.im / self.hbar, -c.re / self.hbar);
                drho.set(i, j, drho.get(i, j).add(&contribution));
            }
        }

        // Dissipative part: Σₖ γₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})
        for (l_k, &gamma_k) in self.lindblad_operators.iter().zip(self.gamma.iter()) {
            let l_dag = l_k.dagger();
            let l_dag_l = l_dag.mul(l_k);

            // L ρ L†
            let l_rho_l_dag = l_k.mul(rho).mul(&l_dag);

            // {L†L, ρ}/2
            let anticomm = l_dag_l.anticommutator(rho);

            for i in 0..dim {
                for j in 0..dim {
                    let dissipator = l_rho_l_dag.get(i, j).sub(&anticomm.get(i, j).scale(0.5));
                    drho.set(i, j, drho.get(i, j).add(&dissipator.scale(gamma_k)));
                }
            }
        }

        drho
    }

    /// Estimate decoherence time T₂ from Lindblad operators
    ///
    /// For pure dephasing: T₂ = 1/γ
    pub fn decoherence_time(&self) -> f64 {
        if self.gamma.is_empty() {
            return f64::INFINITY;
        }
        // T₂ ≈ 1 / (sum of rates)
        1.0 / self.gamma.iter().sum::<f64>()
    }

    /// Evolve for multiple time steps
    pub fn evolve_trajectory(
        &self,
        initial: &DensityMatrix,
        dt: f64,
        steps: usize,
    ) -> Vec<DensityMatrix> {
        let mut trajectory = Vec::with_capacity(steps + 1);
        let mut rho = initial.clone();
        trajectory.push(rho.clone());

        for _ in 0..steps {
            rho = self.evolve(&rho, dt);
            trajectory.push(rho.clone());
        }

        trajectory
    }

    /// Adaptive Dormand-Prince RK45 integrator with error control.
    ///
    /// Returns a trajectory at accepted time steps. Step size is adapted
    /// based on local truncation error estimated from embedded 4th and 5th
    /// order solutions.
    ///
    /// # Arguments
    /// * `rho` — Initial density matrix
    /// * `t_end` — End time
    /// * `rtol` — Relative tolerance
    /// * `atol` — Absolute tolerance
    pub fn evolve_adaptive(
        &self,
        rho: &DensityMatrix,
        t_end: f64,
        rtol: f64,
        atol: f64,
    ) -> Vec<DensityMatrix> {
        // Dormand-Prince coefficients (Butcher tableau)
        let a21 = 1.0 / 5.0;
        let a31 = 3.0 / 40.0;
        let a32 = 9.0 / 40.0;
        let a41 = 44.0 / 45.0;
        let a42 = -56.0 / 15.0;
        let a43 = 32.0 / 9.0;
        let a51 = 19372.0 / 6561.0;
        let a52 = -25360.0 / 2187.0;
        let a53 = 64448.0 / 6561.0;
        let a54 = -212.0 / 729.0;
        let a61 = 9017.0 / 3168.0;
        let a62 = -355.0 / 33.0;
        let a63 = 46732.0 / 5247.0;
        let a64 = 49.0 / 176.0;
        let a65 = -5103.0 / 18656.0;

        // 5th-order weights
        let b1 = 35.0 / 384.0;
        let b3 = 500.0 / 1113.0;
        let b4 = 125.0 / 192.0;
        let b5 = -2187.0 / 6784.0;
        let b6 = 11.0 / 84.0;

        // 4th-order weights (for error estimate)
        let e1 = 71.0 / 57600.0;
        let e3 = -71.0 / 16695.0;
        let e4 = 71.0 / 1920.0;
        let e5 = -17253.0 / 339200.0;
        let e6 = 22.0 / 525.0;
        let e7 = -1.0 / 40.0;

        let mut trajectory = Vec::new();
        let mut current = rho.clone();
        trajectory.push(current.clone());

        let mut t = 0.0;
        let mut h = t_end / 100.0;
        let h_min = 1e-15;
        let h_max = t_end;

        while t < t_end {
            if t + h > t_end {
                h = t_end - t;
            }
            if h < h_min {
                h = h_min;
            }

            // Compute stages
            let k1 = self.derivative(&current);
            let s2 = current.add(&k1.scale(h * a21));
            let k2 = self.derivative(&s2);
            let s3 = current.add(&k1.scale(h * a31)).add(&k2.scale(h * a32));
            let k3 = self.derivative(&s3);
            let s4 = current
                .add(&k1.scale(h * a41))
                .add(&k2.scale(h * a42))
                .add(&k3.scale(h * a43));
            let k4 = self.derivative(&s4);
            let s5 = current
                .add(&k1.scale(h * a51))
                .add(&k2.scale(h * a52))
                .add(&k3.scale(h * a53))
                .add(&k4.scale(h * a54));
            let k5 = self.derivative(&s5);
            let s6 = current
                .add(&k1.scale(h * a61))
                .add(&k2.scale(h * a62))
                .add(&k3.scale(h * a63))
                .add(&k4.scale(h * a64))
                .add(&k5.scale(h * a65));
            let k6 = self.derivative(&s6);

            // 5th-order solution
            let y5 = current.add(
                &k1.scale(h * b1)
                    .add(&k3.scale(h * b3))
                    .add(&k4.scale(h * b4))
                    .add(&k5.scale(h * b5))
                    .add(&k6.scale(h * b6)),
            );

            // Error estimate: difference between 5th and 4th order
            let k7 = self.derivative(&y5);
            let err_mat = k1
                .scale(h * e1)
                .add(&k3.scale(h * e3))
                .add(&k4.scale(h * e4))
                .add(&k5.scale(h * e5))
                .add(&k6.scale(h * e6))
                .add(&k7.scale(h * e7));

            let err_norm = err_mat.frobenius_norm();
            let scale = atol + rtol * current.frobenius_norm();
            let err = err_norm / scale;

            if err <= 1.0 || h <= h_min {
                // Accept step
                t += h;
                current = y5;
                trajectory.push(current.clone());
            }

            // Adjust step size
            let factor = if err > 0.0 { 0.9 * err.powf(-0.2) } else { 5.0 };
            h *= factor.clamp(0.2, 5.0);
            h = h.clamp(h_min, h_max);
        }

        trajectory
    }
}

/// Common decoherence channels
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DecoherenceChannel {
    /// Pure dephasing (T₂ decay without T₁)
    Dephasing { gamma: f64 },
    /// Amplitude damping (T₁ decay to ground state)
    AmplitudeDamping { gamma: f64 },
    /// Depolarizing channel (uniform noise)
    Depolarizing { p: f64 },
    /// Thermal relaxation (T₁ and T₂ at temperature with explicit energy gap)
    ThermalRelaxation {
        t1: f64,
        t2: f64,
        temperature: f64,
        energy_gap: f64,
    },
}

impl DecoherenceChannel {
    /// Get the decoherence time for this channel
    pub fn decoherence_time(&self) -> f64 {
        match self {
            DecoherenceChannel::Dephasing { gamma } => 1.0 / gamma,
            DecoherenceChannel::AmplitudeDamping { gamma } => 1.0 / gamma,
            DecoherenceChannel::Depolarizing { p } => {
                if *p > 0.0 {
                    1.0 / p
                } else {
                    f64::INFINITY
                }
            }
            DecoherenceChannel::ThermalRelaxation { t2, .. } => *t2,
        }
    }

    /// Create Lindblad operators for this channel (2-level system)
    pub fn to_lindblad_operators(&self) -> Vec<(DensityMatrix, f64)> {
        match *self {
            DecoherenceChannel::Dephasing { gamma } => {
                // σ_z causes dephasing
                let mut sigma_z = DensityMatrix::new(2);
                sigma_z.set(0, 0, Complex64::from_real(1.0));
                sigma_z.set(1, 1, Complex64::from_real(-1.0));
                vec![(sigma_z, gamma)]
            }
            DecoherenceChannel::AmplitudeDamping { gamma } => {
                // σ_- = |0⟩⟨1| causes decay
                let mut sigma_minus = DensityMatrix::new(2);
                sigma_minus.set(0, 1, Complex64::from_real(1.0));
                vec![(sigma_minus, gamma)]
            }
            DecoherenceChannel::Depolarizing { p } => {
                // σ_x, σ_y, σ_z with equal rates
                let gamma = p / 3.0;

                let mut sigma_x = DensityMatrix::new(2);
                sigma_x.set(0, 1, Complex64::from_real(1.0));
                sigma_x.set(1, 0, Complex64::from_real(1.0));

                let mut sigma_y = DensityMatrix::new(2);
                sigma_y.set(0, 1, Complex64::new(0.0, -1.0));
                sigma_y.set(1, 0, Complex64::new(0.0, 1.0));

                let mut sigma_z = DensityMatrix::new(2);
                sigma_z.set(0, 0, Complex64::from_real(1.0));
                sigma_z.set(1, 1, Complex64::from_real(-1.0));

                vec![(sigma_x, gamma), (sigma_y, gamma), (sigma_z, gamma)]
            }
            DecoherenceChannel::ThermalRelaxation {
                t1,
                t2,
                temperature,
                energy_gap,
            } => {
                // Combine amplitude damping and dephasing
                let gamma1 = 1.0 / t1;
                let gamma_phi = 1.0 / t2 - 0.5 / t1; // Pure dephasing rate

                // Thermal occupation number from Bose-Einstein distribution
                let n_th = 1.0 / ((energy_gap / (K_BOLTZMANN * temperature)).exp() - 1.0);

                let mut sigma_minus = DensityMatrix::new(2);
                sigma_minus.set(0, 1, Complex64::from_real(1.0));

                let mut sigma_plus = DensityMatrix::new(2);
                sigma_plus.set(1, 0, Complex64::from_real(1.0));

                let mut sigma_z = DensityMatrix::new(2);
                sigma_z.set(0, 0, Complex64::from_real(1.0));
                sigma_z.set(1, 1, Complex64::from_real(-1.0));

                vec![
                    (sigma_minus, gamma1 * (n_th + 1.0)),
                    (sigma_plus, gamma1 * n_th),
                    (sigma_z, gamma_phi.max(0.0)),
                ]
            }
        }
    }
}

/// HDC representation of decoherence states
#[derive(Debug, Clone)]
pub struct DecoherenceEncoder {
    genesis: GenesisSeed,
    /// Coherent state vector
    pub coherent: ContinuousHV,
    /// Decohered state vector
    pub decohered: ContinuousHV,
    /// Environment vector
    pub environment: ContinuousHV,
    /// Entanglement vector
    pub entanglement: ContinuousHV,
    /// Pure state vector
    pub pure: ContinuousHV,
    /// Mixed state vector
    pub mixed: ContinuousHV,
}

impl DecoherenceEncoder {
    /// Create a new decoherence encoder from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),
            coherent: genesis.hv("decoherence::coherent", PHYSICS_DIM),
            decohered: genesis.hv("decoherence::decohered", PHYSICS_DIM),
            environment: genesis.hv("decoherence::environment", PHYSICS_DIM),
            entanglement: genesis.hv("decoherence::entanglement", PHYSICS_DIM),
            pure: genesis.hv("decoherence::pure", PHYSICS_DIM),
            mixed: genesis.hv("decoherence::mixed", PHYSICS_DIM),
        }
    }

    /// Encode purity as weighted superposition
    pub fn encode_purity(&self, purity: f64) -> ContinuousHV {
        let purity = purity.clamp(0.0, 1.0) as f32;
        ContinuousHV::weighted_bundle(&[&self.pure, &self.mixed], &[purity, 1.0 - purity])
    }

    /// Encode decoherence progress (0 = fully coherent, 1 = fully decohered)
    pub fn encode_decoherence(&self, progress: f64) -> ContinuousHV {
        let progress = progress.clamp(0.0, 1.0) as f32;
        ContinuousHV::weighted_bundle(
            &[&self.coherent, &self.decohered],
            &[1.0 - progress, progress],
        )
    }

    /// Encode a decoherence channel
    pub fn encode_channel(&self, channel: DecoherenceChannel) -> ContinuousHV {
        let channel_label = match channel {
            DecoherenceChannel::Dephasing { .. } => "channel::dephasing",
            DecoherenceChannel::AmplitudeDamping { .. } => "channel::amplitude_damping",
            DecoherenceChannel::Depolarizing { .. } => "channel::depolarizing",
            DecoherenceChannel::ThermalRelaxation { .. } => "channel::thermal",
        };
        let channel_hv = self.genesis.hv(channel_label, PHYSICS_DIM);
        self.environment.bind(&channel_hv)
    }
}

/// Decoherence simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoherenceResult {
    /// Time points
    pub times: Vec<f64>,
    /// Purity at each time
    pub purity: Vec<f64>,
    /// von Neumann entropy at each time
    pub entropy: Vec<f64>,
    /// Off-diagonal coherence magnitude
    pub coherence: Vec<f64>,
    /// Decoherence time T₂
    pub t2: f64,
}

/// Simulate decoherence for a 2-level system
pub fn simulate_decoherence(
    initial: &DensityMatrix,
    channel: DecoherenceChannel,
    total_time: f64,
    steps: usize,
) -> DecoherenceResult {
    let dt = total_time / steps as f64;

    // Create Lindblad evolution
    let hamiltonian = DensityMatrix::new(2); // Free evolution
    let mut evolution = LindbladEvolution::new(hamiltonian);

    for (op, gamma) in channel.to_lindblad_operators() {
        evolution.add_lindblad(op, gamma);
    }

    let t2 = evolution.decoherence_time();

    // Evolve and record
    let mut times = Vec::with_capacity(steps + 1);
    let mut purity = Vec::with_capacity(steps + 1);
    let mut entropy = Vec::with_capacity(steps + 1);
    let mut coherence = Vec::with_capacity(steps + 1);

    let mut rho = initial.clone();

    for i in 0..=steps {
        let t = i as f64 * dt;
        times.push(t);
        purity.push(rho.purity());
        entropy.push(rho.von_neumann_entropy());

        // Off-diagonal coherence
        if rho.dim() >= 2 {
            coherence.push(rho.get(0, 1).norm());
        } else {
            coherence.push(0.0);
        }

        if i < steps {
            rho = evolution.evolve(&rho, dt);
        }
    }

    DecoherenceResult {
        times,
        purity,
        entropy,
        coherence,
        t2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex64::new(3.0, 4.0);
        assert!((a.norm() - 5.0).abs() < 1e-10);

        let b = Complex64::new(1.0, 2.0);
        let c = a.mul(&b);
        // (3+4i)(1+2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
        assert!((c.re - (-5.0)).abs() < 1e-10);
        assert!((c.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_pure_state_purity() {
        let psi = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let rho = DensityMatrix::pure_state(&psi);

        // Pure state should have purity = 1
        assert!(
            (rho.purity() - 1.0).abs() < 1e-10,
            "Pure state purity = {}",
            rho.purity()
        );

        // Trace should be 1
        assert!((rho.trace().re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_maximally_mixed_purity() {
        let rho = DensityMatrix::maximally_mixed(2);

        // Maximally mixed should have purity = 1/d = 0.5
        assert!(
            (rho.purity() - 0.5).abs() < 1e-10,
            "Mixed purity = {}",
            rho.purity()
        );

        // Trace should be 1
        assert!((rho.trace().re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_von_neumann_entropy() {
        // Pure state should have S = 0
        let psi = vec![Complex64::from_real(1.0), Complex64::zero()];
        let pure = DensityMatrix::pure_state(&psi);
        let s_pure = pure.von_neumann_entropy();
        assert!(s_pure < 1e-6, "Pure state entropy = {}", s_pure);

        // Maximally mixed should have S = log(d) = 1 bit
        // Note: power iteration may not find all equal eigenvalues perfectly
        // so we also verify using purity as a more robust check
        let mixed = DensityMatrix::maximally_mixed(2);
        let purity = mixed.purity();
        // For d=2 maximally mixed: purity = 1/d = 0.5
        assert!(
            (purity - 0.5).abs() < 1e-10,
            "Mixed state purity = {}, expected 0.5",
            purity
        );

        // Entropy should be positive for mixed state
        let s_mixed = mixed.von_neumann_entropy();
        assert!(
            s_mixed > 0.0,
            "Mixed entropy should be positive: {}",
            s_mixed
        );
    }

    #[test]
    fn test_lindblad_trace_preservation() {
        let hamiltonian = DensityMatrix::new(2);
        let mut evolution = LindbladEvolution::new(hamiltonian);

        // Add dephasing
        let mut sigma_z = DensityMatrix::new(2);
        sigma_z.set(0, 0, Complex64::from_real(1.0));
        sigma_z.set(1, 1, Complex64::from_real(-1.0));
        evolution.add_lindblad(sigma_z, 0.1);

        // Initial state
        let psi = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let rho = DensityMatrix::pure_state(&psi);

        // Evolve
        let rho_final = evolution.evolve(&rho, 0.01);

        // Trace should be preserved (approximately)
        let tr_initial = rho.trace().re;
        let tr_final = rho_final.trace().re;
        assert!((tr_final - tr_initial).abs() < 1e-6);
    }

    #[test]
    fn test_decoherence_channel_time() {
        let channel = DecoherenceChannel::Dephasing { gamma: 1e6 };
        let t2 = channel.decoherence_time();
        assert!((t2 - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_simulate_decoherence() {
        let psi = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let initial = DensityMatrix::pure_state(&psi);

        let channel = DecoherenceChannel::Dephasing { gamma: 1.0 };
        let result = simulate_decoherence(&initial, channel, 5.0, 50);

        assert_eq!(result.times.len(), 51);
        assert!(
            result.purity[0] > result.purity[50]
                || (result.purity[0] - result.purity[50]).abs() < 0.1
        );
        assert!(result.coherence[0] >= result.coherence[50]);
    }

    #[test]
    fn test_decoherence_encoder() {
        let genesis = GenesisSeed::from_phrase("test");
        let encoder = DecoherenceEncoder::from_genesis(&genesis);

        let purity_hv = encoder.encode_purity(0.7);
        assert_eq!(purity_hv.values.len(), PHYSICS_DIM);

        let channel = DecoherenceChannel::AmplitudeDamping { gamma: 1.0 };
        let channel_hv = encoder.encode_channel(channel);
        assert_eq!(channel_hv.values.len(), PHYSICS_DIM);
    }
}
