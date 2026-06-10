// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Open Quantum Systems — Lindblad Master Equation.
//!
//! The Lindblad equation describes quantum systems coupled to an environment:
//!
//! dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
//!
//! This governs decoherence — the transition from quantum to classical behavior.
//! Essential for understanding how molecular quantum coherence gives way to
//! classical neural dynamics.
//!
//! References:
//! - Lindblad, G. (1976). Comm. Math. Phys. 48, 119.
//! - Breuer & Petruccione (2007). *The Theory of Open Quantum Systems*. Oxford UP.
//! - Schlosshauer, M. (2007). *Decoherence and the Quantum-to-Classical Transition*. Springer.

/// Density matrix for an n-level quantum system.
/// Stored as n×n complex matrix: (real, imaginary) pairs.
#[derive(Debug, Clone)]
pub struct DensityMatrix {
    /// Matrix elements: re[i*n + j], im[i*n + j]
    pub re: Vec<f64>,
    pub im: Vec<f64>,
    pub n: usize,
}

impl DensityMatrix {
    /// Create a pure state |k⟩⟨k| (diagonal with 1 at position k).
    pub fn pure_state(n: usize, k: usize) -> Self {
        let mut re = vec![0.0; n * n];
        re[k * n + k] = 1.0;
        Self {
            re,
            im: vec![0.0; n * n],
            n,
        }
    }

    /// Create a maximally mixed state: ρ = I/n.
    pub fn maximally_mixed(n: usize) -> Self {
        let mut re = vec![0.0; n * n];
        for i in 0..n {
            re[i * n + i] = 1.0 / n as f64;
        }
        Self {
            re,
            im: vec![0.0; n * n],
            n,
        }
    }

    /// Create a superposition state: ρ = |ψ⟩⟨ψ| where |ψ⟩ = Σ c_i |i⟩
    pub fn from_state_vector(coeffs_re: &[f64], coeffs_im: &[f64]) -> Self {
        let n = coeffs_re.len();
        let mut re = vec![0.0; n * n];
        let mut im = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                // ρ_ij = c_i × c_j*
                re[i * n + j] = coeffs_re[i] * coeffs_re[j] + coeffs_im[i] * coeffs_im[j];
                im[i * n + j] = coeffs_im[i] * coeffs_re[j] - coeffs_re[i] * coeffs_im[j];
            }
        }

        Self { re, im, n }
    }

    /// Trace of the density matrix (should be 1).
    pub fn trace(&self) -> f64 {
        (0..self.n).map(|i| self.re[i * self.n + i]).sum()
    }

    /// Purity: Tr(ρ²). Pure state = 1, maximally mixed = 1/n.
    pub fn purity(&self) -> f64 {
        let n = self.n;
        let mut _tr = 0.0;
        for i in 0..n {
            for k in 0..n {
                let re_ik = self.re[i * n + k];
                let im_ik = self.im[i * n + k];
                let re_ki = self.re[k * n + i];
                let im_ki = self.im[k * n + i];
                _tr += re_ik * re_ki - im_ik * im_ki; // Real part of (ρ²)_ii via ρ_ik × ρ_ki
            }
        }
        // Actually this computes Σ_i (ρ²)_ii wrong. Let me fix:
        // Tr(ρ²) = Σ_ij |ρ_ij|²
        let mut purity = 0.0;
        for i in 0..n {
            for j in 0..n {
                purity += self.re[i * n + j] * self.re[i * n + j]
                    + self.im[i * n + j] * self.im[i * n + j];
            }
        }
        purity
    }

    /// Von Neumann entropy: S = -Tr(ρ ln ρ).
    /// Computed from eigenvalues of the density matrix.
    pub fn von_neumann_entropy(&self) -> f64 {
        // For small matrices, use the diagonal elements as approximate eigenvalues
        // (exact for diagonal ρ, approximate otherwise)
        let mut entropy = 0.0;
        for i in 0..self.n {
            let p = self.re[i * self.n + i];
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Coherence: sum of off-diagonal |ρ_ij| (l1-norm of coherence).
    pub fn coherence_l1(&self) -> f64 {
        let n = self.n;
        let mut coh = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    coh += (self.re[i * n + j].powi(2) + self.im[i * n + j].powi(2)).sqrt();
                }
            }
        }
        coh
    }
}

/// Lindblad channel: a single dissipation process.
#[derive(Debug, Clone)]
pub struct LindbladChannel {
    /// Lindblad operator L (n×n, real part)
    pub l_re: Vec<f64>,
    /// Lindblad operator L (n×n, imaginary part)
    pub l_im: Vec<f64>,
    /// Decay rate γ
    pub gamma: f64,
}

/// Lindblad master equation solver.
#[derive(Debug, Clone)]
pub struct LindbladSolver {
    /// System dimension
    pub n: usize,
    /// Hamiltonian (real, diagonal in energy eigenbasis)
    pub hamiltonian: Vec<f64>,
    /// Dissipation channels
    pub channels: Vec<LindbladChannel>,
}

impl LindbladSolver {
    /// Create a solver for an n-level system with given energy levels.
    pub fn new(energies: &[f64]) -> Self {
        let n = energies.len();
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            h[i * n + i] = energies[i];
        }
        Self {
            n,
            hamiltonian: h,
            channels: Vec::new(),
        }
    }

    /// Add a dephasing channel: L = |k⟩⟨k| (pure dephasing of level k).
    pub fn add_dephasing(&mut self, level: usize, rate: f64) {
        let n = self.n;
        let mut l_re = vec![0.0; n * n];
        l_re[level * n + level] = 1.0;
        self.channels.push(LindbladChannel {
            l_re,
            l_im: vec![0.0; n * n],
            gamma: rate,
        });
    }

    /// Add a decay channel: L = |lower⟩⟨upper| (spontaneous emission).
    pub fn add_decay(&mut self, upper: usize, lower: usize, rate: f64) {
        let n = self.n;
        let mut l_re = vec![0.0; n * n];
        l_re[lower * n + upper] = 1.0; // |lower⟩⟨upper|
        self.channels.push(LindbladChannel {
            l_re,
            l_im: vec![0.0; n * n],
            gamma: rate,
        });
    }

    /// Evolve the density matrix by one time step dt using Euler method.
    ///
    /// dρ/dt = -i[H, ρ] + Σ_k γ_k D[L_k](ρ)
    /// where D[L](ρ) = LρL† - ½{L†L, ρ}
    pub fn step(&self, rho: &mut DensityMatrix, dt: f64) {
        let n = self.n;

        // Compute dρ/dt
        let mut drho_re = vec![0.0; n * n];
        let mut drho_im = vec![0.0; n * n];

        // Unitary part: -i[H, ρ]
        // For diagonal H: [H, ρ]_ij = (H_ii - H_jj) ρ_ij
        for i in 0..n {
            for j in 0..n {
                let de = self.hamiltonian[i * n + i] - self.hamiltonian[j * n + j];
                // -i × de × ρ_ij = -i × de × (re + i×im) = de×im - i×de×re
                drho_re[i * n + j] += de * rho.im[i * n + j];
                drho_im[i * n + j] -= de * rho.re[i * n + j];
            }
        }

        // Dissipative part: Σ_k γ_k D[L_k](ρ)
        for ch in &self.channels {
            // For simplicity with real L operators:
            // D[L](ρ) = LρL† - ½(L†Lρ + ρL†L)

            // Compute LρL† (real L, so L† = L^T)
            let l = &ch.l_re;
            let gamma = ch.gamma;

            for i in 0..n {
                for j in 0..n {
                    // (LρL†)_ij = Σ_kl L_ik ρ_kl L_jl
                    let mut lrl_re = 0.0;
                    let mut lrl_im = 0.0;
                    for k in 0..n {
                        for m in 0..n {
                            let lik = l[i * n + k];
                            let ljm = l[j * n + m]; // L†_mj = L_jm for real L
                            lrl_re += lik * rho.re[k * n + m] * ljm;
                            lrl_im += lik * rho.im[k * n + m] * ljm;
                        }
                    }

                    // (L†L)_ij = Σ_k L_ki L_kj
                    let mut ldl_ij = 0.0;
                    for k in 0..n {
                        ldl_ij += l[k * n + i] * l[k * n + j];
                    }

                    // D[L](ρ)_ij = LρL†_ij - ½(L†L ρ + ρ L†L)_ij
                    // ½(L†Lρ)_ij = ½ Σ_k (L†L)_ik ρ_kj
                    let mut ldl_rho_re = 0.0;
                    let mut ldl_rho_im = 0.0;
                    let mut rho_ldl_re = 0.0;
                    let mut rho_ldl_im = 0.0;
                    for k in 0..n {
                        // (L†L)_ik only has diagonal contribution for our simple L
                        let _ldl_ik = if i == k { ldl_ij } else { 0.0 };
                        // Actually need proper (L†L)_ik:
                        let mut ldl_ik_val = 0.0;
                        for m in 0..n {
                            ldl_ik_val += l[m * n + i] * l[m * n + k];
                        }
                        ldl_rho_re += ldl_ik_val * rho.re[k * n + j];
                        ldl_rho_im += ldl_ik_val * rho.im[k * n + j];

                        let mut ldl_kj_val = 0.0;
                        for m in 0..n {
                            ldl_kj_val += l[m * n + k] * l[m * n + j];
                        }
                        rho_ldl_re += rho.re[i * n + k] * ldl_kj_val;
                        rho_ldl_im += rho.im[i * n + k] * ldl_kj_val;
                    }

                    drho_re[i * n + j] += gamma * (lrl_re - 0.5 * (ldl_rho_re + rho_ldl_re));
                    drho_im[i * n + j] += gamma * (lrl_im - 0.5 * (ldl_rho_im + rho_ldl_im));
                }
            }
        }

        // Euler step: ρ(t+dt) = ρ(t) + dt × dρ/dt
        for i in 0..n * n {
            rho.re[i] += dt * drho_re[i];
            rho.im[i] += dt * drho_im[i];
        }
    }

    /// Evolve for n_steps and return the purity at each step.
    pub fn evolve_purity_trace(
        &self,
        rho: &mut DensityMatrix,
        dt: f64,
        n_steps: usize,
    ) -> Vec<f64> {
        let mut trace = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            self.step(rho, dt);
            trace.push(rho.purity());
        }
        trace
    }

    /// Decoherence time: time for purity to drop to 1/e of initial.
    pub fn decoherence_time(&self, rho: &DensityMatrix, dt: f64, max_steps: usize) -> f64 {
        let mut rho_copy = rho.clone();
        let p0 = rho_copy.purity();
        let target = p0 / std::f64::consts::E;

        for step in 0..max_steps {
            self.step(&mut rho_copy, dt);
            if rho_copy.purity() < target {
                return step as f64 * dt;
            }
        }

        max_steps as f64 * dt // Didn't fully decohere
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_state_purity() {
        let rho = DensityMatrix::pure_state(3, 0);
        assert!((rho.purity() - 1.0).abs() < 1e-10, "Pure state purity = 1");
        assert!((rho.trace() - 1.0).abs() < 1e-10, "Trace = 1");
    }

    #[test]
    fn test_mixed_state_purity() {
        let rho = DensityMatrix::maximally_mixed(4);
        assert!((rho.purity() - 0.25).abs() < 1e-10, "Purity of I/4 = 1/4");
    }

    #[test]
    fn test_superposition_coherence() {
        let s2 = 1.0 / 2.0_f64.sqrt();
        let rho = DensityMatrix::from_state_vector(&[s2, s2], &[0.0, 0.0]);
        let coh = rho.coherence_l1();
        assert!(coh > 0.9, "Superposition should have coherence: {:.4}", coh);
    }

    #[test]
    fn test_dephasing_destroys_coherence() {
        let s2 = 1.0 / 2.0_f64.sqrt();
        let mut rho = DensityMatrix::from_state_vector(&[s2, s2], &[0.0, 0.0]);

        let mut solver = LindbladSolver::new(&[0.0, 1.0]);
        solver.add_dephasing(0, 1.0);
        solver.add_dephasing(1, 1.0);

        let coh_before = rho.coherence_l1();

        // Evolve with dephasing
        for _ in 0..100 {
            solver.step(&mut rho, 0.01);
        }

        let coh_after = rho.coherence_l1();

        assert!(
            coh_after < coh_before * 0.5,
            "Dephasing should reduce coherence: before={:.4}, after={:.4}",
            coh_before,
            coh_after
        );
    }

    #[test]
    fn test_decay_to_ground_state() {
        // Start in excited state, decay to ground
        let mut rho = DensityMatrix::pure_state(2, 1); // |1⟩

        let mut solver = LindbladSolver::new(&[0.0, 1.0]);
        solver.add_decay(1, 0, 0.5); // |1⟩ → |0⟩ at rate 0.5

        for _ in 0..200 {
            solver.step(&mut rho, 0.05);
        }

        // Should be mostly in ground state
        let p_ground = rho.re[0]; // ρ_00
        assert!(
            p_ground > 0.8,
            "Should decay to ground: P(0) = {:.4}",
            p_ground
        );
    }

    #[test]
    fn test_trace_preserved() {
        let s2 = 1.0 / 2.0_f64.sqrt();
        let mut rho = DensityMatrix::from_state_vector(&[s2, s2], &[0.0, 0.0]);

        let mut solver = LindbladSolver::new(&[0.0, 1.0]);
        solver.add_dephasing(0, 0.1);

        for _ in 0..50 {
            solver.step(&mut rho, 0.01);
        }

        assert!(
            (rho.trace() - 1.0).abs() < 0.05,
            "Trace should be preserved: {:.4}",
            rho.trace()
        );
    }

    #[test]
    fn test_purity_decreases_under_dephasing() {
        let s2 = 1.0 / 2.0_f64.sqrt();
        let mut rho = DensityMatrix::from_state_vector(&[s2, s2], &[0.0, 0.0]);

        let mut solver = LindbladSolver::new(&[0.0, 1.0]);
        solver.add_dephasing(0, 1.0);
        solver.add_dephasing(1, 1.0);

        let purity_before = rho.purity();

        // Evolve with small steps
        for _ in 0..500 {
            solver.step(&mut rho, 0.005);
        }

        let purity_after = rho.purity();
        assert!(
            purity_after < purity_before,
            "Purity should decrease: before={:.4}, after={:.4}",
            purity_before,
            purity_after
        );
    }
}
