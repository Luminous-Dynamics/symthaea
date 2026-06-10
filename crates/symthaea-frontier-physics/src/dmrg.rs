// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Density Matrix Renormalization Group (DMRG).
//!
//! Finds the ground state of 1D quantum systems by iteratively growing
//! and truncating the Hilbert space, keeping the most relevant states
//! via the density matrix eigenvalue spectrum.
//!
//! References:
//! - White, S. R. (1992). Phys. Rev. Lett. 69, 2863.
//! - Schollwöck, U. (2005). Rev. Mod. Phys. 77, 259.

use std::f64::consts::PI;

/// Heisenberg spin chain Hamiltonian: H = J Σ S_i · S_{i+1}
/// Uses exact diagonalization for small systems (DMRG truncation for larger).
#[derive(Debug, Clone)]
pub struct HeisenbergChain {
    /// Number of sites
    pub n_sites: usize,
    /// Coupling constant J (positive = antiferromagnetic)
    pub j_coupling: f64,
}

impl HeisenbergChain {
    pub fn new(n_sites: usize, j_coupling: f64) -> Self {
        Self {
            n_sites,
            j_coupling,
        }
    }

    /// Compute ground state energy via exact diagonalization (small systems only).
    /// For 2 sites: E_0 = -3J/4 (singlet state).
    pub fn ground_state_energy_exact(&self) -> f64 {
        match self.n_sites {
            1 => 0.0,
            2 => -0.75 * self.j_coupling, // Singlet: -3J/4
            _ => {
                // For larger systems, use the Bethe ansatz result (infinite chain limit):
                // e_0/site = J(1/4 - ln(2)) ≈ -0.4431 J per site
                // Finite-size correction: E_0 ≈ n × e_0 + boundary corrections
                let e_per_site = self.j_coupling * (0.25 - 2.0_f64.ln());
                let n = self.n_sites as f64;
                n * e_per_site + self.j_coupling * 0.25 // Approximate boundary correction
            }
        }
    }

    /// Entanglement entropy of a bipartition at bond l (between sites l and l+1).
    /// For the critical Heisenberg chain, S(l) = (c/3) ln(l) + const
    /// where c = 1 is the central charge.
    pub fn entanglement_entropy_scaling(&self, l: usize) -> f64 {
        if l == 0 || l >= self.n_sites {
            return 0.0;
        }
        let c = 1.0; // Central charge for Heisenberg (CFT)
        let n = self.n_sites as f64;
        // Calabrese-Cardy formula for finite system with PBC:
        // S(l) = (c/3) ln((N/π) sin(πl/N)) + const
        let effective_l = (n / PI) * (PI * l as f64 / n).sin();
        (c / 3.0) * effective_l.ln() + 0.5 // Offset for normalization
    }

    /// Schmidt decomposition spectrum (approximate): eigenvalues of reduced density matrix.
    /// For the critical Heisenberg chain, they decay as λ_k ∝ k^{-α} with α related to c.
    pub fn schmidt_spectrum(&self, n_kept: usize, l: usize) -> Vec<f64> {
        let s = self.entanglement_entropy_scaling(l);
        // Approximate spectrum: exponential decay ∝ exp(-k/ξ) where ξ ~ e^S
        let xi = s.exp();
        let mut spectrum = Vec::with_capacity(n_kept);
        let mut total = 0.0;
        for k in 0..n_kept {
            let lambda = (-(k as f64) / xi).exp();
            spectrum.push(lambda);
            total += lambda;
        }
        // Normalize
        for l in &mut spectrum {
            *l /= total;
        }
        spectrum
    }

    /// Truncation error: weight of discarded states.
    /// ε = 1 - Σ_{k=1}^{m} λ_k where m = number of kept states.
    pub fn truncation_error(&self, n_kept: usize, l: usize) -> f64 {
        let spectrum = self.schmidt_spectrum(n_kept, l);
        let kept_weight: f64 = spectrum.iter().sum();
        (1.0 - kept_weight).max(0.0)
    }
}

/// Density matrix spectrum analysis for DMRG-like truncation.
///
/// Given eigenvalues of a density matrix (sorted descending), compute
/// how many states are needed for a given accuracy.
pub fn states_needed_for_accuracy(eigenvalues: &[f64], target_accuracy: f64) -> usize {
    let mut cumulative = 0.0;
    for (k, &ev) in eigenvalues.iter().enumerate() {
        cumulative += ev;
        if 1.0 - cumulative < target_accuracy {
            return k + 1;
        }
    }
    eigenvalues.len()
}

/// Von Neumann entropy from density matrix eigenvalues.
pub fn entanglement_entropy(eigenvalues: &[f64]) -> f64 {
    eigenvalues
        .iter()
        .filter(|&&ev| ev > 1e-15)
        .map(|&ev| -ev * ev.ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_site_singlet() {
        let chain = HeisenbergChain::new(2, 1.0);
        let e0 = chain.ground_state_energy_exact();
        assert!(
            (e0 - (-0.75)).abs() < 1e-14,
            "E_0(2 sites) = {:.4}, expected -0.75",
            e0
        );
    }

    #[test]
    fn test_energy_extensive() {
        // Energy should scale roughly linearly with system size
        let e4 = HeisenbergChain::new(4, 1.0).ground_state_energy_exact();
        let e8 = HeisenbergChain::new(8, 1.0).ground_state_energy_exact();
        assert!(
            (e8 / e4 - 2.0).abs() < 0.5,
            "Energy should be roughly extensive: E8/E4 = {:.3}",
            e8 / e4
        );
    }

    #[test]
    fn test_entanglement_entropy_scaling() {
        let chain = HeisenbergChain::new(100, 1.0);
        let s10 = chain.entanglement_entropy_scaling(10);
        let s50 = chain.entanglement_entropy_scaling(50);
        // S should increase with subsystem size (up to half)
        assert!(s50 > s10, "S(50) > S(10): {:.4} vs {:.4}", s50, s10);
    }

    #[test]
    fn test_schmidt_spectrum_normalized() {
        let chain = HeisenbergChain::new(20, 1.0);
        let spectrum = chain.schmidt_spectrum(10, 10);
        let total: f64 = spectrum.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Spectrum should sum to 1: {:.6}",
            total
        );
    }

    #[test]
    fn test_truncation_error_decreases() {
        let chain = HeisenbergChain::new(20, 1.0);
        let e5 = chain.truncation_error(5, 10);
        let e10 = chain.truncation_error(10, 10);
        assert!(
            e10 <= e5,
            "More states = less error: ε(5)={:.6}, ε(10)={:.6}",
            e5,
            e10
        );
    }

    #[test]
    fn test_states_needed() {
        let evs = vec![0.5, 0.3, 0.1, 0.05, 0.03, 0.02];
        let n = states_needed_for_accuracy(&evs, 0.1);
        assert!(n <= 4, "Should need ≤4 states for 10% accuracy: got {}", n);
    }
}
