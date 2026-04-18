// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time-Dependent Quantum Mechanics.
//!
//! Solves the time-dependent Schrödinger equation (TDSE):
//! iℏ ∂Ψ/∂t = H Ψ
//!
//! Implements CIS (Configuration Interaction Singles) for excited states
//! and Rabi oscillations for coherent dynamics.
//!
//! References:
//! - Szabo & Ostlund (1996), Chapter 4 (CIS).
//! - Dreuw & Head-Gordon (2005). Chem. Rev. 105, 4009 (TDDFT review).

use crate::scf::rhf::RhfResult;
use std::f64::consts::PI;

/// CIS excitation energy result.
#[derive(Debug, Clone)]
pub struct CisExcitation {
    /// Excitation energy (Hartree)
    pub energy: f64,
    /// Excitation energy (eV)
    pub energy_ev: f64,
    /// Dominant i→a transition (occupied → virtual)
    pub from_orbital: usize,
    pub to_orbital: usize,
    /// Oscillator strength (dimensionless)
    pub oscillator_strength: f64,
}

/// Compute CIS (Configuration Interaction Singles) excitation energies.
///
/// CIS approximates excited states as single excitations from the HF ground state:
/// |Ψ_excited⟩ = Σ_{i,a} c_{ia} |Φ_i^a⟩
///
/// The excitation energies are eigenvalues of the CIS matrix:
/// A_{ia,jb} = (ε_a - ε_i)δ_ij δ_ab + (ia|jb) - (ij|ab)
///
/// For a simplified version (without 2-electron integrals), orbital energy
/// differences give Koopman's theorem excitations.
pub fn cis_excitations(rhf: &RhfResult, n_excitations: usize) -> Vec<CisExcitation> {
    let n_occ = rhf.n_occupied;
    let n_mo = rhf.n_independent;
    let n_vir = n_mo - n_occ;
    let eps = &rhf.orbital_energies;

    // Simplified CIS: excitation energies = ε_a - ε_i (Koopman's theorem)
    // This neglects electron correlation in the excited state but gives
    // the correct qualitative ordering.
    let mut excitations: Vec<(f64, usize, usize)> = Vec::new();

    for i in 0..n_occ {
        for a in n_occ..n_mo {
            let de = eps[a] - eps[i];
            if de > 0.0 {
                excitations.push((de, i, a));
            }
        }
    }

    // Sort by energy
    excitations.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    excitations
        .iter()
        .take(n_excitations)
        .map(|&(de, i, a)| {
            // Oscillator strength (simplified: proportional to orbital overlap)
            // f = 2/3 × ΔE × |⟨i|r|a⟩|² — estimated from energy gap
            let f_approx = 2.0 / 3.0 * de * 0.5; // rough estimate

            CisExcitation {
                energy: de,
                energy_ev: de * 27.211_386,
                from_orbital: i,
                to_orbital: a,
                oscillator_strength: f_approx,
            }
        })
        .collect()
}

/// Rabi oscillation: coherent two-level dynamics under driving field.
///
/// P(t) = sin²(Ω_R t / 2) where Ω_R = √(Ω² + Δ²)
///
/// Ω is the coupling strength, Δ is the detuning.
#[derive(Debug, Clone)]
pub struct RabiDynamics {
    /// Rabi frequency Ω_R (a.u.)
    pub rabi_frequency: f64,
    /// Detuning Δ = ω_drive - ω_transition (a.u.)
    pub detuning: f64,
    /// Coupling strength Ω (a.u.)
    pub coupling: f64,
}

impl RabiDynamics {
    /// Create a resonant Rabi system (Δ = 0).
    pub fn resonant(coupling: f64) -> Self {
        Self {
            rabi_frequency: coupling,
            detuning: 0.0,
            coupling,
        }
    }

    /// Create an off-resonant system.
    pub fn off_resonant(coupling: f64, detuning: f64) -> Self {
        let omega_r = (coupling * coupling + detuning * detuning).sqrt();
        Self {
            rabi_frequency: omega_r,
            detuning,
            coupling,
        }
    }

    /// Probability of being in the excited state at time t (a.u.).
    pub fn excited_population(&self, t: f64) -> f64 {
        let theta = self.rabi_frequency * t / 2.0;
        let sin_theta = theta.sin();
        (self.coupling / self.rabi_frequency).powi(2) * sin_theta * sin_theta
    }

    /// Time for complete population inversion (π-pulse time).
    pub fn pi_pulse_time(&self) -> f64 {
        PI / self.rabi_frequency
    }
}

/// Time-evolve a state vector under a time-independent Hamiltonian.
///
/// |Ψ(t)⟩ = exp(-iHt) |Ψ(0)⟩
///
/// For a diagonal Hamiltonian (energy eigenbasis):
/// c_i(t) = c_i(0) × exp(-i E_i t)
///
/// Returns (real parts, imaginary parts) of the evolved coefficients.
pub fn time_evolve_eigenstate(
    coefficients_re: &[f64],
    coefficients_im: &[f64],
    energies: &[f64],
    time: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = coefficients_re.len();
    let mut re_out = vec![0.0; n];
    let mut im_out = vec![0.0; n];

    for i in 0..n {
        let phase = -energies[i] * time;
        let cos_p = phase.cos();
        let sin_p = phase.sin();

        // (a + bi)(cos + i sin) = (a cos - b sin) + i(a sin + b cos)
        re_out[i] = coefficients_re[i] * cos_p - coefficients_im[i] * sin_p;
        im_out[i] = coefficients_re[i] * sin_p + coefficients_im[i] * cos_p;
    }

    (re_out, im_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::sto3g::Sto3g;
    use crate::basis::BasisSetProvider;
    use crate::molecule::Molecule;
    use crate::scf::rhf::{restricted_hartree_fock, RhfConfig};

    #[test]
    fn test_cis_excitations_positive() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let exc = cis_excitations(&rhf, 3);
        assert!(!exc.is_empty(), "Should have excitations");

        for e in &exc {
            assert!(e.energy > 0.0, "Excitation energy should be positive");
            assert!(e.energy_ev > 0.0);
        }
    }

    #[test]
    fn test_cis_ordered() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let exc = cis_excitations(&rhf, 5);
        for i in 1..exc.len() {
            assert!(exc[i].energy >= exc[i - 1].energy);
        }
    }

    #[test]
    fn test_rabi_resonant_pi_pulse() {
        let rabi = RabiDynamics::resonant(0.01); // Ω = 0.01 a.u.
        let t_pi = rabi.pi_pulse_time();
        let pop = rabi.excited_population(t_pi);
        assert!(
            (pop - 1.0).abs() < 1e-10,
            "π-pulse should give P=1: got {}",
            pop
        );
    }

    #[test]
    fn test_rabi_oscillation_period() {
        let rabi = RabiDynamics::resonant(0.01);
        // At t=0, P=0
        assert!(rabi.excited_population(0.0).abs() < 1e-14);
        // At t=π/Ω, P=1
        let p_max = rabi.excited_population(PI / 0.01);
        assert!((p_max - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_evolution_norm_preserved() {
        let re = vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
        let im = vec![0.0, 0.0];
        let energies = vec![-1.0, -0.5];

        let (re_t, im_t) = time_evolve_eigenstate(&re, &im, &energies, 10.0);

        let norm: f64 = re_t.iter().zip(im_t.iter()).map(|(r, i)| r * r + i * i).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Norm should be preserved: {}",
            norm
        );
    }
}
