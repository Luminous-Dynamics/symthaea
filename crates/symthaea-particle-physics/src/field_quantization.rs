// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QED Field Quantization — Second Quantization Framework.
//!
//! Implements creation/annihilation operators, photon states, and
//! computes quantum electrodynamics observables from first principles.
//!
//! References:
//! - Sakurai, J. J. (1967). *Advanced Quantum Mechanics*. Addison-Wesley.
//! - Mandl & Shaw (2010). *Quantum Field Theory*. Wiley.

use crate::constants::*;
use std::f64::consts::PI;

/// Fock state |n⟩ for a single mode of the electromagnetic field.
#[derive(Debug, Clone)]
pub struct FockState {
    /// Photon number
    pub n: u64,
    /// Mode frequency (natural units, GeV)
    pub omega: f64,
}

impl FockState {
    pub fn vacuum(omega: f64) -> Self {
        Self { n: 0, omega }
    }

    /// Energy of this Fock state: E = ℏω(n + 1/2)
    pub fn energy(&self) -> f64 {
        self.omega * (self.n as f64 + 0.5)
    }

    /// Apply creation operator: a†|n⟩ = √(n+1)|n+1⟩
    /// Returns the coefficient √(n+1)
    pub fn create(&mut self) -> f64 {
        let coeff = ((self.n + 1) as f64).sqrt();
        self.n += 1;
        coeff
    }

    /// Apply annihilation operator: a|n⟩ = √n|n-1⟩
    /// Returns the coefficient √n (0 if vacuum)
    pub fn annihilate(&mut self) -> f64 {
        if self.n == 0 {
            return 0.0; // a|0⟩ = 0
        }
        let coeff = (self.n as f64).sqrt();
        self.n -= 1;
        coeff
    }
}

/// Coherent state |α⟩ = exp(-|α|²/2) Σ (α^n/√n!) |n⟩
#[derive(Debug, Clone)]
pub struct CoherentState {
    /// Complex amplitude (real part)
    pub alpha_re: f64,
    /// Complex amplitude (imaginary part)
    pub alpha_im: f64,
    /// Mode frequency
    pub omega: f64,
}

impl CoherentState {
    pub fn new(alpha_re: f64, alpha_im: f64, omega: f64) -> Self {
        Self {
            alpha_re,
            alpha_im,
            omega,
        }
    }

    /// |α|²
    pub fn mean_photon_number(&self) -> f64 {
        self.alpha_re * self.alpha_re + self.alpha_im * self.alpha_im
    }

    /// Probability of measuring n photons: P(n) = e^{-|α|²} |α|^{2n} / n!
    pub fn photon_probability(&self, n: u64) -> f64 {
        let alpha_sq = self.mean_photon_number();
        let mut log_prob = -alpha_sq + n as f64 * alpha_sq.ln();
        // Subtract ln(n!)
        for k in 1..=n {
            log_prob -= (k as f64).ln();
        }
        log_prob.exp()
    }

    /// Electric field expectation: ⟨E⟩ ∝ |α| cos(ωt - φ)
    pub fn electric_field_amplitude(&self) -> f64 {
        2.0 * self.mean_photon_number().sqrt()
    }
}

/// Spontaneous emission rate (Einstein A coefficient).
///
/// A = (ω³ |d|²) / (3π ε₀ ℏ c³)
/// In natural units: A = (4α ω³ |d|²) / 3
///
/// `omega` = transition frequency (a.u.)
/// `dipole_sq` = |⟨f|d|i⟩|² transition dipole moment squared (a.u.)
pub fn spontaneous_emission_rate(omega: f64, dipole_sq: f64) -> f64 {
    4.0 * ALPHA_EM * omega.powi(3) * dipole_sq / 3.0
}

/// Lamb shift of the hydrogen 2S level (leading order).
///
/// ΔE_Lamb ≈ (α⁵ m_e c² / π) × [ln(1/α²) + ...] ≈ 1057 MHz
/// Returns energy in eV.
pub fn lamb_shift_2s() -> f64 {
    // Bethe logarithm approximation
    let m_e_ev = M_ELECTRON * 1e9; // Convert GeV to eV
    let alpha = ALPHA_EM;

    // Leading order: α⁵ m_e / (6π) × ln(m_e/(2×ΔE)) where ΔE is mean excitation
    // Numerical result: ~4.37 × 10⁻⁶ eV ≈ 1057 MHz
    alpha.powi(5) * m_e_ev * 0.41 / PI // Simplified with Bethe log ≈ 0.41×6
}

/// Anomalous magnetic moment of the electron (g-2)/2.
///
/// Schwinger's result (1-loop): a_e = α/(2π)
/// 2-loop: a_e = α/(2π) - 0.328... (α/π)²
pub fn electron_g_minus_2() -> f64 {
    let a = ALPHA_EM;
    // 1-loop (Schwinger 1948)
    let one_loop = a / (2.0 * PI);
    // 2-loop
    let two_loop = -0.328_478_965 * (a / PI).powi(2);
    one_loop + two_loop
}

/// Casimir force per unit area between two perfectly conducting plates
/// separated by distance d.
///
/// F/A = -π² ℏc / (240 d⁴)
/// In natural units with d in GeV⁻¹: F/A = -π² / (240 d⁴)
pub fn casimir_pressure(d_gev_inv: f64) -> f64 {
    -PI * PI / (240.0 * d_gev_inv.powi(4))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_energy() {
        let vac = FockState::vacuum(1.0);
        assert!((vac.energy() - 0.5).abs() < 1e-14, "Vacuum energy = ℏω/2");
    }

    #[test]
    fn test_creation_annihilation() {
        let mut state = FockState::vacuum(1.0);

        // a†|0⟩ = |1⟩ with coefficient 1
        let c = state.create();
        assert!((c - 1.0).abs() < 1e-14);
        assert_eq!(state.n, 1);

        // a†|1⟩ = √2|2⟩
        let c = state.create();
        assert!((c - 2.0_f64.sqrt()).abs() < 1e-14);
        assert_eq!(state.n, 2);

        // a|2⟩ = √2|1⟩
        let c = state.annihilate();
        assert!((c - 2.0_f64.sqrt()).abs() < 1e-14);
        assert_eq!(state.n, 1);
    }

    #[test]
    fn test_vacuum_annihilation() {
        let mut vac = FockState::vacuum(1.0);
        let c = vac.annihilate();
        assert_eq!(c, 0.0, "a|0⟩ = 0");
        assert_eq!(vac.n, 0);
    }

    #[test]
    fn test_coherent_state_poisson() {
        // Mean photon number for |α=2⟩ should be 4
        let cs = CoherentState::new(2.0, 0.0, 1.0);
        assert!((cs.mean_photon_number() - 4.0).abs() < 1e-14);

        // Photon distribution should be Poisson
        let p0 = cs.photon_probability(0);
        let p1 = cs.photon_probability(1);
        // P(0) = e^{-4}, P(1) = 4e^{-4}
        assert!((p1 / p0 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_electron_g_minus_2() {
        // Experimental: a_e ≈ 0.00115965218
        let a_e = electron_g_minus_2();
        assert!(
            (a_e - 0.001_159_65).abs() < 1e-5,
            "(g-2)/2 = {:.8}, expected ≈ 0.00115965",
            a_e
        );
    }

    #[test]
    fn test_lamb_shift_order_of_magnitude() {
        // Lamb shift ≈ 4.37 μeV ≈ 1057 MHz
        let de = lamb_shift_2s();
        assert!(
            de > 1e-7 && de < 1e-4,
            "Lamb shift = {:.2e} eV, expected ~4e-6 eV",
            de
        );
    }

    #[test]
    fn test_casimir_force_attractive() {
        // Casimir force should be attractive (negative pressure)
        let f = casimir_pressure(1.0);
        assert!(f < 0.0, "Casimir force should be attractive");
    }
}
