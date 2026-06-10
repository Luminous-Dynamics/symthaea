// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lattice QCD — Non-perturbative strong interaction physics.
//!
//! Implements Wilson loops, string tension, and hadron mass estimates
//! from the lattice gauge theory framework.
//!
//! References:
//! - Wilson, K. G. (1974). Phys. Rev. D 10, 2445.
//! - Creutz, M. (1983). *Quarks, Gluons and Lattices*. Cambridge UP.
//! - Rothe, H. J. (2012). *Lattice Gauge Theories*. World Scientific.

use crate::constants::*;

/// Strong coupling string tension σ in GeV² (from lattice calculations).
/// σ ≈ (440 MeV)² ≈ 0.194 GeV²
pub const STRING_TENSION: f64 = 0.44 * 0.44;

/// Wilson loop expectation value for an R×T rectangle.
///
/// At strong coupling: ⟨W(R,T)⟩ ~ exp(-σ R T) (area law → confinement)
/// At weak coupling: ⟨W(R,T)⟩ ~ exp(-V(R) T) (perimeter law → deconfinement)
///
/// The area law signals quark confinement.
pub fn wilson_loop_area_law(r: f64, t: f64, sigma: f64) -> f64 {
    (-sigma * r * t).exp()
}

/// Static quark-antiquark potential from lattice QCD.
///
/// V(r) = -4α_s/(3r) + σr + C (Cornell potential)
///
/// Short distance: Coulomb-like (one-gluon exchange)
/// Long distance: linear (string tension / confinement)
pub fn cornell_potential(r_fm: f64) -> f64 {
    // Convert to GeV⁻¹: 1 fm = 5.068 GeV⁻¹
    let r = r_fm * 5.068;
    let alpha_s = 0.3; // effective coupling at ~1 GeV scale
    let sigma = STRING_TENSION;
    let c = -0.25; // constant offset (GeV)

    -4.0 * alpha_s / (3.0 * r) + sigma * r + c
}

/// Hadron mass from constituent quark model (simplified).
///
/// M_hadron ≈ Σ m_constituent + binding
/// Constituent quark masses are ~1/3 of proton mass for u,d.
pub fn hadron_mass_estimate(constituent_masses: &[f64], binding_gev: f64) -> f64 {
    constituent_masses.iter().sum::<f64>() + binding_gev
}

/// Proton mass estimate from constituent quarks.
/// m_p ≈ 3 × m_constituent(u,d) where m_const ≈ 0.34 GeV
pub fn proton_mass_constituent() -> f64 {
    3.0 * 0.336 // Three constituent quarks at ~336 MeV each
}

/// Pion mass from PCAC (Partially Conserved Axial Current).
/// m_π² ≈ (m_u + m_d) × |⟨q̄q⟩| / f_π²
///
/// The pion is anomalously light because it's the pseudo-Goldstone
/// boson of chiral symmetry breaking.
pub fn pion_mass_pcac() -> f64 {
    let m_q = (M_UP + M_DOWN) / 2.0; // Average light quark mass
    let condensate = (0.25_f64).powi(3); // ⟨q̄q⟩ ≈ -(250 MeV)³ in GeV³
    let f_pi = 0.092; // Pion decay constant in GeV

    // m_π² = 2 m_q |⟨q̄q⟩| / f_π²
    let m_pi_sq = 2.0 * m_q * condensate / (f_pi * f_pi);
    m_pi_sq.abs().sqrt()
}

/// Deconfinement temperature from lattice QCD.
/// T_c ≈ 155 MeV (crossover for physical quark masses)
pub const DECONFINEMENT_TEMPERATURE: f64 = 0.155; // GeV

/// QCD vacuum condensate: ⟨ḡG²⟩ ≈ 0.012 GeV⁴ (gluon condensate)
pub const GLUON_CONDENSATE: f64 = 0.012;

/// Topological susceptibility χ_top = ⟨Q²⟩/V ≈ (180 MeV)⁴
pub const TOPOLOGICAL_SUSCEPTIBILITY: f64 = 0.180_f64 * 0.180 * 0.180 * 0.180;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wilson_loop_area_law() {
        // Area law: W decreases exponentially with area
        let w_small = wilson_loop_area_law(1.0, 1.0, STRING_TENSION);
        let w_large = wilson_loop_area_law(2.0, 2.0, STRING_TENSION);
        assert!(w_large < w_small, "Area law: larger loop = smaller W");
        assert!(w_small > 0.0 && w_small < 1.0);
    }

    #[test]
    fn test_cornell_potential_confining() {
        // At large r, potential should increase (confinement)
        let v1 = cornell_potential(0.5); // 0.5 fm
        let v2 = cornell_potential(1.5); // 1.5 fm
        assert!(
            v2 > v1,
            "Confinement: V(1.5fm)={:.3} should > V(0.5fm)={:.3}",
            v2,
            v1
        );
    }

    #[test]
    fn test_cornell_potential_coulomb_short() {
        // At very short r, should be negative (attractive Coulomb)
        let v_short = cornell_potential(0.1);
        assert!(
            v_short < 0.0,
            "Short-range: V(0.1fm) = {:.3} should be negative",
            v_short
        );
    }

    #[test]
    fn test_proton_mass() {
        // Constituent quark model: m_p ≈ 0.938 GeV
        let m_p = proton_mass_constituent();
        assert!(
            (m_p - M_PROTON).abs() < 0.1,
            "Proton mass = {:.3} GeV, expected {:.3}",
            m_p,
            M_PROTON
        );
    }

    #[test]
    fn test_pion_pseudo_goldstone() {
        // Pion should be much lighter than proton (pseudo-Goldstone)
        let m_pi = pion_mass_pcac();
        assert!(
            m_pi < 0.5 * M_PROTON,
            "Pion ({:.3} GeV) should be << proton ({:.3} GeV)",
            m_pi,
            M_PROTON
        );
    }
}
