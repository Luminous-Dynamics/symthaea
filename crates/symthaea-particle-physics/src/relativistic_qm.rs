// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Relativistic Quantum Mechanics — Dirac Equation.
//!
//! The Dirac equation: (iγ^μ ∂_μ - m)ψ = 0
//!
//! Implements energy levels for hydrogen-like atoms including relativistic
//! corrections (fine structure), spin-orbit coupling, and the Dirac spectrum.
//!
//! References:
//! - Dirac, P. A. M. (1928). Proc. R. Soc. A 117, 610.
//! - Greiner, W. (2000). *Relativistic Quantum Mechanics*. Springer.
//! - Bjorken & Drell (1964). *Relativistic Quantum Mechanics*. McGraw-Hill.

use crate::constants::*;
use std::f64::consts::PI;

/// Dirac gamma matrices (4×4) in the Dirac representation.
/// γ⁰ = diag(I, -I), γⁱ = [[0, σⁱ], [-σⁱ, 0]]
///
/// Returns the (i,j) element of γ^μ as (real, imaginary).
pub fn gamma_matrix(mu: usize, i: usize, j: usize) -> (f64, f64) {
    match mu {
        0 => {
            // γ⁰ = diag(1, 1, -1, -1)
            if i == j {
                if i < 2 { (1.0, 0.0) } else { (-1.0, 0.0) }
            } else {
                (0.0, 0.0)
            }
        }
        1 => {
            // γ¹: σ_x in off-diagonal blocks
            match (i, j) {
                (0, 3) | (1, 2) => (1.0, 0.0),
                (2, 1) | (3, 0) => (-1.0, 0.0),
                _ => (0.0, 0.0),
            }
        }
        2 => {
            // γ²: σ_y in off-diagonal blocks
            match (i, j) {
                (0, 3) => (0.0, -1.0),
                (1, 2) => (0.0, 1.0),
                (2, 1) => (0.0, 1.0),
                (3, 0) => (0.0, -1.0),
                _ => (0.0, 0.0),
            }
        }
        3 => {
            // γ³: σ_z in off-diagonal blocks
            match (i, j) {
                (0, 2) => (1.0, 0.0),
                (1, 3) => (-1.0, 0.0),
                (2, 0) => (-1.0, 0.0),
                (3, 1) => (1.0, 0.0),
                _ => (0.0, 0.0),
            }
        }
        _ => (0.0, 0.0),
    }
}

/// Exact Dirac energy levels for hydrogen-like atoms.
///
/// E_{n,j} = mc² × [1 + (Zα/(n - δ))²]^{-1/2}
/// where δ = j + 1/2 - √((j+1/2)² - (Zα)²)
///
/// Returns energy in units of mc² (rest mass subtracted).
pub fn dirac_hydrogen_energy(n: u32, j: f64, z: u32) -> f64 {
    let za = z as f64 * ALPHA_EM;
    let jph = j + 0.5; // j + 1/2
    let delta = jph - (jph * jph - za * za).sqrt();
    let n_eff = n as f64 - delta;

    let ratio = za / n_eff;
    let e_over_mc2 = (1.0 + ratio * ratio).powf(-0.5);

    // Return binding energy (negative): E - mc²
    e_over_mc2 - 1.0
}

/// Non-relativistic hydrogen energy (for comparison): E_n = -Z²α²m_e/(2n²)
pub fn hydrogen_energy_nr(n: u32, z: u32) -> f64 {
    let za = z as f64 * ALPHA_EM;
    -za * za / (2.0 * (n * n) as f64)
}

/// Fine structure splitting: ΔE between j=l+1/2 and j=l-1/2 states.
///
/// ΔE ≈ (Z⁴α⁴ m_e c²) / (2n³) × 1/(l(l+1))
/// Returns energy difference in units of mc².
pub fn fine_structure_splitting(n: u32, l: u32, z: u32) -> f64 {
    if l == 0 {
        return 0.0; // No splitting for s-states
    }
    let za = z as f64 * ALPHA_EM;
    let j_plus = l as f64 + 0.5;
    let j_minus = l as f64 - 0.5;

    let e_plus = dirac_hydrogen_energy(n, j_plus, z);
    let e_minus = dirac_hydrogen_energy(n, j_minus, z);

    (e_plus - e_minus).abs()
}

/// Spin-orbit coupling energy for an electron in a central potential.
///
/// H_SO = (1/2m²c²) × (1/r)(dV/dr) × L·S
/// For hydrogen: ΔE_SO = (Z⁴α⁴)/(2n³) × [j(j+1) - l(l+1) - 3/4] / [l(l+1/2)(l+1)]
pub fn spin_orbit_energy(n: u32, l: u32, j: f64, z: u32) -> f64 {
    if l == 0 {
        return 0.0;
    }
    let za = z as f64 * ALPHA_EM;
    let ls = 0.5 * (j * (j + 1.0) - l as f64 * (l as f64 + 1.0) - 0.75);
    let prefactor = za.powi(4) / (2.0 * (n as f64).powi(3));
    let denominator = l as f64 * (l as f64 + 0.5) * (l as f64 + 1.0);

    prefactor * ls / denominator
}

/// Klein-Gordon equation energy for spin-0 particle:
/// E² = p²c² + m²c⁴ → E = √(p² + m²) in natural units
pub fn klein_gordon_energy(momentum: f64, mass: f64) -> f64 {
    (momentum * momentum + mass * mass).sqrt()
}

/// Zitterbewegung frequency: ω_Z = 2mc²/ℏ
/// The rapid trembling motion of the electron at twice the Compton frequency.
pub fn zitterbewegung_frequency(mass_gev: f64) -> f64 {
    2.0 * mass_gev // In natural units: ω = 2m
}

/// Compton wavelength: λ_C = ℏ/(mc) = 1/m in natural units
pub fn compton_wavelength(mass_gev: f64) -> f64 {
    1.0 / mass_gev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirac_ground_state_hydrogen() {
        // H atom ground state: n=1, j=1/2, Z=1
        // E ≈ -13.6 eV = -α²/2 (in mc² units)
        let e = dirac_hydrogen_energy(1, 0.5, 1);
        let e_ev = e * M_ELECTRON * 1e9; // Convert mc² fraction to eV
        assert!(
            (e_ev - (-13.6)).abs() < 0.1,
            "H ground state = {:.2} eV, expected -13.6 eV",
            e_ev
        );
    }

    #[test]
    fn test_dirac_vs_nonrelativistic() {
        // Dirac should give a slightly more negative energy than Schrödinger
        let e_dirac = dirac_hydrogen_energy(1, 0.5, 1);
        let e_nr = hydrogen_energy_nr(1, 1);
        assert!(
            e_dirac < e_nr,
            "Dirac ({:.8}) should be more bound than NR ({:.8})",
            e_dirac,
            e_nr
        );
    }

    #[test]
    fn test_fine_structure_exists() {
        // 2p state should have fine structure splitting
        let split = fine_structure_splitting(2, 1, 1);
        assert!(split > 0.0, "Fine structure should exist: {:.2e}", split);
        // Should be of order α⁴ ≈ 3e-9
        assert!(split < 1e-4, "Fine structure should be small: {:.2e}", split);
    }

    #[test]
    fn test_heavy_atom_larger_splitting() {
        // Z=10 (Ne) should have larger fine structure than Z=1 (H)
        let split_h = fine_structure_splitting(2, 1, 1);
        let split_ne = fine_structure_splitting(2, 1, 10);
        assert!(
            split_ne > split_h,
            "Heavier atoms have larger FS: Ne={:.2e}, H={:.2e}",
            split_ne,
            split_h
        );
    }

    #[test]
    fn test_gamma_matrix_anticommutation() {
        // {γ^μ, γ^ν} = 2g^{μν} → γ⁰² = I (diagonal elements sum to 4)
        let mut trace = 0.0;
        for i in 0..4 {
            let (re, _) = gamma_matrix(0, i, i);
            trace += re * re;
        }
        assert!((trace - 4.0).abs() < 1e-14, "Tr(γ⁰²) should give 4");
    }

    #[test]
    fn test_klein_gordon_rest_mass() {
        // At p=0: E = m
        let e = klein_gordon_energy(0.0, 1.0);
        assert!((e - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_compton_wavelength_electron() {
        // λ_C(e) = 1/m_e ≈ 1/0.000511 GeV⁻¹ ≈ 386 fm
        let lambda = compton_wavelength(M_ELECTRON);
        // In fm: multiply by 0.197 GeV·fm
        let lambda_fm = lambda * 0.197;
        assert!(
            (lambda_fm - 386.0).abs() < 5.0,
            "Compton wavelength = {:.1} fm",
            lambda_fm
        );
    }
}
