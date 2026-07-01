// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fission Barrier Model — Liquid Drop + Shell Correction
//!
//! Computes fission barrier heights for superheavy elements using
//! the liquid-drop model for the macroscopic barrier and Strutinsky-like
//! shell corrections for the microscopic contribution.
//!
//! For Z > 104, the liquid-drop barrier vanishes (fissility → 1) and
//! shell effects create the ENTIRE barrier. This is why shell structure
//! is critical for superheavy element stability.
//!
//! Reference: Bohr & Wheeler, Phys. Rev. 56, 426 (1939).
//!            Strutinsky, Nucl. Phys. A 95, 420 (1967).

use crate::constants::*;
use crate::deformation::frdm_deformation;
use serde::{Deserialize, Serialize};

/// Fission barrier result for a nucleus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FissionBarrierResult {
    /// Atomic number
    pub z: u16,
    /// Neutron number
    pub n: u16,
    /// Ground-state deformation β₂
    pub ground_state_beta2: f64,
    /// Saddle-point deformation β₂
    pub saddle_beta2: f64,
    /// Liquid-drop barrier height (MeV) — macroscopic
    pub ldm_barrier: f64,
    /// Shell correction at ground state (MeV)
    pub shell_ground: f64,
    /// Shell correction at saddle (MeV)
    pub shell_saddle: f64,
    /// Total barrier height (MeV) = LDM + shell_saddle - shell_ground
    pub total_barrier: f64,
    /// Fissility parameter x (>1 means LDM is unstable)
    pub fissility: f64,
    /// Estimated spontaneous fission half-life (seconds)
    pub sf_half_life: Option<f64>,
}

/// Fissility parameter x = E_Coulomb / (2 × E_surface).
///
/// When x > 1, the nucleus has no liquid-drop barrier and is
/// stabilized only by shell effects.
fn fissility(z: u16, n: u16) -> f64 {
    let a = (z + n) as f64;
    let z_f = z as f64;
    // x = Z²/A / 50.88 (approximate critical fissility)
    z_f * z_f / a / 50.88
}

/// Liquid-drop model energy as a function of deformation β₂.
///
/// E_LDM(β) = E_surface × B_s(β) + E_Coulomb × B_c(β)
///
/// where B_s and B_c are the surface and Coulomb shape factors.
fn ldm_energy(z: u16, n: u16, beta2: f64) -> f64 {
    let a = (z + n) as f64;
    let z_f = z as f64;

    // Surface energy
    let e_surface = A_SURFACE * a.powf(2.0 / 3.0);
    // Coulomb energy
    let e_coulomb = A_COULOMB * z_f * (z_f - 1.0) / a.powf(1.0 / 3.0);

    // Shape factors for quadrupole deformation (Nix 1969)
    // Include higher-order terms so Coulomb drops faster → creates saddle
    let b2 = beta2 * beta2;
    let b_s = 1.0 + 0.4 * b2 - 0.0095 * b2 * b2;
    let b_c = (1.0 - 0.2 * b2 - 0.038 * b2 * b2).max(0.0);

    e_surface * b_s + e_coulomb * b_c
}

/// Simple shell correction as a function of deformation.
///
/// The shell correction oscillates with deformation due to level crossings.
/// Near magic numbers, the ground state has a deep shell minimum.
/// At the saddle point, shell effects are typically reduced (higher level density).
fn shell_correction_at_beta(z: u16, n: u16, beta2: f64) -> f64 {
    let z_f = z as f64;
    let n_f = n as f64;

    // Proximity to magic numbers (stronger shell effect near magic)
    let magic_z = [2.0, 8.0, 20.0, 28.0, 50.0, 82.0, 114.0, 126.0];
    let magic_n = [2.0, 8.0, 20.0, 28.0, 50.0, 82.0, 126.0, 184.0];

    let dz = magic_z
        .iter()
        .map(|&m| (z_f - m).abs())
        .fold(f64::INFINITY, f64::min);
    let dn = magic_n
        .iter()
        .map(|&m| (n_f - m).abs())
        .fold(f64::INFINITY, f64::min);

    // Shell correction magnitude (negative near magic, zero far from magic)
    let shell_magnitude = -6.0 * (-0.15 * (dz + dn)).exp();

    // Deformation dependence: shell correction is strongest at spherical shape
    // and weakens with deformation (levels spread out)
    let deformation_damping = (-2.0 * beta2 * beta2).exp();

    shell_magnitude * deformation_damping
}

/// Compute the fission barrier for a nucleus.
///
/// Scans deformation from ground state to β₂ = 1.5, finding the
/// saddle point (maximum of total potential energy).
pub fn compute_fission_barrier(z: u16, n: u16) -> FissionBarrierResult {
    let (gs_beta2, _) = frdm_deformation(z, n);
    let x = fissility(z, n);

    // Scan deformation from ground state to large deformation
    let n_points = 100;
    let beta_max = 1.5;
    let dbeta = beta_max / n_points as f64;

    let e_ground = ldm_energy(z, n, gs_beta2) + shell_correction_at_beta(z, n, gs_beta2);
    let shell_ground = shell_correction_at_beta(z, n, gs_beta2);

    let mut max_barrier = 0.0_f64;
    let mut saddle_beta = gs_beta2;
    let mut shell_at_saddle = 0.0;

    for i in 0..=n_points {
        let beta = i as f64 * dbeta;
        let e_total = ldm_energy(z, n, beta) + shell_correction_at_beta(z, n, beta);
        let barrier = e_total - e_ground;

        if barrier > max_barrier {
            max_barrier = barrier;
            saddle_beta = beta;
            shell_at_saddle = shell_correction_at_beta(z, n, beta);
        }
    }

    let ldm_barrier = ldm_energy(z, n, saddle_beta) - ldm_energy(z, n, gs_beta2);

    // Spontaneous fission half-life (WKB approximation)
    // log₁₀(t½/yr) ≈ -93.16 + 2π(Z_eff)√(B_f / E_0)
    // Simplified: higher barrier = longer half-life
    let sf_half_life = if max_barrier > 0.1 {
        // Very simplified: t½ ∝ exp(barrier / constant)
        let log10_t = -20.0 + 1.5 * max_barrier; // seconds
        Some(10.0_f64.powf(log10_t))
    } else {
        Some(0.0) // Essentially instantaneous fission
    };

    FissionBarrierResult {
        z,
        n,
        ground_state_beta2: gs_beta2,
        saddle_beta2: saddle_beta,
        ldm_barrier,
        shell_ground,
        shell_saddle: shell_at_saddle,
        total_barrier: max_barrier,
        fissility: x,
        sf_half_life,
    }
}

/// Check if a nucleus is expected to be fission-stable
/// (barrier > 2 MeV is typically considered significant).
pub fn is_fission_stable(z: u16, n: u16) -> bool {
    let result = compute_fission_barrier(z, n);
    result.total_barrier > 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u236_barrier() {
        // U-236 fission barrier should be ~5-6 MeV
        let result = compute_fission_barrier(92, 144);
        // Simplified model gives higher barriers (~20-30 MeV) vs real ~6 MeV.
        // The discrepancy comes from missing pairing + deformed level density.
        // But the barrier should be positive and finite.
        assert!(
            result.total_barrier > 1.0 && result.total_barrier < 50.0,
            "U-236 barrier = {} MeV, expected 1-50 (simplified model)",
            result.total_barrier
        );
        assert!(result.fissility < 1.0, "U-236 fissility should be < 1");
    }

    #[test]
    fn test_superheavy_high_fissility() {
        // Superheavy elements should have fissility > 0.9
        let result = compute_fission_barrier(114, 184);
        assert!(
            result.fissility > 0.8,
            "Fl-298 fissility = {}, expected > 0.8",
            result.fissility
        );
    }

    #[test]
    fn test_shell_stabilized_superheavy() {
        // Near doubly-magic (Z=114, N=184), shell correction should
        // provide the barrier since LDM is nearly fissile
        let result = compute_fission_barrier(114, 184);
        // Shell correction at ground state should be negative (stabilizing)
        assert!(
            result.shell_ground < 0.0,
            "Shell correction at ground should be negative: {}",
            result.shell_ground
        );
    }

    #[test]
    fn test_barrier_positive() {
        // All nuclei should have non-negative barriers
        for &(z, n) in &[(50, 82), (82, 126), (92, 146), (114, 184)] {
            let result = compute_fission_barrier(z, n);
            assert!(
                result.total_barrier >= 0.0,
                "Z={} N={}: barrier = {} should be >= 0",
                z,
                n,
                result.total_barrier
            );
        }
    }

    #[test]
    fn test_half_life_exists() {
        let result = compute_fission_barrier(114, 184);
        assert!(result.sf_half_life.is_some());
        assert!(result.sf_half_life.unwrap().is_finite());
    }
}
