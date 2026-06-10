// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hartree-Fock-Bogoliubov Solver — Architecture Stub
//!
//! Full self-consistent HFB is the gold standard for nuclear structure.
//! This module defines the trait and data structures for future implementation.
//!
//! ## Implementation Options:
//!
//! 1. **HFBTHO FFI** (~500 LOC, 1 week): Call the Fortran HFBTHO code via FFI.
//!    Advantages: battle-tested physics, published results.
//!    Disadvantages: Fortran dependency, harder to deploy.
//!
//! 2. **Native Rust HFB** (~6,000 LOC, 3 months): Full implementation using
//!    nalgebra/faer for eigensolvers. Would be the first Rust HFB code.
//!    Advantages: pure Rust, full control, WASM-compatible.
//!    Disadvantages: significant validation effort.
//!
//! ## Key Components:
//!
//! - Skyrme energy density functional (SLy4, SkM*, UNEDF1)
//! - Transformed Harmonic Oscillator basis (~20 shells, ~400 states)
//! - HFB matrix construction from density-dependent functional
//! - Eigenvalue problem: H|φ⟩ = ε|φ⟩ (400×400 dense)
//! - Self-consistency iteration (50-200 cycles with Broyden mixing)
//! - Pairing via BCS or full Bogoliubov
//!
//! ## References:
//! - Stoitsov et al., Comp. Phys. Comm. 167 (2005) 43 — HFBTHO
//! - Dobaczewski et al., Comp. Phys. Comm. 131 (2000) 164 — HFODD
//! - Bender, Heenen, Reinhard, Rev. Mod. Phys. 75 (2003) 121 — Review

use serde::{Deserialize, Serialize};

/// Skyrme functional parameter sets.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SkyrmeParametrization {
    /// SLy4: Chabanat et al. (1998) — good for neutron-rich nuclei
    SLy4,
    /// SkM*: Bartel et al. (1982) — good for fission barriers
    SkMStar,
    /// UNEDF1: Kortelainen et al. (2012) — optimized for global fit
    UNEDF1,
}

/// Result of an HFB calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfbResult {
    /// Total binding energy (MeV)
    pub binding_energy: f64,
    /// Quadrupole deformation β₂
    pub beta2: f64,
    /// Hexadecapole deformation β₄
    pub beta4: f64,
    /// Neutron pairing gap (MeV)
    pub neutron_pairing_gap: f64,
    /// Proton pairing gap (MeV)
    pub proton_pairing_gap: f64,
    /// RMS charge radius (fm)
    pub charge_radius: f64,
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Whether the calculation converged
    pub converged: bool,
    /// Parametrization used
    pub parametrization: SkyrmeParametrization,
}

/// HFB solver trait — to be implemented by concrete backends.
pub trait HfbSolver {
    /// Solve the HFB equations for nucleus (Z, N).
    fn solve(&self, z: u16, n: u16) -> Result<HfbResult, String>;

    /// Constrained HFB at fixed deformation β₂.
    fn solve_constrained(&self, z: u16, n: u16, beta2: f64) -> Result<HfbResult, String>;

    /// Fission barrier via constrained HFB scan.
    fn fission_barrier(&self, z: u16, n: u16, n_points: usize) -> Result<Vec<(f64, f64)>, String>;
}

/// Placeholder HFB solver that returns estimated values.
///
/// Uses FRDM deformation + DZ binding energy as estimates until
/// a real HFB backend is implemented.
pub struct PlaceholderHfb;

impl HfbSolver for PlaceholderHfb {
    fn solve(&self, z: u16, n: u16) -> Result<HfbResult, String> {
        let (beta2, beta4) = crate::deformation::frdm_deformation(z, n);
        let be = crate::duflo_zuker::dz_binding_energy(z, n);

        Ok(HfbResult {
            binding_energy: be,
            beta2,
            beta4,
            neutron_pairing_gap: 1.2, // Typical pairing gap ~1-2 MeV
            proton_pairing_gap: 1.0,
            charge_radius: 1.2 * ((z + n) as f64).powf(1.0 / 3.0),
            iterations: 0,
            converged: false, // Placeholder — not self-consistent
            parametrization: SkyrmeParametrization::SLy4,
        })
    }

    fn solve_constrained(&self, z: u16, n: u16, beta2: f64) -> Result<HfbResult, String> {
        let mut result = self.solve(z, n)?;
        result.beta2 = beta2;
        Ok(result)
    }

    fn fission_barrier(&self, z: u16, n: u16, n_points: usize) -> Result<Vec<(f64, f64)>, String> {
        let barrier = crate::fission_barrier::compute_fission_barrier(z, n);
        // Return simplified barrier profile
        let points: Vec<(f64, f64)> = (0..=n_points)
            .map(|i| {
                let beta = i as f64 * 1.5 / n_points as f64;
                let e = if beta < barrier.saddle_beta2 {
                    beta / barrier.saddle_beta2 * barrier.total_barrier
                } else {
                    barrier.total_barrier
                        * (1.0 - (beta - barrier.saddle_beta2) / (1.5 - barrier.saddle_beta2))
                };
                (beta, e.max(0.0))
            })
            .collect();
        Ok(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placeholder_solves() {
        let solver = PlaceholderHfb;
        let result = solver.solve(82, 126).unwrap();
        assert!(result.binding_energy > 0.0);
        assert!(!result.converged); // Placeholder is not self-consistent
    }

    #[test]
    fn test_fission_barrier_profile() {
        let solver = PlaceholderHfb;
        let profile = solver.fission_barrier(92, 144, 20).unwrap();
        assert_eq!(profile.len(), 21);
        assert!(profile[0].1 >= 0.0); // Energy at β₂=0
    }
}
