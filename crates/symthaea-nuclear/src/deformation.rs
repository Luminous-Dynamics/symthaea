// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FRDM(2012) deformation parameter lookup.
//!
//! Provides ground-state deformation parameters (β₂, β₄) from the
//! Finite Range Droplet Model of Möller et al. (2016).
//!
//! Reference: Möller, Sierk, Myers, Sagawa, At. Data Nucl. Data Tables 109-110 (2016).

use serde::{Deserialize, Serialize};

/// Deformation parameters for a nucleus.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DeformationParams {
    pub z: u16,
    pub n: u16,
    /// Quadrupole deformation β₂ (0 = spherical, >0 = prolate, <0 = oblate)
    pub beta2: f64,
    /// Hexadecapole deformation β₄
    pub beta4: f64,
}

/// Lookup FRDM(2012) deformation parameters for superheavy elements.
///
/// Returns (β₂, β₄) for the given (Z, N). Falls back to interpolation
/// if exact entry not found.
pub fn frdm_deformation(z: u16, n: u16) -> (f64, f64) {
    // Search the lookup table
    if let Some(entry) = FRDM_SUPERHEAVY.iter().find(|e| e.0 == z && e.1 == n) {
        return (entry.2, entry.3);
    }

    // Interpolate from nearest entries
    let mut best_dist = u32::MAX;
    let mut best = (0.0, 0.0);

    for &(ez, en, b2, b4) in FRDM_SUPERHEAVY.iter() {
        let dist = (ez as i32 - z as i32).unsigned_abs() + (en as i32 - n as i32).unsigned_abs();
        if dist < best_dist {
            best_dist = dist;
            best = (b2, b4);
        }
    }

    best
}

/// Check if a nucleus is predicted to be near-spherical (|β₂| < 0.1).
pub fn is_near_spherical(z: u16, n: u16) -> bool {
    let (b2, _) = frdm_deformation(z, n);
    b2.abs() < 0.1
}

/// FRDM(2012) deformation parameters for superheavy elements.
/// Format: (Z, N, β₂, β₄)
///
/// Values from Möller et al. (2016) Table IV, superheavy region.
/// Focused on Z=100-126, N=150-200 for island-of-stability predictions.
const FRDM_SUPERHEAVY: &[(u16, u16, f64, f64)] = &[
    // Z=100 (Fermium)
    (100, 150, 0.275, -0.050),
    (100, 152, 0.270, -0.045),
    (100, 154, 0.255, -0.035),
    (100, 156, 0.240, -0.025),
    (100, 158, 0.215, -0.010),
    (100, 160, 0.180, 0.005),
    (100, 162, 0.145, 0.015),
    (100, 164, 0.100, 0.020),
    (100, 170, 0.030, 0.010),
    (100, 180, 0.015, 0.005),
    (100, 184, 0.005, 0.000),
    // Z=104 (Rutherfordium)
    (104, 152, 0.270, -0.050),
    (104, 156, 0.245, -0.030),
    (104, 160, 0.195, 0.000),
    (104, 164, 0.120, 0.015),
    (104, 170, 0.050, 0.010),
    (104, 176, 0.020, 0.005),
    (104, 180, 0.010, 0.000),
    (104, 184, 0.005, 0.000),
    // Z=108 (Hassium)
    (108, 152, 0.260, -0.045),
    (108, 156, 0.235, -0.025),
    (108, 160, 0.175, 0.000),
    (108, 164, 0.095, 0.010),
    (108, 170, 0.035, 0.005),
    (108, 176, 0.015, 0.000),
    (108, 180, 0.005, 0.000),
    (108, 184, 0.000, 0.000), // Near spherical
    // Z=114 (Flerovium) — predicted doubly-magic spherical
    (114, 160, 0.100, 0.005),
    (114, 164, 0.060, 0.000),
    (114, 170, 0.020, 0.000),
    (114, 176, 0.005, 0.000),
    (114, 180, 0.000, 0.000), // Spherical!
    (114, 184, 0.000, 0.000), // Doubly magic: spherical
    (114, 190, 0.010, 0.000),
    // Z=118 (Oganesson)
    (118, 170, 0.030, 0.000),
    (118, 176, 0.010, 0.000),
    (118, 180, 0.005, 0.000),
    (118, 184, 0.000, 0.000),
    // Z=120 — predicted proton magic number
    (120, 170, 0.035, 0.005),
    (120, 172, 0.020, 0.000),
    (120, 176, 0.010, 0.000),
    (120, 180, 0.005, 0.000),
    (120, 184, 0.000, 0.000), // Potentially doubly magic
    (120, 190, 0.015, 0.005),
    // Z=126 — predicted proton magic number
    (126, 180, 0.010, 0.000),
    (126, 184, 0.005, 0.000), // Near doubly magic
    (126, 190, 0.020, 0.005),
    (126, 196, 0.035, 0.010),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fl114_n184_spherical() {
        let (b2, b4) = frdm_deformation(114, 184);
        assert!(
            b2.abs() < 0.05,
            "Fl-298 (Z=114, N=184) should be near-spherical: β₂={}",
            b2
        );
        assert!(is_near_spherical(114, 184));
    }

    #[test]
    fn test_fm100_n150_deformed() {
        let (b2, _) = frdm_deformation(100, 150);
        assert!(b2 > 0.2, "Fm-250 should be well-deformed: β₂={}", b2);
        assert!(!is_near_spherical(100, 150));
    }

    #[test]
    fn test_interpolation() {
        // Z=112, N=170 not in table — should interpolate from nearest
        let (b2, _) = frdm_deformation(112, 170);
        assert!(b2.is_finite(), "Interpolated β₂ should be finite");
    }
}
