// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conformal gravity (Mannheim & O'Brien 2012, PRD 85, 124020).
//!
//! The fourth-order Weyl-squared field equations yield, outside a source of
//! N★ solar masses, an additional linear potential on top of Newton:
//!
//!   v²(r) = v_bar²(r) + γ★ N★ c² r / 2 + γ₀ c² r / 2 − κ c² r²
//!
//! where γ★ (per solar mass), γ₀, and κ are claimed-universal constants —
//! zero per-galaxy free parameters once the baryonic mass is fixed.
//!
//! **Honesty flags** (see crate README for citations):
//! - The very derivation of this non-relativistic limit is contested
//!   (Flanagan 2006; Hobson & Lasenby 2021). This module tests the published
//!   phenomenological formula, not the theory's internal consistency.
//! - Mannheim's own fits treat the stellar M/L as a per-galaxy parameter;
//!   we fix Υ = 0.5/0.7 for cross-model fairness, which will worsen his
//!   χ² relative to the published values.

use super::{FittedCurve, RotationModel, baryonic_mass_msun, curve_chi2, v_baryonic_sq};
use crate::constants::{C_M_S, CG_GAMMA_0, CG_GAMMA_STAR, CG_KAPPA, KMS_MS, KPC_M};
use crate::sparc::Galaxy;

pub struct ConformalGravity;

/// The conformal correction to v² [(km/s)²] at radius r for a galaxy of
/// n_star solar masses of baryons: γ★N★c²r/2 + γ₀c²r/2 − κc²r².
pub fn conformal_v_sq_correction(r_kpc: f64, n_star: f64) -> f64 {
    let r_m = r_kpc * KPC_M;
    let c2 = C_M_S * C_M_S;
    let v_sq_si = CG_GAMMA_STAR * n_star * c2 * r_m / 2.0 + CG_GAMMA_0 * c2 * r_m / 2.0
        - CG_KAPPA * c2 * r_m * r_m;
    v_sq_si / (KMS_MS * KMS_MS)
}

impl RotationModel for ConformalGravity {
    fn name(&self) -> &'static str {
        "conformal_mannheim"
    }

    fn n_free_params(&self) -> usize {
        0
    }

    fn fit(&self, galaxy: &Galaxy) -> FittedCurve {
        let n_star = baryonic_mass_msun(galaxy);
        let v_pred: Vec<f64> = galaxy
            .points
            .iter()
            .map(|p| {
                let v_sq = v_baryonic_sq(p) + conformal_v_sq_correction(p.r_kpc, n_star);
                v_sq.max(0.0).sqrt()
            })
            .collect();
        let chi2 = curve_chi2(galaxy, &v_pred);
        FittedCurve {
            v_pred,
            params: vec![],
            chi2,
            converged: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_models::test_util::galaxy_from_rows;

    #[test]
    fn linear_terms_dominate_at_galactic_radii() {
        // At 10 kpc with 1e11 M☉ the correction should be tens-of-km/s scale
        // and positive (κ cutoff subdominant).
        let dv_sq = conformal_v_sq_correction(10.0, 1.0e11);
        let dv = dv_sq.sqrt();
        assert!((30.0..300.0).contains(&dv), "correction → {dv} km/s");
    }

    #[test]
    fn kappa_cutoff_eventually_wins() {
        // The −κc²r² term must turn the correction over at very large radii —
        // conformal gravity predicts curves eventually FALL. (~100+ kpc scale)
        let small = conformal_v_sq_correction(10.0, 1.0e9);
        assert!(small > 0.0);
        let mut turned_negative = false;
        for r in [100.0, 300.0, 1000.0, 3000.0] {
            if conformal_v_sq_correction(r, 1.0e9) < 0.0 {
                turned_negative = true;
                break;
            }
        }
        assert!(turned_negative, "κ term never dominated even at 3 Mpc");
    }

    #[test]
    fn correction_scales_with_baryonic_mass() {
        let low = conformal_v_sq_correction(10.0, 1.0e9);
        let high = conformal_v_sq_correction(10.0, 1.0e11);
        assert!(high > low);
    }

    #[test]
    fn dwarf_galaxy_gets_mostly_universal_terms() {
        // For a tiny dwarf (N★ ~ 1e8), γ₀ dominates γ★N★:
        // γ★·1e8 = 5.42e-31 < γ₀ = 3.06e-28
        let g = galaxy_from_rows(&[(2.0, 25.0, 2.0, 10.0, 15.0)]);
        let fit = ConformalGravity.fit(&g);
        assert!(fit.v_pred[0] > 0.0);
        assert!(fit.converged);
    }
}
