// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Competing gravity models for rotation-curve prediction.
//!
//! Each model implements [`RotationModel`]: given a SPARC galaxy, predict the
//! rotation velocity at every observed radius. Free-parameter counts feed the
//! AIC/BIC accounting in [`crate::fit`] — Newtonian, MOND, and conformal
//! gravity have **zero** per-galaxy free parameters (universal constants +
//! fixed mass-to-light ratios); NFW fits two per galaxy (V200, c).
//!
//! Fairness note: published NFW/SPARC fits often free the mass-to-light
//! ratio Υ per galaxy. We fix Υ = 0.5/0.7 across ALL models so no model
//! gains hidden flexibility. Documented in the crate README.

mod conformal;
mod mond;
mod newtonian;
mod nfw;

pub use conformal::ConformalGravity;
pub use mond::Mond;
pub use newtonian::Newtonian;
pub use nfw::NfwHalo;

use crate::constants::{UPSILON_BULGE, UPSILON_DISK, V_ERR_FLOOR_KMS};
use crate::sparc::{Galaxy, RotationPoint};

/// Result of fitting (or evaluating) one model on one galaxy.
#[derive(Debug, Clone)]
pub struct FittedCurve {
    /// Predicted rotation velocity at each observed radius [km/s]
    pub v_pred: Vec<f64>,
    /// Best-fit per-galaxy parameters (empty for 0-parameter models)
    pub params: Vec<f64>,
    /// χ² against the observed curve (computed with [`curve_chi2`])
    pub chi2: f64,
    /// Whether the fit converged (always true for 0-parameter models)
    pub converged: bool,
}

/// A gravity model that predicts a galaxy's rotation curve.
pub trait RotationModel {
    fn name(&self) -> &'static str;
    /// Per-galaxy free parameters (for AIC/BIC accounting).
    fn n_free_params(&self) -> usize;
    fn fit(&self, galaxy: &Galaxy) -> FittedCurve;
}

/// Squared baryonic rotation velocity [(km/s)²] at one radius, from the
/// SPARC Υ=1 component velocities with fixed mass-to-light ratios.
///
/// **Sign-preserving quadrature**: SPARC encodes central gas depressions as
/// negative Vgas, meaning the gas exerts a net *outward* pull there. The
/// correct composition is v·|v|, not v² — naive squaring silently flips the
/// sign of that contribution (classic SPARC reimplementation bug).
pub fn v_baryonic_sq(p: &RotationPoint) -> f64 {
    p.v_gas * p.v_gas.abs()
        + UPSILON_DISK * p.v_disk * p.v_disk.abs()
        + UPSILON_BULGE * p.v_bul * p.v_bul.abs()
}

/// χ² of a predicted curve against the observed one.
///
/// Uncertainties are floored at [`V_ERR_FLOOR_KMS`] purely to guard division;
/// no statistical error floor is applied.
pub fn curve_chi2(galaxy: &Galaxy, v_pred: &[f64]) -> f64 {
    galaxy
        .points
        .iter()
        .zip(v_pred)
        .map(|(p, vp)| {
            let e = p.e_v_obs.max(V_ERR_FLOOR_KMS);
            let d = (p.v_obs - vp) / e;
            d * d
        })
        .sum()
}

/// Total baryonic mass [M☉]: stars (Υ_d × L[3.6]) + helium-corrected HI gas.
/// Used by conformal gravity's per-mass linear term.
///
/// SPARC's L[3.6] is total luminosity (disk + bulge); applying Υ_d to all of
/// it slightly underestimates bulge-heavy systems — documented approximation.
pub fn baryonic_mass_msun(galaxy: &Galaxy) -> f64 {
    use crate::constants::GAS_HELIUM_FACTOR;
    (UPSILON_DISK * galaxy.luminosity_3p6 + GAS_HELIUM_FACTOR * galaxy.mhi_e9msun) * 1.0e9
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    /// Synthetic galaxy with a given set of (r, v_obs, e, v_gas, v_disk) rows.
    pub fn galaxy_from_rows(rows: &[(f64, f64, f64, f64, f64)]) -> Galaxy {
        Galaxy {
            name: "TEST".to_string(),
            distance_mpc: 10.0,
            inclination_deg: 60.0,
            luminosity_3p6: 5.0,
            sb_eff: 100.0,
            mhi_e9msun: 1.0,
            quality: 1,
            points: rows
                .iter()
                .map(|&(r, v, e, vg, vd)| RotationPoint {
                    r_kpc: r,
                    v_obs: v,
                    e_v_obs: e,
                    v_gas: vg,
                    v_disk: vd,
                    v_bul: 0.0,
                    sb_disk: 50.0,
                    sb_bul: 0.0,
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baryonic_quadrature_preserves_gas_sign() {
        let p = RotationPoint {
            r_kpc: 0.5,
            v_obs: 20.0,
            e_v_obs: 2.0,
            v_gas: -10.0, // central depression
            v_disk: 20.0,
            v_bul: 0.0,
            sb_disk: 100.0,
            sb_bul: 0.0,
        };
        let v_sq = v_baryonic_sq(&p);
        // gas contributes −100, disk contributes +0.5·400 = 200
        assert!((v_sq - 100.0).abs() < 1e-12, "v_sq = {v_sq}");
    }

    #[test]
    fn chi2_of_perfect_prediction_is_zero() {
        let g = test_util::galaxy_from_rows(&[(1.0, 50.0, 2.0, 10.0, 40.0)]);
        let chi2 = curve_chi2(&g, &[50.0]);
        assert_eq!(chi2, 0.0);
    }

    #[test]
    fn chi2_scales_with_normalized_deviation() {
        let g = test_util::galaxy_from_rows(&[(1.0, 50.0, 2.0, 10.0, 40.0)]);
        // 2σ off → χ² = 4
        let chi2 = curve_chi2(&g, &[54.0]);
        assert!((chi2 - 4.0).abs() < 1e-12);
    }
}
