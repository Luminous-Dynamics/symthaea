// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! NFW dark-matter halo (Navarro, Frenk & White 1996).
//!
//!   V_NFW²(r) = V200² · [ln(1+cx) − cx/(1+cx)] / (x·[ln(1+c) − c/(1+c)])
//!
//! with x = r/R200 and R200 = V200/(10·H0). Two free parameters per galaxy
//! (V200, c), fit by Levenberg-Marquardt in log-space — the log
//! parameterization keeps parameters O(1) (required by the fixed
//! finite-difference step in `symthaea_core`'s LM) and enforces positivity
//! without box constraints.
//!
//! This is the flexible reference model in the comparison: its 2 per-galaxy
//! parameters absorb galaxy-level structure by construction, which the
//! AIC/BIC accounting (and the residual-learnability caveat in the README)
//! must and does acknowledge.

use super::{FittedCurve, RotationModel, curve_chi2, v_baryonic_sq};
use crate::constants::{H0_KMS_MPC, V_ERR_FLOOR_KMS};
use crate::sparc::Galaxy;
use symthaea_core::hdc::optimization::{LevenbergMarquardt, OptimizationEngine};

pub struct NfwHalo;

/// NFW halo circular velocity squared [(km/s)²] at radius r.
pub fn nfw_v_sq(r_kpc: f64, v200_kms: f64, c: f64) -> f64 {
    if r_kpc <= 0.0 || v200_kms <= 0.0 || c <= 0.0 {
        return 0.0;
    }
    // R200 [kpc]: R200 [Mpc] = V200/(10·H0)
    let r200_kpc = 1000.0 * v200_kms / (10.0 * H0_KMS_MPC);
    let x = r_kpc / r200_kpc;
    let mu = |y: f64| (1.0 + y).ln() - y / (1.0 + y);
    v200_kms * v200_kms * mu(c * x) / (x * mu(c))
}

/// Model prediction for the full curve given log-space params [ln V200, ln c].
fn predict(galaxy: &Galaxy, ln_params: &[f64]) -> Vec<f64> {
    let v200 = ln_params[0].exp();
    let c = ln_params[1].exp();
    galaxy
        .points
        .iter()
        .map(|p| {
            (v_baryonic_sq(p) + nfw_v_sq(p.r_kpc, v200, c))
                .max(0.0)
                .sqrt()
        })
        .collect()
}

fn weighted_residuals(galaxy: &Galaxy, ln_params: &[f64]) -> Vec<f64> {
    let v_pred = predict(galaxy, ln_params);
    galaxy
        .points
        .iter()
        .zip(&v_pred)
        .map(|(p, vp)| (p.v_obs - vp) / p.e_v_obs.max(V_ERR_FLOOR_KMS))
        .collect()
}

impl RotationModel for NfwHalo {
    fn name(&self) -> &'static str {
        "nfw_halo"
    }

    fn n_free_params(&self) -> usize {
        2
    }

    fn fit(&self, galaxy: &Galaxy) -> FittedCurve {
        // Multi-start: V200 seeded from the outermost observed velocity,
        // concentration from typical halo values.
        let v_outer = galaxy
            .points
            .last()
            .map(|p| p.v_obs)
            .unwrap_or(100.0)
            .max(20.0);
        let starts: Vec<[f64; 2]> = [3.0, 8.0, 15.0]
            .iter()
            .map(|&c0: &f64| [v_outer.ln(), c0.ln()])
            .collect();

        let mut best: Option<(Vec<f64>, f64, bool)> = None;
        for p0 in &starts {
            let res = LevenbergMarquardt::fit(|p| weighted_residuals(galaxy, p), p0, 1e-8, 200);
            let better = best.as_ref().is_none_or(|(_, sse, _)| res.sse < *sse);
            if better {
                best = Some((res.params.clone(), res.sse, res.converged));
            }
        }
        let (mut params, mut sse, mut converged) = best.expect("at least one LM start");

        // Nelder-Mead fallback if no LM start converged.
        if !converged {
            let chi2_fn = |p: &[f64]| {
                weighted_residuals(galaxy, p)
                    .iter()
                    .map(|r| r * r)
                    .sum::<f64>()
            };
            let nm = OptimizationEngine::nelder_mead(&chi2_fn, &params, 0.3, 1e-10);
            if nm.fx < sse {
                params = nm.x;
                sse = nm.fx;
                converged = nm.converged;
            }
        }

        let v_pred = predict(galaxy, &params);
        let chi2 = curve_chi2(galaxy, &v_pred);
        debug_assert!((chi2 - sse).abs() < 1e-6 * (1.0 + sse), "chi2/sse mismatch");
        FittedCurve {
            v_pred,
            params: vec![params[0].exp(), params[1].exp()], // [V200, c]
            chi2,
            converged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_models::test_util::galaxy_from_rows;
    use crate::sparc::{Galaxy, RotationPoint};

    #[test]
    fn nfw_velocity_rises_from_zero_and_stays_finite() {
        let v1 = nfw_v_sq(0.1, 150.0, 10.0);
        let v10 = nfw_v_sq(10.0, 150.0, 10.0);
        let v100 = nfw_v_sq(100.0, 150.0, 10.0);
        assert!(v1 > 0.0 && v10 > v1, "inner curve must rise");
        assert!(v100 > 0.0 && v100.is_finite());
        // At R200 the circular velocity equals V200 by construction
        let r200 = 1000.0 * 150.0 / (10.0 * H0_KMS_MPC);
        let v_at_r200 = nfw_v_sq(r200, 150.0, 10.0).sqrt();
        assert!((v_at_r200 - 150.0).abs() < 1e-9, "V(R200) = {v_at_r200}");
    }

    /// Synthetic-curve parameter recovery: generate a galaxy from known
    /// (V200, c), fit, and demand the truth comes back.
    #[test]
    fn recovers_synthetic_halo_parameters() {
        let (v200_true, c_true) = (160.0, 9.0);
        let radii: Vec<f64> = (1..=25).map(|i| i as f64).collect();
        let points: Vec<RotationPoint> = radii
            .iter()
            .map(|&r| {
                // Simple declining-baryon toy disk
                let v_disk = 80.0 * (-r / 15.0_f64).exp() + 20.0;
                let v_gas = 30.0;
                let v_bar_sq = v_gas * v_gas + crate::constants::UPSILON_DISK * v_disk * v_disk;
                let v_obs = (v_bar_sq + nfw_v_sq(r, v200_true, c_true)).sqrt();
                RotationPoint {
                    r_kpc: r,
                    v_obs,
                    e_v_obs: 2.0,
                    v_gas,
                    v_disk,
                    v_bul: 0.0,
                    sb_disk: 100.0,
                    sb_bul: 0.0,
                }
            })
            .collect();
        let galaxy = Galaxy {
            name: "SYNTH".into(),
            distance_mpc: 10.0,
            inclination_deg: 60.0,
            luminosity_3p6: 5.0,
            sb_eff: 100.0,
            mhi_e9msun: 1.0,
            quality: 1,
            points,
        };

        let fit = NfwHalo.fit(&galaxy);
        let (v200_fit, c_fit) = (fit.params[0], fit.params[1]);
        assert!(
            (v200_fit / v200_true - 1.0).abs() < 0.05,
            "V200: fit {v200_fit} vs true {v200_true}"
        );
        assert!(
            (c_fit / c_true - 1.0).abs() < 0.15,
            "c: fit {c_fit} vs true {c_true}"
        );
        assert!(
            fit.chi2 < 1.0,
            "noiseless synthetic curve must fit to ~0, got {}",
            fit.chi2
        );
    }

    #[test]
    fn fit_improves_on_newtonian_for_flat_curve() {
        // A flat 120 km/s curve that baryons alone cannot sustain
        let rows: Vec<(f64, f64, f64, f64, f64)> = (1..=15)
            .map(|i| {
                let r = i as f64 * 2.0;
                (r, 120.0, 3.0, 40.0, 60.0 * (-r / 10.0_f64).exp() + 10.0)
            })
            .collect();
        let g = galaxy_from_rows(&rows);
        let nfw = NfwHalo.fit(&g);
        let newton = super::super::Newtonian.fit(&g);
        assert!(
            nfw.chi2 < 0.2 * newton.chi2,
            "NFW χ²={} should crush Newtonian χ²={}",
            nfw.chi2,
            newton.chi2
        );
    }
}
