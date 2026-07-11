// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MOND via the Radial Acceleration Relation (McGaugh, Lelli & Schombert
//! 2016, PRL 117, 201101):
//!
//!   g_obs = g_bar / (1 − e^(−√(g_bar/a₀)))
//!
//! with the single universal acceleration scale a₀ = 1.2×10⁻¹⁰ m/s².
//! Zero per-galaxy free parameters: the prediction follows entirely from the
//! observed baryon distribution.
//!
//! Limits: g_bar ≫ a₀ → g_obs → g_bar (Newtonian); g_bar ≪ a₀ →
//! g_obs → √(g_bar·a₀) (deep-MOND flat curves).

use super::{FittedCurve, RotationModel, curve_chi2, v_baryonic_sq};
use crate::constants::{KMS_MS, KPC_M, MOND_A0, accel_si};
use crate::sparc::Galaxy;

pub struct Mond;

/// RAR interpolation: baryonic acceleration → observed acceleration [m/s²].
pub fn rar_g_obs(g_bar: f64) -> f64 {
    if g_bar <= 0.0 {
        return 0.0;
    }
    let x = (g_bar / MOND_A0).sqrt();
    if x < 1e-6 {
        // Series limit: g_bar/(1−e^(−x)) → g_bar/x = √(g_bar·a₀)
        return (g_bar * MOND_A0).sqrt();
    }
    g_bar / (1.0 - (-x).exp())
}

impl RotationModel for Mond {
    fn name(&self) -> &'static str {
        "mond_rar"
    }

    fn n_free_params(&self) -> usize {
        0
    }

    fn fit(&self, galaxy: &Galaxy) -> FittedCurve {
        let v_pred: Vec<f64> = galaxy
            .points
            .iter()
            .map(|p| {
                let g_bar = accel_si(v_baryonic_sq(p).max(0.0), p.r_kpc);
                let g_obs = rar_g_obs(g_bar);
                // v = √(g·r), back to km/s
                (g_obs * p.r_kpc * KPC_M).sqrt() / KMS_MS
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
    fn high_acceleration_limit_is_newtonian() {
        let g_bar = 100.0 * MOND_A0;
        let g_obs = rar_g_obs(g_bar);
        assert!(
            (g_obs / g_bar - 1.0).abs() < 1e-3,
            "ratio = {}",
            g_obs / g_bar
        );
    }

    #[test]
    fn deep_mond_limit_is_sqrt_gbar_a0() {
        let g_bar = 1e-4 * MOND_A0;
        let g_obs = rar_g_obs(g_bar);
        let expected = (g_bar * MOND_A0).sqrt();
        assert!(
            (g_obs / expected - 1.0).abs() < 0.01,
            "g_obs = {g_obs}, expected ≈ {expected}"
        );
    }

    #[test]
    fn rar_is_monotonic_and_amplifying() {
        let mut prev = 0.0;
        for i in 1..100 {
            let g_bar = MOND_A0 * (i as f64) / 10.0;
            let g_obs = rar_g_obs(g_bar);
            assert!(g_obs > prev, "not monotonic at i={i}");
            assert!(g_obs >= g_bar, "RAR must never de-amplify");
            prev = g_obs;
        }
    }

    #[test]
    fn mond_beats_newton_where_baryons_undershoot() {
        // Outer-disk-like point: v_bar ≈ 76 km/s at 20 kpc but v_obs = 120
        let g = galaxy_from_rows(&[(20.0, 120.0, 3.0, 50.0, 80.0)]);
        let mond = Mond.fit(&g);
        let v_bar = super::super::v_baryonic_sq(&g.points[0]).sqrt();
        assert!(mond.v_pred[0] > v_bar, "MOND must amplify above baryonic");
        assert!(
            mond.v_pred[0] > 90.0 && mond.v_pred[0] < 200.0,
            "v_mond = {}",
            mond.v_pred[0]
        );
    }

    /// External-validation gate against the real SPARC sample: the Radial
    /// Acceleration Relation's defining empirical claim (McGaugh, Lelli &
    /// Schombert 2016, PRL 117, 201101) is that log(g_obs) scatters around
    /// log(g_pred = rar_g_obs(g_bar)) by only ~0.11-0.13 dex — the residual
    /// after subtracting the RAR curve is small and (they argue) consistent
    /// with pure measurement error.
    ///
    /// We do NOT reproduce their exact point selection (they additionally
    /// exclude/weight points affected by beam smearing and asymmetric-drift
    /// corrections at small radii; we only apply the galaxy-level Q<=2,
    /// inc>=30deg cut used elsewhere in this crate). So this is not a test
    /// that our scatter equals 0.13 dex to the percent — it is a test that
    /// our RAR implementation produces the right BALLPARK of scatter on
    /// real data. A implementation bug (wrong exponent, wrong acceleration
    /// conversion, a sign error) would blow this out to multiple dex; a
    /// correct implementation should land within a few tenths of a dex of
    /// the published value even with our looser point selection.
    ///
    /// Measured on this crate's implementation (2026-07-07, n=3166 points):
    /// mean=-0.026 dex, scatter=0.177 dex — a bit above the published
    /// 0.11-0.13 dex, consistent with the missing beam-smearing/inner-radius
    /// cuts (which would only ever ADD scatter, never remove it, so a
    /// correct implementation should land at or above the published floor).
    #[test]
    #[ignore = "requires SPARC data: run scripts/download_sparc.sh"]
    fn rar_scatter_is_in_published_ballpark() {
        use crate::sparc::load_sparc;

        let galaxies =
            load_sparc(&crate::test_support::sparc_data_dir()).expect("load_sparc failed");
        let cut: Vec<_> = galaxies
            .into_iter()
            .filter(|g| g.quality <= 2 && g.inclination_deg >= 30.0)
            .collect();
        assert!(
            cut.len() > 100,
            "quality cut left too few galaxies: {}",
            cut.len()
        );

        let mut residuals = Vec::new();
        for g in &cut {
            for p in &g.points {
                if p.r_kpc <= 0.0 || p.v_obs <= 0.0 {
                    continue;
                }
                let g_bar = accel_si(v_baryonic_sq(p).max(0.0), p.r_kpc);
                let g_obs = accel_si(p.v_obs * p.v_obs, p.r_kpc);
                let g_pred = rar_g_obs(g_bar);
                if g_bar <= 0.0 || g_pred <= 0.0 || g_obs <= 0.0 {
                    continue;
                }
                residuals.push(g_obs.log10() - g_pred.log10());
            }
        }
        assert!(
            residuals.len() > 1000,
            "expected thousands of valid points, got {}",
            residuals.len()
        );

        let n = residuals.len() as f64;
        let mean: f64 = residuals.iter().sum::<f64>() / n;
        let variance: f64 = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let scatter = variance.sqrt();
        eprintln!(
            "RAR residuals: n={}, mean={:.4} dex, scatter={:.4} dex",
            residuals.len(),
            mean,
            scatter
        );

        // Systematic bias should be small — a large |mean| would indicate a
        // unit-conversion or amplitude bug, not just methodology difference.
        // Measured value is -0.026 dex; 0.1 leaves ~4x margin.
        assert!(
            mean.abs() < 0.1,
            "systematic bias too large: mean = {mean} dex"
        );
        // Published intrinsic scatter is ~0.11-0.13 dex under their tighter
        // point selection; our looser cut measures 0.177 dex. Lower bound
        // sits just below the published floor (our cut can only add
        // scatter, not remove it, so we shouldn't be much below 0.11); upper
        // bound leaves ~1.7x headroom above our measured value to absorb
        // ordinary dataset/environment variation while still catching a
        // real implementation regression.
        assert!(
            (0.10..0.30).contains(&scatter),
            "RAR scatter out of plausible range: {scatter} dex (n={})",
            residuals.len()
        );
    }
}
