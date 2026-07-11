// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Galaxy-level train/test splitting and held-out evaluation for the HDC
//! residual regressors.
//!
//! All splits are **galaxy-level**, never point-level: points within one
//! galaxy's rotation curve are highly correlated (same distance error,
//! inclination, mass-to-light systematics), so a point-level split would
//! leak information between train and test exactly like the DEAM song-level
//! split guards against track leakage in `symthaea-muse`.

use crate::encoder::{GalaxyPointState, gas_fraction};
use crate::fit::{mae, r_squared};
use crate::gravity_models::{FittedCurve, RotationModel};
use crate::hdc_residual::{HdcResidualRegressor, ResidualExample};
use crate::sparc::Galaxy;

/// One (galaxy, model-fit) pair reduced to residual training examples.
pub(crate) fn galaxy_residual_examples(galaxy: &Galaxy, fit: &FittedCurve) -> Vec<ResidualExample> {
    let gasfrac = gas_fraction(galaxy.mhi_e9msun, galaxy.luminosity_3p6);
    galaxy
        .points
        .iter()
        .zip(&fit.v_pred)
        .map(|(p, vp)| {
            let e = p.e_v_obs.max(crate::constants::V_ERR_FLOOR_KMS);
            ResidualExample {
                state: GalaxyPointState {
                    r_kpc: p.r_kpc,
                    v_gas: p.v_gas,
                    v_disk: p.v_disk,
                    v_bul: p.v_bul,
                    sb_disk: p.sb_disk,
                    sb_bul: p.sb_bul,
                    luminosity_3p6: galaxy.luminosity_3p6,
                    distance_mpc: galaxy.distance_mpc,
                    inclination_deg: galaxy.inclination_deg,
                    gas_fraction: gasfrac,
                },
                target: (p.v_obs - vp) / e,
            }
        })
        .collect()
}

/// Deterministic galaxy-level train/test split (≈90/10), mirroring
/// `symthaea-muse`'s FNV-bucket song split: `fnv(name) % 10 == 0 → test`.
///
/// Takes `&[&Galaxy]` (not `&[Galaxy]`) since every real call site operates
/// on an already-filtered subset (e.g. after a quality cut).
pub fn train_test_split<'a>(galaxies: &[&'a Galaxy]) -> (Vec<&'a Galaxy>, Vec<&'a Galaxy>) {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for &g in galaxies {
        if HdcResidualRegressor::fnv1a(&g.name) % 10 == 0 {
            test.push(g);
        } else {
            train.push(g);
        }
    }
    (train, test)
}

/// Deterministic 5-fold galaxy-level partition (each galaxy assigned to
/// exactly one fold by `fnv(name) % 5`).
pub fn five_fold_split<'a>(galaxies: &[&'a Galaxy]) -> [Vec<&'a Galaxy>; 5] {
    let mut folds: [Vec<&Galaxy>; 5] = Default::default();
    for &g in galaxies {
        let fold = (HdcResidualRegressor::fnv1a(&g.name) % 5) as usize;
        folds[fold].push(g);
    }
    folds
}

/// Extrapolation holdout: the lowest-quintile-by-effective-surface-brightness
/// galaxies, held out entirely. This is the regime (LSB galaxies) where
/// Newtonian/MOND/NFW/conformal predictions diverge most, making it the
/// hardest and most informative extrapolation test.
pub fn low_surface_brightness_holdout<'a>(
    galaxies: &[&'a Galaxy],
) -> (Vec<&'a Galaxy>, Vec<&'a Galaxy>) {
    let mut sorted: Vec<&Galaxy> = galaxies.to_vec();
    sorted.sort_by(|a, b| {
        a.sb_eff
            .partial_cmp(&b.sb_eff)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cutoff = (sorted.len() / 5).max(1);
    let (held_out, kept) = sorted.split_at(cutoff.min(sorted.len()));
    (kept.to_vec(), held_out.to_vec())
}

/// Held-out evaluation summary for one gravity model's residual regressor.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResidualEvalResult {
    pub model_name: String,
    pub n_train_examples: usize,
    pub n_test_examples: usize,
    /// R² of the trained regressor's predictions on held-out points
    pub r_squared: f64,
    /// R² of the mean-predictor baseline (honest floor — a model must beat this)
    pub baseline_r_squared: f64,
    pub mae: f64,
    pub baseline_mae: f64,
}

/// Train a residual regressor for one model on the train-galaxy set and
/// evaluate held-out R²/MAE against the mean-predictor baseline.
pub fn evaluate_residual_regressor(
    model: &dyn RotationModel,
    train_galaxies: &[&Galaxy],
    test_galaxies: &[&Galaxy],
    epochs: usize,
    seed_offset: u64,
) -> ResidualEvalResult {
    let train_examples: Vec<ResidualExample> = train_galaxies
        .iter()
        .flat_map(|g| galaxy_residual_examples(g, &model.fit(g)))
        .collect();
    let test_examples: Vec<ResidualExample> = test_galaxies
        .iter()
        .flat_map(|g| galaxy_residual_examples(g, &model.fit(g)))
        .collect();

    let mut regressor = HdcResidualRegressor::new(seed_offset);
    regressor.train(&train_examples, epochs);

    let observed: Vec<f64> = test_examples.iter().map(|e| e.target).collect();
    let predicted: Vec<f64> = test_examples
        .iter()
        .map(|e| regressor.predict(&e.state))
        .collect();

    let train_mean = if train_examples.is_empty() {
        0.0
    } else {
        train_examples.iter().map(|e| e.target).sum::<f64>() / train_examples.len() as f64
    };
    let baseline: Vec<f64> = vec![train_mean; test_examples.len()];

    ResidualEvalResult {
        model_name: model.name().to_string(),
        n_train_examples: train_examples.len(),
        n_test_examples: test_examples.len(),
        r_squared: r_squared(&observed, &predicted),
        baseline_r_squared: r_squared(&observed, &baseline),
        mae: mae(&observed, &predicted),
        baseline_mae: mae(&observed, &baseline),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_models::Newtonian;
    use crate::sparc::RotationPoint;

    fn synth_galaxy(name: &str, sb_eff: f64) -> Galaxy {
        Galaxy {
            name: name.to_string(),
            distance_mpc: 10.0,
            inclination_deg: 60.0,
            luminosity_3p6: 5.0,
            sb_eff,
            mhi_e9msun: 1.0,
            quality: 1,
            points: (1..=10)
                .map(|i| RotationPoint {
                    r_kpc: i as f64,
                    v_obs: 20.0 + i as f64 * 3.0,
                    e_v_obs: 2.0,
                    v_gas: 10.0,
                    v_disk: 30.0,
                    v_bul: 0.0,
                    sb_disk: 200.0,
                    sb_bul: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn split_is_deterministic_and_covers_all_galaxies() {
        let galaxies: Vec<Galaxy> = (0..30)
            .map(|i| synth_galaxy(&format!("G{i}"), 100.0))
            .collect();
        let refs: Vec<&Galaxy> = galaxies.iter().collect();
        let (train1, test1) = train_test_split(&refs);
        let (train2, test2) = train_test_split(&refs);
        assert_eq!(train1.len(), train2.len());
        assert_eq!(test1.len(), test2.len());
        assert_eq!(train1.len() + test1.len(), galaxies.len());
        // No overlap
        for t in &test1 {
            assert!(!train1.iter().any(|g| g.name == t.name));
        }
    }

    #[test]
    fn five_fold_split_partitions_without_overlap() {
        let galaxies: Vec<Galaxy> = (0..37)
            .map(|i| synth_galaxy(&format!("G{i}"), 100.0))
            .collect();
        let refs: Vec<&Galaxy> = galaxies.iter().collect();
        let folds = five_fold_split(&refs);
        let total: usize = folds.iter().map(|f| f.len()).sum();
        assert_eq!(total, galaxies.len());
    }

    #[test]
    fn lsb_holdout_selects_lowest_surface_brightness() {
        let mut galaxies: Vec<Galaxy> = Vec::new();
        for i in 0..20 {
            galaxies.push(synth_galaxy(&format!("G{i}"), 10.0 + i as f64 * 50.0));
        }
        let refs: Vec<&Galaxy> = galaxies.iter().collect();
        let (kept, held_out) = low_surface_brightness_holdout(&refs);
        assert!(!held_out.is_empty());
        assert_eq!(kept.len() + held_out.len(), galaxies.len());
        let max_held_sb = held_out.iter().map(|g| g.sb_eff).fold(0.0, f64::max);
        let min_kept_sb = kept.iter().map(|g| g.sb_eff).fold(f64::INFINITY, f64::min);
        assert!(max_held_sb <= min_kept_sb, "holdout must be the LSB tail");
    }

    #[test]
    fn evaluate_residual_regressor_runs_end_to_end() {
        let train: Vec<Galaxy> = (0..8)
            .map(|i| synth_galaxy(&format!("TR{i}"), 100.0))
            .collect();
        let test: Vec<Galaxy> = (0..3)
            .map(|i| synth_galaxy(&format!("TE{i}"), 100.0))
            .collect();
        let train_refs: Vec<&Galaxy> = train.iter().collect();
        let test_refs: Vec<&Galaxy> = test.iter().collect();

        let result = evaluate_residual_regressor(&Newtonian, &train_refs, &test_refs, 3, 42);
        assert!(result.r_squared.is_finite());
        assert!(result.baseline_r_squared.is_finite());
        assert!(result.n_train_examples > 0);
        assert!(result.n_test_examples > 0);
    }
}
