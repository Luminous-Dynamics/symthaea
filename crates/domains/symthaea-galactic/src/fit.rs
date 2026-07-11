// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model-comparison statistics with honest free-parameter accounting.
//!
//! Conventions (documented in the README and embedded in the results JSON):
//! - AIC = χ² + 2k, BIC = χ² + k·ln(N), computed **per galaxy** with
//!   k = per-galaxy free parameters (0 for Newtonian/MOND/conformal, 2 for
//!   NFW) and N = points in that galaxy's curve.
//! - Sample totals sum χ² and sum k (NFW: 2 × N_galaxies), so the global
//!   ΔAIC/ΔBIC correctly charge NFW for its 350 fitted parameters.
//! - No statistical error floor is applied to SPARC uncertainties.

/// Reduced χ²: χ²/(N − k). Returns χ² unchanged if dof would be ≤ 0.
pub fn reduced_chi2(chi2: f64, n_points: usize, k_params: usize) -> f64 {
    let dof = n_points.saturating_sub(k_params);
    if dof == 0 { chi2 } else { chi2 / dof as f64 }
}

/// Akaike information criterion (χ² form): AIC = χ² + 2k.
pub fn aic(chi2: f64, k_params: usize) -> f64 {
    chi2 + 2.0 * k_params as f64
}

/// Bayesian information criterion (χ² form): BIC = χ² + k·ln(N).
pub fn bic(chi2: f64, k_params: usize, n_points: usize) -> f64 {
    chi2 + k_params as f64 * (n_points.max(1) as f64).ln()
}

/// Coefficient of determination: 1 − SS_res/SS_tot.
/// Returns 0.0 when the observations have zero variance.
pub fn r_squared(observed: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(observed.len(), predicted.len());
    if observed.is_empty() {
        return 0.0;
    }
    let mean = observed.iter().sum::<f64>() / observed.len() as f64;
    let ss_tot: f64 = observed.iter().map(|o| (o - mean) * (o - mean)).sum();
    let ss_res: f64 = observed
        .iter()
        .zip(predicted)
        .map(|(o, p)| (o - p) * (o - p))
        .sum();
    if ss_tot <= 0.0 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Mean absolute error.
pub fn mae(observed: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(observed.len(), predicted.len());
    if observed.is_empty() {
        return 0.0;
    }
    observed
        .iter()
        .zip(predicted)
        .map(|(o, p)| (o - p).abs())
        .sum::<f64>()
        / observed.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_chi2_divides_by_dof() {
        assert_eq!(reduced_chi2(20.0, 12, 2), 2.0);
        // dof == 0 → return χ² rather than dividing by zero
        assert_eq!(reduced_chi2(20.0, 2, 2), 20.0);
    }

    #[test]
    fn aic_bic_charge_for_parameters() {
        // Equal χ²: the 2-parameter model must lose on both criteria
        let (chi2, n) = (50.0, 20);
        assert!(aic(chi2, 2) > aic(chi2, 0));
        assert!(bic(chi2, 2, n) > bic(chi2, 0, n));
        // BIC penalty exceeds AIC penalty once ln(N) > 2 (N > 7)
        assert!(bic(chi2, 2, n) > aic(chi2, 2));
    }

    #[test]
    fn r_squared_perfect_and_mean_predictor() {
        let obs = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(r_squared(&obs, &obs), 1.0);
        let mean_pred = [2.5; 4];
        assert!(r_squared(&obs, &mean_pred).abs() < 1e-12);
    }

    #[test]
    fn mae_basic() {
        assert_eq!(mae(&[1.0, 3.0], &[2.0, 1.0]), 1.5);
    }
}
