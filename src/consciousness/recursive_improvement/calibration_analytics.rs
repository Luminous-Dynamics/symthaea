// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Rigorous calibration analytics for the Brier-score tracker, powered by
//! `symthaea-statistics`.
//!
//! The tracker records `(confidence, was_correct)` pairs and a running Brier
//! sum, but a bare Brier score can't say *why* it is what it is. This module
//! adds the standard decomposition and diagnostics:
//!
//! - **Murphy's decomposition**: `Brier = Reliability − Resolution +
//!   Uncertainty`. Reliability is calibration error (want low), Resolution is
//!   discrimination (want high), Uncertainty is the irreducible base-rate
//!   variance. The identity is exact when each confidence bin is pure (one
//!   forecast value) and approximate under fixed-width binning.
//! - **Expected Calibration Error (ECE)**: the bin-weighted mean gap between
//!   confidence and observed accuracy.
//! - **Wilson score interval** on the overall accuracy (via the statistics
//!   crate's inverse-normal quantile).

use symthaea_statistics::distributions::normal_quantile;
use symthaea_statistics::mean;

/// A full calibration report over a set of resolved predictions.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationReport {
    /// Number of resolved predictions analysed.
    pub n: usize,
    /// Mean Brier score (0 = perfect, 0.25 = always-0.5 guessing, 1 = worst).
    pub brier: f64,
    /// Reliability (calibration error) — lower is better.
    pub reliability: f64,
    /// Resolution (discrimination) — higher is better.
    pub resolution: f64,
    /// Uncertainty (base-rate variance) — a property of the outcomes, not the model.
    pub uncertainty: f64,
    /// Expected calibration error (bin-weighted |confidence − accuracy|).
    pub ece: f64,
    /// Overall accuracy (fraction correct).
    pub accuracy: f64,
    /// 95% Wilson score interval for the accuracy.
    pub accuracy_ci95: (f64, f64),
}

/// Analyse `(confidence, was_correct)` pairs into a [`CalibrationReport`],
/// binning confidences into `n_bins` fixed-width bins over `[0, 1]`. Returns
/// `None` for an empty input or `n_bins == 0`.
pub fn analyze_pairs(data: &[(f64, bool)], n_bins: usize) -> Option<CalibrationReport> {
    if data.is_empty() || n_bins == 0 {
        return None;
    }
    let n = data.len();
    let nf = n as f64;

    // Overall accuracy (base rate) via the statistics crate's mean.
    let corrects: Vec<f64> = data
        .iter()
        .map(|(_, c)| if *c { 1.0 } else { 0.0 })
        .collect();
    let accuracy = mean(&corrects)?;

    let brier = data
        .iter()
        .map(|(f, c)| (f - if *c { 1.0 } else { 0.0 }).powi(2))
        .sum::<f64>()
        / nf;

    // Group into fixed-width confidence bins.
    let mut bin_sum_conf = vec![0.0f64; n_bins];
    let mut bin_correct = vec![0.0f64; n_bins];
    let mut bin_count = vec![0usize; n_bins];
    for &(f, c) in data {
        let idx = ((f * n_bins as f64).floor() as usize).min(n_bins - 1);
        bin_sum_conf[idx] += f;
        bin_correct[idx] += if c { 1.0 } else { 0.0 };
        bin_count[idx] += 1;
    }

    let (mut reliability, mut resolution, mut ece) = (0.0, 0.0, 0.0);
    for k in 0..n_bins {
        if bin_count[k] == 0 {
            continue;
        }
        let nk = bin_count[k] as f64;
        let mean_conf = bin_sum_conf[k] / nk;
        let obs_freq = bin_correct[k] / nk;
        reliability += nk * (mean_conf - obs_freq).powi(2);
        resolution += nk * (obs_freq - accuracy).powi(2);
        ece += nk * (mean_conf - obs_freq).abs();
    }
    reliability /= nf;
    resolution /= nf;
    ece /= nf;
    let uncertainty = accuracy * (1.0 - accuracy);

    Some(CalibrationReport {
        n,
        brier,
        reliability,
        resolution,
        uncertainty,
        ece,
        accuracy,
        accuracy_ci95: wilson_interval(accuracy, n, 0.95),
    })
}

/// Wilson score interval for a binomial proportion — better than the normal
/// approximation for small `n` or extreme `p`. Uses the statistics crate's
/// inverse-normal quantile for the z-value.
pub fn wilson_interval(p: f64, n: usize, confidence: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let nf = n as f64;
    let z = normal_quantile(0.5 + confidence / 2.0);
    let denom = 1.0 + z * z / nf;
    let center = (p + z * z / (2.0 * nf)) / denom;
    let margin = (z / denom) * (p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt();
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn murphy_decomposition_identity() {
        // Two forecasts at 0.8 (1 of 2 correct) and 0.6 (2 of 2 correct).
        let data = [(0.8, true), (0.8, false), (0.6, true), (0.6, true)];
        let r = analyze_pairs(&data, 10).unwrap();
        assert!((r.brier - 0.25).abs() < 1e-12, "brier {}", r.brier);
        assert!(
            (r.reliability - 0.125).abs() < 1e-12,
            "rel {}",
            r.reliability
        );
        assert!(
            (r.resolution - 0.0625).abs() < 1e-12,
            "res {}",
            r.resolution
        );
        assert!(
            (r.uncertainty - 0.1875).abs() < 1e-12,
            "unc {}",
            r.uncertainty
        );
        // Brier = Reliability − Resolution + Uncertainty (exact, pure bins).
        assert!((r.brier - (r.reliability - r.resolution + r.uncertainty)).abs() < 1e-12);
        assert!((r.ece - 0.35).abs() < 1e-12, "ece {}", r.ece);
    }

    #[test]
    fn perfect_calibration_zero_reliability() {
        // Confidence exactly equals outcome frequency in every bin.
        let data = [(1.0, true), (1.0, true), (0.0, false), (0.0, false)];
        let r = analyze_pairs(&data, 10).unwrap();
        assert!(r.reliability.abs() < 1e-12, "rel {}", r.reliability);
        assert!(r.brier.abs() < 1e-12, "brier {}", r.brier); // perfect predictions
        assert!((r.accuracy - 0.5).abs() < 1e-12);
    }

    #[test]
    fn wilson_interval_contains_and_narrows() {
        // 80/100 correct → 95% CI roughly (0.71, 0.87), and contains 0.8.
        let (lo, hi) = wilson_interval(0.8, 100, 0.95);
        assert!(lo < 0.8 && 0.8 < hi, "({lo}, {hi})");
        assert!(lo > 0.70 && lo < 0.73, "lo {lo}");
        assert!(hi > 0.86 && hi < 0.88, "hi {hi}");
        // More data → tighter interval.
        let (lo2, hi2) = wilson_interval(0.8, 1000, 0.95);
        assert!((hi2 - lo2) < (hi - lo));
    }

    #[test]
    fn empty_is_none() {
        assert!(analyze_pairs(&[], 10).is_none());
        assert!(analyze_pairs(&[(0.5, true)], 0).is_none());
    }
}
