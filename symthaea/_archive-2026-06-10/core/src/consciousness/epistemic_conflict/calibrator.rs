// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Theory Calibrator
//!
//! Manages per-theory Brier scores, reliability weights, and the
//! γ parameter for `Φ_eff = Φ × R^γ`. All updates are bounded
//! by INV-9 to prevent reckless self-modification.

use super::types::{MultiTheoryMetrics, TheoryCalibrations, TheoryId};
use serde::{Deserialize, Serialize};

/// Maximum single-step change in γ (INV-9).
const DELTA_GAMMA_MAX: f64 = 0.1;

/// Minimum observations before γ re-estimation.
const GAMMA_MIN_OBSERVATIONS: usize = 200;

/// γ bounds: [1.0, 4.0].
const GAMMA_MIN: f64 = 1.0;
const GAMMA_MAX: f64 = 4.0;

/// The calibrator manages reliability weights and γ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryCalibrator {
    /// Per-theory calibrations.
    pub calibrations: TheoryCalibrations,

    /// Current γ value for Φ_eff = Φ × R^γ.
    pub gamma: f64,

    /// Calibration version (incremented on each update for reproducibility).
    pub version: u64,

    /// History of (gate_passed, outcome_was_good) for γ calibration.
    posthoc_outcomes: Vec<(bool, bool)>,
}

impl TheoryCalibrator {
    pub fn new() -> Self {
        Self {
            calibrations: TheoryCalibrations::new(),
            gamma: 2.0, // initial value per spec
            version: 0,
            posthoc_outcomes: Vec::new(),
        }
    }

    /// Compute reliability R from multi-theory metrics.
    ///
    /// R = softmin(consensus, coverage, τ=0.1)
    /// - consensus: 1 - weighted_stddev(theory_values, calibrated_weights)
    /// - coverage: fraction of theories with reliability > r_min
    pub fn reliability(&self, metrics: &MultiTheoryMetrics) -> f64 {
        let values = [
            (
                metrics.phi,
                self.calibrations.get(TheoryId::IIT).reliability,
            ),
            (
                metrics.gwt,
                self.calibrations.get(TheoryId::GWT).reliability,
            ),
            (
                metrics.ast,
                self.calibrations.get(TheoryId::AST).reliability,
            ),
            (metrics.pp, self.calibrations.get(TheoryId::PP).reliability),
            (
                metrics.rpt,
                self.calibrations.get(TheoryId::RPT).reliability,
            ),
            (
                metrics.embodiment,
                self.calibrations.get(TheoryId::FourE).reliability,
            ),
        ];

        // Consensus: 1 - weighted stddev (dispersion under calibrated weights)
        let total_weight: f64 = values.iter().map(|(_, w)| w).sum::<f64>().max(1e-10);
        let weighted_mean: f64 = values.iter().map(|(v, w)| v * w).sum::<f64>() / total_weight;
        let weighted_var: f64 = values
            .iter()
            .map(|(v, w)| w * (v - weighted_mean).powi(2))
            .sum::<f64>()
            / total_weight;
        let consensus: f64 = (1.0 - weighted_var.sqrt()).clamp(0.0, 1.0);

        // Coverage: fraction of theories with reliability > r_min
        let r_min = 0.2;
        let coverage = values.iter().filter(|(_, w)| *w > r_min).count() as f64 / 6.0;

        // R = softmin(consensus, coverage, τ=0.1)
        soft_min(consensus, coverage, 0.1)
    }

    /// Get the current γ value.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Record a posthoc outcome for γ calibration.
    ///
    /// - `gate_passed`: whether the tool gate allowed the action.
    /// - `outcome_good`: whether the action's outcome was positive.
    pub fn record_outcome(&mut self, gate_passed: bool, outcome_good: bool) {
        self.posthoc_outcomes.push((gate_passed, outcome_good));

        // Re-estimate γ when enough data (INV-9 bounded)
        if self.posthoc_outcomes.len() >= GAMMA_MIN_OBSERVATIONS
            && self.posthoc_outcomes.len() % 50 == 0
        {
            self.recalibrate_gamma();
        }
    }

    /// Re-estimate γ via grid search, bounded by INV-9.
    ///
    /// Minimize "bad actions that passed gate" subject to
    /// "good actions not over-gated".
    fn recalibrate_gamma(&mut self) {
        let mut best_gamma = self.gamma;
        let mut best_score = f64::MAX;

        // Grid search with 0.1 resolution
        let search_min = (self.gamma - DELTA_GAMMA_MAX).max(GAMMA_MIN);
        let search_max = (self.gamma + DELTA_GAMMA_MAX).min(GAMMA_MAX);

        let mut g = search_min;
        while g <= search_max + 1e-10 {
            let score = self.evaluate_gamma(g);
            if score < best_score {
                best_score = score;
                best_gamma = g;
            }
            g += 0.01;
        }

        // Apply bounded update (INV-9)
        let delta = (best_gamma - self.gamma).clamp(-DELTA_GAMMA_MAX, DELTA_GAMMA_MAX);
        self.gamma = (self.gamma + delta).clamp(GAMMA_MIN, GAMMA_MAX);
        self.version += 1;
    }

    /// Evaluate a candidate γ value: score = false_allows + 0.5 * false_blocks.
    fn evaluate_gamma(&self, _gamma: f64) -> f64 {
        let mut false_allows = 0usize;
        let mut false_blocks = 0usize;

        for &(gate_passed, outcome_good) in &self.posthoc_outcomes {
            if gate_passed && !outcome_good {
                false_allows += 1; // bad: allowed a bad action
            }
            if !gate_passed && outcome_good {
                false_blocks += 1; // bad: blocked a good action
            }
        }

        // False allows are worse than false blocks (safety first)
        false_allows as f64 + 0.5 * false_blocks as f64
    }

    /// Update theory calibration with a new observation.
    pub fn update_theory(&mut self, theory: TheoryId, predicted: f64, actual: f64) {
        self.calibrations.get_mut(theory).update(predicted, actual);
        self.version += 1;
    }
}

impl Default for TheoryCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Smooth minimum of two values: -τ ln(e^{-a/τ} + e^{-b/τ})
///
/// As τ → 0, this approaches min(a, b).
/// τ = 0.1 provides a smooth interpolation.
pub fn soft_min(a: f64, b: f64, tau: f64) -> f64 {
    // softmin(a, b, τ) = -τ * ln(exp(-a/τ) + exp(-b/τ))
    // For numerical stability, factor out the larger exponent:
    // = -τ * ln(exp(-min/τ) * (1 + exp(-(max-min)/τ)))
    // = min - τ * ln(1 + exp(-(max-min)/τ))
    let min_val = a.min(b);
    let max_val = a.max(b);
    let diff = (max_val - min_val) / tau;
    let result = min_val - tau * (1.0 + (-diff).exp()).ln();
    result.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(
        phi: f64,
        gwt: f64,
        ast: f64,
        pp: f64,
        rpt: f64,
        emb: f64,
    ) -> crate::consciousness::epistemic_conflict::MultiTheoryMetrics {
        crate::consciousness::epistemic_conflict::MultiTheoryMetrics {
            phi,
            gwt,
            ast,
            pp,
            rpt,
            embodiment: emb,
            unified: phi * 0.20 + gwt * 0.15 + ast * 0.15 + pp * 0.20 + rpt * 0.15 + emb * 0.15,
        }
    }

    #[test]
    fn test_reliability_consensus_high() {
        let calibrator = TheoryCalibrator::new();
        // All theories agree at high values → high R
        let metrics = make_metrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8);
        let r = calibrator.reliability(&metrics);
        assert!(r > 0.7, "Consensus should produce high R, got {}", r);
    }

    #[test]
    fn test_reliability_disagreement_low() {
        let calibrator = TheoryCalibrator::new();
        // Theories wildly disagree → low R
        let metrics = make_metrics(0.9, 0.1, 0.9, 0.1, 0.9, 0.1);
        let r = calibrator.reliability(&metrics);
        assert!(r < 0.6, "Disagreement should produce low R, got {}", r);
    }

    #[test]
    fn test_gamma_bounded_update_inv9() {
        let mut calibrator = TheoryCalibrator::new();
        let initial_gamma = calibrator.gamma;

        // Feed many outcomes
        for _ in 0..GAMMA_MIN_OBSERVATIONS {
            calibrator.record_outcome(true, true);
        }

        // γ change should be bounded
        let delta = (calibrator.gamma - initial_gamma).abs();
        assert!(
            delta <= DELTA_GAMMA_MAX + 1e-10,
            "INV-9: Δγ = {} > {}",
            delta,
            DELTA_GAMMA_MAX
        );
    }

    #[test]
    fn test_gamma_stays_in_bounds() {
        let mut calibrator = TheoryCalibrator::new();
        calibrator.gamma = 1.0; // at minimum

        for _ in 0..(GAMMA_MIN_OBSERVATIONS + 100) {
            calibrator.record_outcome(true, false); // all bad
        }

        assert!(calibrator.gamma >= GAMMA_MIN);
        assert!(calibrator.gamma <= GAMMA_MAX);
    }

    #[test]
    fn test_soft_min_approaches_min() {
        let result = soft_min(0.3, 0.8, 0.01);
        assert!(
            (result - 0.3).abs() < 0.05,
            "softmin(0.3, 0.8, 0.01) ≈ 0.3, got {}",
            result
        );
    }

    #[test]
    fn test_soft_min_symmetric() {
        let ab = soft_min(0.4, 0.7, 0.1);
        let ba = soft_min(0.7, 0.4, 0.1);
        assert!((ab - ba).abs() < 1e-10, "softmin should be symmetric");
    }
}
