// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Theory Calibrator
//!
//! Manages per-theory Brier/reliability calibration and the γ parameter for
//! `Φ_eff = Φ × R^γ`.
//!
//! ## Gamma authority boundary
//!
//! Posthoc history currently records only `(gate_passed, outcome_was_good)`. That is sufficient
//! for descriptive gate-outcome telemetry, but it cannot identify what the gate *would have done*
//! under a different γ. Candidate-γ fitting therefore remains disabled until replayable decision
//! context (or another independently justified calibration objective) is available and qualified.
//! This is intentionally safer than applying a numerically bounded but causally unidentified
//! self-update.

use super::types::{MultiTheoryMetrics, TheoryCalibrations, TheoryId};
use serde::{Deserialize, Serialize};

/// Current configured γ prior. Automatic adaptation is intentionally frozen; see module docs.
const DEFAULT_GAMMA: f64 = 2.0;

/// The calibrator manages per-theory reliability weights and the current γ setting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryCalibrator {
    /// Per-theory calibrations.
    pub calibrations: TheoryCalibrations,

    /// Current γ value for Φ_eff = Φ × R^γ.
    ///
    /// This remains public for backward compatibility and explicit operator/configuration control,
    /// but `record_outcome` does not mutate it from non-replayable posthoc history.
    pub gamma: f64,

    /// Calibration version, incremented only when an actual calibrated parameter changes.
    pub version: u64,

    /// Descriptive history of (gate_passed, outcome_was_good).
    ///
    /// This history is deliberately *not* treated as sufficient evidence for counterfactual γ
    /// optimization because it lacks the decision context needed to replay alternative γ values.
    posthoc_outcomes: Vec<(bool, bool)>,
}

impl TheoryCalibrator {
    pub fn new() -> Self {
        Self {
            calibrations: TheoryCalibrations::new(),
            gamma: DEFAULT_GAMMA,
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

    /// Total number of posthoc outcomes ever recorded (lifetime, unbounded —
    /// `posthoc_outcomes` is never truncated). Used as `ToolDescriptor.calibration_count`.
    pub fn posthoc_count(&self) -> usize {
        self.posthoc_outcomes.len()
    }

    /// Record a posthoc gate outcome for telemetry/counting.
    ///
    /// - `gate_passed`: whether the tool gate allowed the action.
    /// - `outcome_good`: whether the action's eventual outcome was positive.
    ///
    /// ## Why this does not update γ
    ///
    /// These two booleans describe the decision made under the *already active* γ. They do not
    /// contain enough information to determine whether another candidate γ would have changed the
    /// gate decision. Updating γ from this history would therefore be an unidentified
    /// counterfactual. The history is retained so future replay-capable calibration can migrate
    /// without losing telemetry, but automatic γ adaptation is frozen until that richer evidence
    /// contract is implemented and qualified.
    pub fn record_outcome(&mut self, gate_passed: bool, outcome_good: bool) {
        self.posthoc_outcomes.push((gate_passed, outcome_good));
    }

    /// Update one theory calibration with an observed prediction/outcome pair.
    ///
    /// This path remains adaptive because the update is directly identified by the supplied
    /// `(predicted, actual)` observation and is independently bounded inside `TheoryCalibration`.
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
    fn posthoc_all_good_cannot_move_unidentified_gamma() {
        let mut calibrator = TheoryCalibrator::new();
        let initial_gamma = calibrator.gamma;
        let initial_version = calibrator.version;

        for _ in 0..500 {
            calibrator.record_outcome(true, true);
        }

        assert_eq!(calibrator.gamma, initial_gamma);
        assert_eq!(calibrator.version, initial_version);
        assert_eq!(calibrator.posthoc_count(), 500);
    }

    #[test]
    fn posthoc_all_bad_cannot_move_unidentified_gamma() {
        let mut calibrator = TheoryCalibrator::new();
        let initial_gamma = calibrator.gamma;

        for _ in 0..500 {
            calibrator.record_outcome(true, false);
        }

        assert_eq!(calibrator.gamma, initial_gamma);
        assert_eq!(calibrator.posthoc_count(), 500);
    }

    #[test]
    fn posthoc_mixed_history_cannot_move_unidentified_gamma() {
        let mut calibrator = TheoryCalibrator::new();
        let initial_gamma = calibrator.gamma;

        for i in 0..500 {
            calibrator.record_outcome(i % 2 == 0, i % 3 == 0);
        }

        assert_eq!(calibrator.gamma, initial_gamma);
        assert_eq!(calibrator.posthoc_count(), 500);
    }

    #[test]
    fn identified_theory_update_still_advances_calibration_version() {
        let mut calibrator = TheoryCalibrator::new();
        let initial_version = calibrator.version;
        calibrator.update_theory(TheoryId::IIT, 0.8, 1.0);
        assert_eq!(calibrator.version, initial_version + 1);
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
