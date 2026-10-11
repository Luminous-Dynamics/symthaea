// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compatibility quarantine: Materials representation drift → Safety Agent.
//!
//! `MaterialAgingModel` currently produces an HDC/CfC representation-drift
//! heuristic. It does **not** produce calibrated residual strength, fatigue
//! life, damage, or a qualified structural-health estimate. This bridge is
//! therefore deliberately fail-closed: research drift may be observed, but it
//! cannot authorize a structural operation or mint a positive physical safety
//! level.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `materials` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::SafetyGateResult;
use symthaea_materials::AgingPrediction;

/// Claim ceiling of this compatibility bridge.
pub const MATERIAL_AGING_SAFETY_CLAIM_CLASS: &str =
    "research-heuristic-diagnostic-only-no-physical-authority";

/// Adapter that retains material representation-drift diagnostics while
/// refusing to translate them into positive physical safety authority.
///
/// This type exists as a compatibility quarantine for the previous bridge.
/// A future replacement should consume an evidence-bearing physical
/// degradation/health model with exact applicability and validation.
pub struct MaterialSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl MaterialSafetyAdapter {
    /// Create a new adapter with a fresh Safety Agent.
    pub fn new() -> Self {
        Self {
            agent: SafetyAgent::new(),
            cycle: 0,
        }
    }

    /// Create with a custom Safety Agent.
    pub fn with_agent(agent: SafetyAgent) -> Self {
        Self { agent, cycle: 0 }
    }

    /// Convert a research `AgingPrediction` into diagnostic `SafetyMetrics`.
    ///
    /// The HDC similarity is retained only as research drift telemetry:
    /// - `prediction_error` ← `1 - state_similarity`;
    /// - `temporal_coherence` ← `state_similarity`.
    ///
    /// `consciousness_level` is deliberately set to the conservative floor and
    /// `integrity_critical` is set because this adapter lacks a qualified
    /// physical mapping. Consequently this conversion cannot yield a positive
    /// structural-safety interpretation.
    pub fn to_safety_metrics(&self, prediction: &AgingPrediction) -> SafetyMetrics {
        let similarity = if prediction.state_similarity.is_finite() {
            prediction.state_similarity.clamp(0.0, 1.0)
        } else {
            0.0
        };

        SafetyMetrics {
            cycle: self.cycle,
            consciousness_level: 0.0,
            prediction_error: 1.0 - similarity,
            temporal_coherence: similarity,
            integrity_critical: true,
        }
    }

    /// Assess the research drift through the existing safety-agent telemetry
    /// path. The result is fail-closed and must not be interpreted as a
    /// calibrated material safety assessment.
    pub fn assess(&mut self, prediction: &AgingPrediction) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(prediction);
        self.agent.assess(metrics)
    }

    /// Refuse structural operation authority from an unqualified research
    /// aging heuristic.
    ///
    /// `is_risky` is intentionally ignored: without a qualified physical
    /// mapping, neither risky nor nominal structural operation may be admitted
    /// by this bridge.
    pub fn gate_operation(
        &self,
        _prediction: &AgingPrediction,
        _is_risky: bool,
    ) -> SafetyGateResult {
        SafetyGateResult::Blocked {
            level: SafetyLevel::Red,
            reason: "Material aging HDC/CfC output is a research representation-drift heuristic; no qualified physical degradation mapping is available".to_string(),
        }
    }

    /// Compatibility predicate for callers that previously asked whether the
    /// prediction was physically critical.
    ///
    /// Because the physical proposition is not established, this returns true
    /// fail-closed for every research-only prediction. It does not claim the
    /// material is actually damaged; it states that positive physical safety
    /// is not established by this evidence class.
    pub fn is_critical(_prediction: &AgingPrediction) -> bool {
        true
    }

    /// Access the inner safety agent.
    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }

    /// Return the exact claim ceiling of this bridge.
    pub const fn claim_class(&self) -> &'static str {
        MATERIAL_AGING_SAFETY_CLAIM_CLASS
    }
}

impl Default for MaterialSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Legacy compatibility helper retained fail-closed.
///
/// There is currently no qualified mapping from the research aging model to a
/// physical residual-strength fraction, so no numeric input can mint a positive
/// safety level through this helper. A future physical-health API should use a
/// distinct evidence-bearing type rather than weakening this quarantine.
pub fn remaining_strength_to_safety_level(_remaining_strength: f32) -> SafetyLevel {
    SafetyLevel::Red
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    fn mock_prediction(state_similarity: f32) -> AgingPrediction {
        AgingPrediction {
            horizon_seconds: 31_536_000.0,
            horizon_label: "1 year".to_string(),
            predicted_state: ContinuousHV::random(16_384, 0xBEEF),
            state_similarity,
            remaining_strength: 1.0,
        }
    }

    #[test]
    fn test_research_prediction_cannot_map_to_green() {
        let mut adapter = MaterialSafetyAdapter::new();
        let assessment = adapter.assess(&mock_prediction(0.98));
        assert_eq!(assessment.level, SafetyLevel::Red);
        assert_eq!(assessment.raw_level, SafetyLevel::Red);
    }

    #[test]
    fn test_low_similarity_is_also_fail_closed() {
        let mut adapter = MaterialSafetyAdapter::new();
        let assessment = adapter.assess(&mock_prediction(0.1));
        assert_eq!(assessment.level, SafetyLevel::Red);
    }

    #[test]
    fn test_legacy_strength_mapping_never_mints_positive_level() {
        assert_eq!(remaining_strength_to_safety_level(1.0), SafetyLevel::Red);
        assert_eq!(remaining_strength_to_safety_level(0.9), SafetyLevel::Red);
        assert_eq!(remaining_strength_to_safety_level(0.5), SafetyLevel::Red);
        assert_eq!(remaining_strength_to_safety_level(0.1), SafetyLevel::Red);
        assert_eq!(remaining_strength_to_safety_level(f32::NAN), SafetyLevel::Red);
    }

    #[test]
    fn test_gate_blocks_risky_and_nominal_operations() {
        let adapter = MaterialSafetyAdapter::new();
        let pred = mock_prediction(0.95);
        assert!(!adapter.gate_operation(&pred, true).is_ok());
        assert!(!adapter.gate_operation(&pred, false).is_ok());
    }

    #[test]
    fn test_is_critical_means_physical_safety_not_established() {
        assert!(MaterialSafetyAdapter::is_critical(&mock_prediction(0.99)));
        assert!(MaterialSafetyAdapter::is_critical(&mock_prediction(0.2)));
    }

    #[test]
    fn test_metrics_preserve_drift_only_and_force_fail_closed_integrity() {
        let adapter = MaterialSafetyAdapter::new();
        let pred = mock_prediction(0.8);
        let metrics = adapter.to_safety_metrics(&pred);
        assert_eq!(metrics.consciousness_level, 0.0);
        assert!((metrics.prediction_error - 0.2).abs() < 1.0e-6);
        assert!((metrics.temporal_coherence - 0.8).abs() < 1.0e-6);
        assert!(metrics.integrity_critical);
    }

    #[test]
    fn test_nonfinite_similarity_remains_fail_closed() {
        let adapter = MaterialSafetyAdapter::new();
        let pred = mock_prediction(f32::NAN);
        let metrics = adapter.to_safety_metrics(&pred);
        assert_eq!(metrics.consciousness_level, 0.0);
        assert_eq!(metrics.prediction_error, 1.0);
        assert_eq!(metrics.temporal_coherence, 0.0);
        assert!(metrics.integrity_critical);
    }

    #[test]
    fn test_claim_class_is_explicit() {
        let adapter = MaterialSafetyAdapter::new();
        assert_eq!(adapter.claim_class(), MATERIAL_AGING_SAFETY_CLAIM_CLASS);
    }
}
