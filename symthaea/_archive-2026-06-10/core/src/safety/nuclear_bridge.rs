// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Nuclear Attribution → Safety Agent
//!
//! Maps `AttributionResult` (confidence, source match) and `BackdatedResult`
//! (state drift) into `SafetyMetrics` so the Safety Agent can monitor
//! nuclear material provenance through the same NRC-grade framework.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `nuclear-forensics` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
use symthaea_nuclear_forensics::{AttributionResult, BackdatedResult};

/// Input bundle for a single nuclear safety assessment.
///
/// Combines an attribution result with an optional temporal backdating
/// result to give the adapter both provenance and temporal context.
pub struct NuclearSafetyInput {
    /// Attribution result from the NuclearAttributionAgent.
    pub attribution: AttributionResult,
    /// Optional backdating result for temporal coherence.
    pub backdating: Option<BackdatedResult>,
}

/// Adapter that translates nuclear attribution results into Safety Agent inputs.
///
/// Enables unified safety monitoring of nuclear material provenance alongside
/// consciousness and other domains. Low attribution confidence (ambiguous
/// source) and high temporal drift map to elevated safety levels.
pub struct NuclearSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl NuclearSafetyAdapter {
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

    /// Convert a `NuclearSafetyInput` into `SafetyMetrics`.
    ///
    /// Mapping:
    /// - `consciousness_level` ← attribution confidence (high confidence = "aware" of source)
    /// - `prediction_error` ← 1.0 - confidence (low confidence = high uncertainty)
    /// - `temporal_coherence` ← 1.0 - state_drift (from backdating, if available)
    pub fn to_safety_metrics(&self, input: &NuclearSafetyInput) -> SafetyMetrics {
        let confidence = if input.attribution.confidence.is_finite() {
            input.attribution.confidence.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let temporal_coherence = match &input.backdating {
            Some(bd) => {
                let drift = if bd.state_drift.is_finite() {
                    bd.state_drift.clamp(0.0, 1.0)
                } else {
                    1.0
                };
                1.0 - drift
            }
            None => 0.5, // no temporal data → neutral
        };

        SafetyMetrics {
            cycle: self.cycle,
            consciousness_level: confidence,
            prediction_error: 1.0 - confidence,
            temporal_coherence: temporal_coherence.clamp(0.0, 1.0),
            integrity_critical: false,
        }
    }

    /// Assess a nuclear attribution through the safety agent.
    pub fn assess(&mut self, input: &NuclearSafetyInput) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(input);
        self.agent.assess(metrics)
    }

    /// Check whether a nuclear operation should proceed given attribution confidence.
    ///
    /// Maps confidence to safety level, then applies the safety gate.
    pub fn gate_operation(&self, input: &NuclearSafetyInput, is_risky: bool) -> SafetyGateResult {
        let level = confidence_to_safety_level(input.attribution.confidence);
        safety_gate(level, is_risky)
    }

    /// Whether the attribution is suspicious (low confidence in source identification).
    pub fn is_suspicious(input: &NuclearSafetyInput) -> bool {
        input.attribution.confidence < 0.5
    }

    /// Access the inner safety agent.
    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for NuclearSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map attribution confidence to an NRC safety level.
///
/// Low confidence means the sample's origin is uncertain — higher risk.
pub fn confidence_to_safety_level(confidence: f32) -> SafetyLevel {
    let c = if confidence.is_finite() {
        confidence
    } else {
        0.0
    };
    if c >= 0.6 {
        SafetyLevel::Green
    } else if c >= 0.35 {
        SafetyLevel::Yellow
    } else if c >= 0.15 {
        SafetyLevel::Orange
    } else {
        SafetyLevel::Red
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_nuclear_forensics::{IsotopicSignature, NuclearAttributionAgent};

    fn high_confidence_input() -> NuclearSafetyInput {
        let agent = NuclearAttributionAgent::new();
        let attr = agent.attribute(&IsotopicSignature::highly_enriched_uranium());
        NuclearSafetyInput {
            attribution: attr,
            backdating: None,
        }
    }

    fn low_confidence_input() -> NuclearSafetyInput {
        // Create a maximally perturbed sample for low confidence
        let agent = NuclearAttributionAgent::new();
        let mut sig = IsotopicSignature::natural_uranium();
        // Scramble values to confuse attribution
        sig.u235_u238 = 0.5;
        sig.pu239_pu240 = 0.3;
        sig.pu241_pu239 = 0.4;
        let attr = agent.attribute(&sig);
        NuclearSafetyInput {
            attribution: attr,
            backdating: None,
        }
    }

    #[test]
    fn test_high_confidence_maps_to_green() {
        let mut adapter = NuclearSafetyAdapter::new();
        let assessment = adapter.assess(&high_confidence_input());
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_confidence_to_safety_level() {
        assert_eq!(confidence_to_safety_level(0.9), SafetyLevel::Green);
        assert_eq!(confidence_to_safety_level(0.5), SafetyLevel::Yellow);
        assert_eq!(confidence_to_safety_level(0.25), SafetyLevel::Orange);
        assert_eq!(confidence_to_safety_level(0.1), SafetyLevel::Red);
    }

    #[test]
    fn test_confidence_nan_maps_to_red() {
        assert_eq!(confidence_to_safety_level(f32::NAN), SafetyLevel::Red);
    }

    #[test]
    fn test_gate_allows_high_confidence() {
        let adapter = NuclearSafetyAdapter::new();
        let input = high_confidence_input();
        assert!(adapter.gate_operation(&input, true).is_ok());
    }

    #[test]
    fn test_is_suspicious() {
        assert!(!NuclearSafetyAdapter::is_suspicious(
            &high_confidence_input()
        ));
    }

    #[test]
    fn test_metrics_with_backdating() {
        let adapter = NuclearSafetyAdapter::new();
        let agent = NuclearAttributionAgent::new();
        let attr = agent.attribute(&IsotopicSignature::spent_fuel());
        let decay = symthaea_nuclear_forensics::IsotopeDecayModel::new();
        let bd = decay.predict_at_horizon(&IsotopicSignature::spent_fuel(), 86_400.0);
        let input = NuclearSafetyInput {
            attribution: attr,
            backdating: Some(bd),
        };
        let metrics = adapter.to_safety_metrics(&input);
        assert!(metrics.temporal_coherence.is_finite());
        assert!(metrics.temporal_coherence >= 0.0 && metrics.temporal_coherence <= 1.0);
    }

    #[test]
    fn test_metrics_without_backdating() {
        let adapter = NuclearSafetyAdapter::new();
        let input = high_confidence_input();
        let metrics = adapter.to_safety_metrics(&input);
        assert_eq!(metrics.temporal_coherence, 0.5); // neutral default
    }

    #[test]
    fn test_escalation_sequence() {
        let mut adapter = NuclearSafetyAdapter::new();
        let a1 = adapter.assess(&high_confidence_input());
        assert_eq!(a1.level, SafetyLevel::Green);

        assert_eq!(adapter.agent().history().len(), 1);
    }

    #[test]
    fn test_metrics_nan_clamped() {
        let adapter = NuclearSafetyAdapter::new();
        let input = NuclearSafetyInput {
            attribution: AttributionResult {
                matched_source: symthaea_nuclear_forensics::NuclearSource::NaturalUranium,
                matched_name: "test".to_string(),
                confidence: f32::NAN,
                all_similarities: vec![],
            },
            backdating: None,
        };
        let metrics = adapter.to_safety_metrics(&input);
        assert_eq!(metrics.consciousness_level, 0.0);
        assert_eq!(metrics.prediction_error, 1.0);
    }
}
