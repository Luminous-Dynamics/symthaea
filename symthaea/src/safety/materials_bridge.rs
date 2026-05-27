// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Materials Aging → Safety Agent
//!
//! Maps `AgingPrediction` (remaining strength, state similarity) into
//! `SafetyMetrics` so the Safety Agent can monitor structural integrity
//! through the same NRC-grade framework used for consciousness monitoring.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `materials` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
use symthaea_materials::AgingPrediction;

/// Adapter that translates material aging predictions into Safety Agent inputs.
///
/// Enables unified safety monitoring of structural integrity alongside
/// consciousness and other domains. A degrading material (low remaining
/// strength, high state drift) maps to elevated safety levels.
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

    /// Convert an `AgingPrediction` into `SafetyMetrics`.
    ///
    /// Mapping:
    /// - `consciousness_level` ← remaining_strength (structural integrity proxy)
    /// - `prediction_error` ← 1.0 - state_similarity (higher drift = higher error)
    /// - `temporal_coherence` ← state_similarity (how stable predictions are)
    pub fn to_safety_metrics(&self, prediction: &AgingPrediction) -> SafetyMetrics {
        let remaining = if prediction.remaining_strength.is_finite() {
            prediction.remaining_strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let similarity = if prediction.state_similarity.is_finite() {
            prediction.state_similarity.clamp(0.0, 1.0)
        } else {
            0.0
        };

        SafetyMetrics {
            cycle: self.cycle,
            consciousness_level: remaining,
            prediction_error: 1.0 - similarity,
            temporal_coherence: similarity,
            integrity_critical: false,
        }
    }

    /// Assess a material aging prediction through the safety agent.
    pub fn assess(&mut self, prediction: &AgingPrediction) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(prediction);
        self.agent.assess(metrics)
    }

    /// Check whether a structural operation should proceed given predicted aging.
    ///
    /// Maps remaining_strength to safety level, then applies the safety gate.
    pub fn gate_operation(&self, prediction: &AgingPrediction, is_risky: bool) -> SafetyGateResult {
        let level = remaining_strength_to_safety_level(prediction.remaining_strength);
        safety_gate(level, is_risky)
    }

    /// Whether the predicted state is critically degraded.
    pub fn is_critical(prediction: &AgingPrediction) -> bool {
        prediction.remaining_strength < 0.3
    }

    /// Access the inner safety agent.
    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for MaterialSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map remaining structural strength to an NRC safety level.
pub fn remaining_strength_to_safety_level(remaining_strength: f32) -> SafetyLevel {
    let s = if remaining_strength.is_finite() {
        remaining_strength
    } else {
        0.0
    };
    if s >= 0.6 {
        SafetyLevel::Green
    } else if s >= 0.35 {
        SafetyLevel::Yellow
    } else if s >= 0.15 {
        SafetyLevel::Orange
    } else {
        SafetyLevel::Red
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    fn mock_prediction(remaining_strength: f32, state_similarity: f32) -> AgingPrediction {
        AgingPrediction {
            horizon_seconds: 31_536_000.0,
            horizon_label: "1 year".to_string(),
            predicted_state: ContinuousHV::random(16_384, 0xBEEF),
            state_similarity,
            remaining_strength,
        }
    }

    #[test]
    fn test_healthy_material_maps_to_green() {
        let mut adapter = MaterialSafetyAdapter::new();
        let pred = mock_prediction(0.95, 0.98);
        let assessment = adapter.assess(&pred);
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_degraded_material_escalates() {
        let mut adapter = MaterialSafetyAdapter::new();
        let pred = mock_prediction(0.25, 0.4);
        let assessment = adapter.assess(&pred);
        assert!(assessment.level >= SafetyLevel::Orange);
    }

    #[test]
    fn test_critically_degraded_is_red() {
        let mut adapter = MaterialSafetyAdapter::new();
        let pred = mock_prediction(0.05, 0.1);
        let assessment = adapter.assess(&pred);
        assert_eq!(assessment.level, SafetyLevel::Red);
    }

    #[test]
    fn test_remaining_strength_to_safety_level() {
        assert_eq!(remaining_strength_to_safety_level(0.9), SafetyLevel::Green);
        assert_eq!(remaining_strength_to_safety_level(0.5), SafetyLevel::Yellow);
        assert_eq!(
            remaining_strength_to_safety_level(0.25),
            SafetyLevel::Orange
        );
        assert_eq!(remaining_strength_to_safety_level(0.1), SafetyLevel::Red);
    }

    #[test]
    fn test_remaining_strength_nan_maps_to_red() {
        assert_eq!(
            remaining_strength_to_safety_level(f32::NAN),
            SafetyLevel::Red
        );
    }

    #[test]
    fn test_gate_blocks_risky_at_orange() {
        let adapter = MaterialSafetyAdapter::new();
        let pred = mock_prediction(0.25, 0.4);
        assert!(!adapter.gate_operation(&pred, true).is_ok());
        assert!(adapter.gate_operation(&pred, false).is_ok());
    }

    #[test]
    fn test_gate_blocks_all_at_red() {
        let adapter = MaterialSafetyAdapter::new();
        let pred = mock_prediction(0.05, 0.1);
        assert!(!adapter.gate_operation(&pred, false).is_ok());
    }

    #[test]
    fn test_is_critical() {
        assert!(MaterialSafetyAdapter::is_critical(&mock_prediction(
            0.2, 0.3
        )));
        assert!(!MaterialSafetyAdapter::is_critical(&mock_prediction(
            0.5, 0.7
        )));
    }

    #[test]
    fn test_escalation_sequence() {
        let mut adapter = MaterialSafetyAdapter::new();
        // Simulate material degradation over time
        let a1 = adapter.assess(&mock_prediction(0.9, 0.95));
        assert_eq!(a1.level, SafetyLevel::Green);

        let a2 = adapter.assess(&mock_prediction(0.4, 0.6));
        assert!(a2.level >= SafetyLevel::Yellow);

        let a3 = adapter.assess(&mock_prediction(0.1, 0.2));
        assert!(a3.level >= SafetyLevel::Orange);

        assert_eq!(adapter.agent().history().len(), 3);
    }

    #[test]
    fn test_metrics_nan_clamped() {
        let adapter = MaterialSafetyAdapter::new();
        let pred = mock_prediction(f32::NAN, f32::NAN);
        let metrics = adapter.to_safety_metrics(&pred);
        assert_eq!(metrics.consciousness_level, 0.0);
        assert_eq!(metrics.prediction_error, 1.0);
        assert_eq!(metrics.temporal_coherence, 0.0);
    }
}
