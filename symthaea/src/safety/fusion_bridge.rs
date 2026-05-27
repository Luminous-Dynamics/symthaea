// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Fusion Digital Twin → Safety Agent
//!
//! Maps `FusionTwinOutput` (disruption risk, free energy, action) into
//! `SafetyMetrics` so the Safety Agent can monitor fusion operations
//! through the same NRC-grade framework used for consciousness monitoring.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `fusion-twin` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
use symthaea_physics::fusion_twin::{DisruptionRisk, FusionTwinOutput, PlasmaFepAction};

/// Adapter that translates Fusion Digital Twin outputs into Safety Agent inputs.
///
/// Enables unified safety monitoring across consciousness + fusion domains.
/// A single `SafetyAgent` can consume both consciousness snapshots AND
/// fusion twin outputs, producing a combined safety assessment.
pub struct FusionSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl FusionSafetyAdapter {
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

    /// Convert a `FusionTwinOutput` into `SafetyMetrics`.
    ///
    /// Mapping:
    /// - `consciousness_level` ← 1.0 - free_energy (higher FE = less "conscious" of plasma state)
    /// - `prediction_error` ← free_energy (direct surprise measure)
    /// - `temporal_coherence` ← mean prediction similarity across horizons
    pub fn to_safety_metrics(&self, output: &FusionTwinOutput) -> SafetyMetrics {
        let temporal_coherence = if output.prediction_similarities.is_empty() {
            0.0
        } else {
            let sum: f32 = output.prediction_similarities.iter().map(|(_, s)| *s).sum();
            sum / output.prediction_similarities.len() as f32
        };

        SafetyMetrics {
            cycle: self.cycle,
            consciousness_level: (1.0 - output.free_energy as f32).clamp(0.0, 1.0),
            prediction_error: output.free_energy as f32,
            temporal_coherence: if temporal_coherence.is_finite() {
                temporal_coherence
            } else {
                0.0
            },
            integrity_critical: false,
        }
    }

    /// Assess a fusion twin output through the safety agent.
    pub fn assess(&mut self, output: &FusionTwinOutput) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(output);
        self.agent.assess(metrics)
    }

    /// Check whether a fusion operation should be allowed given the twin's output.
    ///
    /// Maps `DisruptionRisk` to `SafetyLevel`, then applies the safety gate.
    pub fn gate_operation(&self, output: &FusionTwinOutput, is_risky: bool) -> SafetyGateResult {
        let level = disruption_risk_to_safety_level(output.disruption_risk);
        safety_gate(level, is_risky)
    }

    /// Whether the recommended action requires an emergency response.
    pub fn is_emergency(output: &FusionTwinOutput) -> bool {
        matches!(
            output.recommended_action,
            PlasmaFepAction::EmergencyShutdown | PlasmaFepAction::TriggerSoftLanding
        )
    }

    /// Access the inner safety agent.
    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for FusionSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a disruption risk level to the corresponding NRC safety level.
pub fn disruption_risk_to_safety_level(risk: DisruptionRisk) -> SafetyLevel {
    match risk {
        DisruptionRisk::Green => SafetyLevel::Green,
        DisruptionRisk::Yellow => SafetyLevel::Yellow,
        DisruptionRisk::Orange => SafetyLevel::Orange,
        DisruptionRisk::Red => SafetyLevel::Red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_output(
        free_energy: f64,
        risk: DisruptionRisk,
        action: PlasmaFepAction,
    ) -> FusionTwinOutput {
        FusionTwinOutput {
            predictions: vec![(0.001, "1ms".to_string())],
            disruption_risk: risk,
            free_energy,
            recommended_action: action,
            prediction_similarities: vec![(0.001, 0.95), (0.01, 0.90)],
        }
    }

    #[test]
    fn test_healthy_fusion_maps_to_green() {
        let mut adapter = FusionSafetyAdapter::new();
        let output = mock_output(0.05, DisruptionRisk::Green, PlasmaFepAction::MaintainState);
        let assessment = adapter.assess(&output);
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_high_free_energy_maps_to_red() {
        let mut adapter = FusionSafetyAdapter::new();
        let output = mock_output(0.9, DisruptionRisk::Red, PlasmaFepAction::EmergencyShutdown);
        let assessment = adapter.assess(&output);
        assert!(assessment.level >= SafetyLevel::Orange);
    }

    #[test]
    fn test_disruption_risk_to_safety_level_mapping() {
        assert_eq!(
            disruption_risk_to_safety_level(DisruptionRisk::Green),
            SafetyLevel::Green
        );
        assert_eq!(
            disruption_risk_to_safety_level(DisruptionRisk::Yellow),
            SafetyLevel::Yellow
        );
        assert_eq!(
            disruption_risk_to_safety_level(DisruptionRisk::Orange),
            SafetyLevel::Orange
        );
        assert_eq!(
            disruption_risk_to_safety_level(DisruptionRisk::Red),
            SafetyLevel::Red
        );
    }

    #[test]
    fn test_gate_blocks_risky_at_orange() {
        let adapter = FusionSafetyAdapter::new();
        let output = mock_output(0.6, DisruptionRisk::Orange, PlasmaFepAction::InjectGas);
        assert!(!adapter.gate_operation(&output, true).is_ok());
        assert!(adapter.gate_operation(&output, false).is_ok());
    }

    #[test]
    fn test_gate_blocks_all_at_red() {
        let adapter = FusionSafetyAdapter::new();
        let output = mock_output(0.9, DisruptionRisk::Red, PlasmaFepAction::EmergencyShutdown);
        assert!(!adapter.gate_operation(&output, false).is_ok());
        assert!(!adapter.gate_operation(&output, true).is_ok());
    }

    #[test]
    fn test_is_emergency() {
        let emergency = mock_output(0.9, DisruptionRisk::Red, PlasmaFepAction::EmergencyShutdown);
        assert!(FusionSafetyAdapter::is_emergency(&emergency));

        let soft = mock_output(
            0.6,
            DisruptionRisk::Orange,
            PlasmaFepAction::TriggerSoftLanding,
        );
        assert!(FusionSafetyAdapter::is_emergency(&soft));

        let normal = mock_output(0.1, DisruptionRisk::Green, PlasmaFepAction::MaintainState);
        assert!(!FusionSafetyAdapter::is_emergency(&normal));
    }

    #[test]
    fn test_metrics_temporal_coherence() {
        let adapter = FusionSafetyAdapter::new();
        let output = mock_output(0.1, DisruptionRisk::Green, PlasmaFepAction::MaintainState);
        let metrics = adapter.to_safety_metrics(&output);
        assert!(metrics.temporal_coherence > 0.0);
        assert!(metrics.temporal_coherence.is_finite());
    }

    #[test]
    fn test_metrics_empty_similarities() {
        let adapter = FusionSafetyAdapter::new();
        let output = FusionTwinOutput {
            predictions: vec![],
            disruption_risk: DisruptionRisk::Green,
            free_energy: 0.1,
            recommended_action: PlasmaFepAction::MaintainState,
            prediction_similarities: vec![],
        };
        let metrics = adapter.to_safety_metrics(&output);
        assert_eq!(metrics.temporal_coherence, 0.0);
    }

    #[test]
    fn test_end_to_end_escalation_sequence() {
        // Simulate a fusion disruption event: Green → Yellow → Red
        let mut adapter = FusionSafetyAdapter::new();

        // Healthy plasma
        let a1 = adapter.assess(&mock_output(
            0.05,
            DisruptionRisk::Green,
            PlasmaFepAction::MaintainState,
        ));
        assert_eq!(a1.level, SafetyLevel::Green);

        // Degrading plasma
        let a2 = adapter.assess(&mock_output(
            0.5,
            DisruptionRisk::Yellow,
            PlasmaFepAction::InjectGas,
        ));
        assert!(a2.level >= SafetyLevel::Yellow);

        // Emergency
        let a3 = adapter.assess(&mock_output(
            0.95,
            DisruptionRisk::Red,
            PlasmaFepAction::EmergencyShutdown,
        ));
        assert!(a3.level >= SafetyLevel::Orange);

        // Verify cycle counter
        assert_eq!(adapter.agent().history().len(), 3);
    }

    #[test]
    fn test_metrics_consciousness_clamped() {
        let adapter = FusionSafetyAdapter::new();
        // free_energy > 1.0 should clamp consciousness_level to 0.0
        let output = mock_output(1.5, DisruptionRisk::Red, PlasmaFepAction::EmergencyShutdown);
        let metrics = adapter.to_safety_metrics(&output);
        assert_eq!(metrics.consciousness_level, 0.0);
        assert!(metrics.prediction_error.is_finite());
    }

    #[test]
    fn test_gate_allows_green_risky() {
        let adapter = FusionSafetyAdapter::new();
        let output = mock_output(0.05, DisruptionRisk::Green, PlasmaFepAction::MaintainState);
        assert!(adapter.gate_operation(&output, true).is_ok());
    }

    #[test]
    fn test_all_disruption_risks_map_correctly() {
        let pairs = [
            (DisruptionRisk::Green, SafetyLevel::Green),
            (DisruptionRisk::Yellow, SafetyLevel::Yellow),
            (DisruptionRisk::Orange, SafetyLevel::Orange),
            (DisruptionRisk::Red, SafetyLevel::Red),
        ];
        for (risk, expected) in pairs {
            assert_eq!(disruption_risk_to_safety_level(risk), expected);
        }
    }
}
