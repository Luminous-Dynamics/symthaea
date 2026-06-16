// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Threat Assessment -> Safety Agent
//!
//! Maps `ThreatOutput` (free energy, threat level, action) into
//! `SafetyMetrics` so the Safety Agent can monitor threat assessments
//! through the same NRC-grade framework.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `threat-assessment` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
#[cfg(test)]
use symthaea_physics::threat::ThreatFepAction;
use symthaea_physics::threat::{ThreatLevel, ThreatOutput};

/// Adapter that translates Threat Assessment outputs into Safety Agent inputs.
pub struct ThreatSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl ThreatSafetyAdapter {
    pub fn new() -> Self {
        Self {
            agent: SafetyAgent::new(),
            cycle: 0,
        }
    }

    pub fn with_agent(agent: SafetyAgent) -> Self {
        Self { agent, cycle: 0 }
    }

    pub fn to_safety_metrics(&self, output: &ThreatOutput) -> SafetyMetrics {
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

    pub fn assess(&mut self, output: &ThreatOutput) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(output);
        self.agent.assess(metrics)
    }

    pub fn gate_operation(&self, output: &ThreatOutput, is_risky: bool) -> SafetyGateResult {
        let level = threat_to_safety_level(output.threat_level);
        safety_gate(level, is_risky)
    }

    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for ThreatSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a threat level to the corresponding NRC safety level.
pub fn threat_to_safety_level(level: ThreatLevel) -> SafetyLevel {
    match level {
        ThreatLevel::Green => SafetyLevel::Green,
        ThreatLevel::Yellow => SafetyLevel::Yellow,
        ThreatLevel::Orange => SafetyLevel::Orange,
        ThreatLevel::Red => SafetyLevel::Red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_output(free_energy: f64, threat: ThreatLevel, action: ThreatFepAction) -> ThreatOutput {
        ThreatOutput {
            free_energy,
            recommended_action: action,
            threat_level: threat,
            prediction_similarities: vec![(0.001, 0.95), (0.01, 0.90)],
        }
    }

    #[test]
    fn test_healthy_threat_maps_to_green() {
        let mut adapter = ThreatSafetyAdapter::new();
        let output = mock_output(0.05, ThreatLevel::Green, ThreatFepAction::Monitor);
        let assessment = adapter.assess(&output);
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_threat_to_safety_level_mapping() {
        assert_eq!(
            threat_to_safety_level(ThreatLevel::Green),
            SafetyLevel::Green
        );
        assert_eq!(
            threat_to_safety_level(ThreatLevel::Yellow),
            SafetyLevel::Yellow
        );
        assert_eq!(
            threat_to_safety_level(ThreatLevel::Orange),
            SafetyLevel::Orange
        );
        assert_eq!(threat_to_safety_level(ThreatLevel::Red), SafetyLevel::Red);
    }

    #[test]
    fn test_gate_blocks_risky_at_orange() {
        let adapter = ThreatSafetyAdapter::new();
        let output = mock_output(0.6, ThreatLevel::Orange, ThreatFepAction::ActivateResponse);
        assert!(!adapter.gate_operation(&output, true).is_ok());
        assert!(adapter.gate_operation(&output, false).is_ok());
    }

    #[test]
    fn test_gate_blocks_all_at_red() {
        let adapter = ThreatSafetyAdapter::new();
        let output = mock_output(0.9, ThreatLevel::Red, ThreatFepAction::EmergencyProtocol);
        assert!(!adapter.gate_operation(&output, false).is_ok());
        assert!(!adapter.gate_operation(&output, true).is_ok());
    }
}
