// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Datacenter Digital Twin -> Safety Agent
//!
//! Maps `DatacenterOutput` (free energy, safety level, action) into
//! `SafetyMetrics` so the Safety Agent can monitor datacenter operations
//! through the same NRC-grade framework.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `datacenter` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
#[cfg(test)]
use symthaea_physics::datacenter::DatacenterFepAction;
use symthaea_physics::datacenter::{DatacenterOutput, DatacenterSafetyLevel};

/// Adapter that translates Datacenter outputs into Safety Agent inputs.
pub struct DatacenterSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl DatacenterSafetyAdapter {
    pub fn new() -> Self {
        Self {
            agent: SafetyAgent::new(),
            cycle: 0,
        }
    }

    pub fn with_agent(agent: SafetyAgent) -> Self {
        Self { agent, cycle: 0 }
    }

    pub fn to_safety_metrics(&self, output: &DatacenterOutput) -> SafetyMetrics {
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

    pub fn assess(&mut self, output: &DatacenterOutput) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(output);
        self.agent.assess(metrics)
    }

    pub fn gate_operation(&self, output: &DatacenterOutput, is_risky: bool) -> SafetyGateResult {
        let level = datacenter_safety_to_level(output.safety_level);
        safety_gate(level, is_risky)
    }

    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for DatacenterSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a datacenter safety level to the corresponding NRC safety level.
pub fn datacenter_safety_to_level(level: DatacenterSafetyLevel) -> SafetyLevel {
    match level {
        DatacenterSafetyLevel::Green => SafetyLevel::Green,
        DatacenterSafetyLevel::Yellow => SafetyLevel::Yellow,
        DatacenterSafetyLevel::Orange => SafetyLevel::Orange,
        DatacenterSafetyLevel::Red => SafetyLevel::Red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_output(
        free_energy: f64,
        safety: DatacenterSafetyLevel,
        action: DatacenterFepAction,
    ) -> DatacenterOutput {
        DatacenterOutput {
            free_energy,
            recommended_action: action,
            safety_level: safety,
            prediction_similarities: vec![(0.001, 0.95), (0.01, 0.90)],
        }
    }

    #[test]
    fn test_healthy_datacenter_maps_to_green() {
        let mut adapter = DatacenterSafetyAdapter::new();
        let output = mock_output(
            0.05,
            DatacenterSafetyLevel::Green,
            DatacenterFepAction::Maintain,
        );
        let assessment = adapter.assess(&output);
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_datacenter_safety_to_level_mapping() {
        assert_eq!(
            datacenter_safety_to_level(DatacenterSafetyLevel::Green),
            SafetyLevel::Green
        );
        assert_eq!(
            datacenter_safety_to_level(DatacenterSafetyLevel::Yellow),
            SafetyLevel::Yellow
        );
        assert_eq!(
            datacenter_safety_to_level(DatacenterSafetyLevel::Orange),
            SafetyLevel::Orange
        );
        assert_eq!(
            datacenter_safety_to_level(DatacenterSafetyLevel::Red),
            SafetyLevel::Red
        );
    }

    #[test]
    fn test_gate_blocks_risky_at_orange() {
        let adapter = DatacenterSafetyAdapter::new();
        let output = mock_output(
            0.6,
            DatacenterSafetyLevel::Orange,
            DatacenterFepAction::ThrottleCompute,
        );
        assert!(!adapter.gate_operation(&output, true).is_ok());
        assert!(adapter.gate_operation(&output, false).is_ok());
    }

    #[test]
    fn test_gate_blocks_all_at_red() {
        let adapter = DatacenterSafetyAdapter::new();
        let output = mock_output(
            0.9,
            DatacenterSafetyLevel::Red,
            DatacenterFepAction::EmergencyCooldown,
        );
        assert!(!adapter.gate_operation(&output, false).is_ok());
        assert!(!adapter.gate_operation(&output, true).is_ok());
    }
}
