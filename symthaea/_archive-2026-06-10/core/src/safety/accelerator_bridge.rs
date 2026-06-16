// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Accelerator Digital Twin -> Safety Agent
//!
//! Maps `AcceleratorOutput` (free energy, safety level, action) into
//! `SafetyMetrics` so the Safety Agent can monitor particle accelerator
//! operations through the same NRC-grade framework.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `accelerator` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
#[cfg(test)]
use symthaea_physics::accelerator::AcceleratorFepAction;
use symthaea_physics::accelerator::{AcceleratorOutput, AcceleratorSafetyLevel};

/// Adapter that translates Accelerator outputs into Safety Agent inputs.
pub struct AcceleratorSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl AcceleratorSafetyAdapter {
    pub fn new() -> Self {
        Self {
            agent: SafetyAgent::new(),
            cycle: 0,
        }
    }

    pub fn with_agent(agent: SafetyAgent) -> Self {
        Self { agent, cycle: 0 }
    }

    pub fn to_safety_metrics(&self, output: &AcceleratorOutput) -> SafetyMetrics {
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

    pub fn assess(&mut self, output: &AcceleratorOutput) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(output);
        self.agent.assess(metrics)
    }

    pub fn gate_operation(&self, output: &AcceleratorOutput, is_risky: bool) -> SafetyGateResult {
        let level = accelerator_safety_to_level(output.safety_level);
        safety_gate(level, is_risky)
    }

    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for AcceleratorSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map an accelerator safety level to the corresponding NRC safety level.
pub fn accelerator_safety_to_level(level: AcceleratorSafetyLevel) -> SafetyLevel {
    match level {
        AcceleratorSafetyLevel::Green => SafetyLevel::Green,
        AcceleratorSafetyLevel::Yellow => SafetyLevel::Yellow,
        AcceleratorSafetyLevel::Orange => SafetyLevel::Orange,
        AcceleratorSafetyLevel::Red => SafetyLevel::Red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_output(
        free_energy: f64,
        safety: AcceleratorSafetyLevel,
        action: AcceleratorFepAction,
    ) -> AcceleratorOutput {
        AcceleratorOutput {
            free_energy,
            recommended_action: action,
            safety_level: safety,
            prediction_similarities: vec![(0.001, 0.95), (0.01, 0.90)],
        }
    }

    #[test]
    fn test_healthy_accelerator_maps_to_green() {
        let mut adapter = AcceleratorSafetyAdapter::new();
        let output = mock_output(
            0.05,
            AcceleratorSafetyLevel::Green,
            AcceleratorFepAction::MaintainBeam,
        );
        let assessment = adapter.assess(&output);
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_accelerator_safety_to_level_mapping() {
        assert_eq!(
            accelerator_safety_to_level(AcceleratorSafetyLevel::Green),
            SafetyLevel::Green
        );
        assert_eq!(
            accelerator_safety_to_level(AcceleratorSafetyLevel::Yellow),
            SafetyLevel::Yellow
        );
        assert_eq!(
            accelerator_safety_to_level(AcceleratorSafetyLevel::Orange),
            SafetyLevel::Orange
        );
        assert_eq!(
            accelerator_safety_to_level(AcceleratorSafetyLevel::Red),
            SafetyLevel::Red
        );
    }

    #[test]
    fn test_gate_blocks_risky_at_orange() {
        let adapter = AcceleratorSafetyAdapter::new();
        let output = mock_output(
            0.6,
            AcceleratorSafetyLevel::Orange,
            AcceleratorFepAction::ReduceIntensity,
        );
        assert!(!adapter.gate_operation(&output, true).is_ok());
        assert!(adapter.gate_operation(&output, false).is_ok());
    }

    #[test]
    fn test_gate_blocks_all_at_red() {
        let adapter = AcceleratorSafetyAdapter::new();
        let output = mock_output(
            0.9,
            AcceleratorSafetyLevel::Red,
            AcceleratorFepAction::BeamDump,
        );
        assert!(!adapter.gate_operation(&output, false).is_ok());
        assert!(!adapter.gate_operation(&output, true).is_ok());
    }
}
