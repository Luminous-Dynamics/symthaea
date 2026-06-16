// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Fission Reactor Digital Twin -> Safety Agent
//!
//! Maps `FissionOutput` (free energy, safety level, action) into
//! `SafetyMetrics` so the Safety Agent can monitor fission reactor
//! operations through the same NRC-grade framework.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `fission-reactor` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
#[cfg(test)]
use symthaea_physics::fission::FissionFepAction;
use symthaea_physics::fission::{FissionOutput, FissionSafetyLevel};

/// Adapter that translates Fission reactor outputs into Safety Agent inputs.
pub struct FissionSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl FissionSafetyAdapter {
    pub fn new() -> Self {
        Self {
            agent: SafetyAgent::new(),
            cycle: 0,
        }
    }

    pub fn with_agent(agent: SafetyAgent) -> Self {
        Self { agent, cycle: 0 }
    }

    pub fn to_safety_metrics(&self, output: &FissionOutput) -> SafetyMetrics {
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

    pub fn assess(&mut self, output: &FissionOutput) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(output);
        self.agent.assess(metrics)
    }

    pub fn gate_operation(&self, output: &FissionOutput, is_risky: bool) -> SafetyGateResult {
        let level = fission_safety_to_level(output.safety_level);
        safety_gate(level, is_risky)
    }

    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for FissionSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a fission safety level to the corresponding NRC safety level.
pub fn fission_safety_to_level(level: FissionSafetyLevel) -> SafetyLevel {
    match level {
        FissionSafetyLevel::Green => SafetyLevel::Green,
        FissionSafetyLevel::Yellow => SafetyLevel::Yellow,
        FissionSafetyLevel::Orange => SafetyLevel::Orange,
        FissionSafetyLevel::Red => SafetyLevel::Red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_output(
        free_energy: f64,
        safety: FissionSafetyLevel,
        action: FissionFepAction,
    ) -> FissionOutput {
        FissionOutput {
            free_energy,
            recommended_action: action,
            safety_level: safety,
            prediction_similarities: vec![(0.001, 0.95), (0.01, 0.90)],
        }
    }

    #[test]
    fn test_healthy_fission_maps_to_green() {
        let mut adapter = FissionSafetyAdapter::new();
        let output = mock_output(
            0.05,
            FissionSafetyLevel::Green,
            FissionFepAction::MaintainReactor,
        );
        let assessment = adapter.assess(&output);
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_fission_safety_to_level_mapping() {
        assert_eq!(
            fission_safety_to_level(FissionSafetyLevel::Green),
            SafetyLevel::Green
        );
        assert_eq!(
            fission_safety_to_level(FissionSafetyLevel::Yellow),
            SafetyLevel::Yellow
        );
        assert_eq!(
            fission_safety_to_level(FissionSafetyLevel::Orange),
            SafetyLevel::Orange
        );
        assert_eq!(
            fission_safety_to_level(FissionSafetyLevel::Red),
            SafetyLevel::Red
        );
    }

    #[test]
    fn test_gate_blocks_risky_at_orange() {
        let adapter = FissionSafetyAdapter::new();
        let output = mock_output(
            0.6,
            FissionSafetyLevel::Orange,
            FissionFepAction::ReducePower,
        );
        assert!(!adapter.gate_operation(&output, true).is_ok());
        assert!(adapter.gate_operation(&output, false).is_ok());
    }

    #[test]
    fn test_gate_blocks_all_at_red() {
        let adapter = FissionSafetyAdapter::new();
        let output = mock_output(0.9, FissionSafetyLevel::Red, FissionFepAction::SCRAM);
        assert!(!adapter.gate_operation(&output, false).is_ok());
        assert!(!adapter.gate_operation(&output, true).is_ok());
    }
}
