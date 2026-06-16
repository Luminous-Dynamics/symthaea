// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Grid Digital Twin -> Safety Agent
//!
//! Maps `GridOutput` (free energy, safety level, action) into
//! `SafetyMetrics` so the Safety Agent can monitor grid operations
//! through the same NRC-grade framework used for consciousness monitoring.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `grid-scaling` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
#[cfg(test)]
use symthaea_physics::grid::GridFepAction;
use symthaea_physics::grid::{GridOutput, GridSafetyLevel};

/// Adapter that translates Grid Digital Twin outputs into Safety Agent inputs.
pub struct GridSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl GridSafetyAdapter {
    pub fn new() -> Self {
        Self {
            agent: SafetyAgent::new(),
            cycle: 0,
        }
    }

    pub fn with_agent(agent: SafetyAgent) -> Self {
        Self { agent, cycle: 0 }
    }

    pub fn to_safety_metrics(&self, output: &GridOutput) -> SafetyMetrics {
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

    pub fn assess(&mut self, output: &GridOutput) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(output);
        self.agent.assess(metrics)
    }

    pub fn gate_operation(&self, output: &GridOutput, is_risky: bool) -> SafetyGateResult {
        let level = grid_safety_to_level(output.safety_level);
        safety_gate(level, is_risky)
    }

    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for GridSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a grid safety level to the corresponding NRC safety level.
pub fn grid_safety_to_level(level: GridSafetyLevel) -> SafetyLevel {
    match level {
        GridSafetyLevel::Green => SafetyLevel::Green,
        GridSafetyLevel::Yellow => SafetyLevel::Yellow,
        GridSafetyLevel::Orange => SafetyLevel::Orange,
        GridSafetyLevel::Red => SafetyLevel::Red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_output(free_energy: f64, safety: GridSafetyLevel, action: GridFepAction) -> GridOutput {
        GridOutput {
            free_energy,
            recommended_action: action,
            safety_level: safety,
            prediction_similarities: vec![(0.001, 0.95), (0.01, 0.90)],
        }
    }

    #[test]
    fn test_healthy_grid_maps_to_green() {
        let mut adapter = GridSafetyAdapter::new();
        let output = mock_output(0.05, GridSafetyLevel::Green, GridFepAction::MaintainGrid);
        let assessment = adapter.assess(&output);
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_grid_safety_to_level_mapping() {
        assert_eq!(
            grid_safety_to_level(GridSafetyLevel::Green),
            SafetyLevel::Green
        );
        assert_eq!(
            grid_safety_to_level(GridSafetyLevel::Yellow),
            SafetyLevel::Yellow
        );
        assert_eq!(
            grid_safety_to_level(GridSafetyLevel::Orange),
            SafetyLevel::Orange
        );
        assert_eq!(grid_safety_to_level(GridSafetyLevel::Red), SafetyLevel::Red);
    }

    #[test]
    fn test_gate_blocks_risky_at_orange() {
        let adapter = GridSafetyAdapter::new();
        let output = mock_output(0.6, GridSafetyLevel::Orange, GridFepAction::ActivateReserve);
        assert!(!adapter.gate_operation(&output, true).is_ok());
        assert!(adapter.gate_operation(&output, false).is_ok());
    }

    #[test]
    fn test_gate_blocks_all_at_red() {
        let adapter = GridSafetyAdapter::new();
        let output = mock_output(
            0.9,
            GridSafetyLevel::Red,
            GridFepAction::EmergencyCurtailment,
        );
        assert!(!adapter.gate_operation(&output, false).is_ok());
        assert!(!adapter.gate_operation(&output, true).is_ok());
    }
}
