// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Strategic Materials -> Safety Agent
//!
//! Maps strategic material free energy into `SafetyMetrics` so the Safety
//! Agent can monitor material service conditions through the same NRC-grade
//! framework.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `strategic-materials` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
#[allow(unused_imports)]
use symthaea_materials::strategic::StrategicFepAction;

/// Adapter that translates strategic material free energy into Safety Agent inputs.
pub struct StrategicMaterialsSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl StrategicMaterialsSafetyAdapter {
    pub fn new() -> Self {
        Self {
            agent: SafetyAgent::new(),
            cycle: 0,
        }
    }

    pub fn with_agent(agent: SafetyAgent) -> Self {
        Self { agent, cycle: 0 }
    }

    /// Assess strategic material safety directly from free energy.
    pub fn assess_direct(&mut self, free_energy: f64) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = SafetyMetrics {
            cycle: self.cycle,
            consciousness_level: (1.0 - free_energy as f32).clamp(0.0, 1.0),
            prediction_error: free_energy as f32,
            temporal_coherence: 0.5,
            integrity_critical: false,
        };
        self.agent.assess(metrics)
    }

    /// Gate a strategic material operation based on free energy.
    pub fn gate_direct(&self, free_energy: f64, is_risky: bool) -> SafetyGateResult {
        let level = fe_to_safety_level(free_energy);
        safety_gate(level, is_risky)
    }

    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for StrategicMaterialsSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map free energy to an NRC safety level.
fn fe_to_safety_level(fe: f64) -> SafetyLevel {
    if fe > 0.7 {
        SafetyLevel::Red
    } else if fe > 0.5 {
        SafetyLevel::Orange
    } else if fe > 0.1 {
        SafetyLevel::Yellow
    } else {
        SafetyLevel::Green
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthy_strategic_maps_to_green() {
        let mut adapter = StrategicMaterialsSafetyAdapter::new();
        let assessment = adapter.assess_direct(0.05);
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_high_fe_escalates() {
        let mut adapter = StrategicMaterialsSafetyAdapter::new();
        let assessment = adapter.assess_direct(0.9);
        assert!(assessment.level >= SafetyLevel::Orange);
    }

    #[test]
    fn test_gate_blocks_risky_at_orange() {
        let adapter = StrategicMaterialsSafetyAdapter::new();
        assert!(!adapter.gate_direct(0.6, true).is_ok());
        assert!(adapter.gate_direct(0.6, false).is_ok());
    }

    #[test]
    fn test_gate_blocks_all_at_red() {
        let adapter = StrategicMaterialsSafetyAdapter::new();
        assert!(!adapter.gate_direct(0.8, false).is_ok());
        assert!(!adapter.gate_direct(0.8, true).is_ok());
    }
}
