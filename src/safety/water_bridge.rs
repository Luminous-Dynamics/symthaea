// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain bridge: Water Quality → Safety Agent
//!
//! Maps water quality readings and FEP agent outputs into `SafetyMetrics`
//! so the Safety Agent can monitor water infrastructure through the same
//! NRC-grade framework used for consciousness monitoring.
//!
//! # Feature gates
//!
//! Requires both `safety-agents` and `water-prediction` features.

use super::agent::{SafetyAgent, SafetyAssessment, SafetyLevel, SafetyMetrics};
use super::gate::{SafetyGateResult, safety_gate};
use symthaea_cell_foundry::hydrology::{WaterFepAction, WaterFepAgent, WaterQualityReading};

/// Input bundle for a single water safety assessment.
///
/// Combines a water quality reading with the FEP agent's analysis
/// (free energy + recommended action) into a single struct for the adapter.
pub struct WaterSafetyInput {
    /// The raw water quality reading.
    pub reading: WaterQualityReading,
    /// Free energy (surprise) computed by the WaterFepAgent.
    pub free_energy: f64,
    /// Action recommended by the FEP agent.
    pub recommended_action: WaterFepAction,
}

impl WaterSafetyInput {
    /// Build a safety input from a reading and a FEP agent.
    pub fn from_agent(reading: &WaterQualityReading, agent: &WaterFepAgent) -> Self {
        Self {
            reading: reading.clone(),
            free_energy: agent.compute_free_energy(reading),
            recommended_action: agent.select_action(reading),
        }
    }
}

/// Adapter that translates water quality data into Safety Agent inputs.
///
/// Enables unified safety monitoring of water infrastructure alongside
/// consciousness and other domains. Unsafe water (low potability, high
/// free energy) maps to elevated safety levels.
pub struct WaterSafetyAdapter {
    agent: SafetyAgent,
    cycle: usize,
}

impl WaterSafetyAdapter {
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

    /// Convert a `WaterSafetyInput` into `SafetyMetrics`.
    ///
    /// Mapping:
    /// - `consciousness_level` ← potability (water safety proxy)
    /// - `prediction_error` ← free_energy (surprise from FEP agent)
    /// - `temporal_coherence` ← 1.0 - normalized deviation from potable standards
    pub fn to_safety_metrics(&self, input: &WaterSafetyInput) -> SafetyMetrics {
        let potability = if input.reading.potability.is_finite() {
            (input.reading.potability as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let fe = if input.free_energy.is_finite() {
            input.free_energy as f32
        } else {
            1.0
        };

        let deviation = input.reading.deviation_from_potable() as f32;
        let coherence = if deviation.is_finite() {
            (1.0 - deviation.min(1.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        SafetyMetrics {
            cycle: self.cycle,
            consciousness_level: potability,
            prediction_error: fe,
            temporal_coherence: coherence,
            integrity_critical: false,
        }
    }

    /// Assess a water quality input through the safety agent.
    pub fn assess(&mut self, input: &WaterSafetyInput) -> SafetyAssessment {
        self.cycle += 1;
        let metrics = self.to_safety_metrics(input);
        self.agent.assess(metrics)
    }

    /// Check whether a water operation should proceed given current quality.
    ///
    /// Maps potability to safety level, then applies the safety gate.
    pub fn gate_operation(&self, input: &WaterSafetyInput, is_risky: bool) -> SafetyGateResult {
        let level = potability_to_safety_level(input.reading.potability);
        safety_gate(level, is_risky)
    }

    /// Whether the FEP agent recommends emergency shutoff.
    pub fn is_emergency(input: &WaterSafetyInput) -> bool {
        matches!(input.recommended_action, WaterFepAction::EmergencyShutoff)
    }

    /// Access the inner safety agent.
    pub fn agent(&self) -> &SafetyAgent {
        &self.agent
    }
}

impl Default for WaterSafetyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Map water potability to an NRC safety level.
pub fn potability_to_safety_level(potability: f64) -> SafetyLevel {
    let p = if potability.is_finite() {
        potability as f32
    } else {
        0.0
    };
    if p >= 0.6 {
        SafetyLevel::Green
    } else if p >= 0.35 {
        SafetyLevel::Yellow
    } else if p >= 0.15 {
        SafetyLevel::Orange
    } else {
        SafetyLevel::Red
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_input() -> WaterSafetyInput {
        let reading = WaterQualityReading::clean();
        let agent = WaterFepAgent::new();
        WaterSafetyInput::from_agent(&reading, &agent)
    }

    fn contaminated_input() -> WaterSafetyInput {
        let reading = WaterQualityReading::contaminated();
        let agent = WaterFepAgent::new();
        WaterSafetyInput::from_agent(&reading, &agent)
    }

    #[test]
    fn test_clean_water_maps_to_green() {
        let mut adapter = WaterSafetyAdapter::new();
        let assessment = adapter.assess(&clean_input());
        assert_eq!(assessment.level, SafetyLevel::Green);
    }

    #[test]
    fn test_contaminated_water_escalates() {
        let mut adapter = WaterSafetyAdapter::new();
        let assessment = adapter.assess(&contaminated_input());
        assert!(assessment.level >= SafetyLevel::Orange);
    }

    #[test]
    fn test_potability_to_safety_level() {
        assert_eq!(potability_to_safety_level(1.0), SafetyLevel::Green);
        assert_eq!(potability_to_safety_level(0.5), SafetyLevel::Yellow);
        assert_eq!(potability_to_safety_level(0.25), SafetyLevel::Orange);
        assert_eq!(potability_to_safety_level(0.1), SafetyLevel::Red);
    }

    #[test]
    fn test_potability_nan_maps_to_red() {
        assert_eq!(potability_to_safety_level(f64::NAN), SafetyLevel::Red);
    }

    #[test]
    fn test_gate_blocks_risky_contaminated() {
        let adapter = WaterSafetyAdapter::new();
        let input = contaminated_input();
        // Contaminated water (potability=0.1) → Red → blocks everything
        assert!(!adapter.gate_operation(&input, true).is_ok());
    }

    #[test]
    fn test_gate_allows_clean() {
        let adapter = WaterSafetyAdapter::new();
        let input = clean_input();
        assert!(adapter.gate_operation(&input, true).is_ok());
    }

    #[test]
    fn test_is_emergency() {
        assert!(WaterSafetyAdapter::is_emergency(&contaminated_input()));
        assert!(!WaterSafetyAdapter::is_emergency(&clean_input()));
    }

    #[test]
    fn test_escalation_sequence() {
        let mut adapter = WaterSafetyAdapter::new();

        let a1 = adapter.assess(&clean_input());
        assert_eq!(a1.level, SafetyLevel::Green);

        let a2 = adapter.assess(&contaminated_input());
        assert!(a2.level >= SafetyLevel::Orange);

        assert_eq!(adapter.agent().history().len(), 2);
    }

    #[test]
    fn test_metrics_nan_clamped() {
        let adapter = WaterSafetyAdapter::new();
        let input = WaterSafetyInput {
            reading: WaterQualityReading {
                potability: f64::NAN,
                ..WaterQualityReading::clean()
            },
            free_energy: f64::NAN,
            recommended_action: WaterFepAction::Maintain,
        };
        let metrics = adapter.to_safety_metrics(&input);
        assert_eq!(metrics.consciousness_level, 0.0);
        assert_eq!(metrics.prediction_error, 1.0);
    }
}
