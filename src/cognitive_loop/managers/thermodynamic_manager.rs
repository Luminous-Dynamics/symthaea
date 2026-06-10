// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # ThermodynamicManager — CognitiveSubsystem for Unified Thermodynamics
//!
//! Implements the `CognitiveSubsystem` trait at interval 43 (co-prime with
//! 7, 11, 13, 19, 29, 37, 41) to integrate all thermodynamic computation
//! into the cognitive loop's proposal-based architecture.
//!
//! Owns `ThermodynamicIntegration` (which owns the unified state + physics bridge).
//! Each cycle, observes the `CycleSnapshot`, builds a `ThermodynamicInput`,
//! runs the full integration pipeline, and returns a `SubsystemOutput` with
//! LR/exploration/confidence proposals.
//!
//! ## Science
//!
//! Prigogine (1977), Landauer (1961), Szilard (1929), Carnot (1824),
//! Jarzynski (1997), Crooks (1999), Onsager (1931), Kauffman (1993).

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};
use super::super::thermodynamic_integration::{ThermodynamicInput, ThermodynamicIntegration};
use crate::consciousness::consciousness_thermodynamics::ConsciousnessPhase;
use crate::consciousness::dissipative_consciousness::ThermodynamicRegime;

/// ThermodynamicManager: CognitiveSubsystem at interval 43.
///
/// Integrates DissipativeConsciousness, ConsciousnessThermodynamicsAnalyzer,
/// HierarchicalFreeEnergy, and ThermodynamicPhysicsBridge into the proposal
/// system via a single `process()` call per interval.
pub struct ThermodynamicManager {
    /// Core integration engine (unified state + physics bridge + feedback loops).
    pub integration: ThermodynamicIntegration,

    /// Externally-set input that the cycle populates before process() runs.
    /// The cycle writes module outputs here, then process() reads it.
    input: ThermodynamicInput,

    /// Cached memory suppression flag from last cycle.
    last_memory_suppressed: bool,
}

impl Default for ThermodynamicManager {
    fn default() -> Self {
        Self {
            integration: ThermodynamicIntegration::default(),
            input: ThermodynamicInput::default(),
            last_memory_suppressed: false,
        }
    }
}

impl ThermodynamicManager {
    /// Co-prime scheduling interval.
    ///
    /// 43 is co-prime with all existing manager intervals:
    /// - GWT: 7, Ethics: 11, Learning: 13, Memory: 19, Consciousness: 29
    /// - Governance: 37, Swarm: 41, Cantor: 47, Spectrum: 53, Metabolism: 59
    pub const INTERVAL: u32 = 43;

    /// Set dissipative consciousness data (from primitive tier).
    pub fn set_dissipative(
        &mut self,
        health: f64,
        regime: ThermodynamicRegime,
        entropy_rate: f64,
        order: f64,
        criticality: f64,
        lambda: f64,
        efficiency: f64,
    ) {
        self.input.dissipative_health = health;
        self.input.dissipative_regime = regime;
        self.input.entropy_production_rate = entropy_rate;
        self.input.order_parameter = order;
        self.input.criticality_distance = criticality;
        self.input.lambda_parameter = lambda;
        self.input.dissipation_efficiency = efficiency;
        self.input.has_dissipative = true;
    }

    /// Set consciousness thermodynamics analyzer data.
    pub fn set_analyzer(
        &mut self,
        entropy: f64,
        free_energy: f64,
        temperature: f64,
        phase: ConsciousnessPhase,
    ) {
        self.input.analyzer_entropy = entropy;
        self.input.analyzer_free_energy = free_energy;
        self.input.analyzer_temperature = temperature;
        self.input.analyzer_phase = phase;
        self.input.has_analyzer = true;
    }

    /// Set hierarchical free energy data.
    pub fn set_hfe(&mut self, total: f64, complexity: f64, accuracy: f64) {
        self.input.hfe_total = total;
        self.input.hfe_complexity = complexity;
        self.input.hfe_accuracy = accuracy;
        self.input.has_hfe = true;
    }

    /// Set energy budget data from SubstrateManager.
    pub fn set_energy(&mut self, per_cycle: f64, total: f64, throughput: f32) {
        self.input.energy_per_cycle = per_cycle;
        self.input.total_energy_spent = total;
        self.input.energy_throughput_multiplier = throughput;
    }

    /// Access unified thermodynamic state (for telemetry export).
    pub fn state(&self) -> &super::super::thermodynamic_state::UnifiedThermodynamicState {
        &self.integration.state
    }

    /// Access physics bridge (for telemetry export).
    pub fn bridge(
        &self,
    ) -> &super::super::thermodynamic_physics_bridge::ThermodynamicPhysicsBridge {
        &self.integration.bridge
    }

    /// Whether Landauer memory pressure is suppressing consolidation.
    pub fn memory_consolidation_suppressed(&self) -> bool {
        self.last_memory_suppressed
    }
}

impl CognitiveSubsystem for ThermodynamicManager {
    fn name(&self) -> &'static str {
        "thermodynamic_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        // Populate core snapshot fields into input
        self.input.unified_psi = snapshot.unified_psi;
        self.input.coherence = snapshot.coherence;
        self.input.prediction_error = snapshot.prediction_error;
        self.input.attention_sensitivity = snapshot.phi_attention_weight;
        self.input.metabolic_stress = snapshot.thermodynamic_load;

        // Run full thermodynamic integration pipeline
        let feedback = self.integration.run_cycle(&self.input);

        // Translate ThermodynamicFeedback → SubsystemOutput
        let mut output = SubsystemOutput::NEUTRAL;

        // LR modulation: compound all LR feedback
        output.lr_modulation = feedback.lr_factor as f64;

        // Exploration: translate exploration_factor into delta
        // factor < 1.0 → negative delta (dampen), > 1.0 → positive delta (boost)
        output.exploration_delta = (feedback.exploration_factor as f64 - 1.0) * 0.1;

        // Confidence: insight boosts, Prigogine violation reduces
        if feedback.insight_detected {
            output.confidence_delta += 0.01;
        }
        if feedback.prigogine_violated {
            output.confidence_delta -= 0.005;
        }

        // Flags: memory consolidation suppression
        if feedback.memory_consolidation_suppressed {
            output.flags |= output_flags::REQUEST_CONSOLIDATION;
        }

        self.last_memory_suppressed = feedback.memory_consolidation_suppressed;

        output
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_neutral_output() {
        let mut mgr = ThermodynamicManager::default();
        let snapshot = CycleSnapshot::default();
        let output = mgr.process(&snapshot);
        // No modules set → near-neutral output
        assert!(output.lr_modulation >= 0.9 && output.lr_modulation <= 1.1);
        assert!(output.exploration_delta.abs() < 0.1);
    }

    #[test]
    fn test_process_with_all_modules() {
        let mut mgr = ThermodynamicManager::default();
        mgr.set_dissipative(
            0.7,
            ThermodynamicRegime::EdgeOfChaos,
            0.15,
            0.6,
            0.03,
            0.27,
            0.5,
        );
        mgr.set_analyzer(0.45, -0.15, 0.42, ConsciousnessPhase::Normal);
        mgr.set_hfe(0.6, 0.35, 0.25);
        mgr.set_energy(1e-10, 5e-8, 1.0);

        let mut snapshot = CycleSnapshot::default();
        snapshot.unified_psi = 0.7;
        snapshot.coherence = 0.8;
        snapshot.prediction_error = 0.2;

        let output = mgr.process(&snapshot);
        assert!(output.lr_modulation.is_finite());
        assert!(output.exploration_delta.is_finite());
        assert!(output.confidence_delta.is_finite());
    }

    #[test]
    fn test_interval_is_43() {
        let mgr = ThermodynamicManager::default();
        assert_eq!(mgr.interval(), 43);
    }

    #[test]
    fn test_name() {
        let mgr = ThermodynamicManager::default();
        assert_eq!(mgr.name(), "thermodynamic_manager");
    }

    #[test]
    fn test_state_accessors() {
        let mgr = ThermodynamicManager::default();
        let state = mgr.state();
        assert_eq!(state.canonical_entropy, 0.5);
        let bridge = mgr.bridge();
        assert_eq!(bridge.carnot_efficiency, 0.0);
    }

    #[test]
    fn test_memory_pressure_flag() {
        let mut mgr = ThermodynamicManager::default();
        mgr.set_energy(1e-15, 0.0, 1.0); // Tiny budget
        mgr.set_analyzer(0.5, -0.1, 0.4, ConsciousnessPhase::Normal);

        let mut snapshot = CycleSnapshot::default();
        snapshot.prediction_error = 10.0; // Huge error → lots of bits to consolidate
        snapshot.unified_psi = 0.5;

        let output = mgr.process(&snapshot);
        if mgr.memory_consolidation_suppressed() {
            assert!(output.flags & output_flags::REQUEST_CONSOLIDATION != 0);
        }
    }
}
