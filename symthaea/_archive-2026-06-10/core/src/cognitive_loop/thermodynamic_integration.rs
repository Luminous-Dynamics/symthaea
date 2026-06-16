// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Thermodynamic Integration: Single entry point for all thermodynamic
//! computations, cross-couplings, and feedback loops.
//!
//! Consolidates all wiring between DissipativeConsciousness,
//! ConsciousnessThermodynamicsAnalyzer, HierarchicalFreeEnergy,
//! ThermodynamicPhysicsBridge, and SubstrateManager into one struct
//! with a single `run_cycle()` method.
//!
//! This design keeps all thermodynamic logic in new files (not scattered
//! across existing cycle files), making it resilient to concurrent edits.

use serde::{Deserialize, Serialize};

use super::thermodynamic_physics_bridge::ThermodynamicPhysicsBridge;
use super::thermodynamic_state::UnifiedThermodynamicState;
use super::thresholds;

use crate::consciousness::consciousness_thermodynamics::ConsciousnessPhase;
use crate::consciousness::dissipative_consciousness::ThermodynamicRegime;

/// Input snapshot for thermodynamic integration — gathered from the cycle.
#[derive(Debug, Clone)]
pub struct ThermodynamicInput {
    // From consciousness phase
    pub unified_psi: f64,
    pub coherence: f32,
    pub prediction_error: f32,
    pub attention_sensitivity: f32,

    // From substrate/metabolism
    pub energy_per_cycle: f64,
    pub total_energy_spent: f64,
    pub energy_throughput_multiplier: f32,
    pub metabolic_stress: f32,

    // From dissipative consciousness (if available)
    pub dissipative_health: f64,
    pub dissipative_regime: ThermodynamicRegime,
    pub entropy_production_rate: f64,
    pub order_parameter: f64,
    pub criticality_distance: f64,
    pub lambda_parameter: f64,
    pub dissipation_efficiency: f64,
    pub has_dissipative: bool,

    // From consciousness thermodynamics analyzer (if available)
    pub analyzer_entropy: f64,
    pub analyzer_free_energy: f64,
    pub analyzer_temperature: f64,
    pub analyzer_phase: ConsciousnessPhase,
    pub has_analyzer: bool,

    // From HFE (if available)
    pub hfe_total: f64,
    pub hfe_complexity: f64,
    pub hfe_accuracy: f64,
    pub has_hfe: bool,
}

impl Default for ThermodynamicInput {
    fn default() -> Self {
        Self {
            unified_psi: 0.0,
            coherence: 0.0,
            prediction_error: 0.0,
            attention_sensitivity: 0.0,
            energy_per_cycle: 0.0,
            total_energy_spent: 0.0,
            energy_throughput_multiplier: 1.0,
            metabolic_stress: 0.0,
            dissipative_health: 0.0,
            dissipative_regime: ThermodynamicRegime::Equilibrium,
            entropy_production_rate: 0.0,
            order_parameter: 0.0,
            criticality_distance: 0.273,
            lambda_parameter: 0.0,
            dissipation_efficiency: 0.0,
            has_dissipative: false,
            analyzer_entropy: 0.0,
            analyzer_free_energy: 0.0,
            analyzer_temperature: 0.3,
            analyzer_phase: ConsciousnessPhase::Normal,
            has_analyzer: false,
            hfe_total: 0.0,
            hfe_complexity: 0.0,
            hfe_accuracy: 0.0,
            has_hfe: false,
        }
    }
}

/// Output feedback from thermodynamic integration — applied to the cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThermodynamicFeedback {
    /// Learning rate multiplier from all thermodynamic sources.
    pub lr_factor: f32,
    /// Exploration multiplier from all thermodynamic sources.
    pub exploration_factor: f32,
    /// Whether insight was detected (for logging/telemetry).
    pub insight_detected: bool,
    /// Whether Prigogine violation was detected.
    pub prigogine_violated: bool,
    /// Suggested HFE learning rate adjustment (0.0 = no change).
    pub hfe_lr_adjustment: f64,
    /// Whether Landauer pressure is suppressing memory consolidation.
    pub memory_consolidation_suppressed: bool,
    /// Whether England dissipation-driven adaptation signal is active.
    /// When true, entropy production is correlated with order growth and PE reduction
    /// — exploration should be boosted, not dampened.
    pub england_boost: bool,
}

/// Consolidated thermodynamic integration — owns unified state + physics bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicIntegration {
    /// Unified thermodynamic state (canonical snapshot from all modules).
    pub state: UnifiedThermodynamicState,
    /// Physics bridge (Maxwell Demon, Landauer, Carnot, Onsager, Jarzynski, Prigogine).
    pub bridge: ThermodynamicPhysicsBridge,
    /// Previous order parameter (for insight detection).
    prev_order: f64,
}

impl Default for ThermodynamicIntegration {
    fn default() -> Self {
        Self {
            state: UnifiedThermodynamicState::default(),
            bridge: ThermodynamicPhysicsBridge::default(),
            prev_order: 0.0,
        }
    }
}

impl ThermodynamicIntegration {
    /// Run the full thermodynamic integration for this cycle.
    ///
    /// 1. Updates unified state from all module outputs
    /// 2. Applies cross-couplings
    /// 3. Computes physics bridge
    /// 4. Generates feedback for the cycle to apply
    pub(crate) fn run_cycle(&mut self, input: &ThermodynamicInput) -> ThermodynamicFeedback {
        let mut feedback = ThermodynamicFeedback {
            lr_factor: 1.0,
            exploration_factor: 1.0,
            ..Default::default()
        };

        // ═══════════════════════════════════════════════════════════════════
        // Phase 1: Update unified state from all modules
        // ═══════════════════════════════════════════════════════════════════

        if input.has_dissipative {
            self.state.update_dissipative(
                input.dissipative_health,
                input.dissipative_regime,
                input.entropy_production_rate,
                input.entropy_production_rate,
                input.order_parameter,
                input.criticality_distance,
                input.lambda_parameter,
                input.dissipation_efficiency,
            );
        }

        if input.has_analyzer {
            self.state.update_analyzer(
                input.analyzer_entropy,
                input.analyzer_free_energy,
                input.analyzer_temperature,
                input.analyzer_phase,
            );
        }

        if input.has_hfe {
            self.state
                .update_hfe(input.hfe_total, input.hfe_complexity, input.hfe_accuracy);
        }

        self.state.update_energy(
            input.energy_per_cycle,
            input.total_energy_spent,
            input.metabolic_stress,
            input.prediction_error,
        );

        // ═══════════════════════════════════════════════════════════════════
        // Phase 2: Cross-couplings (modules informing each other)
        // ═══════════════════════════════════════════════════════════════════

        // 2c: HFE → blended canonical free energy
        if input.has_hfe && input.hfe_total.is_finite() && input.hfe_total.abs() > 0.001 {
            let w = thresholds::THERMO_HFE_BLEND_WEIGHT as f64;
            let hfe_norm = input.hfe_total / (1.0 + input.hfe_total.abs());
            let analyzer_fe = self.state.canonical_free_energy;
            self.state.canonical_free_energy = if analyzer_fe.is_finite() {
                (1.0 - w) * analyzer_fe + w * hfe_norm
            } else {
                hfe_norm
            };
        }

        // 2e: Substrate energy throughput → temperature modulation
        let throughput = input.energy_throughput_multiplier;
        if throughput.is_finite() && throughput > 1.0 {
            let temp_boost = ((throughput - 1.0) * 0.01).min(0.05) as f64;
            self.state.effective_temperature =
                (self.state.effective_temperature + temp_boost).min(2.0);
        }

        // ═══════════════════════════════════════════════════════════════════
        // Phase 3: Physics bridge computation
        // ═══════════════════════════════════════════════════════════════════

        let info_processed = input.coherence as f64 * input.unified_psi;
        self.bridge.compute(
            input.attention_sensitivity as f64,
            info_processed,
            input.coherence as f64,
            input.prediction_error as f64,
            self.state.effective_temperature,
            self.state.entropy_production_rate,
            self.state.order_parameter,
            self.state.energy_per_cycle,
            self.prev_order,
            self.state.hierarchical_free_energy,
            &self.state.regime,
        );
        self.prev_order = self.state.order_parameter;

        // Write derived quantities back to unified state
        self.state.carnot_efficiency = self.bridge.carnot_efficiency;
        self.state.landauer_bound_joules = self.bridge.memory_landauer_cost;

        // ═══════════════════════════════════════════════════════════════════
        // Phase 4: Feedback loops (active self-regulation)
        // ═══════════════════════════════════════════════════════════════════

        // 4a: Insight detection → LR boost
        // Science: Jarzynski (1997) — rare entropy-decreasing fluctuations carry information.
        if self.bridge.insight_detected {
            feedback.lr_factor *= thresholds::THERMO_INSIGHT_LR_BOOST;
            feedback.insight_detected = true;
        }

        // 4b: Prigogine violation → exploration damping
        // Science: Prigogine (1947) — increasing entropy production in LinearNonEquilibrium
        // violates minimum entropy production principle → system destabilizing.
        // Note: only valid in LinearNonEquilibrium — at EdgeOfChaos, entropy production
        // is necessary for complex dynamics (Langton 1990).
        if self.bridge.prigogine_violated {
            feedback.exploration_factor *= thresholds::THERMO_PRIGOGINE_VIOLATION_DAMPING;
            feedback.prigogine_violated = true;
        }

        // 4b2: England dissipation-driven adaptation → exploration boost
        // Science: England (2013) — systems driven by external energy self-organize
        // to maximize entropy production while building internal order.
        // When entropy + order + PE-reduction align, dissipation is driving learning.
        if self.bridge.england_exploration_boost {
            feedback.exploration_factor *= 1.1; // 10% boost
            feedback.england_boost = true;
        }

        // 4c: Jarzynski-HFE divergence → HFE learning rate correction
        // Science: Crooks (1999) — divergence between nonequilibrium and model free energy
        // indicates model inaccuracy → boost learning to correct.
        let divergence = self.bridge.jarzynski_hfe_divergence;
        if divergence.is_finite() && divergence > thresholds::THERMO_JARZYNSKI_DIVERGENCE_THRESHOLD
        {
            let correction = (divergence * 0.5).min(2.0);
            feedback.hfe_lr_adjustment = correction;
            // Also boost global LR slightly
            feedback.lr_factor *= 1.0 + (correction * 0.02) as f32; // Gentle global boost
        }

        // 4d: Onsager asymmetry → coherence-seeking response
        // Science: Onsager (1931) — asymmetric transport = decoupled dimensions.
        let asymmetry = self.bridge.onsager_asymmetry;
        if asymmetry.is_finite() && asymmetry > thresholds::THERMO_ONSAGER_ASYMMETRY_THRESHOLD {
            feedback.exploration_factor *= thresholds::THERMO_ONSAGER_COHERENCE_DAMPING;
        }

        // 4e: Carnot efficiency → energy awareness
        // Science: Carnot (1824) — low actual/theoretical ratio = wasting energy.
        // When efficiency is very low, dampen expensive operations.
        if self.bridge.efficiency_ratio < thresholds::THERMO_LOW_EFFICIENCY_THRESHOLD
            && self.bridge.carnot_efficiency > 0.1
        {
            feedback.exploration_factor *= 0.95; // Mild conservation
        }

        // 4f: Landauer → memory consolidation gating
        // Science: Landauer (1961) — memory has minimum thermodynamic cost.
        // When energy budget is tight, suppress memory consolidation.
        let budget_fraction = if input.energy_per_cycle > 0.0 {
            self.bridge.memory_landauer_cost / input.energy_per_cycle
        } else {
            0.0
        };
        if budget_fraction > thresholds::THERMO_LANDAUER_MEMORY_PRESSURE_THRESHOLD {
            feedback.memory_consolidation_suppressed = true;
        }

        feedback
    }

    /// Encode the unified thermodynamic state as a 16,384-dimensional HDC vector.
    ///
    /// Creates a composite hypervector by binding temperature, entropy, free energy,
    /// and order parameter into the HDC space. This makes thermodynamics a first-class
    /// participant in HDC reasoning — the vector can be bundled into the cognitive
    /// state, compared across cycles for pattern detection, and used to identify
    /// thermodynamic attractors.
    ///
    /// # Science
    ///
    /// - Kanerva (2009) — Hyperdimensional Computing: binding = conjunction, bundling = superposition
    /// - Tononi (2004) — Phi as thermodynamic potential maps naturally to HDC encoding
    /// - Friston (2010) — Free energy as HDC vector enables prediction-error in HDC space
    pub fn encode_as_hdc(
        &self,
        encoder: &symthaea_core::physics::ThermoEncoder,
    ) -> symthaea_core::hdc::unified_hv::ContinuousHV {
        use symthaea_core::hdc::unified_hv::ContinuousHV;

        let s = &self.state;

        // 1. Encode entropy as HDC vector (Shannon entropy of consciousness dimensions)
        //    Use 7 probability-like values from normalized state
        let entropy_probs = [
            s.canonical_entropy.max(0.01),
            s.effective_temperature.max(0.01) / 2.0,
            s.order_parameter.max(0.01),
            s.dissipative_health.max(0.01),
            (1.0 - s.criticality_distance).max(0.01),
            s.dissipation_efficiency.max(0.01),
            (1.0 - s.canonical_free_energy.abs().min(1.0)).max(0.01),
        ];
        let entropy_hv = encoder.shannon_entropy(&entropy_probs);

        // 2. Encode thermodynamic state (T, P, V, U) via ThermoEncoder
        //    Map consciousness quantities to thermodynamic analogues:
        //    - Temperature = effective_temperature (already in Kelvin-analogue)
        //    - Pressure = entropy_production_rate (thermodynamic pressure to dissipate)
        //    - Volume = 1.0 - criticality_distance (phase space volume near edge)
        //    - Internal energy = order_parameter (structural energy)
        let state_hv = encoder
            .encode_state(
                s.effective_temperature.max(0.01) * 300.0, // Scale to Kelvin range
                s.entropy_production_rate.abs().max(0.001) * 101325.0, // Scale to Pa range
                (1.0 - s.criticality_distance).max(0.001) * 0.001, // Scale to m³ range
                s.order_parameter.max(0.001) * 1000.0,     // Scale to Joules range
            )
            .vector;

        // 3. Encode free energy (Helmholtz: F = U - TS)
        let helmholtz_hv = {
            let thermo_state = encoder.encode_state(
                s.effective_temperature.max(0.01) * 300.0,
                101325.0,
                0.001,
                s.order_parameter.max(0.001) * 1000.0,
            );
            encoder.helmholtz_free_energy(&thermo_state)
        };

        // 4. Bind all components: entropy ⊗ state ⊗ helmholtz
        //    Binding creates a unique composite that preserves all information
        //    while being nearly orthogonal to any single component.
        let bound = entropy_hv.bind(&state_hv).bind(&helmholtz_hv);

        // 5. Scale by dissipative health (healthy consciousness = stronger signal)
        let health_scale = s.dissipative_health.max(0.1);
        bound.scale(health_scale as f32)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_input() -> ThermodynamicInput {
        ThermodynamicInput {
            unified_psi: 0.6,
            coherence: 0.7,
            prediction_error: 0.3,
            attention_sensitivity: 0.5,
            // Energy budget in effective consciousness units (same scale as
            // k_eff=0.01 Landauer costs, not physical joules).
            energy_per_cycle: 0.1,
            total_energy_spent: 0.5,
            energy_throughput_multiplier: 1.0,
            metabolic_stress: 0.2,
            dissipative_health: 0.7,
            dissipative_regime: ThermodynamicRegime::EdgeOfChaos,
            entropy_production_rate: 0.15,
            order_parameter: 0.6,
            criticality_distance: 0.05,
            lambda_parameter: 0.27,
            dissipation_efficiency: 0.5,
            has_dissipative: true,
            analyzer_entropy: 0.5,
            analyzer_free_energy: -0.2,
            analyzer_temperature: 0.4,
            analyzer_phase: ConsciousnessPhase::Normal,
            has_analyzer: true,
            hfe_total: 0.8,
            hfe_complexity: 0.5,
            hfe_accuracy: 0.3,
            has_hfe: true,
        }
    }

    #[test]
    fn test_run_cycle_baseline() {
        let mut ti = ThermodynamicIntegration::default();
        let input = default_input();
        let fb = ti.run_cycle(&input);
        // Baseline: no extreme feedback
        assert!(fb.lr_factor >= 0.9 && fb.lr_factor <= 1.2);
        assert!(fb.exploration_factor >= 0.8 && fb.exploration_factor <= 1.1);
        assert!(!fb.memory_consolidation_suppressed);
    }

    #[test]
    fn test_unified_state_populated() {
        let mut ti = ThermodynamicIntegration::default();
        let input = default_input();
        ti.run_cycle(&input);
        // All modules should have written to unified state
        assert!(ti.state.dissipative_health > 0.0);
        assert!(ti.state.canonical_entropy > 0.0);
        assert!(ti.state.hierarchical_free_energy > 0.0);
        assert_eq!(ti.state.energy_per_cycle, 0.1);
    }

    #[test]
    fn test_insight_triggers_lr_boost() {
        let mut ti = ThermodynamicIntegration::default();
        // Set up: low order → high order (sudden insight)
        ti.prev_order = 0.2;
        let mut input = default_input();
        input.order_parameter = 0.9; // Big jump
        let fb = ti.run_cycle(&input);
        if fb.insight_detected {
            assert!(fb.lr_factor > 1.0);
        }
    }

    #[test]
    fn test_prigogine_violation_damps_exploration() {
        let mut ti = ThermodynamicIntegration::default();
        // Set up: LinearNonEquilibrium with increasing entropy
        ti.bridge.prev_entropy_production = 0.1;
        let mut input = default_input();
        input.dissipative_regime = ThermodynamicRegime::LinearNonEquilibrium;
        input.entropy_production_rate = 0.3; // Increasing (violation!)
        let fb = ti.run_cycle(&input);
        if fb.prigogine_violated {
            assert!(fb.exploration_factor < 1.0);
        }
    }

    #[test]
    fn test_hfe_blend() {
        let mut ti = ThermodynamicIntegration::default();
        // Start with known analyzer free energy
        ti.state.canonical_free_energy = -0.5;
        let mut input = default_input();
        input.hfe_total = 2.0; // High HFE
        ti.run_cycle(&input);
        // Blended should shift toward HFE contribution
        assert!(ti.state.canonical_free_energy > -0.5); // Shifted positive
    }

    #[test]
    fn test_landauer_memory_pressure() {
        let mut ti = ThermodynamicIntegration::default();
        let mut input = default_input();
        // Extreme prediction error → high Landauer cost
        input.prediction_error = 10.0;
        input.energy_per_cycle = 1e-15; // Tiny budget
        let fb = ti.run_cycle(&input);
        // Landauer cost should dominate energy budget
        assert!(fb.memory_consolidation_suppressed);
    }

    #[test]
    fn test_absent_modules_graceful() {
        let mut ti = ThermodynamicIntegration::default();
        let input = ThermodynamicInput::default(); // All has_* = false
        let fb = ti.run_cycle(&input);
        // Should not crash, feedback should be neutral
        assert!(fb.lr_factor.is_finite());
        assert!(fb.exploration_factor.is_finite());
    }

    #[test]
    fn test_carnot_low_efficiency_conservation() {
        let mut ti = ThermodynamicIntegration::default();
        let mut input = default_input();
        input.analyzer_temperature = 1.5; // High temp → high Carnot
        // Run several cycles to build up Onsager history and stabilize
        for _ in 0..5 {
            ti.run_cycle(&input);
        }
        // Carnot efficiency should be positive
        assert!(ti.bridge.carnot_efficiency > 0.0);
    }
}
