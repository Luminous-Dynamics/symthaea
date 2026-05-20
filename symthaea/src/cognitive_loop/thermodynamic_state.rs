// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Thermodynamic State: canonical snapshot aggregating all live
//! thermodynamic subsystems (DissipativeConsciousness, ConsciousnessThermodynamicsAnalyzer,
//! HierarchicalFreeEnergy, SubstrateManager, MetabolismManager).
//!
//! All modules write to this struct after computing; cross-couplings read from
//! the previous cycle's snapshot. EMA smoothing prevents jitter.
//!
//! Science: Sterling (2012) — allostatic timescales require smoothing to avoid
//! oscillatory feedback cascades.

use serde::{Deserialize, Serialize};

use crate::consciousness::consciousness_thermodynamics::ConsciousnessPhase;
use crate::consciousness::dissipative_consciousness::ThermodynamicRegime;
use symthaea_core::physics::thermodynamics::ThermodynamicLedger;

use super::thresholds;

/// Unified thermodynamic snapshot aggregating all live subsystems.
///
/// Written to by 3 modules (dissipative, analyzer, HFE) each cycle.
/// Read from by cross-couplings and telemetry export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedThermodynamicState {
    // ═══════════════════════════════════════════════════════════════════════
    // Core Ledger (Joules)
    // ═══════════════════════════════════════════════════════════════════════
    /// Explicit energy ledger for cognitive operations.
    pub ledger: ThermodynamicLedger,

    // ═══════════════════════════════════════════════════════════════════════
    // Canonical state (reconciled from multiple sources)
    // ═══════════════════════════════════════════════════════════════════════
    /// Shannon entropy from ConsciousnessThermodynamicsAnalyzer [0, 1].
    pub canonical_entropy: f64,
    /// Helmholtz free energy, blended from Analyzer + HFE.
    pub canonical_free_energy: f64,
    /// Effective temperature from Analyzer (variance + arousal).
    pub effective_temperature: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Dissipative consciousness (Prigogine far-from-equilibrium)
    // ═══════════════════════════════════════════════════════════════════════
    /// Entropy production rate dS/dt from DissipativeConsciousness [-1, 1].
    pub entropy_production_rate: f64,
    /// Order parameter √(Φ × coherence) [0, 1].
    pub order_parameter: f64,
    /// Distance from edge of chaos (|λ - 0.273|).
    pub criticality_distance: f64,
    /// Langton's λ parameter (optimal ≈ 0.273).
    pub lambda_parameter: f64,
    /// Dissipative health score [0, 1].
    pub dissipative_health: f64,
    /// Dissipation efficiency (order per unit entropy).
    pub dissipation_efficiency: f64,
    /// Current thermodynamic regime.
    pub regime: ThermodynamicRegime,

    // ═══════════════════════════════════════════════════════════════════════
    // Consciousness thermodynamics analyzer (7D phase state)
    // ═══════════════════════════════════════════════════════════════════════
    /// Consciousness phase (Frozen/Normal/Critical/Flow/Chaotic/Unified).
    pub consciousness_phase: ConsciousnessPhase,

    // ═══════════════════════════════════════════════════════════════════════
    // Hierarchical free energy (Friston predictive processing)
    // ═══════════════════════════════════════════════════════════════════════
    /// Total hierarchical free energy across all levels.
    pub hierarchical_free_energy: f64,
    /// HFE complexity component (KL divergence from prior).
    pub hfe_complexity: f64,
    /// HFE accuracy component (expected log-likelihood).
    pub hfe_accuracy: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Energy budget (SubstrateManager + MetabolismManager)
    // ═══════════════════════════════════════════════════════════════════════
    /// Substrate energy per cycle (joules).
    pub energy_per_cycle: f64,
    /// Cumulative energy spent since init (joules).
    pub total_energy_spent: f64,
    /// Metabolic stress composite from real hardware [0, 1].
    pub metabolic_stress: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // Derived quantities (computed from above, populated later phases)
    // ═══════════════════════════════════════════════════════════════════════
    /// Carnot efficiency (Phase 4: 1 - T_cold/T_hot). 0.0 until wired.
    pub carnot_efficiency: f64,
    /// Landauer bound for memory consolidation this cycle (joules). 0.0 until wired.
    pub landauer_bound_joules: f64,
}

impl Default for UnifiedThermodynamicState {
    fn default() -> Self {
        Self {
            ledger: ThermodynamicLedger::new(6.0), // 6 Joules capacity
            canonical_entropy: 0.5,
            canonical_free_energy: 0.0,

            effective_temperature: 0.3, // Warm default (Normal phase)
            entropy_production_rate: 0.0,
            order_parameter: 0.0,
            criticality_distance: 0.273, // Max distance from target
            lambda_parameter: 0.0,
            dissipative_health: 0.0,
            dissipation_efficiency: 0.0,
            regime: ThermodynamicRegime::Equilibrium,
            consciousness_phase: ConsciousnessPhase::Normal,
            hierarchical_free_energy: 0.0,
            hfe_complexity: 0.0,
            hfe_accuracy: 0.0,
            energy_per_cycle: 0.0,
            total_energy_spent: 0.0,
            metabolic_stress: 0.0,
            carnot_efficiency: 0.0,
            landauer_bound_joules: 0.0,
        }
    }
}

impl UnifiedThermodynamicState {
    /// EMA-smooth a value toward a target. Returns updated value.
    #[inline]
    fn ema(current: f64, target: f64, alpha: f64) -> f64 {
        let result = current + alpha * (target - current);
        if result.is_finite() {
            result
        } else {
            current // NaN guard: retain previous value
        }
    }

    /// Update dissipative consciousness fields (Phase B, every cycle).
    pub(crate) fn update_dissipative(
        &mut self,
        health: f64,
        regime: ThermodynamicRegime,
        entropy_rate: f64,
        entropy_production: f64,
        order: f64,
        criticality: f64,
        lambda: f64,
        efficiency: f64,
    ) {
        let alpha = thresholds::THERMO_UNIFIED_EMA_ALPHA as f64;
        self.dissipative_health = Self::ema(self.dissipative_health, health, alpha);
        self.regime = regime;
        self.entropy_production_rate =
            Self::ema(self.entropy_production_rate, entropy_production, alpha);
        self.order_parameter = Self::ema(self.order_parameter, order, alpha);
        self.criticality_distance = criticality; // Instant (derived from λ)
        self.lambda_parameter = lambda; // Instant
        self.dissipation_efficiency = Self::ema(self.dissipation_efficiency, efficiency, alpha);
        // Also update entropy_production_rate from the raw rate
        let _ = entropy_rate; // Captured in entropy_production above
    }

    /// Update ConsciousnessThermodynamicsAnalyzer fields (late consciousness, urgency-gated).
    pub(crate) fn update_analyzer(
        &mut self,
        entropy: f64,
        free_energy: f64,
        temperature: f64,
        phase: ConsciousnessPhase,
    ) {
        let alpha = thresholds::THERMO_UNIFIED_EMA_ALPHA as f64;
        self.canonical_entropy = Self::ema(self.canonical_entropy, entropy, alpha);
        self.canonical_free_energy = Self::ema(self.canonical_free_energy, free_energy, alpha);
        self.effective_temperature = Self::ema(self.effective_temperature, temperature, alpha);
        self.consciousness_phase = phase;
    }

    /// Update HierarchicalFreeEnergy fields (late consciousness, urgency-gated).
    pub(crate) fn update_hfe(&mut self, total_fe: f64, complexity: f64, accuracy: f64) {
        let alpha = thresholds::THERMO_UNIFIED_EMA_ALPHA as f64;
        self.hierarchical_free_energy = Self::ema(self.hierarchical_free_energy, total_fe, alpha);
        self.hfe_complexity = Self::ema(self.hfe_complexity, complexity, alpha);
        self.hfe_accuracy = Self::ema(self.hfe_accuracy, accuracy, alpha);
    }

    /// Update energy budget fields from SubstrateManager + MetabolismManager.
    pub(crate) fn update_energy(
        &mut self,
        energy_per_cycle: f64,
        total_spent: f64,
        metabolic: f32,
        cognitive_load: f32,
    ) {
        self.energy_per_cycle = energy_per_cycle;
        self.total_energy_spent = total_spent;
        self.metabolic_stress = if metabolic.is_finite() {
            metabolic
        } else {
            self.metabolic_stress
        };

        // Deduct cognitive load from ledger (mapping load 1.0 to 1 Joule)
        if cognitive_load > 0.0 {
            self.ledger.deduct(cognitive_load as f64);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values_sensible() {
        let state = UnifiedThermodynamicState::default();
        assert_eq!(state.canonical_entropy, 0.0);
        assert_eq!(state.effective_temperature, 0.3);
        assert_eq!(state.regime, ThermodynamicRegime::Equilibrium);
        assert_eq!(state.consciousness_phase, ConsciousnessPhase::Normal);
        assert_eq!(state.criticality_distance, 0.273);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut state = UnifiedThermodynamicState::default();
        state.update_dissipative(
            0.8,
            ThermodynamicRegime::EdgeOfChaos,
            0.3,
            0.3,
            0.7,
            0.05,
            0.27,
            0.5,
        );
        // After one update with alpha=0.05, should be 0 + 0.05*(0.8 - 0) = 0.04
        assert!(
            (state.dissipative_health - 0.04).abs() < 0.01,
            "expected ~0.04, got {}",
            state.dissipative_health,
        );
        // Second update should converge further
        state.update_dissipative(
            0.8,
            ThermodynamicRegime::EdgeOfChaos,
            0.3,
            0.3,
            0.7,
            0.05,
            0.27,
            0.5,
        );
        assert!(state.dissipative_health > 0.04);
        assert!(state.dissipative_health < 0.8);
    }

    #[test]
    fn test_nan_guard_in_ema() {
        let mut state = UnifiedThermodynamicState::default();
        state.canonical_entropy = 0.5;
        state.update_analyzer(f64::NAN, 0.0, 0.0, ConsciousnessPhase::Normal);
        // NaN target should retain previous value
        assert_eq!(state.canonical_entropy, 0.5);
    }

    #[test]
    fn test_update_analyzer() {
        let mut state = UnifiedThermodynamicState::default();
        state.update_analyzer(0.7, -0.3, 0.5, ConsciousnessPhase::Critical);
        assert!(state.canonical_entropy > 0.0);
        assert!(state.canonical_free_energy < 0.0);
        assert_eq!(state.consciousness_phase, ConsciousnessPhase::Critical);
    }

    #[test]
    fn test_update_hfe() {
        let mut state = UnifiedThermodynamicState::default();
        state.update_hfe(1.5, 0.8, 0.7);
        assert!(state.hierarchical_free_energy > 0.0);
        assert!(state.hfe_complexity > 0.0);
    }

    #[test]
    fn test_update_energy() {
        let mut state = UnifiedThermodynamicState::default();
        state.update_energy(1e-10, 5e-8, 0.45, 0.0);
        assert_eq!(state.energy_per_cycle, 1e-10);
        assert_eq!(state.total_energy_spent, 5e-8);
        assert_eq!(state.metabolic_stress, 0.45);
    }

    #[test]
    fn test_energy_nan_guard() {
        let mut state = UnifiedThermodynamicState::default();
        state.metabolic_stress = 0.3;
        state.update_energy(0.0, 0.0, f32::NAN, 0.0);
        assert_eq!(state.metabolic_stress, 0.3); // Retained
    }
}
