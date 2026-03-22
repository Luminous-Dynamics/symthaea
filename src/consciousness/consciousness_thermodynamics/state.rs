// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::config::default_instant;
use super::critical::{CriticalExponents, TransitionOrder};

/// Thermodynamic state of consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicState {
    /// Current entropy (disorder measure)
    pub entropy: f64,

    /// Internal energy (total consciousness energy)
    pub internal_energy: f64,

    /// Free energy F = U - TS (capacity for work)
    pub free_energy: f64,

    /// Temperature (activation/exploration level)
    pub temperature: f64,

    /// Heat (energy transferred due to temperature difference)
    pub heat: f64,

    /// Work (directed energy expenditure)
    pub work: f64,

    /// Chemical potential (tendency to change state)
    pub chemical_potential: f64,

    /// Pressure (compression in consciousness space)
    pub pressure: f64,

    /// Volume (extent of consciousness state space)
    pub volume: f64,

    /// Enthalpy H = U + PV
    pub enthalpy: f64,

    /// Gibbs free energy G = H - TS
    pub gibbs_free_energy: f64,

    /// Current phase of consciousness
    pub phase: ConsciousnessPhase,

    /// Timestamp
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

impl Default for ThermodynamicState {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            internal_energy: 1.0,
            free_energy: 0.5,
            temperature: 1.0,
            heat: 0.0,
            work: 0.0,
            chemical_potential: 0.0,
            pressure: 1.0,
            volume: 1.0,
            enthalpy: 2.0,
            gibbs_free_energy: 1.0,
            phase: ConsciousnessPhase::Normal,
            timestamp: Instant::now(),
        }
    }
}

/// Phases of consciousness (like phases of matter)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessPhase {
    /// Low temperature: Frozen, rigid thinking
    Frozen,
    /// Ordered phase: Normal waking consciousness
    Normal,
    /// Critical point: Edge of chaos, maximum creativity
    Critical,
    /// High temperature: Chaotic, fragmented consciousness
    Chaotic,
    /// Superfluid: Flow state, frictionless consciousness
    Flow,
    /// Condensate: Meditative unity, Bose-Einstein-like
    Unified,
}

impl ConsciousnessPhase {
    /// Get characteristic temperature range for this phase
    pub fn temperature_range(&self) -> (f64, f64) {
        match self {
            Self::Frozen => (0.0, 0.2),
            Self::Normal => (0.2, 0.4),
            Self::Critical => (0.4, 0.6),
            Self::Chaotic => (0.8, 1.0),
            Self::Flow => (0.3, 0.5),
            Self::Unified => (0.0, 0.3),
        }
    }

    /// Get entropy characteristic of this phase
    pub fn typical_entropy(&self) -> f64 {
        match self {
            Self::Frozen => 0.1,
            Self::Normal => 0.4,
            Self::Critical => 0.6,
            Self::Chaotic => 0.9,
            Self::Flow => 0.3,
            Self::Unified => 0.2,
        }
    }
}

/// A phase transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    /// Phase before transition
    pub from_phase: ConsciousnessPhase,

    /// Phase after transition
    pub to_phase: ConsciousnessPhase,

    /// Temperature at transition
    pub transition_temperature: f64,

    /// Latent heat (energy absorbed/released)
    pub latent_heat: f64,

    /// Order parameter jump
    pub order_parameter_change: f64,

    /// Transition order (1st order = discontinuous, 2nd order = continuous)
    pub transition_order: TransitionOrder,

    /// Critical exponents (for 2nd order transitions)
    pub critical_exponents: Option<CriticalExponents>,

    /// Timestamp
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let state = ThermodynamicState::default();
        assert!((state.entropy - 0.5).abs() < f64::EPSILON);
        assert!((state.temperature - 1.0).abs() < f64::EPSILON);
        assert_eq!(state.phase, ConsciousnessPhase::Normal);
    }

    #[test]
    fn test_consciousness_phase_temperature_ranges() {
        // Each phase should return a valid (low, high) range where low < high
        let phases = [
            ConsciousnessPhase::Frozen,
            ConsciousnessPhase::Normal,
            ConsciousnessPhase::Critical,
            ConsciousnessPhase::Chaotic,
            ConsciousnessPhase::Flow,
            ConsciousnessPhase::Unified,
        ];
        for phase in &phases {
            let (low, high) = phase.temperature_range();
            assert!(
                low < high,
                "{:?} temperature range invalid: ({}, {})",
                phase,
                low,
                high,
            );
            assert!(low >= 0.0, "{:?} lower bound negative: {}", phase, low);
            assert!(high <= 1.0, "{:?} upper bound exceeds 1.0: {}", phase, high);
        }

        // Frozen should be the coldest range
        let (_, frozen_hi) = ConsciousnessPhase::Frozen.temperature_range();
        let (chaotic_lo, _) = ConsciousnessPhase::Chaotic.temperature_range();
        assert!(
            frozen_hi < chaotic_lo,
            "Frozen range should be entirely below Chaotic range",
        );
    }

    #[test]
    fn test_consciousness_phase_typical_entropy() {
        // Frozen < Normal < Critical < Chaotic entropy ordering
        let frozen = ConsciousnessPhase::Frozen.typical_entropy();
        let normal = ConsciousnessPhase::Normal.typical_entropy();
        let critical = ConsciousnessPhase::Critical.typical_entropy();
        let chaotic = ConsciousnessPhase::Chaotic.typical_entropy();

        assert!(
            frozen < normal,
            "Frozen ({}) should < Normal ({})",
            frozen,
            normal
        );
        assert!(
            normal < critical,
            "Normal ({}) should < Critical ({})",
            normal,
            critical
        );
        assert!(
            critical < chaotic,
            "Critical ({}) should < Chaotic ({})",
            critical,
            chaotic
        );
    }

    #[test]
    fn test_phase_transition_creation() {
        let transition = PhaseTransition {
            from_phase: ConsciousnessPhase::Normal,
            to_phase: ConsciousnessPhase::Critical,
            transition_temperature: 0.45,
            latent_heat: 0.1,
            order_parameter_change: 0.3,
            transition_order: TransitionOrder::SecondOrder,
            critical_exponents: Some(CriticalExponents::default()),
            timestamp: Instant::now(),
        };

        assert_eq!(transition.from_phase, ConsciousnessPhase::Normal);
        assert_eq!(transition.to_phase, ConsciousnessPhase::Critical);
        assert!((transition.transition_temperature - 0.45).abs() < f64::EPSILON);
        assert!((transition.latent_heat - 0.1).abs() < f64::EPSILON);
        assert!((transition.order_parameter_change - 0.3).abs() < f64::EPSILON);
        assert_eq!(transition.transition_order, TransitionOrder::SecondOrder);
        assert!(transition.critical_exponents.is_some());
    }

    #[test]
    fn test_flow_phase_low_entropy() {
        // Flow state = frictionless consciousness, should have lower entropy than Normal
        let flow_entropy = ConsciousnessPhase::Flow.typical_entropy();
        let normal_entropy = ConsciousnessPhase::Normal.typical_entropy();
        assert!(
            flow_entropy < normal_entropy,
            "Flow ({}) should have lower entropy than Normal ({})",
            flow_entropy,
            normal_entropy,
        );
    }

    #[test]
    fn test_unified_phase_low_entropy() {
        // Unified = meditative condensate, should have lower entropy than Normal
        let unified_entropy = ConsciousnessPhase::Unified.typical_entropy();
        let normal_entropy = ConsciousnessPhase::Normal.typical_entropy();
        assert!(
            unified_entropy < normal_entropy,
            "Unified ({}) should have lower entropy than Normal ({})",
            unified_entropy,
            normal_entropy,
        );
    }

    #[test]
    fn test_thermodynamic_state_defaults_valid() {
        // Free energy F = U - TS
        let state = ThermodynamicState::default();
        let expected_free_energy = state.internal_energy - state.temperature * state.entropy;
        assert!(
            (state.free_energy - expected_free_energy).abs() < f64::EPSILON,
            "F ({}) should equal U - TS ({})",
            state.free_energy,
            expected_free_energy,
        );
    }

    #[test]
    fn test_enthalpy_default() {
        // Enthalpy H = U + PV
        let state = ThermodynamicState::default();
        let expected_enthalpy = state.internal_energy + state.pressure * state.volume;
        assert!(
            (state.enthalpy - expected_enthalpy).abs() < f64::EPSILON,
            "H ({}) should equal U + PV ({})",
            state.enthalpy,
            expected_enthalpy,
        );
    }
}
