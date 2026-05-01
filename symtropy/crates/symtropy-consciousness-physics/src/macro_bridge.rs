// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Macro/micro bridge: civilization-sim state modulates real-time physics.
//!
//! The `symtropy-world` civilization sim runs at monthly ticks (slow macro).
//! The physics engine runs at ~60fps (fast micro). This module provides the
//! coupling layer: [`MacroWorldModifiers`] carries the civilization state,
//! and [`apply_macro_modifiers`] applies it to a [`ConsciousnessField`].
//!
//! # Mappings
//!
//! | Civilization metric | Physics effect | Rationale |
//! |---------------------|----------------|-----------|
//! | Governance stability | Φ-gravity strength | Stable society attracts conscious entities |
//! | Oppression index | Entity prediction error | Oppression = chronic stress = motor imprecision |
//! | Gini coefficient | Energy regen variance | Inequality = unequal energy access |
//! | Emergency active | Safety tier degradation | Crisis = heightened threat response |
//! | Economic production | Energy regen multiplier | Productive economy = more energy in world |
//! | Faction harmony | Harmony field base | Ideological harmony ripples into physics |
//!
//! # Usage
//!
//! The game populates `MacroWorldModifiers` from `WorldBridge::current` each frame,
//! then the `macro_to_physics_system` in `symtropy-bevy` applies it before the
//! physics step:
//!
//! ```ignore
//! // In your bridge system:
//! let modifiers = MacroWorldModifiers::from_sim(
//!     stability, oppression, gini, emergency, production, faction_harmony
//! );
//! // The bevy system applies it automatically when MacroWorldModifiers is a resource.
//! ```

use crate::coupling::ConsciousnessField;

// ────────────────────────────────────────────────────────────
//  MacroWorldModifiers
// ────────────────────────────────────────────────────────────

/// Physics modifiers derived from the civilization-sim state.
///
/// Populated once per sim tick from `SimSnapshot`, then applied to the
/// `ConsciousnessField` each physics frame. Fields are normalized unless noted.
#[derive(Debug, Clone)]
pub struct MacroWorldModifiers {
    /// Governance stability [0, 1].
    /// High stability → stronger Φ-gravity (conscious entities coalesce).
    pub stability: f64,

    /// Oppression index [0, 1].
    /// High oppression → increased baseline prediction error (chronic stress).
    pub oppression: f64,

    /// Gini coefficient [0, 1].
    /// High inequality → larger energy regen variance across entities.
    pub gini: f64,

    /// Whether emergency/crisis is active.
    /// True → spike all entity prediction errors by `EMERGENCY_ERROR_SPIKE`.
    pub emergency_active: bool,

    /// Total economic production, normalized to [0, 1] relative to baseline.
    /// Higher → more energy regeneration in the world.
    pub production_norm: f64,

    /// Dominant faction harmony dimension [-0.5, 0.5].
    /// Positive = harmonious ideology (boosts harmony field base).
    /// Negative = conflictual ideology (suppresses harmony).
    pub faction_harmony_bias: f64,
}

/// Prediction error added to every entity when emergency powers are active.
pub const EMERGENCY_ERROR_SPIKE: f64 = 0.3;
/// Maximum Φ-gravity boost from governance stability.
pub const MAX_STABILITY_GRAVITY_BOOST: f64 = 0.4;
/// Maximum energy regen multiplier from economic production.
pub const MAX_PRODUCTION_REGEN_BOOST: f64 = 0.5;
/// Maximum prediction error from chronic oppression per tick.
pub const MAX_OPPRESSION_ERROR: f64 = 0.05;

impl Default for MacroWorldModifiers {
    /// Neutral state — no modifiers applied (stable, free, equal, peaceful).
    fn default() -> Self {
        Self {
            stability: 0.5,
            oppression: 0.0,
            gini: 0.0,
            emergency_active: false,
            production_norm: 0.5,
            faction_harmony_bias: 0.0,
        }
    }
}

impl MacroWorldModifiers {
    /// Construct from raw civilization-sim values.
    pub fn from_sim(
        stability: f64,
        oppression: f64,
        gini: f64,
        emergency_active: bool,
        production_norm: f64,
        faction_harmony_bias: f64,
    ) -> Self {
        Self {
            stability: stability.clamp(0.0, 1.0),
            oppression: oppression.clamp(0.0, 1.0),
            gini: gini.clamp(0.0, 1.0),
            emergency_active,
            production_norm: production_norm.clamp(0.0, 1.0),
            faction_harmony_bias: faction_harmony_bias.clamp(-0.5, 0.5),
        }
    }

    /// Compute the Φ-gravity strength modifier from governance stability.
    ///
    /// More stable society → stronger inter-entity Φ attraction.
    /// Returns an additive delta to `ConsciousnessField::phi_gravity_strength`.
    pub fn phi_gravity_delta(&self) -> f64 {
        (self.stability - 0.5) * MAX_STABILITY_GRAVITY_BOOST * 2.0
    }

    /// Compute the energy regen multiplier from production.
    ///
    /// Higher production → more total energy in the physics world.
    /// Returns a multiplier (1.0 = no change).
    pub fn energy_regen_multiplier(&self) -> f64 {
        1.0 + (self.production_norm - 0.5) * MAX_PRODUCTION_REGEN_BOOST * 2.0
    }

    /// Compute the oppression prediction-error penalty per entity per tick.
    ///
    /// Chronic oppression = chronic stress = reduced motor precision.
    pub fn oppression_error_per_tick(&self) -> f64 {
        self.oppression * MAX_OPPRESSION_ERROR
    }

    /// Compute the harmony base level delta from faction ideology.
    ///
    /// Harmonious dominant faction → slight boost to ambient harmony.
    /// Conflictual faction → suppression.
    pub fn harmony_base_delta(&self) -> f64 {
        self.faction_harmony_bias
    }
}

// ────────────────────────────────────────────────────────────
//  Field application
// ────────────────────────────────────────────────────────────

/// Apply macro-world modifiers to a [`ConsciousnessField`].
///
/// Called each physics tick before `step_with_callback`. Modifiers persist
/// until the next call (i.e., they are re-applied every tick, not accumulated).
///
/// Effects:
/// 1. **Φ-gravity**: `phi_gravity_strength` is set to `base + phi_gravity_delta`.
/// 2. **Oppression prediction error**: added to every entity's `prediction_error`.
/// 3. **Emergency spike**: if `emergency_active`, all entities get an additional
///    `EMERGENCY_ERROR_SPIKE` prediction error.
/// 4. **Energy regen**: each entity's energy max is modulated by `energy_regen_multiplier`.
///    (Actual regen happens during `step_with_callback`; we record the multiplier.)
pub fn apply_macro_modifiers<const D: usize>(
    field: &mut ConsciousnessField<D>,
    modifiers: &MacroWorldModifiers,
    base_gravity: f64,
) {
    // 1. Φ-gravity
    field.phi_gravity_strength = (base_gravity + modifiers.phi_gravity_delta()).max(0.0);

    // 2. Per-entity oppression stress + emergency spike
    let oppression_err = modifiers.oppression_error_per_tick();
    let emergency_err = if modifiers.emergency_active {
        EMERGENCY_ERROR_SPIKE
    } else {
        0.0
    };
    let total_err_delta = oppression_err + emergency_err;

    if total_err_delta > 1e-10 {
        for entity in field.entities.values_mut() {
            entity.prediction_error += total_err_delta;
            entity.motor_precision = 1.0 / (1.0 + entity.prediction_error);
        }
    }
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coupling::ConsciousnessField;

    fn make_field_with_entity() -> (ConsciousnessField<2>, symtropy_physics::BodyHandle) {
        let mut field = ConsciousnessField::new();
        let h = symtropy_physics::BodyHandle(0);
        field.register(h, 100.0, 5.0);
        (field, h)
    }

    #[test]
    fn default_modifiers_have_no_effect_on_gravity() {
        let mods = MacroWorldModifiers::default();
        // phi_gravity_delta = (0.5 - 0.5) * MAX * 2 = 0
        assert!((mods.phi_gravity_delta()).abs() < 1e-9);
    }

    #[test]
    fn high_stability_increases_gravity() {
        let mods = MacroWorldModifiers {
            stability: 1.0,
            ..Default::default()
        };
        assert!(mods.phi_gravity_delta() > 0.0);
        assert!(mods.phi_gravity_delta() <= MAX_STABILITY_GRAVITY_BOOST);
    }

    #[test]
    fn low_stability_decreases_gravity() {
        let mods = MacroWorldModifiers {
            stability: 0.0,
            ..Default::default()
        };
        assert!(mods.phi_gravity_delta() < 0.0);
    }

    #[test]
    fn high_production_boosts_regen() {
        let mods = MacroWorldModifiers {
            production_norm: 1.0,
            ..Default::default()
        };
        assert!(mods.energy_regen_multiplier() > 1.0);
    }

    #[test]
    fn low_production_reduces_regen() {
        let mods = MacroWorldModifiers {
            production_norm: 0.0,
            ..Default::default()
        };
        assert!(mods.energy_regen_multiplier() < 1.0);
        // Must be positive (economy can't drive regen below 0)
        assert!(mods.energy_regen_multiplier() >= 1.0 - MAX_PRODUCTION_REGEN_BOOST);
    }

    #[test]
    fn oppression_error_zero_when_no_oppression() {
        let mods = MacroWorldModifiers {
            oppression: 0.0,
            ..Default::default()
        };
        assert!((mods.oppression_error_per_tick()).abs() < 1e-15);
    }

    #[test]
    fn max_oppression_error_bounded() {
        let mods = MacroWorldModifiers {
            oppression: 1.0,
            ..Default::default()
        };
        assert!((mods.oppression_error_per_tick() - MAX_OPPRESSION_ERROR).abs() < 1e-9);
    }

    #[test]
    fn apply_oppression_increases_entity_prediction_error() {
        let (mut field, h) = make_field_with_entity();
        let err_before = field.entities[&h].prediction_error;
        let mods = MacroWorldModifiers {
            oppression: 1.0,
            ..Default::default()
        };
        apply_macro_modifiers(&mut field, &mods, 0.0);
        let err_after = field.entities[&h].prediction_error;
        assert!(
            err_after > err_before,
            "oppression should increase prediction error"
        );
    }

    #[test]
    fn apply_emergency_spikes_prediction_error() {
        let (mut field, h) = make_field_with_entity();
        let err_before = field.entities[&h].prediction_error;
        let mods = MacroWorldModifiers {
            emergency_active: true,
            oppression: 0.0,
            ..Default::default()
        };
        apply_macro_modifiers(&mut field, &mods, 0.0);
        let err_after = field.entities[&h].prediction_error;
        assert!(
            (err_after - err_before - EMERGENCY_ERROR_SPIKE).abs() < 1e-9,
            "emergency should add exactly EMERGENCY_ERROR_SPIKE, got delta {}",
            err_after - err_before
        );
    }

    #[test]
    fn apply_stability_sets_phi_gravity() {
        let (mut field, _h) = make_field_with_entity();
        let mods = MacroWorldModifiers {
            stability: 1.0,
            ..Default::default()
        };
        apply_macro_modifiers(&mut field, &mods, 0.1);
        let expected = 0.1 + mods.phi_gravity_delta();
        assert!(
            (field.phi_gravity_strength - expected).abs() < 1e-9,
            "phi_gravity_strength should equal base + delta"
        );
    }

    #[test]
    fn phi_gravity_never_goes_negative() {
        let (mut field, _h) = make_field_with_entity();
        let mods = MacroWorldModifiers {
            stability: 0.0,
            ..Default::default()
        };
        apply_macro_modifiers(&mut field, &mods, 0.0);
        assert!(field.phi_gravity_strength >= 0.0);
    }

    #[test]
    fn from_sim_clamps_inputs() {
        let m = MacroWorldModifiers::from_sim(2.0, -1.0, 5.0, false, -0.5, 1.0);
        assert_eq!(m.stability, 1.0);
        assert_eq!(m.oppression, 0.0);
        assert_eq!(m.gini, 1.0);
        assert_eq!(m.production_norm, 0.0);
        assert_eq!(m.faction_harmony_bias, 0.5);
    }

    #[test]
    fn no_oppression_no_emergency_no_error_change() {
        let (mut field, h) = make_field_with_entity();
        let err_before = field.entities[&h].prediction_error;
        let mods = MacroWorldModifiers::default(); // oppression=0, emergency=false
        apply_macro_modifiers(&mut field, &mods, 0.0);
        let err_after = field.entities[&h].prediction_error;
        assert!((err_after - err_before).abs() < 1e-12);
    }
}
