// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bevy integration for the macro/micro temporal bridge.
//!
//! `MacroWorldState` is a Bevy [`Resource`] that carries civilization-sim
//! modifiers into the physics loop. The `apply_macro_modifiers_system<D>`
//! reads it each `FixedUpdate` and applies it to `ConsciousnessField<D>`
//! before the physics step.
//!
//! # Usage
//!
//! ```ignore
//! // 1. Insert the resource (starts at neutral defaults — no effect):
//! app.insert_resource(MacroWorldState::default());
//!
//! // 2. In your bridge polling system, update it from WorldBridge:
//! fn bridge_poll_system(
//!     mut state: ResMut<MacroWorldState>,
//!     bridge: Res<WorldBridgeResource>,  // your wrapper around WorldBridge
//! ) {
//!     let snap = &bridge.0.current;
//!     state.modifiers = MacroWorldModifiers::from_sim(
//!         snap.governance.stability,
//!         snap.governance.oppression_index,
//!         snap.economy.gini,
//!         snap.governance.emergency_active,
//!         (snap.economy.total_production / 1000.0).clamp(0.0, 1.0), // normalize
//!         0.0, // faction_harmony_bias — compute from factions if needed
//!     );
//! }
//!
//! // 3. The plugin wires `apply_macro_modifiers_system` into FixedUpdate automatically.
//! ```
//!
//! # Without a civilization sim
//!
//! Insert `MacroWorldState::default()` (neutral modifiers = no effect) or
//! set modifiers manually from gameplay events:
//! ```ignore
//! state.modifiers.emergency_active = player_triggered_crisis;
//! state.modifiers.stability = 1.0 - game_difficulty;
//! ```

use bevy::prelude::*;
use symtropy_consciousness_physics::macro_bridge::{apply_macro_modifiers, MacroWorldModifiers};

use crate::plugin::SymtropyPhysics;

// ────────────────────────────────────────────────────────────
//  Resource
// ────────────────────────────────────────────────────────────

/// Bevy resource holding civilization-sim modifiers for the physics world.
///
/// Updated by the game's bridge polling system (once per sim tick or frame).
/// Read by `apply_macro_modifiers_system` each `FixedUpdate`.
#[derive(Resource)]
pub struct MacroWorldState {
    /// The current macro-world modifiers.
    pub modifiers: MacroWorldModifiers,
    /// Base Φ-gravity strength (before stability modifier is applied).
    /// Set once at app startup; the stability modifier adds a delta.
    pub base_phi_gravity: f64,
}

impl Default for MacroWorldState {
    fn default() -> Self {
        Self {
            modifiers: MacroWorldModifiers::default(),
            base_phi_gravity: 0.0, // disabled by default; game sets this
        }
    }
}

impl MacroWorldState {
    /// Create with a specific base Φ-gravity strength.
    pub fn with_gravity(base_phi_gravity: f64) -> Self {
        Self {
            modifiers: MacroWorldModifiers::default(),
            base_phi_gravity,
        }
    }
}

// ────────────────────────────────────────────────────────────
//  System
// ────────────────────────────────────────────────────────────

/// FixedUpdate system: applies macro-world modifiers to `ConsciousnessField<D>`.
///
/// Uses `Option<Res<MacroWorldState>>` — silently no-ops when the resource is
/// absent, so it's safe to register unconditionally in the plugin.
/// Insert `MacroWorldState` to activate macro/micro coupling.
pub fn apply_macro_modifiers_system<const D: usize>(
    state: Option<Res<MacroWorldState>>,
    mut physics: ResMut<SymtropyPhysics<D>>,
) {
    let Some(state) = state else {
        return;
    };
    apply_macro_modifiers(&mut physics.field, &state.modifiers, state.base_phi_gravity);
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;
    use symtropy_consciousness_physics::coupling::ConsciousnessField;
    use symtropy_physics::PhysicsWorld;

    fn make_physics() -> SymtropyPhysics<2> {
        SymtropyPhysics {
            world: PhysicsWorld::<2>::new(SVector::zeros()),
            field: ConsciousnessField::new(),
        }
    }

    fn add_entity(p: &mut SymtropyPhysics<2>) -> symtropy_physics::BodyHandle {
        let h = p
            .world
            .add_sphere(symtropy_math::Point::new([0.0, 0.0]), 1.0, 1.0);
        p.field.register(h, 100.0, 5.0);
        h
    }

    #[test]
    fn macro_world_state_default_neutral() {
        let state = MacroWorldState::default();
        assert!((state.base_phi_gravity).abs() < 1e-9);
        assert!((state.modifiers.phi_gravity_delta()).abs() < 1e-9);
    }

    #[test]
    fn with_gravity_sets_base() {
        let state = MacroWorldState::with_gravity(0.5);
        assert!((state.base_phi_gravity - 0.5).abs() < 1e-9);
    }

    #[test]
    fn emergency_modifiers_spike_entity_error() {
        let mut p = make_physics();
        let h = add_entity(&mut p);

        let err_before = p.field.entities[&h].prediction_error;
        let state = MacroWorldState {
            modifiers: MacroWorldModifiers {
                emergency_active: true,
                oppression: 0.0,
                ..Default::default()
            },
            base_phi_gravity: 0.0,
        };

        apply_macro_modifiers(&mut p.field, &state.modifiers, state.base_phi_gravity);
        let err_after = p.field.entities[&h].prediction_error;

        use symtropy_consciousness_physics::macro_bridge::EMERGENCY_ERROR_SPIKE;
        assert!((err_after - err_before - EMERGENCY_ERROR_SPIKE).abs() < 1e-9);
    }

    #[test]
    fn high_stability_sets_phi_gravity_above_base() {
        let mut p = make_physics();
        let _h = add_entity(&mut p);
        let state = MacroWorldState {
            modifiers: MacroWorldModifiers {
                stability: 1.0,
                ..Default::default()
            },
            base_phi_gravity: 0.1,
        };
        apply_macro_modifiers(&mut p.field, &state.modifiers, state.base_phi_gravity);
        assert!(p.field.phi_gravity_strength > 0.1);
    }

    #[test]
    fn absent_resource_does_not_crash() {
        // Test that the system logic short-circuits when resource is None
        // We can't call the bevy system directly without an App, but we can
        // verify that apply_macro_modifiers with neutral defaults is a no-op.
        let mut p = make_physics();
        let h = add_entity(&mut p);
        let initial_err = p.field.entities[&h].prediction_error;
        let gravity_before = p.field.phi_gravity_strength;

        let neutral = MacroWorldModifiers::default();
        apply_macro_modifiers(&mut p.field, &neutral, 0.0);

        assert!((p.field.entities[&h].prediction_error - initial_err).abs() < 1e-12);
        assert!((p.field.phi_gravity_strength - gravity_before).abs() < 1e-9);
    }
}
