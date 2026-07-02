// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symtropy Bevy plugin: the primary entry point for using the engine in Bevy.

pub use crate::biometrics::{PlayerBiometrics, biometric_to_phi_system};
pub use crate::macro_bridge::{MacroWorldState, apply_macro_modifiers_system};
use bevy::prelude::*;
use nalgebra::SVector;
use symtropy_consciousness_physics::coupling::ConsciousnessField;
use symtropy_physics::world::PhysicsWorld;

/// Re-exported from [`symtropy-bevy-core`]. Identical type used by both
/// crates, so an entity with `PhysicsBody` works with either the permissive
/// or the Phi-coupled plugin.
pub use symtropy_bevy_core::{PhysicsBody, sync_transforms};

/// Resource that owns the physics world and coupling field.
#[derive(Resource, Reflect)]
pub struct SymtropyPhysics<const D: usize> {
    #[reflect(ignore)]
    pub world: PhysicsWorld<D>,
    #[reflect(ignore)]
    pub field: ConsciousnessField<D>,
}

impl<const D: usize> SymtropyPhysics<D> {
    pub fn new(gravity: SVector<f64, D>) -> Self {
        Self {
            world: PhysicsWorld::new(gravity),
            field: ConsciousnessField::new(),
        }
    }

    pub fn with_gravity(gravity: SVector<f64, D>) -> Self {
        Self::new(gravity)
    }
}

impl<const D: usize> Default for SymtropyPhysics<D> {
    fn default() -> Self {
        Self::new(SVector::zeros())
    }
}

/// Configuration for the Symtropy physics plugin.
pub struct SymtropyPhysicsPluginConfig<const D: usize> {
    pub gravity: SVector<f64, D>,
    /// Whether to enable debug gizmos (default: true if feature enabled).
    pub debug_gizmos: bool,
}

impl<const D: usize> Default for SymtropyPhysicsPluginConfig<D> {
    fn default() -> Self {
        Self {
            gravity: SVector::zeros(),
            debug_gizmos: cfg!(feature = "debug-gizmos"),
        }
    }
}

/// The main Symtropy physics plugin.
pub struct SymtropyPhysicsPlugin<const D: usize> {
    pub config: SymtropyPhysicsPluginConfig<D>,
}

impl<const D: usize> Default for SymtropyPhysicsPlugin<D> {
    fn default() -> Self {
        Self {
            config: SymtropyPhysicsPluginConfig::default(),
        }
    }
}

impl<const D: usize> SymtropyPhysicsPlugin<D> {
    /// Create a plugin with the given gravity.
    pub fn with_gravity(gravity: impl Into<SVector<f64, D>>) -> Self {
        Self {
            config: SymtropyPhysicsPluginConfig {
                gravity: gravity.into(),
                debug_gizmos: cfg!(feature = "debug-gizmos"),
            },
        }
    }

    /// Create with full configuration.
    pub fn with_config(config: SymtropyPhysicsPluginConfig<D>) -> Self {
        Self { config }
    }
}

impl<const D: usize> Plugin for SymtropyPhysicsPlugin<D> {
    fn build(&self, app: &mut App) {
        app.insert_resource(SymtropyPhysics::<{ D }>::with_gravity(self.config.gravity));

        app.add_systems(
            FixedUpdate,
            (
                biometric_to_phi_system::<{ D }>,
                apply_macro_modifiers_system::<{ D }>,
                physics_step::<{ D }>,
                sync_transforms_specialized::<{ D }>,
            )
                .chain(),
        );

        #[cfg(feature = "debug-gizmos")]
        if self.config.debug_gizmos {
            app.add_systems(
                Update,
                (
                    crate::debug::draw_debug_gizmos::<{ D }>,
                    crate::debug::draw_harmony_fields::<{ D }>,
                    crate::debug::draw_phi_heatmap::<{ D }>,
                ),
            );
        }

        #[cfg(feature = "inspector")]
        app.add_plugins(bevy_inspector_egui::quick::ResourceInspectorPlugin::<
            SymtropyPhysics<{ D }>,
        >::default());

        app.add_plugins(crate::animation::RoboticAnimationPlugin);
        app.add_plugins(crate::audio::RoboticAudioPlugin);
        app.add_plugins(crate::scripting::RoboticScriptingPlugin);
    }
}

/// Physics step system.
fn physics_step<const D: usize>(time: Res<Time>, mut physics: ResMut<SymtropyPhysics<D>>) {
    let dt = time.delta_secs() as f64;
    // Split mutable borrow to satisfy borrow checker
    let physics = &mut *physics;
    physics.world.step_with_callback(dt, &mut physics.field);
}

/// Internal wrapper to map SymtropyPhysics to the permissive sync_transforms.
fn sync_transforms_specialized<const D: usize>(
    physics: Res<SymtropyPhysics<D>>,
    mut query: Query<(&PhysicsBody, &mut Transform)>,
) {
    for (body_comp, mut transform) in query.iter_mut() {
        if let Some(body) = physics.world.body(body_comp.handle) {
            let pos = body.position();
            if D >= 1 {
                transform.translation.x = pos[0] as f32;
            }
            if D >= 2 {
                transform.translation.y = pos[1] as f32;
            }
            if D >= 3 {
                transform.translation.z = pos[2] as f32;
            }
        }
    }
}
