// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The core Bevy plugin and systems.

use bevy::prelude::*;
use nalgebra::SVector;
use symtropy_bevy_core::PhysicsBody;
use symtropy_consciousness_physics::coupling::ConsciousnessField;
use symtropy_physics::PhysicsWorld;

use crate::biometrics::biometric_to_phi_system;
use crate::macro_bridge::apply_macro_modifiers_system;

/// Combined physics world + Phi-coupling field as a Bevy Resource.
///
/// Access this in your systems to add bodies, query Phi, consume energy, etc.
///
/// ```ignore
/// fn my_system(mut physics: ResMut<SymtropyPhysics<2>>) {
///     let handle = physics.world.add_sphere(Point::new([0.0, 5.0]), 1.0, 1.0);
///     physics.field.register(handle, 100.0, 10.0);
/// }
/// ```
#[derive(Resource)]
pub struct SymtropyPhysics<const D: usize> {
    /// The N-dimensional rigid body world.
    pub world: PhysicsWorld<D>,
    /// The Phi-coupling field (implements `PhysicsCallback<D>`).
    pub field: ConsciousnessField<D>,
}

impl<const D: usize> Default for SymtropyPhysics<D> {
    fn default() -> Self {
        Self {
            world: PhysicsWorld::new(SVector::zeros()),
            field: ConsciousnessField::new(),
        }
    }
}

impl<const D: usize> SymtropyPhysics<D> {
    /// Create with custom gravity.
    pub fn with_gravity(gravity: SVector<f64, D>) -> Self {
        Self {
            world: PhysicsWorld::new(gravity),
            field: ConsciousnessField::new(),
        }
    }
}

// `PhysicsBody` is now re-exported from `symtropy-bevy-core` (identical
// definition; see lib.rs). This crate no longer owns the type — any entity
// marked with `PhysicsBody` is compatible with both the permissive
// `BevyPhysicsPlugin` and the Phi-coupled `SymtropyPhysicsPlugin`.

/// Plugin configuration.
pub struct SymtropyPhysicsPluginConfig<const D: usize> {
    /// Initial gravity vector.
    pub gravity: SVector<f64, D>,
    /// Whether to enable debug gizmo rendering.
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

/// Drop-in Bevy plugin for Phi-coupled N-dimensional physics.
///
/// # Usage
///
/// ```ignore
/// // 2D with default settings (no gravity, debug gizmos on)
/// app.add_plugins(SymtropyPhysicsPlugin::<2>::default());
///
/// // 3D with gravity
/// app.add_plugins(SymtropyPhysicsPlugin::<3>::with_gravity([0.0, -9.81, 0.0]));
/// ```
pub struct SymtropyPhysicsPlugin<const D: usize> {
    config: SymtropyPhysicsPluginConfig<D>,
}

impl<const D: usize> Default for SymtropyPhysicsPlugin<D> {
    fn default() -> Self {
        Self {
            config: SymtropyPhysicsPluginConfig::default(),
        }
    }
}

impl<const D: usize> SymtropyPhysicsPlugin<D> {
    /// Create with custom gravity.
    pub fn with_gravity(gravity: [f64; D]) -> Self {
        Self {
            config: SymtropyPhysicsPluginConfig {
                gravity: SVector::from(gravity),
                debug_gizmos: cfg!(feature = "debug-gizmos"),
            },
        }
    }

    /// Create with full configuration.
    pub fn with_config(config: SymtropyPhysicsPluginConfig<D>) -> Self {
        Self { config }
    }
}

// We need separate Plugin impls for each D we support (Bevy Plugin trait isn't const-generic).
// Implement for the common dimensions.

impl Plugin for SymtropyPhysicsPlugin<2> {
    fn build(&self, app: &mut App) {
        app.insert_resource(SymtropyPhysics::<2>::with_gravity(self.config.gravity));
        // biometric and macro systems use Option params — no-ops when resources absent.
        app.add_systems(
            FixedUpdate,
            (
                biometric_to_phi_system::<2>,
                apply_macro_modifiers_system::<2>,
                physics_step::<2>,
                sync_transforms::<2>,
            )
                .chain(),
        );
        #[cfg(feature = "debug-gizmos")]
        if self.config.debug_gizmos {
            app.add_systems(Update, crate::debug::draw_debug_gizmos::<2>);
        }
    }
}

impl Plugin for SymtropyPhysicsPlugin<3> {
    fn build(&self, app: &mut App) {
        app.insert_resource(SymtropyPhysics::<3>::with_gravity(self.config.gravity));
        app.add_systems(
            FixedUpdate,
            (
                biometric_to_phi_system::<3>,
                apply_macro_modifiers_system::<3>,
                physics_step::<3>,
                sync_transforms::<3>,
            )
                .chain(),
        );
        #[cfg(feature = "debug-gizmos")]
        if self.config.debug_gizmos {
            app.add_systems(Update, crate::debug::draw_debug_gizmos::<3>);
        }
    }
}

impl Plugin for SymtropyPhysicsPlugin<4> {
    fn build(&self, app: &mut App) {
        app.insert_resource(SymtropyPhysics::<4>::with_gravity(self.config.gravity));
        app.add_systems(
            FixedUpdate,
            (
                biometric_to_phi_system::<4>,
                apply_macro_modifiers_system::<4>,
                physics_step::<4>,
                sync_transforms::<4>,
            )
                .chain(),
        );
        #[cfg(feature = "debug-gizmos")]
        if self.config.debug_gizmos {
            app.add_systems(Update, crate::debug::draw_debug_gizmos::<4>);
        }
    }
}

/// Physics step system: ticks prediction error decay, then steps the world
/// with the Phi-coupling field as the callback.
fn physics_step<const D: usize>(mut physics: ResMut<SymtropyPhysics<D>>, time: Res<Time<Fixed>>) {
    let dt = time.delta_secs_f64();
    let SymtropyPhysics {
        ref mut world,
        ref mut field,
    } = *physics;
    field.tick_prediction_errors();
    world.step_with_callback(dt, field);
}

/// Sync physics body positions to Bevy Transforms.
///
/// For 2D: writes (x, y) to translation.x/y.
/// For 3D: writes (x, y, z) to translation.
/// For 4D: writes (x, y, z) to translation (w dropped).
fn sync_transforms<const D: usize>(
    physics: Res<SymtropyPhysics<D>>,
    mut query: Query<(&PhysicsBody, &mut Transform)>,
) {
    for (body_comp, mut transform) in &mut query {
        if let Some(body) = physics.world.body(body_comp.handle) {
            let pos = body.position();
            if D >= 1 {
                transform.translation.x = pos.coord(0) as f32;
            }
            if D >= 2 {
                transform.translation.y = pos.coord(1) as f32;
            }
            if D >= 3 {
                transform.translation.z = pos.coord(2) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;
    use symthaea_consciousness_equation::ConsciousnessInputs;
    use symtropy_math::Point;

    // --- SymtropyPhysics resource construction ---

    #[test]
    fn symtropy_physics_2d_default_has_zero_gravity() {
        let p = SymtropyPhysics::<2>::default();
        let g = p.world.gravity;
        assert_eq!(g[0], 0.0);
        assert_eq!(g[1], 0.0);
    }

    #[test]
    fn symtropy_physics_3d_default_constructs() {
        let _p = SymtropyPhysics::<3>::default();
    }

    #[test]
    fn symtropy_physics_with_gravity_stores_gravity() {
        let p = SymtropyPhysics::<2>::with_gravity(SVector::from([0.0, -9.81]));
        let g = p.world.gravity;
        assert!((g[1] - (-9.81)).abs() < 1e-9);
    }

    // --- PhysicsBody ---

    #[test]
    fn physics_body_new_stores_handle_and_radius() {
        let mut world = symtropy_physics::PhysicsWorld::<2>::new(SVector::zeros());
        let pt = Point::new([0.0, 0.0]);
        let h = world.add_sphere(pt, 1.0, 1.0);
        let body = PhysicsBody::new(h, 0.5);
        assert_eq!(body.handle, h);
        assert!((body.visual_radius - 0.5).abs() < 1e-9);
    }

    // --- PluginConfig ---

    #[test]
    fn plugin_config_default_has_zero_gravity_2d() {
        let cfg = SymtropyPhysicsPluginConfig::<2>::default();
        assert_eq!(cfg.gravity, SVector::<f64, 2>::zeros());
    }

    #[test]
    fn plugin_config_default_has_zero_gravity_3d() {
        let cfg = SymtropyPhysicsPluginConfig::<3>::default();
        assert_eq!(cfg.gravity, SVector::<f64, 3>::zeros());
    }

    // --- Physics world operations ---

    #[test]
    fn can_add_sphere_to_physics_world() {
        let mut p = SymtropyPhysics::<2>::default();
        let pt = Point::new([0.0, 5.0]);
        let h = p.world.add_sphere(pt, 1.0, 1.0);
        assert!(p.world.body(h).is_some());
    }

    #[test]
    fn can_add_and_query_multiple_bodies() {
        let mut p = SymtropyPhysics::<2>::default();
        let h1 = p.world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
        let h2 = p.world.add_sphere(Point::new([5.0, 0.0]), 1.0, 1.0);
        assert!(p.world.body(h1).is_some());
        assert!(p.world.body(h2).is_some());
    }

    #[test]
    fn can_register_body_with_consciousness_field() {
        let mut p = SymtropyPhysics::<2>::default();
        let h = p.world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
        p.field.register(h, 100.0, 10.0);
        let state = p.field.entities.get(&h);
        assert!(state.is_some());
    }

    #[test]
    fn manual_physics_step_does_not_panic() {
        let mut p = SymtropyPhysics::<2>::default();
        p.world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
        let SymtropyPhysics {
            ref mut world,
            ref mut field,
        } = p;
        world.step_with_callback(1.0 / 60.0, field);
    }

    #[test]
    fn gravity_step_moves_body_downward() {
        let mut p = SymtropyPhysics::<2>::with_gravity(SVector::from([0.0, -9.81]));
        let h = p.world.add_sphere(Point::new([0.0, 10.0]), 1.0, 1.0);
        let y0 = p.world.body(h).unwrap().position().coord(1);
        let SymtropyPhysics {
            ref mut world,
            ref mut field,
        } = p;
        for _ in 0..10 {
            world.step_with_callback(1.0 / 60.0, field);
        }
        let y1 = p.world.body(h).unwrap().position().coord(1);
        assert!(
            y1 < y0,
            "gravity should pull body downward: y0={y0}, y1={y1}"
        );
    }

    #[test]
    fn field_phi_updates_after_consciousness_inputs() {
        let mut p = SymtropyPhysics::<2>::default();
        let h = p.world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
        p.field.register(h, 100.0, 10.0);
        let inputs = ConsciousnessInputs {
            phi: 0.75,
            broadcast: 0.75,
            working_memory: 0.75,
            attention: 0.75,
            recurrence: 0.75,
            embodiment: 0.75,
            knowledge: 0.75,
            synchrony: 0.75,
        };
        p.field.update_entity(h, &inputs, Point::origin());
        let state = p.field.entities.get(&h).unwrap();
        assert!(state.phi() > 0.0);
    }

    // --- Plugin construction ---

    #[test]
    fn plugin_with_gravity_constructor_does_not_panic() {
        let _plugin = SymtropyPhysicsPlugin::<2>::with_gravity([0.0, -9.81]);
    }

    // --- Headless Plugin registration ---

    #[test]
    fn plugin_2d_registers_resource() {
        let mut app = App::new();
        SymtropyPhysicsPlugin::<2>::default().build(&mut app);
        assert!(app.world().contains_resource::<SymtropyPhysics<2>>());
    }

    #[test]
    fn plugin_3d_registers_resource() {
        let mut app = App::new();
        SymtropyPhysicsPlugin::<3>::default().build(&mut app);
        assert!(app.world().contains_resource::<SymtropyPhysics<3>>());
    }

    #[test]
    fn plugin_4d_registers_resource() {
        let mut app = App::new();
        SymtropyPhysicsPlugin::<4>::default().build(&mut app);
        assert!(app.world().contains_resource::<SymtropyPhysics<4>>());
    }

    #[test]
    fn plugin_2d_does_not_interfere_with_plugin_3d() {
        let mut app = App::new();
        SymtropyPhysicsPlugin::<2>::default().build(&mut app);
        SymtropyPhysicsPlugin::<3>::default().build(&mut app);
        assert!(app.world().contains_resource::<SymtropyPhysics<2>>());
        assert!(app.world().contains_resource::<SymtropyPhysics<3>>());
    }

    #[test]
    fn plugin_resource_is_mutable_post_registration() {
        let mut app = App::new();
        SymtropyPhysicsPlugin::<2>::default().build(&mut app);
        let mut res = app.world_mut().resource_mut::<SymtropyPhysics<2>>();
        let pt = Point::new([1.0, 2.0]);
        let h = res.world.add_sphere(pt, 0.5, 1.0);
        assert!(res.world.body(h).is_some());
    }

    #[test]
    fn plugin_with_gravity_resource_accessible() {
        let mut app = App::new();
        SymtropyPhysicsPlugin::<2>::with_gravity([0.0, -9.81]).build(&mut app);
        let res = app.world().resource::<SymtropyPhysics<2>>();
        let g = res.world.gravity;
        assert!((g[1] - (-9.81)).abs() < 1e-9);
    }
}
