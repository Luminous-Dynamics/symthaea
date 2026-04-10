// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The core Bevy plugin and systems.

use bevy::prelude::*;
use nalgebra::SVector;
use symtropy_consciousness_physics::coupling::ConsciousnessField;
use symtropy_physics::body::BodyHandle;
use symtropy_physics::PhysicsWorld;

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

/// Bevy component linking an entity to a physics body.
///
/// Attach this to a Bevy entity (with `Sprite`, `Mesh3d`, etc.)
/// to have its `Transform` automatically synced from the physics world.
#[derive(Component)]
pub struct PhysicsBody {
    /// Handle to the body in the physics world.
    pub handle: BodyHandle,
    /// Visual radius for debug rendering.
    pub visual_radius: f32,
}

impl PhysicsBody {
    pub fn new(handle: BodyHandle, visual_radius: f32) -> Self {
        Self { handle, visual_radius }
    }
}

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
        app.add_systems(FixedUpdate, (physics_step::<2>, sync_transforms::<2>).chain());
        #[cfg(feature = "debug-gizmos")]
        if self.config.debug_gizmos {
            app.add_systems(Update, crate::debug::draw_debug_gizmos::<2>);
        }
    }
}

impl Plugin for SymtropyPhysicsPlugin<3> {
    fn build(&self, app: &mut App) {
        app.insert_resource(SymtropyPhysics::<3>::with_gravity(self.config.gravity));
        app.add_systems(FixedUpdate, (physics_step::<3>, sync_transforms::<3>).chain());
        #[cfg(feature = "debug-gizmos")]
        if self.config.debug_gizmos {
            app.add_systems(Update, crate::debug::draw_debug_gizmos::<3>);
        }
    }
}

impl Plugin for SymtropyPhysicsPlugin<4> {
    fn build(&self, app: &mut App) {
        app.insert_resource(SymtropyPhysics::<4>::with_gravity(self.config.gravity));
        app.add_systems(FixedUpdate, (physics_step::<4>, sync_transforms::<4>).chain());
        #[cfg(feature = "debug-gizmos")]
        if self.config.debug_gizmos {
            app.add_systems(Update, crate::debug::draw_debug_gizmos::<4>);
        }
    }
}

/// Physics step system: ticks prediction error decay, then steps the world
/// with the Phi-coupling field as the callback.
fn physics_step<const D: usize>(
    mut physics: ResMut<SymtropyPhysics<D>>,
    time: Res<Time<Fixed>>,
) {
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
            if D >= 1 { transform.translation.x = pos.coord(0) as f32; }
            if D >= 2 { transform.translation.y = pos.coord(1) as f32; }
            if D >= 3 { transform.translation.z = pos.coord(2) as f32; }
        }
    }
}
