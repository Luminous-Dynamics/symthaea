// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! # symtropy-bevy-core
//!
//! Permissively-licensed Bevy plugin for [`symtropy-physics`] N-dimensional physics.
//! Zero AGPL dependencies.
//!
//! Use [`BevyPhysicsPlugin`] for drop-in physics with no coupling, or
//! [`BevyPhysicsCorePlugin`] + [`step_physics`] with your own
//! [`PhysicsCallback`] resource to couple any per-body metric to physics.
//!
//! For Φ (integrated information) coupling, see the AGPL sibling crate
//! `symtropy-bevy` — same API, adds `ConsciousnessField` integration.

use bevy::prelude::*;
use nalgebra::SVector;
use symtropy_physics::{body::BodyHandle, CollisionEvent, PhysicsCallback, PhysicsWorld};

// --- Resource wrapping the physics world ---

/// Bevy resource wrapping an N-dimensional [`PhysicsWorld`].
///
/// Access this in your systems to add bodies, step simulation manually,
/// or query state.
#[derive(Resource)]
pub struct BevyPhysics<const D: usize> {
    /// The N-dimensional rigid body world.
    pub world: PhysicsWorld<D>,
}

impl<const D: usize> Default for BevyPhysics<D> {
    fn default() -> Self {
        Self {
            world: PhysicsWorld::new(SVector::zeros()),
        }
    }
}

impl<const D: usize> BevyPhysics<D> {
    /// Create with custom gravity.
    pub fn with_gravity(gravity: [f64; D]) -> Self {
        Self {
            world: PhysicsWorld::new(SVector::from(gravity)),
        }
    }
}

// --- Linking component ---

/// Bevy component linking an entity to a physics body.
///
/// Attach to a Bevy entity with a `Transform`; `sync_transforms` writes the
/// physics body's position into it each `FixedUpdate`.
#[derive(Component)]
pub struct PhysicsBody {
    /// Handle to the body in the physics world.
    pub handle: BodyHandle,
    /// Visual radius for debug rendering.
    pub visual_radius: f32,
}

impl PhysicsBody {
    /// Create a new component for a given body handle and visual radius.
    pub fn new(handle: BodyHandle, visual_radius: f32) -> Self {
        Self {
            handle,
            visual_radius,
        }
    }
}

// --- Default no-coupling callback (resource flavor) ---

/// Bevy `Resource` form of the identity "no-coupling" callback.
///
/// Forces, impulses, and friction pass through unchanged. Use when you want
/// N-dimensional physics without any per-body state coupling. Implements
/// [`PhysicsCallback<D>`] for all `D`.
#[derive(Resource, Default)]
pub struct NoCouplingResource;

impl<const D: usize> PhysicsCallback<D> for NoCouplingResource {
    fn modulate_force(&self, _: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
        *force
    }
    fn modulate_impulse(&self, impulse: f64, _: &SVector<f64, D>) -> f64 {
        impulse
    }
    fn friction_multiplier(&self, _: &SVector<f64, D>, _: BodyHandle) -> f64 {
        1.0
    }
    fn on_collision(&mut self, _: &CollisionEvent<D>) {}
    fn record_dissipation(&mut self, _: f64) {}
    fn record_work(&mut self, _: BodyHandle, _: f64) {}
    fn apply_trauma(&mut self, _: &CollisionEvent<D>) {}
}

// --- Systems ---

/// Step the physics world with a generic callback `Resource`.
///
/// Bring your own `C: PhysicsCallback<D> + Resource` to couple a custom metric
/// to physics, or use [`NoCouplingResource`] for uncoupled physics-only behavior.
pub fn step_physics<const D: usize, C: PhysicsCallback<D> + Resource>(
    mut physics: ResMut<BevyPhysics<D>>,
    mut cb: ResMut<C>,
    time: Res<Time<Fixed>>,
) {
    physics
        .world
        .step_with_callback(time.delta_secs_f64(), &mut *cb);
}

/// Sync physics body positions to Bevy `Transform`s.
///
/// 2D: writes `(x, y)` to `translation.x`/`.y`.
/// 3D: writes `(x, y, z)` to `translation`.
/// 4D: writes `(x, y, z)` to `translation` (w dropped — use `symtropy-render-bridge`
/// for cross-section projection).
pub fn sync_transforms<const D: usize>(
    physics: Res<BevyPhysics<D>>,
    mut query: Query<(&PhysicsBody, &mut Transform)>,
) {
    for (body_comp, mut transform) in &mut query {
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

// --- Plugins ---

/// Minimal Bevy plugin: registers [`BevyPhysics<D>`] + [`sync_transforms`] only.
///
/// Use this when you're supplying your own step system with a custom
/// [`PhysicsCallback`] resource.
pub struct BevyPhysicsCorePlugin<const D: usize> {
    /// Initial gravity.
    pub gravity: SVector<f64, D>,
}

impl<const D: usize> Default for BevyPhysicsCorePlugin<D> {
    fn default() -> Self {
        Self {
            gravity: SVector::zeros(),
        }
    }
}

impl<const D: usize> BevyPhysicsCorePlugin<D> {
    /// Create with the given gravity vector.
    pub fn with_gravity(gravity: [f64; D]) -> Self {
        Self {
            gravity: SVector::from(gravity),
        }
    }
}

/// Full Bevy plugin: registers [`BevyPhysics<D>`] + [`NoCouplingResource`] + a
/// default [`step_physics`] system + [`sync_transforms`].
///
/// Drop in for N-dimensional physics with no coupling. For custom couplings,
/// use [`BevyPhysicsCorePlugin`] and register your own step system.
pub struct BevyPhysicsPlugin<const D: usize> {
    /// Initial gravity.
    pub gravity: SVector<f64, D>,
}

impl<const D: usize> Default for BevyPhysicsPlugin<D> {
    fn default() -> Self {
        Self {
            gravity: SVector::zeros(),
        }
    }
}

impl<const D: usize> BevyPhysicsPlugin<D> {
    /// Create with the given gravity vector.
    pub fn with_gravity(gravity: [f64; D]) -> Self {
        Self {
            gravity: SVector::from(gravity),
        }
    }
}

// Per-dim Plugin impls — Bevy's Plugin trait isn't const-generic.

macro_rules! impl_core_plugin {
    ($d:literal) => {
        impl Plugin for BevyPhysicsCorePlugin<$d> {
            fn build(&self, app: &mut App) {
                app.insert_resource(BevyPhysics::<$d> {
                    world: PhysicsWorld::new(self.gravity),
                });
                app.add_systems(FixedUpdate, sync_transforms::<$d>);
            }
        }
    };
}

macro_rules! impl_full_plugin {
    ($d:literal) => {
        impl Plugin for BevyPhysicsPlugin<$d> {
            fn build(&self, app: &mut App) {
                app.insert_resource(BevyPhysics::<$d> {
                    world: PhysicsWorld::new(self.gravity),
                });
                app.insert_resource(NoCouplingResource);
                app.add_systems(
                    FixedUpdate,
                    (
                        step_physics::<$d, NoCouplingResource>,
                        sync_transforms::<$d>,
                    )
                        .chain(),
                );
            }
        }
    };
}

impl_core_plugin!(2);
impl_core_plugin!(3);
impl_core_plugin!(4);

impl_full_plugin!(2);
impl_full_plugin!(3);
impl_full_plugin!(4);

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use symtropy_math::Point;

    #[test]
    fn bevy_physics_default_has_zero_gravity_2d() {
        let p = BevyPhysics::<2>::default();
        let g = p.world.gravity;
        assert_eq!(g[0], 0.0);
        assert_eq!(g[1], 0.0);
    }

    #[test]
    fn bevy_physics_with_gravity_stores_gravity() {
        let p = BevyPhysics::<3>::with_gravity([0.0, -9.81, 0.0]);
        let g = p.world.gravity;
        assert!((g[1] - (-9.81)).abs() < 1e-9);
    }

    #[test]
    fn physics_body_new_stores_handle_and_radius() {
        let mut world = PhysicsWorld::<2>::new(SVector::zeros());
        let h = world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
        let b = PhysicsBody::new(h, 0.5);
        assert_eq!(b.handle, h);
        assert!((b.visual_radius - 0.5).abs() < 1e-9);
    }

    #[test]
    fn no_coupling_modulate_force_is_identity() {
        let cb = NoCouplingResource;
        let mut world = PhysicsWorld::<2>::new(SVector::zeros());
        let h = world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
        let f_in = SVector::from([3.14, -2.71]);
        let f_out = <NoCouplingResource as PhysicsCallback<2>>::modulate_force(&cb, h, &f_in);
        assert_eq!(f_in, f_out);
    }

    #[test]
    fn no_coupling_friction_multiplier_is_one() {
        let cb = NoCouplingResource;
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        let h = world.add_sphere(Point::new([0.0, 0.0, 0.0]), 1.0, 1.0);
        let point = SVector::from([0.0, 0.0, 0.0]);
        let mu = <NoCouplingResource as PhysicsCallback<3>>::friction_multiplier(&cb, &point, h);
        assert!((mu - 1.0).abs() < 1e-9);
    }

    #[test]
    fn core_plugin_2d_registers_resource() {
        let mut app = App::new();
        BevyPhysicsCorePlugin::<2>::default().build(&mut app);
        assert!(app.world().contains_resource::<BevyPhysics<2>>());
    }

    #[test]
    fn full_plugin_2d_registers_both_resources() {
        let mut app = App::new();
        BevyPhysicsPlugin::<2>::default().build(&mut app);
        assert!(app.world().contains_resource::<BevyPhysics<2>>());
        assert!(app.world().contains_resource::<NoCouplingResource>());
    }

    #[test]
    fn full_plugin_3d_registers_both_resources() {
        let mut app = App::new();
        BevyPhysicsPlugin::<3>::default().build(&mut app);
        assert!(app.world().contains_resource::<BevyPhysics<3>>());
        assert!(app.world().contains_resource::<NoCouplingResource>());
    }

    #[test]
    fn full_plugin_4d_registers_both_resources() {
        let mut app = App::new();
        BevyPhysicsPlugin::<4>::default().build(&mut app);
        assert!(app.world().contains_resource::<BevyPhysics<4>>());
        assert!(app.world().contains_resource::<NoCouplingResource>());
    }

    #[test]
    fn plugin_with_gravity_resource_accessible() {
        let mut app = App::new();
        BevyPhysicsPlugin::<2>::with_gravity([0.0, -9.81]).build(&mut app);
        let res = app.world().resource::<BevyPhysics<2>>();
        let g = res.world.gravity;
        assert!((g[1] - (-9.81)).abs() < 1e-9);
    }

    #[test]
    fn manual_step_with_no_coupling_gravity_pulls_body_down() {
        let mut p = BevyPhysics::<2>::with_gravity([0.0, -9.81]);
        let h = p.world.add_sphere(Point::new([0.0, 10.0]), 1.0, 1.0);
        let y0 = p.world.body(h).unwrap().position().coord(1);
        let mut cb = NoCouplingResource;
        for _ in 0..10 {
            p.world.step_with_callback(1.0 / 60.0, &mut cb);
        }
        let y1 = p.world.body(h).unwrap().position().coord(1);
        assert!(y1 < y0, "gravity should pull body down: y0={y0}, y1={y1}");
    }

    #[test]
    fn can_add_multiple_bodies() {
        let mut p = BevyPhysics::<3>::default();
        let h1 = p.world.add_sphere(Point::new([0.0, 0.0, 0.0]), 1.0, 1.0);
        let h2 = p.world.add_sphere(Point::new([5.0, 0.0, 0.0]), 1.0, 1.0);
        assert!(p.world.body(h1).is_some());
        assert!(p.world.body(h2).is_some());
    }
}
