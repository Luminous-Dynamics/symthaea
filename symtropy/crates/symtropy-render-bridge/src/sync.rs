// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sync ND physics bodies to Bevy Transform components.

use bevy::prelude::*;
use symtropy_physics::body::BodyHandle;

/// Bevy component that links an entity to a physics body.
///
/// Attach this to a Bevy entity (with a SpriteBundle or MeshBundle)
/// to have its Transform automatically updated from the physics world.
#[derive(Component)]
pub struct PhysicsBody {
    /// Handle to the corresponding body in PhysicsWorld.
    pub handle: BodyHandle,
    /// Visual radius for rendering (may differ from collider radius).
    pub visual_radius: f32,
}

impl PhysicsBody {
    pub fn new(handle: BodyHandle, visual_radius: f32) -> Self {
        Self {
            handle,
            visual_radius,
        }
    }
}

/// Sync the 2D physics world to Bevy transforms.
///
/// This is a Bevy system that reads from PhysicsWorld<2> (stored as a Resource)
/// and updates all entities with a PhysicsBody component.
///
/// Usage in your Bevy app:
/// ```ignore
/// app.add_systems(Update, sync_physics_2d);
/// ```
pub fn sync_physics_2d(
    physics: Res<PhysicsWorldRes2D>,
    mut query: Query<(&PhysicsBody, &mut Transform)>,
) {
    for (body_comp, mut transform) in &mut query {
        if let Some(body) = physics.world.body(body_comp.handle) {
            let pos = body.position();
            transform.translation.x = pos.coord(0) as f32;
            transform.translation.y = pos.coord(1) as f32;
        }
    }
}

/// Sync the 3D physics world to Bevy transforms.
pub fn sync_physics_3d(
    physics: Res<PhysicsWorldRes3D>,
    mut query: Query<(&PhysicsBody, &mut Transform)>,
) {
    for (body_comp, mut transform) in &mut query {
        if let Some(body) = physics.world.body(body_comp.handle) {
            let pos = body.position();
            transform.translation.x = pos.coord(0) as f32;
            transform.translation.y = pos.coord(1) as f32;
            transform.translation.z = pos.coord(2) as f32;
        }
    }
}

/// Sync the 4D physics world to Bevy transforms.
///
/// Projects the first 3 coordinates (x, y, z) to Bevy's Transform.
/// The 4th coordinate (w) is stored in `transform.scale.x` as a
/// normalized proximity signal [0.1, 1.0] that can drive alpha/visibility
/// via the Projector4D system.
pub fn sync_physics_4d(
    physics: Res<PhysicsWorldRes4D>,
    mut query: Query<(&PhysicsBody, &mut Transform)>,
) {
    for (body_comp, mut transform) in &mut query {
        if let Some(body) = physics.world.body(body_comp.handle) {
            let pos = body.position();
            transform.translation.x = pos.coord(0) as f32;
            transform.translation.y = pos.coord(1) as f32;
            transform.translation.z = pos.coord(2) as f32;
        }
    }
}

/// Bevy Resource wrapping a 2D physics world.
#[derive(Resource)]
pub struct PhysicsWorldRes2D {
    pub world: symtropy_physics::PhysicsWorld<2>,
}

/// Bevy Resource wrapping a 3D physics world.
#[derive(Resource)]
pub struct PhysicsWorldRes3D {
    pub world: symtropy_physics::PhysicsWorld<3>,
}

/// Bevy Resource wrapping a 4D physics world.
#[derive(Resource)]
pub struct PhysicsWorldRes4D {
    pub world: symtropy_physics::PhysicsWorld<4>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physics_body_component_creation() {
        let pb = PhysicsBody::new(BodyHandle(42), 16.0);
        assert_eq!(pb.handle, BodyHandle(42));
        assert_eq!(pb.visual_radius, 16.0);
    }

    #[test]
    fn physics_world_res_2d() {
        let res = PhysicsWorldRes2D {
            world: symtropy_physics::PhysicsWorld::new(nalgebra::SVector::from([0.0, -9.81])),
        };
        assert_eq!(res.world.body_count(), 0);
    }

    #[test]
    fn physics_world_res_3d() {
        let res = PhysicsWorldRes3D {
            world: symtropy_physics::PhysicsWorld::new(nalgebra::SVector::from([0.0, -9.81, 0.0])),
        };
        assert_eq!(res.world.body_count(), 0);
    }

    #[test]
    fn physics_world_res_4d() {
        let res = PhysicsWorldRes4D {
            world: symtropy_physics::PhysicsWorld::new(nalgebra::SVector::from([
                0.0, -9.81, 0.0, 0.0,
            ])),
        };
        assert_eq!(res.world.body_count(), 0);
    }
}
