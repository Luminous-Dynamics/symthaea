// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified robotics spawning for Bevy.

use crate::plugin::SymtropyPhysics;
use bevy::prelude::*;
use symtropy_math::Point;
use symtropy_robotics_bridge_core::agent::spawn_robot_body;
use symtropy_robotics_bridge_core::platform::PlatformType;

/// Spawn a robot using the native Symtropy N-Dimensional solver.
pub fn spawn_robot_native<const D: usize>(
    commands: &mut Commands,
    physics: &mut SymtropyPhysics<D>,
    platform: PlatformType,
    position: Point<D>,
    name: impl Into<String>,
) -> Entity {
    let handle = spawn_robot_body(&mut physics.world, platform, position);

    // In Bevy, we spawn an entity and attach the PhysicsBody component
    commands
        .spawn((
            crate::PhysicsBody::new(handle, platform.default_radius() as f32),
            Name::new(name.into()),
            // Transform is synced by SymtropyPhysicsPlugin
        ))
        .id()
}

/// Spawn a robot using the Rapier3D backend (if enabled).
#[cfg(feature = "rapier3d")]
pub fn spawn_robot_rapier3d(
    commands: &mut Commands,
    platform: PlatformType,
    position: Vec3,
    name: impl Into<String>,
) -> Entity {
    use rapier3d::prelude::*;
    use symtropy_rapier3d_bridge::{RapierCollider, RapierRigidBody};

    // Note: This assumes a Rapier3D world is active in the Bevy app
    // (e.g. via bevy_rapier3d or our own bridge integration).
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![position.x, position.y, position.z])
        .build();
    let collider = ColliderBuilder::ball(platform.default_radius() as f32).build();

    commands
        .spawn((
            RapierRigidBody(rigid_body),
            RapierCollider(collider),
            Name::new(name.into()),
        ))
        .id()
}
