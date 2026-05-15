// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bridge between Rapier3D and Symtropy's state-coupling framework.
//!
//! This crate allows using Rapier3D as the physics backend while preserving
//! Symtropy's first-class state-coupling (Phi, harmony, energy budgets).

use ::nalgebra::SVector;
use rapier3d::prelude::*;
use symtropy_physics::body::BodyHandle;
use symtropy_physics::world::PhysicsCallback;

use bevy::prelude::Component;

/// Wrapper to make Rapier RigidBody a Bevy Component.
#[derive(Component)]
pub struct RapierRigidBody(pub RigidBody);

/// Wrapper to make Rapier Collider a Bevy Component.
#[derive(Component)]
pub struct RapierCollider(pub Collider);

/// A wrapper that applies a Symtropy `PhysicsCallback` to a Rapier3D world.
pub struct RapierPhysicsBridge<C: PhysicsCallback<3>> {
    pub callback: C,
}

impl<C: PhysicsCallback<3>> RapierPhysicsBridge<C> {
    pub fn new(callback: C) -> Self {
        Self { callback }
    }

    /// Step the Rapier world and apply modulated forces/impulses.
    pub fn step(
        &mut self,
        _dt: f32,
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        integration_parameters: &IntegrationParameters,
        island_manager: &mut IslandManager,
        broad_phase: &mut DefaultBroadPhase,
        narrow_phase: &mut NarrowPhase,
        impulse_joint_set: &mut ImpulseJointSet,
        multibody_joint_set: &mut MultibodyJointSet,
        ccd_solver: &mut CCDSolver,
        physics_hooks: &dyn PhysicsHooks,
        event_handler: &dyn EventHandler,
    ) {
        // 1. Modulate external forces before the step
        for (handle, body) in rigid_body_set.iter_mut() {
            let body_handle = BodyHandle(handle.into_raw_parts().0 as usize);
            let force = body.user_force();
            // Manually convert from Rapier's nalgebra version (0.33) to Symtropy's (0.34)
            let force_f64 = SVector::<f64, 3>::new(force.x as f64, force.y as f64, force.z as f64);
            let modulated_force = self.callback.modulate_force(body_handle, &force_f64);

            // Convert back to Rapier's nalgebra version
            let rapier_force = rapier3d::na::Vector3::new(
                modulated_force[0] as f32,
                modulated_force[1] as f32,
                modulated_force[2] as f32,
            );

            body.reset_forces(true);
            body.add_force(rapier_force, true);
        }

        // 2. Perform the physics step
        // Note: Real-time modulation of impulses within the solver requires
        // custom PhysicsHooks. For this baseline, we step then apply correction.
        let mut pipeline = PhysicsPipeline::new();
        pipeline.step(
            &rapier3d::na::vector![0.0, -9.81, 0.0],
            integration_parameters,
            island_manager,
            broad_phase,
            narrow_phase,
            rigid_body_set,
            collider_set,
            impulse_joint_set,
            multibody_joint_set,
            ccd_solver,
            None,
            physics_hooks,
            event_handler,
        );

        // 3. Post-step: energy accounting and trauma application
        // (This would iterate over contact events and call self.callback.on_collision)
    }
}

/// Helper to spawn a robot using Rapier3D.
pub fn spawn_robot_rapier3d(
    rigid_body_set: &mut RigidBodySet,
    collider_set: &mut ColliderSet,
    platform: symtropy_robotics_bridge_core::platform::PlatformType,
    position: symtropy_math::Point<3>,
) -> RigidBodyHandle {
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![
            position.0[0] as f32,
            position.0[1] as f32,
            position.0[2] as f32
        ])
        .build();
    let handle = rigid_body_set.insert(rigid_body);
    let collider = ColliderBuilder::ball(platform.default_radius() as f32).build();
    collider_set.insert_with_parent(collider, handle, rigid_body_set);
    handle
}
