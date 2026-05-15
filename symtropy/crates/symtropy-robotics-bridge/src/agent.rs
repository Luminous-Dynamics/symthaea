// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use bevy::ecs::relationship::Relationship;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use symthaea_bevy_brain::CognitiveBrain;
use symtropy_bevy_core::{BevyPhysics, PhysicsBody};

/// Marker for the root entity of a robotic agent.
#[derive(Component, Serialize, Deserialize, Clone, Debug)]
pub struct RoboticAgent {
    pub model_name: String,
}

/// Marker for a motorized joint in a robotic kinematic chain.
#[derive(Component, Serialize, Deserialize, Clone, Debug)]
pub struct RoboticJoint {
    pub joint_name: String,
    pub motor_index: usize,
}

pub struct RoboticBrainPlugin;

impl Plugin for RoboticBrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            (propagate_sensory_input_system, apply_motor_commands_system),
        );
    }
}

/// Feeds physical state (pose, velocity) from PhysicsBody into the CognitiveBrain's LTC input.
fn propagate_sensory_input_system(
    mut query: Query<(&mut CognitiveBrain, &Transform, &PhysicsBody)>,
) {
    for (mut brain, transform, _body) in &mut query {
        // String-based perception input for the brain
        brain.perception_input = format!(
            "pos: {:.2}, {:.2}, {:.2}",
            transform.translation.x, transform.translation.y, transform.translation.z
        );
    }
}

/// Translates LTC output neurons into PD motor drive targets for kinematic joints.
fn apply_motor_commands_system(
    brains: Query<(&CognitiveBrain, &RoboticAgent)>,
    mut joints: Query<(&RoboticJoint, &mut PhysicsBody, &ChildOf)>,
    mut physics: ResMut<BevyPhysics<3>>,
) {
    for (joint, body, parent) in &mut joints {
        if let Ok((brain, _agent)) = brains.get(parent.get()) {
            let output = &brain.motor_output;
            if output.len() > joint.motor_index {
                let _target_pos = output[joint.motor_index] as f64;
                if let Some(_rb) = physics.world.body_mut(body.handle) {
                    // Assuming a MotorDrive is attached to the body in the physics world
                    // rb.apply_torque(...) or set PD target
                }
            }
        }
    }
}

pub fn spawn_robot(
    commands: &mut Commands,
    name: &str,
    pos: Vec3,
    brain: CognitiveBrain,
    body_handle: symtropy_physics::body::BodyHandle,
) -> Entity {
    commands
        .spawn((
            RoboticAgent {
                model_name: name.to_string(),
            },
            brain,
            PhysicsBody {
                handle: body_handle,
                visual_radius: 0.5,
            },
            Transform::from_translation(pos),
            GlobalTransform::default(),
        ))
        .id()
}
