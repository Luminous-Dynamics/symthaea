// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use bevy::ecs::relationship::Relationship;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use symthaea_bevy_brain::CognitiveBrain;
use symthaea_fep::HapticSemanticBinder;
use symtropy_bevy_core::{BevyPhysics, PhysicsBody};

/// Marker for the root entity of a robotic agent.
#[derive(Component, Serialize, Deserialize, Clone, Debug)]
pub struct RoboticAgent {
    pub model_name: String,
}

/// Component that maps physical state to semantic HDC space.
#[derive(Component)]
pub struct RoboticHapticBinder {
    pub binder: HapticSemanticBinder,
}

/// Proximity sensor: detects nearby objects within a radius.
#[derive(Component, Serialize, Deserialize, Clone, Debug)]
pub struct RoboticProximitySensor {
    pub radius: f32,
    /// Detected entities and their distances.
    #[serde(skip)]
    pub detections: Vec<(Entity, f32)>,
}

impl Default for RoboticProximitySensor {
    fn default() -> Self {
        Self {
            radius: 5.0,
            detections: Vec::new(),
        }
    }
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
            (
                update_proximity_sensors_system,
                propagate_sensory_input_system,
                apply_motor_commands_system,
            )
                .chain(),
        );
    }
}

/// Updates proximity sensors by querying the physics world or Bevy transforms.
fn update_proximity_sensors_system(
    mut sensors: Query<(&mut RoboticProximitySensor, &Transform, Entity)>,
    others: Query<(&Transform, Entity), (Without<RoboticProximitySensor>, With<PhysicsBody>)>,
) {
    for (mut sensor, transform, sensor_entity) in &mut sensors {
        sensor.detections.clear();
        for (other_transform, other_entity) in &others {
            if sensor_entity == other_entity {
                continue;
            }
            let dist = transform.translation.distance(other_transform.translation);
            if dist <= sensor.radius {
                sensor.detections.push((other_entity, dist));
            }
        }
        // Sort by distance
        sensor
            .detections
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    }
}

/// Feeds physical state (pose, velocity, proximity) into the CognitiveBrain's LTC input.
/// Transitions from string-based to HDC-based perception using HapticSemanticBinder.
fn propagate_sensory_input_system(
    mut query: Query<(
        &mut CognitiveBrain,
        &Transform,
        &PhysicsBody,
        Option<&RoboticHapticBinder>,
        Option<&RoboticProximitySensor>,
    )>,
    physics: Res<BevyPhysics<3>>,
) {
    for (mut brain, transform, body_comp, haptic_binder, proximity) in &mut query {
        if let Some(hb) = haptic_binder {
            // Base state: [pos(3), rot(4), vel(3), ang_vel(3)] (13 dims)
            let mut state = vec![0.0f32; 13];
            state[0] = transform.translation.x;
            state[1] = transform.translation.y;
            state[2] = transform.translation.z;
            state[3] = transform.rotation.x;
            state[4] = transform.rotation.y;
            state[5] = transform.rotation.z;
            state[6] = transform.rotation.w;

            if let Some(body) = physics.world.body(body_comp.handle) {
                state[7] = body.linear_velocity[0] as f32;
                state[8] = body.linear_velocity[1] as f32;
                state[9] = body.linear_velocity[2] as f32;
                // Bivector 3D has 3 components: xy, xz, yz
                state[10] = body.angular_velocity.get(0, 1) as f32;
                state[11] = body.angular_velocity.get(0, 2) as f32;
                state[12] = body.angular_velocity.get(1, 2) as f32;
            }

            // Optional: Append proximity data (first 3 closest objects: [dist, dx, dy, dz] * 3 = 12 dims)
            // Total dims would be 13 + 12 = 25.
            // For now, keeping it at 13 to match the binder initialization in spawn_robot.
            // Future: Dynamically resize binder or use a fixed large capacity.

            // Project into 16,384D semantic space
            let health = vec![1.0f32; state.len()];
            brain.perception_hv = Some(hb.binder.bind_with_health(&state, &health));
        } else {
            // Legacy path
            let prox_str = if let Some(p) = proximity {
                format!(", proximity: {}", p.detections.len())
            } else {
                String::new()
            };
            brain.perception_input = format!(
                "pos: {:.2}, {:.2}, {:.2}{}",
                transform.translation.x, transform.translation.y, transform.translation.z, prox_str
            );
        }
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
    // Create a binder for the standard 13-dim state
    let haptic_binder = RoboticHapticBinder {
        binder: HapticSemanticBinder::new(13, 16384),
    };

    commands
        .spawn((
            RoboticAgent {
                model_name: name.to_string(),
            },
            brain,
            haptic_binder,
            RoboticProximitySensor::default(),
            PhysicsBody {
                handle: body_handle,
                visual_radius: 0.5,
            },
            Transform::from_translation(pos),
            GlobalTransform::default(),
        ))
        .id()
}
