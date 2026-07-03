// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use bevy::ecs::relationship::Relationship;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use symthaea_bevy_brain::CognitiveBrain;
use symthaea_fep::HapticSemanticBinder;
use symtropy_bevy_core::{BevyPhysics, PhysicsBody};
use symtropy_math::Bivector;

/// Marker for the root entity of a robotic agent (ECS tag component only —
/// for the consciousness-driving agent with `.tick()`/`.phi()`, see
/// [`RoboticAgent`] further down this file).
#[derive(Component, Serialize, Deserialize, Clone, Debug)]
pub struct RoboticAgentTag {
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

/// Direct-drive proportional torque gain applied per unit of normalized
/// motor output ([-1.0, 1.0] LTC output -> Newton-meters). A full actuator
/// model (gearing, current limits, per-joint PD gains) is out of scope for
/// this generic ECS bridge — platform demos that need real PD position
/// control implement their own controller (e.g.
/// `symtropy-manipulator-demo`'s `admittance_control.rs`) and don't route
/// through this Bevy system at all.
const MOTOR_TORQUE_GAIN: f64 = 5.0;

/// Translates LTC output neurons into torque applied to kinematic joints.
///
/// `RoboticJoint` doesn't currently carry per-joint rotation-axis metadata,
/// so every joint is driven as a single-DOF hinge rotating in the physics
/// world's xy plane (bivector components `(0, 1)`). Multi-axis joints would
/// need an explicit axis/plane field added to `RoboticJoint`.
fn apply_motor_commands_system(
    brains: Query<(&CognitiveBrain, &RoboticAgentTag)>,
    mut joints: Query<(&RoboticJoint, &mut PhysicsBody, &ChildOf)>,
    mut physics: ResMut<BevyPhysics<3>>,
) {
    for (joint, body, parent) in &mut joints {
        if let Ok((brain, _agent)) = brains.get(parent.get()) {
            let output = &brain.motor_output;
            if output.len() > joint.motor_index {
                let target = output[joint.motor_index] as f64;
                if let Some(rb) = physics.world.body_mut(body.handle) {
                    let mut torque = Bivector::zero();
                    torque.set(0, 1, target * MOTOR_TORQUE_GAIN);
                    rb.apply_torque(torque);
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
            RoboticAgentTag {
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

// ---------------------------------------------------------------------
// Headless consciousness-driven RoboticAgent
// ---------------------------------------------------------------------
//
// This is a *different* concept from `RoboticAgentTag` above: it's not a
// Bevy component at all, but the concrete type the platform demos
// (`crates/apps/symtropy-*-demo`) and this crate's `examples/*.rs` Φ
// benchmarks construct directly via `RoboticAgent::new(handle, platform,
// name)` and drive headlessly via `.tick(observation, danger_level)` /
// `.phi()`. It implements the trait of the same name from
// `symtropy-robotics-bridge-core` (re-exported from this crate's root as
// `RoboticAgentTrait` to avoid colliding with this concrete struct) so
// `tick_motor_commands()` / `MotorPlanner` dispatch works generically over
// it. Callers must `use symtropy_robotics_bridge::RoboticAgentTrait;` to
// bring `.tick()` / `.phi()` / `.bottleneck()` / `.can_act()` into scope —
// `RoboticAgent::new()` and the public `caution` / `safety_tier` fields
// don't need it.

use symthaea_consciousness_equation::ConsciousnessInputs;
use symtropy_consciousness_physics::{EntityConsciousness, SafetyTier};
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge_core::RoboticAgent as CoreRoboticAgent;
use symtropy_robotics_bridge_core::platform::PlatformType;

/// EMA smoothing factor for the `caution` accumulator — how quickly caution
/// tracks the current danger level (~6-7 ticks to reach 63% of a step
/// change).
const CAUTION_EMA_ALPHA: f64 = 0.15;

/// Default energy budget for the internal [`EntityConsciousness`]. Large
/// enough that these headless demo/benchmark agents never hit the
/// low-energy `SafetyTier::Red` collapse path from budget exhaustion —
/// only from Φ itself, which is what the demos/benchmarks are measuring.
const DEFAULT_ENERGY_BUDGET: f64 = 1_000.0;

/// A consciousness-driven robotic agent: wraps the real
/// `MasterConsciousnessEquation` pipeline (via
/// `symtropy_consciousness_physics::EntityConsciousness`) behind the simple
/// `tick(observation, danger) -> motor_gain` contract the platform demos and
/// Φ-gated-safety benchmark examples expect.
///
/// Construct with [`RoboticAgent::new`]; drive with `.tick()` / read `.phi()`
/// via the [`symtropy_robotics_bridge_core::RoboticAgent`] trait (import it
/// from this crate's root as `RoboticAgentTrait`).
pub struct RoboticAgent {
    body: BodyHandle,
    platform: PlatformType,
    pub name: String,
    consciousness: EntityConsciousness,
    /// EMA of recent danger levels. Rises under sustained danger, decays
    /// back down once danger subsides — a softer behavioral signal
    /// independent of the harder Φ-gated `safety_tier`.
    pub caution: f64,
    /// Current Φ-gated safety tier (mirrors `self.consciousness.safety_tier`
    /// after each tick; exposed directly so callers don't need
    /// `symtropy-consciousness-physics` in scope just to read it).
    pub safety_tier: SafetyTier,
}

impl RoboticAgent {
    /// Create a new agent for `platform`, tracking physics body `body`, with
    /// a display `name` (logging/telemetry only, no uniqueness requirement).
    pub fn new(body: BodyHandle, platform: PlatformType, name: impl Into<String>) -> Self {
        Self {
            body,
            platform,
            name: name.into(),
            consciousness: EntityConsciousness::new(DEFAULT_ENERGY_BUDGET),
            caution: 0.0,
            safety_tier: SafetyTier::Green,
        }
    }
}

impl CoreRoboticAgent for RoboticAgent {
    fn body(&self) -> BodyHandle {
        self.body
    }

    fn platform(&self) -> PlatformType {
        self.platform
    }

    /// Run one perception-action tick through the real
    /// `MasterConsciousnessEquation` (via `EntityConsciousness::compute`) and
    /// return the resulting motor gain in `[0, 1]`.
    ///
    /// `observation` is platform-specific (see each demo's
    /// `consciousness_bridge.rs` for its packing convention). This method is
    /// deliberately shape-agnostic: `observation[0]` is treated as a
    /// prediction-error-like signal, and any remaining channels are averaged
    /// into an embodiment-grounding signal — never a PE-to-Phi shortcut, the
    /// full equation always runs.
    fn tick(&mut self, observation: &[f64], danger_level: f64) -> f64 {
        let danger = danger_level.clamp(0.0, 1.0);
        let prediction_error = observation.first().copied().unwrap_or(0.0).clamp(0.0, 2.0);
        let embodiment = if observation.len() > 2 {
            (observation[2..].iter().sum::<f64>() / (observation.len() - 2) as f64).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let inputs = ConsciousnessInputs {
            phi: (1.0 - danger).clamp(0.0, 1.0),
            broadcast: (1.0 - prediction_error / 2.0).clamp(0.0, 1.0),
            working_memory: 0.7,
            attention: (1.0 - danger).clamp(0.0, 1.0),
            recurrence: 0.6,
            embodiment,
            knowledge: 0.5,
            synchrony: (1.0 - prediction_error / 2.0).clamp(0.0, 1.0),
        };

        self.consciousness.compute(&inputs);
        self.safety_tier = self.consciousness.safety_tier;

        // Sustained danger raises caution; sustained safety lets it settle.
        self.caution += (danger - self.caution) * CAUTION_EMA_ALPHA;

        self.consciousness.effective_motor_gain().clamp(0.0, 1.0)
    }

    fn phi(&self) -> f64 {
        self.consciousness.phi()
    }

    fn bottleneck(&self) -> &str {
        self.consciousness.bottleneck()
    }

    fn can_act(&self) -> bool {
        self.safety_tier.allows_output()
    }
}

#[cfg(test)]
mod consciousness_agent_tests {
    use super::*;

    #[test]
    fn tick_produces_valid_phi_and_gain() {
        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Manipulator, "test");
        let gain = agent.tick(&[0.01, 0.0, 0.5, 0.5], 0.0);
        assert!((0.0..=1.0).contains(&gain));
        assert!((0.0..=1.0).contains(&agent.phi()));
    }

    #[test]
    fn sustained_danger_raises_caution() {
        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Manipulator, "test");
        let initial = agent.caution;
        for _ in 0..20 {
            agent.tick(&[0.8, 0.95], 0.95);
        }
        assert!(agent.caution > initial);
    }

    #[test]
    fn low_danger_decreases_caution_after_high_danger() {
        let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Manipulator, "test");
        for _ in 0..10 {
            agent.tick(&[0.8, 0.95], 0.95);
        }
        let high = agent.caution;
        for _ in 0..50 {
            agent.tick(&[0.01, 0.0], 0.0);
        }
        assert!(agent.caution < high);
    }
}
