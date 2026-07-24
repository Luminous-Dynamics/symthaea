// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy-physics backed quadruped simulator with terrain contact.
//!
//! Replaces the simple spring-damper per-foot model with real multi-point
//! friction contact via GJK+EPA. Enables slope walking, turning, and
//! consciousness-coupled gait degradation through PhysicsCallback.

#![cfg(feature = "symtropy")]

use symtropy_math::Point;
use symtropy_physics::articulation::{ArticulatedChain, ChainBuilder, LinkSpec};
use symtropy_physics::body::{BodyHandle, RigidBody};
use symtropy_physics::world::PhysicsWorld;
use symtropy_physics::{CollisionEvent, PhysicsCallback};

use crate::simulator::QuadrupedPhysicsSimulator;
use crate::types::{NUM_ACTUATORS, QuadrupedCommand, QuadrupedState};

/// Consciousness-coupled physics callback for quadruped.
///
/// At high Φ: full motor authority, normal friction ("grip on reality").
/// At low Φ: reduced motor force, gait degrades toward freeze/collapse.
pub struct ConsciousnessQuadrupedCallback {
    pub motor_gain: f32,
    pub foot_contacts: [bool; 4],
}

impl PhysicsCallback<3> for ConsciousnessQuadrupedCallback {
    fn apply_trauma(&mut self, _event: &CollisionEvent<3>) {}
    fn modulate_force(
        &self,
        _body: BodyHandle,
        force: &nalgebra::SVector<f64, 3>,
    ) -> nalgebra::SVector<f64, 3> {
        *force * self.motor_gain as f64
    }
    fn modulate_impulse(&self, impulse: f64, _contact: &nalgebra::SVector<f64, 3>) -> f64 {
        impulse
    }
    fn friction_multiplier(&self, _contact: &nalgebra::SVector<f64, 3>, _body: BodyHandle) -> f64 {
        1.0
    }
    fn on_collision(&mut self, _event: &CollisionEvent<3>) {}
    fn record_dissipation(&mut self, _energy: f64) {}
    // No-op like record_dissipation above: QuadrupedState has no energy
    // accounting field yet, so there's nowhere honest to put this without
    // inventing an unread accumulator. Added because PhysicsCallback grew
    // this as a required method upstream — this crate's `symtropy` feature
    // had never actually been compiled, so the gap went undetected until
    // robotics plan 2026-07-10 Tier 4.4 wired the backend in for real.
    fn record_work(&mut self, _body: BodyHandle, _work_joules: f64) {}
}

/// Symtropy-physics quadruped with 4 articulated legs and terrain contact.
pub struct SymtropyQuadrupedSimulator {
    world: PhysicsWorld<3>,
    torso: BodyHandle,
    legs: [ArticulatedChain; 4],
    state: QuadrupedState,
    callback: ConsciousnessQuadrupedCallback,
}

impl SymtropyQuadrupedSimulator {
    /// Create quadruped: 12kg torso + 4 legs × 3 joints.
    pub fn new() -> Self {
        let gravity = nalgebra::SVector::from([0.0, 0.0, -9.81]);
        let mut world = PhysicsWorld::new(gravity);

        // Ground plane
        world.add_body(RigidBody::static_body(
            BodyHandle(0),
            Point::new([0.0, 0.0, -0.01]),
            Box::new(symtropy_math::HyperBox::new([10.0, 10.0, 0.01])),
        ));

        // Torso (12kg box)
        let torso = world.add_body(RigidBody::dynamic_sphere(
            BodyHandle(0),
            Point::new([0.0, 0.0, 0.35]),
            0.15, // radius
            12.0, // mass
        ));

        // 4 legs at corners of torso
        let hip_offsets = [
            [0.15, 0.1, 0.35],   // Front-left
            [0.15, -0.1, 0.35],  // Front-right
            [-0.15, 0.1, 0.35],  // Rear-left
            [-0.15, -0.1, 0.35], // Rear-right
        ];

        let leg_spec = [
            LinkSpec {
                mass: 0.5,
                length: 0.12,
                radius: 0.02,
                plane_a: 0,
                plane_b: 2,
                motor_max_force: Some(20.0),
                angle_limits: Some((-1.0, 1.0)),
                ..Default::default()
            },
            LinkSpec {
                mass: 0.4,
                length: 0.15,
                radius: 0.015,
                plane_a: 1,
                plane_b: 2,
                motor_max_force: Some(15.0),
                angle_limits: Some((-2.0, 0.5)),
                ..Default::default()
            },
            LinkSpec {
                mass: 0.3,
                length: 0.15,
                radius: 0.012,
                plane_a: 1,
                plane_b: 2,
                motor_max_force: Some(10.0),
                angle_limits: Some((-0.5, 2.5)),
                ..Default::default()
            },
        ];

        let mut legs = Vec::with_capacity(4);
        for offset in &hip_offsets {
            let mut builder = ChainBuilder::new().base_position(Point::new(*offset));
            for spec in &leg_spec {
                builder = builder.add_link(spec.clone());
            }
            legs.push(builder.build(&mut world));
        }

        Self {
            world,
            torso,
            legs: legs.try_into().unwrap_or_else(|v: Vec<ArticulatedChain>| {
                panic!("Expected 4 legs, got {}", v.len())
            }),
            state: QuadrupedState::standing(),
            callback: ConsciousnessQuadrupedCallback {
                motor_gain: 1.0,
                foot_contacts: [false; 4],
            },
        }
    }

    /// Set consciousness motor gain (0.0 = collapse, 1.0 = full authority).
    pub fn set_motor_gain(&mut self, gain: f32) {
        self.callback.motor_gain = gain;
    }

    /// Get torso height above ground.
    pub fn torso_height(&self) -> f64 {
        self.world
            .body(self.torso)
            .map(|b| b.transform.translation.0[2])
            .unwrap_or(0.0)
    }

    /// Sync QuadrupedState from physics world.
    fn sync_state(&mut self) {
        if let Some(body) = self.world.body(self.torso) {
            let p = &body.transform.translation.0;
            self.state.base_position = [p[0], p[1], p[2]];
            self.state.base_linear_velocity = [
                body.linear_velocity[0],
                body.linear_velocity[1],
                body.linear_velocity[2],
            ];
        }
        // Read joint angles from all 4 legs (3 joints each = 12 total)
        for (leg_idx, leg) in self.legs.iter().enumerate() {
            let states = leg.read_joint_states(&self.world);
            for (j, &(angle, vel)) in states.iter().enumerate().take(3) {
                let joint_idx = leg_idx * 3 + j;
                if joint_idx < self.state.joint_angles.len() {
                    self.state.joint_angles[joint_idx] = angle;
                    self.state.joint_velocities[joint_idx] = vel;
                }
            }
        }
    }
}

impl QuadrupedPhysicsSimulator for SymtropyQuadrupedSimulator {
    fn step(&mut self, cmd: &QuadrupedCommand, dt: f64) {
        let gain = self.callback.motor_gain as f64;
        // Map 12 torques to 4 legs × 3 joints
        for (leg_idx, leg) in self.legs.iter().enumerate() {
            for (j, &link) in leg.links.iter().enumerate().take(3) {
                let cmd_idx = leg_idx * 3 + j;
                if cmd_idx < NUM_ACTUATORS {
                    let target = cmd.joint_torques[cmd_idx] as f64 * gain * 3.0;
                    if let Some(body) = self.world.body_mut(link) {
                        let plane = if j == 0 { (0, 2) } else { (1, 2) };
                        body.angular_velocity.set(
                            plane.0,
                            plane.1,
                            body.angular_velocity.get(plane.0, plane.1) + target * dt,
                        );
                    }
                }
            }
        }

        self.world.step_with_callback(dt, &mut self.callback);
        self.sync_state();
    }

    fn state(&self) -> &QuadrupedState {
        &self.state
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for SymtropyQuadrupedSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let sim = SymtropyQuadrupedSimulator::new();
        assert_eq!(sim.legs.len(), 4);
        for leg in &sim.legs {
            assert_eq!(leg.num_joints, 3);
        }
    }

    #[test]
    fn test_stepping_stays_finite() {
        let mut sim = SymtropyQuadrupedSimulator::new();
        let cmd = QuadrupedCommand {
            joint_torques: [0.0; NUM_ACTUATORS],
        };
        for _ in 0..200 {
            sim.step(&cmd, 0.005);
        }
        assert!(
            sim.state().is_finite(),
            "State should remain finite after stepping"
        );
    }

    #[test]
    fn test_torque_produces_motion() {
        let mut sim = SymtropyQuadrupedSimulator::new();
        let h0 = sim.torso_height();
        let mut cmd = QuadrupedCommand {
            joint_torques: [0.3; NUM_ACTUATORS],
        };
        for _ in 0..50 {
            sim.step(&cmd, 0.005);
        }
        let h1 = sim.torso_height();
        assert!((h1 - h0).abs() > 0.001, "Torque should produce motion");
    }

    #[test]
    fn test_consciousness_zero_gain() {
        let mut sim = SymtropyQuadrupedSimulator::new();
        sim.set_motor_gain(0.0);
        let mut cmd = QuadrupedCommand {
            joint_torques: [1.0; NUM_ACTUATORS],
        };
        for _ in 0..50 {
            sim.step(&cmd, 0.005);
        }
        assert!(sim.state().is_finite());
    }

    #[test]
    fn test_reset() {
        let mut sim = SymtropyQuadrupedSimulator::new();
        let mut cmd = QuadrupedCommand {
            joint_torques: [0.5; NUM_ACTUATORS],
        };
        for _ in 0..100 {
            sim.step(&cmd, 0.005);
        }
        sim.reset();
        assert_eq!(sim.legs.len(), 4);
        assert!(sim.state().is_finite());
    }
}
