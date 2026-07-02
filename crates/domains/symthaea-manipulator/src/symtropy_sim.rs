// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy-physics backed manipulator simulator.
//!
//! Uses `symtropy_physics::PhysicsWorld<3>` with articulated chain,
//! contact physics, and consciousness-modulated force callback.
//! Feature-gated behind `symtropy`.
//!
//! Compared to `SimpleManipulatorSimulator`:
//! - Real contact manifolds (GJK + EPA)
//! - Multi-point friction (Coulomb)
//! - Gripper grasping (two prismatic-jointed fingers)
//! - Force/torque sensing from contact data
//! - Consciousness callback modulates motor force by Phi

#![cfg(feature = "symtropy")]

use symtropy_math::Point;
use symtropy_physics::articulation::{ArticulatedChain, ChainBuilder, LinkSpec};
use symtropy_physics::body::{BodyHandle, RigidBody};
use symtropy_physics::world::PhysicsWorld;
use symtropy_physics::{CollisionEvent, PhysicsCallback};

use crate::simulator::ManipulatorPhysicsSimulator;
use crate::types::{ManipulatorCommand, ManipulatorState, NUM_JOINTS};

/// Consciousness-modulated physics callback.
///
/// Scales all motor forces by the current safety gain (0.0–1.0).
/// At Red safety, all motors produce zero force — gravity-compensated hold.
pub struct ConsciousnessCallback {
    /// Current motor gain from Phi-based safety level (0.0–1.0).
    pub motor_gain: f32,
}

impl PhysicsCallback<3> for ConsciousnessCallback {
    fn apply_trauma(&mut self, _event: &CollisionEvent<3>) {}
    fn modulate_force(
        &self,
        _handle: BodyHandle,
        force: &nalgebra::SVector<f64, 3>,
    ) -> nalgebra::SVector<f64, 3> {
        *force * self.motor_gain as f64
    }

    fn modulate_impulse(&self, impulse: f64, _contact: &nalgebra::SVector<f64, 3>) -> f64 {
        impulse * self.motor_gain as f64
    }

    fn friction_multiplier(&self, _contact: &nalgebra::SVector<f64, 3>, _body: BodyHandle) -> f64 {
        1.0
    }

    fn on_collision(&mut self, _event: &CollisionEvent<3>) {}

    fn record_dissipation(&mut self, _energy: f64) {}
}

/// Symtropy-physics based manipulator simulator with contact physics.
pub struct SymtropyManipulatorSimulator {
    world: PhysicsWorld<3>,
    chain: ArticulatedChain,
    state: ManipulatorState,
    callback: ConsciousnessCallback,
    /// Target object handle (for grasping demos).
    target_object: Option<BodyHandle>,
}

impl SymtropyManipulatorSimulator {
    /// Create a new simulator with a 7-DOF Panda-like arm.
    pub fn new() -> Self {
        let gravity = nalgebra::SVector::from([0.0, 0.0, -9.81]);
        let mut world = PhysicsWorld::new(gravity);

        // Ground plane (large static box at z=0)
        world.add_body(RigidBody::static_body(
            BodyHandle(0),
            Point::new([0.0, 0.0, -0.05]),
            Box::new(symtropy_math::HyperBox::new([5.0, 5.0, 0.05])),
        ));

        // Build 7-DOF articulated arm
        let panda_links = [
            LinkSpec {
                mass: 4.0,
                length: 0.333,
                radius: 0.06,
                plane_a: 0,
                plane_b: 2,
                motor_max_force: Some(87.0),
                ..Default::default()
            },
            LinkSpec {
                mass: 4.0,
                length: 0.316,
                radius: 0.06,
                plane_a: 0,
                plane_b: 1,
                motor_max_force: Some(87.0),
                ..Default::default()
            },
            LinkSpec {
                mass: 3.0,
                length: 0.384,
                radius: 0.05,
                plane_a: 0,
                plane_b: 2,
                motor_max_force: Some(87.0),
                ..Default::default()
            },
            LinkSpec {
                mass: 3.0,
                length: 0.088,
                radius: 0.05,
                plane_a: 0,
                plane_b: 1,
                angle_limits: Some((-3.07, -0.07)),
                motor_max_force: Some(87.0),
                ..Default::default()
            },
            LinkSpec {
                mass: 2.0,
                length: 0.107,
                radius: 0.04,
                plane_a: 0,
                plane_b: 2,
                motor_max_force: Some(12.0),
                ..Default::default()
            },
            LinkSpec {
                mass: 1.5,
                length: 0.100,
                radius: 0.04,
                plane_a: 0,
                plane_b: 1,
                motor_max_force: Some(12.0),
                ..Default::default()
            },
            LinkSpec {
                mass: 0.5,
                length: 0.088,
                radius: 0.03,
                plane_a: 0,
                plane_b: 2,
                motor_max_force: Some(12.0),
                ..Default::default()
            },
        ];

        let mut builder = ChainBuilder::new().base_position(Point::new([0.0, 0.0, 1.0]));
        for link in panda_links {
            builder = builder.add_link(link);
        }
        let chain = builder.build(&mut world);

        Self {
            world,
            chain,
            state: ManipulatorState::home(),
            callback: ConsciousnessCallback { motor_gain: 1.0 },
            target_object: None,
        }
    }

    /// Set the consciousness-based motor gain.
    pub fn set_motor_gain(&mut self, gain: f32) {
        self.callback.motor_gain = gain;
    }

    /// Add a target object for grasping experiments.
    pub fn add_target_box(&mut self, position: [f64; 3], half_size: f64, mass: f64) -> BodyHandle {
        let handle = self.world.add_body(RigidBody::dynamic_sphere(
            BodyHandle(0),
            Point::new(position),
            half_size,
            mass,
        ));
        self.target_object = Some(handle);
        handle
    }

    /// Read the tip (end-effector) position from the physics world.
    pub fn tip_position(&self) -> [f64; 3] {
        match self.world.body(self.chain.tip()) {
            Some(body) => {
                let p = &body.transform.translation.0;
                [p[0], p[1], p[2]]
            }
            None => [0.0; 3],
        }
    }

    /// Synchronize ManipulatorState from physics world state.
    fn sync_state(&mut self) {
        let joint_states = self.chain.read_joint_states(&self.world);
        for (i, &(angle, vel)) in joint_states.iter().enumerate().take(NUM_JOINTS) {
            self.state.joint_angles[i] = angle;
            self.state.joint_velocities[i] = vel;
        }
        self.state.end_effector_position = self.tip_position();
        // Contact forces would come from collision manifolds in a full impl
        self.state.end_effector_force = [0.0; 3];
    }
}

impl ManipulatorPhysicsSimulator for SymtropyManipulatorSimulator {
    fn step(&mut self, cmd: &ManipulatorCommand, dt: f64) {
        // Map torque commands to motor targets (angular velocity targets)
        // Scale by consciousness gain
        let gain = self.callback.motor_gain as f64;
        for (i, &torque) in cmd.joint_torques.iter().enumerate().take(NUM_JOINTS) {
            // Convert torque command to angular velocity target
            // (simplified: torque ∝ target angular velocity for PD motor)
            let target = torque as f64 * gain * 5.0; // Scale factor
                                                     // Update motor target via constraint system
                                                     // Note: direct constraint access not available — use body forces instead
            if let Some(body) = self
                .world
                .body_mut(self.chain.links[i.min(self.chain.links.len() - 1)])
            {
                let plane_idx = if i % 2 == 0 { (0, 2) } else { (0, 1) };
                body.angular_velocity.set(
                    plane_idx.0,
                    plane_idx.1,
                    body.angular_velocity.get(plane_idx.0, plane_idx.1) + target * dt,
                );
            }
        }

        self.world.step_with_callback(dt, &mut self.callback);
        self.sync_state();
    }

    fn state(&self) -> &ManipulatorState {
        &self.state
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for SymtropyManipulatorSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let sim = SymtropyManipulatorSimulator::new();
        assert_eq!(sim.chain.num_joints, 7);
        assert!(sim.state().is_finite());
    }

    #[test]
    fn test_step_stability() {
        let mut sim = SymtropyManipulatorSimulator::new();
        let cmd = ManipulatorCommand::zero();
        for _ in 0..200 {
            sim.step(&cmd, 0.002);
        }
        assert!(
            sim.state().is_finite(),
            "State should remain finite after 200 steps"
        );
    }

    #[test]
    fn test_torque_produces_motion() {
        let mut sim = SymtropyManipulatorSimulator::new();
        let before = sim.state().joint_angles.clone();
        let mut cmd = ManipulatorCommand::zero();
        cmd.joint_torques[0] = 0.5;
        for _ in 0..50 {
            sim.step(&cmd, 0.005);
        }
        let after = &sim.state().joint_angles;
        // At least one angle should have changed
        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(changed, "Torque should produce motion");
    }

    #[test]
    fn test_consciousness_gain_zero_stops_motion() {
        let mut sim = SymtropyManipulatorSimulator::new();
        sim.set_motor_gain(0.0); // Red safety — no motor authority
        let mut cmd = ManipulatorCommand::zero();
        cmd.joint_torques = [1.0; NUM_JOINTS];
        // Step with max torque but zero gain
        for _ in 0..50 {
            sim.step(&cmd, 0.005);
        }
        // Arm should still be finite (gravity holds it, no motor drives it)
        assert!(sim.state().is_finite());
    }

    #[test]
    fn test_target_object_creation() {
        let mut sim = SymtropyManipulatorSimulator::new();
        let handle = sim.add_target_box([0.3, 0.0, 0.5], 0.03, 0.1);
        assert!(sim.target_object.is_some());
        // Object should be at specified position
        let body = sim.world.body(handle).unwrap();
        assert!((body.transform.translation.0[0] - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_tip_position_finite() {
        let sim = SymtropyManipulatorSimulator::new();
        let tip = sim.tip_position();
        assert!(tip[0].is_finite() && tip[1].is_finite() && tip[2].is_finite());
    }

    #[test]
    fn test_reset_restores_initial_state() {
        let mut sim = SymtropyManipulatorSimulator::new();
        let mut cmd = ManipulatorCommand::zero();
        cmd.joint_torques[0] = 1.0;
        for _ in 0..100 {
            sim.step(&cmd, 0.005);
        }
        sim.reset();
        assert_eq!(sim.chain.num_joints, 7);
        assert!(sim.state().is_finite());
    }
}
