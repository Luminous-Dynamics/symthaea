// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy-physics exoskeleton: human-robot co-embodiment with consciousness-mediated impedance.
//!
//! Two parallel 3-joint chains (human legs + exo legs) coupled at the hip.
//! At high Φ: exo actively assists (stiff, high motor force).
//! At low Φ: exo becomes transparent (compliant, low impedance, backdrivable).
//!
//! This is the unique consciousness-coupling demo — no other framework offers
//! adaptive impedance driven by a measurable consciousness metric.

#![cfg(feature = "symtropy")]

use symtropy_math::Point;
use symtropy_physics::articulation::{ArticulatedChain, ChainBuilder, LinkSpec};
use symtropy_physics::body::{BodyHandle, RigidBody};
use symtropy_physics::world::PhysicsWorld;
use symtropy_physics::{CollisionEvent, PhysicsCallback};

use crate::simulator::ExoskeletonPhysicsSimulator;
use crate::types::{ExoskeletonCommand, ExoskeletonState, NUM_ACTUATORS, NUM_JOINTS};

/// Consciousness-mediated impedance callback.
///
/// High Φ → stiff assist (motor_gain=1.0, high friction).
/// Low Φ → transparent compliance (motor_gain→0, low friction = backdrivable).
pub struct ConsciousnessExoCallback {
    pub motor_gain: f32,
    pub stiffness_factor: f64,
}

impl PhysicsCallback<3> for ConsciousnessExoCallback {
    fn apply_trauma(&mut self, _event: &CollisionEvent<3>) {}
    fn modulate_force(
        &self,
        _body: BodyHandle,
        force: &nalgebra::SVector<f64, 3>,
    ) -> nalgebra::SVector<f64, 3> {
        // Exo forces scaled by consciousness-mediated stiffness
        *force * self.motor_gain as f64 * self.stiffness_factor
    }
    fn modulate_impulse(&self, impulse: f64, _contact: &nalgebra::SVector<f64, 3>) -> f64 {
        impulse
    }
    fn friction_multiplier(&self, _contact: &nalgebra::SVector<f64, 3>, _body: BodyHandle) -> f64 {
        // At low consciousness, reduce friction → more backdrivable
        0.5 + 0.5 * self.motor_gain as f64
    }
    fn on_collision(&mut self, _event: &CollisionEvent<3>) {}
    fn record_dissipation(&mut self, _energy: f64) {}
}

/// Symtropy exoskeleton with human-robot coupled chains.
pub struct SymtropyExoskeletonSimulator {
    world: PhysicsWorld<3>,
    /// Left leg exo chain (hip → thigh → shank).
    left_exo: ArticulatedChain,
    /// Right leg exo chain.
    right_exo: ArticulatedChain,
    state: ExoskeletonState,
    callback: ConsciousnessExoCallback,
    /// Simulated human effort (sinusoidal gait torques).
    gait_phase: f64,
}

impl SymtropyExoskeletonSimulator {
    pub fn new() -> Self {
        let gravity = nalgebra::SVector::from([0.0, 0.0, -9.81]);
        let mut world = PhysicsWorld::new(gravity);

        // Ground plane
        world.add_body(RigidBody::static_body(
            BodyHandle(0),
            Point::new([0.0, 0.0, -0.01]),
            Box::new(symtropy_math::HyperBox::new([5.0, 5.0, 0.01])),
        ));

        // Exo leg spec: 3 joints per leg (hip, thigh, shank)
        let leg_spec = [
            LinkSpec {
                mass: 1.5,
                length: 0.15,
                radius: 0.04,
                plane_a: 0,
                plane_b: 2,
                motor_max_force: Some(40.0),
                angle_limits: Some((-0.5, 1.5)),
                ..Default::default()
            },
            LinkSpec {
                mass: 1.2,
                length: 0.40,
                radius: 0.035,
                plane_a: 1,
                plane_b: 2,
                motor_max_force: Some(30.0),
                angle_limits: Some((0.0, 2.4)),
                ..Default::default()
            },
            LinkSpec {
                mass: 0.8,
                length: 0.40,
                radius: 0.03,
                plane_a: 1,
                plane_b: 2,
                motor_max_force: Some(20.0),
                angle_limits: Some((-0.5, 0.5)),
                ..Default::default()
            },
        ];

        // Left leg at (0, 0.1, 0.9)
        let mut left_builder = ChainBuilder::new().base_position(Point::new([0.0, 0.1, 0.9]));
        for spec in &leg_spec {
            left_builder = left_builder.add_link(spec.clone());
        }
        let left_exo = left_builder.build(&mut world);

        // Right leg at (0, -0.1, 0.9)
        let mut right_builder = ChainBuilder::new().base_position(Point::new([0.0, -0.1, 0.9]));
        for spec in &leg_spec {
            right_builder = right_builder.add_link(spec.clone());
        }
        let right_exo = right_builder.build(&mut world);

        Self {
            world,
            left_exo,
            right_exo,
            state: ExoskeletonState::standing(),
            callback: ConsciousnessExoCallback {
                motor_gain: 1.0,
                stiffness_factor: 1.0,
            },
            gait_phase: 0.0,
        }
    }

    /// Set consciousness-mediated impedance.
    pub fn set_consciousness_impedance(&mut self, motor_gain: f32, stiffness: f64) {
        self.callback.motor_gain = motor_gain;
        self.callback.stiffness_factor = stiffness;
    }

    /// Sync state from physics world.
    fn sync_state(&mut self) {
        // Left leg joints (0-2)
        let left_states = self.left_exo.read_joint_states(&self.world);
        for (j, &(angle, vel)) in left_states.iter().enumerate().take(3) {
            self.state.joint_angles[j] = angle;
            self.state.joint_velocities[j] = vel;
        }
        // Right leg joints (3-5)
        let right_states = self.right_exo.read_joint_states(&self.world);
        for (j, &(angle, vel)) in right_states.iter().enumerate().take(3) {
            self.state.joint_angles[3 + j] = angle;
            self.state.joint_velocities[3 + j] = vel;
        }
    }
}

impl ExoskeletonPhysicsSimulator for SymtropyExoskeletonSimulator {
    fn step(&mut self, cmd: &ExoskeletonCommand, dt: f64) {
        let gain = self.callback.motor_gain as f64;

        // Apply exo torques to left leg
        for (j, &link) in self.left_exo.links.iter().enumerate().take(3) {
            let torque = cmd.joint_torques[j] as f64 * gain * cmd.stiffness_gain as f64;
            if let Some(body) = self.world.body_mut(link) {
                let plane = if j == 0 { (0, 2) } else { (1, 2) };
                body.angular_velocity.set(
                    plane.0,
                    plane.1,
                    body.angular_velocity.get(plane.0, plane.1) + torque * dt,
                );
            }
        }

        // Apply exo torques to right leg
        for (j, &link) in self.right_exo.links.iter().enumerate().take(3) {
            let torque = cmd.joint_torques[3 + j] as f64 * gain * cmd.stiffness_gain as f64;
            if let Some(body) = self.world.body_mut(link) {
                let plane = if j == 0 { (0, 2) } else { (1, 2) };
                body.angular_velocity.set(
                    plane.0,
                    plane.1,
                    body.angular_velocity.get(plane.0, plane.1) + torque * dt,
                );
            }
        }

        // Simulate human gait effort (sinusoidal)
        self.gait_phase += dt * 2.0 * std::f64::consts::PI; // 1 Hz gait
        let human_effort = self.gait_phase.sin() * 5.0; // 5 Nm peak
        for j in 0..NUM_JOINTS {
            self.state.human_torques[j] = human_effort * if j < 3 { 1.0 } else { -1.0 };
        }

        self.world.step_with_callback(dt, &mut self.callback);
        self.sync_state();
    }

    fn state(&self) -> &ExoskeletonState {
        &self.state
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for SymtropyExoskeletonSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let sim = SymtropyExoskeletonSimulator::new();
        assert_eq!(sim.left_exo.num_joints, 3);
        assert_eq!(sim.right_exo.num_joints, 3);
    }

    #[test]
    fn test_stepping_finite() {
        let mut sim = SymtropyExoskeletonSimulator::new();
        let cmd = ExoskeletonCommand::zero();
        for _ in 0..200 {
            sim.step(&cmd, 0.005);
        }
        assert!(sim.state().is_finite());
    }

    #[test]
    fn test_high_consciousness_active_assist() {
        let mut sim = SymtropyExoskeletonSimulator::new();
        sim.set_consciousness_impedance(1.0, 1.5); // High stiffness
        let mut cmd = ExoskeletonCommand::zero();
        cmd.joint_torques = [0.5; NUM_ACTUATORS];
        cmd.stiffness_gain = 1.0;
        for _ in 0..50 {
            sim.step(&cmd, 0.005);
        }
        // Should have moved joints
        let angles = &sim.state().joint_angles;
        let moved = angles.iter().any(|a| a.abs() > 0.001);
        assert!(moved, "Active assist should move joints");
    }

    #[test]
    fn test_low_consciousness_transparent() {
        let mut sim = SymtropyExoskeletonSimulator::new();
        sim.set_consciousness_impedance(0.0, 0.1); // Low Φ → transparent
        let mut cmd = ExoskeletonCommand::zero();
        cmd.joint_torques = [1.0; NUM_ACTUATORS]; // Max command
        for _ in 0..50 {
            sim.step(&cmd, 0.005);
        }
        assert!(sim.state().is_finite(), "Transparent mode should be stable");
    }

    #[test]
    fn test_reset() {
        let mut sim = SymtropyExoskeletonSimulator::new();
        let cmd = ExoskeletonCommand::zero();
        for _ in 0..100 {
            sim.step(&cmd, 0.005);
        }
        sim.reset();
        assert_eq!(sim.left_exo.num_joints, 3);
        assert!(sim.state().is_finite());
    }
}
