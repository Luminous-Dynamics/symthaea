// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 6DOF AUV physics simulator using hydrodynamic model.

use crate::hydrodynamics::{self, HydrodynamicConfig};
use crate::types::{AuvCommand, AuvState, NUM_ACTUATORS};

/// Trait for AUV physics simulation backends.
pub trait AuvPhysicsSimulator {
    fn step(&mut self, cmd: &AuvCommand, dt: f64);
    fn state(&self) -> &AuvState;
    fn reset(&mut self, depth: f64);
    fn apply_external_force(&mut self, force: [f64; 3]);
}

/// Simple Rust-native 6DOF hydrodynamic simulator.
pub struct SimpleAuvSimulator {
    state: AuvState,
    hydro_config: HydrodynamicConfig,
    thruster_state: [f64; NUM_ACTUATORS],
    external_force: [f64; 3],
}

impl SimpleAuvSimulator {
    pub fn new() -> Self {
        Self {
            state: AuvState::neutral_buoyancy(10.0),
            hydro_config: HydrodynamicConfig::default(),
            thruster_state: [0.0; NUM_ACTUATORS],
            external_force: [0.0; 3],
        }
    }
}

impl Default for SimpleAuvSimulator {
    fn default() -> Self { Self::new() }
}

impl AuvPhysicsSimulator for SimpleAuvSimulator {
    fn step(&mut self, cmd: &AuvCommand, dt: f64) {
        let (thrust_force, thrust_moment) = hydrodynamics::thruster_forces(
            &self.hydro_config, &mut self.thruster_state, &cmd.thrusters, dt,
        );
        let hydro = hydrodynamics::compute_forces(
            &self.hydro_config, &self.state.linear_velocity, &self.state.angular_velocity, self.state.depth,
        );
        let eff = hydrodynamics::effective_mass(&self.hydro_config);

        // Linear dynamics
        for i in 0..3 {
            let f = thrust_force[i] + hydro.force[i] + self.external_force[i];
            self.state.linear_velocity[i] += (f / eff[i]) * dt;
            self.state.position[i] += self.state.linear_velocity[i] * dt;
        }

        // Angular dynamics
        for i in 0..3 {
            let m = thrust_moment[i] + hydro.moment[i];
            self.state.angular_velocity[i] += (m / eff[3 + i].max(0.1)) * dt;
        }

        // Quaternion integration
        let [qw, qx, qy, qz] = self.state.quaternion;
        let [wx, wy, wz] = self.state.angular_velocity;
        let h = 0.5 * dt;
        self.state.quaternion[0] += (-qx * wx - qy * wy - qz * wz) * h;
        self.state.quaternion[1] += (qw * wx + qy * wz - qz * wy) * h;
        self.state.quaternion[2] += (qw * wy - qx * wz + qz * wx) * h;
        self.state.quaternion[3] += (qw * wz + qx * wy - qy * wx) * h;
        let norm = self.state.quaternion.iter().map(|q| q * q).sum::<f64>().sqrt();
        if norm > 1e-10 { for q in &mut self.state.quaternion { *q /= norm; } }

        // Surface constraint
        self.state.depth = self.state.position[2].max(0.0);
        if self.state.position[2] < 0.0 {
            self.state.position[2] = 0.0;
            self.state.linear_velocity[2] = self.state.linear_velocity[2].max(0.0);
        }

        // Derived state
        self.state.pressure = 101.325 + self.state.depth * 9.81;
        self.state.buoyancy_force = hydro.buoyancy;
        for i in 0..NUM_ACTUATORS {
            self.state.thruster_feedback[i] = self.thruster_state[i] / self.hydro_config.max_thruster_force;
        }
        self.external_force = [0.0; 3];
    }

    fn state(&self) -> &AuvState { &self.state }

    fn reset(&mut self, depth: f64) {
        self.state = AuvState::neutral_buoyancy(depth);
        self.thruster_state = [0.0; NUM_ACTUATORS];
        self.external_force = [0.0; 3];
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        self.external_force = force;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_thrust_accelerates() {
        let mut sim = SimpleAuvSimulator::new();
        for _ in 0..100 { sim.step(&AuvCommand::forward(0.5), 0.01); }
        assert!(sim.state().speed() > 0.0);
    }

    #[test]
    fn test_drag_limits_speed() {
        let mut sim = SimpleAuvSimulator::new();
        for _ in 0..10000 { sim.step(&AuvCommand::forward(0.5), 0.01); }
        assert!(sim.state().speed() < 10.0, "Drag should limit speed: {}", sim.state().speed());
    }

    #[test]
    fn test_surface_constraint() {
        let mut sim = SimpleAuvSimulator::new();
        sim.reset(1.0);
        for _ in 0..5000 { sim.step(&AuvCommand::descend(-0.5), 0.01); }
        assert!(sim.state().position[2] >= 0.0);
    }

    #[test]
    fn test_external_force() {
        let mut sim = SimpleAuvSimulator::new();
        sim.apply_external_force([100.0, 0.0, 0.0]);
        sim.step(&AuvCommand::zero(), 0.01);
        assert!(sim.state().linear_velocity[0] > 0.0);
    }

    #[test]
    fn test_quaternion_normalized() {
        let mut sim = SimpleAuvSimulator::new();
        let cmd = AuvCommand { thrusters: [0.5, -0.3, 0.2, 0.0, 0.1, -0.1, 0.3, -0.2] };
        for _ in 0..1000 { sim.step(&cmd, 0.01); }
        let q = sim.state().quaternion;
        let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");
    }

    #[test]
    fn test_reset() {
        let mut sim = SimpleAuvSimulator::new();
        sim.step(&AuvCommand::forward(1.0), 0.1);
        sim.reset(50.0);
        assert_eq!(sim.state().speed(), 0.0);
        assert!((sim.state().depth - 50.0).abs() < 1e-10);
    }
}
