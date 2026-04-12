// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 6DOF AUV physics simulator using hydrodynamic model.

use crate::hydrodynamics::{self, HydrodynamicConfig};
use crate::types::{AuvCommand, AuvState, NUM_ACTUATORS};

/// Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians.
/// Uses ZYX convention (aerospace standard).
fn quaternion_to_euler(q: &[f64; 4]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
    let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
    let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
    [roll, pitch, yaw]
}

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
    /// Energy reservoir (joules). Depletes with thrust.
    /// When exhausted, thrusters produce zero force (drift mode).
    energy_remaining_j: f64,
    /// Total energy consumed (joules, monotonically increasing).
    energy_consumed_j: f64,
    /// Energy capacity (joules). Default: 500 kJ (~14 hours at cruise).
    energy_capacity_j: f64,
}

impl SimpleAuvSimulator {
    pub fn new() -> Self {
        let capacity = 500_000.0; // 500 kJ — typical small AUV battery
        Self {
            state: AuvState::neutral_buoyancy(10.0),
            hydro_config: HydrodynamicConfig::default(),
            thruster_state: [0.0; NUM_ACTUATORS],
            external_force: [0.0; 3],
            energy_remaining_j: capacity,
            energy_consumed_j: 0.0,
            energy_capacity_j: capacity,
        }
    }

    /// Create with a custom energy budget (joules).
    pub fn with_energy_budget(energy_j: f64) -> Self {
        let mut sim = Self::new();
        sim.energy_capacity_j = energy_j;
        sim.energy_remaining_j = energy_j;
        sim
    }

    /// Remaining energy as fraction [0, 1].
    pub fn energy_fraction(&self) -> f64 {
        if self.energy_capacity_j <= 0.0 {
            return 0.0;
        }
        (self.energy_remaining_j / self.energy_capacity_j).clamp(0.0, 1.0)
    }

    /// Total energy consumed (joules).
    pub fn energy_consumed(&self) -> f64 {
        self.energy_consumed_j
    }

    /// Whether the battery is exhausted.
    pub fn energy_exhausted(&self) -> bool {
        self.energy_remaining_j <= 0.0
    }
}

impl Default for SimpleAuvSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl AuvPhysicsSimulator for SimpleAuvSimulator {
    fn step(&mut self, cmd: &AuvCommand, dt: f64) {
        // Energy-gated thruster commands: if battery exhausted, no thrust
        let effective_cmd = if self.energy_remaining_j > 0.0 {
            cmd.clone()
        } else {
            AuvCommand::zero() // Drift mode
        };

        let (thrust_force, thrust_moment) = hydrodynamics::thruster_forces(
            &self.hydro_config,
            &mut self.thruster_state,
            &effective_cmd.thrusters,
            dt,
        );

        // Energy depletion: P = Σ(F_i²) / η (power proportional to force²)
        // Efficiency η = 0.4 (typical electric thruster)
        // Energy = Power × dt
        let thrust_power: f64 = self.thruster_state.iter().map(|f| f * f).sum::<f64>() / 0.4;
        let energy_this_step = thrust_power * dt;
        self.energy_consumed_j += energy_this_step;
        self.energy_remaining_j = (self.energy_remaining_j - energy_this_step).max(0.0);

        // Extract Euler angles (roll, pitch, yaw) from quaternion for restoring moments
        let q = &self.state.quaternion;
        let orientation_angles = quaternion_to_euler(q);
        let hydro = hydrodynamics::compute_forces(
            &self.hydro_config,
            &self.state.linear_velocity,
            &self.state.angular_velocity,
            &orientation_angles,
            self.state.depth,
        );
        let eff = hydrodynamics::effective_mass(&self.hydro_config);

        // Physical velocity limits for a 50kg torpedo-class AUV.
        // Prevents explicit Euler divergence when quadratic drag coefficient
        // times velocity² exceeds inertia per timestep (stiff ODE).
        // Science: REMUS 100 max speed ~2.5 m/s; 5 m/s is generous upper bound.
        const MAX_LINEAR_VELOCITY: f64 = 5.0; // m/s
        const MAX_ANGULAR_VELOCITY: f64 = 3.0; // rad/s

        // Linear dynamics with velocity-limited integration.
        // The quadratic drag term F = -c*v*|v| is stiff: explicit Euler
        // diverges when |v| is large relative to dt. We apply the force
        // update then clamp, which is equivalent to an implicit drag floor.
        for i in 0..3 {
            let f = thrust_force[i] + hydro.force[i] + self.external_force[i];
            self.state.linear_velocity[i] += (f / eff[i]) * dt;
            self.state.linear_velocity[i] =
                self.state.linear_velocity[i].clamp(-MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY);
            self.state.position[i] += self.state.linear_velocity[i] * dt;
        }

        // Angular dynamics with velocity-limited integration.
        // Same stiffness issue: roll inertia (0.5 kg·m²) is small relative
        // to angular drag (623 Nm at 1 rad/s), causing oscillation→NaN at
        // dt=0.01 under full thrust. Clamping prevents the divergence.
        for i in 0..3 {
            let m = thrust_moment[i] + hydro.moment[i];
            self.state.angular_velocity[i] += (m / eff[3 + i].max(0.1)) * dt;
            self.state.angular_velocity[i] =
                self.state.angular_velocity[i].clamp(-MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY);
        }

        // Quaternion integration (semi-implicit Euler)
        let [qw, qx, qy, qz] = self.state.quaternion;
        let [wx, wy, wz] = self.state.angular_velocity;
        let h = 0.5 * dt;
        self.state.quaternion[0] += (-qx * wx - qy * wy - qz * wz) * h;
        self.state.quaternion[1] += (qw * wx + qy * wz - qz * wy) * h;
        self.state.quaternion[2] += (qw * wy - qx * wz + qz * wx) * h;
        self.state.quaternion[3] += (qw * wz + qx * wy - qy * wx) * h;
        let norm = self
            .state
            .quaternion
            .iter()
            .map(|q| q * q)
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for q in &mut self.state.quaternion {
                *q /= norm;
            }
        }

        // NaN guard: if any state variable goes non-finite (e.g., from
        // extreme external forces or numerical edge cases), reset velocity
        // to zero rather than propagating corruption through the loop.
        if !self.state.linear_velocity.iter().all(|v| v.is_finite()) {
            self.state.linear_velocity = [0.0; 3];
        }
        if !self.state.angular_velocity.iter().all(|v| v.is_finite()) {
            self.state.angular_velocity = [0.0; 3];
        }
        if !self.state.quaternion.iter().all(|v| v.is_finite()) {
            self.state.quaternion = [1.0, 0.0, 0.0, 0.0]; // Identity
        }

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
            self.state.thruster_feedback[i] =
                self.thruster_state[i] / self.hydro_config.max_thruster_force;
        }
        self.external_force = [0.0; 3];
    }

    fn state(&self) -> &AuvState {
        &self.state
    }

    fn reset(&mut self, depth: f64) {
        self.state = AuvState::neutral_buoyancy(depth);
        self.thruster_state = [0.0; NUM_ACTUATORS];
        self.external_force = [0.0; 3];
        self.energy_remaining_j = self.energy_capacity_j;
        self.energy_consumed_j = 0.0;
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
        for _ in 0..100 {
            sim.step(&AuvCommand::forward(0.5), 0.01);
        }
        assert!(sim.state().speed() > 0.0);
    }

    #[test]
    fn test_drag_limits_speed() {
        let mut sim = SimpleAuvSimulator::new();
        for _ in 0..10000 {
            sim.step(&AuvCommand::forward(0.5), 0.01);
        }
        assert!(
            sim.state().speed() < 10.0,
            "Drag should limit speed: {}",
            sim.state().speed()
        );
    }

    #[test]
    fn test_surface_constraint() {
        let mut sim = SimpleAuvSimulator::new();
        sim.reset(1.0);
        for _ in 0..5000 {
            sim.step(&AuvCommand::descend(-0.5), 0.01);
        }
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
        let cmd = AuvCommand {
            thrusters: [0.5, -0.3, 0.2, 0.0, 0.1, -0.1, 0.3, -0.2],
        };
        for _ in 0..1000 {
            sim.step(&cmd, 0.01);
        }
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
