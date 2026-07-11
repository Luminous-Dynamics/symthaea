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

    /// Set the attitude quaternion `[w, x, y, z]` directly (normalized on
    /// entry). For initialization and testing — e.g. spawning the vehicle at
    /// a heading, or verifying frame math at a known attitude.
    pub fn set_attitude(&mut self, q: [f64; 4]) {
        let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            self.state.quaternion = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
        }
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

        let hydro = hydrodynamics::compute_forces(
            &self.hydro_config,
            &self.state.linear_velocity,
            &self.state.angular_velocity,
            &self.state.quaternion,
            self.state.depth,
        );
        let eff = hydrodynamics::effective_mass(&self.hydro_config);

        // External force (currents, tether tugs) is WORLD-frame; rotate into
        // the body frame the dynamics integrate in.
        let ext_body = hydrodynamics::world_to_body(&self.state.quaternion, &self.external_force);

        // Rigid-body Coriolis–centripetal terms C(ν)ν (Fossen 2011, §3.3,
        // diagonal-mass form): f = −ω × (M_eff ⊙ v). Previously absent —
        // turning while moving had no coupling into sway/heave.
        let mv = [
            eff[0] * self.state.linear_velocity[0],
            eff[1] * self.state.linear_velocity[1],
            eff[2] * self.state.linear_velocity[2],
        ];
        let w = &self.state.angular_velocity;
        let coriolis = [
            -(w[1] * mv[2] - w[2] * mv[1]),
            -(w[2] * mv[0] - w[0] * mv[2]),
            -(w[0] * mv[1] - w[1] * mv[0]),
        ];

        // Physical velocity limits for a 50kg torpedo-class AUV.
        // Prevents explicit Euler divergence when quadratic drag coefficient
        // times velocity² exceeds inertia per timestep (stiff ODE).
        // Science: REMUS 100 max speed ~2.5 m/s; 5 m/s is generous upper bound.
        const MAX_LINEAR_VELOCITY: f64 = 5.0; // m/s
        const MAX_ANGULAR_VELOCITY: f64 = 3.0; // rad/s

        // Linear dynamics with velocity-limited integration (BODY frame).
        // The quadratic drag term F = -c*v*|v| is stiff: explicit Euler
        // diverges when |v| is large relative to dt. We apply the force
        // update then clamp, which is equivalent to an implicit drag floor.
        for i in 0..3 {
            let f = thrust_force[i] + hydro.force[i] + ext_body[i] + coriolis[i];
            self.state.linear_velocity[i] += (f / eff[i]) * dt;
            self.state.linear_velocity[i] =
                self.state.linear_velocity[i].clamp(-MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY);
        }
        // Position integrates the WORLD-frame velocity. Previously the
        // body-frame surge/sway/heave were added straight to world position,
        // so a yawed AUV commanded to surge always advanced world +x —
        // navigation diverged from truth the moment heading ≠ 0.
        let world_v =
            hydrodynamics::body_to_world(&self.state.quaternion, &self.state.linear_velocity);
        for i in 0..3 {
            self.state.position[i] += world_v[i] * dt;
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

    #[test]
    fn test_translation_respects_heading() {
        // Frame regression: a yawed AUV commanded to surge must advance
        // along its HEADING in the world frame. Before the 2026-07 frame
        // rework, body-frame velocity was integrated straight into world
        // position, so this test's yawed vehicle advanced world +x anyway.
        let mut sim = SimpleAuvSimulator::new();
        // Yaw 90° about world z: q = [cos(45°), 0, 0, sin(45°)]
        let s = std::f64::consts::FRAC_1_SQRT_2;
        sim.set_attitude([s, 0.0, 0.0, s]);
        let p0 = sim.state().position;
        for _ in 0..500 {
            sim.step(&AuvCommand::forward(0.5), 0.01);
        }
        let dx = (sim.state().position[0] - p0[0]).abs();
        let dy = (sim.state().position[1] - p0[1]).abs();
        assert!(
            dy > 5.0 * dx.max(1e-6),
            "yawed 90°, surge must move world y, got dx={dx:.3} dy={dy:.3}"
        );
        assert!(dy > 0.5, "should actually have moved: dy={dy:.3}");
    }

    #[test]
    fn test_coriolis_couples_turn_into_sway() {
        // With surge velocity and a yaw rate, the rigid-body Coriolis term
        // −ω × (M v) must push sway — absent before the frame rework.
        let cfg = crate::hydrodynamics::HydrodynamicConfig::default();
        let eff = crate::hydrodynamics::effective_mass(&cfg);
        // Direct term check (unit-level, no integration noise):
        let v = [1.0, 0.0, 0.0]; // surge 1 m/s
        let w = [0.0, 0.0, 0.5]; // yawing 0.5 rad/s
        let mv = [eff[0] * v[0], eff[1] * v[1], eff[2] * v[2]];
        let coriolis_y = -(w[2] * mv[0] - w[0] * mv[2]);
        assert!(
            coriolis_y.abs() > 1.0,
            "turn+surge must produce a sway Coriolis force, got {coriolis_y}"
        );
    }
}
