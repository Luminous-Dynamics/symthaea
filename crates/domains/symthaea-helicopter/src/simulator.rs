// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics simulator trait and simple implementation for SAR helicopter.

use crate::rotor_dynamics::{RotorDynamics, RotorOutput};
use crate::types::{HelicopterCommand, HelicopterState};
use crate::wind_model::{WindConfig, WindModel};

/// Trait for helicopter physics simulation backends.
pub trait HelicopterPhysicsSimulator {
    /// Advance one timestep with the given motor command.
    fn step(&mut self, cmd: &HelicopterCommand, dt: f64);
    /// Current flight state.
    fn state(&self) -> &HelicopterState;
    /// Reset to hover at the given altitude.
    fn reset(&mut self, altitude: f64);
    /// Reset to hover with a deterministic perturbation.
    fn reset_with_perturbation(&mut self, altitude: f64, perturbation: f64, seed: u64);
    /// Apply an external force (world-frame, Newtons) for the next step.
    fn apply_external_force(&mut self, force: [f64; 3]);
}

/// Simple Rust-native physics simulator for SAR helicopter.
///
/// Uses the [`RotorDynamics`] model for thrust/torque computation and
/// semi-implicit Euler integration for body dynamics.
///
/// Atmosphere: owns a [`WindModel`] — each step the gust state evolves,
/// aerodynamic drag acts on *relative* airspeed (vehicle velocity minus
/// wind), and rotor thrust is multiplied by the Cheeseman-Bennett ground
/// effect ratio. The default wind config is calm (zero steady wind, zero
/// gusts), which reproduces still-air drag exactly; enable wind explicitly
/// via [`SimpleHelicopterSimulator::set_wind`] with a seed so runs stay
/// deterministic.
///
/// Mass: 500 kg (light SAR helicopter, e.g., Robinson R44 class).
pub struct SimpleHelicopterSimulator {
    state: HelicopterState,
    rotor: RotorDynamics,
    wind: WindModel,
    mass: f64,
    inertia: [f64; 3],
    external_force: [f64; 3],
    drag_coeff: f64,
    angular_damping: f64,
}

impl SimpleHelicopterSimulator {
    /// Create a new simulator hovering at 20m (calm air).
    pub fn new() -> Self {
        Self {
            state: HelicopterState::hover(20.0),
            rotor: RotorDynamics::new(),
            wind: WindModel::calm(),
            mass: 500.0,
            inertia: [1500.0, 1500.0, 2000.0],
            external_force: [0.0; 3],
            drag_coeff: 0.15, // Higher drag than quadrotor (larger body)
            angular_damping: 2.0,
        }
    }

    /// Enable wind with the given config and gust seed (deterministic).
    pub fn set_wind(&mut self, config: WindConfig, seed: u64) {
        self.wind = WindModel::with_seed(config, seed);
    }

    /// Access the current wind model.
    pub fn wind_model(&self) -> &WindModel {
        &self.wind
    }
}

impl Default for SimpleHelicopterSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl HelicopterPhysicsSimulator for SimpleHelicopterSimulator {
    fn step(&mut self, cmd: &HelicopterCommand, dt: f64) {
        let g = 9.81;

        // 1. Step rotor dynamics
        let rotor_out: RotorOutput = self.rotor.step(
            cmd.thrust as f64,
            cmd.tail_rotor as f64,
            cmd.collective as f64,
            cmd.cyclic_lon as f64,
            cmd.cyclic_lat as f64,
            dt,
        );

        // Update rotor feedback in state
        self.state.main_rotor_rpm = self.rotor.state().main_rpm;
        self.state.tail_rotor_rpm = self.rotor.state().tail_rpm;
        self.state.collective_pitch = cmd.collective as f64 * 0.26; // Normalized → radians
        self.state.cyclic_lon_feedback = cmd.cyclic_lon as f64 * 0.15;
        self.state.cyclic_lat_feedback = cmd.cyclic_lat as f64 * 0.15;

        // 2. Handle autorotation override
        if let Some(descent_rate) = rotor_out.descent_rate_override {
            self.state.linear_velocity[2] = descent_rate;
            self.state.position[2] += self.state.linear_velocity[2] * dt;
            if self.state.position[2] < 0.0 {
                self.state.position[2] = 0.0;
                self.state.linear_velocity = [0.0; 3];
            }
            return;
        }

        // 3. Thrust vector in world frame (rotate body-z by quaternion),
        //    augmented by ground effect near the surface (Cheeseman-Bennett).
        let [w, x, y, z] = self.state.quaternion;
        let thrust = rotor_out.thrust_force * self.wind.ground_effect_ratio(self.state.position[2]);

        let fx = 2.0 * (x * z + w * y) * thrust + self.external_force[0];
        let fy = 2.0 * (y * z - w * x) * thrust + self.external_force[1];
        let fz = (1.0 - 2.0 * (x * x + y * y)) * thrust + self.external_force[2];

        // 4. Linear acceleration with quadratic drag: F_drag = -c × |v_rel| × v_rel
        // Quadratic drag better models aerodynamic forces at helicopter speeds
        // (Fossen 2011). The drag coefficient absorbs 0.5 × ρ × Cd × A / mass.
        // Drag acts on airspeed relative to the wind: with the default calm
        // config wind_vel is exactly [0,0,0] and this reduces to still-air drag.
        let wind_vel = self.wind.wind_velocity_step(dt);
        let drag = self.drag_coeff / self.mass;
        let vx = self.state.linear_velocity[0] - wind_vel[0];
        let vy = self.state.linear_velocity[1] - wind_vel[1];
        let vz = self.state.linear_velocity[2] - wind_vel[2];
        let ax = fx / self.mass - drag * vx.abs() * vx;
        let ay = fy / self.mass - drag * vy.abs() * vy;
        let az = fz / self.mass - g - drag * vz.abs() * vz;

        // 5. Semi-implicit Euler
        self.state.linear_velocity[0] += ax * dt;
        self.state.linear_velocity[1] += ay * dt;
        self.state.linear_velocity[2] += az * dt;

        self.state.position[0] += self.state.linear_velocity[0] * dt;
        self.state.position[1] += self.state.linear_velocity[1] * dt;
        self.state.position[2] += self.state.linear_velocity[2] * dt;

        // Ground constraint
        if self.state.position[2] < 0.0 {
            self.state.position[2] = 0.0;
            self.state.linear_velocity[2] = 0.0;
        }

        // 6. Angular dynamics
        // Cyclic → body moments + gyroscopic precession from rotor model
        let moment_roll = cmd.cyclic_lat as f64 * 500.0 / self.inertia[0]
            + rotor_out.precession_roll / self.inertia[0];
        let moment_pitch = cmd.cyclic_lon as f64 * 500.0 / self.inertia[1]
            + rotor_out.precession_pitch / self.inertia[1];
        // Pedal → yaw moment, offset by torque reaction
        let moment_yaw = (cmd.pedal as f64 * 300.0 - rotor_out.torque_reaction) / self.inertia[2];

        self.state.angular_velocity[0] += moment_roll * dt;
        self.state.angular_velocity[1] += moment_pitch * dt;
        self.state.angular_velocity[2] += moment_yaw * dt;

        // Angular damping
        let ang_decay = (-self.angular_damping * dt).exp();
        for av in &mut self.state.angular_velocity {
            *av *= ang_decay;
        }

        // 7. Quaternion integration (semi-implicit)
        let [qw, qx, qy, qz] = self.state.quaternion;
        let [wx, wy, wz] = self.state.angular_velocity;
        let half_dt = 0.5 * dt;
        let dqw = (-qx * wx - qy * wy - qz * wz) * half_dt;
        let dqx = (qw * wx + qy * wz - qz * wy) * half_dt;
        let dqy = (qw * wy - qx * wz + qz * wx) * half_dt;
        let dqz = (qw * wz + qx * wy - qy * wx) * half_dt;

        self.state.quaternion[0] += dqw;
        self.state.quaternion[1] += dqx;
        self.state.quaternion[2] += dqy;
        self.state.quaternion[3] += dqz;

        // Normalize quaternion
        let norm = (self.state.quaternion[0].powi(2)
            + self.state.quaternion[1].powi(2)
            + self.state.quaternion[2].powi(2)
            + self.state.quaternion[3].powi(2))
        .sqrt();
        if norm > 1e-10 {
            for q in &mut self.state.quaternion {
                *q /= norm;
            }
        }

        // Clear one-shot external force
        self.external_force = [0.0; 3];
    }

    fn state(&self) -> &HelicopterState {
        &self.state
    }

    fn reset(&mut self, altitude: f64) {
        self.state = HelicopterState::hover(altitude);
        self.rotor.reset();
        self.wind.reset();
        self.external_force = [0.0; 3];
    }

    fn reset_with_perturbation(&mut self, altitude: f64, perturbation: f64, _seed: u64) {
        self.reset(altitude);
        // Add lateral velocity as perturbation
        self.state.linear_velocity[0] = perturbation;
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        self.external_force = force;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hover_stability() {
        let mut sim = SimpleHelicopterSimulator::new();
        let cmd = HelicopterCommand::hover();
        let initial_alt = sim.state().altitude();

        // Run 1 second at 300Hz
        for _ in 0..300 {
            sim.step(&cmd, 1.0 / 300.0);
        }

        let final_alt = sim.state().altitude();
        // Should stay roughly at hover altitude (within 5m)
        assert!(
            (final_alt - initial_alt).abs() < 5.0,
            "Hover should be roughly stable: initial={initial_alt}, final={final_alt}"
        );
        assert!(sim.state().is_finite(), "State must remain finite");
    }

    #[test]
    fn test_zero_command_descends() {
        let mut sim = SimpleHelicopterSimulator::new();
        let cmd = HelicopterCommand::zero();
        let initial_alt = sim.state().altitude();

        for _ in 0..3000 {
            sim.step(&cmd, 1.0 / 300.0);
        }

        let final_alt = sim.state().altitude();
        assert!(
            final_alt < initial_alt,
            "Zero thrust should descend: {final_alt} < {initial_alt}"
        );
    }

    #[test]
    fn test_external_force() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.apply_external_force([1000.0, 0.0, 0.0]); // 1000N lateral push
        sim.step(&HelicopterCommand::hover(), 0.01);

        assert!(
            sim.state().linear_velocity[0] > 0.0,
            "External force should accelerate in X"
        );
    }

    #[test]
    fn test_ground_constraint() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.reset(1.0); // Start at 1m
        let cmd = HelicopterCommand::zero();

        // Run until should hit ground
        for _ in 0..10000 {
            sim.step(&cmd, 0.001);
        }

        assert!(
            sim.state().position[2] >= 0.0,
            "Ground constraint should prevent negative altitude"
        );
    }

    #[test]
    fn test_quaternion_normalized() {
        let mut sim = SimpleHelicopterSimulator::new();
        let cmd = HelicopterCommand {
            collective: 0.3,
            cyclic_lon: 0.5,
            cyclic_lat: -0.3,
            pedal: 0.2,
            thrust: 0.6,
            tail_rotor: 0.5,
        };

        for _ in 0..1000 {
            sim.step(&cmd, 1.0 / 300.0);
        }

        let q = sim.state().quaternion;
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Quaternion should remain normalized: norm={norm}"
        );
    }

    #[test]
    fn test_reset() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.step(&HelicopterCommand::hover(), 0.01);
        sim.reset(50.0);
        assert!((sim.state().altitude() - 50.0).abs() < 1e-10);
        assert_eq!(sim.state().speed(), 0.0);
    }

    #[test]
    fn test_autorotation_descent() {
        let mut sim = SimpleHelicopterSimulator::new();
        // Force low RPM
        sim.rotor.state_mut().main_rpm = 500.0;
        let cmd = HelicopterCommand::zero();
        sim.step(&cmd, 0.01);

        // Should be descending in autorotation
        assert!(sim.rotor.state().in_autorotation);
    }

    #[test]
    fn test_rotor_feedback_in_state() {
        let mut sim = SimpleHelicopterSimulator::new();
        sim.step(&HelicopterCommand::hover(), 0.01);
        assert!(sim.state().main_rotor_rpm > 0.0);
        assert!(sim.state().tail_rotor_rpm > 0.0);
    }

    #[test]
    fn test_steady_wind_pushes_downwind() {
        let mut calm = SimpleHelicopterSimulator::new();
        let mut windy = SimpleHelicopterSimulator::new();
        windy.set_wind(
            WindConfig {
                steady_wind: [10.0, 0.0, 0.0],
                gust_intensity: 0.0, // deterministic: no gusts
                ..WindConfig::default()
            },
            42,
        );

        let cmd = HelicopterCommand::hover();
        for _ in 0..900 {
            calm.step(&cmd, 1.0 / 300.0);
            windy.step(&cmd, 1.0 / 300.0);
        }

        assert!(
            windy.state().linear_velocity[0] > calm.state().linear_velocity[0],
            "Steady +x wind must push the helicopter downwind: windy vx={}, calm vx={}",
            windy.state().linear_velocity[0],
            calm.state().linear_velocity[0]
        );
        assert!(
            windy.state().linear_velocity[0] > 0.0,
            "Downwind drift must be positive: vx={}",
            windy.state().linear_velocity[0]
        );
    }

    #[test]
    fn test_gusty_wind_deterministic_with_seed() {
        let make = || {
            let mut sim = SimpleHelicopterSimulator::new();
            sim.set_wind(WindConfig::moderate_wind(), 7);
            sim
        };
        let mut a = make();
        let mut b = make();
        let cmd = HelicopterCommand::hover();
        for _ in 0..300 {
            a.step(&cmd, 1.0 / 300.0);
            b.step(&cmd, 1.0 / 300.0);
        }
        assert_eq!(a.state().position, b.state().position);
        assert_eq!(a.state().linear_velocity, b.state().linear_velocity);
    }

    #[test]
    fn test_ground_effect_boosts_thrust_near_ground() {
        // Same hover command, low vs high altitude: ground effect must
        // produce a higher climb rate (or slower descent) near the ground.
        let mut low = SimpleHelicopterSimulator::new();
        let mut high = SimpleHelicopterSimulator::new();
        low.reset(3.0);
        high.reset(100.0);

        let cmd = HelicopterCommand::hover();
        for _ in 0..300 {
            low.step(&cmd, 1.0 / 300.0);
            high.step(&cmd, 1.0 / 300.0);
        }

        let low_climb = low.state().altitude() - 3.0;
        let high_climb = high.state().altitude() - 100.0;
        assert!(
            low_climb > high_climb,
            "Ground effect should augment thrust near ground: low_climb={low_climb}, high_climb={high_climb}"
        );
    }
}
