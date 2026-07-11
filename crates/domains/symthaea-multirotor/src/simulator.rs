// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics simulator trait and simple ballistic implementation.
//!
//! Defines a `PhysicsSimulator` trait so that the training loop can run against
//! different physics backends (simple ballistic, MuJoCo, etc.).

use crate::types::{FlightState, QuadrotorCommand};

/// Trait for quadrotor physics simulation backends.
pub trait PhysicsSimulator {
    /// Advance one timestep with the given motor command.
    fn step(&mut self, cmd: &QuadrotorCommand, dt: f64);

    /// Current flight state.
    fn state(&self) -> &FlightState;

    /// Reset to a hover at the given altitude.
    fn reset(&mut self, altitude: f64);

    /// Reset with a deterministic perturbation.
    fn reset_with_perturbation(&mut self, altitude: f64, perturbation: f64, seed: u64);

    /// Apply an external force (world-frame, Newtons) for the next step.
    /// The force is accumulated and cleared after each `step()` call.
    fn apply_external_force(&mut self, force: [f64; 3]);
}

/// Simple physics model for pure-Rust testing.
///
/// Simulates a rigid body with thrust, moments, and aerodynamic drag (no rotor dynamics).
/// Good enough for verifying the multi-rate loop and training convergence.
pub struct SimplePhysicsSimulator {
    state: FlightState,
    mass: f64,
    inertia: [f64; 3],
    external_force: [f64; 3],
    /// Translational drag coefficient (N·s/m). Approximate for Crazyflie-size quad.
    drag_coeff: f64,
    /// Angular damping coefficient (1/s). Used as `exp(-c * dt)` per step.
    angular_damping: f64,
    /// Current effective motor outputs [thrust, roll, pitch, yaw] after lag filter.
    motor_state: [f64; 4],
    /// First-order motor response time constant (seconds). Crazyflie BL motors ≈ 20ms.
    motor_time_constant: f64,
    /// Whether to apply motor lag filter (default: true).
    enable_motor_lag: bool,
}

impl SimplePhysicsSimulator {
    /// Create a new simulator at default hover altitude (0.1m).
    pub fn new() -> Self {
        Self {
            state: FlightState::hover(0.1),
            mass: 0.027,
            inertia: [1.4e-5, 1.4e-5, 2.2e-5],
            external_force: [0.0; 3],
            drag_coeff: 0.01,     // N·s/m — empirical for Crazyflie-size
            angular_damping: 5.0, // 1/s — moderate rotational damping
            motor_state: [QuadrotorCommand::HOVER_THRUST as f64, 0.0, 0.0, 0.0],
            motor_time_constant: 0.02, // 20ms — Crazyflie BL motor response
            enable_motor_lag: true,
        }
    }

    /// Motor response time constant in seconds.
    pub fn motor_time_constant(&self) -> f64 {
        self.motor_time_constant
    }

    /// Thrust required to exactly counteract gravity for this airframe (N).
    /// Safety fallbacks must derive their thrust from this instead of
    /// hardcoding airframe-specific Newton values.
    pub fn hover_thrust(&self) -> f64 {
        self.mass * 9.81
    }

    /// Enable or disable motor lag simulation.
    pub fn set_motor_lag(&mut self, enabled: bool) {
        self.enable_motor_lag = enabled;
    }
}

impl Default for SimplePhysicsSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsSimulator for SimplePhysicsSimulator {
    fn step(&mut self, cmd: &QuadrotorCommand, dt: f64) {
        let g = 9.81;

        // Motor lag: first-order exponential filter
        let effective = if self.enable_motor_lag {
            let alpha = 1.0 - (-dt / self.motor_time_constant).exp();
            let cmd_vals = cmd.to_ctrl();
            for i in 0..4 {
                self.motor_state[i] += alpha * (cmd_vals[i] - self.motor_state[i]);
            }
            QuadrotorCommand {
                thrust: self.motor_state[0] as f32,
                roll_moment: self.motor_state[1] as f32,
                pitch_moment: self.motor_state[2] as f32,
                yaw_moment: self.motor_state[3] as f32,
            }
        } else {
            *cmd
        };

        let thrust = effective.thrust as f64;
        let [w, x, y, z] = self.state.quaternion;

        // Simplified rotation of thrust vector (body z → world)
        let fx = 2.0 * (x * z + w * y) * thrust + self.external_force[0];
        let fy = 2.0 * (y * z - w * x) * thrust + self.external_force[1];
        let fz = (1.0 - 2.0 * (x * x + y * y)) * thrust + self.external_force[2];

        // Linear acceleration
        let ax = fx / self.mass;
        let ay = fy / self.mass;
        let az = fz / self.mass - g;

        // Translational drag: F_drag = -k * v (velocity-proportional)
        let drag = self.drag_coeff / self.mass;
        let ax = ax - drag * self.state.linear_velocity[0];
        let ay = ay - drag * self.state.linear_velocity[1];
        let az = az - drag * self.state.linear_velocity[2];

        // Semi-implicit Euler: velocity from new acceleration, position from new velocity
        self.state.linear_velocity[0] += ax * dt;
        self.state.linear_velocity[1] += ay * dt;
        self.state.linear_velocity[2] += az * dt;

        self.state.position[0] += self.state.linear_velocity[0] * dt;
        self.state.position[1] += self.state.linear_velocity[1] * dt;
        self.state.position[2] += self.state.linear_velocity[2] * dt;

        // Terminal velocity clamp: prevents unrealistic freefall speeds.
        // Drag-limited descent for a ~1.5kg quadrotor is approximately 5 m/s.
        const MAX_DESCENT_RATE: f64 = -5.0;
        if self.state.linear_velocity[2] < MAX_DESCENT_RATE {
            self.state.linear_velocity[2] = MAX_DESCENT_RATE;
        }

        // Ground constraint
        if self.state.position[2] < 0.0 {
            self.state.position[2] = 0.0;
            self.state.linear_velocity[2] = 0.0;
        }

        // Angular acceleration from moments
        let alpha_x = effective.roll_moment as f64 / self.inertia[0];
        let alpha_y = effective.pitch_moment as f64 / self.inertia[1];
        let alpha_z = effective.yaw_moment as f64 / self.inertia[2];

        // Update angular velocity
        self.state.angular_velocity[0] += alpha_x * dt;
        self.state.angular_velocity[1] += alpha_y * dt;
        self.state.angular_velocity[2] += alpha_z * dt;

        // Angular damping: exponential decay, dt-dependent for physical consistency
        let ang_decay = (-self.angular_damping * dt).exp();
        for av in &mut self.state.angular_velocity {
            *av *= ang_decay;
        }

        // Update quaternion from angular velocity
        let [wx, wy, wz] = self.state.angular_velocity;
        let half_dt = dt * 0.5;
        let dw = w - half_dt * (x * wx + y * wy + z * wz);
        let dx = x + half_dt * (w * wx + y * wz - z * wy);
        let dy = y + half_dt * (w * wy + z * wx - x * wz);
        let dz = z + half_dt * (w * wz + x * wy - y * wx);
        self.state.quaternion = normalize_quat([dw, dx, dy, dz]);

        self.state.timestamp += dt;

        // Clear external force after each step
        self.external_force = [0.0; 3];
    }

    fn state(&self) -> &FlightState {
        &self.state
    }

    fn reset(&mut self, altitude: f64) {
        self.state = FlightState::hover(altitude);
        self.external_force = [0.0; 3];
        self.motor_state = [QuadrotorCommand::HOVER_THRUST as f64, 0.0, 0.0, 0.0];
    }

    fn reset_with_perturbation(&mut self, altitude: f64, perturbation: f64, seed: u64) {
        self.state = FlightState::hover(altitude);
        self.external_force = [0.0; 3];
        self.motor_state = [QuadrotorCommand::HOVER_THRUST as f64, 0.0, 0.0, 0.0];

        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        self.state.position[0] += perturbation * next_f64() * 0.1;
        self.state.position[1] += perturbation * next_f64() * 0.1;
        self.state.position[2] += perturbation * next_f64() * 0.05;
        let tilt = perturbation * next_f64() * 0.1;
        self.state.quaternion = normalize_quat([1.0, tilt, tilt * 0.5, 0.0]);
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        self.external_force[0] += force[0];
        self.external_force[1] += force[1];
        self.external_force[2] += force[2];
    }
}

fn normalize_quat(q: [f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-10 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_object_compiles() {
        let sim: Box<dyn PhysicsSimulator> = Box::new(SimplePhysicsSimulator::new());
        assert!(sim.state().altitude() > 0.0);
    }

    #[test]
    fn test_simple_physics_hover() {
        let mut sim = SimplePhysicsSimulator::new();
        let cmd = QuadrotorCommand::hover();
        for _ in 0..100 {
            sim.step(&cmd, 0.002);
        }
        assert!(
            (sim.state().altitude() - 0.1).abs() < 0.05,
            "Hover should maintain altitude: got {}",
            sim.state().altitude()
        );
    }

    #[test]
    fn test_drag_limits_velocity() {
        let mut sim = SimplePhysicsSimulator::new();
        let cmd = QuadrotorCommand::hover();

        // Apply a lateral impulse
        sim.apply_external_force([0.05, 0.0, 0.0]);
        sim.step(&cmd, 0.002);
        let v_after_impulse = sim.state().linear_velocity[0];

        // Run for many steps with no external force — drag should decelerate
        for _ in 0..500 {
            sim.step(&cmd, 0.002);
        }
        let v_after_drag = sim.state().linear_velocity[0];

        assert!(
            v_after_drag.abs() < v_after_impulse.abs(),
            "Drag should decelerate: impulse_v={v_after_impulse:.6}, final_v={v_after_drag:.6}"
        );
    }

    #[test]
    fn test_angular_damping_dt_dependent() {
        // Two sims with different dt should produce similar angular decay over same wall-time.
        let mut sim_fast = SimplePhysicsSimulator::new();
        let mut sim_slow = SimplePhysicsSimulator::new();

        // Give both an angular kick
        sim_fast.state.angular_velocity = [1.0, 0.0, 0.0];
        sim_slow.state.angular_velocity = [1.0, 0.0, 0.0];

        let cmd = QuadrotorCommand::hover();
        let _total_time = 0.1; // 100ms

        // sim_fast: 50 steps at dt=0.002
        for _ in 0..50 {
            sim_fast.step(&cmd, 0.002);
        }
        // sim_slow: 10 steps at dt=0.01
        for _ in 0..10 {
            sim_slow.step(&cmd, 0.01);
        }

        let w_fast = sim_fast.state.angular_velocity[0];
        let w_slow = sim_slow.state.angular_velocity[0];

        // With dt-dependent damping, both should produce similar results
        assert!(
            (w_fast - w_slow).abs() < 0.05,
            "dt-dependent damping should give consistent results: fast={w_fast:.4}, slow={w_slow:.4}"
        );
    }

    #[test]
    fn test_external_force_displaces() {
        let mut sim = SimplePhysicsSimulator::new();
        let cmd = QuadrotorCommand::hover();

        // Apply a strong lateral force
        sim.apply_external_force([0.1, 0.0, 0.0]);
        sim.step(&cmd, 0.002);

        // Force should have displaced the quad
        assert!(sim.state().position[0].abs() > 0.0);
        // Force clears after step
        sim.step(&cmd, 0.002);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut sim = SimplePhysicsSimulator::new();
        sim.apply_external_force([1.0, 1.0, 1.0]);
        sim.reset(0.5);
        assert!((sim.state().altitude() - 0.5).abs() < 1e-10);
        assert!(sim.state().speed() < 1e-10);
    }

    #[test]
    fn test_long_horizon_hover_stability() {
        // With drag, hover command should maintain altitude over 2000 steps (4s).
        // This is the regression test that catches drag-less divergence.
        let mut sim = SimplePhysicsSimulator::new();
        let cmd = QuadrotorCommand::hover();

        for step in 0..2000 {
            sim.step(&cmd, 0.002);

            // Position should remain bounded throughout
            let pos_err = (sim.state().position[0].powi(2)
                + sim.state().position[1].powi(2)
                + (sim.state().position[2] - 0.1).powi(2))
            .sqrt();

            assert!(
                pos_err < 0.10,
                "Position error unbounded at step {step}: {pos_err:.4}m (drag should prevent this)"
            );
        }

        // Final altitude should be close to initial
        assert!(
            (sim.state().altitude() - 0.1).abs() < 0.05,
            "Long-horizon hover failed: final altitude = {:.4}m",
            sim.state().altitude()
        );
    }

    #[test]
    fn test_motor_lag_smooths_step_input() {
        let mut sim = SimplePhysicsSimulator::new();
        assert!(sim.enable_motor_lag);

        // Command full thrust — motor state starts at hover, should ramp up
        let full_thrust = QuadrotorCommand {
            thrust: QuadrotorCommand::MAX_THRUST,
            roll_moment: 0.0,
            pitch_moment: 0.0,
            yaw_moment: 0.0,
        };

        sim.step(&full_thrust, 0.002);
        // After one step, motor_state should be between hover and max (not max yet)
        assert!(
            sim.motor_state[0] > QuadrotorCommand::HOVER_THRUST as f64,
            "Motor should start ramping up"
        );
        assert!(
            sim.motor_state[0] < QuadrotorCommand::MAX_THRUST as f64,
            "Motor should not reach max instantly: {}",
            sim.motor_state[0]
        );
    }

    #[test]
    fn test_motor_lag_disabled_passthrough() {
        let mut sim = SimplePhysicsSimulator::new();
        sim.set_motor_lag(false);

        let full_thrust = QuadrotorCommand {
            thrust: QuadrotorCommand::MAX_THRUST,
            roll_moment: 0.0,
            pitch_moment: 0.0,
            yaw_moment: 0.0,
        };

        sim.step(&full_thrust, 0.002);
        // Without lag, the command should be applied directly
        // (motor_state is NOT updated when lag is disabled)
        // The state should show full thrust effect immediately
        assert!(
            sim.state().altitude() > 0.1,
            "Full thrust without lag should immediately accelerate upward"
        );
    }

    #[test]
    fn test_motor_lag_converges_to_steady_state() {
        let mut sim = SimplePhysicsSimulator::new();

        let target = QuadrotorCommand {
            thrust: 0.4,
            roll_moment: 0.001,
            pitch_moment: 0.0,
            yaw_moment: 0.0,
        };

        // Run for 200ms (100 steps at 2ms) — well beyond 20ms time constant
        for _ in 0..100 {
            sim.step(&target, 0.002);
        }

        // Motor state should converge close to target (within 1%)
        assert!(
            (sim.motor_state[0] - 0.4).abs() < 0.004,
            "Thrust should converge: {}",
            sim.motor_state[0]
        );
        assert!(
            (sim.motor_state[1] - 0.001).abs() < 0.0001,
            "Roll moment should converge: {}",
            sim.motor_state[1]
        );
    }

    #[test]
    fn test_drag_bounds_velocity_after_gust() {
        // After a strong lateral gust, drag should bring velocity back toward zero
        let mut sim = SimplePhysicsSimulator::new();
        let cmd = QuadrotorCommand::hover();

        // Strong lateral gust
        sim.apply_external_force([0.1, 0.0, 0.0]);
        sim.step(&cmd, 0.002);

        // Run for 2 seconds — velocity should decay significantly
        for _ in 0..1000 {
            sim.step(&cmd, 0.002);
        }

        let vx = sim.state().linear_velocity[0];
        assert!(
            vx.abs() < 0.5,
            "Drag should limit lateral velocity after gust: vx={vx:.4}"
        );
    }

    mod proptest_physics {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Arbitrary commands and timesteps must never produce NaN/Inf.
            #[test]
            fn arbitrary_inputs_stay_finite(
                thrust in -1.0f32..2.0,
                roll in -0.1f32..0.1,
                pitch in -0.1f32..0.1,
                yaw in -0.1f32..0.1,
                dt in 0.0001f64..0.05,
                steps in 1usize..200,
            ) {
                let mut sim = SimplePhysicsSimulator::new();
                let cmd = QuadrotorCommand { thrust, roll_moment: roll, pitch_moment: pitch, yaw_moment: yaw };
                for _ in 0..steps {
                    sim.step(&cmd, dt);
                }
                let s = sim.state();
                prop_assert!(s.is_finite(), "State diverged to NaN/Inf: pos={:?} vel={:?}", s.position, s.linear_velocity);
            }

            /// Position must stay non-negative in z (ground constraint).
            #[test]
            fn altitude_never_negative(
                thrust in -1.0f32..2.0,
                dt in 0.001f64..0.02,
            ) {
                let mut sim = SimplePhysicsSimulator::new();
                let cmd = QuadrotorCommand { thrust, roll_moment: 0.0, pitch_moment: 0.0, yaw_moment: 0.0 };
                for _ in 0..500 {
                    sim.step(&cmd, dt);
                }
                prop_assert!(sim.state().position[2] >= 0.0, "Altitude went negative: {}", sim.state().position[2]);
            }

            /// Descent rate must be bounded by MAX_DESCENT_RATE.
            #[test]
            fn descent_rate_bounded(
                thrust in -1.0f32..0.0,
                dt in 0.001f64..0.02,
            ) {
                let mut sim = SimplePhysicsSimulator::new();
                let cmd = QuadrotorCommand { thrust, roll_moment: 0.0, pitch_moment: 0.0, yaw_moment: 0.0 };
                for _ in 0..1000 {
                    sim.step(&cmd, dt);
                }
                prop_assert!(sim.state().linear_velocity[2] >= -5.1, "Descent rate exceeded limit: {}", sim.state().linear_velocity[2]);
            }
        }
    }
}
