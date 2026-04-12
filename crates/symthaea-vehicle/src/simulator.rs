// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics simulator trait and bicycle-model implementation for vehicle dynamics.
//!
//! Defines `VehiclePhysicsSimulator` trait for pluggable backends.
//! The `BicycleModelSimulator` provides a pure-Rust single-track (bicycle) model with:
//! - Pacejka-inspired tire lateral force (linear + saturation)
//! - Longitudinal dynamics (engine, braking, aerodynamic drag, rolling resistance)
//! - Weight transfer (longitudinal → pitch, affects tire grip distribution)
//! - Rate-limited steering actuator
//! - Friction scaling for perturbation (black ice, rain)
//!
//! The bicycle model is the standard for vehicle dynamics control research.
//! It collapses the 4-wheel system into a 2-axle system, accurate up to ~0.5g
//! lateral acceleration — sufficient for highway autonomy and the training crucible.

use crate::types::{ChassisParams, VehicleCommand, VehicleState};

/// Trait for vehicle physics simulation backends.
pub trait VehiclePhysicsSimulator {
    /// Advance one timestep with the given motor command.
    fn step(&mut self, cmd: &VehicleCommand, dt: f64);

    /// Current vehicle state.
    fn state(&self) -> &VehicleState;

    /// Reset to default state (stopped at origin).
    fn reset(&mut self);

    /// Reset with a deterministic perturbation (randomized initial state).
    fn reset_with_perturbation(&mut self, perturbation: f64, seed: u64);

    /// Apply an external force to the vehicle (world-frame, Newtons) for the next step.
    /// [longitudinal, lateral] in body frame.
    fn apply_external_force(&mut self, force: [f64; 2]);

    /// Set the friction scale (1.0 = dry, 0.0 = pure ice).
    fn set_friction_scale(&mut self, scale: f64);

    /// Get current friction scale.
    fn friction_scale(&self) -> f64;

    /// Set per-axle friction scales for asymmetric grip (tire blowout).
    /// Multiplied with the global friction_scale: effective = global * axle.
    fn set_axle_friction(&mut self, front: f64, rear: f64);
}

/// Bicycle-model (single-track) vehicle simulator.
///
/// The bicycle model is the workhorse of vehicle dynamics control. It treats
/// the vehicle as having two "tires": one at the front axle and one at the rear.
/// This captures the essential lateral dynamics (understeer/oversteer, yaw stability)
/// while remaining computationally trivial — perfect for a 6-watt mind.
///
/// ## Dynamics
///
/// Lateral:
///   m * (dvy/dt + V * r) = Fyf + Fyr
///   Iz * dr/dt = a * Fyf - b * Fyr
///
/// Where:
///   αf = δ - atan2(vy + a*r, V)   (front slip angle)
///   αr = -atan2(vy - b*r, V)       (rear slip angle)
///   Fyf = -Cf * f(αf)              (front lateral force)
///   Fyr = -Cr * f(αr)              (rear lateral force)
///   f(α) = α * (1 - |α|/α_max)    (simplified Pacejka saturation)
///
/// Longitudinal:
///   m * dV/dt = Fx - Fdrag - Frolling + Fy_coupling
///   Fx = throttle * max_engine_force - brake * max_brake_force
///   Fdrag = 0.5 * CdA * ρ * V²
///   Frolling = Cr * m * g
pub struct BicycleModelSimulator {
    state: VehicleState,
    params: ChassisParams,
    external_force: [f64; 2],
    friction_scale: f64,
    /// Per-axle friction multiplier (1.0 = normal, <1.0 = blowout/degradation).
    front_friction_scale: f64,
    rear_friction_scale: f64,
    /// Initial state for resetting (set by the task/scenario).
    initial_speed: f64,
}

/// Maximum tire slip angle before full saturation (rad). ~15 degrees.
const ALPHA_MAX: f64 = 0.26;

/// Gravitational acceleration (m/s²).
const G: f64 = 9.81;

/// Minimum speed for lateral dynamics to avoid singularity (m/s).
const V_MIN: f64 = 0.5;

impl BicycleModelSimulator {
    /// Create a new simulator with default chassis params, starting from rest.
    pub fn new() -> Self {
        Self {
            state: VehicleState::stopped(),
            params: ChassisParams::default(),
            external_force: [0.0; 2],
            friction_scale: 1.0,
            front_friction_scale: 1.0,
            rear_friction_scale: 1.0,
            initial_speed: 0.0,
        }
    }

    /// Create a simulator with custom chassis parameters.
    pub fn with_params(params: ChassisParams) -> Self {
        Self {
            state: VehicleState::stopped(),
            params,
            external_force: [0.0; 2],
            friction_scale: 1.0,
            front_friction_scale: 1.0,
            rear_friction_scale: 1.0,
            initial_speed: 0.0,
        }
    }

    /// Create a simulator starting at a given speed (m/s).
    pub fn at_speed(speed: f64) -> Self {
        Self {
            state: VehicleState::cruising(speed),
            params: ChassisParams::default(),
            external_force: [0.0; 2],
            friction_scale: 1.0,
            front_friction_scale: 1.0,
            rear_friction_scale: 1.0,
            initial_speed: speed,
        }
    }

    /// Get reference to chassis parameters.
    pub fn params(&self) -> &ChassisParams {
        &self.params
    }

    /// Set the vehicle's world-frame position and heading directly.
    pub fn set_position(&mut self, x: f64, y: f64, heading: f64) {
        self.state.position_x = x;
        self.state.position_y = y;
        self.state.heading = heading;
    }

    /// Simplified Pacejka tire force: linear region with saturation.
    ///
    /// f(α) = Cs * α * (1 - |α| / α_max) for |α| < α_max
    /// f(α) = Cs * α_max * 0.25 * sign(α) for |α| >= α_max (saturated)
    ///
    /// This captures the essential nonlinearity: grip increases linearly with
    /// slip angle, peaks near α_max, then decays. The saturation plateau prevents
    /// force from going to zero (which would be numerically catastrophic).
    fn tire_lateral_force(cornering_stiffness: f64, slip_angle: f64, friction: f64) -> f64 {
        let cs = cornering_stiffness * friction;
        let alpha = slip_angle;
        let alpha_abs = alpha.abs();

        if alpha_abs < ALPHA_MAX {
            // Linear region with gradual saturation
            -cs * alpha * (1.0 - alpha_abs / ALPHA_MAX * 0.5)
        } else {
            // Saturated — peak force with sign
            -cs * ALPHA_MAX * 0.5 * alpha.signum()
        }
    }

    /// Compute weight transfer from longitudinal acceleration.
    /// Returns (front_load_fraction, rear_load_fraction).
    fn weight_transfer(&self, ax: f64) -> (f64, f64) {
        let p = &self.params;
        let base_front = p.rear_axle_dist / p.wheelbase;
        let base_rear = p.front_axle_dist / p.wheelbase;

        // Weight transfer: ΔFz = m * ax * h / L
        let transfer = p.mass * ax * p.cg_height / p.wheelbase;
        let total_weight = p.mass * G;

        let front_load = (base_front * total_weight - transfer) / total_weight;
        let rear_load = (base_rear * total_weight + transfer) / total_weight;

        (front_load.max(0.05), rear_load.max(0.05))
    }
}

impl Default for BicycleModelSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl VehiclePhysicsSimulator for BicycleModelSimulator {
    fn step(&mut self, cmd: &VehicleCommand, dt: f64) {
        let p = &self.params;
        let cmd = cmd.clamped();

        // 1. Steering actuator: rate-limited approach to commanded angle
        let target_steer = cmd.steering as f64 * p.max_steering_angle;
        let steer_error = target_steer - self.state.steering_angle;
        let max_steer_change = p.steering_rate_limit * dt;
        self.state.steering_angle += steer_error.clamp(-max_steer_change, max_steer_change);

        // 2. Longitudinal forces
        let v = self.state.speed.max(0.0);

        // Engine force: power-limited at high speed.
        // F = min(max_engine_force, max_power / max(v, 1.0))
        let power_limited_force = p.max_power / v.max(1.0);
        let engine_force = cmd.throttle as f64 * power_limited_force.min(p.max_engine_force);

        let brake_force = cmd.brake as f64 * p.max_brake_force * self.friction_scale;
        let drag_force = 0.5 * p.drag_coeff * p.air_density * v * v;

        // Rolling resistance only applies when moving (not when parked)
        let rolling_force = if v > 0.01 {
            p.rolling_resistance * p.mass * G
        } else {
            0.0
        };

        let fx_total =
            engine_force - brake_force - drag_force - rolling_force + self.external_force[0];

        let ax = fx_total / p.mass;
        self.state.longitudinal_accel = ax;

        // 3. Weight transfer for tire grip modulation
        let (front_load, rear_load) = self.weight_transfer(ax);

        // 4. Lateral dynamics (only active above V_MIN to avoid singularity)
        let delta = self.state.steering_angle;
        let vy = self.state.lateral_velocity;
        let r = self.state.yaw_rate;

        if v > V_MIN {
            // Tire slip angles
            let alpha_f = delta - ((vy + p.front_axle_dist * r) / v).atan();
            let alpha_r = -((vy - p.rear_axle_dist * r) / v).atan();

            self.state.tire_slip_front = alpha_f.abs();
            self.state.tire_slip_rear = alpha_r.abs();

            // Lateral tire forces with weight transfer
            let fyf = Self::tire_lateral_force(
                p.front_cornering_stiffness,
                alpha_f,
                self.friction_scale * self.front_friction_scale * front_load,
            );
            let fyr = Self::tire_lateral_force(
                p.rear_cornering_stiffness,
                alpha_r,
                self.friction_scale * self.rear_friction_scale * rear_load,
            );

            // Lateral acceleration: Fy_total / m - V * r (body frame)
            let ay = (fyf + fyr + self.external_force[1]) / p.mass;
            self.state.lateral_accel = ay;

            // State derivatives
            let dvy = ay - v * r;
            let dr = (p.front_axle_dist * fyf - p.rear_axle_dist * fyr) / p.yaw_inertia;

            self.state.lateral_velocity += dvy * dt;
            self.state.yaw_rate += dr * dt;
        } else {
            // Low-speed kinematic model: yaw rate from steering geometry
            if delta.abs() > 1e-6 {
                self.state.yaw_rate = v * delta.tan() / p.wheelbase;
            } else {
                self.state.yaw_rate *= (1.0 - 5.0 * dt).max(0.0); // decay
            }
            self.state.lateral_velocity *= (1.0 - 10.0 * dt).max(0.0); // decay
            self.state.lateral_accel = 0.0;
            self.state.tire_slip_front = 0.0;
            self.state.tire_slip_rear = 0.0;
        }

        // 5. Integrate speed (prevent reversing from braking)
        self.state.speed += ax * dt;
        if self.state.speed < 0.0 && cmd.brake > 0.0 {
            self.state.speed = 0.0;
            self.state.lateral_velocity = 0.0;
            self.state.yaw_rate = 0.0;
        }

        // 6. Yaw rate damping (tire aligning torque)
        self.state.yaw_rate *= (1.0 - 0.5 * dt).max(0.0);

        // 7. Pitch from longitudinal weight transfer
        // Simplified: pitch ∝ -ax * h / (wheelbase * g)
        let target_pitch = -ax * p.cg_height / (p.wheelbase * G);
        self.state.pitch = 0.8 * self.state.pitch + 0.2 * target_pitch;

        // 8. Integrate heading and position (world frame)
        self.state.heading += self.state.yaw_rate * dt;
        // Normalize heading to [-pi, pi]
        while self.state.heading > std::f64::consts::PI {
            self.state.heading -= 2.0 * std::f64::consts::PI;
        }
        while self.state.heading < -std::f64::consts::PI {
            self.state.heading += 2.0 * std::f64::consts::PI;
        }

        let cos_h = self.state.heading.cos();
        let sin_h = self.state.heading.sin();
        self.state.position_x += (v * cos_h - self.state.lateral_velocity * sin_h) * dt;
        self.state.position_y += (v * sin_h + self.state.lateral_velocity * cos_h) * dt;

        // 9. Update throttle/brake feedback
        self.state.throttle_position = cmd.throttle as f64;
        self.state.brake_pressure = cmd.brake as f64;

        self.state.timestamp += dt;
        self.external_force = [0.0; 2];
    }

    fn state(&self) -> &VehicleState {
        &self.state
    }

    fn reset(&mut self) {
        self.state = if self.initial_speed > 0.0 {
            VehicleState::cruising(self.initial_speed)
        } else {
            VehicleState::stopped()
        };
        self.external_force = [0.0; 2];
        self.friction_scale = 1.0;
        self.front_friction_scale = 1.0;
        self.rear_friction_scale = 1.0;
    }

    fn reset_with_perturbation(&mut self, perturbation: f64, seed: u64) {
        self.reset();

        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        // Perturb initial speed (±10% of initial)
        let speed_base = self.initial_speed.max(5.0);
        self.state.speed += perturbation * next_f64() * speed_base * 0.1;
        self.state.speed = self.state.speed.max(0.0);

        // Perturb heading (±5 degrees)
        self.state.heading += perturbation * next_f64() * 0.087;

        // Perturb lateral velocity (±0.5 m/s)
        self.state.lateral_velocity += perturbation * next_f64() * 0.5;

        // Perturb yaw rate (±0.1 rad/s)
        self.state.yaw_rate += perturbation * next_f64() * 0.1;

        // Perturb lateral position (±0.5m)
        self.state.position_y += perturbation * next_f64() * 0.5;
    }

    fn apply_external_force(&mut self, force: [f64; 2]) {
        self.external_force[0] += force[0];
        self.external_force[1] += force[1];
    }

    fn set_friction_scale(&mut self, scale: f64) {
        self.friction_scale = scale.clamp(0.0, 2.0);
    }

    fn friction_scale(&self) -> f64 {
        self.friction_scale
    }

    fn set_axle_friction(&mut self, front: f64, rear: f64) {
        self.front_friction_scale = front.clamp(0.0, 2.0);
        self.rear_friction_scale = rear.clamp(0.0, 2.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_object_compiles() {
        let sim: Box<dyn VehiclePhysicsSimulator> = Box::new(BicycleModelSimulator::new());
        assert!(sim.state().speed >= 0.0);
    }

    #[test]
    fn test_stopped_stays_stopped() {
        let mut sim = BicycleModelSimulator::new();
        let cmd = VehicleCommand::zero();
        for _ in 0..100 {
            sim.step(&cmd, 0.005);
        }
        assert!(
            sim.state().speed.abs() < 0.01,
            "Zero command from rest should stay near zero: {}",
            sim.state().speed
        );
    }

    #[test]
    fn test_throttle_accelerates() {
        let mut sim = BicycleModelSimulator::new();
        let cmd = VehicleCommand {
            steering: 0.0,
            throttle: 0.5,
            brake: 0.0,
        };
        for _ in 0..200 {
            sim.step(&cmd, 0.005);
        }
        assert!(
            sim.state().speed > 1.0,
            "Throttle should produce speed: {}",
            sim.state().speed
        );
    }

    #[test]
    fn test_brake_decelerates() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        let cmd = VehicleCommand {
            steering: 0.0,
            throttle: 0.0,
            brake: 1.0,
        };
        for _ in 0..400 {
            sim.step(&cmd, 0.005);
        }
        assert!(
            sim.state().speed < 5.0,
            "Full brake should decelerate: {}",
            sim.state().speed
        );
    }

    #[test]
    fn test_brake_does_not_reverse() {
        let mut sim = BicycleModelSimulator::at_speed(2.0);
        let cmd = VehicleCommand {
            steering: 0.0,
            throttle: 0.0,
            brake: 1.0,
        };
        for _ in 0..1000 {
            sim.step(&cmd, 0.005);
        }
        assert!(
            sim.state().speed >= 0.0,
            "Braking should not reverse: {}",
            sim.state().speed
        );
    }

    #[test]
    fn test_steering_changes_heading() {
        let mut sim = BicycleModelSimulator::at_speed(10.0);
        let cmd = VehicleCommand {
            steering: 0.5,
            throttle: 0.3,
            brake: 0.0,
        };
        for _ in 0..200 {
            sim.step(&cmd, 0.005);
        }
        assert!(
            sim.state().heading.abs() > 0.01,
            "Steering should change heading: {}",
            sim.state().heading
        );
        assert!(
            sim.state().yaw_rate.abs() > 0.01,
            "Should have non-zero yaw rate: {}",
            sim.state().yaw_rate
        );
    }

    #[test]
    fn test_drag_limits_top_speed() {
        let mut sim = BicycleModelSimulator::new();
        let cmd = VehicleCommand {
            steering: 0.0,
            throttle: 1.0,
            brake: 0.0,
        };
        for _ in 0..20000 {
            sim.step(&cmd, 0.005);
        }
        let speed = sim.state().speed;
        assert!(speed < 60.0, "Drag should limit top speed: {speed} m/s");
        // Should reach a reasonable top speed
        assert!(speed > 20.0, "Should reach meaningful speed: {speed} m/s");
    }

    #[test]
    fn test_external_force_affects_state() {
        let mut sim = BicycleModelSimulator::at_speed(10.0);
        sim.apply_external_force([0.0, 5000.0]); // lateral push
        let cmd = VehicleCommand::zero();
        sim.step(&cmd, 0.005);
        // Should induce lateral velocity or yaw
        let lat = sim.state().lateral_velocity.abs();
        let yaw = sim.state().yaw_rate.abs();
        assert!(
            lat > 0.0 || yaw > 0.0,
            "External force should affect dynamics: lat={lat}, yaw={yaw}"
        );
    }

    #[test]
    fn test_friction_scale() {
        let mut sim = BicycleModelSimulator::at_speed(10.0);
        sim.set_friction_scale(0.1); // ice
        assert!((sim.friction_scale() - 0.1).abs() < 1e-10);

        // Steer on ice — should produce larger slip
        let cmd = VehicleCommand {
            steering: 1.0,
            throttle: 0.3,
            brake: 0.0,
        };
        for _ in 0..100 {
            sim.step(&cmd, 0.005);
        }
        let slip = sim.state().tire_slip_front;
        assert!(slip > 0.0, "Should have tire slip on ice: {slip}");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        let cmd = VehicleCommand {
            steering: 0.5,
            throttle: 1.0,
            brake: 0.0,
        };
        for _ in 0..500 {
            sim.step(&cmd, 0.005);
        }

        sim.reset();
        assert!((sim.state().speed - 13.4).abs() < 0.1);
        assert!(sim.state().heading.abs() < 1e-10);
        assert!(sim.state().position_x.abs() < 1e-10);
    }

    #[test]
    fn test_perturbation_changes_state() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        sim.reset_with_perturbation(1.0, 42);
        assert!(
            (sim.state().speed - 13.4).abs() > 0.01
                || sim.state().heading.abs() > 0.001
                || sim.state().lateral_velocity.abs() > 0.001,
            "Perturbation should change state"
        );
    }

    #[test]
    fn test_state_stays_finite() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        let cmds = [
            VehicleCommand {
                steering: 1.0,
                throttle: 1.0,
                brake: 0.0,
            },
            VehicleCommand {
                steering: -1.0,
                throttle: 0.0,
                brake: 1.0,
            },
            VehicleCommand {
                steering: 0.5,
                throttle: 0.5,
                brake: 0.5,
            },
        ];
        for step in 0..3000 {
            sim.step(&cmds[step % cmds.len()], 0.005);
            assert!(
                sim.state().is_finite(),
                "State became non-finite at step {step}: {:?}",
                sim.state()
            );
        }
    }

    #[test]
    fn test_weight_transfer_under_braking() {
        let mut sim = BicycleModelSimulator::at_speed(20.0);
        let cmd = VehicleCommand {
            steering: 0.0,
            throttle: 0.0,
            brake: 0.8,
        };
        sim.step(&cmd, 0.005);
        assert!(
            sim.state().pitch > 0.0,
            "Braking should produce nose-down pitch (positive): {}",
            sim.state().pitch
        );
    }

    #[test]
    fn test_position_advances_with_speed() {
        let mut sim = BicycleModelSimulator::at_speed(10.0);
        let cmd = VehicleCommand {
            steering: 0.0,
            throttle: 0.3,
            brake: 0.0,
        };
        for _ in 0..200 {
            sim.step(&cmd, 0.005);
        }
        assert!(
            sim.state().position_x > 5.0,
            "Position should advance: x={}",
            sim.state().position_x
        );
    }

    #[test]
    fn test_heading_normalizes() {
        let mut sim = BicycleModelSimulator::at_speed(10.0);
        let cmd = VehicleCommand {
            steering: 1.0,
            throttle: 0.3,
            brake: 0.0,
        };
        for _ in 0..5000 {
            sim.step(&cmd, 0.005);
        }
        assert!(
            sim.state().heading.abs() <= std::f64::consts::PI + 0.01,
            "Heading should stay normalized: {}",
            sim.state().heading
        );
    }

    #[test]
    fn test_axle_friction_front_blowout_induces_yaw() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        sim.set_axle_friction(0.3, 1.0); // front blowout: front grip drops to 30%

        let cmd = VehicleCommand {
            steering: 0.3,
            throttle: 0.3,
            brake: 0.0,
        };
        for _ in 0..200 {
            sim.step(&cmd, 0.005);
        }

        let state = sim.state();
        assert!(
            state.is_finite(),
            "State should remain finite under blowout"
        );

        // Compare with normal handling
        let mut sim_normal = BicycleModelSimulator::at_speed(13.4);
        let cmd_n = cmd;
        for _ in 0..200 {
            sim_normal.step(&cmd_n, 0.005);
        }

        assert!(
            (state.heading - sim_normal.state().heading).abs() > 0.001,
            "Blowout should change handling: blowout heading={}, normal={}",
            state.heading,
            sim_normal.state().heading
        );
    }

    #[test]
    fn test_axle_friction_rear_blowout_oversteer() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        sim.set_axle_friction(1.0, 0.3); // rear blowout

        let cmd = VehicleCommand {
            steering: 0.3,
            throttle: 0.3,
            brake: 0.0,
        };
        for _ in 0..200 {
            sim.step(&cmd, 0.005);
        }

        assert!(
            sim.state().is_finite(),
            "State should remain finite under rear blowout"
        );
        assert!(
            sim.state().yaw_rate.abs() > 0.0 || sim.state().heading.abs() > 0.001,
            "Rear blowout should affect dynamics"
        );
    }

    #[test]
    fn test_axle_friction_resets() {
        let mut sim = BicycleModelSimulator::at_speed(13.4);
        sim.set_axle_friction(0.3, 0.5);
        sim.reset();
        assert!((sim.friction_scale() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_steering_rate_limited() {
        let mut sim = BicycleModelSimulator::at_speed(10.0);
        let cmd = VehicleCommand {
            steering: 1.0, // full lock
            throttle: 0.3,
            brake: 0.0,
        };
        sim.step(&cmd, 0.005);
        let max_change = sim.params().steering_rate_limit * 0.005;
        assert!(
            sim.state().steering_angle.abs() <= max_change + 1e-6,
            "Steering should be rate-limited: {} vs max {}",
            sim.state().steering_angle,
            max_change
        );
    }

    mod proptest_physics {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Arbitrary commands and timesteps must never produce NaN/Inf.
            #[test]
            fn arbitrary_inputs_stay_finite(
                steering in -1.0f32..1.0,
                throttle in 0.0f32..1.0,
                brake in 0.0f32..1.0,
                dt in 0.001f64..0.02,
                steps in 1usize..500,
            ) {
                let mut sim = BicycleModelSimulator::at_speed(10.0);
                let cmd = VehicleCommand { steering, throttle, brake };
                for _ in 0..steps {
                    sim.step(&cmd, dt);
                }
                prop_assert!(sim.state().is_finite(), "Vehicle state diverged: speed={}", sim.state().speed);
            }

            /// Speed must never go negative (vehicle doesn't reverse from braking).
            #[test]
            fn speed_never_negative(
                brake in 0.5f32..1.0,
                dt in 0.001f64..0.02,
            ) {
                let mut sim = BicycleModelSimulator::at_speed(5.0);
                let cmd = VehicleCommand { steering: 0.0, throttle: 0.0, brake };
                for _ in 0..1000 {
                    sim.step(&cmd, dt);
                }
                prop_assert!(sim.state().speed >= 0.0, "Speed went negative: {}", sim.state().speed);
            }

            /// Friction scaling must not cause divergence.
            #[test]
            fn friction_scaling_stays_finite(
                friction in 0.01f64..2.0,
                steering in -1.0f32..1.0,
                throttle in 0.0f32..1.0,
            ) {
                let mut sim = BicycleModelSimulator::at_speed(15.0);
                sim.set_friction_scale(friction);
                let cmd = VehicleCommand { steering, throttle, brake: 0.0 };
                for _ in 0..200 {
                    sim.step(&cmd, 0.005);
                }
                prop_assert!(sim.state().is_finite(), "Diverged at friction={friction}: speed={}", sim.state().speed);
            }
        }
    }
}
