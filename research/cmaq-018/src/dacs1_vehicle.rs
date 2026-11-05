//! DACS-1 analytic vehicle/actuator reference profile.
//!
//! This is a deterministic, forward-only, flat-road kinematic bicycle model
//! with first-order throttle/brake/steering actuators and an explicit lateral
//! acceleration envelope. All quantities are SI. It is a conformance reference,
//! not a validated tire model, simulator, or production vehicle controller.

use std::f64::consts::{PI, TAU};

const GRAVITY_MPS2: f64 = 9.806_65;
const STATE_EPSILON: f64 = 1.0e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleState {
    pub x_m: f64,
    pub y_m: f64,
    pub yaw_rad: f64,
    pub speed_mps: f64,
}

impl VehicleState {
    pub fn validate(self) -> Result<(), VehicleError> {
        if !self.x_m.is_finite()
            || !self.y_m.is_finite()
            || !self.yaw_rad.is_finite()
            || !self.speed_mps.is_finite()
            || self.speed_mps < 0.0
        {
            return Err(VehicleError::InvalidState);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActuatorCommand {
    /// Normalized accelerator request in [0, 1].
    pub throttle: f64,
    /// Normalized service-brake request in [0, 1].
    pub brake: f64,
    /// Normalized steering-wheel request in [-1, 1].
    pub steering: f64,
}

impl ActuatorCommand {
    pub fn validate(self) -> Result<(), VehicleError> {
        if !self.throttle.is_finite()
            || !self.brake.is_finite()
            || !self.steering.is_finite()
            || !(0.0..=1.0).contains(&self.throttle)
            || !(0.0..=1.0).contains(&self.brake)
            || !(-1.0..=1.0).contains(&self.steering)
        {
            return Err(VehicleError::InvalidCommand);
        }
        if self.throttle > 0.0 && self.brake > 0.0 {
            return Err(VehicleError::ConflictingPedals);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActuatorState {
    pub throttle: f64,
    pub brake: f64,
    pub steering_angle_rad: f64,
}

impl Default for ActuatorState {
    fn default() -> Self {
        Self {
            throttle: 0.0,
            brake: 0.0,
            steering_angle_rad: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleConfig {
    pub wheelbase_m: f64,
    pub max_steering_angle_rad: f64,
    pub max_steering_rate_rad_s: f64,
    pub max_throttle_accel_mps2: f64,
    pub max_brake_decel_mps2: f64,
    pub rolling_resistance_mps2: f64,
    /// Aerodynamic deceleration coefficient in 1/m: a_drag = -c * v * |v|.
    pub aerodynamic_drag_coeff_per_m: f64,
    pub max_lateral_accel_mps2: f64,
    pub tire_friction_coefficient: f64,
    pub throttle_time_constant_s: f64,
    pub brake_time_constant_s: f64,
    pub steering_time_constant_s: f64,
}

impl Default for VehicleConfig {
    fn default() -> Self {
        Self {
            wheelbase_m: 2.8,
            max_steering_angle_rad: 0.55,
            max_steering_rate_rad_s: 1.2,
            max_throttle_accel_mps2: 3.5,
            max_brake_decel_mps2: 8.0,
            rolling_resistance_mps2: 0.15,
            aerodynamic_drag_coeff_per_m: 0.002,
            max_lateral_accel_mps2: 8.0,
            tire_friction_coefficient: 0.9,
            throttle_time_constant_s: 0.25,
            brake_time_constant_s: 0.10,
            steering_time_constant_s: 0.15,
        }
    }
}

impl VehicleConfig {
    pub fn validate(self) -> Result<(), VehicleError> {
        let positive = [
            self.wheelbase_m,
            self.max_steering_angle_rad,
            self.max_steering_rate_rad_s,
            self.max_throttle_accel_mps2,
            self.max_brake_decel_mps2,
            self.max_lateral_accel_mps2,
            self.tire_friction_coefficient,
            self.throttle_time_constant_s,
            self.brake_time_constant_s,
            self.steering_time_constant_s,
        ];
        if positive.iter().any(|value| !value.is_finite() || *value <= 0.0)
            || !self.rolling_resistance_mps2.is_finite()
            || self.rolling_resistance_mps2 < 0.0
            || !self.aerodynamic_drag_coeff_per_m.is_finite()
            || self.aerodynamic_drag_coeff_per_m < 0.0
            || self.max_steering_angle_rad >= PI / 2.0
        {
            return Err(VehicleError::InvalidConfiguration);
        }
        Ok(())
    }

    #[must_use]
    pub fn lateral_accel_limit_mps2(self) -> f64 {
        self.max_lateral_accel_mps2.min(self.tire_friction_coefficient * GRAVITY_MPS2)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOutcome {
    pub state: VehicleState,
    pub actuators: ActuatorState,
    pub longitudinal_accel_mps2: f64,
    pub lateral_accel_mps2: f64,
    pub yaw_rate_rps: f64,
    pub lateral_saturated: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VehicleError {
    InvalidConfiguration,
    InvalidState,
    InvalidCommand,
    ConflictingPedals,
    InvalidTimeStep,
}

#[derive(Clone, Copy, Debug)]
pub struct AnalyticVehicle {
    config: VehicleConfig,
}

impl AnalyticVehicle {
    pub fn new(config: VehicleConfig) -> Result<Self, VehicleError> {
        config.validate()?;
        Ok(Self { config })
    }

    #[must_use]
    pub fn config(self) -> VehicleConfig {
        self.config
    }

    /// Apply one actuator command for dt seconds.
    ///
    /// The profile intentionally has no reverse gear, road grade, bank angle,
    /// load transfer, tire relaxation, or lateral slip state. Braking cannot
    /// reverse the vehicle; a separate reverse-gear contract would be required.
    pub fn step(
        self,
        state: VehicleState,
        actuators: ActuatorState,
        command: ActuatorCommand,
        dt_s: f64,
    ) -> Result<StepOutcome, VehicleError> {
        state.validate()?;
        command.validate()?;
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(VehicleError::InvalidTimeStep);
        }
        if !actuators.throttle.is_finite()
            || !actuators.brake.is_finite()
            || !actuators.steering_angle_rad.is_finite()
            || !(0.0..=1.0).contains(&actuators.throttle)
            || !(0.0..=1.0).contains(&actuators.brake)
        {
            return Err(VehicleError::InvalidState);
        }

        let throttle = first_order(
            actuators.throttle,
            command.throttle,
            dt_s,
            self.config.throttle_time_constant_s,
        );
        let brake = first_order(
            actuators.brake,
            command.brake,
            dt_s,
            self.config.brake_time_constant_s,
        );

        let target_steering =
            command.steering * self.config.max_steering_angle_rad;
        let lagged_steering = first_order(
            actuators.steering_angle_rad,
            target_steering,
            dt_s,
            self.config.steering_time_constant_s,
        );
        let steering_delta = (lagged_steering - actuators.steering_angle_rad)
            .clamp(
                -self.config.max_steering_rate_rad_s * dt_s,
                self.config.max_steering_rate_rad_s * dt_s,
            );
        let steering_angle_rad = actuators.steering_angle_rad + steering_delta;

        let rolling = if state.speed_mps > STATE_EPSILON {
            self.config.rolling_resistance_mps2
        } else {
            0.0
        };
        let aero = self.config.aerodynamic_drag_coeff_per_m
            * state.speed_mps
            * state.speed_mps;
        let requested_longitudinal_accel = self.config.max_throttle_accel_mps2 * throttle
            - self.config.max_brake_decel_mps2 * brake
            - rolling
            - aero;
        let next_speed = (state.speed_mps + requested_longitudinal_accel * dt_s).max(0.0);
        let longitudinal_accel_mps2 = (next_speed - state.speed_mps) / dt_s;

        let raw_yaw_rate = if next_speed <= STATE_EPSILON {
            0.0
        } else {
            next_speed / self.config.wheelbase_m * steering_angle_rad.tan()
        };
        let lateral_limit = self.config.lateral_accel_limit_mps2();
        let raw_lateral_accel = next_speed * raw_yaw_rate;
        let lateral_saturated = raw_lateral_accel.abs() > lateral_limit;
        let yaw_rate_rps = if lateral_saturated {
            raw_lateral_accel.signum() * lateral_limit / next_speed
        } else {
            raw_yaw_rate
        };
        let lateral_accel_mps2 = next_speed * yaw_rate_rps;

        let mid_yaw = state.yaw_rad + 0.5 * yaw_rate_rps * dt_s;
        let next_state = VehicleState {
            x_m: state.x_m + next_speed * mid_yaw.cos() * dt_s,
            y_m: state.y_m + next_speed * mid_yaw.sin() * dt_s,
            yaw_rad: wrap_angle(state.yaw_rad + yaw_rate_rps * dt_s),
            speed_mps: next_speed,
        };

        Ok(StepOutcome {
            state: next_state,
            actuators: ActuatorState {
                throttle,
                brake,
                steering_angle_rad,
            },
            longitudinal_accel_mps2,
            lateral_accel_mps2,
            yaw_rate_rps,
            lateral_saturated,
        })
    }
}

fn first_order(current: f64, target: f64, dt_s: f64, time_constant_s: f64) -> f64 {
    let alpha = dt_s / (time_constant_s + dt_s);
    current + alpha * (target - current)
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vehicle(config: VehicleConfig) -> AnalyticVehicle {
        AnalyticVehicle::new(config).expect("valid analytic vehicle config")
    }

    fn state(speed_mps: f64) -> VehicleState {
        VehicleState {
            x_m: 0.0,
            y_m: 0.0,
            yaw_rad: 0.0,
            speed_mps,
        }
    }

    fn command(throttle: f64, brake: f64, steering: f64) -> ActuatorCommand {
        ActuatorCommand {
            throttle,
            brake,
            steering,
        }
    }

    #[test]
    fn neutral_input_coasts_down_without_instant_stopping() {
        let model = vehicle(VehicleConfig::default());
        let result = model
            .step(state(12.0), ActuatorState::default(), command(0.0, 0.0, 0.0), 0.5)
            .unwrap();

        assert!(result.state.speed_mps < 12.0);
        assert!(result.state.speed_mps > 0.0);
        assert!(result.state.x_m > 0.0);
        assert_eq!(result.yaw_rate_rps, 0.0);
    }

    #[test]
    fn braking_decelerates_and_never_invents_reverse_motion() {
        let model = vehicle(VehicleConfig::default());
        let mut current_state = state(1.0);
        let mut actuators = ActuatorState::default();
        for _ in 0..30 {
            let result = model
                .step(current_state, actuators, command(0.0, 1.0, 0.0), 0.1)
                .unwrap();
            current_state = result.state;
            actuators = result.actuators;
        }
        assert_eq!(current_state.speed_mps, 0.0);
    }

    #[test]
    fn throttle_has_declared_actuator_lag() {
        let model = vehicle(VehicleConfig::default());
        let result = model
            .step(state(10.0), ActuatorState::default(), command(1.0, 0.0, 0.0), 0.1)
            .unwrap();

        assert!(result.state.speed_mps > 10.0);
        assert!(result.actuators.throttle < 1.0);
        assert!(result.longitudinal_accel_mps2 < VehicleConfig::default().max_throttle_accel_mps2);
    }

    #[test]
    fn steering_at_speed_changes_heading_with_correct_sign() {
        let model = vehicle(VehicleConfig::default());
        let result = model
            .step(state(15.0), ActuatorState::default(), command(0.0, 0.0, 0.8), 0.25)
            .unwrap();

        assert!(result.state.yaw_rad > 0.0);
        assert!(result.yaw_rate_rps > 0.0);
        assert!(result.lateral_accel_mps2 > 0.0);
    }

    #[test]
    fn steering_at_rest_does_not_rotate_vehicle() {
        let model = vehicle(VehicleConfig::default());
        let result = model
            .step(state(0.0), ActuatorState::default(), command(0.0, 0.0, 1.0), 0.5)
            .unwrap();

        assert_eq!(result.state.speed_mps, 0.0);
        assert_eq!(result.state.yaw_rad, 0.0);
        assert_eq!(result.yaw_rate_rps, 0.0);
    }

    #[test]
    fn lateral_acceleration_respects_declared_friction_envelope() {
        let config = VehicleConfig {
            tire_friction_coefficient: 0.25,
            ..VehicleConfig::default()
        };
        let model = vehicle(config);
        let mut current_state = state(25.0);
        let mut actuators = ActuatorState::default();
        for _ in 0..20 {
            let result = model
                .step(current_state, actuators, command(0.0, 0.0, 1.0), 0.1)
                .unwrap();
            current_state = result.state;
            actuators = result.actuators;
            assert!(result.lateral_accel_mps2.abs() <= config.lateral_accel_limit_mps2() + 1e-12);
        }
        assert!(model.config().lateral_accel_limit_mps2() < config.max_lateral_accel_mps2);
    }

    #[test]
    fn steering_rate_limit_bounds_first_actuator_step() {
        let config = VehicleConfig::default();
        let model = vehicle(config);
        let result = model
            .step(state(5.0), ActuatorState::default(), command(0.0, 0.0, 1.0), 0.01)
            .unwrap();
        assert!(
            result.actuators.steering_angle_rad.abs()
                <= config.max_steering_rate_rad_s * 0.01 + 1e-12
        );
    }

    #[test]
    fn mirrored_steering_and_state_produce_mirrored_trajectory() {
        let model = vehicle(VehicleConfig::default());
        let left = VehicleState {
            x_m: 3.0,
            y_m: 1.5,
            yaw_rad: 0.15,
            speed_mps: 13.0,
        };
        let right = VehicleState {
            y_m: -left.y_m,
            yaw_rad: -left.yaw_rad,
            ..left
        };
        let left_result = model
            .step(left, ActuatorState::default(), command(0.0, 0.0, 0.6), 0.2)
            .unwrap();
        let right_result = model
            .step(right, ActuatorState::default(), command(0.0, 0.0, -0.6), 0.2)
            .unwrap();

        assert!((left_result.state.x_m - right_result.state.x_m).abs() < 1e-12);
        assert!((left_result.state.y_m + right_result.state.y_m).abs() < 1e-12);
        assert!((left_result.state.yaw_rad + right_result.state.yaw_rad).abs() < 1e-12);
        assert!((left_result.state.speed_mps - right_result.state.speed_mps).abs() < 1e-12);
        assert!((left_result.lateral_accel_mps2 + right_result.lateral_accel_mps2).abs() < 1e-12);
    }

    #[test]
    fn conflicting_pedals_fail_closed() {
        let model = vehicle(VehicleConfig::default());
        assert_eq!(
            model.step(state(10.0), ActuatorState::default(), command(0.2, 0.1, 0.0), 0.1),
            Err(VehicleError::ConflictingPedals)
        );
    }

    #[test]
    fn non_finite_commands_states_and_time_steps_are_rejected() {
        let model = vehicle(VehicleConfig::default());
        assert_eq!(
            model.step(state(10.0), ActuatorState::default(), command(f64::NAN, 0.0, 0.0), 0.1),
            Err(VehicleError::InvalidCommand)
        );
        assert_eq!(
            model.step(state(10.0), ActuatorState::default(), command(0.0, 0.0, 0.0), 0.0),
            Err(VehicleError::InvalidTimeStep)
        );
        assert_eq!(
            model.step(state(-1.0), ActuatorState::default(), command(0.0, 0.0, 0.0), 0.1),
            Err(VehicleError::InvalidState)
        );
    }

    #[test]
    fn invalid_vehicle_configuration_is_rejected() {
        let config = VehicleConfig {
            wheelbase_m: 0.0,
            ..VehicleConfig::default()
        };
        assert_eq!(
            AnalyticVehicle::new(config),
            Err(VehicleError::InvalidConfiguration)
        );
    }
}
