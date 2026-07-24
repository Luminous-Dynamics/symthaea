// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reduced-order engine, governor, transmission, and freewheel transients.
//!
//! Fuel availability alone does not imply instantaneous shaft power. This model
//! adds engine spool lag, governor response, torque limiting, transmission loss,
//! and one-way freewheel behavior so an engine flameout cannot drag rotor energy
//! through a fictitious rigid shaft.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DrivetrainTransientConfig {
    pub nominal_rotor_rpm: f64,
    pub maximum_engine_power_w: f64,
    pub maximum_shaft_torque_nm: f64,
    pub transmission_efficiency: f64,
    pub spool_up_time_constant_s: f64,
    pub spool_down_time_constant_s: f64,
    pub governor_time_constant_s: f64,
    pub governor_rpm_gain: f64,
    pub freewheel_release_slip_rpm: f64,
}

impl Default for DrivetrainTransientConfig {
    fn default() -> Self {
        Self {
            nominal_rotor_rpm: 3_300.0,
            maximum_engine_power_w: 420_000.0,
            maximum_shaft_torque_nm: 1_500.0,
            transmission_efficiency: 0.94,
            spool_up_time_constant_s: 2.5,
            spool_down_time_constant_s: 1.0,
            governor_time_constant_s: 0.35,
            governor_rpm_gain: 1.5,
            freewheel_release_slip_rpm: 75.0,
        }
    }
}

impl DrivetrainTransientConfig {
    pub fn validate(&self) -> Result<(), DrivetrainTransientError> {
        let positive = [
            self.nominal_rotor_rpm,
            self.maximum_engine_power_w,
            self.maximum_shaft_torque_nm,
            self.spool_up_time_constant_s,
            self.spool_down_time_constant_s,
            self.governor_time_constant_s,
            self.freewheel_release_slip_rpm,
        ];
        if positive
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || !self.transmission_efficiency.is_finite()
            || !(0.0..=1.0).contains(&self.transmission_efficiency)
            || !self.governor_rpm_gain.is_finite()
            || self.governor_rpm_gain < 0.0
        {
            return Err(DrivetrainTransientError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrivetrainTransientError {
    InvalidConfiguration,
    InvalidInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DrivetrainInput {
    pub throttle_command: f64,
    pub rotor_rpm: f64,
    pub demanded_rotor_power_w: f64,
    pub engine_available: bool,
    pub dt_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DrivetrainTransientState {
    pub engine_speed_fraction: f64,
    pub governor_fraction: f64,
    pub equivalent_engine_rotor_rpm: f64,
    pub clutch_engaged: bool,
    pub delivered_shaft_power_w: f64,
    pub delivered_shaft_torque_nm: f64,
    pub torque_limited: bool,
    pub power_deficit_w: f64,
    pub cumulative_disengagements: u64,
}

#[derive(Debug, Clone)]
pub struct DrivetrainTransientModel {
    config: DrivetrainTransientConfig,
    state: DrivetrainTransientState,
}

impl Default for DrivetrainTransientModel {
    fn default() -> Self {
        Self::new(DrivetrainTransientConfig::default())
            .expect("default drivetrain transient configuration must remain valid")
    }
}

impl DrivetrainTransientModel {
    pub fn new(config: DrivetrainTransientConfig) -> Result<Self, DrivetrainTransientError> {
        config.validate()?;
        Ok(Self {
            config,
            state: DrivetrainTransientState {
                engine_speed_fraction: 0.0,
                governor_fraction: 0.0,
                equivalent_engine_rotor_rpm: 0.0,
                clutch_engaged: false,
                delivered_shaft_power_w: 0.0,
                delivered_shaft_torque_nm: 0.0,
                torque_limited: false,
                power_deficit_w: 0.0,
                cumulative_disengagements: 0,
            },
        })
    }

    pub fn state(&self) -> DrivetrainTransientState {
        self.state
    }

    pub fn reset(&mut self, engine_speed_fraction: f64) -> Result<(), DrivetrainTransientError> {
        if !engine_speed_fraction.is_finite() || !(0.0..=1.0).contains(&engine_speed_fraction) {
            return Err(DrivetrainTransientError::InvalidInput);
        }
        self.state = DrivetrainTransientState {
            engine_speed_fraction,
            governor_fraction: engine_speed_fraction,
            equivalent_engine_rotor_rpm: engine_speed_fraction * self.config.nominal_rotor_rpm,
            clutch_engaged: engine_speed_fraction > 0.0,
            delivered_shaft_power_w: 0.0,
            delivered_shaft_torque_nm: 0.0,
            torque_limited: false,
            power_deficit_w: 0.0,
            cumulative_disengagements: 0,
        };
        Ok(())
    }

    pub fn step(
        &mut self,
        input: DrivetrainInput,
    ) -> Result<DrivetrainTransientState, DrivetrainTransientError> {
        self.config.validate()?;
        if !input.throttle_command.is_finite()
            || !(0.0..=1.0).contains(&input.throttle_command)
            || !input.rotor_rpm.is_finite()
            || input.rotor_rpm < 0.0
            || !input.demanded_rotor_power_w.is_finite()
            || input.demanded_rotor_power_w < 0.0
            || !input.dt_s.is_finite()
            || input.dt_s <= 0.0
        {
            return Err(DrivetrainTransientError::InvalidInput);
        }

        let engine_target = if input.engine_available {
            input.throttle_command
        } else {
            0.0
        };
        let spool_tau = if engine_target >= self.state.engine_speed_fraction {
            self.config.spool_up_time_constant_s
        } else {
            self.config.spool_down_time_constant_s
        };
        self.state.engine_speed_fraction += first_order_alpha(input.dt_s, spool_tau)
            * (engine_target - self.state.engine_speed_fraction);
        self.state.engine_speed_fraction = self.state.engine_speed_fraction.clamp(0.0, 1.0);
        self.state.equivalent_engine_rotor_rpm =
            self.state.engine_speed_fraction * self.config.nominal_rotor_rpm;

        let rpm_error_fraction = if self.config.nominal_rotor_rpm > 0.0 {
            (self.config.nominal_rotor_rpm - input.rotor_rpm) / self.config.nominal_rotor_rpm
        } else {
            0.0
        };
        let governor_target = (input.throttle_command
            + self.config.governor_rpm_gain * rpm_error_fraction)
            .clamp(0.0, 1.0);
        self.state.governor_fraction +=
            first_order_alpha(input.dt_s, self.config.governor_time_constant_s)
                * (governor_target - self.state.governor_fraction);
        self.state.governor_fraction = self.state.governor_fraction.clamp(0.0, 1.0);

        let was_engaged = self.state.clutch_engaged;
        self.state.clutch_engaged = input.engine_available
            && self.state.equivalent_engine_rotor_rpm + self.config.freewheel_release_slip_rpm
                >= input.rotor_rpm;
        if was_engaged && !self.state.clutch_engaged {
            self.state.cumulative_disengagements =
                self.state.cumulative_disengagements.saturating_add(1);
        }

        if !self.state.clutch_engaged {
            self.state.delivered_shaft_power_w = 0.0;
            self.state.delivered_shaft_torque_nm = 0.0;
            self.state.torque_limited = false;
            self.state.power_deficit_w = input.demanded_rotor_power_w;
            return Ok(self.state);
        }

        let available_engine_power_w = self.config.maximum_engine_power_w
            * self.state.engine_speed_fraction
            * self.state.governor_fraction;
        let pre_torque_power_w = input.demanded_rotor_power_w.min(available_engine_power_w)
            * self.config.transmission_efficiency;
        let rotor_omega_rad_s = rpm_to_rad_s(input.rotor_rpm).max(1.0);
        let unconstrained_torque_nm = pre_torque_power_w / rotor_omega_rad_s;
        self.state.torque_limited = unconstrained_torque_nm > self.config.maximum_shaft_torque_nm;
        self.state.delivered_shaft_torque_nm =
            unconstrained_torque_nm.min(self.config.maximum_shaft_torque_nm);
        self.state.delivered_shaft_power_w =
            self.state.delivered_shaft_torque_nm * rotor_omega_rad_s;
        self.state.power_deficit_w =
            (input.demanded_rotor_power_w - self.state.delivered_shaft_power_w).max(0.0);
        Ok(self.state)
    }
}

fn first_order_alpha(dt_s: f64, time_constant_s: f64) -> f64 {
    1.0 - (-dt_s / time_constant_s).exp()
}

fn rpm_to_rad_s(rpm: f64) -> f64 {
    rpm * std::f64::consts::TAU / 60.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(engine_available: bool) -> DrivetrainInput {
        DrivetrainInput {
            throttle_command: 1.0,
            rotor_rpm: 3_300.0,
            demanded_rotor_power_w: 300_000.0,
            engine_available,
            dt_s: 0.1,
        }
    }

    #[test]
    fn engine_power_spools_in_instead_of_appearing_instantly() {
        let mut model = DrivetrainTransientModel::default();
        let first = model.step(input(true)).unwrap();
        assert!(first.engine_speed_fraction > 0.0);
        assert!(first.engine_speed_fraction < 0.2);
        assert!(first.delivered_shaft_power_w < 300_000.0);
    }

    #[test]
    fn flameout_disengages_freewheel_and_does_not_drag_rotor() {
        let mut model = DrivetrainTransientModel::default();
        model.reset(1.0).unwrap();
        let nominal = model.step(input(true)).unwrap();
        assert!(nominal.clutch_engaged);
        let failed = model.step(input(false)).unwrap();
        assert!(!failed.clutch_engaged);
        assert_eq!(failed.delivered_shaft_torque_nm, 0.0);
        assert_eq!(failed.power_deficit_w, 300_000.0);
    }

    #[test]
    fn shaft_torque_is_explicitly_limited() {
        let config = DrivetrainTransientConfig {
            maximum_shaft_torque_nm: 100.0,
            ..DrivetrainTransientConfig::default()
        };
        let mut model = DrivetrainTransientModel::new(config).unwrap();
        model.reset(1.0).unwrap();
        let state = model.step(input(true)).unwrap();
        assert!(state.torque_limited);
        assert!(state.delivered_shaft_torque_nm <= 100.0);
    }

    #[test]
    fn invalid_time_step_fails_closed() {
        let mut model = DrivetrainTransientModel::default();
        let mut bad = input(true);
        bad.dt_s = 0.0;
        assert_eq!(
            model.step(bad).unwrap_err(),
            DrivetrainTransientError::InvalidInput
        );
    }
}
