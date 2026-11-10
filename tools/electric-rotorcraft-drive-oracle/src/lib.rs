// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent, single-path electric drive-chain reference oracle.
//!
//! The model maps demanded rotor-side torque and RPM through an explicitly
//! supplied transmission, motor and inverter efficiency to DC-bus power.
//! Efficiencies are operating-point inputs, not defaults or inferred maps.
//!
//! The model is deliberately independent of the production aircraft,
//! helicopter, multirotor, battery and thermal implementations. It contains
//! no third-party dependencies and is intended for deterministic analytical
//! reference cases only.
//!
//! Losses are reported at their component boundaries. This crate does not
//! decide which thermal node receives a loss or claim that all loss energy is
//! deposited in the battery.

#![forbid(unsafe_code)]

use std::fmt;
use std::f64::consts::TAU;

pub const MODEL_VERSION: &str = "electric-rotorcraft-drive-oracle-v0.1.0";

/// Explicit operating-point efficiencies and bus limits.
///
/// A direct-drive architecture must explicitly set transmission efficiency
/// to 1.0. No efficiency or bus-limit defaults are provided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveConfig {
    /// Mechanical transmission efficiency from motor shaft to rotor shaft.
    pub transmission_efficiency: f64,
    /// Motor efficiency from motor electrical input to motor shaft output.
    pub motor_efficiency: f64,
    /// Inverter efficiency from DC-bus input to motor electrical output.
    pub inverter_efficiency: f64,
    /// Minimum characterized DC-bus voltage for this operating profile (V).
    pub minimum_dc_bus_voltage_v: f64,
    /// Maximum permitted DC-bus current for this operating profile (A).
    pub maximum_dc_bus_current_a: f64,
    /// Maximum permitted DC-bus input power for this operating profile (W).
    pub maximum_dc_bus_power_w: f64,
    /// Maximum absolute energy-closure residual accepted for one substep (J).
    pub energy_balance_tolerance_j: f64,
}

impl DriveConfig {
    pub fn validate(&self) -> Result<(), DriveError> {
        let efficiencies = [
            self.transmission_efficiency,
            self.motor_efficiency,
            self.inverter_efficiency,
        ];
        if efficiencies
            .iter()
            .any(|value| !value.is_finite() || !(0.0 < *value && *value <= 1.0))
        {
            return Err(DriveError::InvalidConfiguration(
                "all conversion efficiencies must be finite and in (0, 1]",
            ));
        }

        let positive = [
            self.minimum_dc_bus_voltage_v,
            self.maximum_dc_bus_current_a,
            self.maximum_dc_bus_power_w,
            self.energy_balance_tolerance_j,
        ];
        if positive
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(DriveError::InvalidConfiguration(
                "bus limits and energy tolerance must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// One positive-power rotor-side operating point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveInput {
    /// Required rotor-side shaft torque (N m).
    pub rotor_shaft_torque_nm: f64,
    /// Rotor-side rotational speed (revolutions per minute).
    pub rotor_speed_rpm: f64,
    /// Actual or independently modelled DC-bus voltage at this operating point (V).
    pub dc_bus_voltage_v: f64,
    /// Duration represented by this operating point (s).
    pub duration_s: f64,
}

/// Component-boundary power and energy accounting for one drive path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveReport {
    pub model_version: &'static str,
    pub rotor_angular_velocity_rad_s: f64,
    pub rotor_shaft_power_w: f64,
    /// Motor mechanical output before transmission loss (W).
    pub motor_shaft_output_power_w: f64,
    pub transmission_loss_w: f64,
    /// Motor electrical input after inverter, before motor conversion (W).
    pub motor_electrical_input_power_w: f64,
    pub motor_loss_w: f64,
    /// DC-bus input power before inverter conversion loss (W).
    pub dc_bus_input_power_w: f64,
    pub inverter_loss_w: f64,
    pub dc_bus_current_a: f64,
    pub total_conversion_loss_w: f64,
    /// DC input power minus rotor shaft power and all three conversion losses (W).
    pub power_balance_residual_w: f64,
    pub rotor_shaft_energy_j: f64,
    pub transmission_loss_energy_j: f64,
    pub motor_loss_energy_j: f64,
    pub inverter_loss_energy_j: f64,
    pub dc_bus_input_energy_j: f64,
    /// DC input energy minus rotor energy and all component loss energies (J).
    pub energy_balance_residual_j: f64,
}

/// Explicit non-PASS outcomes for unsupported or infeasible operating points.
#[derive(Debug, Clone, PartialEq)]
pub enum DriveError {
    InvalidConfiguration(&'static str),
    InvalidOperatingPoint(&'static str),
    DcBusUndervoltage {
        measured_voltage_v: f64,
        minimum_voltage_v: f64,
    },
    DcBusPowerLimitExceeded {
        required_power_w: f64,
        maximum_power_w: f64,
    },
    DcBusCurrentLimitExceeded {
        required_current_a: f64,
        maximum_current_a: f64,
    },
    NumericalOverflow,
    EnergyBalanceExceeded {
        residual_j: f64,
        tolerance_j: f64,
    },
}

impl fmt::Display for DriveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DriveError {}

/// Evaluate one rotor-side motoring operating point.
///
/// Power flow is:
///
/// rotor shaft -> transmission -> motor -> inverter -> DC bus
///
/// The caller supplies measured or explicitly modelled efficiencies for the
/// exact torque/RPM/temperature operating point. This function does not
/// interpolate maps, extrapolate outside a map, or silently derate demand.
pub fn evaluate_drive(
    config: &DriveConfig,
    input: DriveInput,
) -> Result<DriveReport, DriveError> {
    config.validate()?;

    let finite_inputs = [
        input.rotor_shaft_torque_nm,
        input.rotor_speed_rpm,
        input.dc_bus_voltage_v,
        input.duration_s,
    ];
    if finite_inputs.iter().any(|value| !value.is_finite()) {
        return Err(DriveError::InvalidOperatingPoint(
            "torque, speed, bus voltage and duration must be finite",
        ));
    }
    if input.rotor_shaft_torque_nm <= 0.0 {
        return Err(DriveError::InvalidOperatingPoint(
            "rotor shaft torque must be positive for this motoring-only model",
        ));
    }
    if input.rotor_speed_rpm <= 0.0 {
        return Err(DriveError::InvalidOperatingPoint(
            "rotor speed must be positive for this motoring-only model",
        ));
    }
    if input.dc_bus_voltage_v <= 0.0 || input.duration_s <= 0.0 {
        return Err(DriveError::InvalidOperatingPoint(
            "DC-bus voltage and duration must be positive",
        ));
    }
    if input.dc_bus_voltage_v < config.minimum_dc_bus_voltage_v {
        return Err(DriveError::DcBusUndervoltage {
            measured_voltage_v: input.dc_bus_voltage_v,
            minimum_voltage_v: config.minimum_dc_bus_voltage_v,
        });
    }

    let rotor_angular_velocity_rad_s = input.rotor_speed_rpm * TAU / 60.0;
    let rotor_shaft_power_w = input.rotor_shaft_torque_nm * rotor_angular_velocity_rad_s;
    let motor_shaft_output_power_w =
        rotor_shaft_power_w / config.transmission_efficiency;
    let transmission_loss_w = motor_shaft_output_power_w - rotor_shaft_power_w;
    let motor_electrical_input_power_w =
        motor_shaft_output_power_w / config.motor_efficiency;
    let motor_loss_w = motor_electrical_input_power_w - motor_shaft_output_power_w;
    let dc_bus_input_power_w =
        motor_electrical_input_power_w / config.inverter_efficiency;
    let inverter_loss_w = dc_bus_input_power_w - motor_electrical_input_power_w;
    let dc_bus_current_a = dc_bus_input_power_w / input.dc_bus_voltage_v;
    let total_conversion_loss_w = transmission_loss_w + motor_loss_w + inverter_loss_w;
    let power_balance_residual_w =
        dc_bus_input_power_w - rotor_shaft_power_w - total_conversion_loss_w;

    let derived = [
        rotor_angular_velocity_rad_s,
        rotor_shaft_power_w,
        motor_shaft_output_power_w,
        transmission_loss_w,
        motor_electrical_input_power_w,
        motor_loss_w,
        dc_bus_input_power_w,
        inverter_loss_w,
        dc_bus_current_a,
        total_conversion_loss_w,
        power_balance_residual_w,
    ];
    if derived.iter().any(|value| !value.is_finite())
        || derived[..10].iter().any(|value| *value < 0.0)
        || rotor_shaft_power_w <= 0.0
        || dc_bus_input_power_w <= 0.0
        || dc_bus_current_a <= 0.0
    {
        return Err(DriveError::NumericalOverflow);
    }

    if dc_bus_input_power_w > config.maximum_dc_bus_power_w {
        return Err(DriveError::DcBusPowerLimitExceeded {
            required_power_w: dc_bus_input_power_w,
            maximum_power_w: config.maximum_dc_bus_power_w,
        });
    }
    if dc_bus_current_a > config.maximum_dc_bus_current_a {
        return Err(DriveError::DcBusCurrentLimitExceeded {
            required_current_a: dc_bus_current_a,
            maximum_current_a: config.maximum_dc_bus_current_a,
        });
    }

    let rotor_shaft_energy_j = rotor_shaft_power_w * input.duration_s;
    let transmission_loss_energy_j = transmission_loss_w * input.duration_s;
    let motor_loss_energy_j = motor_loss_w * input.duration_s;
    let inverter_loss_energy_j = inverter_loss_w * input.duration_s;
    let dc_bus_input_energy_j = dc_bus_input_power_w * input.duration_s;
    let energy_balance_residual_j = dc_bus_input_energy_j
        - rotor_shaft_energy_j
        - transmission_loss_energy_j
        - motor_loss_energy_j
        - inverter_loss_energy_j;

    let energies = [
        rotor_shaft_energy_j,
        transmission_loss_energy_j,
        motor_loss_energy_j,
        inverter_loss_energy_j,
        dc_bus_input_energy_j,
        energy_balance_residual_j,
    ];
    if energies.iter().any(|value| !value.is_finite()) {
        return Err(DriveError::NumericalOverflow);
    }
    if energy_balance_residual_j.abs() > config.energy_balance_tolerance_j {
        return Err(DriveError::EnergyBalanceExceeded {
            residual_j: energy_balance_residual_j,
            tolerance_j: config.energy_balance_tolerance_j,
        });
    }

    Ok(DriveReport {
        model_version: MODEL_VERSION,
        rotor_angular_velocity_rad_s,
        rotor_shaft_power_w,
        motor_shaft_output_power_w,
        transmission_loss_w,
        motor_electrical_input_power_w,
        motor_loss_w,
        dc_bus_input_power_w,
        inverter_loss_w,
        dc_bus_current_a,
        total_conversion_loss_w,
        power_balance_residual_w,
        rotor_shaft_energy_j,
        transmission_loss_energy_j,
        motor_loss_energy_j,
        inverter_loss_energy_j,
        dc_bus_input_energy_j,
        energy_balance_residual_j,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DriveConfig {
        DriveConfig {
            transmission_efficiency: 1.0,
            motor_efficiency: 1.0,
            inverter_efficiency: 1.0,
            minimum_dc_bus_voltage_v: 300.0,
            maximum_dc_bus_current_a: 500.0,
            maximum_dc_bus_power_w: 200_000.0,
            energy_balance_tolerance_j: 1.0e-8,
        }
    }

    fn input_for_power(power_w: f64, bus_voltage_v: f64, duration_s: f64) -> DriveInput {
        let speed_rpm = 3_000.0;
        let angular_velocity = speed_rpm * TAU / 60.0;
        DriveInput {
            rotor_shaft_torque_nm: power_w / angular_velocity,
            rotor_speed_rpm: speed_rpm,
            dc_bus_voltage_v: bus_voltage_v,
            duration_s,
        }
    }

    #[test]
    fn unit_efficiency_path_matches_rotor_power_and_bus_current() {
        let report = evaluate_drive(&config(), input_for_power(10_000.0, 400.0, 2.0)).unwrap();
        assert!((report.rotor_shaft_power_w - 10_000.0).abs() < 1.0e-9);
        assert!((report.dc_bus_input_power_w - 10_000.0).abs() < 1.0e-9);
        assert!((report.dc_bus_current_a - 25.0).abs() < 1.0e-12);
        assert_eq!(report.total_conversion_loss_w, 0.0);
    }

    #[test]
    fn three_stage_efficiency_composition_matches_closed_form() {
        let mut model = config();
        model.transmission_efficiency = 0.90;
        model.motor_efficiency = 0.95;
        model.inverter_efficiency = 0.98;
        let report = evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)).unwrap();
        let expected = 10_000.0 / (0.90 * 0.95 * 0.98);
        assert!((report.dc_bus_input_power_w - expected).abs() < 1.0e-9);
        assert!(report.dc_bus_input_power_w > report.motor_electrical_input_power_w);
        assert!(report.motor_electrical_input_power_w > report.motor_shaft_output_power_w);
        assert!(report.motor_shaft_output_power_w > report.rotor_shaft_power_w);
    }

    #[test]
    fn transmission_loss_is_isolated_at_its_declared_boundary() {
        let mut model = config();
        model.transmission_efficiency = 0.80;
        let report = evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)).unwrap();
        assert!((report.motor_shaft_output_power_w - 12_500.0).abs() < 1.0e-9);
        assert!((report.transmission_loss_w - 2_500.0).abs() < 1.0e-9);
        assert_eq!(report.motor_loss_w, 0.0);
        assert_eq!(report.inverter_loss_w, 0.0);
    }

    #[test]
    fn motor_loss_is_isolated_at_its_declared_boundary() {
        let mut model = config();
        model.motor_efficiency = 0.80;
        let report = evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)).unwrap();
        assert!((report.motor_electrical_input_power_w - 12_500.0).abs() < 1.0e-9);
        assert!((report.motor_loss_w - 2_500.0).abs() < 1.0e-9);
        assert_eq!(report.transmission_loss_w, 0.0);
        assert_eq!(report.inverter_loss_w, 0.0);
    }

    #[test]
    fn inverter_loss_is_isolated_at_its_declared_boundary() {
        let mut model = config();
        model.inverter_efficiency = 0.80;
        let report = evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)).unwrap();
        assert!((report.dc_bus_input_power_w - 12_500.0).abs() < 1.0e-9);
        assert!((report.inverter_loss_w - 2_500.0).abs() < 1.0e-9);
        assert_eq!(report.transmission_loss_w, 0.0);
        assert_eq!(report.motor_loss_w, 0.0);
    }

    #[test]
    fn changing_bus_voltage_changes_current_not_required_power() {
        let model = config();
        let low_voltage = evaluate_drive(&model, input_for_power(20_000.0, 400.0, 1.0)).unwrap();
        let high_voltage = evaluate_drive(&model, input_for_power(20_000.0, 800.0, 1.0)).unwrap();
        assert!((low_voltage.dc_bus_input_power_w - high_voltage.dc_bus_input_power_w).abs() < 1.0e-9);
        assert!((low_voltage.dc_bus_current_a - 50.0).abs() < 1.0e-12);
        assert!((high_voltage.dc_bus_current_a - 25.0).abs() < 1.0e-12);
    }

    #[test]
    fn undervoltage_is_a_typed_non_pass() {
        let mut model = config();
        model.minimum_dc_bus_voltage_v = 350.0;
        assert!(matches!(
            evaluate_drive(&model, input_for_power(10_000.0, 300.0, 1.0)),
            Err(DriveError::DcBusUndervoltage { .. })
        ));
    }

    #[test]
    fn current_limit_is_a_typed_non_pass() {
        let mut model = config();
        model.maximum_dc_bus_current_a = 20.0;
        assert!(matches!(
            evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)),
            Err(DriveError::DcBusCurrentLimitExceeded { .. })
        ));
    }

    #[test]
    fn bus_power_limit_is_a_typed_non_pass() {
        let mut model = config();
        model.maximum_dc_bus_power_w = 9_000.0;
        assert!(matches!(
            evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)),
            Err(DriveError::DcBusPowerLimitExceeded { .. })
        ));
    }

    #[test]
    fn invalid_efficiencies_are_rejected_without_defaults() {
        for efficiency in [0.0, -0.1, 1.01, f64::NAN, f64::INFINITY] {
            let mut model = config();
            model.motor_efficiency = efficiency;
            assert!(matches!(
                evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)),
                Err(DriveError::InvalidConfiguration(_))
            ));
        }
    }

    #[test]
    fn invalid_torque_speed_voltage_and_duration_are_rejected() {
        let model = config();
        let mut input = input_for_power(10_000.0, 400.0, 1.0);
        input.rotor_shaft_torque_nm = 0.0;
        assert!(matches!(
            evaluate_drive(&model, input),
            Err(DriveError::InvalidOperatingPoint(_))
        ));

        input = input_for_power(10_000.0, 400.0, 1.0);
        input.rotor_speed_rpm = -1.0;
        assert!(matches!(
            evaluate_drive(&model, input),
            Err(DriveError::InvalidOperatingPoint(_))
        ));

        input = input_for_power(10_000.0, 400.0, 1.0);
        input.dc_bus_voltage_v = 0.0;
        assert!(matches!(
            evaluate_drive(&model, input),
            Err(DriveError::InvalidOperatingPoint(_))
        ));

        input = input_for_power(10_000.0, 400.0, 1.0);
        input.duration_s = f64::NAN;
        assert!(matches!(
            evaluate_drive(&model, input),
            Err(DriveError::InvalidOperatingPoint(_))
        ));
    }

    #[test]
    fn substep_energy_closes_and_component_losses_remain_separate() {
        let mut model = config();
        model.transmission_efficiency = 0.90;
        model.motor_efficiency = 0.95;
        model.inverter_efficiency = 0.98;
        let report = evaluate_drive(&model, input_for_power(10_000.0, 400.0, 12.0)).unwrap();
        let accounted = report.rotor_shaft_energy_j
            + report.transmission_loss_energy_j
            + report.motor_loss_energy_j
            + report.inverter_loss_energy_j;
        assert!((accounted - report.dc_bus_input_energy_j).abs() < 1.0e-7);
        assert!(report.transmission_loss_energy_j > 0.0);
        assert!(report.motor_loss_energy_j > 0.0);
        assert!(report.inverter_loss_energy_j > 0.0);
    }

    #[test]
    fn direct_drive_requires_explicit_unity_transmission_efficiency() {
        let mut model = config();
        model.transmission_efficiency = 1.0;
        let report = evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)).unwrap();
        assert_eq!(report.transmission_loss_w, 0.0);
    }

    #[test]
    fn extreme_finite_inputs_fail_closed_on_intermediate_overflow() {
        let model = config();
        let input = DriveInput {
            rotor_shaft_torque_nm: f64::MAX,
            rotor_speed_rpm: f64::MAX,
            dc_bus_voltage_v: 400.0,
            duration_s: 1.0,
        };
        assert_eq!(
            evaluate_drive(&model, input),
            Err(DriveError::NumericalOverflow)
        );
    }
}
