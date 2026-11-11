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

use std::f64::consts::TAU;
use std::fmt;

pub const MODEL_VERSION: &str = "electric-rotorcraft-drive-oracle-v0.3.0";

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

/// Positive motoring speed ratio, defined as motor speed divided by rotor speed.
///
/// This is a kinematic ratio only. Transmission efficiency remains separately
/// supplied through DriveConfig. Reverse power flow and direction changes are
/// outside this motoring-only reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmissionKinematics {
    pub motor_to_rotor_speed_ratio: f64,
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

/// Drive-chain power report augmented with the motor-side shaft operating point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveKinematicReport {
    pub drive: DriveReport,
    pub motor_to_rotor_speed_ratio: f64,
    pub motor_angular_velocity_rad_s: f64,
    pub motor_speed_rpm: f64,
    pub motor_shaft_output_torque_nm: f64,
}


/// A motor efficiency interval linked to one exact map and evidence manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct MotorEfficiencyEvidence {
    pub map_id: String,
    pub configuration_digest: String,
    pub context_digest: String,
    pub evidence_manifest_digest: String,
    pub motor_torque_nm: f64,
    pub motor_speed_rpm: f64,
    pub efficiency_lower_bound: f64,
    pub efficiency_nominal: f64,
    pub efficiency_upper_bound: f64,
}

/// Required configuration and operating context for the consuming aircraft.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveMapExpectedContext<'a> {
    pub configuration_digest: &'a str,
    pub context_digest: &'a str,
}

/// Motor-side shaft state, derived independently of motor electrical efficiency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotorShaftOperatingPoint {
    pub rotor_angular_velocity_rad_s: f64,
    pub rotor_shaft_power_w: f64,
    pub motor_angular_velocity_rad_s: f64,
    pub motor_speed_rpm: f64,
    pub motor_shaft_output_power_w: f64,
    pub motor_shaft_output_torque_nm: f64,
}

/// Three drive-chain evaluations over one map-qualified motor-efficiency interval.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveMapEnvelope {
    pub map_id: String,
    pub configuration_digest: String,
    pub context_digest: String,
    pub evidence_manifest_digest: String,
    pub motor_operating_point: MotorShaftOperatingPoint,
    pub efficiency_lower_bound: f64,
    pub efficiency_nominal: f64,
    pub efficiency_upper_bound: f64,
    /// Lowest efficiency / highest DC demand. Typed failures remain visible.
    pub adverse_efficiency_scenario: Result<DriveReport, DriveError>,
    pub nominal_efficiency_scenario: Result<DriveReport, DriveError>,
    /// Highest efficiency / lowest DC demand. Typed failures remain visible.
    pub favourable_efficiency_scenario: Result<DriveReport, DriveError>,
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
pub fn evaluate_drive(config: &DriveConfig, input: DriveInput) -> Result<DriveReport, DriveError> {
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
    let motor_shaft_output_power_w = rotor_shaft_power_w / config.transmission_efficiency;
    let transmission_loss_w = motor_shaft_output_power_w - rotor_shaft_power_w;
    let motor_electrical_input_power_w = motor_shaft_output_power_w / config.motor_efficiency;
    let motor_loss_w = motor_electrical_input_power_w - motor_shaft_output_power_w;
    let dc_bus_input_power_w = motor_electrical_input_power_w / config.inverter_efficiency;
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


/// Derive the motor-side shaft operating point from rotor demand and an
/// explicit transmission speed ratio/efficiency, without needing motor or
/// inverter efficiency. Use this before querying a motor efficiency map.
pub fn derive_motor_shaft_operating_point(
    input: DriveInput,
    transmission: TransmissionKinematics,
    transmission_efficiency: f64,
) -> Result<MotorShaftOperatingPoint, DriveError> {
    let ratio = transmission.motor_to_rotor_speed_ratio;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(DriveError::InvalidConfiguration(
            "motor-to-rotor speed ratio must be finite and positive for motoring",
        ));
    }
    if !transmission_efficiency.is_finite()
        || !(0.0 < transmission_efficiency && transmission_efficiency <= 1.0)
    {
        return Err(DriveError::InvalidConfiguration(
            "transmission efficiency must be finite and in (0, 1]",
        ));
    }
    if !input.rotor_shaft_torque_nm.is_finite()
        || !input.rotor_speed_rpm.is_finite()
        || input.rotor_shaft_torque_nm <= 0.0
        || input.rotor_speed_rpm <= 0.0
    {
        return Err(DriveError::InvalidOperatingPoint(
            "rotor torque and speed must be finite and positive for motoring",
        ));
    }

    let rotor_angular_velocity_rad_s = input.rotor_speed_rpm * TAU / 60.0;
    let rotor_shaft_power_w = input.rotor_shaft_torque_nm * rotor_angular_velocity_rad_s;
    let motor_angular_velocity_rad_s = rotor_angular_velocity_rad_s * ratio;
    let motor_speed_rpm = motor_angular_velocity_rad_s * 60.0 / TAU;
    let motor_shaft_output_power_w = rotor_shaft_power_w / transmission_efficiency;
    let motor_shaft_output_torque_nm =
        motor_shaft_output_power_w / motor_angular_velocity_rad_s;

    let derived = [
        rotor_angular_velocity_rad_s,
        rotor_shaft_power_w,
        motor_angular_velocity_rad_s,
        motor_speed_rpm,
        motor_shaft_output_power_w,
        motor_shaft_output_torque_nm,
    ];
    if derived.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(DriveError::NumericalOverflow);
    }

    Ok(MotorShaftOperatingPoint {
        rotor_angular_velocity_rad_s,
        rotor_shaft_power_w,
        motor_angular_velocity_rad_s,
        motor_speed_rpm,
        motor_shaft_output_power_w,
        motor_shaft_output_torque_nm,
    })
}

/// Evaluate one drive chain at lower, nominal and upper map-qualified motor
/// efficiencies. Expected configuration and context digests are checked before
/// any scenario is evaluated. Each scenario retains its own typed failure.
pub fn evaluate_drive_with_map_evidence(
    config: &DriveConfig,
    input: DriveInput,
    transmission: TransmissionKinematics,
    expected_context: DriveMapExpectedContext<'_>,
    evidence: &MotorEfficiencyEvidence,
) -> Result<DriveMapEnvelope, DriveError> {
    if evidence.map_id.trim().is_empty()
        || evidence.configuration_digest.trim().is_empty()
        || evidence.context_digest.trim().is_empty()
        || evidence.evidence_manifest_digest.trim().is_empty()
        || expected_context.configuration_digest.trim().is_empty()
        || expected_context.context_digest.trim().is_empty()
    {
        return Err(DriveError::InvalidConfiguration(
            "map, configuration, context and evidence identities must not be empty",
        ));
    }
    if evidence.configuration_digest != expected_context.configuration_digest {
        return Err(DriveError::InvalidConfiguration(
            "motor efficiency map configuration digest does not match drive configuration",
        ));
    }
    if evidence.context_digest != expected_context.context_digest {
        return Err(DriveError::InvalidConfiguration(
            "motor efficiency map context digest does not match drive operating context",
        ));
    }

    let efficiencies = [
        evidence.efficiency_lower_bound,
        evidence.efficiency_nominal,
        evidence.efficiency_upper_bound,
    ];
    if efficiencies
        .iter()
        .any(|value| !value.is_finite() || !(0.0 < *value && *value <= 1.0))
        || !(evidence.efficiency_lower_bound
            <= evidence.efficiency_nominal
            && evidence.efficiency_nominal
                <= evidence.efficiency_upper_bound)
    {
        return Err(DriveError::InvalidConfiguration(
            "motor efficiency bounds must be finite and ordered in (0, 1]",
        ));
    }

    let motor_operating_point = derive_motor_shaft_operating_point(
        input,
        transmission,
        config.transmission_efficiency,
    )?;
    if evidence.motor_torque_nm.to_bits()
        != motor_operating_point.motor_shaft_output_torque_nm.to_bits()
        || evidence.motor_speed_rpm.to_bits() != motor_operating_point.motor_speed_rpm.to_bits()
    {
        return Err(DriveError::InvalidOperatingPoint(
            "motor efficiency evidence does not match the current motor torque-speed point",
        ));
    }

    let evaluate_at_efficiency = |efficiency: f64| {
        let scenario_config = DriveConfig {
            motor_efficiency: efficiency,
            ..*config
        };
        evaluate_drive(&scenario_config, input)
    };

    Ok(DriveMapEnvelope {
        map_id: evidence.map_id.clone(),
        configuration_digest: evidence.configuration_digest.clone(),
        context_digest: evidence.context_digest.clone(),
        evidence_manifest_digest: evidence.evidence_manifest_digest.clone(),
        motor_operating_point,
        efficiency_lower_bound: evidence.efficiency_lower_bound,
        efficiency_nominal: evidence.efficiency_nominal,
        efficiency_upper_bound: evidence.efficiency_upper_bound,
        adverse_efficiency_scenario: evaluate_at_efficiency(evidence.efficiency_lower_bound),
        nominal_efficiency_scenario: evaluate_at_efficiency(evidence.efficiency_nominal),
        favourable_efficiency_scenario: evaluate_at_efficiency(evidence.efficiency_upper_bound),
    })
}

/// Evaluate the existing rotor-side drive-chain reference and derive the motor
/// shaft operating point using an explicit gearbox speed ratio.
///
/// The ratio convention is g = motor_speed / rotor_speed for positive
/// motoring. The existing transmission efficiency determines motor-side
/// mechanical power; this function uses that power and the transformed motor
/// angular velocity to calculate motor shaft torque. It does not infer a
/// gearbox ratio, efficiency map, motor limit, or physical operating domain.
pub fn evaluate_drive_with_kinematics(
    config: &DriveConfig,
    input: DriveInput,
    transmission: TransmissionKinematics,
) -> Result<DriveKinematicReport, DriveError> {
    let motor_point = derive_motor_shaft_operating_point(
        input,
        transmission,
        config.transmission_efficiency,
    )?;
    let drive = evaluate_drive(config, input)?;

    Ok(DriveKinematicReport {
        drive,
        motor_to_rotor_speed_ratio: transmission.motor_to_rotor_speed_ratio,
        motor_angular_velocity_rad_s: motor_point.motor_angular_velocity_rad_s,
        motor_speed_rpm: motor_point.motor_speed_rpm,
        motor_shaft_output_torque_nm: motor_point.motor_shaft_output_torque_nm,
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

    fn map_evidence(
        model: &DriveConfig,
        input: DriveInput,
        transmission: TransmissionKinematics,
    ) -> MotorEfficiencyEvidence {
        let point = derive_motor_shaft_operating_point(
            input,
            transmission,
            model.transmission_efficiency,
        )
        .unwrap();
        MotorEfficiencyEvidence {
            map_id: "motor-map-v1".to_owned(),
            configuration_digest: "aircraft-config-sha256".to_owned(),
            context_digest: "motor-context-sha256".to_owned(),
            evidence_manifest_digest: "measurement-manifest-sha256".to_owned(),
            motor_torque_nm: point.motor_shaft_output_torque_nm,
            motor_speed_rpm: point.motor_speed_rpm,
            efficiency_lower_bound: 0.90,
            efficiency_nominal: 0.95,
            efficiency_upper_bound: 0.98,
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
        assert!(
            (low_voltage.dc_bus_input_power_w - high_voltage.dc_bus_input_power_w).abs() < 1.0e-9
        );
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
            let mut transmission = config();
            transmission.transmission_efficiency = efficiency;
            assert!(matches!(
                evaluate_drive(&transmission, input_for_power(10_000.0, 400.0, 1.0)),
                Err(DriveError::InvalidConfiguration(_))
            ));

            let mut motor = config();
            motor.motor_efficiency = efficiency;
            assert!(matches!(
                evaluate_drive(&motor, input_for_power(10_000.0, 400.0, 1.0)),
                Err(DriveError::InvalidConfiguration(_))
            ));

            let mut inverter = config();
            inverter.inverter_efficiency = efficiency;
            assert!(matches!(
                evaluate_drive(&inverter, input_for_power(10_000.0, 400.0, 1.0)),
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
    fn identical_frozen_inputs_produce_identical_reports() {
        let mut model = config();
        model.transmission_efficiency = 0.91;
        model.motor_efficiency = 0.96;
        model.inverter_efficiency = 0.985;
        let input = input_for_power(18_500.0, 420.0, 7.5);
        let first = evaluate_drive(&model, input).unwrap();
        let second = evaluate_drive(&model, input).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_bus_limits_are_rejected() {
        let mut model = config();
        model.minimum_dc_bus_voltage_v = 0.0;
        assert!(matches!(
            evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)),
            Err(DriveError::InvalidConfiguration(_))
        ));

        model = config();
        model.maximum_dc_bus_current_a = f64::INFINITY;
        assert!(matches!(
            evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)),
            Err(DriveError::InvalidConfiguration(_))
        ));

        model = config();
        model.maximum_dc_bus_power_w = -1.0;
        assert!(matches!(
            evaluate_drive(&model, input_for_power(10_000.0, 400.0, 1.0)),
            Err(DriveError::InvalidConfiguration(_))
        ));
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

    #[test]
    fn motor_shaft_operating_point_is_independent_of_motor_efficiency() {
        let input = input_for_power(10_000.0, 400.0, 12.0);
        let point = derive_motor_shaft_operating_point(
            input,
            TransmissionKinematics {
                motor_to_rotor_speed_ratio: 5.0,
            },
            0.90,
        )
        .unwrap();
        assert!((point.motor_speed_rpm - 15_000.0).abs() < 1.0e-9);
        assert!((point.motor_shaft_output_power_w - 11_111.111_111_111_11).abs() < 1.0e-8);
        assert!((point.motor_shaft_output_torque_nm - 7.073_553_026_306_46).abs() < 1.0e-9);
    }

    #[test]
    fn map_evidence_produces_ordered_drive_demand_envelope() {
        let mut model = config();
        model.transmission_efficiency = 0.90;
        model.inverter_efficiency = 0.98;
        let input = input_for_power(10_000.0, 400.0, 12.0);
        let transmission = TransmissionKinematics {
            motor_to_rotor_speed_ratio: 5.0,
        };
        let evidence = map_evidence(&model, input, transmission);
        let result = evaluate_drive_with_map_evidence(
            &model,
            input,
            transmission,
            DriveMapExpectedContext {
                configuration_digest: "aircraft-config-sha256",
                context_digest: "motor-context-sha256",
            },
            &evidence,
        )
        .unwrap();

        let adverse = result.adverse_efficiency_scenario.unwrap();
        let nominal = result.nominal_efficiency_scenario.unwrap();
        let favourable = result.favourable_efficiency_scenario.unwrap();
        assert!(adverse.dc_bus_input_power_w > nominal.dc_bus_input_power_w);
        assert!(nominal.dc_bus_input_power_w > favourable.dc_bus_input_power_w);
        assert!(
            (adverse.dc_bus_input_power_w - 10_000.0 / (0.90 * 0.90 * 0.98)).abs()
                < 1.0e-8
        );
        assert!(
            (nominal.dc_bus_input_power_w - 10_000.0 / (0.90 * 0.95 * 0.98)).abs()
                < 1.0e-8
        );
        assert!(
            (favourable.dc_bus_input_power_w - 10_000.0 / (0.90 * 0.98 * 0.98)).abs()
                < 1.0e-8
        );
        assert_eq!(result.map_id, "motor-map-v1");
        assert_eq!(
            result.evidence_manifest_digest,
            "measurement-manifest-sha256"
        );
        assert!((result.motor_operating_point.motor_speed_rpm - 15_000.0).abs() < 1.0e-9);
    }

    #[test]
    fn adverse_bus_limit_failure_is_not_hidden_by_nominal_pass() {
        let mut model = config();
        model.transmission_efficiency = 0.90;
        model.inverter_efficiency = 0.98;
        model.maximum_dc_bus_power_w = 12_000.0;
        let input = input_for_power(10_000.0, 400.0, 12.0);
        let transmission = TransmissionKinematics {
            motor_to_rotor_speed_ratio: 5.0,
        };
        let evidence = map_evidence(&model, input, transmission);
        let result = evaluate_drive_with_map_evidence(
            &model,
            input,
            transmission,
            DriveMapExpectedContext {
                configuration_digest: "aircraft-config-sha256",
                context_digest: "motor-context-sha256",
            },
            &evidence,
        )
        .unwrap();

        assert!(matches!(
            result.adverse_efficiency_scenario,
            Err(DriveError::DcBusPowerLimitExceeded { .. })
        ));
        assert!(result.nominal_efficiency_scenario.is_ok());
        assert!(result.favourable_efficiency_scenario.is_ok());
    }

    #[test]
    fn map_configuration_and_context_mismatches_fail_closed() {
        let model = config();
        let input = input_for_power(10_000.0, 400.0, 1.0);
        let transmission = TransmissionKinematics {
            motor_to_rotor_speed_ratio: 1.0,
        };
        let evidence = map_evidence(&model, input, transmission);
        assert!(matches!(
            evaluate_drive_with_map_evidence(
                &model,
                input,
                transmission,
                DriveMapExpectedContext {
                    configuration_digest: "other-aircraft",
                    context_digest: "motor-context-sha256",
                },
                &evidence,
            ),
            Err(DriveError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            evaluate_drive_with_map_evidence(
                &model,
                input,
                transmission,
                DriveMapExpectedContext {
                    configuration_digest: "aircraft-config-sha256",
                    context_digest: "other-temperature-or-firmware-context",
                },
                &evidence,
            ),
            Err(DriveError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn stale_motor_operating_point_evidence_is_rejected() {
        let model = config();
        let input = input_for_power(10_000.0, 400.0, 1.0);
        let transmission = TransmissionKinematics {
            motor_to_rotor_speed_ratio: 1.0,
        };
        let mut evidence = map_evidence(&model, input, transmission);
        evidence.motor_speed_rpm += 1.0;
        assert!(matches!(
            evaluate_drive_with_map_evidence(
                &model,
                input,
                transmission,
                DriveMapExpectedContext {
                    configuration_digest: "aircraft-config-sha256",
                    context_digest: "motor-context-sha256",
                },
                &evidence,
            ),
            Err(DriveError::InvalidOperatingPoint(_))
        ));
    }

    #[test]
    fn invalid_map_efficiency_interval_and_missing_provenance_fail_closed() {
        let model = config();
        let input = input_for_power(10_000.0, 400.0, 1.0);
        let transmission = TransmissionKinematics {
            motor_to_rotor_speed_ratio: 1.0,
        };
        let mut evidence = map_evidence(&model, input, transmission);
        evidence.efficiency_lower_bound = 0.99;
        evidence.efficiency_nominal = 0.95;
        assert!(matches!(
            evaluate_drive_with_map_evidence(
                &model,
                input,
                transmission,
                DriveMapExpectedContext {
                    configuration_digest: "aircraft-config-sha256",
                    context_digest: "motor-context-sha256",
                },
                &evidence,
            ),
            Err(DriveError::InvalidConfiguration(_))
        ));

        let mut evidence = map_evidence(&model, input, transmission);
        evidence.efficiency_lower_bound = 0.0;
        assert!(matches!(
            evaluate_drive_with_map_evidence(
                &model,
                input,
                transmission,
                DriveMapExpectedContext {
                    configuration_digest: "aircraft-config-sha256",
                    context_digest: "motor-context-sha256",
                },
                &evidence,
            ),
            Err(DriveError::InvalidConfiguration(_))
        ));

        let mut evidence = map_evidence(&model, input, transmission);
        evidence.evidence_manifest_digest.clear();
        assert!(matches!(
            evaluate_drive_with_map_evidence(
                &model,
                input,
                transmission,
                DriveMapExpectedContext {
                    configuration_digest: "aircraft-config-sha256",
                    context_digest: "motor-context-sha256",
                },
                &evidence,
            ),
            Err(DriveError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn five_to_one_gearbox_maps_rotor_demand_to_motor_torque_speed() {
        let mut model = config();
        model.transmission_efficiency = 0.90;
        model.motor_efficiency = 0.95;
        model.inverter_efficiency = 0.98;
        let input = input_for_power(10_000.0, 400.0, 12.0);
        let report = evaluate_drive_with_kinematics(
            &model,
            input,
            TransmissionKinematics {
                motor_to_rotor_speed_ratio: 5.0,
            },
        )
        .unwrap();

        let expected_motor_angular_velocity = input.rotor_speed_rpm * TAU / 60.0 * 5.0;
        let expected_motor_power = 10_000.0 / 0.90;
        let expected_motor_torque = expected_motor_power / expected_motor_angular_velocity;
        assert!((report.motor_speed_rpm - 15_000.0).abs() < 1.0e-9);
        assert!(
            (report.motor_angular_velocity_rad_s - expected_motor_angular_velocity).abs()
                < 1.0e-12
        );
        assert!(
            (report.drive.motor_shaft_output_power_w - expected_motor_power).abs() < 1.0e-8
        );
        assert!(
            (report.motor_shaft_output_torque_nm - expected_motor_torque).abs() < 1.0e-12
        );
        assert!(
            (report.motor_shaft_output_torque_nm * report.motor_angular_velocity_rad_s
                - report.drive.motor_shaft_output_power_w)
                .abs()
                < 1.0e-8
        );
        assert!(
            (report.drive.dc_bus_input_power_w - 10_000.0 / (0.90 * 0.95 * 0.98)).abs()
                < 1.0e-8
        );
    }

    #[test]
    fn direct_drive_kinematics_preserve_rotor_speed_and_torque() {
        let model = config();
        let input = input_for_power(10_000.0, 400.0, 1.0);
        let report = evaluate_drive_with_kinematics(
            &model,
            input,
            TransmissionKinematics {
                motor_to_rotor_speed_ratio: 1.0,
            },
        )
        .unwrap();
        assert!((report.motor_speed_rpm - input.rotor_speed_rpm).abs() < 1.0e-12);
        assert!(
            (report.motor_shaft_output_torque_nm - input.rotor_shaft_torque_nm).abs() < 1.0e-12
        );
    }

    #[test]
    fn invalid_gear_ratios_are_rejected() {
        let model = config();
        let input = input_for_power(10_000.0, 400.0, 1.0);
        for ratio in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                evaluate_drive_with_kinematics(
                    &model,
                    input,
                    TransmissionKinematics {
                        motor_to_rotor_speed_ratio: ratio,
                    }
                ),
                Err(DriveError::InvalidConfiguration(_))
            ));
        }
    }

    #[test]
    fn gear_ratio_overflow_and_underflow_fail_closed() {
        let model = config();
        let input = input_for_power(10_000.0, 400.0, 1.0);
        for ratio in [f64::MAX, f64::MIN_POSITIVE] {
            assert_eq!(
                evaluate_drive_with_kinematics(
                    &model,
                    input,
                    TransmissionKinematics {
                        motor_to_rotor_speed_ratio: ratio,
                    }
                ),
                Err(DriveError::NumericalOverflow)
            );
        }
    }

    #[test]
    fn changing_gear_ratio_changes_motor_torque_speed_not_chain_power() {
        let mut model = config();
        model.transmission_efficiency = 0.90;
        let input = input_for_power(10_000.0, 400.0, 1.0);
        let low_ratio = evaluate_drive_with_kinematics(
            &model,
            input,
            TransmissionKinematics {
                motor_to_rotor_speed_ratio: 2.0,
            },
        )
        .unwrap();
        let high_ratio = evaluate_drive_with_kinematics(
            &model,
            input,
            TransmissionKinematics {
                motor_to_rotor_speed_ratio: 5.0,
            },
        )
        .unwrap();
        assert_eq!(
            low_ratio.drive.dc_bus_input_power_w,
            high_ratio.drive.dc_bus_input_power_w
        );
        assert!(high_ratio.motor_speed_rpm > low_ratio.motor_speed_rpm);
        assert!(high_ratio.motor_shaft_output_torque_nm < low_ratio.motor_shaft_output_torque_nm);
    }
}
