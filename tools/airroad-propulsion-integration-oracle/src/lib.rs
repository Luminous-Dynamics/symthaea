// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pinned, synthetic end-to-end composition of the electric propulsion evidence
//! oracles. This crate qualifies software hand-offs only; it is not a physical
//! motor, battery, thermal, aircraft, safety, or certification model.

#![forbid(unsafe_code)]

use electric_propulsion_map_oracle::{
    EfficiencyMap2D, MapError, MapLookup, MapNode, MapQuery,
};
use electric_propulsion_observation_oracle::{
    CalibrationStatus, MeasurementChannel, MeasurementQuantity, MeasurementSample,
    MeasurementUnit, MotorMapOperatingPoint, ObservationError, RunManifest, SampleQuality,
    SignConvention, TimeDomain, derive_motor_map_operating_point, synchronize_at,
};
use electric_rotorcraft_drive_oracle::{
    DriveConfig, DriveError, DriveInput, DriveMapEnvelope, DriveMapExpectedContext,
    MotorEfficiencyEvidence, TransmissionKinematics, derive_motor_shaft_operating_point,
    evaluate_drive_with_map_evidence,
};
use electric_thermal_loss_ledger_oracle::{
    ElectricalLoad, ElectricalLoadKind, ElectricalLoadReport, HeatAllocationReport,
    HeatLossKind, HeatLossSource, LedgerError, ThermalNodeAllocation,
    allocate_loss_heat, evaluate_electrical_loads,
};
use std::f64::consts::TAU;
use std::fmt;

const CONFIGURATION_DIGEST: &str = "synthetic-aircraft-configuration-sha256";
const CONTEXT_DIGEST: &str = "synthetic-hot-day-hover-context-sha256";
const EVIDENCE_MANIFEST_DIGEST: &str = "synthetic-propulsion-run-manifest-sha256";
const MAP_CONTENT_DIGEST: &str = "synthetic-motor-map-content-sha256";
const ANCHOR_TIMESTAMP_NS: i64 = 1_000_000;

/// Errors retain the failing composition boundary instead of collapsing all
/// failures into a generic infeasible result.
#[derive(Debug)]
pub enum IntegrationError {
    Observation(ObservationError),
    Map(MapError),
    Drive(DriveError),
    Ledger(LedgerError),
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for IntegrationError {}

impl From<ObservationError> for IntegrationError {
    fn from(value: ObservationError) -> Self {
        Self::Observation(value)
    }
}

impl From<MapError> for IntegrationError {
    fn from(value: MapError) -> Self {
        Self::Map(value)
    }
}

impl From<DriveError> for IntegrationError {
    fn from(value: DriveError) -> Self {
        Self::Drive(value)
    }
}

impl From<LedgerError> for IntegrationError {
    fn from(value: LedgerError) -> Self {
        Self::Ledger(value)
    }
}

/// All outputs from the single-substep synthetic composition fixture.
#[derive(Debug)]
pub struct IntegrationReport {
    pub motor_observation: MotorMapOperatingPoint,
    pub map_lookup: MapLookup,
    pub drive_envelope: DriveMapEnvelope,
    pub electrical_loads: ElectricalLoadReport,
    pub thermal_allocation: HeatAllocationReport,
}

struct CoreResult {
    manifest: RunManifest,
    observation: MotorMapOperatingPoint,
    map: EfficiencyMap2D,
    map_lookup: MapLookup,
    evidence: MotorEfficiencyEvidence,
    drive_config: DriveConfig,
    drive_input: DriveInput,
    transmission: TransmissionKinematics,
    drive_envelope: DriveMapEnvelope,
}

fn sample(sequence: u64, timestamp_ns: i64, value: f64, uncertainty: f64) -> MeasurementSample {
    MeasurementSample {
        source_sequence: sequence,
        timestamp_ns,
        value,
        absolute_uncertainty: uncertainty,
        quality: SampleQuality::Valid,
    }
}

fn channel(
    channel_id: &str,
    quantity: MeasurementQuantity,
    unit: MeasurementUnit,
    value: f64,
    uncertainty: f64,
) -> MeasurementChannel {
    MeasurementChannel {
        channel_id: channel_id.to_owned(),
        physical_boundary_id: "motor-shaft-01".to_owned(),
        run_id: "synthetic-run-001".to_owned(),
        asset_id: "synthetic-motor-serial-001".to_owned(),
        configuration_digest: CONFIGURATION_DIGEST.to_owned(),
        context_digest: CONTEXT_DIGEST.to_owned(),
        quantity,
        unit,
        sign_convention: SignConvention::PositiveMotoring,
        calibration_id: format!("synthetic-calibration-{channel_id}"),
        calibration_certificate_digest: format!("synthetic-calibration-certificate-{channel_id}"),
        calibration_status: CalibrationStatus::Current,
        raw_data_digest: format!("synthetic-raw-data-{channel_id}"),
        clock_source_id: "synthetic-tai-clock-01".to_owned(),
        clock_uncertainty_ns: 10,
        samples: vec![
            sample(1, ANCHOR_TIMESTAMP_NS - 20, value * 0.999, uncertainty),
            sample(2, ANCHOR_TIMESTAMP_NS, value, uncertainty),
            sample(3, ANCHOR_TIMESTAMP_NS + 20, value * 1.001, uncertainty),
        ],
    }
}

fn build_core(maximum_dc_bus_power_w: f64) -> Result<CoreResult, IntegrationError> {
    // Frozen reference: 10 kW at the rotor shaft and 3,000 rotor RPM.
    let rotor_speed_rpm = 3_000.0;
    let rotor_angular_velocity_rad_s = rotor_speed_rpm * TAU / 60.0;
    let drive_input = DriveInput {
        rotor_shaft_torque_nm: 10_000.0 / rotor_angular_velocity_rad_s,
        rotor_speed_rpm,
        dc_bus_voltage_v: 400.0,
        duration_s: 12.0,
    };
    let transmission = TransmissionKinematics {
        motor_to_rotor_speed_ratio: 5.0,
    };
    let transmission_efficiency = 0.90;

    // Derive the motor-shaft point once; the synthetic measurement samples use
    // these exact values so the later bit-exact evidence/operating-point join
    // is checking lineage rather than a rounded, hand-copied coordinate.
    let expected_motor_point = derive_motor_shaft_operating_point(
        drive_input,
        transmission,
        transmission_efficiency,
    )?;

    let manifest = RunManifest {
        run_id: "synthetic-run-001".to_owned(),
        asset_id: "synthetic-motor-serial-001".to_owned(),
        configuration_digest: CONFIGURATION_DIGEST.to_owned(),
        context_digest: CONTEXT_DIGEST.to_owned(),
        procedure_revision: "synthetic-procedure-v1".to_owned(),
        evidence_manifest_digest: EVIDENCE_MANIFEST_DIGEST.to_owned(),
        time_domain: TimeDomain::Tai,
        maximum_clock_uncertainty_ns: 100,
        maximum_sample_time_error_ns: 100,
    };
    let channels = vec![
        channel(
            "motor-torque",
            MeasurementQuantity::Torque,
            MeasurementUnit::NewtonMetre,
            expected_motor_point.motor_shaft_output_torque_nm,
            0.001,
        ),
        channel(
            "motor-speed",
            MeasurementQuantity::RotationalSpeed,
            MeasurementUnit::RevolutionsPerMinute,
            expected_motor_point.motor_speed_rpm,
            0.05,
        ),
    ];
    let synchronized = synchronize_at(
        &manifest,
        &channels,
        &["motor-torque", "motor-speed"],
        ANCHOR_TIMESTAMP_NS,
    )?;
    let observation = derive_motor_map_operating_point(
        &synchronized,
        "motor-torque",
        "motor-speed",
        "motor-shaft-01",
    )?;

    let torque = observation.torque_value_nm;
    let speed = observation.speed_value_rpm;
    let map = EfficiencyMap2D::new(
        "synthetic-motor-map-001",
        MAP_CONTENT_DIGEST,
        observation.configuration_digest.clone(),
        observation.context_digest.clone(),
        observation.evidence_manifest_digest.clone(),
        vec![torque * 0.5, torque * 1.5],
        vec![speed * 0.5, speed * 1.5],
        vec![
            MapNode {
                efficiency: 0.95,
                absolute_uncertainty_bound: 0.004,
            };
            4
        ],
        0.001,
    )?;
    let map_lookup = map.lookup(MapQuery {
        configuration_digest: &observation.configuration_digest,
        context_digest: &observation.context_digest,
        motor_torque_nm: torque,
        motor_speed_rpm: speed,
    })?;

    // Explicit adapter between independently versioned map and drive oracles.
    let evidence = MotorEfficiencyEvidence {
        map_id: map_lookup.map_id.clone(),
        map_digest: map_lookup.map_digest.clone(),
        map_model_version: map_lookup.model_version.to_owned(),
        torque_cell_lower_index: map_lookup.torque_cell_lower_index,
        speed_cell_lower_index: map_lookup.speed_cell_lower_index,
        interpolation_weights: map_lookup.interpolation_weights,
        configuration_digest: map_lookup.configuration_digest.clone(),
        context_digest: map_lookup.context_digest.clone(),
        evidence_manifest_digest: map_lookup.evidence_manifest_digest.clone(),
        motor_torque_nm: map_lookup.motor_torque_nm,
        motor_speed_rpm: map_lookup.motor_speed_rpm,
        efficiency_lower_bound: map_lookup.efficiency_lower_bound,
        efficiency_nominal: map_lookup.efficiency,
        efficiency_upper_bound: map_lookup.efficiency_upper_bound,
    };
    let drive_config = DriveConfig {
        transmission_efficiency,
        motor_efficiency: map_lookup.efficiency,
        inverter_efficiency: 0.98,
        minimum_dc_bus_voltage_v: 350.0,
        maximum_dc_bus_current_a: 500.0,
        maximum_dc_bus_power_w,
        energy_balance_tolerance_j: 1.0e-5,
    };
    let drive_envelope = evaluate_drive_with_map_evidence(
        &drive_config,
        drive_input,
        transmission,
        DriveMapExpectedContext {
            configuration_digest: &observation.configuration_digest,
            context_digest: &observation.context_digest,
        },
        &evidence,
    )?;

    Ok(CoreResult {
        manifest,
        observation,
        map,
        map_lookup,
        evidence,
        drive_config,
        drive_input,
        transmission,
        drive_envelope,
    })
}

/// Run a deterministic, synthetic one-substep evidence-composition fixture.
///
/// The frozen case is 10 kW rotor-shaft demand at 3,000 RPM, a 5:1
/// motor-to-rotor speed ratio, 90% transmission efficiency, a context-bound
/// motor map, 98% inverter efficiency, a 400 V DC bus, and 12 seconds.
///
/// Electrical meter loads and thermal heat-source allocations are separate:
/// conversion losses appear once in electrical demand and are independently
/// allocated (or left explicitly unallocated) in the thermal ledger.
pub fn run_synthetic_fixture() -> Result<IntegrationReport, IntegrationError> {
    let core = build_core(100_000.0)?;
    let adverse_drive = *core
        .drive_envelope
        .adverse_efficiency_scenario
        .as_ref()
        .map_err(|error| IntegrationError::Drive(error.clone()))?;

    let evidence_digest = core.manifest.evidence_manifest_digest.clone();
    let electrical_loads = evaluate_electrical_loads(
        &[
            ElectricalLoad {
                load_id: "synthetic-propulsion-dc".to_owned(),
                meter_boundary_id: "synthetic-propulsion-dc-boundary".to_owned(),
                kind: ElectricalLoadKind::PropulsionDcInput,
                power_w: adverse_drive.dc_bus_input_power_w,
                evidence_manifest_digest: evidence_digest.clone(),
            },
            ElectricalLoad {
                load_id: "synthetic-tms".to_owned(),
                meter_boundary_id: "synthetic-tms-boundary".to_owned(),
                kind: ElectricalLoadKind::ThermalManagement,
                power_w: 250.0,
                evidence_manifest_digest: "synthetic-tms-evidence-sha256".to_owned(),
            },
            ElectricalLoad {
                load_id: "synthetic-auxiliary".to_owned(),
                meter_boundary_id: "synthetic-auxiliary-boundary".to_owned(),
                kind: ElectricalLoadKind::OtherAuxiliary,
                power_w: 100.0,
                evidence_manifest_digest: "synthetic-auxiliary-evidence-sha256".to_owned(),
            },
            ElectricalLoad {
                load_id: "synthetic-distribution-loss".to_owned(),
                meter_boundary_id: "synthetic-distribution-loss-boundary".to_owned(),
                kind: ElectricalLoadKind::UpstreamDistributionLoss,
                power_w: 50.0,
                evidence_manifest_digest: "synthetic-distribution-evidence-sha256".to_owned(),
            },
        ],
        core.drive_input.duration_s,
    )?;

    let thermal_allocation = allocate_loss_heat(&[
        HeatLossSource {
            source_id: "synthetic-transmission-loss".to_owned(),
            component_id: "synthetic-gearbox-01".to_owned(),
            kind: HeatLossKind::Transmission,
            loss_power_w: adverse_drive.transmission_loss_w,
            evidence_manifest_digest: evidence_digest.clone(),
            allocations: vec![ThermalNodeAllocation {
                thermal_node_id: "gearbox-housing".to_owned(),
                fraction: 0.80,
            }],
        },
        HeatLossSource {
            source_id: "synthetic-motor-loss".to_owned(),
            component_id: "synthetic-motor-01".to_owned(),
            kind: HeatLossKind::Motor,
            loss_power_w: adverse_drive.motor_loss_w,
            evidence_manifest_digest: evidence_digest.clone(),
            allocations: vec![ThermalNodeAllocation {
                thermal_node_id: "motor-case".to_owned(),
                fraction: 0.90,
            }],
        },
        HeatLossSource {
            source_id: "synthetic-inverter-loss".to_owned(),
            component_id: "synthetic-inverter-01".to_owned(),
            kind: HeatLossKind::Inverter,
            loss_power_w: adverse_drive.inverter_loss_w,
            evidence_manifest_digest: evidence_digest,
            allocations: vec![ThermalNodeAllocation {
                thermal_node_id: "inverter-baseplate".to_owned(),
                fraction: 0.70,
            }],
        },
    ])?;

    Ok(IntegrationReport {
        motor_observation: core.observation,
        map_lookup: core.map_lookup,
        drive_envelope: core.drive_envelope,
        electrical_loads,
        thermal_allocation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_fixture_preserves_evidence_and_closes_both_ledgers() {
        let report = run_synthetic_fixture().unwrap();
        let adverse = report
            .drive_envelope
            .adverse_efficiency_scenario
            .as_ref()
            .unwrap();
        let nominal = report
            .drive_envelope
            .nominal_efficiency_scenario
            .as_ref()
            .unwrap();
        let favourable = report
            .drive_envelope
            .favourable_efficiency_scenario
            .as_ref()
            .unwrap();

        assert!((adverse.rotor_shaft_power_w - 10_000.0).abs() < 1.0e-8);
        assert!((report.motor_observation.shaft_power_w - 10_000.0 / 0.90).abs() < 1.0e-8);
        assert!((report.drive_envelope.motor_operating_point.motor_speed_rpm - 15_000.0).abs()
            < 1.0e-8);
        assert!(adverse.dc_bus_input_power_w > nominal.dc_bus_input_power_w);
        assert!(nominal.dc_bus_input_power_w > favourable.dc_bus_input_power_w);

        assert_eq!(
            report.motor_observation.configuration_digest,
            report.map_lookup.configuration_digest
        );
        assert_eq!(
            report.motor_observation.context_digest,
            report.map_lookup.context_digest
        );
        assert_eq!(
            report.motor_observation.evidence_manifest_digest,
            report.map_lookup.evidence_manifest_digest
        );
        assert_eq!(report.map_lookup.map_digest, MAP_CONTENT_DIGEST);
        assert_eq!(
            report.drive_envelope.map_digest,
            report.map_lookup.map_digest
        );
        assert_eq!(
            report.drive_envelope.evidence_manifest_digest,
            report.motor_observation.evidence_manifest_digest
        );

        assert!((adverse.dc_bus_input_energy_j
            - adverse.dc_bus_input_power_w * 12.0)
            .abs()
            < 1.0e-7);
        assert!(adverse.energy_balance_residual_j.abs() < 1.0e-5);

        assert_eq!(
            report.electrical_loads.propulsion_dc_input_power_w,
            Some(adverse.dc_bus_input_power_w)
        );
        assert_eq!(
            report.electrical_loads.thermal_management_power_w,
            Some(250.0)
        );
        assert_eq!(
            report.electrical_loads.other_auxiliary_power_w,
            Some(100.0)
        );
        assert_eq!(
            report.electrical_loads.upstream_distribution_loss_power_w,
            Some(50.0)
        );
        assert!(
            (report.electrical_loads.battery_terminal_power_w
                - (adverse.dc_bus_input_power_w + 400.0))
                .abs()
                < 1.0e-8
        );
        assert!(
            (report.electrical_loads.battery_terminal_energy_j
                - report.electrical_loads.battery_terminal_power_w * 12.0)
                .abs()
                < 1.0e-7
        );

        assert!(
            (report.thermal_allocation.total_declared_loss_power_w
                - adverse.total_conversion_loss_w)
                .abs()
                < 1.0e-8
        );
        assert!(
            (report.thermal_allocation.total_allocated_heat_power_w
                + report.thermal_allocation.total_unallocated_loss_power_w
                - report.thermal_allocation.total_declared_loss_power_w)
                .abs()
                < 1.0e-8
        );
        assert!(report.thermal_allocation.total_unallocated_loss_power_w > 0.0);
        assert_eq!(report.thermal_allocation.sources.len(), 3);
        assert!(!report
            .thermal_allocation
            .sources
            .iter()
            .any(|source| source.kind == HeatLossKind::ThermalManagementInternal));
    }

    #[test]
    fn map_oracle_rejects_wrong_context_and_out_of_domain_coordinates() {
        let core = build_core(100_000.0).unwrap();
        let context_mismatch = core.map.lookup(MapQuery {
            configuration_digest: CONFIGURATION_DIGEST,
            context_digest: "different-context",
            motor_torque_nm: core.observation.torque_value_nm,
            motor_speed_rpm: core.observation.speed_value_rpm,
        });
        assert_eq!(context_mismatch, Err(MapError::ContextMismatch));

        let out_of_domain = core.map.lookup(MapQuery {
            configuration_digest: CONFIGURATION_DIGEST,
            context_digest: CONTEXT_DIGEST,
            motor_torque_nm: core.observation.torque_value_nm * 2.0,
            motor_speed_rpm: core.observation.speed_value_rpm,
        });
        assert!(matches!(
            out_of_domain,
            Err(MapError::OutOfDomain { axis: "torque", .. })
        ));
    }

    #[test]
    fn drive_consumer_rejects_mismatched_configuration_before_evaluation() {
        let core = build_core(100_000.0).unwrap();
        let result = evaluate_drive_with_map_evidence(
            &core.drive_config,
            core.drive_input,
            core.transmission,
            DriveMapExpectedContext {
                configuration_digest: "different-aircraft-configuration",
                context_digest: CONTEXT_DIGEST,
            },
            &core.evidence,
        );
        assert!(matches!(
            result,
            Err(DriveError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn adverse_map_scenario_failure_is_not_hidden_by_nominal_feasibility() {
        // The 0.5% absolute efficiency bound makes adverse demand exceed this
        // limit while nominal and favourable demand remain below it.
        let core = build_core(11_960.0).unwrap();
        assert!(matches!(
            core.drive_envelope.adverse_efficiency_scenario,
            Err(DriveError::DcBusPowerLimitExceeded { .. })
        ));
        assert!(core.drive_envelope.nominal_efficiency_scenario.is_ok());
        assert!(core.drive_envelope.favourable_efficiency_scenario.is_ok());
    }
}
