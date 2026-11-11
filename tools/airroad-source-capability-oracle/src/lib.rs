// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Read-only join between a shared aircraft DC-bus demand report and an
//! evidence-bound, state-conditioned source capability profile.
//!
//! Battery physics and capability generation remain upstream. This crate
//! validates point identity, interval coverage, supplied limits and reserve;
//! it does not calculate OCV, resistance, voltage sag or SOC evolution.

#![forbid(unsafe_code)]

use electric_rotorcraft_drive_oracle::{
    DcBusMissionReport, DcBusReport, DcBusScenarioSummary, DcBusSegmentResult,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const MODEL_VERSION: &str = "airroad-source-capability-oracle-v0.1.0";

/// Ordered non-probabilistic lower/nominal/upper bounds for an energy quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyBounds {
    pub lower_j: f64,
    pub nominal_j: f64,
    pub upper_j: f64,
}

impl EnergyBounds {
    fn validate(self) -> bool {
        self.lower_j.is_finite()
            && self.nominal_j.is_finite()
            && self.upper_j.is_finite()
            && self.lower_j >= 0.0
            && self.lower_j <= self.nominal_j
            && self.nominal_j <= self.upper_j
    }
}

/// Source energy available above the hard BMS floor and protected at mission end.
/// These values are outputs from an upstream battery capability model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceEnergyBudget {
    pub usable_energy_above_hard_floor: EnergyBounds,
    pub protected_end_reserve: EnergyBounds,
}

/// One scenario-specific source capability evaluation at an exact bus operating
/// point and exact mission-segment duration.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCapabilityCase {
    pub assessed_bus_power_w: f64,
    pub assessed_bus_current_a: f64,
    pub assessed_bus_voltage_v: f64,
    pub maximum_deliverable_power_w: f64,
    pub maximum_deliverable_current_a: f64,
    pub minimum_deliverable_voltage_v: f64,
    pub valid_for_duration_s: f64,
    pub evidence_manifest_digest: String,
}

/// State-conditioned source capability for one exact mission interval.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCapabilityInterval {
    pub segment_id: String,
    pub start_timestamp_ns: u64,
    pub end_timestamp_ns: u64,
    pub operating_context_digest: String,
    pub source_state_digest: String,
    pub adverse: SourceCapabilityCase,
    pub nominal: SourceCapabilityCase,
    pub favourable: SourceCapabilityCase,
}

/// Evidence-bound capability profile for one source attached to one DC bus.
/// v0.1 permits one declared source per bus and rejects duplicate source IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCapabilityProfile {
    pub source_id: String,
    pub battery_instance_id: String,
    pub bus_id: String,
    pub source_boundary_id: String,
    pub aircraft_configuration_digest: String,
    pub as_installed_pack_configuration_digest: String,
    pub mission_profile_digest: String,
    pub source_model_id: String,
    pub source_model_version: String,
    pub source_model_output_digest: String,
    pub source_evidence_manifest_digest: String,
    pub source_state_manifest_digest: String,
    pub energy_evidence_manifest_digest: String,
    pub energy_budget: SourceEnergyBudget,
    pub intervals: Vec<SourceCapabilityInterval>,
}

/// Inputs expected to describe one exact aircraft mission / battery configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCapabilityJoinInput {
    pub aircraft_configuration_digest: String,
    pub mission_profile_digest: String,
    pub bus_demand: DcBusMissionReport,
    pub source_profiles: Vec<SourceCapabilityProfile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityScenario {
    Adverse,
    Nominal,
    Favourable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceCapabilityDisposition {
    FeasibleWithinSuppliedCapability,
    Infeasible,
    UnknownMissingCapability,
    NonPassIdentityMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SourceCapabilityFailureKind {
    UpstreamBusScenarioInfeasible,
    DemandExceedsPowerCapability {
        demanded_w: f64,
        available_w: f64,
    },
    DemandExceedsCurrentCapability {
        demanded_a: f64,
        available_a: f64,
    },
    DeliverableVoltageBelowRequired {
        deliverable_v: f64,
        required_v: f64,
    },
    MissionEnergyViolatesProtectedReserve {
        demanded_energy_j: f64,
        usable_energy_j: f64,
        protected_reserve_j: f64,
    },
    OperatingPointMismatch(&'static str),
    InvalidCapability(&'static str),
    MissingCapability(&'static str),
    IdentityMismatch(&'static str),
    NumericalOverflow,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceCapabilityFailure {
    pub bus_id: String,
    pub segment_id: Option<String>,
    pub scenario: CapabilityScenario,
    pub kind: SourceCapabilityFailureKind,
}

/// The bus operating point and corresponding source limits that were checked.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCapabilityIntervalReport {
    pub segment_id: String,
    pub duration_s: f64,
    pub demanded_power_w: f64,
    pub demanded_current_a: f64,
    pub required_bus_voltage_v: f64,
    pub maximum_deliverable_power_w: f64,
    pub maximum_deliverable_current_a: f64,
    pub minimum_deliverable_voltage_v: f64,
    pub power_margin_w: f64,
    pub current_margin_a: f64,
    pub voltage_margin_v: f64,
    pub bus_energy_j: f64,
    pub source_state_digest: String,
    pub operating_context_digest: String,
    pub evidence_manifest_digest: String,
}

/// Result for one bus and one rectangular uncertainty scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCapabilityScenarioReport {
    pub scenario: CapabilityScenario,
    pub disposition: SourceCapabilityDisposition,
    pub demanded_energy_j: Option<f64>,
    pub usable_energy_j: Option<f64>,
    pub protected_reserve_j: Option<f64>,
    pub remaining_energy_margin_j: Option<f64>,
    pub interval_reports: Vec<SourceCapabilityIntervalReport>,
    pub failures: Vec<SourceCapabilityFailure>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceBusCapabilityReport {
    pub bus_id: String,
    pub source_id: Option<String>,
    pub battery_instance_id: Option<String>,
    pub source_boundary_id: String,
    pub bus_source_evidence_manifest_digest: String,
    pub as_installed_pack_configuration_digest: Option<String>,
    pub source_evidence_manifest_digest: Option<String>,
    pub source_state_manifest_digest: Option<String>,
    pub energy_evidence_manifest_digest: Option<String>,
    pub source_model_id: Option<String>,
    pub source_model_version: Option<String>,
    pub source_model_output_digest: Option<String>,
    pub adverse: SourceCapabilityScenarioReport,
    pub nominal: SourceCapabilityScenarioReport,
    pub favourable: SourceCapabilityScenarioReport,
}

/// Complete source-capability qualification report for the exact bus mission.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCapabilityJoinReport {
    pub model_version: &'static str,
    pub aircraft_configuration_digest: String,
    pub mission_profile_digest: String,
    pub bus_demand_model_version: &'static str,
    pub buses: Vec<SourceBusCapabilityReport>,
}

/// Structural failures that prevent a deterministic source-profile join.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceCapabilityJoinError {
    InvalidInput(&'static str),
    DuplicateIdentity(&'static str),
    UnexpectedBus(String),
    InvalidBusDemandConfiguration(String),
}

impl fmt::Display for SourceCapabilityJoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SourceCapabilityJoinError {}

fn same_f64(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits()
}

fn empty_scenario_report(
    scenario: CapabilityScenario,
    disposition: SourceCapabilityDisposition,
    bus_id: &str,
    kind: SourceCapabilityFailureKind,
) -> SourceCapabilityScenarioReport {
    SourceCapabilityScenarioReport {
        scenario,
        disposition,
        demanded_energy_j: None,
        usable_energy_j: None,
        protected_reserve_j: None,
        remaining_energy_margin_j: None,
        interval_reports: Vec::new(),
        failures: vec![SourceCapabilityFailure {
            bus_id: bus_id.to_owned(),
            segment_id: None,
            scenario,
            kind,
        }],
    }
}

fn scenario_summary(
    bus: &DcBusReport,
    scenario: CapabilityScenario,
) -> &Result<DcBusScenarioSummary, Vec<electric_rotorcraft_drive_oracle::DcBusScenarioFailure>> {
    match scenario {
        CapabilityScenario::Adverse => &bus.adverse_scenario,
        CapabilityScenario::Nominal => &bus.nominal_scenario,
        CapabilityScenario::Favourable => &bus.favourable_scenario,
    }
}

fn capability_case(
    interval: &SourceCapabilityInterval,
    scenario: CapabilityScenario,
) -> &SourceCapabilityCase {
    match scenario {
        CapabilityScenario::Adverse => &interval.adverse,
        CapabilityScenario::Nominal => &interval.nominal,
        CapabilityScenario::Favourable => &interval.favourable,
    }
}

fn energy_bounds_for(
    profile: &SourceCapabilityProfile,
    scenario: CapabilityScenario,
) -> (f64, f64) {
    let usable = profile.energy_budget.usable_energy_above_hard_floor;
    let reserve = profile.energy_budget.protected_end_reserve;
    match scenario {
        CapabilityScenario::Adverse => (usable.lower_j, reserve.upper_j),
        CapabilityScenario::Nominal => (usable.nominal_j, reserve.nominal_j),
        CapabilityScenario::Favourable => (usable.upper_j, reserve.lower_j),
    }
}

fn scenario_reference_segments(bus: &DcBusReport) -> Option<&[DcBusSegmentResult]> {
    [
        &bus.adverse_scenario,
        &bus.nominal_scenario,
        &bus.favourable_scenario,
    ]
    .into_iter()
    .find_map(|result| result.as_ref().ok().map(|summary| summary.segments.as_slice()))
}

fn validate_profile_identity(
    input: &SourceCapabilityJoinInput,
    bus: &DcBusReport,
    profile: &SourceCapabilityProfile,
) -> Result<(), &'static str> {
    if profile.aircraft_configuration_digest != input.aircraft_configuration_digest
        || bus.configuration_digest != input.aircraft_configuration_digest
        || bus.source_evidence_manifest_digest.trim().is_empty()
        || profile.mission_profile_digest != input.mission_profile_digest
        || profile.bus_id != bus.bus_id
        || profile.source_boundary_id != bus.source_boundary_id
        || profile.source_model_id.trim().is_empty()
        || profile.source_model_version.trim().is_empty()
        || profile.source_model_output_digest.trim().is_empty()
        || profile.source_evidence_manifest_digest.trim().is_empty()
        || profile.source_state_manifest_digest.trim().is_empty()
        || profile.energy_evidence_manifest_digest.trim().is_empty()
        || profile.source_id.trim().is_empty()
        || profile.battery_instance_id.trim().is_empty()
        || profile.as_installed_pack_configuration_digest.trim().is_empty()
    {
        return Err("source, bus, model, mission, pack or evidence identity mismatch");
    }
    if !profile.energy_budget.usable_energy_above_hard_floor.validate()
        || !profile.energy_budget.protected_end_reserve.validate()
    {
        return Err(
            "usable energy and protected reserve bounds must be finite, non-negative and ordered",
        );
    }
    Ok(())
}

fn validate_case(
    case: &SourceCapabilityCase,
    segment: &DcBusSegmentResult,
    scenario: CapabilityScenario,
    bus_id: &str,
) -> Result<(), SourceCapabilityFailure> {
    let failure = |kind| SourceCapabilityFailure {
        bus_id: bus_id.to_owned(),
        segment_id: Some(segment.segment_id.clone()),
        scenario,
        kind,
    };

    if case.evidence_manifest_digest.trim().is_empty() {
        return Err(failure(SourceCapabilityFailureKind::MissingCapability(
            "per-interval capability evidence digest is absent",
        )));
    }
    let values = [
        case.assessed_bus_power_w,
        case.assessed_bus_current_a,
        case.assessed_bus_voltage_v,
        case.maximum_deliverable_power_w,
        case.maximum_deliverable_current_a,
        case.minimum_deliverable_voltage_v,
        case.valid_for_duration_s,
    ];
    if values.iter().any(|value| !value.is_finite())
        || case.assessed_bus_power_w < 0.0
        || case.assessed_bus_current_a < 0.0
        || case.assessed_bus_voltage_v <= 0.0
        || case.maximum_deliverable_power_w <= 0.0
        || case.maximum_deliverable_current_a <= 0.0
        || case.minimum_deliverable_voltage_v <= 0.0
        || case.valid_for_duration_s <= 0.0
    {
        return Err(failure(SourceCapabilityFailureKind::InvalidCapability(
            "source capability values must be finite and in their physical domains",
        )));
    }
    if !same_f64(case.assessed_bus_power_w, segment.total_bus_power_w)
        || !same_f64(case.assessed_bus_current_a, segment.total_bus_current_a)
        || !same_f64(case.assessed_bus_voltage_v, segment.bus_voltage_v)
    {
        return Err(failure(SourceCapabilityFailureKind::OperatingPointMismatch(
            "assessed bus power/current/voltage differs from the bus report",
        )));
    }
    if case.valid_for_duration_s + f64::EPSILON < segment.duration_s {
        return Err(failure(SourceCapabilityFailureKind::MissingCapability(
            "source capability is not valid for the full mission interval duration",
        )));
    }
    Ok(())
}

fn assess_scenario(
    bus: &DcBusReport,
    profile: &SourceCapabilityProfile,
    reference_segments: &[DcBusSegmentResult],
    scenario: CapabilityScenario,
) -> SourceCapabilityScenarioReport {
    let Some(segments) = scenario_summary(bus, scenario)
        .as_ref()
        .ok()
        .map(|summary| summary.segments.as_slice())
    else {
        return empty_scenario_report(
            scenario,
            SourceCapabilityDisposition::Infeasible,
            &bus.bus_id,
            SourceCapabilityFailureKind::UpstreamBusScenarioInfeasible,
        );
    };
    let mut report = SourceCapabilityScenarioReport {
        scenario,
        disposition: SourceCapabilityDisposition::FeasibleWithinSuppliedCapability,
        demanded_energy_j: None,
        usable_energy_j: None,
        protected_reserve_j: None,
        remaining_energy_margin_j: None,
        interval_reports: Vec::with_capacity(segments.len()),
        failures: Vec::new(),
    };

    if profile.intervals.len() != reference_segments.len()
        || segments.len() != reference_segments.len()
    {
        return empty_scenario_report(
            scenario,
            SourceCapabilityDisposition::UnknownMissingCapability,
            &bus.bus_id,
            SourceCapabilityFailureKind::MissingCapability(
                "source capability intervals do not cover the complete bus mission schedule",
            ),
        );
    }

    for (index, segment) in segments.iter().enumerate() {
        let reference = &reference_segments[index];
        let interval = &profile.intervals[index];
        if interval.segment_id != reference.segment_id
            || interval.start_timestamp_ns != reference.start_timestamp_ns
            || interval.end_timestamp_ns != reference.end_timestamp_ns
            || interval.operating_context_digest.trim().is_empty()
            || interval.source_state_digest.trim().is_empty()
            || segment.segment_id != reference.segment_id
            || segment.start_timestamp_ns != reference.start_timestamp_ns
            || segment.end_timestamp_ns != reference.end_timestamp_ns
        {
            report.failures.push(SourceCapabilityFailure {
                bus_id: bus.bus_id.clone(),
                segment_id: Some(reference.segment_id.clone()),
                scenario,
                kind: SourceCapabilityFailureKind::IdentityMismatch(
                    "source capability interval, mission context or bus schedule is inconsistent",
                ),
            });
            continue;
        }

        let case = capability_case(interval, scenario);
        if let Err(failure) = validate_case(case, segment, scenario, &bus.bus_id) {
            report.failures.push(failure);
            continue;
        }

        let power_margin = case.maximum_deliverable_power_w - segment.total_bus_power_w;
        let current_margin = case.maximum_deliverable_current_a - segment.total_bus_current_a;
        let voltage_margin = case.minimum_deliverable_voltage_v - segment.bus_voltage_v;
        if power_margin < 0.0 {
            report.failures.push(SourceCapabilityFailure {
                bus_id: bus.bus_id.clone(),
                segment_id: Some(segment.segment_id.clone()),
                scenario,
                kind: SourceCapabilityFailureKind::DemandExceedsPowerCapability {
                    demanded_w: segment.total_bus_power_w,
                    available_w: case.maximum_deliverable_power_w,
                },
            });
        }
        if current_margin < 0.0 {
            report.failures.push(SourceCapabilityFailure {
                bus_id: bus.bus_id.clone(),
                segment_id: Some(segment.segment_id.clone()),
                scenario,
                kind: SourceCapabilityFailureKind::DemandExceedsCurrentCapability {
                    demanded_a: segment.total_bus_current_a,
                    available_a: case.maximum_deliverable_current_a,
                },
            });
        }
        if voltage_margin < 0.0 {
            report.failures.push(SourceCapabilityFailure {
                bus_id: bus.bus_id.clone(),
                segment_id: Some(segment.segment_id.clone()),
                scenario,
                kind: SourceCapabilityFailureKind::DeliverableVoltageBelowRequired {
                    deliverable_v: case.minimum_deliverable_voltage_v,
                    required_v: segment.bus_voltage_v,
                },
            });
        }

        report.interval_reports.push(SourceCapabilityIntervalReport {
            segment_id: segment.segment_id.clone(),
            duration_s: segment.duration_s,
            demanded_power_w: segment.total_bus_power_w,
            demanded_current_a: segment.total_bus_current_a,
            required_bus_voltage_v: segment.bus_voltage_v,
            maximum_deliverable_power_w: case.maximum_deliverable_power_w,
            maximum_deliverable_current_a: case.maximum_deliverable_current_a,
            minimum_deliverable_voltage_v: case.minimum_deliverable_voltage_v,
            power_margin_w: power_margin,
            current_margin_a: current_margin,
            voltage_margin_v: voltage_margin,
            bus_energy_j: segment.total_bus_energy_j,
            source_state_digest: interval.source_state_digest.clone(),
            operating_context_digest: interval.operating_context_digest.clone(),
            evidence_manifest_digest: case.evidence_manifest_digest.clone(),
        });
    }

    let demanded_energy = scenario_summary(bus, scenario)
        .as_ref()
        .map(|summary| summary.total_bus_energy_j)
        .unwrap_or(0.0);
    let (usable_energy, protected_reserve) = energy_bounds_for(profile, scenario);
    let required_energy = demanded_energy + protected_reserve;
    if !required_energy.is_finite() {
        report.failures.push(SourceCapabilityFailure {
            bus_id: bus.bus_id.clone(),
            segment_id: None,
            scenario,
            kind: SourceCapabilityFailureKind::NumericalOverflow,
        });
    } else if required_energy > usable_energy {
        report.failures.push(SourceCapabilityFailure {
            bus_id: bus.bus_id.clone(),
            segment_id: None,
            scenario,
            kind: SourceCapabilityFailureKind::MissionEnergyViolatesProtectedReserve {
                demanded_energy_j: demanded_energy,
                usable_energy_j: usable_energy,
                protected_reserve_j: protected_reserve,
            },
        });
    }

    let identity_failure = report.failures.iter().any(|failure| {
        matches!(
            &failure.kind,
            SourceCapabilityFailureKind::IdentityMismatch(_)
                | SourceCapabilityFailureKind::OperatingPointMismatch(_)
        )
    });
    let unknown_failure = report.failures.iter().any(|failure| {
        matches!(
            &failure.kind,
            SourceCapabilityFailureKind::MissingCapability(_)
                | SourceCapabilityFailureKind::InvalidCapability(_)
        )
    });
    let infeasible_failure = !report.failures.is_empty();

    report.demanded_energy_j = Some(demanded_energy);
    report.usable_energy_j = Some(usable_energy);
    report.protected_reserve_j = Some(protected_reserve);
    report.remaining_energy_margin_j = Some(usable_energy - protected_reserve - demanded_energy);
    report.disposition = if identity_failure {
        SourceCapabilityDisposition::NonPassIdentityMismatch
    } else if infeasible_failure && !unknown_failure {
        SourceCapabilityDisposition::Infeasible
    } else if unknown_failure {
        SourceCapabilityDisposition::UnknownMissingCapability
    } else {
        SourceCapabilityDisposition::FeasibleWithinSuppliedCapability
    };
    report
}

/// Join per-bus mission demand with exact, scenario-matched source capability.
///
/// The source model supplies the capability values. This function only checks
/// point identity, duration validity, power/current/voltage limits, and energy
/// plus reserve. It never computes battery electrochemistry or voltage sag.
pub fn qualify_source_capability(
    input: &SourceCapabilityJoinInput,
) -> Result<SourceCapabilityJoinReport, SourceCapabilityJoinError> {
    if input.aircraft_configuration_digest.trim().is_empty()
        || input.mission_profile_digest.trim().is_empty()
        || input.bus_demand.configuration_digest != input.aircraft_configuration_digest
        || input.bus_demand.buses.is_empty()
    {
        return Err(SourceCapabilityJoinError::InvalidInput(
            "mission/configuration identity and complete bus-demand report are required",
        ));
    }

    let mut profiles_by_bus = BTreeMap::new();
    let mut source_ids = BTreeSet::new();
    let mut battery_instances = BTreeSet::new();
    for profile in &input.source_profiles {
        if profile.bus_id.trim().is_empty()
            || profile.source_id.trim().is_empty()
            || profile.battery_instance_id.trim().is_empty()
        {
            return Err(SourceCapabilityJoinError::InvalidInput(
                "source, battery and bus identities must not be empty",
            ));
        }
        if profiles_by_bus
            .insert(profile.bus_id.clone(), profile)
            .is_some()
        {
            return Err(SourceCapabilityJoinError::DuplicateIdentity(
                "bus source profile",
            ));
        }
        if !source_ids.insert(profile.source_id.clone()) {
            return Err(SourceCapabilityJoinError::DuplicateIdentity("source ID"));
        }
        if !battery_instances.insert(profile.battery_instance_id.clone()) {
            return Err(SourceCapabilityJoinError::DuplicateIdentity("battery instance ID"));
        }
    }

    let mut known_buses = BTreeSet::new();
    for bus in &input.bus_demand.buses {
        if bus.bus_id.trim().is_empty()
            || bus.source_boundary_id.trim().is_empty()
            || bus.configuration_digest != input.aircraft_configuration_digest
            || bus.source_evidence_manifest_digest.trim().is_empty()
        {
            return Err(SourceCapabilityJoinError::InvalidBusDemandConfiguration(
                bus.bus_id.clone(),
            ));
        }
        if !known_buses.insert(bus.bus_id.as_str()) {
            return Err(SourceCapabilityJoinError::DuplicateIdentity("bus ID"));
        }
    }
    for profile in &input.source_profiles {
        if !known_buses.contains(profile.bus_id.as_str()) {
            return Err(SourceCapabilityJoinError::UnexpectedBus(profile.bus_id.clone()));
        }
    }

    let mut buses = input.bus_demand.buses.clone();
    buses.sort_by(|left, right| left.bus_id.cmp(&right.bus_id));
    let mut reports = Vec::with_capacity(buses.len());

    for bus in &buses {
        let Some(profile) = profiles_by_bus.get(&bus.bus_id).copied() else {
            let missing = |scenario| empty_scenario_report(
                scenario,
                SourceCapabilityDisposition::UnknownMissingCapability,
                &bus.bus_id,
                SourceCapabilityFailureKind::MissingCapability(
                    "no state-conditioned source profile is supplied for this bus",
                ),
            );
            reports.push(SourceBusCapabilityReport {
                bus_id: bus.bus_id.clone(),
                source_id: None,
                battery_instance_id: None,
                source_boundary_id: bus.source_boundary_id.clone(),
                bus_source_evidence_manifest_digest: bus.source_evidence_manifest_digest.clone(),
                as_installed_pack_configuration_digest: None,
                source_evidence_manifest_digest: None,
                source_state_manifest_digest: None,
                energy_evidence_manifest_digest: None,
                source_model_id: None,
                source_model_version: None,
                source_model_output_digest: None,
                adverse: missing(CapabilityScenario::Adverse),
                nominal: missing(CapabilityScenario::Nominal),
                favourable: missing(CapabilityScenario::Favourable),
            });
            continue;
        };

        let identity_error = validate_profile_identity(input, bus, profile);
        if let Err(reason) = identity_error {
            let non_pass = |scenario| empty_scenario_report(
                scenario,
                SourceCapabilityDisposition::NonPassIdentityMismatch,
                &bus.bus_id,
                SourceCapabilityFailureKind::IdentityMismatch(reason),
            );
            reports.push(SourceBusCapabilityReport {
                bus_id: bus.bus_id.clone(),
                source_id: Some(profile.source_id.clone()),
                battery_instance_id: Some(profile.battery_instance_id.clone()),
                source_boundary_id: profile.source_boundary_id.clone(),
                bus_source_evidence_manifest_digest: bus.source_evidence_manifest_digest.clone(),
                as_installed_pack_configuration_digest: Some(
                    profile.as_installed_pack_configuration_digest.clone(),
                ),
                source_evidence_manifest_digest: Some(
                    profile.source_evidence_manifest_digest.clone(),
                ),
                source_state_manifest_digest: Some(profile.source_state_manifest_digest.clone()),
                energy_evidence_manifest_digest: Some(
                    profile.energy_evidence_manifest_digest.clone(),
                ),
                source_model_id: Some(profile.source_model_id.clone()),
                source_model_version: Some(profile.source_model_version.clone()),
                source_model_output_digest: Some(profile.source_model_output_digest.clone()),
                adverse: non_pass(CapabilityScenario::Adverse),
                nominal: non_pass(CapabilityScenario::Nominal),
                favourable: non_pass(CapabilityScenario::Favourable),
            });
            continue;
        }

        let reference_segments = scenario_reference_segments(bus);
        let (adverse, nominal, favourable) = if let Some(reference) = reference_segments {
            (
                assess_scenario(bus, profile, reference, CapabilityScenario::Adverse),
                assess_scenario(bus, profile, reference, CapabilityScenario::Nominal),
                assess_scenario(bus, profile, reference, CapabilityScenario::Favourable),
            )
        } else {
            let failed = |scenario| empty_scenario_report(
                scenario,
                SourceCapabilityDisposition::Infeasible,
                &bus.bus_id,
                SourceCapabilityFailureKind::UpstreamBusScenarioInfeasible,
            );
            (
                failed(CapabilityScenario::Adverse),
                failed(CapabilityScenario::Nominal),
                failed(CapabilityScenario::Favourable),
            )
        };

        reports.push(SourceBusCapabilityReport {
            bus_id: bus.bus_id.clone(),
            source_id: Some(profile.source_id.clone()),
            battery_instance_id: Some(profile.battery_instance_id.clone()),
            source_boundary_id: profile.source_boundary_id.clone(),
            bus_source_evidence_manifest_digest: bus.source_evidence_manifest_digest.clone(),
            as_installed_pack_configuration_digest: Some(
                profile.as_installed_pack_configuration_digest.clone(),
            ),
            source_evidence_manifest_digest: Some(profile.source_evidence_manifest_digest.clone()),
            source_state_manifest_digest: Some(profile.source_state_manifest_digest.clone()),
            energy_evidence_manifest_digest: Some(profile.energy_evidence_manifest_digest.clone()),
            source_model_id: Some(profile.source_model_id.clone()),
            source_model_version: Some(profile.source_model_version.clone()),
            source_model_output_digest: Some(profile.source_model_output_digest.clone()),
            adverse,
            nominal,
            favourable,
        });
    }

    Ok(SourceCapabilityJoinReport {
        model_version: MODEL_VERSION,
        aircraft_configuration_digest: input.aircraft_configuration_digest.clone(),
        mission_profile_digest: input.mission_profile_digest.clone(),
        bus_demand_model_version: input.bus_demand.model_version,
        buses: reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use electric_rotorcraft_drive_oracle::{
        DcBusReport, DcBusScenarioSummary, DcBusSegmentResult,
    };

    fn segment(
        id: &str,
        start_ns: u64,
        end_ns: u64,
        power_w: f64,
        current_a: f64,
        voltage_v: f64,
    ) -> DcBusSegmentResult {
        let duration_s = (end_ns - start_ns) as f64 / 1_000_000_000.0;
        DcBusSegmentResult {
            segment_id: id.to_owned(),
            start_timestamp_ns: start_ns,
            end_timestamp_ns: end_ns,
            duration_s,
            bus_voltage_v: voltage_v,
            propulsion_power_w: power_w,
            external_load_power_w: 0.0,
            total_bus_power_w: power_w,
            total_bus_current_a: current_a,
            total_bus_energy_j: power_w * duration_s,
            branches: Vec::new(),
            external_loads: Vec::new(),
        }
    }

    fn bus_summary(segments: Vec<DcBusSegmentResult>) -> DcBusScenarioSummary {
        let total_energy_j = segments.iter().map(|segment| segment.total_bus_energy_j).sum();
        let active_duration_s = segments.iter().map(|segment| segment.duration_s).sum();
        DcBusScenarioSummary {
            segment_count: segments.len(),
            active_duration_s,
            propulsion_energy_j: total_energy_j,
            external_load_energy_j: 0.0,
            total_bus_energy_j: total_energy_j,
            average_bus_power_w: total_energy_j / active_duration_s,
            peak_bus_power_w: segments
                .iter()
                .map(|segment| segment.total_bus_power_w)
                .fold(0.0, f64::max),
            peak_bus_current_a: segments
                .iter()
                .map(|segment| segment.total_bus_current_a)
                .fold(0.0, f64::max),
            segments,
        }
    }

    fn bus_report() -> DcBusMissionReport {
        let adverse = bus_summary(vec![
            segment("climb", 0, 10_000_000_000, 600.0, 1.5, 400.0),
            segment("cruise", 10_000_000_000, 30_000_000_000, 300.0, 0.75, 400.0),
        ]);
        let nominal = bus_summary(vec![
            segment("climb", 0, 10_000_000_000, 500.0, 1.25, 400.0),
            segment("cruise", 10_000_000_000, 30_000_000_000, 250.0, 0.625, 400.0),
        ]);
        let favourable = bus_summary(vec![
            segment("climb", 0, 10_000_000_000, 400.0, 1.0, 400.0),
            segment("cruise", 10_000_000_000, 30_000_000_000, 200.0, 0.5, 400.0),
        ]);
        DcBusMissionReport {
            model_version: "electric-rotorcraft-drive-oracle-v0.7.0",
            configuration_digest: "aircraft-config".to_owned(),
            buses: vec![DcBusReport {
                bus_id: "main-bus".to_owned(),
                source_boundary_id: "battery-main-terminals".to_owned(),
                source_evidence_manifest_digest: "bus-source-evidence".to_owned(),
                configuration_digest: "aircraft-config".to_owned(),
                branch_ids: vec!["left".to_owned(), "right".to_owned()],
                adverse_scenario: Ok(adverse),
                nominal_scenario: Ok(nominal),
                favourable_scenario: Ok(favourable),
            }],
        }
    }

    fn source_case(
        segment: &DcBusSegmentResult,
        max_power_w: f64,
        max_current_a: f64,
        min_voltage_v: f64,
    ) -> SourceCapabilityCase {
        SourceCapabilityCase {
            assessed_bus_power_w: segment.total_bus_power_w,
            assessed_bus_current_a: segment.total_bus_current_a,
            assessed_bus_voltage_v: segment.bus_voltage_v,
            maximum_deliverable_power_w: max_power_w,
            maximum_deliverable_current_a: max_current_a,
            minimum_deliverable_voltage_v: min_voltage_v,
            valid_for_duration_s: segment.duration_s,
            evidence_manifest_digest: format!("capability-{}", segment.segment_id),
        }
    }

    fn source_interval(
        adverse_segment: &DcBusSegmentResult,
        nominal_segment: &DcBusSegmentResult,
        favourable_segment: &DcBusSegmentResult,
    ) -> SourceCapabilityInterval {
        SourceCapabilityInterval {
            segment_id: nominal_segment.segment_id.clone(),
            start_timestamp_ns: nominal_segment.start_timestamp_ns,
            end_timestamp_ns: nominal_segment.end_timestamp_ns,
            operating_context_digest: format!("pack-context-{}", nominal_segment.segment_id),
            source_state_digest: format!("battery-state-{}", nominal_segment.segment_id),
            adverse: source_case(adverse_segment, 800.0, 2.0, 395.0),
            nominal: source_case(nominal_segment, 700.0, 1.8, 395.0),
            favourable: source_case(favourable_segment, 650.0, 1.7, 395.0),
        }
    }

    fn profile(bus: &DcBusReport) -> SourceCapabilityProfile {
        let adverse = bus.adverse_scenario.as_ref().unwrap();
        let nominal = bus.nominal_scenario.as_ref().unwrap();
        let favourable = bus.favourable_scenario.as_ref().unwrap();
        SourceCapabilityProfile {
            source_id: format!("source-{}", bus.bus_id),
            battery_instance_id: format!("pack-{}", bus.bus_id),
            bus_id: bus.bus_id.clone(),
            source_boundary_id: bus.source_boundary_id.clone(),
            aircraft_configuration_digest: "aircraft-config".to_owned(),
            as_installed_pack_configuration_digest: "pack-config-v7".to_owned(),
            mission_profile_digest: "mission-profile-sha256".to_owned(),
            source_model_id: "qualified-battery-capability-model".to_owned(),
            source_model_version: "battery-capability-model-v3".to_owned(),
            source_model_output_digest: "source-output-digest".to_owned(),
            source_evidence_manifest_digest: "source-evidence-digest".to_owned(),
            source_state_manifest_digest: "source-state-manifest".to_owned(),
            energy_evidence_manifest_digest: "energy-budget-manifest".to_owned(),
            energy_budget: SourceEnergyBudget {
                usable_energy_above_hard_floor: EnergyBounds {
                    lower_j: 20_000.0,
                    nominal_j: 24_000.0,
                    upper_j: 28_000.0,
                },
                protected_end_reserve: EnergyBounds {
                    lower_j: 2_000.0,
                    nominal_j: 3_000.0,
                    upper_j: 4_000.0,
                },
            },
            intervals: vec![
                source_interval(
                    &adverse.segments[0],
                    &nominal.segments[0],
                    &favourable.segments[0],
                ),
                source_interval(
                    &adverse.segments[1],
                    &nominal.segments[1],
                    &favourable.segments[1],
                ),
            ],
        }
    }

    fn input() -> SourceCapabilityJoinInput {
        let bus_demand = bus_report();
        let source_profiles = vec![profile(&bus_demand.buses[0])];
        SourceCapabilityJoinInput {
            aircraft_configuration_digest: "aircraft-config".to_owned(),
            mission_profile_digest: "mission-profile-sha256".to_owned(),
            bus_demand,
            source_profiles,
        }
    }

    #[test]
    fn matched_source_capability_passes_all_three_scenarios() {
        let report = qualify_source_capability(&input()).unwrap();
        let bus = &report.buses[0];
        assert_eq!(
            bus.adverse.disposition,
            SourceCapabilityDisposition::FeasibleWithinSuppliedCapability
        );
        assert_eq!(
            bus.nominal.disposition,
            SourceCapabilityDisposition::FeasibleWithinSuppliedCapability
        );
        assert_eq!(
            bus.favourable.disposition,
            SourceCapabilityDisposition::FeasibleWithinSuppliedCapability
        );
        assert_eq!(bus.nominal.interval_reports.len(), 2);
    }

    #[test]
    fn cold_or_aged_adverse_source_capability_can_fail_while_nominal_passes() {
        let mut input = input();
        let profile = &mut input.source_profiles[0];
        profile.intervals[0].adverse.maximum_deliverable_power_w = 500.0;
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(report.buses[0].adverse.disposition, SourceCapabilityDisposition::Infeasible);
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::FeasibleWithinSuppliedCapability
        );
    }

    #[test]
    fn bus_current_and_power_limits_are_checked_separately() {
        let mut input = input();
        input.source_profiles[0].intervals[0].nominal.maximum_deliverable_power_w = 400.0;
        input.source_profiles[0].intervals[0].nominal.maximum_deliverable_current_a = 1.0;
        let report = qualify_source_capability(&input).unwrap();
        let failures = &report.buses[0].nominal.failures;
        assert!(failures.iter().any(|failure| matches!(
            &failure.kind,
            SourceCapabilityFailureKind::DemandExceedsPowerCapability { .. }
        )));
        assert!(failures.iter().any(|failure| matches!(
            &failure.kind,
            SourceCapabilityFailureKind::DemandExceedsCurrentCapability { .. }
        )));
    }

    #[test]
    fn minimum_deliverable_terminal_voltage_is_checked_at_the_requested_bus_voltage() {
        let mut input = input();
        input.source_profiles[0].intervals[0].nominal.minimum_deliverable_voltage_v = 390.0;
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::Infeasible
        );
        assert!(report.buses[0].nominal.failures.iter().any(|failure| matches!(
            &failure.kind,
            SourceCapabilityFailureKind::DeliverableVoltageBelowRequired { .. }
        )));
    }

    #[test]
    fn mission_energy_plus_protected_reserve_must_fit_usable_energy() {
        let mut input = input();
        input.source_profiles[0].energy_budget.usable_energy_above_hard_floor.lower_j = 5_000.0;
        input.source_profiles[0].energy_budget.usable_energy_above_hard_floor.nominal_j = 5_000.0;
        input.source_profiles[0].energy_budget.usable_energy_above_hard_floor.upper_j = 5_000.0;
        input.source_profiles[0].energy_budget.protected_end_reserve.lower_j = 1_000.0;
        input.source_profiles[0].energy_budget.protected_end_reserve.nominal_j = 1_000.0;
        input.source_profiles[0].energy_budget.protected_end_reserve.upper_j = 1_000.0;
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::Infeasible
        );
        assert!(report.buses[0].nominal.failures.iter().any(|failure| matches!(
            &failure.kind,
            SourceCapabilityFailureKind::MissionEnergyViolatesProtectedReserve { .. }
        )));
    }

    #[test]
    fn missing_source_profile_is_unknown_not_static_nameplate_pass() {
        let mut input = input();
        input.source_profiles.clear();
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].adverse.disposition,
            SourceCapabilityDisposition::UnknownMissingCapability
        );
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::UnknownMissingCapability
        );
    }

    #[test]
    fn stale_source_boundary_or_configuration_is_a_non_pass() {
        let mut input = input();
        input.source_profiles[0].source_boundary_id = "other-source".to_owned();
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::NonPassIdentityMismatch
        );
        let mut input = input();
        input.source_profiles[0].aircraft_configuration_digest = "other-config".to_owned();
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::NonPassIdentityMismatch
        );
    }

    #[test]
    fn missing_interval_evidence_is_unknown_not_zero_load() {
        let mut input = input();
        input.source_profiles[0].intervals.pop();
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::UnknownMissingCapability
        );
    }

    #[test]
    fn capability_assessed_for_another_operating_point_is_a_non_pass() {
        let mut input = input();
        input.source_profiles[0].intervals[0].nominal.assessed_bus_power_w += 1.0;
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::NonPassIdentityMismatch
        );
    }

    #[test]
    fn capability_must_cover_the_entire_segment_duration() {
        let mut input = input();
        input.source_profiles[0].intervals[0].nominal.valid_for_duration_s = 5.0;
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::UnknownMissingCapability
        );
    }

    #[test]
    fn tms_energy_is_not_added_twice_to_the_bus_energy_ledger() {
        let mut input = input();
        let nominal = input.bus_demand.buses[0]
            .nominal_scenario
            .as_mut()
            .unwrap();
        nominal.external_load_energy_j = 3_000.0;
        nominal.propulsion_energy_j = nominal.total_bus_energy_j - 3_000.0;
        let nominal_bus_energy = nominal.total_bus_energy_j;
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.demanded_energy_j,
            Some(nominal_bus_energy)
        );
        assert_eq!(
            report.buses[0].nominal.remaining_energy_margin_j,
            Some(24_000.0 - 3_000.0 - nominal_bus_energy)
        );
    }

    #[test]
    fn duplicate_source_and_battery_instances_fail_closed() {
        let mut input = input();
        let mut second = input.source_profiles[0].clone();
        second.bus_id = "aux-bus".to_owned();
        input.source_profiles.push(second);
        assert_eq!(
            qualify_source_capability(&input),
            Err(SourceCapabilityJoinError::DuplicateIdentity("source ID"))
        );
    }

    #[test]
    fn unordered_energy_bounds_and_non_finite_limits_fail_closed_as_unknown() {
        let mut input = input();
        input.source_profiles[0].energy_budget.usable_energy_above_hard_floor.lower_j = 50_000.0;
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::NonPassIdentityMismatch
        );

        let mut input = input();
        input.source_profiles[0].intervals[0].nominal.maximum_deliverable_power_w = f64::NAN;
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::UnknownMissingCapability
        );
    }

    #[test]
    fn source_profiles_and_bus_reports_are_order_invariant() {
        let mut forward = input();
        let mut second_bus = forward.bus_demand.buses[0].clone();
        second_bus.bus_id = "aux-bus".to_owned();
        second_bus.source_boundary_id = "battery-aux-terminals".to_owned();
        second_bus.source_evidence_manifest_digest = "aux-bus-evidence".to_owned();
        let second_profile = profile(&second_bus);
        forward.bus_demand.buses.push(second_bus);
        forward.source_profiles.push(second_profile);

        let mut reversed = forward.clone();
        reversed.bus_demand.buses.reverse();
        reversed.source_profiles.reverse();

        let first_report = qualify_source_capability(&forward).unwrap();
        let second_report = qualify_source_capability(&reversed).unwrap();
        assert_eq!(first_report, second_report);
        assert_eq!(first_report.buses[0].bus_id, "aux-bus");
        assert_eq!(
            first_report.buses[0].source_boundary_id,
            "battery-aux-terminals"
        );
        assert_eq!(
            first_report.buses[1].battery_instance_id.as_deref(),
            Some("pack-main-bus")
        );
    }

    #[test]
    fn report_preserves_exact_bus_pack_and_source_evidence_lineage() {
        let report = qualify_source_capability(&input()).unwrap();
        let bus = &report.buses[0];
        assert_eq!(
            bus.bus_source_evidence_manifest_digest,
            "bus-source-evidence"
        );
        assert_eq!(
            bus.as_installed_pack_configuration_digest.as_deref(),
            Some("pack-config-v7")
        );
        assert_eq!(
            bus.source_evidence_manifest_digest.as_deref(),
            Some("source-evidence-digest")
        );
        assert_eq!(
            bus.source_state_manifest_digest.as_deref(),
            Some("source-state-manifest")
        );
        assert_eq!(
            bus.energy_evidence_manifest_digest.as_deref(),
            Some("energy-budget-manifest")
        );
        assert_eq!(
            bus.source_model_output_digest.as_deref(),
            Some("source-output-digest")
        );
    }

    #[test]
    fn upstream_bus_failure_cannot_be_hidden_by_a_source_profile() {
        let mut input = input();
        input.bus_demand.buses[0].nominal_scenario = Err(vec![]);
        let report = qualify_source_capability(&input).unwrap();
        assert_eq!(
            report.buses[0].nominal.disposition,
            SourceCapabilityDisposition::Infeasible
        );
        assert!(matches!(
            &report.buses[0].nominal.failures[0].kind,
            SourceCapabilityFailureKind::UpstreamBusScenarioInfeasible
        ));
    }
}
