// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mission-segment thermal feasibility overlay with temperature derating.
//!
//! This module consumes a validated mission report, reconstructs the component
//! losses at each mission segment, and drives the multi-node thermal RC network.
//! Battery, inverter, and motor power limits are checked against temperature-
//! dependent derating curves before and after every thermal substep.
//!
//! This is a thermal feasibility overlay, not yet a fully iterated
//! electro-thermal solver: updated temperatures do not feed back into the
//! battery OCV/resistance or propulsion-efficiency maps and the mission energy
//! solution is not re-solved. No thermal-runaway, cell-hot-spot, coolant-flow,
//! or airworthiness claim is made.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::thermal_network::{
    ThermalNetwork, ThermalNetworkError, ThermalState,
};
use crate::{FlightPhase, MissionInput, MissionReport, SegmentResult};

const GRAVITY_MPS2: f64 = 9.806_65;
const SECONDS_PER_HOUR: f64 = 3_600.0;
const MAX_SUBSTEPS_PER_SEGMENT: usize = 1_000_000;
const MODEL_VERSION: &str = "electric-aircraft-mission-thermal-overlay-v0.1.0";

/// Power boundary used when comparing a mission demand with a derating curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComponentPowerBasis {
    /// Total pack terminal power, including declared auxiliary load.
    BatteryTerminal,
    /// DC input power attributable to the traction inverter, excluding auxiliaries.
    InverterDcInput,
    /// Mechanical shaft output required from the motor.
    MotorShaftOutput,
}

/// One point on a temperature-dependent continuous-power limit curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalDeratingPoint {
    pub temperature_k: f64,
    pub maximum_power_w: f64,
}

/// A derating curve bound to one physical thermal node and one power boundary.
///
/// Temperatures must be strictly increasing. Power limits may be non-monotonic so
/// measured cold-side and hot-side derating can both be represented. Values
/// between measured points are linearly interpolated; temperatures outside the
/// characterized domain are rejected (no extrapolation).
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalDeratingCurve {
    pub component_id: String,
    pub node_id: String,
    pub power_basis: ComponentPowerBasis,
    pub points: Vec<ThermalDeratingPoint>,
}

impl ThermalDeratingCurve {
    pub fn validate(&self) -> Result<(), MissionThermalError> {
        if self.component_id.trim().is_empty() || self.node_id.trim().is_empty() {
            return Err(MissionThermalError::InvalidConfiguration(
                "component and thermal node identifiers must be non-empty",
            ));
        }
        if self.points.len() < 2 {
            return Err(MissionThermalError::InvalidCurve {
                component_id: self.component_id.clone(),
                reason: "at least two characterization points are required",
            });
        }
        let mut previous_temperature = f64::NEG_INFINITY;
        for point in &self.points {
            if !point.temperature_k.is_finite()
                || point.temperature_k <= 0.0
                || !point.maximum_power_w.is_finite()
                || point.maximum_power_w < 0.0
                || point.temperature_k <= previous_temperature
            {
                return Err(MissionThermalError::InvalidCurve {
                    component_id: self.component_id.clone(),
                    reason: "temperatures must increase strictly and power limits must be finite and nonnegative",
                });
            }
            previous_temperature = point.temperature_k;
        }
        Ok(())
    }

    /// Resolve the available power at a measured node temperature.
    ///
    /// There is deliberately no extrapolation beyond the supplied temperature
    /// interval: an uncharacterized thermal state is not treated as safe.
    pub fn available_power_w(
        &self,
        temperature_k: f64,
    ) -> Result<f64, MissionThermalError> {
        self.validate()?;
        if !temperature_k.is_finite() {
            return Err(MissionThermalError::InvalidConfiguration(
                "derating lookup temperature must be finite",
            ));
        }
        let first = self.points.first().expect("validated curve has points");
        let last = self.points.last().expect("validated curve has points");
        if temperature_k < first.temperature_k || temperature_k > last.temperature_k {
            return Err(MissionThermalError::CurveOutOfDomain {
                component_id: self.component_id.clone(),
                temperature_k,
                minimum_temperature_k: first.temperature_k,
                maximum_temperature_k: last.temperature_k,
            });
        }
        for pair in self.points.windows(2) {
            let lower = pair[0];
            let upper = pair[1];
            if temperature_k >= lower.temperature_k && temperature_k <= upper.temperature_k {
                let fraction = (temperature_k - lower.temperature_k)
                    / (upper.temperature_k - lower.temperature_k);
                return Ok(lower.maximum_power_w
                    + fraction * (upper.maximum_power_w - lower.maximum_power_w));
            }
        }
        Ok(last.maximum_power_w)
    }
}

/// Per-segment ambient boundary and any separately characterized heat sources.
///
/// Additional heat must exclude battery I²R, motor copper/iron, and inverter
/// conversion losses already reconstructed from the mission report. It is
/// intended for independently measured items such as pump or auxiliary
/// electronics heat that is actually deposited into a modeled node.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentThermalProfile {
    pub segment_id: String,
    pub ambient_temperature_k: f64,
    pub additional_heat_generation_w: BTreeMap<String, f64>,
}

/// Controls thermal timestep, derating maps, and mission-aligned ambient inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionThermalConfig {
    /// Maximum thermal integration step. Each mission segment is subdivided
    /// into steps no larger than this value.
    pub maximum_step_s: f64,
    /// Exactly one curve is required for battery, inverter, and motor power.
    pub derating_curves: Vec<ThermalDeratingCurve>,
    /// One ambient / external-heat profile is required for each mission segment,
    /// in the same order and with the same ID.
    pub segment_profiles: Vec<SegmentThermalProfile>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionThermalSegmentReport {
    pub segment_id: String,
    pub duration_s: f64,
    pub substeps: usize,
    pub maximum_temperature_k: f64,
    pub maximum_temperature_node_id: String,
    /// Minimum available-minus-requested power over all components and substeps.
    pub minimum_power_margin_w: f64,
    pub heat_input_j: f64,
    pub ambient_rejected_energy_j: f64,
    pub stored_energy_change_j: f64,
    pub maximum_absolute_energy_balance_residual_j: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionThermalReport {
    pub model_version: &'static str,
    pub source_energy_model_version: &'static str,
    /// Upstream propulsion evidence records copied from the source MissionReport.
    pub source_characterization_evidence_records: Vec<crate::characterization::EvidenceRecord>,
    /// Upstream battery evidence records copied from the source MissionReport.
    pub source_battery_characterization_evidence_records: Vec<crate::characterization::EvidenceRecord>,
    /// Source MissionReport flag; this overlay copies but does not re-establish byte verification.
    pub source_evidence_content_digests_verified: bool,
    pub segments: Vec<MissionThermalSegmentReport>,
    pub final_state: ThermalState,
    pub maximum_temperature_k: f64,
    pub maximum_temperature_node_id: String,
    pub minimum_power_margin_w: f64,
    pub total_heat_input_j: f64,
    pub total_ambient_rejected_energy_j: f64,
    pub total_stored_energy_change_j: f64,
    pub maximum_absolute_energy_balance_residual_j: f64,
}

impl MissionThermalReport {
    /// A report only exists when every modeled node and derating constraint passed.
    pub fn is_thermally_feasible(&self) -> bool {
        self.minimum_power_margin_w.is_finite()
            && self.minimum_power_margin_w >= -1e-8
            && self.maximum_temperature_k.is_finite()
            && self.maximum_temperature_k > 0.0
            && self.maximum_absolute_energy_balance_residual_j.is_finite()
            && self.maximum_absolute_energy_balance_residual_j >= 0.0
            && self.final_state.temperatures_k.iter().all(|temperature| {
                temperature.is_finite() && *temperature > 0.0
            })
            && self.segments.iter().all(|segment| {
                segment.minimum_power_margin_w.is_finite()
                    && segment.minimum_power_margin_w >= -1e-8
                    && segment.maximum_temperature_k.is_finite()
                    && segment.maximum_temperature_k > 0.0
                    && segment.maximum_absolute_energy_balance_residual_j.is_finite()
                    && segment.maximum_absolute_energy_balance_residual_j >= 0.0
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MissionThermalError {
    InvalidConfiguration(&'static str),
    InvalidCurve {
        component_id: String,
        reason: &'static str,
    },
    CurveOutOfDomain {
        component_id: String,
        temperature_k: f64,
        minimum_temperature_k: f64,
        maximum_temperature_k: f64,
    },
    DuplicateComponentId(String),
    DuplicatePowerBasis(ComponentPowerBasis),
    UnknownThermalNode(String),
    MissionInput(crate::ModelError),
    Characterization(crate::characterization::CharacterizationError),
    MissionReportShapeMismatch(&'static str),
    MissionReportInputMismatch {
        segment_id: String,
        field: &'static str,
    },
    InvalidMissionProfile {
        segment_id: String,
        reason: &'static str,
    },
    PowerLimitExceeded {
        segment_id: String,
        component_id: String,
        requested_power_w: f64,
        available_power_w: f64,
        temperature_k: f64,
    },
    ThermalNetwork(ThermalNetworkError),
}

impl fmt::Display for MissionThermalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MissionThermalError {}

/// Replay the mission's modeled heat losses through the thermal network.
///
/// The function verifies that the report's segment identities and the
/// reconstructable power / energy terms agree with the supplied mission input.
/// It then checks component power demand against thermal derating curves at
/// every substep and advances the thermal network with component and external
/// heat sources. Any out-of-domain curve lookup, power-limit violation, node
/// temperature ceiling, or energy-accounting error fails the evaluation.
///
/// Important: the energy report is not recomputed as the thermal state evolves.
/// This is an independent thermal feasibility overlay that prepares the next
/// step: iterative coupling of temperature, OCV/resistance, efficiency, and
/// available power maps.
pub fn evaluate_mission_thermal_overlay(
    mission: &MissionInput,
    energy_report: &MissionReport,
    network: &ThermalNetwork,
    initial_state: &ThermalState,
    config: &MissionThermalConfig,
) -> Result<MissionThermalReport, MissionThermalError> {
    network
        .validate()
        .map_err(MissionThermalError::ThermalNetwork)?;
    validate_mission_report_pair(mission, energy_report)?;
    validate_config(network, mission, config)?;
    validate_initial_state(network, initial_state)?;

    let node_indices = network
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id.as_str(), index))
        .collect::<BTreeMap<_, _>>();

    let (mut mission_maximum_temperature_k, mut mission_maximum_node_id) =
        maximum_node_temperature(network, initial_state);
    let mut state = initial_state.clone();
    let mut total_heat_input_j = 0.0;
    let mut total_ambient_rejected_energy_j = 0.0;
    let mut total_stored_energy_change_j = 0.0;
    let mut maximum_absolute_energy_balance_residual_j: f64 = 0.0;
    let mut minimum_power_margin_w = f64::INFINITY;
    let mut segment_reports = Vec::with_capacity(mission.segments.len());

    for ((mission_segment, result), profile) in mission
        .segments
        .iter()
        .zip(&energy_report.segments)
        .zip(&config.segment_profiles)
    {
        let duration_s = mission_segment.duration_s;
        let heat_by_basis = component_heat_generation_w(result, duration_s)?;
        let demand_by_basis = component_power_demand_w(result);

        let mut heat_generation_w = vec![0.0_f64; network.nodes.len()];
        for curve in &config.derating_curves {
            let index = *node_indices
                .get(curve.node_id.as_str())
                .ok_or_else(|| MissionThermalError::UnknownThermalNode(curve.node_id.clone()))?;
            heat_generation_w[index] += heat_by_basis[&curve.power_basis];
        }
        for (node_id, heat_w) in &profile.additional_heat_generation_w {
            let index = *node_indices
                .get(node_id.as_str())
                .ok_or_else(|| MissionThermalError::UnknownThermalNode(node_id.clone()))?;
            heat_generation_w[index] += *heat_w;
        }

        let (mut segment_maximum_temperature_k, mut segment_maximum_node_id) =
            maximum_node_temperature(network, &state);
        let mut segment_minimum_power_margin_w = f64::INFINITY;
        let mut segment_heat_input_j = 0.0;
        let mut segment_ambient_rejected_energy_j = 0.0;
        let mut segment_stored_energy_change_j = 0.0;
        let mut segment_maximum_absolute_energy_balance_residual_j: f64 = 0.0;

        let step_count_f = (duration_s / config.maximum_step_s).ceil();
        if !step_count_f.is_finite() || step_count_f < 1.0
            || step_count_f > MAX_SUBSTEPS_PER_SEGMENT as f64
        {
            return Err(MissionThermalError::InvalidMissionProfile {
                segment_id: mission_segment.id.clone(),
                reason: "thermal substep count is outside the supported range",
            });
        }
        let step_count = step_count_f as usize;
        let mut elapsed_s = 0.0;

        for step_index in 0..step_count {
            let dt_s = (duration_s - elapsed_s).min(config.maximum_step_s);
            if !dt_s.is_finite() || dt_s <= 0.0 {
                return Err(MissionThermalError::InvalidMissionProfile {
                    segment_id: mission_segment.id.clone(),
                    reason: "thermal substep duration is invalid",
                });
            }
            let margin_before = check_derating(
                &mission_segment.id,
                &config.derating_curves,
                &node_indices,
                &state,
                &demand_by_basis,
            )?;
            segment_minimum_power_margin_w =
                segment_minimum_power_margin_w.min(margin_before);
            minimum_power_margin_w = minimum_power_margin_w.min(margin_before);

            let step = network
                .step(
                    &state,
                    &heat_generation_w,
                    dt_s,
                    profile.ambient_temperature_k,
                )
                .map_err(MissionThermalError::ThermalNetwork)?;
            state = step.state;

            let margin_after = check_derating(
                &mission_segment.id,
                &config.derating_curves,
                &node_indices,
                &state,
                &demand_by_basis,
            )?;
            segment_minimum_power_margin_w =
                segment_minimum_power_margin_w.min(margin_after);
            minimum_power_margin_w = minimum_power_margin_w.min(margin_after);

            if step.maximum_temperature_k > segment_maximum_temperature_k {
                segment_maximum_temperature_k = step.maximum_temperature_k;
                segment_maximum_node_id = step.maximum_temperature_node_id.clone();
            }
            if step.maximum_temperature_k > mission_maximum_temperature_k {
                mission_maximum_temperature_k = step.maximum_temperature_k;
                mission_maximum_node_id = step.maximum_temperature_node_id.clone();
            }

            segment_heat_input_j += step.heat_input_j;
            segment_ambient_rejected_energy_j += step.ambient_rejected_energy_j;
            segment_stored_energy_change_j += step.stored_energy_change_j;
            segment_maximum_absolute_energy_balance_residual_j = segment_maximum_absolute_energy_balance_residual_j
                .max(step.energy_balance_residual_j.abs());
            total_heat_input_j += step.heat_input_j;
            total_ambient_rejected_energy_j += step.ambient_rejected_energy_j;
            total_stored_energy_change_j += step.stored_energy_change_j;
            maximum_absolute_energy_balance_residual_j = maximum_absolute_energy_balance_residual_j
                .max(step.energy_balance_residual_j.abs());

            if !segment_heat_input_j.is_finite()
                || !segment_ambient_rejected_energy_j.is_finite()
                || !segment_stored_energy_change_j.is_finite()
                || !total_heat_input_j.is_finite()
                || !total_ambient_rejected_energy_j.is_finite()
                || !total_stored_energy_change_j.is_finite()
            {
                return Err(MissionThermalError::InvalidConfiguration(
                    "mission thermal energy accounting overflowed",
                ));
            }
            elapsed_s += dt_s;
            if step_index + 1 == step_count {
                // Assign the exact segment duration to the final boundary so
                // floating-point summation cannot silently shorten a segment.
                elapsed_s = duration_s;
            }
        }

        segment_reports.push(MissionThermalSegmentReport {
            segment_id: mission_segment.id.clone(),
            duration_s,
            substeps: step_count,
            maximum_temperature_k: segment_maximum_temperature_k,
            maximum_temperature_node_id: segment_maximum_node_id,
            minimum_power_margin_w: segment_minimum_power_margin_w,
            heat_input_j: segment_heat_input_j,
            ambient_rejected_energy_j: segment_ambient_rejected_energy_j,
            stored_energy_change_j: segment_stored_energy_change_j,
            maximum_absolute_energy_balance_residual_j:
                segment_maximum_absolute_energy_balance_residual_j,
        });
    }

    Ok(MissionThermalReport {
        model_version: MODEL_VERSION,
        source_energy_model_version: energy_report.model_version,
        source_characterization_evidence_records:
            energy_report.characterization_evidence_records.clone(),
        source_battery_characterization_evidence_records:
            energy_report.battery_characterization_evidence_records.clone(),
        source_evidence_content_digests_verified:
            energy_report.evidence_content_digests_verified,
        segments: segment_reports,
        final_state: state,
        maximum_temperature_k: mission_maximum_temperature_k,
        maximum_temperature_node_id: mission_maximum_node_id,
        minimum_power_margin_w,
        total_heat_input_j,
        total_ambient_rejected_energy_j,
        total_stored_energy_change_j,
        maximum_absolute_energy_balance_residual_j,
    })
}

fn validate_config(
    network: &ThermalNetwork,
    mission: &MissionInput,
    config: &MissionThermalConfig,
) -> Result<(), MissionThermalError> {
    if !config.maximum_step_s.is_finite() || config.maximum_step_s <= 0.0 {
        return Err(MissionThermalError::InvalidConfiguration(
            "maximum thermal step must be finite and positive",
        ));
    }
    if config.derating_curves.len() != 3 {
        return Err(MissionThermalError::InvalidConfiguration(
            "exactly one battery, inverter, and motor derating curve is required",
        ));
    }
    if config.segment_profiles.len() != mission.segments.len() {
        return Err(MissionThermalError::MissionReportShapeMismatch(
            "one thermal profile is required per mission segment",
        ));
    }

    let known_nodes = network
        .nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut component_ids = BTreeSet::new();
    let mut power_bases = BTreeSet::new();
    for curve in &config.derating_curves {
        curve.validate()?;
        if !known_nodes.contains(curve.node_id.as_str()) {
            return Err(MissionThermalError::UnknownThermalNode(
                curve.node_id.clone(),
            ));
        }
        if !component_ids.insert(curve.component_id.as_str()) {
            return Err(MissionThermalError::DuplicateComponentId(
                curve.component_id.clone(),
            ));
        }
        if !power_bases.insert(curve.power_basis) {
            return Err(MissionThermalError::DuplicatePowerBasis(
                curve.power_basis,
            ));
        }
    }
    for (segment, profile) in mission
        .segments
        .iter()
        .zip(&config.segment_profiles)
    {
        if segment.id != profile.segment_id {
            return Err(MissionThermalError::InvalidMissionProfile {
                segment_id: profile.segment_id.clone(),
                reason: "thermal profile IDs must match mission order exactly",
            });
        }
        if !profile.ambient_temperature_k.is_finite()
            || profile.ambient_temperature_k <= 0.0
        {
            return Err(MissionThermalError::InvalidMissionProfile {
                segment_id: profile.segment_id.clone(),
                reason: "ambient temperature must be finite and above absolute zero",
            });
        }
        for (node_id, heat_w) in &profile.additional_heat_generation_w {
            if !known_nodes.contains(node_id.as_str()) {
                return Err(MissionThermalError::UnknownThermalNode(node_id.clone()));
            }
            if !heat_w.is_finite() || *heat_w < 0.0 {
                return Err(MissionThermalError::InvalidMissionProfile {
                    segment_id: profile.segment_id.clone(),
                    reason: "additional heat sources must be finite and nonnegative",
                });
            }
        }
    }
    Ok(())
}

fn validate_initial_state(
    network: &ThermalNetwork,
    state: &ThermalState,
) -> Result<(), MissionThermalError> {
    if state.temperatures_k.len() != network.nodes.len() {
        return Err(MissionThermalError::InvalidConfiguration(
            "initial temperature vector must match thermal network node count",
        ));
    }
    for (node, temperature) in network.nodes.iter().zip(&state.temperatures_k) {
        if !temperature.is_finite()
            || *temperature <= 0.0
            || *temperature > node.maximum_temperature_k
        {
            return Err(MissionThermalError::InvalidConfiguration(
                "initial temperatures must be positive and within every node ceiling",
            ));
        }
    }
    Ok(())
}

fn validate_mission_report_pair(
    mission: &MissionInput,
    report: &MissionReport,
) -> Result<(), MissionThermalError> {
    crate::validate_input(mission).map_err(MissionThermalError::MissionInput)?;
    if mission.segments.len() != report.segments.len() {
        return Err(MissionThermalError::MissionReportShapeMismatch(
            "mission and energy report segment counts differ",
        ));
    }

    let battery = &mission.battery;
    let capacity_wh = battery.nominal_energy_wh * battery.state_of_health;
    let expected_mass = mission.aircraft.empty_mass_kg
        + mission.aircraft.payload_mass_kg
        + battery.pack_mass_kg;
    let expected_specific_energy = battery.nominal_energy_wh / battery.pack_mass_kg;
    let expected_available_energy_wh =
        (battery.initial_state_of_charge - battery.reserve_state_of_charge) * capacity_wh;
    if !close_enough(expected_mass, report.all_up_mass_kg)
        || !close_enough(expected_specific_energy, report.battery_specific_energy_wh_per_kg)
        || !close_enough(expected_available_energy_wh, report.available_energy_to_reserve_wh)
    {
        return Err(MissionThermalError::MissionReportInputMismatch {
            segment_id: "<mission>".into(),
            field: "all-up mass, pack specific energy, or available reserve energy",
        });
    }

    let propulsion_record_ids = report
        .characterization_evidence_records
        .iter()
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    let battery_record_ids = report
        .battery_characterization_evidence_records
        .iter()
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    if propulsion_record_ids != report.characterization_evidence_ids
        || battery_record_ids != report.battery_characterization_evidence_ids
    {
        return Err(MissionThermalError::MissionReportInputMismatch {
            segment_id: "<mission>".into(),
            field: "characterization evidence IDs do not match source evidence records",
        });
    }
    for record in report
        .characterization_evidence_records
        .iter()
        .chain(&report.battery_characterization_evidence_records)
    {
        record
            .validate()
            .map_err(MissionThermalError::Characterization)?;
    }

    let mut previous_soc = battery.initial_state_of_charge;
    let mut total_terminal_wh = 0.0;
    let mut total_chemical_wh = 0.0;
    let mut peak_power_w: f64 = 0.0;
    let mut peak_current_a: f64 = 0.0;
    let mut minimum_voltage_v = f64::INFINITY;
    let mut maximum_pack_temperature_k = battery.initial_temperature_k;

    for (segment, result) in mission.segments.iter().zip(&report.segments) {
        if segment.id != result.id || segment.phase != result.phase {
            return Err(MissionThermalError::MissionReportInputMismatch {
                segment_id: segment.id.clone(),
                field: "segment ID or flight phase",
            });
        }

        let dt = segment.duration_s;
        let expected_drag_n = report.all_up_mass_kg * GRAVITY_MPS2
            / segment.lift_to_drag_ratio;
        let climb_power_w = report.all_up_mass_kg
            * GRAVITY_MPS2
            * segment.climb_rate_mps.max(0.0);
        let useful_power_w = expected_drag_n * segment.true_airspeed_mps + climb_power_w;

        let efficiencies = [
            result.motor_efficiency_used,
            result.inverter_efficiency_used,
            result.propeller_efficiency_used,
        ];
        if !efficiencies.iter().all(|efficiency| {
            efficiency.is_finite() && *efficiency > 0.0 && *efficiency <= 1.0
        }) {
            return Err(MissionThermalError::MissionReportInputMismatch {
                segment_id: segment.id.clone(),
                field: "resolved propulsion efficiency",
            });
        }

        let expected_shaft_w = useful_power_w / result.propeller_efficiency_used;
        let expected_motor_input_w = expected_shaft_w / result.motor_efficiency_used;
        let expected_inverter_input_w = expected_motor_input_w / result.inverter_efficiency_used;
        let expected_battery_power_w = expected_inverter_input_w + segment.auxiliary_power_w;

        // Verify the battery equivalent-circuit identity independently of the
        // mission evaluator: P_terminal = I * (OCV - I*R).
        let current_a = result.battery_current_a;
        let ocv_v = result.battery_open_circuit_voltage_used_v;
        let resistance_ohm = result.battery_internal_resistance_used_ohm;
        let terminal_voltage_v = ocv_v - current_a * resistance_ohm;
        let electrical_terminal_power_w = current_a * terminal_voltage_v;
        let terminal_wh = electrical_terminal_power_w * dt / SECONDS_PER_HOUR;
        let chemical_wh = current_a * ocv_v * dt / SECONDS_PER_HOUR;
        let resistive_heat_wh =
            current_a * current_a * resistance_ohm * dt / SECONDS_PER_HOUR;
        let expected_soc = previous_soc - chemical_wh / capacity_wh;

        let numeric_values = [
            expected_drag_n,
            expected_shaft_w,
            expected_motor_input_w,
            expected_inverter_input_w,
            expected_battery_power_w,
            current_a,
            ocv_v,
            resistance_ohm,
            terminal_voltage_v,
            electrical_terminal_power_w,
            terminal_wh,
            chemical_wh,
            resistive_heat_wh,
            expected_soc,
            result.aerodynamic_drag_n,
            result.propulsive_shaft_power_w,
            result.battery_power_w,
            result.terminal_voltage_v,
            result.terminal_energy_wh,
            result.chemical_energy_wh,
            result.resistive_heat_wh,
            result.pack_temperature_end_k,
            result.state_of_charge_end,
        ];
        if numeric_values.iter().any(|value| !value.is_finite())
            || current_a < 0.0
            || ocv_v <= 0.0
            || resistance_ohm < 0.0
            || terminal_voltage_v <= 0.0
            || electrical_terminal_power_w < 0.0
            || electrical_terminal_power_w > battery.maximum_continuous_power_w + 1e-9
            || current_a > battery.maximum_continuous_current_a + 1e-9
            || result.state_of_charge_end < battery.reserve_state_of_charge
            || result.state_of_charge_end > 1.0
            || result.pack_temperature_end_k <= 0.0
            || result.pack_temperature_end_k > battery.maximum_temperature_k
        {
            return Err(MissionThermalError::MissionReportInputMismatch {
                segment_id: segment.id.clone(),
                field: "non-finite or physically invalid segment electrical / thermal state",
            });
        }

        let calculated = [
            expected_drag_n,
            expected_shaft_w,
            expected_battery_power_w,
            electrical_terminal_power_w,
            terminal_voltage_v,
            terminal_wh,
            chemical_wh,
            resistive_heat_wh,
            expected_soc,
        ];
        let reported = [
            result.aerodynamic_drag_n,
            result.propulsive_shaft_power_w,
            result.battery_power_w,
            result.battery_current_a * result.terminal_voltage_v,
            result.terminal_voltage_v,
            result.terminal_energy_wh,
            result.chemical_energy_wh,
            result.resistive_heat_wh,
            result.state_of_charge_end,
        ];
        if calculated
            .iter()
            .zip(reported)
            .any(|(expected, actual)| !close_enough(*expected, actual))
        {
            return Err(MissionThermalError::MissionReportInputMismatch {
                segment_id: segment.id.clone(),
                field: "drag, propulsion/battery power, voltage, energy, I²R heat, or SOC transition",
            });
        }

        // Also assert the physical identity that chemical draw equals terminal
        // delivery plus equivalent internal ohmic loss.
        if !close_enough(chemical_wh, terminal_wh + resistive_heat_wh) {
            return Err(MissionThermalError::MissionReportInputMismatch {
                segment_id: segment.id.clone(),
                field: "battery chemical / terminal / resistive energy balance",
            });
        }

        previous_soc = result.state_of_charge_end;
        total_terminal_wh += terminal_wh;
        total_chemical_wh += chemical_wh;
        peak_power_w = peak_power_w.max(electrical_terminal_power_w);
        peak_current_a = peak_current_a.max(current_a);
        minimum_voltage_v = minimum_voltage_v.min(terminal_voltage_v);
        maximum_pack_temperature_k =
            maximum_pack_temperature_k.max(result.pack_temperature_end_k);
    }

    let expected_energy_with_uncertainty =
        total_chemical_wh * (1.0 + mission.mission_uncertainty_fraction);
    let expected_energy_margin_wh =
        expected_available_energy_wh - expected_energy_with_uncertainty;
    if !expected_energy_margin_wh.is_finite() || expected_energy_margin_wh < -1e-9 {
        return Err(MissionThermalError::MissionReportInputMismatch {
            segment_id: "<mission>".into(),
            field: "mission energy reserve is infringed",
        });
    }
    let expected_final_soc = report
        .segments
        .last()
        .map_or(battery.initial_state_of_charge, |segment| segment.state_of_charge_end);

    let aggregate_pairs = [
        (total_terminal_wh, report.mission_terminal_energy_wh),
        (total_chemical_wh, report.mission_chemical_energy_wh),
        (expected_energy_with_uncertainty, report.energy_required_with_uncertainty_wh),
        (expected_energy_margin_wh, report.energy_margin_wh),
        (peak_power_w, report.peak_battery_power_w),
        (peak_current_a, report.peak_battery_current_a),
        (minimum_voltage_v, report.minimum_terminal_voltage_v),
        (maximum_pack_temperature_k, report.maximum_pack_temperature_k),
        (expected_final_soc, report.final_state_of_charge),
    ];
    if aggregate_pairs
        .iter()
        .any(|(expected, actual)| !close_enough(*expected, *actual))
    {
        return Err(MissionThermalError::MissionReportInputMismatch {
            segment_id: "<mission>".into(),
            field: "aggregate energy, SOC, peak power/current, minimum voltage, or maximum pack temperature",
        });
    }

    Ok(())
}

fn component_heat_generation_w(
    result: &SegmentResult,
    duration_s: f64,
) -> Result<BTreeMap<ComponentPowerBasis, f64>, MissionThermalError> {
    let shaft_power_w = result.propulsive_shaft_power_w;
    let motor_efficiency = result.motor_efficiency_used;
    let inverter_efficiency = result.inverter_efficiency_used;
    let motor_input_w = shaft_power_w / motor_efficiency;
    let inverter_input_w = motor_input_w / inverter_efficiency;
    let heat = BTreeMap::from([
        (
            ComponentPowerBasis::BatteryTerminal,
            result.resistive_heat_wh * SECONDS_PER_HOUR / duration_s,
        ),
        (
            ComponentPowerBasis::InverterDcInput,
            inverter_input_w - motor_input_w,
        ),
        (
            ComponentPowerBasis::MotorShaftOutput,
            motor_input_w - shaft_power_w,
        ),
    ]);
    if heat.values().any(|value| !value.is_finite() || *value < -1e-8) {
        return Err(MissionThermalError::MissionReportInputMismatch {
            segment_id: result.id.clone(),
            field: "component loss reconstruction",
        });
    }
    Ok(heat
        .into_iter()
        .map(|(basis, watts)| (basis, watts.max(0.0)))
        .collect())
}

fn component_power_demand_w(
    result: &SegmentResult,
) -> BTreeMap<ComponentPowerBasis, f64> {
    let motor_input_w = result.propulsive_shaft_power_w / result.motor_efficiency_used;
    let inverter_input_w = motor_input_w / result.inverter_efficiency_used;
    BTreeMap::from([
        (ComponentPowerBasis::BatteryTerminal, result.battery_power_w),
        (ComponentPowerBasis::InverterDcInput, inverter_input_w),
        (ComponentPowerBasis::MotorShaftOutput, result.propulsive_shaft_power_w),
    ])
}

fn check_derating(
    segment_id: &str,
    curves: &[ThermalDeratingCurve],
    node_indices: &BTreeMap<&str, usize>,
    state: &ThermalState,
    demand_by_basis: &BTreeMap<ComponentPowerBasis, f64>,
) -> Result<f64, MissionThermalError> {
    let mut minimum_margin_w = f64::INFINITY;
    for curve in curves {
        let index = *node_indices
            .get(curve.node_id.as_str())
            .ok_or_else(|| MissionThermalError::UnknownThermalNode(curve.node_id.clone()))?;
        let temperature_k = state.temperatures_k[index];
        let available_power_w = curve.available_power_w(temperature_k)?;
        let requested_power_w = *demand_by_basis
            .get(&curve.power_basis)
            .ok_or(MissionThermalError::InvalidConfiguration(
                "component power demand is missing",
            ))?;
        let margin_w = available_power_w - requested_power_w;
        minimum_margin_w = minimum_margin_w.min(margin_w);
        if margin_w < -1e-8 {
            return Err(MissionThermalError::PowerLimitExceeded {
                segment_id: segment_id.into(),
                component_id: curve.component_id.clone(),
                requested_power_w,
                available_power_w,
                temperature_k,
            });
        }
    }
    Ok(minimum_margin_w)
}

fn maximum_node_temperature(
    network: &ThermalNetwork,
    state: &ThermalState,
) -> (f64, String) {
    network
        .nodes
        .iter()
        .zip(&state.temperatures_k)
        .max_by(|(_, first), (_, second)| first.total_cmp(second))
        .map(|(node, temperature)| (*temperature, node.id.clone()))
        .unwrap_or((f64::NAN, String::new()))
}

fn close_enough(expected: f64, actual: f64) -> bool {
    expected.is_finite()
        && actual.is_finite()
        && (expected - actual).abs() <= 1e-8 * expected.abs().max(actual.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AircraftConfig, BatteryConfig, FlightPhase, MissionSegment, PropulsionConfig,
        evaluate_mission,
    };
    use crate::thermal_network::{ThermalLink, ThermalNode};

    fn baseline_mission() -> MissionInput {
        MissionInput {
            aircraft: AircraftConfig {
                empty_mass_kg: 350.0,
                payload_mass_kg: 40.0,
                maximum_takeoff_mass_kg: 700.0,
            },
            battery: BatteryConfig {
                nominal_energy_wh: 40_000.0,
                pack_mass_kg: 200.0,
                nominal_open_circuit_voltage_v: 400.0,
                maximum_continuous_power_w: 80_000.0,
                maximum_continuous_current_a: 250.0,
                internal_resistance_ohm: 0.04,
                thermal_capacity_j_per_k: 80_000.0,
                thermal_conductance_w_per_k: 100.0,
                maximum_temperature_k: 333.15,
                initial_temperature_k: 298.15,
                ambient_temperature_k: 293.15,
                state_of_health: 0.95,
                initial_state_of_charge: 0.95,
                reserve_state_of_charge: 0.20,
            },
            propulsion: PropulsionConfig {
                motor_efficiency: 0.94,
                inverter_efficiency: 0.97,
                propeller_efficiency: 0.82,
            },
            mission_uncertainty_fraction: 0.10,
            segments: vec![
                MissionSegment {
                    id: "climb".into(),
                    phase: FlightPhase::Climb,
                    duration_s: 120.0,
                    true_airspeed_mps: 40.0,
                    lift_to_drag_ratio: 14.0,
                    climb_rate_mps: 2.0,
                    auxiliary_power_w: 800.0,
                },
                MissionSegment {
                    id: "cruise".into(),
                    phase: FlightPhase::Cruise,
                    duration_s: 600.0,
                    true_airspeed_mps: 50.0,
                    lift_to_drag_ratio: 16.0,
                    climb_rate_mps: 0.0,
                    auxiliary_power_w: 600.0,
                },
            ],
        }
    }

    fn network() -> ThermalNetwork {
        ThermalNetwork {
            nodes: vec![
                ThermalNode {
                    id: "battery".into(),
                    heat_capacity_j_per_k: 80_000.0,
                    ambient_conductance_w_per_k: 90.0,
                    maximum_temperature_k: 390.0,
                },
                ThermalNode {
                    id: "inverter".into(),
                    heat_capacity_j_per_k: 8_000.0,
                    ambient_conductance_w_per_k: 30.0,
                    maximum_temperature_k: 420.0,
                },
                ThermalNode {
                    id: "motor".into(),
                    heat_capacity_j_per_k: 20_000.0,
                    ambient_conductance_w_per_k: 45.0,
                    maximum_temperature_k: 430.0,
                },
                ThermalNode {
                    id: "coolant".into(),
                    heat_capacity_j_per_k: 25_000.0,
                    ambient_conductance_w_per_k: 20.0,
                    maximum_temperature_k: 390.0,
                },
            ],
            links: vec![
                ThermalLink {
                    first_node_id: "battery".into(),
                    second_node_id: "coolant".into(),
                    conductance_w_per_k: 12.0,
                },
                ThermalLink {
                    first_node_id: "inverter".into(),
                    second_node_id: "coolant".into(),
                    conductance_w_per_k: 8.0,
                },
                ThermalLink {
                    first_node_id: "motor".into(),
                    second_node_id: "coolant".into(),
                    conductance_w_per_k: 10.0,
                },
            ],
        }
    }

    fn curve(
        component_id: &str,
        node_id: &str,
        basis: ComponentPowerBasis,
        low_temperature_power_w: f64,
        high_temperature_power_w: f64,
    ) -> ThermalDeratingCurve {
        ThermalDeratingCurve {
            component_id: component_id.into(),
            node_id: node_id.into(),
            power_basis: basis,
            points: vec![
                ThermalDeratingPoint {
                    temperature_k: 280.0,
                    maximum_power_w: low_temperature_power_w,
                },
                ThermalDeratingPoint {
                    temperature_k: 390.0,
                    maximum_power_w: high_temperature_power_w,
                },
            ],
        }
    }

    fn config() -> MissionThermalConfig {
        MissionThermalConfig {
            maximum_step_s: 10.0,
            derating_curves: vec![
                curve("battery-pack", "battery", ComponentPowerBasis::BatteryTerminal, 90_000.0, 65_000.0),
                curve("traction-inverter", "inverter", ComponentPowerBasis::InverterDcInput, 90_000.0, 65_000.0),
                curve("traction-motor", "motor", ComponentPowerBasis::MotorShaftOutput, 90_000.0, 65_000.0),
            ],
            segment_profiles: vec![
                SegmentThermalProfile {
                    segment_id: "climb".into(),
                    ambient_temperature_k: 303.15,
                    additional_heat_generation_w: BTreeMap::from([("coolant".into(), 120.0)]),
                },
                SegmentThermalProfile {
                    segment_id: "cruise".into(),
                    ambient_temperature_k: 303.15,
                    additional_heat_generation_w: BTreeMap::from([("coolant".into(), 80.0)]),
                },
            ],
        }
    }

    fn initial_state() -> ThermalState {
        ThermalState {
            temperatures_k: vec![298.15, 298.15, 298.15, 298.15],
        }
    }

    #[test]
    fn mission_report_is_replayed_through_component_thermal_network() {
        let mission = baseline_mission();
        let energy = evaluate_mission(&mission).unwrap();
        let result = evaluate_mission_thermal_overlay(
            &mission,
            &energy,
            &network(),
            &initial_state(),
            &config(),
        )
        .unwrap();

        assert!(result.is_thermally_feasible());
        assert_eq!(result.source_energy_model_version, energy.model_version);
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].substeps, 12);
        assert!(result.minimum_power_margin_w > 0.0);
        assert!(result.maximum_temperature_k >= 298.15);
        assert!(result.maximum_absolute_energy_balance_residual_j < 1e-6);
        assert_eq!(result.final_state.temperatures_k.len(), 4);
    }

    #[test]
    fn thermal_derating_limit_rejects_high_power_at_hot_start() {
        let mission = baseline_mission();
        let energy = evaluate_mission(&mission).unwrap();
        let mut hot_state = initial_state();
        hot_state.temperatures_k = vec![350.0, 350.0, 350.0, 350.0];
        let mut hot_config = config();
        for curve in &mut hot_config.derating_curves {
            curve.points = vec![
                ThermalDeratingPoint { temperature_k: 280.0, maximum_power_w: 90_000.0 },
                ThermalDeratingPoint { temperature_k: 390.0, maximum_power_w: 1_000.0 },
            ];
        }

        assert!(matches!(
            evaluate_mission_thermal_overlay(&mission, &energy, &network(), &hot_state, &hot_config),
            Err(MissionThermalError::PowerLimitExceeded { .. })
        ));
    }

    #[test]
    fn thermal_derating_rejects_uncharacterized_temperature() {
        let mission = baseline_mission();
        let energy = evaluate_mission(&mission).unwrap();
        let mut cold_state = initial_state();
        cold_state.temperatures_k = vec![270.0, 270.0, 270.0, 270.0];

        assert!(matches!(
            evaluate_mission_thermal_overlay(&mission, &energy, &network(), &cold_state, &config()),
            Err(MissionThermalError::CurveOutOfDomain { .. })
        ));
    }

    #[test]
    fn derating_curve_preserves_cold_and_hot_power_limits() {
        let battery_curve = ThermalDeratingCurve {
            component_id: "battery".into(),
            node_id: "battery".into(),
            power_basis: ComponentPowerBasis::BatteryTerminal,
            points: vec![
                ThermalDeratingPoint { temperature_k: 270.0, maximum_power_w: 10_000.0 },
                ThermalDeratingPoint { temperature_k: 300.0, maximum_power_w: 90_000.0 },
                ThermalDeratingPoint { temperature_k: 390.0, maximum_power_w: 65_000.0 },
            ],
        };
        battery_curve.validate().unwrap();
        assert!((battery_curve.available_power_w(285.0).unwrap() - 50_000.0).abs() < 1e-9);
        assert!((battery_curve.available_power_w(390.0).unwrap() - 65_000.0).abs() < 1e-9);
    }

    #[test]
    fn source_evidence_records_and_digest_status_are_preserved() {
        use crate::characterization::{EvidenceClass, EvidenceRecord};

        let mission = baseline_mission();
        let mut energy = evaluate_mission(&mission).unwrap();
        let propulsion_record = EvidenceRecord {
            id: "motor-map".into(),
            uri: "artifact://test/motor-map".into(),
            sha256: "a".repeat(64),
            class: EvidenceClass::PhysicalMeasurement,
            description: "Synthetic provenance fixture".into(),
        };
        let battery_record = EvidenceRecord {
            id: "battery-map".into(),
            uri: "artifact://test/battery-map".into(),
            sha256: "b".repeat(64),
            class: EvidenceClass::PhysicalMeasurement,
            description: "Synthetic provenance fixture".into(),
        };
        energy.characterization_evidence_ids.push(propulsion_record.id.clone());
        energy.characterization_evidence_records.push(propulsion_record.clone());
        energy.battery_characterization_evidence_ids.push(battery_record.id.clone());
        energy.battery_characterization_evidence_records.push(battery_record.clone());
        energy.evidence_content_digests_verified = true;

        let result = evaluate_mission_thermal_overlay(
            &mission, &energy, &network(), &initial_state(), &config()
        ).unwrap();

        assert_eq!(result.source_characterization_evidence_records, vec![propulsion_record]);
        assert_eq!(result.source_battery_characterization_evidence_records, vec![battery_record]);
        assert!(result.source_evidence_content_digests_verified);
    }

    #[test]
    fn evidence_id_record_mismatch_is_rejected() {
        let mission = baseline_mission();
        let mut energy = evaluate_mission(&mission).unwrap();
        energy.characterization_evidence_ids.push("missing-record".into());

        assert!(matches!(
            evaluate_mission_thermal_overlay(&mission, &energy, &network(), &initial_state(), &config()),
            Err(MissionThermalError::MissionReportInputMismatch { .. })
        ));
    }

    #[test]
    fn battery_current_and_voltage_identity_must_match_report() {
        let mission = baseline_mission();
        let mut energy = evaluate_mission(&mission).unwrap();
        energy.segments[0].battery_current_a *= 0.99;

        assert!(matches!(
            evaluate_mission_thermal_overlay(&mission, &energy, &network(), &initial_state(), &config()),
            Err(MissionThermalError::MissionReportInputMismatch { .. })
        ));
    }

    #[test]
    fn aggregate_soc_and_energy_totals_must_match_segment_records() {
        let mission = baseline_mission();
        let mut energy = evaluate_mission(&mission).unwrap();
        energy.final_state_of_charge = (energy.final_state_of_charge - 0.01).max(0.0);

        assert!(matches!(
            evaluate_mission_thermal_overlay(&mission, &energy, &network(), &initial_state(), &config()),
            Err(MissionThermalError::MissionReportInputMismatch { .. })
        ));
    }

    #[test]
    fn report_must_match_mission_power_and_duration() {
        let mission = baseline_mission();
        let mut energy = evaluate_mission(&mission).unwrap();
        energy.segments[0].terminal_energy_wh += 1.0;

        assert!(matches!(
            evaluate_mission_thermal_overlay(&mission, &energy, &network(), &initial_state(), &config()),
            Err(MissionThermalError::MissionReportInputMismatch { .. })
        ));
    }

    #[test]
    fn thermal_profile_order_must_match_mission_order() {
        let mission = baseline_mission();
        let energy = evaluate_mission(&mission).unwrap();
        let mut bad_config = config();
        bad_config.segment_profiles.swap(0, 1);

        assert!(matches!(
            evaluate_mission_thermal_overlay(&mission, &energy, &network(), &initial_state(), &bad_config),
            Err(MissionThermalError::InvalidMissionProfile { .. })
        ));
    }

    #[test]
    fn node_temperature_ceiling_remains_fail_closed() {
        let mission = baseline_mission();
        let energy = evaluate_mission(&mission).unwrap();
        let mut overheated_network = network();
        overheated_network.nodes[0].maximum_temperature_k = 298.2;
        let mut high_heat_config = config();
        high_heat_config.segment_profiles[0]
            .additional_heat_generation_w
            .insert("battery".into(), 10_000.0);

        assert!(matches!(
            evaluate_mission_thermal_overlay(
                &mission,
                &energy,
                &overheated_network,
                &initial_state(),
                &high_heat_config,
            ),
            Err(MissionThermalError::ThermalNetwork(
                ThermalNetworkError::TemperatureLimitExceeded { .. }
            ))
        ));
    }

    #[test]
    fn configuration_requires_exact_component_derating_boundaries() {
        let mission = baseline_mission();
        let energy = evaluate_mission(&mission).unwrap();
        let mut bad_config = config();
        bad_config.derating_curves.pop();

        assert!(matches!(
            evaluate_mission_thermal_overlay(&mission, &energy, &network(), &initial_state(), &bad_config),
            Err(MissionThermalError::InvalidConfiguration(_))
        ));
    }
}
