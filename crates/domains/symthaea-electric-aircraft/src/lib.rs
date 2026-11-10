// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see repository root for terms.

//! Reduced-order electric fixed-wing mission-energy feasibility model.
//!
//! This is a small, inspectable physics kernel. It estimates steady-segment
//! drag, climb power, propulsion-chain losses, battery current/voltage sag,
//! resistive heating, state-of-charge consumption, and protected reserve.
//!
//! It is NOT a flight-dynamics solver, battery qualification model, aero model,
//! flight-control system, or certification means of compliance. Inputs such as
//! lift-to-drag ratio and component efficiencies must come from separately
//! qualified sources.

use std::collections::BTreeMap;
use std::fmt;

pub mod characterization;
pub mod uncertainty;

use characterization::{
    BatteryCharacterization, BatteryOperatingPoint, CharacterizationError, EvidenceRecord,
    PropulsionCharacterization, PropulsionOperatingPoint,
};

const STANDARD_GRAVITY_MPS2: f64 = 9.806_65;
const SECONDS_PER_HOUR: f64 = 3_600.0;

#[derive(Debug, Clone, PartialEq)]
pub struct AircraftConfig {
    /// Empty mass must exclude the separately declared battery pack mass.
    pub empty_mass_kg: f64,
    pub payload_mass_kg: f64,
    pub maximum_takeoff_mass_kg: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatteryConfig {
    /// Nominal pack energy at beginning of life, measured at the pack terminals (Wh).
    pub nominal_energy_wh: f64,
    pub pack_mass_kg: f64,
    pub nominal_open_circuit_voltage_v: f64,
    pub maximum_continuous_power_w: f64,
    pub maximum_continuous_current_a: f64,
    /// Pack-level equivalent DC resistance at the model reference condition (ohm).
    pub internal_resistance_ohm: f64,
    /// Lumped pack thermal capacitance (J/K).
    pub thermal_capacity_j_per_k: f64,
    /// Lumped pack-to-ambient thermal conductance (W/K); zero means adiabatic.
    pub thermal_conductance_w_per_k: f64,
    pub maximum_temperature_k: f64,
    pub initial_temperature_k: f64,
    pub ambient_temperature_k: f64,
    pub state_of_health: f64,
    pub initial_state_of_charge: f64,
    /// Protected minimum state of charge; mission energy may not consume this reserve.
    pub reserve_state_of_charge: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropulsionConfig {
    pub motor_efficiency: f64,
    pub inverter_efficiency: f64,
    pub propeller_efficiency: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightPhase {
    Climb,
    Cruise,
    Descent,
    Approach,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionSegment {
    pub id: String,
    pub phase: FlightPhase,
    pub duration_s: f64,
    pub true_airspeed_mps: f64,
    /// Must come from an aerodynamic polar or other declared source.
    pub lift_to_drag_ratio: f64,
    /// Positive is climb; negative is descent. Descent energy recovery is not credited.
    pub climb_rate_mps: f64,
    pub auxiliary_power_w: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionInput {
    pub aircraft: AircraftConfig,
    pub battery: BatteryConfig,
    pub propulsion: PropulsionConfig,
    /// Screening margin only; not a statistical interval or regulatory reserve.
    pub mission_uncertainty_fraction: f64,
    pub segments: Vec<MissionSegment>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentResult {
    pub id: String,
    pub phase: FlightPhase,
    pub aerodynamic_drag_n: f64,
    /// Mechanical shaft power delivered to the propeller after propeller losses.
    pub propulsive_shaft_power_w: f64,
    pub motor_efficiency_used: f64,
    pub inverter_efficiency_used: f64,
    pub propeller_efficiency_used: f64,
    pub battery_open_circuit_voltage_used_v: f64,
    pub battery_internal_resistance_used_ohm: f64,
    pub battery_power_w: f64,
    pub battery_current_a: f64,
    pub terminal_voltage_v: f64,
    pub terminal_energy_wh: f64,
    pub chemical_energy_wh: f64,
    pub resistive_heat_wh: f64,
    pub pack_temperature_end_k: f64,
    pub state_of_charge_end: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionReport {
    pub model_version: &'static str,
    pub all_up_mass_kg: f64,
    pub battery_specific_energy_wh_per_kg: f64,
    pub available_energy_to_reserve_wh: f64,
    pub mission_terminal_energy_wh: f64,
    pub mission_chemical_energy_wh: f64,
    pub energy_required_with_uncertainty_wh: f64,
    pub energy_margin_wh: f64,
    pub peak_battery_power_w: f64,
    pub peak_battery_current_a: f64,
    pub minimum_terminal_voltage_v: f64,
    pub maximum_pack_temperature_k: f64,
    pub final_state_of_charge: f64,
    /// Evidence dataset IDs used by calibrated propulsion maps; empty for scalar mode.
    pub characterization_evidence_ids: Vec<String>,
    /// Full source metadata and digests for calibrated propulsion maps.
    pub characterization_evidence_records: Vec<EvidenceRecord>,
    /// Battery OCV/resistance evidence dataset IDs; empty for scalar battery mode.
    pub battery_characterization_evidence_ids: Vec<String>,
    /// Full source metadata and digests for characterized battery maps.
    pub battery_characterization_evidence_records: Vec<EvidenceRecord>,
    /// True only when exact caller-supplied dataset bytes passed SHA-256
    /// checks through the fully verified evaluator. This is not a signature,
    /// publisher, authority or configuration-approval verdict.
    pub evidence_content_digests_verified: bool,
    pub segments: Vec<SegmentResult>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    InvalidAircraftConfig(&'static str),
    InvalidBatteryConfig(&'static str),
    InvalidPropulsionConfig(&'static str),
    Characterization(CharacterizationError),
    InvalidMission(&'static str),
    DuplicateSegmentId(String),
    MaximumTakeoffMassExceeded { actual_kg: f64, maximum_kg: f64 },
    BatteryPowerLimitExceeded { segment_id: String, requested_w: f64, maximum_w: f64 },
    BatteryCurrentLimitExceeded { segment_id: String, requested_a: f64, maximum_a: f64 },
    BatteryVoltageCollapse { segment_id: String },
    BatteryTemperatureLimitExceeded { segment_id: String, temperature_k: f64, maximum_k: f64 },
    ReserveInfringed { required_wh: f64, available_wh: f64 },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ModelError {}

pub fn evaluate_mission(input: &MissionInput) -> Result<MissionReport, ModelError> {
    let scalar_propulsion = scalar_propulsion_point(&input.propulsion);
    let battery = &input.battery;
    let scalar_battery = BatteryOperatingPoint {
        open_circuit_voltage_v: battery.nominal_open_circuit_voltage_v,
        internal_resistance_ohm: battery.internal_resistance_ohm,
    };
    evaluate_mission_with_providers(
        input,
        |_, _| Ok(scalar_propulsion),
        |_, _| Ok(scalar_battery.clone()),
        Vec::new(),
        Vec::new(),
    )
}

/// Evaluate with evidence-bound propulsion efficiency maps.
///
/// Each map is sampled only inside its characterized domain. Its nominal
/// efficiency minus the supplied absolute uncertainty bound is used for
/// energy sizing. This is a conservative screening convention, not a
/// statistical confidence interval or certification method.
pub fn evaluate_mission_characterized(
    input: &MissionInput,
    characterization: &PropulsionCharacterization,
) -> Result<MissionReport, ModelError> {
    characterization.validate().map_err(ModelError::Characterization)?;
    let evidence_records = characterization.evidence_records();
    let battery = &input.battery;
    let scalar_battery = BatteryOperatingPoint {
        open_circuit_voltage_v: battery.nominal_open_circuit_voltage_v,
        internal_resistance_ohm: battery.internal_resistance_ohm,
    };
    evaluate_mission_with_providers(
        input,
        |segment, useful_power_w| {
            characterization
                .operating_point(segment.true_airspeed_mps, useful_power_w)
                .map_err(ModelError::Characterization)
        },
        |_, _| Ok(scalar_battery.clone()),
        evidence_records,
        Vec::new(),
    )
}

/// Evaluate with SOC/temperature-dependent pack open-circuit voltage and
/// resistance surfaces while retaining scalar propulsion efficiencies.
pub fn evaluate_mission_battery_characterized(
    input: &MissionInput,
    characterization: &BatteryCharacterization,
) -> Result<MissionReport, ModelError> {
    characterization.validate().map_err(ModelError::Characterization)?;
    let evidence_records = characterization.evidence_records();
    let scalar_propulsion = scalar_propulsion_point(&input.propulsion);
    evaluate_mission_with_providers(
        input,
        |_, _| Ok(scalar_propulsion),
        |soc, temperature| {
            characterization
                .operating_point(soc, temperature)
                .map_err(ModelError::Characterization)
        },
        Vec::new(),
        evidence_records,
    )
}

/// Evaluate with both evidence-bound propulsion maps and battery OCV/resistance
/// surfaces. Every map must cover the actual operating point; extrapolation is
/// rejected.
pub fn evaluate_mission_fully_characterized(
    input: &MissionInput,
    propulsion: &PropulsionCharacterization,
    battery: &BatteryCharacterization,
) -> Result<MissionReport, ModelError> {
    propulsion.validate().map_err(ModelError::Characterization)?;
    battery.validate().map_err(ModelError::Characterization)?;
    let propulsion_evidence_records = propulsion.evidence_records();
    let battery_evidence_records = battery.evidence_records();
    evaluate_mission_with_providers(
        input,
        |segment, useful_power_w| {
            propulsion
                .operating_point(segment.true_airspeed_mps, useful_power_w)
                .map_err(ModelError::Characterization)
        },
        |soc, temperature| {
            battery
                .operating_point(soc, temperature)
                .map_err(ModelError::Characterization)
        },
        propulsion_evidence_records,
        battery_evidence_records,
    )
}

/// Evaluate a fully characterized mission only after verifying the exact
/// caller-supplied bytes for every propulsion and battery evidence record.
/// This verifies SHA-256 content integrity, but it does not fetch URIs or
/// authenticate publishers, signatures, authority, or configuration approval.
pub fn evaluate_mission_fully_verified(
    input: &MissionInput,
    propulsion: &PropulsionCharacterization,
    battery: &BatteryCharacterization,
    evidence_payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<MissionReport, ModelError> {
    propulsion
        .verify_evidence_payloads(evidence_payloads)
        .map_err(ModelError::Characterization)?;
    battery
        .verify_evidence_payloads(evidence_payloads)
        .map_err(ModelError::Characterization)?;
    let mut report = evaluate_mission_fully_characterized(input, propulsion, battery)?;
    report.evidence_content_digests_verified = true;
    Ok(report)
}

fn scalar_propulsion_point(config: &PropulsionConfig) -> PropulsionOperatingPoint {
    PropulsionOperatingPoint {
        motor_efficiency_used: config.motor_efficiency,
        inverter_efficiency_used: config.inverter_efficiency,
        propeller_efficiency_used: config.propeller_efficiency,
        motor_load_fraction: 0.0,
        inverter_load_fraction: 0.0,
        propeller_advance_ratio: 0.0,
    }
}

fn evaluate_mission_with_providers<F, G>(
    input: &MissionInput,
    mut efficiency_provider: F,
    mut battery_provider: G,
    characterization_evidence_records: Vec<EvidenceRecord>,
    battery_characterization_evidence_records: Vec<EvidenceRecord>,
) -> Result<MissionReport, ModelError>
where
    F: FnMut(&MissionSegment, f64) -> Result<PropulsionOperatingPoint, ModelError>,
    G: FnMut(f64, f64) -> Result<BatteryOperatingPoint, ModelError>,
{
    validate_input(input)?;

    let mass_kg = input.aircraft.empty_mass_kg
        + input.aircraft.payload_mass_kg
        + input.battery.pack_mass_kg;
    if !mass_kg.is_finite() {
        return Err(ModelError::InvalidAircraftConfig("all-up mass overflowed"));
    }
    if mass_kg > input.aircraft.maximum_takeoff_mass_kg {
        return Err(ModelError::MaximumTakeoffMassExceeded {
            actual_kg: mass_kg,
            maximum_kg: input.aircraft.maximum_takeoff_mass_kg,
        });
    }

    let battery = &input.battery;
    let capacity_wh = battery.nominal_energy_wh * battery.state_of_health;
    let available_energy_wh = (battery.initial_state_of_charge - battery.reserve_state_of_charge)
        * capacity_wh;
    if !capacity_wh.is_finite() || !available_energy_wh.is_finite() {
        return Err(ModelError::InvalidBatteryConfig("capacity or reserve energy overflowed"));
    }

    let mut soc = battery.initial_state_of_charge;
    let mut temperature_k = battery.initial_temperature_k;
    let mut terminal_energy_wh = 0.0;
    let mut chemical_energy_wh = 0.0;
    let mut peak_power_w: f64 = 0.0;
    let mut peak_current_a: f64 = 0.0;
    let mut minimum_voltage_v = f64::INFINITY;
    let mut maximum_temperature_k = temperature_k;
    let mut results = Vec::with_capacity(input.segments.len());

    for segment in &input.segments {
        let weight_n = mass_kg * STANDARD_GRAVITY_MPS2;
        // Reduced-order, steady-flight approximation: lift ~= weight and
        // drag = lift/(L/D). Transient and manoeuvre load factors are not modelled.
        let drag_n = weight_n / segment.lift_to_drag_ratio;
        let aerodynamic_power_w = drag_n * segment.true_airspeed_mps;
        let climb_power_w = mass_kg
            * STANDARD_GRAVITY_MPS2
            * segment.climb_rate_mps.max(0.0);
        let useful_propulsive_power_w = aerodynamic_power_w + climb_power_w;
        let operating_point = efficiency_provider(segment, useful_propulsive_power_w)?;
        let chain_efficiency = operating_point.motor_efficiency_used
            * operating_point.inverter_efficiency_used
            * operating_point.propeller_efficiency_used;
        if !chain_efficiency.is_finite() || chain_efficiency <= 0.0 {
            return Err(ModelError::InvalidPropulsionConfig(
                "resolved propulsion efficiency must be finite and positive",
            ));
        }
        let propeller_shaft_power_w =
            useful_propulsive_power_w / operating_point.propeller_efficiency_used;
        let battery_power_w = propeller_shaft_power_w
            / (operating_point.motor_efficiency_used * operating_point.inverter_efficiency_used)
            + segment.auxiliary_power_w;
        if !battery_power_w.is_finite() || !propeller_shaft_power_w.is_finite() {
            return Err(ModelError::InvalidMission("computed segment power overflowed"));
        }

        let battery_operating_point = battery_provider(soc, temperature_k)?;
        let open_circuit_voltage_v = battery_operating_point.open_circuit_voltage_v;
        let internal_resistance_ohm = battery_operating_point.internal_resistance_ohm;
        if !open_circuit_voltage_v.is_finite()
            || open_circuit_voltage_v <= 0.0
            || !internal_resistance_ohm.is_finite()
            || internal_resistance_ohm < 0.0
        {
            return Err(ModelError::InvalidBatteryConfig(
                "resolved battery OCV and internal resistance must be finite and positive",
            ));
        }

        if battery_power_w > battery.maximum_continuous_power_w + 1e-9 {
            return Err(ModelError::BatteryPowerLimitExceeded {
                segment_id: segment.id.clone(),
                requested_w: battery_power_w,
                maximum_w: battery.maximum_continuous_power_w,
            });
        }

        let current_a = current_for_terminal_power(
            battery_power_w,
            open_circuit_voltage_v,
            internal_resistance_ohm,
        )
        .ok_or_else(|| ModelError::BatteryVoltageCollapse { segment_id: segment.id.clone() })?;
        if !current_a.is_finite() {
            return Err(ModelError::BatteryVoltageCollapse { segment_id: segment.id.clone() });
        }
        if current_a > battery.maximum_continuous_current_a + 1e-9 {
            return Err(ModelError::BatteryCurrentLimitExceeded {
                segment_id: segment.id.clone(),
                requested_a: current_a,
                maximum_a: battery.maximum_continuous_current_a,
            });
        }

        let terminal_voltage_v = open_circuit_voltage_v - current_a * internal_resistance_ohm;
        if !terminal_voltage_v.is_finite() || terminal_voltage_v <= 0.0 {
            return Err(ModelError::BatteryVoltageCollapse { segment_id: segment.id.clone() });
        }

        let dt = segment.duration_s;
        let terminal_wh = battery_power_w * dt / SECONDS_PER_HOUR;
        // Chemical energy draw uses open-circuit voltage; the difference from
        // terminal energy is the internal I²R loss in this equivalent model.
        let chemical_wh = current_a * open_circuit_voltage_v * dt / SECONDS_PER_HOUR;
        let heat_w = current_a * current_a * internal_resistance_ohm;
        let heat_wh = heat_w * dt / SECONDS_PER_HOUR;

        let thermal_decay = if battery.thermal_conductance_w_per_k > 0.0 {
            (-battery.thermal_conductance_w_per_k * dt / battery.thermal_capacity_j_per_k).exp()
        } else {
            1.0
        };
        let new_temperature_k = if battery.thermal_conductance_w_per_k > 0.0 {
            let equilibrium_k = battery.ambient_temperature_k
                + heat_w / battery.thermal_conductance_w_per_k;
            equilibrium_k + (temperature_k - equilibrium_k) * thermal_decay
        } else {
            temperature_k + heat_w * dt / battery.thermal_capacity_j_per_k
        };
        if !new_temperature_k.is_finite() || new_temperature_k > battery.maximum_temperature_k {
            return Err(ModelError::BatteryTemperatureLimitExceeded {
                segment_id: segment.id.clone(),
                temperature_k: new_temperature_k,
                maximum_k: battery.maximum_temperature_k,
            });
        }

        soc -= chemical_wh / capacity_wh;
        terminal_energy_wh += terminal_wh;
        chemical_energy_wh += chemical_wh;
        if !soc.is_finite() || !terminal_energy_wh.is_finite() || !chemical_energy_wh.is_finite() {
            return Err(ModelError::InvalidMission("mission state or energy accumulation overflowed"));
        }
        peak_power_w = peak_power_w.max(battery_power_w);
        peak_current_a = peak_current_a.max(current_a);
        minimum_voltage_v = minimum_voltage_v.min(terminal_voltage_v);
        maximum_temperature_k = maximum_temperature_k.max(new_temperature_k);
        temperature_k = new_temperature_k;

        results.push(SegmentResult {
            id: segment.id.clone(),
            phase: segment.phase,
            aerodynamic_drag_n: drag_n,
            propulsive_shaft_power_w: propeller_shaft_power_w,
            motor_efficiency_used: operating_point.motor_efficiency_used,
            inverter_efficiency_used: operating_point.inverter_efficiency_used,
            propeller_efficiency_used: operating_point.propeller_efficiency_used,
            battery_open_circuit_voltage_used_v: open_circuit_voltage_v,
            battery_internal_resistance_used_ohm: internal_resistance_ohm,
            battery_power_w,
            battery_current_a: current_a,
            terminal_voltage_v,
            terminal_energy_wh: terminal_wh,
            chemical_energy_wh: chemical_wh,
            resistive_heat_wh: heat_wh,
            pack_temperature_end_k: temperature_k,
            state_of_charge_end: soc,
        });
    }

    let energy_required_with_uncertainty_wh = chemical_energy_wh
        * (1.0 + input.mission_uncertainty_fraction);
    let energy_margin_wh = available_energy_wh - energy_required_with_uncertainty_wh;
    if !energy_required_with_uncertainty_wh.is_finite() || !energy_margin_wh.is_finite() {
        return Err(ModelError::InvalidMission("mission reserve arithmetic overflowed"));
    }
    if energy_margin_wh < -1e-9 {
        return Err(ModelError::ReserveInfringed {
            required_wh: energy_required_with_uncertainty_wh,
            available_wh: available_energy_wh,
        });
    }

    Ok(MissionReport {
        model_version: "electric-fixed-wing-energy-v0.3.0",
        all_up_mass_kg: mass_kg,
        battery_specific_energy_wh_per_kg: battery.nominal_energy_wh / battery.pack_mass_kg,
        available_energy_to_reserve_wh: available_energy_wh,
        mission_terminal_energy_wh: terminal_energy_wh,
        mission_chemical_energy_wh: chemical_energy_wh,
        energy_required_with_uncertainty_wh,
        energy_margin_wh,
        peak_battery_power_w: peak_power_w,
        peak_battery_current_a: peak_current_a,
        minimum_terminal_voltage_v: minimum_voltage_v,
        maximum_pack_temperature_k,
        final_state_of_charge: soc,
        characterization_evidence_ids: characterization_evidence_records
            .iter()
            .map(|record| record.id.clone())
            .collect(),
        characterization_evidence_records,
        battery_characterization_evidence_ids: battery_characterization_evidence_records
            .iter()
            .map(|record| record.id.clone())
            .collect(),
        battery_characterization_evidence_records,
        evidence_content_digests_verified: false,
        segments: results,
    })
}

fn current_for_terminal_power(
    power_w: f64,
    open_circuit_voltage_v: f64,
    resistance_ohm: f64,
) -> Option<f64> {
    if power_w == 0.0 {
        return Some(0.0);
    }
    if resistance_ohm == 0.0 {
        return Some(power_w / open_circuit_voltage_v);
    }
    let discriminant = open_circuit_voltage_v * open_circuit_voltage_v
        - 4.0 * resistance_ohm * power_w;
    if !discriminant.is_finite() || discriminant < 0.0 {
        return None;
    }
    Some((open_circuit_voltage_v - discriminant.sqrt()) / (2.0 * resistance_ohm))
}

fn validate_input(input: &MissionInput) -> Result<(), ModelError> {
    let aircraft = &input.aircraft;
    if !aircraft.empty_mass_kg.is_finite() || aircraft.empty_mass_kg <= 0.0
        || !aircraft.payload_mass_kg.is_finite() || aircraft.payload_mass_kg < 0.0
        || !aircraft.maximum_takeoff_mass_kg.is_finite() || aircraft.maximum_takeoff_mass_kg <= 0.0
    {
        return Err(ModelError::InvalidAircraftConfig(
            "masses must be finite, with positive empty mass and MTOW",
        ));
    }

    let b = &input.battery;
    let positive = [
        b.nominal_energy_wh,
        b.pack_mass_kg,
        b.nominal_open_circuit_voltage_v,
        b.maximum_continuous_power_w,
        b.maximum_continuous_current_a,
        b.thermal_capacity_j_per_k,
        b.maximum_temperature_k,
        b.initial_temperature_k,
    ];
    if positive.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || !b.internal_resistance_ohm.is_finite() || b.internal_resistance_ohm < 0.0
        || !b.thermal_conductance_w_per_k.is_finite() || b.thermal_conductance_w_per_k < 0.0
        || !b.ambient_temperature_k.is_finite() || b.ambient_temperature_k <= 0.0
        || !(0.0..=1.0).contains(&b.state_of_health)
        || !(0.0..=1.0).contains(&b.initial_state_of_charge)
        || !(0.0..=1.0).contains(&b.reserve_state_of_charge)
        || b.state_of_health == 0.0
        || b.initial_state_of_charge <= b.reserve_state_of_charge
        || b.maximum_temperature_k <= b.ambient_temperature_k
        || b.initial_temperature_k > b.maximum_temperature_k
    {
        return Err(ModelError::InvalidBatteryConfig(
            "invalid capacity, electrical, thermal, SOH, or SOC parameter",
        ));
    }

    let p = &input.propulsion;
    for efficiency in [p.motor_efficiency, p.inverter_efficiency, p.propeller_efficiency] {
        if !efficiency.is_finite() || efficiency <= 0.0 || efficiency > 1.0 {
            return Err(ModelError::InvalidPropulsionConfig(
                "efficiencies must be in (0, 1]",
            ));
        }
    }

    if !input.mission_uncertainty_fraction.is_finite()
        || input.mission_uncertainty_fraction < 0.0
        || input.segments.is_empty()
    {
        return Err(ModelError::InvalidMission(
            "uncertainty must be nonnegative and at least one segment is required",
        ));
    }

    let mut ids = std::collections::BTreeSet::new();
    for segment in &input.segments {
        if segment.id.trim().is_empty() {
            return Err(ModelError::InvalidMission("segment ID must be nonempty"));
        }
        if !ids.insert(segment.id.as_str()) {
            return Err(ModelError::DuplicateSegmentId(segment.id.clone()));
        }
        if !segment.duration_s.is_finite() || segment.duration_s <= 0.0
            || !segment.true_airspeed_mps.is_finite() || segment.true_airspeed_mps <= 0.0
            || !segment.lift_to_drag_ratio.is_finite() || segment.lift_to_drag_ratio <= 0.0
            || !segment.climb_rate_mps.is_finite()
            || !segment.auxiliary_power_w.is_finite() || segment.auxiliary_power_w < 0.0
        {
            return Err(ModelError::InvalidMission(
                "segment duration, speed, L/D, climb rate, or auxiliary load is invalid",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> MissionInput {
        MissionInput {
            aircraft: AircraftConfig {
                empty_mass_kg: 350.0,
                payload_mass_kg: 50.0,
                maximum_takeoff_mass_kg: 700.0,
            },
            battery: BatteryConfig {
                nominal_energy_wh: 40_000.0,
                pack_mass_kg: 240.0,
                nominal_open_circuit_voltage_v: 400.0,
                maximum_continuous_power_w: 75_000.0,
                maximum_continuous_current_a: 250.0,
                internal_resistance_ohm: 0.04,
                thermal_capacity_j_per_k: 80_000.0,
                thermal_conductance_w_per_k: 140.0,
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
            mission_uncertainty_fraction: 0.20,
            segments: vec![
                MissionSegment {
                    id: "climb".into(),
                    phase: FlightPhase::Climb,
                    duration_s: 120.0,
                    true_airspeed_mps: 35.0,
                    lift_to_drag_ratio: 12.0,
                    climb_rate_mps: 3.0,
                    auxiliary_power_w: 1_000.0,
                },
                MissionSegment {
                    id: "cruise".into(),
                    phase: FlightPhase::Cruise,
                    duration_s: 1_200.0,
                    true_airspeed_mps: 55.0,
                    lift_to_drag_ratio: 15.0,
                    climb_rate_mps: 0.0,
                    auxiliary_power_w: 800.0,
                },
                MissionSegment {
                    id: "descent".into(),
                    phase: FlightPhase::Descent,
                    duration_s: 300.0,
                    true_airspeed_mps: 40.0,
                    lift_to_drag_ratio: 12.0,
                    climb_rate_mps: -2.0,
                    auxiliary_power_w: 700.0,
                },
            ],
        }
    }

    #[test]
    fn baseline_mission_produces_finite_energy_and_reserve_report() {
        let report = evaluate_mission(&baseline()).unwrap();
        assert!(report.mission_terminal_energy_wh.is_finite());
        assert!(report.mission_terminal_energy_wh > 0.0);
        assert!(report.energy_margin_wh > 0.0);
        assert!(report.final_state_of_charge < 0.95);
        assert_eq!(report.segments.len(), 3);
        let modeled_resistive_loss_wh: f64 = report.segments
            .iter()
            .map(|segment| segment.resistive_heat_wh)
            .sum();
        assert!(
            (report.mission_chemical_energy_wh
                - report.mission_terminal_energy_wh
                - modeled_resistive_loss_wh)
                .abs() < 1e-6
        );
    }

    #[test]
    fn higher_mass_increases_required_energy() {
        let light = baseline();
        let mut heavy = baseline();
        heavy.aircraft.payload_mass_kg += 40.0;
        let light_report = evaluate_mission(&light).unwrap();
        let heavy_report = evaluate_mission(&heavy).unwrap();
        assert!(heavy_report.mission_terminal_energy_wh > light_report.mission_terminal_energy_wh);
    }

    #[test]
    fn descent_does_not_create_unqualified_regeneration_credit() {
        let mut descending = baseline();
        let mut level = baseline();
        descending.segments[2].climb_rate_mps = -8.0;
        level.segments[2].climb_rate_mps = 0.0;
        let a = evaluate_mission(&descending).unwrap();
        let b = evaluate_mission(&level).unwrap();
        assert!((a.mission_terminal_energy_wh - b.mission_terminal_energy_wh).abs() < 1e-9);
    }

    #[test]
    fn over_mtow_is_rejected() {
        let mut input = baseline();
        input.aircraft.maximum_takeoff_mass_kg = 500.0;
        assert!(matches!(
            evaluate_mission(&input),
            Err(ModelError::MaximumTakeoffMassExceeded { .. })
        ));
    }

    #[test]
    fn insufficient_energy_reserve_is_rejected() {
        let mut input = baseline();
        input.battery.initial_state_of_charge = 0.35;
        assert!(matches!(
            evaluate_mission(&input),
            Err(ModelError::ReserveInfringed { .. })
        ));
    }

    #[test]
    fn battery_power_limit_is_enforced() {
        let mut input = baseline();
        input.battery.maximum_continuous_power_w = 5_000.0;
        assert!(matches!(
            evaluate_mission(&input),
            Err(ModelError::BatteryPowerLimitExceeded { .. })
        ));
    }

    #[test]
    fn battery_current_limit_is_enforced_independently() {
        let mut input = baseline();
        input.battery.maximum_continuous_current_a = 100.0;
        assert!(matches!(
            evaluate_mission(&input),
            Err(ModelError::BatteryCurrentLimitExceeded { .. })
        ));
    }

    #[test]
    fn infeasible_terminal_power_causes_voltage_collapse() {
        let mut input = baseline();
        input.battery.internal_resistance_ohm = 1.0;
        assert!(matches!(
            evaluate_mission(&input),
            Err(ModelError::BatteryVoltageCollapse { .. })
        ));
    }

    #[test]
    fn invalid_aero_input_is_rejected() {
        let mut input = baseline();
        input.segments[0].lift_to_drag_ratio = 0.0;
        assert!(matches!(
            evaluate_mission(&input),
            Err(ModelError::InvalidMission(_))
        ));
    }

    #[test]
    fn duplicate_segment_ids_are_rejected() {
        let mut input = baseline();
        input.segments[1].id = "climb".into();
        assert!(matches!(
            evaluate_mission(&input),
            Err(ModelError::DuplicateSegmentId(_))
        ));
    }

    #[test]
    fn thermal_limit_is_enforced() {
        let mut input = baseline();
        input.battery.thermal_conductance_w_per_k = 0.0;
        input.battery.thermal_capacity_j_per_k = 100.0;
        input.battery.maximum_temperature_k = 300.0;
        assert!(matches!(
            evaluate_mission(&input),
            Err(ModelError::BatteryTemperatureLimitExceeded { .. })
        ));
    }

    #[test]
    fn uncertainty_margin_makes_mission_harder_to_qualify() {
        let mut low = baseline();
        let mut high = baseline();
        low.mission_uncertainty_fraction = 0.0;
        high.mission_uncertainty_fraction = 0.5;
        let a = evaluate_mission(&low).unwrap();
        let b = evaluate_mission(&high).unwrap();
        assert!(b.energy_required_with_uncertainty_wh > a.energy_required_with_uncertainty_wh);
    }

    fn battery_characterization() -> BatteryCharacterization {
        use characterization::{
            EvidenceClass, EvidenceRecord, ScalarGrid2D, ScalarGridPoint,
        };

        let evidence = |id: &str| EvidenceRecord {
            id: id.into(),
            uri: format!("artifact://test/{id}"),
            sha256: "c".repeat(64),
            class: EvidenceClass::Simulation,
            description: "Synthetic unit-test fixture; not physical evidence".into(),
        };
        let grid = |id: &str, evidence_id: &str, values: Vec<Vec<f64>>, uncertainty: f64| {
            ScalarGrid2D {
                id: id.into(),
                x_axis: "state_of_charge".into(),
                x_unit: "1".into(),
                x_values: vec![0.2, 0.6, 0.95],
                y_axis: "temperature".into(),
                y_unit: "K".into(),
                y_values: vec![273.0, 298.0, 323.0],
                evidence: evidence(evidence_id),
                values: values
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|value| ScalarGridPoint {
                                value,
                                absolute_uncertainty: uncertainty,
                            })
                            .collect()
                    })
                    .collect(),
            }
        };
        BatteryCharacterization {
            open_circuit_voltage: grid(
                "pack-ocv",
                "pack-ocv-data",
                vec![
                    vec![340.0, 350.0, 345.0],
                    vec![365.0, 375.0, 370.0],
                    vec![385.0, 395.0, 390.0],
                ],
                1.5,
            ),
            internal_resistance: grid(
                "pack-resistance",
                "pack-resistance-data",
                vec![
                    vec![0.070, 0.065, 0.070],
                    vec![0.055, 0.050, 0.060],
                    vec![0.050, 0.045, 0.055],
                ],
                0.002,
            ),
        }
    }

    fn propulsion_characterization() -> PropulsionCharacterization {
        use characterization::{EfficiencyCurve, EfficiencyPoint, EvidenceClass, EvidenceRecord};

        let evidence = |id: &str| EvidenceRecord {
            id: id.into(),
            uri: format!("artifact://test/{id}"),
            sha256: "b".repeat(64),
            class: EvidenceClass::Simulation,
            description: "Synthetic unit-test fixture; not physical evidence".into(),
        };
        let curve = |id: &str, axis: &str, points: Vec<EfficiencyPoint>| EfficiencyCurve {
            id: id.into(),
            x_axis: axis.into(),
            x_unit: "1".into(),
            evidence: evidence(&format!("{id}-evidence")),
            points,
        };
        PropulsionCharacterization {
            motor_efficiency_by_load: curve(
                "motor",
                "shaft_load_fraction",
                vec![
                    EfficiencyPoint { x: 0.1, efficiency: 0.92, absolute_uncertainty: 0.01 },
                    EfficiencyPoint { x: 1.0, efficiency: 0.96, absolute_uncertainty: 0.01 },
                ],
            ),
            inverter_efficiency_by_load: curve(
                "inverter",
                "input_load_fraction",
                vec![
                    EfficiencyPoint { x: 0.1, efficiency: 0.97, absolute_uncertainty: 0.005 },
                    EfficiencyPoint { x: 1.0, efficiency: 0.985, absolute_uncertainty: 0.005 },
                ],
            ),
            propeller_efficiency_by_advance_ratio: curve(
                "propeller",
                "advance_ratio",
                vec![
                    EfficiencyPoint { x: 0.4, efficiency: 0.80, absolute_uncertainty: 0.02 },
                    EfficiencyPoint { x: 1.0, efficiency: 0.86, absolute_uncertainty: 0.02 },
                ],
            ),
            motor_rated_shaft_power_w: 80_000.0,
            inverter_rated_input_power_w: 90_000.0,
            propeller_diameter_m: 1.6,
            propeller_rpm: 2_400.0,
        }
    }

    #[test]
    fn characterized_mission_uses_conservative_efficiency_bounds_and_records_evidence() {
        let input = baseline();
        let characterization = propulsion_characterization();
        let scalar = evaluate_mission(&input).unwrap();
        let mapped = evaluate_mission_characterized(&input, &characterization).unwrap();

        assert_eq!(mapped.model_version, "electric-fixed-wing-energy-v0.3.0");
        assert_eq!(
            mapped.characterization_evidence_ids,
            vec!["inverter-evidence", "motor-evidence", "propeller-evidence"]
        );
        assert!(mapped.mission_terminal_energy_wh > scalar.mission_terminal_energy_wh);
        // Independent numerical oracle for this synthetic map fixture.
        assert!((mapped.mission_terminal_energy_wh - 14_922.574_361_978_19).abs() < 1e-5);
        assert!((mapped.mission_chemical_energy_wh - 15_052.058_448_099_895).abs() < 1e-5);
        assert!((mapped.energy_margin_wh - 10_437.529_862_280_127).abs() < 1e-5);
        assert!((mapped.segments[0].propeller_efficiency_used - 0.794_687_5).abs() < 1e-12);
        assert!(mapped.segments.iter().all(|segment| {
            segment.motor_efficiency_used > 0.0
                && segment.motor_efficiency_used < 1.0
                && segment.inverter_efficiency_used > 0.0
                && segment.inverter_efficiency_used < 1.0
                && segment.propeller_efficiency_used > 0.0
                && segment.propeller_efficiency_used < 1.0
        }));
    }

    #[test]
    fn characterized_mission_rejects_unmeasured_operating_region() {
        let input = baseline();
        let mut characterization = propulsion_characterization();
        characterization.propeller_rpm = 4_000.0;
        assert!(matches!(
            evaluate_mission_characterized(&input, &characterization),
            Err(ModelError::Characterization(
                CharacterizationError::OutsideCalibrationDomain { .. }
            ))
        ));
    }

    #[test]
    fn battery_characterized_mission_uses_conservative_surfaces_and_records_evidence() {
        let input = baseline();
        let propulsion = propulsion_characterization();
        let battery = battery_characterization();
        let propulsion_only = evaluate_mission_characterized(&input, &propulsion).unwrap();
        let fully_mapped =
            evaluate_mission_fully_characterized(&input, &propulsion, &battery).unwrap();

        assert_eq!(fully_mapped.model_version, "electric-fixed-wing-energy-v0.3.0");
        assert_eq!(
            fully_mapped.battery_characterization_evidence_ids,
            vec!["pack-ocv-data", "pack-resistance-data"]
        );
        assert!(fully_mapped.segments[0].battery_open_circuit_voltage_used_v < 400.0);
        assert!(fully_mapped.segments[0].battery_internal_resistance_used_ohm > 0.04);
        assert!(fully_mapped.mission_chemical_energy_wh
            > propulsion_only.mission_chemical_energy_wh);
        // Independent numerical oracle for this synthetic SOC/temperature map fixture.
        assert!((fully_mapped.mission_chemical_energy_wh - 15_089.329_326_948_14).abs() < 1e-5);
        assert!((fully_mapped.energy_margin_wh - 10_392.804_807_662_233).abs() < 1e-5);
    }

    #[test]
    fn fully_verified_mission_checks_exact_dataset_bytes_before_evaluation() {
        use std::collections::BTreeMap;

        let input = baseline();
        let mut propulsion = propulsion_characterization();
        let mut battery = battery_characterization();
        let expected_sha256 =
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        for record in [
            &mut propulsion.motor_efficiency_by_load.evidence,
            &mut propulsion.inverter_efficiency_by_load.evidence,
            &mut propulsion.propeller_efficiency_by_advance_ratio.evidence,
            &mut battery.open_circuit_voltage.evidence,
            &mut battery.internal_resistance.evidence,
        ] {
            record.sha256 = expected_sha256.into();
        }
        let evidence_ids = [
            "motor-evidence",
            "inverter-evidence",
            "propeller-evidence",
            "pack-ocv-data",
            "pack-resistance-data",
        ];
        let payloads = evidence_ids
            .into_iter()
            .map(|id| (id.to_string(), b"abc".to_vec()))
            .collect::<BTreeMap<_, _>>();

        let report =
            evaluate_mission_fully_verified(&input, &propulsion, &battery, &payloads).unwrap();
        assert_eq!(report.model_version, "electric-fixed-wing-energy-v0.3.0");
        assert_eq!(report.characterization_evidence_ids.len(), 3);
        assert_eq!(report.battery_characterization_evidence_ids.len(), 2);

        let mut corrupt = payloads;
        corrupt.insert("motor-evidence".into(), b"changed bytes".to_vec());
        assert!(matches!(
            evaluate_mission_fully_verified(&input, &propulsion, &battery, &corrupt),
            Err(ModelError::Characterization(
                CharacterizationError::EvidenceDigestMismatch { .. }
            ))
        ));
    }

    #[test]
    fn battery_characterized_mission_rejects_out_of_domain_soc() {
        let mut input = baseline();
        input.battery.initial_state_of_charge = 0.15;
        input.battery.reserve_state_of_charge = 0.10;
        assert!(matches!(
            evaluate_mission_battery_characterized(&input, &battery_characterization()),
            Err(ModelError::Characterization(
                CharacterizationError::GridOutsideCalibrationDomain { .. }
            ))
        ));
    }

    #[test]
    fn conservative_mission_changes_do_not_improve_energy_margin() {
        let base = evaluate_mission(&baseline()).unwrap();

        let mut faster = baseline();
        for segment in &mut faster.segments {
            segment.true_airspeed_mps *= 1.10;
        }
        let faster_report = evaluate_mission(&faster).unwrap();
        assert!(faster_report.energy_required_with_uncertainty_wh
            > base.energy_required_with_uncertainty_wh);

        let mut lower_soh = baseline();
        lower_soh.battery.state_of_health = 0.85;
        let lower_soh_report = evaluate_mission(&lower_soh).unwrap();
        assert!(lower_soh_report.energy_margin_wh < base.energy_margin_wh);

        let mut higher_reserve = baseline();
        higher_reserve.battery.reserve_state_of_charge = 0.30;
        let higher_reserve_report = evaluate_mission(&higher_reserve).unwrap();
        assert!(higher_reserve_report.energy_margin_wh < base.energy_margin_wh);

        let mut lower_efficiency = baseline();
        lower_efficiency.propulsion.motor_efficiency = 0.90;
        let lower_efficiency_report = evaluate_mission(&lower_efficiency).unwrap();
        assert!(lower_efficiency_report.energy_required_with_uncertainty_wh
            > base.energy_required_with_uncertainty_wh);
    }

}
