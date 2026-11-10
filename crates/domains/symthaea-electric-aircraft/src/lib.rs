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

use std::fmt;

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
    pub propulsive_shaft_power_w: f64,
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
    pub segments: Vec<SegmentResult>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    InvalidAircraftConfig(&'static str),
    InvalidBatteryConfig(&'static str),
    InvalidPropulsionConfig(&'static str),
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

    let chain_efficiency = input.propulsion.motor_efficiency
        * input.propulsion.inverter_efficiency
        * input.propulsion.propeller_efficiency;

    for segment in &input.segments {
        let weight_n = mass_kg * STANDARD_GRAVITY_MPS2;
        // Reduced-order, steady-flight approximation: lift ~= weight and
        // drag = lift/(L/D). Transient and manoeuvre load factors are not modelled.
        let drag_n = weight_n / segment.lift_to_drag_ratio;
        let aerodynamic_power_w = drag_n * segment.true_airspeed_mps;
        let climb_power_w = mass_kg
            * STANDARD_GRAVITY_MPS2
            * segment.climb_rate_mps.max(0.0);
        let shaft_power_w = aerodynamic_power_w + climb_power_w;
        let battery_power_w = shaft_power_w / chain_efficiency + segment.auxiliary_power_w;
        if !battery_power_w.is_finite() {
            return Err(ModelError::InvalidMission("computed segment power overflowed"));
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
            battery.nominal_open_circuit_voltage_v,
            battery.internal_resistance_ohm,
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

        let terminal_voltage_v = battery.nominal_open_circuit_voltage_v
            - current_a * battery.internal_resistance_ohm;
        if !terminal_voltage_v.is_finite() || terminal_voltage_v <= 0.0 {
            return Err(ModelError::BatteryVoltageCollapse { segment_id: segment.id.clone() });
        }

        let dt = segment.duration_s;
        let terminal_wh = battery_power_w * dt / SECONDS_PER_HOUR;
        // Chemical energy draw uses open-circuit voltage; the difference from
        // terminal energy is the internal I²R loss in this equivalent model.
        let chemical_wh = current_a * battery.nominal_open_circuit_voltage_v
            * dt / SECONDS_PER_HOUR;
        let heat_w = current_a * current_a * battery.internal_resistance_ohm;
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
            propulsive_shaft_power_w: shaft_power_w,
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
        model_version: "electric-fixed-wing-energy-v0.1.0",
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
