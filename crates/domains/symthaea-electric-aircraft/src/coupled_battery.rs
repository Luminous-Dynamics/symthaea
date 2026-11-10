// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded, time-resolved battery/electro-thermal feedback for one mission substep.
//!
//! This is the first coupled production slice: the battery OCV and resistance
//! maps are re-sampled at the candidate pack temperature, terminal current and
//! I²R heat are recomputed, and the existing multi-node implicit thermal
//! network is advanced until the battery-node temperature fixed point closes.
//!
//! Propulsion demand and non-battery heat inputs are frozen over this substep.
//! Motor/inverter efficiency feedback, dynamic TMS flow, cell-level physics,
//! ageing evolution and aircraft-level re-optimization remain separate work.

use std::fmt;

use crate::characterization::{
    BatteryCharacterization, BatteryOperatingPoint, CharacterizationError, EvidenceRecord,
};
use crate::thermal_network::{
    ThermalNetwork, ThermalNetworkError, ThermalState,
};
use crate::BatteryConfig;

const MODEL_VERSION: &str = "electric-aircraft-coupled-battery-step-v0.1.0";
const SECONDS_PER_HOUR: f64 = 3_600.0;
const MAX_ITERATIONS_HARD_LIMIT: usize = 100_000;

#[derive(Debug, Clone, PartialEq)]
pub struct CoupledBatteryStepConfig {
    /// Thermal node receiving pack-level I²R heat.
    pub battery_node_id: String,
    pub dt_s: f64,
    pub ambient_temperature_k: f64,
    pub maximum_iterations: usize,
    pub absolute_temperature_tolerance_k: f64,
    pub relative_temperature_tolerance: f64,
    /// Deterministic under-relaxation in (0, 1]. No adaptive fallback is used.
    pub relaxation_factor: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoupledBatteryStepReport {
    pub model_version: &'static str,
    pub iterations: usize,
    /// Temperature at which the converged conservative OCV/R operating point was sampled.
    pub battery_characterization_temperature_k: f64,
    pub battery_node_temperature_end_k: f64,
    pub battery_open_circuit_voltage_v: f64,
    pub battery_internal_resistance_ohm: f64,
    pub required_terminal_power_w: f64,
    pub terminal_voltage_v: f64,
    pub battery_current_a: f64,
    pub battery_resistive_heat_w: f64,
    pub terminal_energy_j: f64,
    pub chemical_energy_j: f64,
    pub battery_resistive_heat_j: f64,
    /// Chemical - terminal - ohmic heat; independently reported from network balance.
    pub battery_energy_balance_residual_j: f64,
    pub thermal_network_energy_balance_residual_j: f64,
    pub temperature_fixed_point_residual_k: f64,
    pub state_of_charge_start: f64,
    pub state_of_charge_end: f64,
    pub committed_thermal_state: ThermalState,
    /// Provenance is preserved but not byte-verified by this function.
    pub battery_characterization_evidence_records: Vec<EvidenceRecord>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoupledBatteryStepError {
    InvalidConfiguration(&'static str),
    InvalidBatteryState(&'static str),
    BatteryNodeMissing(String),
    Characterization(CharacterizationError),
    ThermalNetwork(ThermalNetworkError),
    PowerLimitExceeded { requested_w: f64, maximum_w: f64 },
    CurrentLimitExceeded { requested_a: f64, maximum_a: f64 },
    ElectricalInfeasible {
        requested_power_w: f64,
        open_circuit_voltage_v: f64,
        internal_resistance_ohm: f64,
    },
    NonPositiveTerminalVoltage { voltage_v: f64 },
    BatteryTemperatureLimitExceeded { temperature_k: f64, maximum_k: f64 },
    ReserveInfringed { state_of_charge: f64, reserve: f64 },
    IterationLimit { iterations: usize, last_temperature_residual_k: f64 },
    ElectricalBalanceExceeded { residual_w: f64 },
    BatteryEnergyBalanceExceeded { residual_j: f64 },
    ThermalEnergyBalanceExceeded { residual_j: f64 },
}

impl fmt::Display for CoupledBatteryStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CoupledBatteryStepError {}

/// Solve one battery/thermal substep without mutating caller-owned state.
///
/// `required_terminal_power_w` is a frozen demand at the battery terminals,
/// including all declared propulsion and auxiliary loads. The supplied heat
/// vector contains all non-pack-I²R heat sources, one nonnegative value per
/// thermal node. Pack I²R heat is added exactly once at `battery_node_id`.
///
/// The temperature iteration always replays the thermal network from the
/// same initial state. A state is returned only after the temperature fixed
/// point, electrical identity, reserve, component limits and both energy
/// balances pass. Out-of-domain characterization and non-convergence fail
/// closed; there is no stale-point or nominal-value fallback.
pub fn solve_coupled_battery_substep(
    battery: &BatteryConfig,
    characterization: &BatteryCharacterization,
    network: &ThermalNetwork,
    initial_thermal_state: &ThermalState,
    state_of_charge: f64,
    required_terminal_power_w: f64,
    non_battery_heat_generation_w: &[f64],
    config: &CoupledBatteryStepConfig,
) -> Result<CoupledBatteryStepReport, CoupledBatteryStepError> {
    validate_inputs(
        battery,
        characterization,
        network,
        initial_thermal_state,
        state_of_charge,
        required_terminal_power_w,
        non_battery_heat_generation_w,
        config,
    )?;

    let battery_node_index = network
        .nodes
        .iter()
        .position(|node| node.id == config.battery_node_id)
        .ok_or_else(|| CoupledBatteryStepError::BatteryNodeMissing(config.battery_node_id.clone()))?;

    let mut candidate_temperature_k = initial_thermal_state.temperatures_k[battery_node_index];
    let mut last_temperature_residual_k = 0.0;

    for iteration in 1..=config.maximum_iterations {
        if candidate_temperature_k > battery.maximum_temperature_k {
            return Err(CoupledBatteryStepError::BatteryTemperatureLimitExceeded {
                temperature_k: candidate_temperature_k,
                maximum_k: battery.maximum_temperature_k,
            });
        }

        let operating_point = characterization
            .operating_point(state_of_charge, candidate_temperature_k)
            .map_err(CoupledBatteryStepError::Characterization)?;
        let current_a = current_for_terminal_power(
            required_terminal_power_w,
            operating_point.open_circuit_voltage_v,
            operating_point.internal_resistance_ohm,
        )
        .ok_or(CoupledBatteryStepError::ElectricalInfeasible {
            requested_power_w: required_terminal_power_w,
            open_circuit_voltage_v: operating_point.open_circuit_voltage_v,
            internal_resistance_ohm: operating_point.internal_resistance_ohm,
        })?;

        if current_a > battery.maximum_continuous_current_a + 1e-9 {
            return Err(CoupledBatteryStepError::CurrentLimitExceeded {
                requested_a: current_a,
                maximum_a: battery.maximum_continuous_current_a,
            });
        }

        let terminal_voltage_v = operating_point.open_circuit_voltage_v
            - current_a * operating_point.internal_resistance_ohm;
        if !terminal_voltage_v.is_finite() || terminal_voltage_v <= 0.0 {
            return Err(CoupledBatteryStepError::NonPositiveTerminalVoltage {
                voltage_v: terminal_voltage_v,
            });
        }

        let reconstructed_power_w = current_a * terminal_voltage_v;
        let electrical_balance_residual_w = reconstructed_power_w - required_terminal_power_w;
        let electrical_balance_tolerance_w = 1e-10 * required_terminal_power_w.abs().max(1.0);
        if !electrical_balance_residual_w.is_finite()
            || electrical_balance_residual_w.abs() > electrical_balance_tolerance_w
        {
            return Err(CoupledBatteryStepError::ElectricalBalanceExceeded {
                residual_w: electrical_balance_residual_w,
            });
        }

        let resistive_heat_w = current_a
            * current_a
            * operating_point.internal_resistance_ohm;
        if !resistive_heat_w.is_finite() || resistive_heat_w < 0.0 {
            return Err(CoupledBatteryStepError::InvalidConfiguration(
                "computed pack I²R heat is invalid",
            ));
        }

        let mut heat_generation_w = non_battery_heat_generation_w.to_vec();
        heat_generation_w[battery_node_index] += resistive_heat_w;
        if !heat_generation_w[battery_node_index].is_finite() {
            return Err(CoupledBatteryStepError::InvalidConfiguration(
                "combined battery-node heat overflowed",
            ));
        }

        let thermal_step = network
            .step(
                initial_thermal_state,
                &heat_generation_w,
                config.dt_s,
                config.ambient_temperature_k,
            )
            .map_err(CoupledBatteryStepError::ThermalNetwork)?;
        let next_temperature_k = thermal_step.state.temperatures_k[battery_node_index];
        if next_temperature_k > battery.maximum_temperature_k {
            return Err(CoupledBatteryStepError::BatteryTemperatureLimitExceeded {
                temperature_k: next_temperature_k,
                maximum_k: battery.maximum_temperature_k,
            });
        }

        last_temperature_residual_k = next_temperature_k - candidate_temperature_k;
        let temperature_tolerance_k = config.absolute_temperature_tolerance_k
            + config.relative_temperature_tolerance
                * next_temperature_k.abs().max(candidate_temperature_k.abs());

        if last_temperature_residual_k.abs() <= temperature_tolerance_k {
            let terminal_energy_j = required_terminal_power_w * config.dt_s;
            let battery_resistive_heat_j = resistive_heat_w * config.dt_s;
            let chemical_energy_j =
                current_a * operating_point.open_circuit_voltage_v * config.dt_s;
            let battery_energy_balance_residual_j =
                chemical_energy_j - terminal_energy_j - battery_resistive_heat_j;
            let battery_energy_tolerance_j = 1e-10 * chemical_energy_j.abs().max(1.0);
            if !battery_energy_balance_residual_j.is_finite()
                || battery_energy_balance_residual_j.abs() > battery_energy_tolerance_j
            {
                return Err(CoupledBatteryStepError::BatteryEnergyBalanceExceeded {
                    residual_j: battery_energy_balance_residual_j,
                });
            }

            let thermal_energy_scale_j = thermal_step
                .heat_input_j
                .abs()
                .max(thermal_step.ambient_rejected_energy_j.abs())
                .max(thermal_step.stored_energy_change_j.abs())
                .max(1.0);
            let thermal_energy_tolerance_j = 1e-9 * thermal_energy_scale_j;
            if !thermal_step.energy_balance_residual_j.is_finite()
                || thermal_step.energy_balance_residual_j.abs() > thermal_energy_tolerance_j
            {
                return Err(CoupledBatteryStepError::ThermalEnergyBalanceExceeded {
                    residual_j: thermal_step.energy_balance_residual_j,
                });
            }

            let capacity_j = battery.nominal_energy_wh
                * battery.state_of_health
                * SECONDS_PER_HOUR;
            if !capacity_j.is_finite() || capacity_j <= 0.0 {
                return Err(CoupledBatteryStepError::InvalidBatteryState(
                    "usable pack energy capacity overflowed or is non-positive",
                ));
            }
            let state_of_charge_end =
                state_of_charge - chemical_energy_j / capacity_j;
            if !state_of_charge_end.is_finite()
                || state_of_charge_end < battery.reserve_state_of_charge - 1e-12
            {
                return Err(CoupledBatteryStepError::ReserveInfringed {
                    state_of_charge: state_of_charge_end,
                    reserve: battery.reserve_state_of_charge,
                });
            }

            return Ok(CoupledBatteryStepReport {
                model_version: MODEL_VERSION,
                iterations: iteration,
                battery_characterization_temperature_k: candidate_temperature_k,
                battery_node_temperature_end_k: next_temperature_k,
                battery_open_circuit_voltage_v: operating_point.open_circuit_voltage_v,
                battery_internal_resistance_ohm: operating_point.internal_resistance_ohm,
                required_terminal_power_w,
                terminal_voltage_v,
                battery_current_a: current_a,
                battery_resistive_heat_w: resistive_heat_w,
                terminal_energy_j,
                chemical_energy_j,
                battery_resistive_heat_j,
                battery_energy_balance_residual_j,
                thermal_network_energy_balance_residual_j: thermal_step.energy_balance_residual_j,
                temperature_fixed_point_residual_k: last_temperature_residual_k,
                state_of_charge_start: state_of_charge,
                state_of_charge_end,
                committed_thermal_state: thermal_step.state,
                battery_characterization_evidence_records: characterization.evidence_records(),
            });
        }

        candidate_temperature_k += config.relaxation_factor * last_temperature_residual_k;
        if !candidate_temperature_k.is_finite() || candidate_temperature_k <= 0.0 {
            return Err(CoupledBatteryStepError::InvalidBatteryState(
                "temperature iteration produced a non-positive or non-finite candidate",
            ));
        }
    }

    Err(CoupledBatteryStepError::IterationLimit {
        iterations: config.maximum_iterations,
        last_temperature_residual_k,
    })
}

fn validate_inputs(
    battery: &BatteryConfig,
    characterization: &BatteryCharacterization,
    network: &ThermalNetwork,
    initial_thermal_state: &ThermalState,
    state_of_charge: f64,
    required_terminal_power_w: f64,
    non_battery_heat_generation_w: &[f64],
    config: &CoupledBatteryStepConfig,
) -> Result<(), CoupledBatteryStepError> {
    characterization
        .validate()
        .map_err(CoupledBatteryStepError::Characterization)?;
    network.validate().map_err(CoupledBatteryStepError::ThermalNetwork)?;

    if config.battery_node_id.trim().is_empty()
        || !config.dt_s.is_finite()
        || config.dt_s <= 0.0
        || !config.ambient_temperature_k.is_finite()
        || config.ambient_temperature_k <= 0.0
        || config.maximum_iterations == 0
        || config.maximum_iterations > MAX_ITERATIONS_HARD_LIMIT
        || !config.absolute_temperature_tolerance_k.is_finite()
        || config.absolute_temperature_tolerance_k <= 0.0
        || !config.relative_temperature_tolerance.is_finite()
        || config.relative_temperature_tolerance < 0.0
        || !config.relaxation_factor.is_finite()
        || config.relaxation_factor <= 0.0
        || config.relaxation_factor > 1.0
    {
        return Err(CoupledBatteryStepError::InvalidConfiguration(
            "step configuration, tolerances, iteration bound, or relaxation factor is invalid",
        ));
    }

    if !battery.nominal_energy_wh.is_finite()
        || battery.nominal_energy_wh <= 0.0
        || !battery.state_of_health.is_finite()
        || battery.state_of_health <= 0.0
        || battery.state_of_health > 1.0
        || !battery.maximum_continuous_power_w.is_finite()
        || battery.maximum_continuous_power_w <= 0.0
        || !battery.maximum_continuous_current_a.is_finite()
        || battery.maximum_continuous_current_a <= 0.0
        || !battery.maximum_temperature_k.is_finite()
        || battery.maximum_temperature_k <= 0.0
        || !battery.reserve_state_of_charge.is_finite()
        || !(0.0..=1.0).contains(&battery.reserve_state_of_charge)
    {
        return Err(CoupledBatteryStepError::InvalidBatteryState(
            "battery capacity, SOH, limits, temperature ceiling, or reserve is invalid",
        ));
    }

    if !state_of_charge.is_finite()
        || !(battery.reserve_state_of_charge..=1.0).contains(&state_of_charge)
    {
        return Err(CoupledBatteryStepError::InvalidBatteryState(
            "starting SOC must be finite and at or above reserve, no greater than one",
        ));
    }
    if !required_terminal_power_w.is_finite() || required_terminal_power_w < 0.0 {
        return Err(CoupledBatteryStepError::InvalidConfiguration(
            "required terminal power must be finite and nonnegative",
        ));
    }
    if required_terminal_power_w > battery.maximum_continuous_power_w + 1e-9 {
        return Err(CoupledBatteryStepError::PowerLimitExceeded {
            requested_w: required_terminal_power_w,
            maximum_w: battery.maximum_continuous_power_w,
        });
    }
    if initial_thermal_state.temperatures_k.len() != network.nodes.len()
        || non_battery_heat_generation_w.len() != network.nodes.len()
    {
        return Err(CoupledBatteryStepError::ThermalNetwork(
            ThermalNetworkError::InvalidState(
                "temperature and external heat vectors must match the thermal node count",
            ),
        ));
    }
    for (node, temperature) in network.nodes.iter().zip(&initial_thermal_state.temperatures_k) {
        if !temperature.is_finite() || *temperature <= 0.0 {
            return Err(CoupledBatteryStepError::ThermalNetwork(
                ThermalNetworkError::InvalidState(
                    "initial node temperatures must be finite and above absolute zero",
                ),
            ));
        }
        if *temperature > node.maximum_temperature_k {
            return Err(CoupledBatteryStepError::ThermalNetwork(
                ThermalNetworkError::TemperatureLimitExceeded {
                    node_id: node.id.clone(),
                    temperature_k: *temperature,
                    maximum_temperature_k: node.maximum_temperature_k,
                },
            ));
        }
    }
    if non_battery_heat_generation_w
        .iter()
        .any(|heat| !heat.is_finite() || *heat < 0.0)
    {
        return Err(CoupledBatteryStepError::InvalidConfiguration(
            "non-battery heat sources must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn current_for_terminal_power(
    terminal_power_w: f64,
    open_circuit_voltage_v: f64,
    internal_resistance_ohm: f64,
) -> Option<f64> {
    if !terminal_power_w.is_finite()
        || terminal_power_w < 0.0
        || !open_circuit_voltage_v.is_finite()
        || open_circuit_voltage_v <= 0.0
        || !internal_resistance_ohm.is_finite()
        || internal_resistance_ohm < 0.0
    {
        return None;
    }
    if terminal_power_w == 0.0 {
        return Some(0.0);
    }
    if internal_resistance_ohm == 0.0 {
        return Some(terminal_power_w / open_circuit_voltage_v);
    }

    let discriminant = open_circuit_voltage_v * open_circuit_voltage_v
        - 4.0 * internal_resistance_ohm * terminal_power_w;
    if !discriminant.is_finite() || discriminant < 0.0 {
        return None;
    }
    let denominator = open_circuit_voltage_v + discriminant.sqrt();
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let current_a = 2.0 * terminal_power_w / denominator;
    current_a.is_finite().then_some(current_a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::characterization::{
        BatteryCharacterization, EvidenceClass, EvidenceRecord, ScalarGrid2D, ScalarGridPoint,
    };
    use crate::thermal_network::{ThermalNode, ThermalNetwork};

    fn evidence(id: &str) -> EvidenceRecord {
        EvidenceRecord {
            id: id.into(),
            uri: format!("https://example.invalid/{id}.csv"),
            sha256: "a".repeat(64),
            class: EvidenceClass::Simulation,
            description: "synthetic test fixture; not physical measurement".into(),
        }
    }

    fn grid(id: &str, evidence_id: &str, rows: [[f64; 2]; 2], temperature_max_k: f64) -> ScalarGrid2D {
        ScalarGrid2D {
            id: id.into(),
            x_axis: "state_of_charge".into(),
            x_unit: "1".into(),
            x_values: vec![0.2, 1.0],
            y_axis: "temperature".into(),
            y_unit: "K".into(),
            y_values: vec![280.0, temperature_max_k],
            evidence: evidence(evidence_id),
            values: rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|value| ScalarGridPoint {
                            value: *value,
                            absolute_uncertainty: 0.0,
                        })
                        .collect()
                })
                .collect(),
        }
    }

    fn characterization(resistance_rows: [[f64; 2]; 2], temperature_max_k: f64) -> BatteryCharacterization {
        BatteryCharacterization {
            open_circuit_voltage: grid(
                "pack-ocv",
                "ocv-fixture",
                [[400.0, 400.0], [400.0, 400.0]],
                temperature_max_k,
            ),
            internal_resistance: grid(
                "pack-resistance",
                "resistance-fixture",
                resistance_rows,
                temperature_max_k,
            ),
        }
    }

    fn battery() -> BatteryConfig {
        BatteryConfig {
            nominal_energy_wh: 100_000.0,
            pack_mass_kg: 250.0,
            nominal_open_circuit_voltage_v: 400.0,
            maximum_continuous_power_w: 100_000.0,
            maximum_continuous_current_a: 500.0,
            internal_resistance_ohm: 0.04,
            thermal_capacity_j_per_k: 100_000.0,
            thermal_conductance_w_per_k: 0.0,
            maximum_temperature_k: 600.0,
            initial_temperature_k: 298.15,
            ambient_temperature_k: 298.15,
            state_of_health: 1.0,
            initial_state_of_charge: 0.8,
            reserve_state_of_charge: 0.2,
        }
    }

    fn network(maximum_temperature_k: f64, heat_capacity_j_per_k: f64) -> ThermalNetwork {
        ThermalNetwork {
            nodes: vec![ThermalNode {
                id: "battery".into(),
                heat_capacity_j_per_k,
                ambient_conductance_w_per_k: 0.0,
                maximum_temperature_k,
            }],
            links: Vec::new(),
        }
    }

    fn config(dt_s: f64) -> CoupledBatteryStepConfig {
        CoupledBatteryStepConfig {
            battery_node_id: "battery".into(),
            dt_s,
            ambient_temperature_k: 298.15,
            maximum_iterations: 1_000,
            absolute_temperature_tolerance_k: 1e-10,
            relative_temperature_tolerance: 1e-12,
            relaxation_factor: 0.5,
        }
    }

    #[test]
    fn constant_resistance_step_matches_frozen_analytic_electrical_reference() {
        let result = solve_coupled_battery_substep(
            &battery(),
            &characterization([[0.04, 0.04], [0.04, 0.04]], 360.0),
            &network(360.0, 100_000.0),
            &ThermalState { temperatures_k: vec![298.15] },
            0.8,
            40_000.0,
            &[0.0],
            &config(60.0),
        )
        .unwrap();

        assert!((result.battery_current_a - 101.020_514_433_6).abs() < 1e-9);
        assert!((result.battery_resistive_heat_w - 408.205_773_457_5).abs() < 1e-8);
        assert!(result.battery_energy_balance_residual_j.abs() < 1e-6);
        assert!(result.thermal_network_energy_balance_residual_j.abs() < 1e-6);
        assert!(result.temperature_fixed_point_residual_k.abs() < 1e-9);
        assert_eq!(result.battery_characterization_evidence_records.len(), 2);
    }

    #[test]
    fn resistance_is_resampled_as_temperature_rises_and_feedback_converges() {
        let result = solve_coupled_battery_substep(
            &battery(),
            &characterization([[0.03, 0.05], [0.03, 0.05]], 380.0),
            &network(380.0, 50_000.0),
            &ThermalState { temperatures_k: vec![320.0] },
            0.8,
            20_000.0,
            &[0.0],
            &config(30.0),
        )
        .unwrap();

        assert!(result.iterations > 1);
        assert!(result.battery_node_temperature_end_k > 320.0);
        assert!(result.battery_internal_resistance_ohm > 0.03);
        assert!(result.temperature_fixed_point_residual_k.abs() < 1e-8);
    }

    #[test]
    fn no_load_keeps_current_and_pack_heat_at_zero() {
        let result = solve_coupled_battery_substep(
            &battery(),
            &characterization([[0.04, 0.04], [0.04, 0.04]], 360.0),
            &network(360.0, 100_000.0),
            &ThermalState { temperatures_k: vec![298.15] },
            0.8,
            0.0,
            &[0.0],
            &config(10.0),
        )
        .unwrap();

        assert_eq!(result.battery_current_a, 0.0);
        assert_eq!(result.battery_resistive_heat_w, 0.0);
        assert_eq!(result.state_of_charge_end, 0.8);
        assert_eq!(result.battery_node_temperature_end_k, 298.15);
    }

    #[test]
    fn characterization_out_of_domain_is_not_extrapolated() {
        let result = solve_coupled_battery_substep(
            &battery(),
            &characterization([[0.04, 0.04], [0.04, 0.04]], 310.0),
            &network(360.0, 100_000.0),
            &ThermalState { temperatures_k: vec![320.0] },
            0.8,
            1_000.0,
            &[0.0],
            &config(1.0),
        );
        assert!(matches!(
            result,
            Err(CoupledBatteryStepError::Characterization(
                CharacterizationError::GridOutsideCalibrationDomain { .. }
            ))
        ));
    }

    #[test]
    fn power_current_and_reserve_limits_are_typed() {
        let mut low_power = battery();
        low_power.maximum_continuous_power_w = 5_000.0;
        assert!(matches!(
            solve_coupled_battery_substep(
                &low_power,
                &characterization([[0.04, 0.04], [0.04, 0.04]], 360.0),
                &network(360.0, 100_000.0),
                &ThermalState { temperatures_k: vec![298.15] },
                0.8,
                6_000.0,
                &[0.0],
                &config(1.0),
            ),
            Err(CoupledBatteryStepError::PowerLimitExceeded { .. })
        ));

        let mut low_current = battery();
        low_current.maximum_continuous_current_a = 10.0;
        assert!(matches!(
            solve_coupled_battery_substep(
                &low_current,
                &characterization([[0.04, 0.04], [0.04, 0.04]], 360.0),
                &network(360.0, 100_000.0),
                &ThermalState { temperatures_k: vec![298.15] },
                0.8,
                10_000.0,
                &[0.0],
                &config(1.0),
            ),
            Err(CoupledBatteryStepError::CurrentLimitExceeded { .. })
        ));

        let mut small_capacity = battery();
        small_capacity.nominal_energy_wh = 1_000.0;
        assert!(matches!(
            solve_coupled_battery_substep(
                &small_capacity,
                &characterization([[0.04, 0.04], [0.04, 0.04]], 360.0),
                &network(360.0, 1_000_000.0),
                &ThermalState { temperatures_k: vec![298.15] },
                0.8,
                40_000.0,
                &[0.0],
                &config(360.0),
            ),
            Err(CoupledBatteryStepError::ReserveInfringed { .. })
        ));
    }

    #[test]
    fn thermal_limit_and_iteration_limit_fail_closed_without_a_partial_report() {
        let thermal_failure = solve_coupled_battery_substep(
            &battery(),
            &characterization([[0.04, 0.04], [0.04, 0.04]], 360.0),
            &network(298.2, 100_000.0),
            &ThermalState { temperatures_k: vec![298.15] },
            0.8,
            40_000.0,
            &[0.0],
            &config(60.0),
        );
        assert!(matches!(
            thermal_failure,
            Err(CoupledBatteryStepError::ThermalNetwork(
                ThermalNetworkError::TemperatureLimitExceeded { .. }
            ))
        ));

        let mut slow = config(600.0);
        slow.maximum_iterations = 1;
        slow.relaxation_factor = 0.1;
        slow.absolute_temperature_tolerance_k = 1e-12;
        slow.relative_temperature_tolerance = 0.0;
        let iteration_failure = solve_coupled_battery_substep(
            &battery(),
            &characterization([[0.04, 0.04], [0.04, 0.05]], 590.0),
            &network(590.0, 1_000.0),
            &ThermalState { temperatures_k: vec![298.15] },
            0.8,
            40_000.0,
            &[0.0],
            &slow,
        );
        assert!(matches!(
            iteration_failure,
            Err(CoupledBatteryStepError::IterationLimit { iterations: 1, .. })
        ));
    }

    #[test]
    fn impossible_power_root_is_not_repaired_by_reducing_demand() {
        let result = solve_coupled_battery_substep(
            &battery(),
            &characterization([[0.5, 0.5], [0.5, 0.5]], 360.0),
            &network(360.0, 100_000.0),
            &ThermalState { temperatures_k: vec![298.15] },
            0.8,
            100_000.0,
            &[0.0],
            &config(1.0),
        );
        assert!(matches!(result, Err(CoupledBatteryStepError::ElectricalInfeasible { .. })));
    }
    #[test]
    fn zero_reserve_is_valid_and_energy_capacity_overflow_fails_closed() {
        let mut zero_reserve = battery();
        zero_reserve.reserve_state_of_charge = 0.0;
        let no_load = solve_coupled_battery_substep(
            &zero_reserve,
            &characterization([[0.04, 0.04], [0.04, 0.04]], 360.0),
            &network(360.0, 100_000.0),
            &ThermalState { temperatures_k: vec![298.15] },
            0.8,
            0.0,
            &[0.0],
            &config(1.0),
        )
        .unwrap();
        assert_eq!(no_load.state_of_charge_end, 0.8);

        let mut overflowing_capacity = battery();
        overflowing_capacity.nominal_energy_wh = f64::MAX;
        assert!(matches!(
            solve_coupled_battery_substep(
                &overflowing_capacity,
                &characterization([[0.04, 0.04], [0.04, 0.04]], 360.0),
                &network(360.0, 100_000.0),
                &ThermalState { temperatures_k: vec![298.15] },
                0.8,
                0.0,
                &[0.0],
                &config(1.0),
            ),
            Err(CoupledBatteryStepError::InvalidBatteryState(_))
        ));
    }

}
