// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Differential tests against the separately maintained scalar reference oracle.
//!
//! The oracle source is compiled as a test-only module. It does not share the
//! production battery fixed-point iteration or multi-node thermal network.
//! The matched cases deliberately reduce the production network to one node
//! and freeze OCV so both implementations solve the same mathematical plant.

#[path = "../../../../tools/electric-aircraft-coupled-oracle/src/lib.rs"]
mod reference_oracle;

use reference_oracle::{
    OracleConfig, OracleState, SubstepInput, solve_substep_bisection,
};
use symthaea_electric_aircraft::characterization::{
    BatteryCharacterization, EvidenceClass, EvidenceRecord, ScalarGrid2D, ScalarGridPoint,
};
use symthaea_electric_aircraft::coupled_battery::{
    CoupledBatteryStepConfig, CoupledBatteryStepReport, solve_coupled_battery_substep,
};
use symthaea_electric_aircraft::thermal_network::{
    ThermalNetwork, ThermalNode, ThermalState,
};
use symthaea_electric_aircraft::BatteryConfig;

const REFERENCE_TEMPERATURE_K: f64 = 298.15;
const MINIMUM_TEMPERATURE_K: f64 = 280.0;
const MAXIMUM_TEMPERATURE_K: f64 = 360.0;
const INITIAL_SOC: f64 = 0.8;
const RESERVE_SOC: f64 = 0.2;
const NOMINAL_ENERGY_WH: f64 = 100_000.0;
const STATE_OF_HEALTH: f64 = 0.9;
const OCV_V: f64 = 400.0;
const RESISTANCE_REFERENCE_OHM: f64 = 0.04;

struct MatchedCase {
    battery: BatteryConfig,
    characterization: BatteryCharacterization,
    network: ThermalNetwork,
    initial_state: ThermalState,
    production_config: CoupledBatteryStepConfig,
    oracle_config: OracleConfig,
    oracle_state: OracleState,
    terminal_power_w: f64,
    duration_s: f64,
}

fn evidence(id: &str) -> EvidenceRecord {
    EvidenceRecord {
        id: id.into(),
        uri: format!("https://example.invalid/{id}.csv"),
        sha256: "a".repeat(64),
        class: EvidenceClass::Simulation,
        description: "synthetic cross-implementation contract fixture".into(),
    }
}

fn grid(
    id: &str,
    evidence_id: &str,
    values: [[f64; 2]; 2],
) -> ScalarGrid2D {
    ScalarGrid2D {
        id: id.into(),
        x_axis: "state_of_charge".into(),
        x_unit: "1".into(),
        x_values: vec![0.2, 1.0],
        y_axis: "temperature".into(),
        y_unit: "K".into(),
        y_values: vec![MINIMUM_TEMPERATURE_K, MAXIMUM_TEMPERATURE_K],
        evidence: evidence(evidence_id),
        values: values
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

fn matched_case(
    initial_temperature_k: f64,
    thermal_capacity_j_per_k: f64,
    thermal_conductance_w_per_k: f64,
    resistance_temperature_coefficient_per_k: f64,
    terminal_power_w: f64,
    duration_s: f64,
) -> MatchedCase {
    let resistance_at = |temperature_k: f64| {
        RESISTANCE_REFERENCE_OHM
            * (1.0 + resistance_temperature_coefficient_per_k
                * (temperature_k - REFERENCE_TEMPERATURE_K))
    };
    let resistance_values = [
        resistance_at(MINIMUM_TEMPERATURE_K),
        resistance_at(MAXIMUM_TEMPERATURE_K),
    ];

    MatchedCase {
        battery: BatteryConfig {
            nominal_energy_wh: NOMINAL_ENERGY_WH,
            pack_mass_kg: 250.0,
            nominal_open_circuit_voltage_v: OCV_V,
            maximum_continuous_power_w: 150_000.0,
            maximum_continuous_current_a: 500.0,
            internal_resistance_ohm: RESISTANCE_REFERENCE_OHM,
            thermal_capacity_j_per_k,
            thermal_conductance_w_per_k,
            maximum_temperature_k: MAXIMUM_TEMPERATURE_K,
            initial_temperature_k,
            ambient_temperature_k: REFERENCE_TEMPERATURE_K,
            state_of_health: STATE_OF_HEALTH,
            initial_state_of_charge: INITIAL_SOC,
            reserve_state_of_charge: RESERVE_SOC,
        },
        characterization: BatteryCharacterization {
            open_circuit_voltage: grid(
                "ocv-grid",
                "ocv-synthetic-fixture",
                [[OCV_V, OCV_V], [OCV_V, OCV_V]],
            ),
            internal_resistance: grid(
                "resistance-grid",
                "resistance-synthetic-fixture",
                [
                    [resistance_values[0], resistance_values[1]],
                    [resistance_values[0], resistance_values[1]],
                ],
            ),
        },
        network: ThermalNetwork {
            nodes: vec![ThermalNode {
                id: "battery".into(),
                heat_capacity_j_per_k,
                ambient_conductance_w_per_k: thermal_conductance_w_per_k,
                maximum_temperature_k: MAXIMUM_TEMPERATURE_K,
            }],
            links: Vec::new(),
        },
        initial_state: ThermalState {
            temperatures_k: vec![initial_temperature_k],
        },
        production_config: CoupledBatteryStepConfig {
            battery_node_id: "battery".into(),
            dt_s: duration_s,
            ambient_temperature_k: REFERENCE_TEMPERATURE_K,
            maximum_iterations: 1_000,
            absolute_temperature_tolerance_k: 1e-11,
            relative_temperature_tolerance: 1e-13,
            relaxation_factor: 0.5,
        },
        oracle_config: OracleConfig {
            capacity_wh: NOMINAL_ENERGY_WH * STATE_OF_HEALTH,
            nominal_ocv_v: OCV_V,
            ocv_soc_slope_v: 0.0,
            soc_reference: 0.5,
            resistance_reference_ohm: RESISTANCE_REFERENCE_OHM,
            resistance_temperature_coefficient_per_k,
            reference_temperature_k: REFERENCE_TEMPERATURE_K,
            thermal_capacity_j_per_k,
            thermal_conductance_w_per_k,
            ambient_temperature_k: REFERENCE_TEMPERATURE_K,
            minimum_temperature_k: MINIMUM_TEMPERATURE_K,
            maximum_temperature_k: MAXIMUM_TEMPERATURE_K,
            reserve_soc: RESERVE_SOC,
            maximum_continuous_power_w: 150_000.0,
            maximum_continuous_current_a: 500.0,
            temperature_tolerance_k: 1e-10,
            maximum_iterations: 1_000,
            energy_balance_tolerance_j: 1e-3,
        },
        oracle_state: OracleState {
            temperature_k: initial_temperature_k,
            soc: INITIAL_SOC,
        },
        terminal_power_w,
        duration_s,
    }
}

fn solve_both(case: &MatchedCase) -> (CoupledBatteryStepReport, reference_oracle::SubstepReport) {
    let production = solve_coupled_battery_substep(
        &case.battery,
        &case.characterization,
        &case.network,
        &case.initial_state,
        INITIAL_SOC,
        case.terminal_power_w,
        &[0.0],
        &case.production_config,
    )
    .unwrap();

    let reference = solve_substep_bisection(
        &case.oracle_config,
        case.oracle_state,
        SubstepInput {
            terminal_power_w: case.terminal_power_w,
            duration_s: case.duration_s,
        },
    )
    .unwrap();

    (production, reference)
}

fn assert_close(name: &str, production: f64, reference: f64, tolerance: f64) {
    assert!(
        (production - reference).abs() <= tolerance,
        "{name} mismatch: production={production:.15e}, reference={reference:.15e}, tolerance={tolerance:.3e}"
    );
}

fn assert_reports_match(
    production: &CoupledBatteryStepReport,
    reference: &reference_oracle::SubstepReport,
) {
    assert_close(
        "open-circuit voltage",
        production.battery_open_circuit_voltage_v,
        reference.operating_point.open_circuit_voltage_v,
        1e-10,
    );
    assert_close(
        "internal resistance",
        production.battery_internal_resistance_ohm,
        reference.operating_point.resistance_ohm,
        1e-9,
    );
    assert_close(
        "battery current",
        production.battery_current_a,
        reference.operating_point.current_a,
        1e-7,
    );
    assert_close(
        "terminal voltage",
        production.terminal_voltage_v,
        reference.operating_point.terminal_voltage_v,
        1e-7,
    );
    assert_close(
        "terminal energy",
        production.terminal_energy_j,
        reference.terminal_energy_wh * 3_600.0,
        1e-5,
    );
    assert_close(
        "chemical energy",
        production.chemical_energy_j,
        reference.chemical_energy_wh * 3_600.0,
        1e-5,
    );
    assert_close(
        "resistive heat energy",
        production.battery_resistive_heat_j,
        reference.resistive_heat_j,
        1e-5,
    );
    assert_close(
        "end temperature",
        production.battery_node_temperature_end_k,
        reference.next_state.temperature_k,
        2e-8,
    );
    assert_close(
        "end SOC",
        production.state_of_charge_end,
        reference.next_state.soc,
        2e-10,
    );
}

#[test]
fn production_substep_matches_independent_oracle_for_constant_resistance() {
    let case = matched_case(
        REFERENCE_TEMPERATURE_K,
        100_000.0,
        25.0,
        0.0,
        40_000.0,
        60.0,
    );
    let (production, reference) = solve_both(&case);
    assert_reports_match(&production, &reference);
}

#[test]
fn production_substep_matches_bisection_oracle_with_temperature_resistance_feedback() {
    let case = matched_case(
        320.0,
        50_000.0,
        10.0,
        0.002,
        20_000.0,
        30.0,
    );
    let (production, reference) = solve_both(&case);
    assert!(production.iterations > 1);
    assert!(reference.temperature_iterations > 1);
    assert_reports_match(&production, &reference);
}
