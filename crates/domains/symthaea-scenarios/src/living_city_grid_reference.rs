// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Analytical reference check for the existing Symthaea radial-feeder solver.
//!
//! This is deliberately a small model-code/reference-equation check. It does
//! not establish that linearized DistFlow is adequate for every distribution
//! network, nor does it validate nonlinear AC power flow or grid-outage dynamics.
//! The reference case is a single branch with a closed-form voltage solution.

use symthaea_grid_physics::feeder::{Feeder, Line, Node};

pub const GRID_REFERENCE_SUITE_VERSION: u32 = 1;
pub const GRID_REFERENCE_CASE_ID: &str = "lindistflow-single-branch-100kw-v1";
pub const GRID_REFERENCE_VOLTAGE_TOLERANCE_V: f64 = 1e-6;
pub const GRID_REFERENCE_POWER_TOLERANCE_KW: f64 = 1e-9;

const BASE_VOLTAGE_V: f64 = 7_200.0;
const LINE_RESISTANCE_OHM: f64 = 1.0;
const LINE_REACTANCE_OHM: f64 = 0.0;
const LOAD_KW: f64 = 100.0;
const LOAD_KVAR: f64 = 0.0;

#[derive(Debug, Clone, PartialEq)]
pub struct GridReferenceResult {
    pub suite_version: u32,
    pub case_id: &'static str,
    pub analytical_voltage_v: f64,
    pub solver_voltage_v: f64,
    pub absolute_voltage_error_v: f64,
    pub expected_branch_power_kw: f64,
    pub solver_branch_power_kw: f64,
    pub absolute_branch_power_error_kw: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridReferenceError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for GridReferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "grid reference case {} failed: {}", self.field, self.reason)
    }
}

impl std::error::Error for GridReferenceError {}

fn invalid(field: &'static str, reason: &'static str) -> GridReferenceError {
    GridReferenceError { field, reason }
}

/// Run the fixed, versioned analytical feeder reference case.
///
/// LinDistFlow single-branch reference:
/// V_child^2 = V_base^2 - 2 * (R * P_watts + X * Q_var).
/// The solver is accepted only if voltage and branch-power outputs agree with
/// this independent closed-form calculation within the fixed regression tolerances.
pub fn run_grid_reference_case() -> Result<GridReferenceResult, GridReferenceError> {
    let expected_voltage_squared = BASE_VOLTAGE_V.powi(2)
        - 2.0
            * (LINE_RESISTANCE_OHM * LOAD_KW * 1_000.0
                + LINE_REACTANCE_OHM * LOAD_KVAR * 1_000.0);
    if !expected_voltage_squared.is_finite() || expected_voltage_squared <= 0.0 {
        return Err(invalid(
            "analytical_voltage_squared",
            "reference equation produced a non-positive or non-finite result",
        ));
    }
    let analytical_voltage_v = expected_voltage_squared.sqrt();

    let feeder = Feeder::new(
        BASE_VOLTAGE_V,
        vec![
            Node::root(),
            Node::load(
                0,
                Line {
                    resistance_ohm: LINE_RESISTANCE_OHM,
                    reactance_ohm: LINE_REACTANCE_OHM,
                },
                LOAD_KW,
                LOAD_KVAR,
            ),
        ],
    )
    .map_err(|_| invalid("feeder_topology", "reference feeder topology was rejected"))?;
    let solution = feeder.solve();
    let solver_voltage_v = solution.voltage_v(1);
    let solver_branch_power_kw = solution.branch_power_kw[1];
    let absolute_voltage_error_v = (solver_voltage_v - analytical_voltage_v).abs();
    let absolute_branch_power_error_kw = (solver_branch_power_kw - LOAD_KW).abs();

    if !solver_voltage_v.is_finite()
        || absolute_voltage_error_v > GRID_REFERENCE_VOLTAGE_TOLERANCE_V
    {
        return Err(invalid(
            "voltage_v",
            "solver voltage differs from the analytical reference beyond the declared tolerance",
        ));
    }
    if !solver_branch_power_kw.is_finite()
        || absolute_branch_power_error_kw > GRID_REFERENCE_POWER_TOLERANCE_KW
    {
        return Err(invalid(
            "branch_power_kw",
            "solver branch-power aggregation differs from the analytical load beyond tolerance",
        ));
    }

    Ok(GridReferenceResult {
        suite_version: GRID_REFERENCE_SUITE_VERSION,
        case_id: GRID_REFERENCE_CASE_ID,
        analytical_voltage_v,
        solver_voltage_v,
        absolute_voltage_error_v,
        expected_branch_power_kw: LOAD_KW,
        solver_branch_power_kw,
        absolute_branch_power_error_kw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_branch_solver_matches_closed_form_lindistflow_reference() {
        let result = run_grid_reference_case().unwrap();
        assert_eq!(result.suite_version, 1);
        assert_eq!(result.case_id, GRID_REFERENCE_CASE_ID);
        assert!(result.solver_voltage_v < BASE_VOLTAGE_V);
        assert!(
            result.absolute_voltage_error_v <= GRID_REFERENCE_VOLTAGE_TOLERANCE_V,
            "voltage error {} V exceeded tolerance",
            result.absolute_voltage_error_v
        );
        assert!(
            result.absolute_branch_power_error_kw <= GRID_REFERENCE_POWER_TOLERANCE_KW,
            "branch power error {} kW exceeded tolerance",
            result.absolute_branch_power_error_kw
        );
        assert_eq!(result.solver_branch_power_kw, 100.0);
        assert!(
            (result.analytical_voltage_v.powi(2) - 51_640_000.0).abs() < 1e-6,
            "analytical squared voltage should follow the fixed reference equation"
        );
    }
}
