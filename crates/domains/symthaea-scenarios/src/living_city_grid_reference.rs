// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Analytical reference checks for the existing Symthaea radial-feeder solver.
//!
//! These are deliberately small model-code/reference-equation checks. They do
//! not establish that linearized DistFlow is adequate for every distribution
//! network, nor do they validate nonlinear AC power flow or outage dynamics.
//! Each case is deterministic and has a closed-form voltage profile.

use symthaea_grid_physics::feeder::{Feeder, Line, Node};

pub const GRID_REFERENCE_SUITE_VERSION: u32 = 2;
pub const GRID_REFERENCE_SUITE_ID: &str = "lindistflow-analytical-reference-suite";
pub const GRID_REFERENCE_CASE_IDS: [&str; 4] = [
    "lindistflow-single-branch-100kw-v1",
    "lindistflow-single-branch-100kvar-v1",
    "lindistflow-two-segment-50kw-v1",
    "lindistflow-single-branch-100kw-der-v1",
];
pub const GRID_REFERENCE_VOLTAGE_TOLERANCE_V: f64 = 1e-6;
pub const GRID_REFERENCE_POWER_TOLERANCE_KW: f64 = 1e-9;
pub const GRID_REFERENCE_VALIDATION_KIND: &str = "analytical_equation_regression";
pub const GRID_REFERENCE_SOLVER_ENVELOPE: &str =
    "linearized DistFlow; single-phase equivalent; balanced radial topology; quadratic branch-loss term omitted; no unbalanced phases, meshed networks, transient/dynamic stability, or nonlinear AC validation";

#[derive(Debug, Clone, PartialEq)]
pub struct GridReferenceResult {
    pub suite_version: u32,
    pub suite_id: &'static str,
    pub case_id: &'static str,
    pub validation_kind: &'static str,
    pub solver_envelope: &'static str,
    /// Analytical voltage at the terminal/monitored bus.
    pub analytical_voltage_v: f64,
    /// Solver voltage at the same bus.
    pub solver_voltage_v: f64,
    pub absolute_voltage_error_v: f64,
    /// Expected branch power at the terminal/monitored bus.
    pub expected_branch_power_kw: f64,
    pub solver_branch_power_kw: f64,
    pub absolute_branch_power_error_kw: f64,
    pub checked_bus_count: usize,
    pub max_absolute_voltage_error_v: f64,
    pub max_absolute_branch_power_error_kw: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridReferenceError {
    pub case_id: &'static str,
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for GridReferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "grid reference case {} failed at {}: {}",
            self.case_id, self.field, self.reason
        )
    }
}

impl std::error::Error for GridReferenceError {}

fn invalid(
    case_id: &'static str,
    field: &'static str,
    reason: &'static str,
) -> GridReferenceError {
    GridReferenceError {
        case_id,
        field,
        reason,
    }
}

/// Check all bus voltages and expected branch real-power values against
/// analytical LinDistFlow results. None means the root/slack bus has no branch
/// power assigned at that index; non-root branches need an expected value.
fn solve_and_check(
    case_id: &'static str,
    feeder: Feeder,
    expected_voltage_squared_by_bus: &[f64],
    expected_branch_power_kw_by_bus: &[Option<f64>],
) -> Result<GridReferenceResult, GridReferenceError> {
    if expected_voltage_squared_by_bus.len() != feeder.nodes.len()
        || expected_branch_power_kw_by_bus.len() != feeder.nodes.len()
    {
        return Err(invalid(
            case_id,
            "reference_dimensions",
            "reference vectors must have exactly one entry per feeder bus",
        ));
    }
    if expected_voltage_squared_by_bus
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(invalid(
            case_id,
            "analytical_voltage_squared",
            "all analytical bus voltage squares must be finite and positive",
        ));
    }

    let solution = feeder.solve();
    let mut max_voltage_error_v: f64 = 0.0;
    let mut max_branch_power_error_kw: f64 = 0.0;
    let mut terminal_expected_branch_power = None;
    let mut terminal_solver_branch_power = None;
    let last_bus = feeder.nodes.len() - 1;

    for (bus, expected_voltage_squared) in
        expected_voltage_squared_by_bus.iter().copied().enumerate()
    {
        let expected_voltage_v = expected_voltage_squared.sqrt();
        let actual_voltage_v = solution.voltage_v(bus);
        if !actual_voltage_v.is_finite() {
            return Err(invalid(
                case_id,
                "voltage_v",
                "solver returned a non-finite bus voltage",
            ));
        }
        let error_v = (actual_voltage_v - expected_voltage_v).abs();
        max_voltage_error_v = max_voltage_error_v.max(error_v);

        if let Some(expected_power_kw) = expected_branch_power_kw_by_bus[bus] {
            if !expected_power_kw.is_finite() {
                return Err(invalid(
                    case_id,
                    "expected_branch_power_kw",
                    "expected branch power must be finite",
                ));
            }
            let actual_power_kw = solution.branch_power_kw[bus];
            if !actual_power_kw.is_finite() {
                return Err(invalid(
                    case_id,
                    "branch_power_kw",
                    "solver returned non-finite branch power",
                ));
            }
            let error_kw = (actual_power_kw - expected_power_kw).abs();
            max_branch_power_error_kw = max_branch_power_error_kw.max(error_kw);
            if bus == last_bus {
                terminal_expected_branch_power = Some(expected_power_kw);
                terminal_solver_branch_power = Some(actual_power_kw);
            }
        }
    }

    if max_voltage_error_v > GRID_REFERENCE_VOLTAGE_TOLERANCE_V {
        return Err(invalid(
            case_id,
            "voltage_v",
            "solver voltage profile exceeds the analytical reference tolerance",
        ));
    }
    if max_branch_power_error_kw > GRID_REFERENCE_POWER_TOLERANCE_KW {
        return Err(invalid(
            case_id,
            "branch_power_kw",
            "solver branch-power aggregation exceeds the analytical reference tolerance",
        ));
    }

    let terminal_expected_branch_power = terminal_expected_branch_power.ok_or_else(|| {
        invalid(
            case_id,
            "expected_branch_power_kw",
            "terminal bus must declare an expected branch power",
        )
    })?;
    let terminal_solver_branch_power = terminal_solver_branch_power.ok_or_else(|| {
        invalid(
            case_id,
            "branch_power_kw",
            "terminal bus branch power was not checked",
        )
    })?;
    let analytical_voltage_v = expected_voltage_squared_by_bus[last_bus].sqrt();
    let solver_voltage_v = solution.voltage_v(last_bus);

    Ok(GridReferenceResult {
        suite_version: GRID_REFERENCE_SUITE_VERSION,
        suite_id: GRID_REFERENCE_SUITE_ID,
        case_id,
        validation_kind: GRID_REFERENCE_VALIDATION_KIND,
        solver_envelope: GRID_REFERENCE_SOLVER_ENVELOPE,
        analytical_voltage_v,
        solver_voltage_v,
        absolute_voltage_error_v: (solver_voltage_v - analytical_voltage_v).abs(),
        expected_branch_power_kw: terminal_expected_branch_power,
        solver_branch_power_kw: terminal_solver_branch_power,
        absolute_branch_power_error_kw: (terminal_solver_branch_power
            - terminal_expected_branch_power)
            .abs(),
        checked_bus_count: feeder.nodes.len(),
        max_absolute_voltage_error_v: max_voltage_error_v,
        max_absolute_branch_power_error_kw: max_branch_power_error_kw,
    })
}

/// Case 1: one branch, 7.2 kV base, 1 ohm resistance, 100 kW real load.
pub fn run_grid_reference_case() -> Result<GridReferenceResult, GridReferenceError> {
    run_real_power_case()
}

/// Run the complete versioned analytical suite. Any reference mismatch fails
/// the suite; results are emitted in the fixed declared case order.
pub fn run_grid_reference_suite() -> Result<Vec<GridReferenceResult>, GridReferenceError> {
    Ok(vec![
        run_real_power_case()?,
        run_reactive_power_case()?,
        run_two_segment_case()?,
        run_der_injection_case()?,
    ])
}

fn run_real_power_case() -> Result<GridReferenceResult, GridReferenceError> {
    let feeder = Feeder::new(
        7_200.0,
        vec![
            Node::root(),
            Node::load(
                0,
                Line {
                    resistance_ohm: 1.0,
                    reactance_ohm: 0.0,
                },
                100.0,
                0.0,
            ),
        ],
    )
    .map_err(|_| invalid(
        GRID_REFERENCE_CASE_IDS[0],
        "feeder_topology",
        "reference feeder topology was rejected",
    ))?;

    // V_child^2 = 7200^2 - 2 * (1 * 100_000) = 51_640_000.
    solve_and_check(
        GRID_REFERENCE_CASE_IDS[0],
        feeder,
        &[7_200.0_f64.powi(2), 51_640_000.0],
        &[None, Some(100.0)],
    )
}

fn run_reactive_power_case() -> Result<GridReferenceResult, GridReferenceError> {
    let feeder = Feeder::new(
        7_200.0,
        vec![
            Node::root(),
            Node::load(
                0,
                Line {
                    resistance_ohm: 0.0,
                    reactance_ohm: 1.0,
                },
                0.0,
                100.0,
            ),
        ],
    )
    .map_err(|_| invalid(
        GRID_REFERENCE_CASE_IDS[1],
        "feeder_topology",
        "reference feeder topology was rejected",
    ))?;

    // V_child^2 = 7200^2 - 2 * (1 * 100_000) = 51_640_000.
    solve_and_check(
        GRID_REFERENCE_CASE_IDS[1],
        feeder,
        &[7_200.0_f64.powi(2), 51_640_000.0],
        &[None, Some(0.0)],
    )
}

fn run_two_segment_case() -> Result<GridReferenceResult, GridReferenceError> {
    let feeder = Feeder::new(
        12_000.0,
        vec![
            Node::root(),
            Node::load(
                0,
                Line {
                    resistance_ohm: 0.5,
                    reactance_ohm: 0.0,
                },
                0.0,
                0.0,
            ),
            Node::load(
                1,
                Line {
                    resistance_ohm: 0.5,
                    reactance_ohm: 0.0,
                },
                50.0,
                0.0,
            ),
        ],
    )
    .map_err(|_| invalid(
        GRID_REFERENCE_CASE_IDS[2],
        "feeder_topology",
        "reference feeder topology was rejected",
    ))?;

    // A and B both carry the downstream 50 kW. Each 0.5 ohm segment drops
    // 2 * 0.5 * 50_000 = 50_000 V^2 in the declared LinDistFlow equation.
    solve_and_check(
        GRID_REFERENCE_CASE_IDS[2],
        feeder,
        &[144_000_000.0, 143_950_000.0, 143_900_000.0],
        &[None, Some(50.0), Some(50.0)],
    )
}

fn run_der_injection_case() -> Result<GridReferenceResult, GridReferenceError> {
    let feeder = Feeder::new(
        7_200.0,
        vec![
            Node::root(),
            Node::load(
                0,
                Line {
                    resistance_ohm: 1.0,
                    reactance_ohm: 0.0,
                },
                -100.0,
                0.0,
            ),
        ],
    )
    .map_err(|_| invalid(
        GRID_REFERENCE_CASE_IDS[3],
        "feeder_topology",
        "reference feeder topology was rejected",
    ))?;

    // Negative net load means 100 kW of net distributed generation:
    // V_child^2 = 7200^2 - 2 * (1 * -100_000) = 52_040_000.
    solve_and_check(
        GRID_REFERENCE_CASE_IDS[3],
        feeder,
        &[7_200.0_f64.powi(2), 52_040_000.0],
        &[None, Some(-100.0)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_branch_solver_matches_closed_form_lindistflow_reference() {
        let result = run_grid_reference_case().unwrap();
        assert_eq!(result.suite_version, GRID_REFERENCE_SUITE_VERSION);
        assert_eq!(result.case_id, GRID_REFERENCE_CASE_IDS[0]);
        assert!(result.solver_voltage_v < 7_200.0);
        assert!(result.absolute_voltage_error_v <= GRID_REFERENCE_VOLTAGE_TOLERANCE_V);
        assert!(result.absolute_branch_power_error_kw <= GRID_REFERENCE_POWER_TOLERANCE_KW);
        assert_eq!(result.solver_branch_power_kw, 100.0);
        assert!((result.analytical_voltage_v.powi(2) - 51_640_000.0).abs() < 1e-6);
    }

    #[test]
    fn all_reference_cases_match_their_closed_form_results() {
        let results = run_grid_reference_suite().unwrap();
        assert_eq!(results.len(), GRID_REFERENCE_CASE_IDS.len());
        assert_eq!(
            results.iter().map(|result| result.case_id).collect::<Vec<_>>(),
            GRID_REFERENCE_CASE_IDS.to_vec()
        );
        assert!(results.iter().all(|result| {
            result.max_absolute_voltage_error_v <= GRID_REFERENCE_VOLTAGE_TOLERANCE_V
                && result.max_absolute_branch_power_error_kw
                    <= GRID_REFERENCE_POWER_TOLERANCE_KW
        }));
    }

    #[test]
    fn reactive_power_case_checks_voltage_without_real_power_flow() {
        let result = run_reactive_power_case().unwrap();
        assert_eq!(result.solver_branch_power_kw, 0.0);
        assert!(result.analytical_voltage_v < 7_200.0);
    }

    #[test]
    fn two_segment_case_checks_downstream_power_aggregation() {
        let result = run_two_segment_case().unwrap();
        assert_eq!(result.solver_branch_power_kw, 50.0);
        assert_eq!(result.checked_bus_count, 3);
        assert!(result.solver_voltage_v < 12_000.0);
    }

    #[test]
    fn der_injection_case_checks_voltage_rise() {
        let result = run_der_injection_case().unwrap();
        assert_eq!(result.solver_branch_power_kw, -100.0);
        assert!(result.solver_voltage_v > 7_200.0);
    }
}
