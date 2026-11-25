// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::{
    env,
    error::Error,
    fs,
    io::{Error as IoError, ErrorKind},
    process::ExitCode,
};

use serde::Deserialize;
use symthaea_plasma_probes::solve_small_positive_mass_response_qp;

#[derive(Debug, Deserialize)]
struct QpInput {
    schema_version: String,
    case_id: String,
    input_values: Vec<f64>,
    cell_volumes: Vec<f64>,
    response_coefficients: Vec<f64>,
    target_mass: f64,
    target_response: f64,
    expected: QpExpected,
}

#[derive(Debug, Deserialize)]
struct QpExpected {
    projected_values: Vec<f64>,
    positive_cell_count: usize,
    achieved_mass: f64,
    mass_residual: f64,
    achieved_response: f64,
    response_residual: f64,
    lagrange_multipliers: [f64; 2],
    weighted_correction_norm: f64,
    objective_value: f64,
    maximum_primal_violation: f64,
    maximum_dual_violation: f64,
    maximum_stationarity_residual: f64,
    maximum_complementarity_residual: f64,
}

fn assert_close(actual: f64, expected: f64, field: &str) -> Result<(), Box<dyn Error>> {
    let tolerance = 1.0e-10_f64.max(expected.abs() * 1.0e-9);
    if !actual.is_finite() || (actual - expected).abs() > tolerance {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!("{field} mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"),
        )
        .into());
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let input_path = args.next().ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            "usage: small_positive_response_qp_audit <record.json>",
        )
    })?;
    if args.next().is_some() {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: small_positive_response_qp_audit <record.json>",
        )
        .into());
    }

    let input_text = fs::read_to_string(&input_path)?;
    let input: QpInput = serde_json::from_str(&input_text)?;
    if input.schema_version != "0.1" {
        return Err(IoError::new(ErrorKind::InvalidData, "unsupported schema_version").into());
    }

    let report = solve_small_positive_mass_response_qp(
        &input.input_values,
        &input.cell_volumes,
        &input.response_coefficients,
        input.target_mass,
        input.target_response,
    )?;
    if report.projected_values.len() != input.expected.projected_values.len() {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "projected value count differs from expected fixture",
        )
        .into());
    }
    for (index, (actual, expected)) in report
        .projected_values
        .iter()
        .zip(&input.expected.projected_values)
        .enumerate()
    {
        assert_close(*actual, *expected, &format!("projected_values[{index}]"))?;
    }
    for (actual, expected, field) in [
        (report.achieved_mass, input.expected.achieved_mass, "achieved_mass"),
        (report.mass_residual, input.expected.mass_residual, "mass_residual"),
        (report.achieved_response, input.expected.achieved_response, "achieved_response"),
        (report.response_residual, input.expected.response_residual, "response_residual"),
        (
            report.lagrange_multipliers[0],
            input.expected.lagrange_multipliers[0],
            "lagrange_multiplier_mass",
        ),
        (
            report.lagrange_multipliers[1],
            input.expected.lagrange_multipliers[1],
            "lagrange_multiplier_response",
        ),
        (
            report.weighted_correction_norm,
            input.expected.weighted_correction_norm,
            "weighted_correction_norm",
        ),
        (report.objective_value, input.expected.objective_value, "objective_value"),
        (
            report.maximum_primal_violation,
            input.expected.maximum_primal_violation,
            "maximum_primal_violation",
        ),
        (
            report.maximum_dual_violation,
            input.expected.maximum_dual_violation,
            "maximum_dual_violation",
        ),
        (
            report.maximum_stationarity_residual,
            input.expected.maximum_stationarity_residual,
            "maximum_stationarity_residual",
        ),
        (
            report.maximum_complementarity_residual,
            input.expected.maximum_complementarity_residual,
            "maximum_complementarity_residual",
        ),
    ] {
        assert_close(actual, expected, field)?;
    }
    if report.positive_cell_count != input.expected.positive_cell_count {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "positive cell count differs from expected fixture",
        )
        .into());
    }

    let output = serde_json::json!({
        "audit_status": "PASS",
        "known_answer_verified": true,
        "case_id": input.case_id,
        "claim_boundary": "bounded small-system QP reference only; no physical solver result",
        "report": report
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("small_positive_response_qp_audit=FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
