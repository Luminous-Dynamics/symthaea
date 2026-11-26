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
use symthaea_plasma_probes::project_nonnegative_preserving_mass;

#[derive(Debug, Deserialize)]
struct ProjectionInput {
    schema_version: String,
    case_id: String,
    input_values: Vec<f64>,
    cell_volumes: Vec<f64>,
    target_mass: f64,
    expected: ProjectionExpected,
}

#[derive(Debug, Deserialize)]
struct ProjectionExpected {
    projected_values: Vec<f64>,
    achieved_mass: f64,
    mass_residual: f64,
    lagrange_multiplier: f64,
    active_positive_cell_count: usize,
    weighted_correction_norm: f64,
    minimum_projected_value: f64,
}

fn assert_close(actual: f64, expected: f64, field: &str) -> Result<(), Box<dyn Error>> {
    let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-10);
    if !actual.is_finite() || (actual - expected).abs() > tolerance {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            format!(
                "{field} mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"
            ),
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
            "usage: positive_mass_projection_audit <record.json>",
        )
    })?;
    if args.next().is_some() {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: positive_mass_projection_audit <record.json>",
        )
        .into());
    }

    let input_text = fs::read_to_string(&input_path)?;
    let input: ProjectionInput = serde_json::from_str(&input_text)?;
    if input.schema_version != "0.1" {
        return Err(IoError::new(ErrorKind::InvalidData, "unsupported schema_version").into());
    }

    let report = project_nonnegative_preserving_mass(
        &input.input_values,
        &input.cell_volumes,
        input.target_mass,
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
    assert_close(
        report.achieved_mass,
        input.expected.achieved_mass,
        "achieved_mass",
    )?;
    assert_close(
        report.mass_residual,
        input.expected.mass_residual,
        "mass_residual",
    )?;
    assert_close(
        report.lagrange_multiplier,
        input.expected.lagrange_multiplier,
        "lagrange_multiplier",
    )?;
    assert_close(
        report.weighted_correction_norm,
        input.expected.weighted_correction_norm,
        "weighted_correction_norm",
    )?;
    assert_close(
        report.minimum_projected_value,
        input.expected.minimum_projected_value,
        "minimum_projected_value",
    )?;
    if report.active_positive_cell_count != input.expected.active_positive_cell_count {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "active_positive_cell_count differs from expected fixture",
        )
        .into());
    }

    let output = serde_json::json!({
        "audit_status": "PASS",
        "known_answer_verified": true,
        "case_id": input.case_id,
        "claim_boundary": "synthetic mass-and-positivity projection only",
        "report": report
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("positive_mass_projection_audit=FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
