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
    let output = serde_json::json!({
        "audit_status": "PASS",
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
