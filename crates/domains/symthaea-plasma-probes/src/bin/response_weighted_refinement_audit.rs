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
use symthaea_plasma_probes::{
    ComplexResponse, ModeResponseMatrix, plan_response_weighted_refinement,
};

#[derive(Debug, Deserialize)]
struct RefinementInput {
    schema_version: String,
    case_id: String,
    input_modes: Vec<u32>,
    output_modes: Vec<u32>,
    response_entries: Vec<ResponseEntry>,
    coefficients: Vec<ModeCoefficient>,
    encoded_bytes: Vec<ModeBytes>,
    output_qoi_weights: Vec<f64>,
    response_loss_budget: f64,
}

#[derive(Debug, Deserialize)]
struct ResponseEntry {
    input_mode: u32,
    output_mode: u32,
    real: f64,
    imag: f64,
}

#[derive(Debug, Deserialize)]
struct ModeCoefficient {
    mode: u32,
    real: f64,
    imag: f64,
}

#[derive(Debug, Deserialize)]
struct ModeBytes {
    mode: u32,
    bytes: u64,
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let input_path = args.next().ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            "usage: response_weighted_refinement_audit <record.json>",
        )
    })?;
    if args.next().is_some() {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: response_weighted_refinement_audit <record.json>",
        )
        .into());
    }

    let input_text = fs::read_to_string(&input_path)?;
    let input: RefinementInput = serde_json::from_str(&input_text)?;
    if input.schema_version != "0.1" {
        return Err(IoError::new(ErrorKind::InvalidData, "unsupported schema_version").into());
    }

    let mut response = ModeResponseMatrix::new(input.input_modes, input.output_modes)?;
    for entry in input.response_entries {
        response.set(
            entry.input_mode,
            entry.output_mode,
            ComplexResponse::new(entry.real, entry.imag)?,
        )?;
    }
    let coefficients = input
        .coefficients
        .into_iter()
        .map(|entry| {
            Ok((
                entry.mode,
                ComplexResponse::new(entry.real, entry.imag)?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let encoded_bytes = input
        .encoded_bytes
        .into_iter()
        .map(|entry| (entry.mode, entry.bytes))
        .collect::<Vec<_>>();

    let plan = plan_response_weighted_refinement(
        &response,
        &coefficients,
        &encoded_bytes,
        &input.output_qoi_weights,
        input.response_loss_budget,
    )?;
    let output = serde_json::json!({
        "audit_status": "PASS",
        "case_id": input.case_id,
        "claim_boundary": "scoring primitive only; no physical solver result",
        "plan": plan
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("response_weighted_refinement_audit=FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
