// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run the Rust Bayesian EVSI inspection policy.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use symthaea_semiconductor::benchmark::{
    EvsiReportV1, EvsiSubjectV1, evaluate_evsi_v1,
};

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 1 {
        return Err("usage: semi-inspection-evsi <evsi-subject.json>".into());
    }
    let input_path = PathBuf::from(&arguments[0]);
    let input_text = fs::read_to_string(&input_path)
        .map_err(|error| format!("cannot read {}: {error}", input_path.display()))?;
    let subject: EvsiSubjectV1 = serde_json::from_str(&input_text)
        .map_err(|error| format!("invalid EVSI JSON: {error}"))?;
    let report: EvsiReportV1 = evaluate_evsi_v1(&subject)
        .map_err(|error| format!("EVSI evaluation rejected input: {error}"))?;
    let canonical = serde_json::to_value(&report)
        .map_err(|error| format!("cannot canonicalize EVSI report: {error}"))?;
    let rendered = serde_json::to_string_pretty(&canonical)
        .map_err(|error| format!("cannot render EVSI report: {error}"))?;
    println!("{rendered}");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
