// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run the Rust SEMI-BENCH-001 inspection-decision oracle.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use symthaea_semiconductor::benchmark::{
    InspectionDecisionReportV1, InspectionDecisionSubjectV1, evaluate_inspection_decision_v1,
};

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 1 {
        return Err("usage: semi-inspection-decision <inspection-subject.json>".into());
    }
    let input_path = PathBuf::from(&arguments[0]);
    let input_text = fs::read_to_string(&input_path)
        .map_err(|error| format!("cannot read {}: {error}", input_path.display()))?;
    let subject: InspectionDecisionSubjectV1 = serde_json::from_str(&input_text)
        .map_err(|error| format!("invalid inspection-decision JSON: {error}"))?;
    let report: InspectionDecisionReportV1 = evaluate_inspection_decision_v1(&subject)
        .map_err(|error| format!("inspection-decision evaluation rejected input: {error}"))?;

    // serde_json::Value canonicalizes object keys, matching the Python
    // reference's sort_keys=True output contract.
    let canonical = serde_json::to_value(&report)
        .map_err(|error| format!("cannot canonicalize report: {error}"))?;
    let rendered = serde_json::to_string_pretty(&canonical)
        .map_err(|error| format!("cannot render report: {error}"))?;
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
