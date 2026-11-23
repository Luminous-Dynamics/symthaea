// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Command-line entry point for the SEMI-BENCH-001 Rust binary metric scorer.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use symthaea_semiconductor::benchmark::{BinaryScoreSubjectV1, score_binary_subject_v1};

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 1 {
        return Err("usage: semi-binary-metrics <score-subject.json>".to_string());
    }

    let input_path = PathBuf::from(&arguments[0]);
    let input = fs::read_to_string(&input_path)
        .map_err(|error| format!("cannot read {}: {error}", input_path.display()))?;
    let subject: BinaryScoreSubjectV1 = serde_json::from_str(&input)
        .map_err(|error| format!("invalid score subject JSON: {error}"))?;
    let report = score_binary_subject_v1(&subject)
        .map_err(|error| format!("SEMI-BENCH scorer rejected input: {error}"))?;
    let canonical = serde_json::to_value(report)
        .map_err(|error| format!("cannot construct report JSON: {error}"))?;
    let rendered = serde_json::to_string_pretty(&canonical)
        .map_err(|error| format!("cannot render report JSON: {error}"))?;
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
