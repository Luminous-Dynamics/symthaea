// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Digest-bound audit of a SECOM feature/label file pair.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use symthaea_semiconductor::benchmark::{SecomRawExpectationsV1, audit_secom_raw_v1};

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 3 {
        return Err(
            "usage: semi-secom-audit <secom.data> <secom_labels.data> <expectations.json>"
                .to_string(),
        );
    }
    let feature_path = PathBuf::from(&arguments[0]);
    let label_path = PathBuf::from(&arguments[1]);
    let expectation_path = PathBuf::from(&arguments[2]);
    let feature_bytes = fs::read(&feature_path)
        .map_err(|error| format!("cannot read {}: {error}", feature_path.display()))?;
    let label_bytes = fs::read(&label_path)
        .map_err(|error| format!("cannot read {}: {error}", label_path.display()))?;
    let expected_text = fs::read_to_string(&expectation_path)
        .map_err(|error| format!("cannot read {}: {error}", expectation_path.display()))?;
    let expectations: SecomRawExpectationsV1 = serde_json::from_str(&expected_text)
        .map_err(|error| format!("invalid expectations JSON: {error}"))?;
    let report = audit_secom_raw_v1(&feature_bytes, &label_bytes, &expectations)
        .map_err(|error| format!("SECOM raw audit rejected input: {error}"))?;
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("cannot render audit report: {error}"))?;
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
