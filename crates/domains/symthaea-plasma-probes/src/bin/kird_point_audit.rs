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

use symthaea_plasma_probes::validate_kird_record_json;

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let record_path = args.next().ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            "usage: kird_point_audit <record.json>",
        )
    })?;
    if args.next().is_some() {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: kird_point_audit <record.json>",
        )
        .into());
    }

    let input = fs::read_to_string(&record_path)?;
    let report = validate_kird_record_json(&input)?;
    let output = serde_json::json!({
        "audit_status": "PASS",
        "record": report
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("kird_point_audit=FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
