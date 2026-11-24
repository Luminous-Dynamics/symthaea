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

use symthaea_plasma_probes::validate_sensor_actuator_timing_contract_json;

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let contract_path = args.next().ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            "usage: sensor_actuator_contract_audit <contract.json>",
        )
    })?;
    if args.next().is_some() {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "usage: sensor_actuator_contract_audit <contract.json>",
        )
        .into());
    }

    let input = fs::read_to_string(&contract_path)?;
    let report = validate_sensor_actuator_timing_contract_json(&input)?;
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "audit_status": "PASS",
        "timing": report
    }))?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sensor_actuator_contract_audit=FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
