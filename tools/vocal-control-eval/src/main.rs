// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evaluate one provenance-bound vocal control intervention case or multi-seed suite JSON file.

use std::process::ExitCode;

use symthaea_vocal_control_eval::{
    ControlCase, ControlSuite, Qualification, evaluate, evaluate_suite,
};

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(error) => {
            eprintln!("vocal-control-eval: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let path = std::env::args_os().nth(1).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: vocal-control-eval <control-case.json>",
        )
    })?;
    let bytes = std::fs::read(path)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    if value.get("replicates").is_some() {
        let suite: ControlSuite = serde_json::from_value(value)?;
        let report = evaluate_suite(&suite).map_err(std::io::Error::other)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(report.qualification == Qualification::Qualified)
    } else {
        let case: ControlCase = serde_json::from_value(value)?;
        let report = evaluate(&case).map_err(std::io::Error::other)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(report.qualification == Qualification::Qualified)
    }
}
