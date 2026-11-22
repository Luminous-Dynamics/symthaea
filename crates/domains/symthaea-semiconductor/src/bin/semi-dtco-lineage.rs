// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Command-line DTCO lineage validator for SEMI-BENCH-001.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use symthaea_semiconductor::benchmark::{DtcoLineageManifestV1, validate_dtco_lineage_v1};

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().skip(1).collect();
    if arguments.len() != 1 {
        return Err("usage: semi-dtco-lineage <lineage-manifest.json>".to_string());
    }

    let input_path = PathBuf::from(&arguments[0]);
    let input = fs::read_to_string(&input_path)
        .map_err(|error| format!("cannot read {}: {error}", input_path.display()))?;
    let manifest: DtcoLineageManifestV1 = serde_json::from_str(&input)
        .map_err(|error| format!("invalid DTCO lineage JSON: {error}"))?;
    let report = validate_dtco_lineage_v1(&manifest)
        .map_err(|error| format!("DTCO lineage rejected input: {error}"))?;
    let rendered = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("cannot render DTCO lineage report: {error}"))?;
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
