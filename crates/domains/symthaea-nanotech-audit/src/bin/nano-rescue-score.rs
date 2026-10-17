// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../rescue_metrics.rs"]
mod rescue_metrics;

use std::path::PathBuf;
use std::process::ExitCode;
use symthaea_nanotech_audit::audit_run;

fn main() -> ExitCode {
    let mut args = std::env::args_os();
    let program = args.next().unwrap_or_default();
    let Some(run_dir) = args.next() else {
        eprintln!(
            "usage: {} <run-directory>",
            PathBuf::from(program).display()
        );
        eprintln!(
            "The scorer requires a passing structural audit; a score report is not a scientific promotion verdict."
        );
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one argument: <run-directory>");
        return ExitCode::from(2);
    }

    let run_dir = PathBuf::from(run_dir);
    let audit = audit_run(&run_dir);
    if audit.status != "artifact_integrity_pass" {
        let details = serde_json::to_string_pretty(&audit)
            .unwrap_or_else(|error| format!("could not serialize audit report: {error}"));
        eprintln!(
            "NANO-013 scoring refused because structural audit status was {}:\n{}",
            audit.status, details
        );
        return ExitCode::from(1);
    }

    match rescue_metrics::score_run(&run_dir) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("NANO-013 scoring failed closed: {error}");
            ExitCode::from(1)
        }
    }
}
