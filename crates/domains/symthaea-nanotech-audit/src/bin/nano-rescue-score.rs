// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../rescue_metrics.rs"]
mod rescue_metrics;

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args_os();
    let program = args.next().unwrap_or_default();
    let Some(run_dir) = args.next() else {
        eprintln!(
            "usage: {} <run-directory>",
            PathBuf::from(program).display()
        );
        eprintln!(
            "Run the symthaea-nanotech-audit structural verifier first; scoring is not a scientific promotion verdict."
        );
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one argument: <run-directory>");
        return ExitCode::from(2);
    }

    match rescue_metrics::score_run(&PathBuf::from(run_dir)) {
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
