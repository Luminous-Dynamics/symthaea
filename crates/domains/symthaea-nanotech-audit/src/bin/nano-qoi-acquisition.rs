// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::process::ExitCode;
use symthaea_nanotech_audit::qoi_acquisition::run_qoi_acquisition_benchmark;

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() > 3 || args.iter().any(|arg| arg == "--help") {
        eprintln!("usage: nano-qoi-acquisition [reference-budget] [random-trials] [seed]");
        eprintln!("Defaults: budget=8, random-trials=256, seed=20903.");
        eprintln!("This runs a synthetic known-truth benchmark, not an atomistic simulation.");
        return if args.iter().any(|arg| arg == "--help") {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(2)
        };
    }

    let parse_usize = |index: usize, default: usize, label: &str| -> Result<usize, String> {
        match args.get(index) {
            Some(value) => value
                .parse::<usize>()
                .map_err(|error| format!("invalid {label}: {error}")),
            None => Ok(default),
        }
    };
    let parse_seed = || -> Result<u64, String> {
        match args.get(2) {
            Some(value) => value.parse::<u64>().map_err(|error| format!("invalid seed: {error}")),
            None => Ok(0x51a7),
        }
    };

    let budget = match parse_usize(0, 8, "reference-budget") {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let trials = match parse_usize(1, 256, "random-trials") {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let seed = match parse_seed() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };

    match run_qoi_acquisition_benchmark(budget, trials, seed) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not serialize acquisition report: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("synthetic QoI acquisition benchmark failed: {error}");
            ExitCode::from(1)
        }
    }
}
