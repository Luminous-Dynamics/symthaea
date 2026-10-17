// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

mod benchmark;

use serde_json::Value;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() {
    eprintln!(
        "Usage: hifi-benchmark-check <split-manifest.json>\n\
         Checks declared group-level split integrity for open hi-fi benchmarks.\n\
         This does not evaluate model quality or validate the science of a study."
    );
}

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(path_arg) = args.next() else {
        usage();
        return ExitCode::from(2);
    };
    let path = PathBuf::from(path_arg);
    if args.next().is_some() {
        usage();
        return ExitCode::from(2);
    }

    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("ERROR: cannot read {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let manifest: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("ERROR: invalid JSON in {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    match benchmark::validate_split_manifest(&manifest) {
        Ok(rows) => {
            let benchmark_id = manifest
                .get("benchmark_id")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            println!(
                "SPLIT CONTRACT OK: benchmark={benchmark_id} observations={rows}; model quality not evaluated"
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("SPLIT CONTRACT FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
