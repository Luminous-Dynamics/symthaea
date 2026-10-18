// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, process};

use symthaea_habitat_vertical_bench::{validate_fixture_str, EMBEDDED_FIXTURE};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() > 1 {
        eprintln!("usage: validate-habitat-vertical [fixture.json]");
        process::exit(2);
    }

    let input_storage = if let Some(path) = args.first() {
        match fs::read_to_string(path) {
            Ok(contents) => Some(contents),
            Err(error) => {
                eprintln!("could not read fixture {}: {error}", path);
                process::exit(2);
            }
        }
    } else {
        None
    };
    let input = input_storage.as_deref().unwrap_or(EMBEDDED_FIXTURE);

    let report = match validate_fixture_str(input) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("invalid JSON fixture: {error}");
            process::exit(2);
        }
    };

    if report.is_valid() {
        println!("STRUCTURAL FIXTURE VALIDATION: PASS");
        println!("scope: synthetic structure only; no physical or product qualification");
        println!("negative controls are declared, not executed by this binary");
        return;
    }

    eprintln!(
        "STRUCTURAL FIXTURE VALIDATION: FAIL ({} findings)",
        report.issues.len()
    );
    for issue in report.issues {
        eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
    }
    process::exit(1);
}
