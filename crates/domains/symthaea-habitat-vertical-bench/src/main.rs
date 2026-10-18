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
        let value: serde_json::Value = match serde_json::from_str(input) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("fixture changed between validation and summary parsing: {error}");
                process::exit(2);
            }
        };
        let cost = match symthaea_habitat_vertical_bench::accounting::calculate_cost_comparison(&value) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("cost known-answer calculation failed after structural pass: {error}");
                process::exit(1);
            }
        };
        let mass_case_id = value
            .get("synthetic_material_balance_cases")
            .and_then(serde_json::Value::as_array)
            .and_then(|cases| cases.first())
            .and_then(|case| case.get("case_id"))
            .and_then(serde_json::Value::as_str);
        let Some(mass_case_id) = mass_case_id else {
            eprintln!("no valid material-balance case was found after structural pass");
            process::exit(1);
        };
        let mass = match symthaea_habitat_vertical_bench::accounting::calculate_material_balance(
            &value,
            mass_case_id,
        ) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("material-balance known-answer calculation failed after structural pass: {error}");
                process::exit(1);
            }
        };

        println!("STRUCTURAL FIXTURE VALIDATION: PASS");
        println!(
            "synthetic known-answer candidate/reference cost ratio: {} basis points; target_met={}",
            cost.ratio_basis_points_floor, cost.target_met
        );
        println!(
            "synthetic material-balance closure: {} basis points; accepted mass yield: {} basis points; unaccounted={} {}",
            mass.closure_basis_points_floor,
            mass.accepted_yield_basis_points_floor,
            mass.unaccounted_mass_mg,
            mass.mass_unit
        );
        println!("scope: synthetic arithmetic and structure only; no physical or product qualification");
        println!("negative controls are declared, not all executed by this binary");
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
