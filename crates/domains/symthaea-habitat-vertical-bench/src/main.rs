// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, process};

use symthaea_habitat_vertical_bench::scenario_campaign::{
    validate_scenario_campaign_str, EMBEDDED_SCENARIO_CAMPAIGN,
};
use symthaea_habitat_vertical_bench::resilience_options::{
    validate_resilience_options_str, EMBEDDED_RESILIENCE_OPTIONS,
};
use symthaea_habitat_vertical_bench::assumption_falsification::{
    validate_assumption_falsification_str, EMBEDDED_HABITAT_ASSUMPTIONS,
};
use symthaea_habitat_vertical_bench::event_replay::{validate_event_log_str, EMBEDDED_EVENT_LOG};
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

        if input_storage.is_none() {
            let campaign = match validate_scenario_campaign_str(EMBEDDED_SCENARIO_CAMPAIGN) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded scenario campaign JSON: {error}");
                    process::exit(2);
                }
            };
            if !campaign.is_valid() {
                eprintln!(
                    "SCENARIO CAMPAIGN DEFINITION: FAIL ({} findings)",
                    campaign.issues.len()
                );
                for issue in campaign.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic disruption campaign definition: PASS ({} scenarios; numeric results={}; simulation=NOT RUN)",
                campaign.scenario_count, campaign.numeric_result_count
            );
            let options = match validate_resilience_options_str(EMBEDDED_RESILIENCE_OPTIONS) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded resilience-options JSON: {error}");
                    process::exit(2);
                }
            };
            if !options.is_valid() {
                eprintln!(
                    "RESILIENCE OPTIONS DEFINITION: FAIL ({} findings)",
                    options.issues.len()
                );
                for issue in options.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic resilience option matrix: PASS ({} options x {} scenarios = {} cells; robust frontier=NOT RUN; probabilities=NOT ASSUMED)",
                options.option_count, options.scenario_count, options.outcome_count
            );
            let assumptions = match validate_assumption_falsification_str(EMBEDDED_HABITAT_ASSUMPTIONS) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded assumption-falsification JSON: {error}");
                    process::exit(2);
                }
            };
            if !assumptions.is_valid() {
                eprintln!(
                    "ASSUMPTION FALSIFICATION DEFINITIONS: FAIL ({} findings)",
                    assumptions.issues.len()
                );
                for issue in assumptions.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "factory assumption/test definitions: PASS ({} assumptions; {} candidate tests; {} critical assumptions; selection=NOT RUN; observations=0)",
                assumptions.assumption_count,
                assumptions.experiment_count,
                assumptions.critical_assumption_count
            );
            let event_replay = match validate_event_log_str(EMBEDDED_EVENT_LOG, EMBEDDED_FIXTURE) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded event-replay JSON: {error}");
                    process::exit(2);
                }
            };
            if !event_replay.is_valid() {
                eprintln!("EVENT REPLAY DEFINITION: FAIL ({} findings)", event_replay.issues.len());
                for issue in event_replay.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic event replay contract: PASS ({} unique events; {} exact redeliveries; digest=NOT COMPUTED; physical evidence=NOT ESTABLISHED)",
                event_replay.unique_event_count,
                event_replay.duplicate_delivery_count
            );
        } else {
            println!("embedded disruption campaign: not checked against the alternate fixture");
            println!("embedded resilience option matrix: not checked against the alternate fixture");
            println!("embedded assumption/test definitions: not checked against the alternate fixture");
        }

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
