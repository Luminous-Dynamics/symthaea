// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::process::ExitCode;

use symthaea_nanotech_audit::rare_event_bounds::{
    assess_rare_event_risk, RareEventEvidence,
};

fn parse_bool(value: &str, name: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("{name} must be exactly true or false")),
    }
}

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 9 || args.first().is_some_and(|arg| arg == "--help") {
        eprintln!(
            "usage: nano-risk-bound <events> <units> <confidence> <unit-label> <independent> <common-probability> <common-horizon> <complete-outcomes> <equal-probability>"
        );
        eprintln!(
            "Boolean flags are declarations that require independent justification; this utility cannot validate the science or sampling design."
        );
        return if args.len() == 9 { ExitCode::SUCCESS } else { ExitCode::from(2) };
    }

    let parse = || -> Result<RareEventEvidence, String> {
        Ok(RareEventEvidence {
            observed_events: args[0]
                .parse::<u64>()
                .map_err(|error| format!("invalid events count: {error}"))?,
            observed_units: args[1]
                .parse::<u64>()
                .map_err(|error| format!("invalid units count: {error}"))?,
            confidence_level: args[2]
                .parse::<f64>()
                .map_err(|error| format!("invalid confidence level: {error}"))?,
            unit_label: args[3].clone(),
            independent_units_justified: parse_bool(&args[4], "independent")?,
            common_event_probability_justified: parse_bool(&args[5], "common-probability")?,
            common_exposure_horizon: parse_bool(&args[6], "common-horizon")?,
            all_outcomes_observed: parse_bool(&args[7], "complete-outcomes")?,
            equal_probability_sampling: parse_bool(&args[8], "equal-probability")?,
        })
    };

    let evidence = match parse() {
        Ok(evidence) => evidence,
        Err(error) => {
            eprintln!("invalid rare-event input: {error}");
            return ExitCode::from(2);
        }
    };

    match assess_rare_event_risk(&evidence) {
        Ok(assessment) => match serde_json::to_string_pretty(&assessment) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not serialize assessment: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("rare-event bound calculation failed: {error}");
            ExitCode::from(1)
        }
    }
}
