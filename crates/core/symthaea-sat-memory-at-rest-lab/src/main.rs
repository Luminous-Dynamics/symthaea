// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::env;
use std::fs;
use std::process;

use symthaea_sat_memory_at_rest_lab::{run_campaign, CampaignStatus, ProfileMode};

fn usage(program: &str) -> ! {
    eprintln!("Usage: {program} <output-directory> <vulnerable|protected-read|protected-audit>");
    process::exit(64);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args.first().map(String::as_str).unwrap_or("sat-memory-at-rest-lab");
    let (output_dir, mode) = match args.as_slice() {
        [_, output, mode] => {
            let mode = ProfileMode::parse(mode).unwrap_or_else(|error| {
                eprintln!("{error}");
                usage(program)
            });
            (output.as_str(), mode)
        }
        _ => usage(program),
    };

    if let Err(error) = fs::create_dir_all(output_dir) {
        eprintln!("cannot create output directory {output_dir}: {error}");
        process::exit(73);
    }

    match run_campaign(output_dir, mode) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("campaign report serialization cannot fail")
            );
            if report.campaign_status == CampaignStatus::NotEvaluable {
                process::exit(3);
            }
        }
        Err(error) => {
            eprintln!("memory lab failed: {error}");
            process::exit(2);
        }
    }
}
