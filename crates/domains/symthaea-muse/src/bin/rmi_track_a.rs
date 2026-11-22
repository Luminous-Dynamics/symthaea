// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run the deterministic offline RMI-Bench Track A synthetic suite.
//!
//! Usage:
//!   cargo run --release -p symthaea-muse --bin rmi_track_a -- [trials_per_target] [seed_root] [report.json]

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use symthaea_muse::rmi_track_a_runner::run_rmi_track_a_suite_v1;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);

    let trials_per_target = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(8);

    let seed_root = args
        .next()
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(42);

    let output_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("rmi-track-a-report.json"));

    if args.next().is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: rmi_track_a [trials_per_target] [seed_root] [report.json]",
        )
        .into());
    }

    let report = run_rmi_track_a_suite_v1(trials_per_target, seed_root)?;
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&output_path, json)?;

    println!("RMI Track A synthetic fixture suite");
    println!("Trials: {}", report.payload.summary.trials);
    println!("Evaluator errors: {}", report.payload.summary.evaluator_errors);
    println!(
        "Fixed-agent false positives: {}",
        report.payload.summary.trials
            - report.payload.summary.fixed_agent_zero_sensitivity
    );
    println!(
        "One-way prompt false positives: {}",
        report.payload.summary.trials
            - report.payload.summary.one_way_prompt_zero_sensitivity
    );
    println!(
        "Yoked replay false positives: {}",
        report.payload.summary.trials
            - report.payload.summary.yoked_replay_zero_sensitivity
    );
    println!(
        "Wrong-feature target-sensitivity false positives: {}",
        report.payload.summary.trials
            - report.payload.summary.unrelated_feature_zero_target_sensitivity
    );
    println!(
        "Imitation sensitivity: {}/{}",
        report.payload.summary.imitation_agent_target_sensitive,
        report.payload.summary.trials
    );
    println!(
        "Synthetic positive control: {}/{}",
        report.payload.summary.positive_agent_target_sensitive,
        report.payload.summary.trials
    );
    println!(
        "Reciprocal positive control: {}/{}",
        report.payload.summary.reciprocal_positive_control_both_directions,
        report.payload.summary.trials
    );
    println!(
        "Synthetic fixture expectations satisfied: {}",
        report
            .payload
            .summary
            .synthetic_fixture_expectations_satisfied
    );
    println!("Report SHA-256: {}", report.payload_sha256);
    println!("Report written to: {}", output_path.display());

    if !report.payload.summary.synthetic_fixture_expectations_satisfied {
        return Err(std::io::Error::other(
            "one or more synthetic fixture expectations failed",
        )
        .into());
    }
    Ok(())
}
