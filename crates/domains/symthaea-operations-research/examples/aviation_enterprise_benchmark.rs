// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Run the bounded synthetic AECB-001 fixture and print one JSON run manifest.
//!
//! Example:
//! cargo run -p symthaea-operations-research --example aviation_enterprise_benchmark
//!
//! This is a research fixture, not airline dispatch or maintenance authority.

use std::process::ExitCode;

use symthaea_operations_research::aviation_enterprise::{
    reference_scenario, solve_exact, verify_plan, RunManifest, VerificationStatus,
};

fn main() -> ExitCode {
    let scenario = reference_scenario();
    let plan = match solve_exact(&scenario) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("AECB-001 scenario rejected: {error}");
            return ExitCode::FAILURE;
        }
    };

    let verification = verify_plan(&scenario, &plan);
    let manifest = RunManifest::from_run(&scenario, 0, &plan, &verification);
    println!("{}", manifest.to_json());

    match verification.status {
        VerificationStatus::Verified => ExitCode::SUCCESS,
        VerificationStatus::Rejected | VerificationStatus::Indeterminate => ExitCode::FAILURE,
    }
}
