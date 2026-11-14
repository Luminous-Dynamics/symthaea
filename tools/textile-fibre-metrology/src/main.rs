// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;
use textile_fibre_metrology::{
    assess_distribution, DistributionAssessment, FibreLengthDistribution, MethodDefinition,
    ReceiverCriterion,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputBundle {
    method: MethodDefinition,
    distribution: FibreLengthDistribution,
    receiver_criterion: ReceiverCriterion,
}

#[derive(Debug, Serialize)]
struct OutputBundle {
    claim_ceiling: &'static str,
    assessment: DistributionAssessment,
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 || args[1] != "verify" {
        return Err(
            "usage: textile-fibre-metrology verify --input BUNDLE.json".to_owned()
        );
    }
    let input_index = args
        .iter()
        .position(|arg| arg == "--input")
        .ok_or_else(|| "missing --input BUNDLE.json".to_owned())?;
    let input_path = args
        .get(input_index + 1)
        .ok_or_else(|| "missing value after --input".to_owned())?;
    let _ = PathBuf::from(input_path);
    let bytes = fs::read(input_path)
        .map_err(|error| format!("cannot read input bundle: {error}"))?;
    let bundle: InputBundle = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid input bundle: {error}"))?;
    let assessment = assess_distribution(
        &bundle.distribution,
        &bundle.method,
        &bundle.receiver_criterion,
    );
    let output = OutputBundle {
        claim_ceiling: "method-scoped fibre-length metrology only; not yarn or receiver acceptance",
        assessment,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("cannot serialize assessment: {error}"))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAIL: {error}");
        std::process::exit(2);
    }
}
