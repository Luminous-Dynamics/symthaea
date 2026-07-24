// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate an independently generated dynamics-oracle dataset.

use std::path::PathBuf;

use symthaea_humanoid::DynamicsOracleDataset;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let Some(path) = args.next().map(PathBuf::from) else {
        eprintln!("usage: symthaea_oracle_dataset <dataset.json> <candidate-build-id>");
        std::process::exit(2);
    };
    let Some(candidate_build_id) = args.next().and_then(|value| value.into_string().ok()) else {
        eprintln!("candidate build identity is required");
        std::process::exit(2);
    };
    if args.next().is_some() || candidate_build_id.trim().is_empty() {
        eprintln!("unexpected arguments or empty candidate build identity");
        std::process::exit(2);
    }
    let dataset = match DynamicsOracleDataset::load_json(&path) {
        Ok(dataset) => dataset,
        Err(error) => {
            eprintln!("failed to load {}: {error}", path.display());
            std::process::exit(1);
        }
    };
    if !dataset.validate_for_candidate(&candidate_build_id) {
        eprintln!("dataset failed independent-oracle admission");
        std::process::exit(1);
    }
    println!(
        "dataset={} generator={} engine={} cases={} morphology={:?}",
        dataset.manifest.dataset_id,
        dataset.manifest.generator_build_id,
        dataset.manifest.engine_id,
        dataset.cases.len(),
        dataset.manifest.morphology,
    );
}
