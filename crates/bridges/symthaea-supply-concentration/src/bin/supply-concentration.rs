// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, process};
use symthaea_discovery::CandidateId;
use symthaea_supply_concentration::{
    calculate_supply_concentration, CandidateAggregationPolicy, SupplyDataset,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("supply-concentration: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let data_path = args.next().ok_or("missing <supply-data.json>")?;
    let source_path = args.next().ok_or("missing <source-document>")?;
    let candidate_id = args.next().ok_or("missing <candidate-id>")?;
    let formula = args.next().ok_or("missing <formula>")?;
    let aggregation = args.next().ok_or("missing <max|mass-weighted>")?;
    if args.next().is_some() {
        return Err("usage: supply-concentration <supply-data.json> <source-document> <candidate-id> <formula> <max|mass-weighted>".into());
    }

    let aggregation_policy = match aggregation.as_str() {
        "max" => CandidateAggregationPolicy::MaximumElementHhi,
        "mass-weighted" => CandidateAggregationPolicy::MassWeightedMeanHhi,
        _ => return Err("aggregation must be exactly 'max' or 'mass-weighted'".into()),
    };

    let data_bytes = fs::read(data_path)?;
    let source_document = fs::read(source_path)?;
    let dataset = SupplyDataset::from_json_bytes(&data_bytes)?;
    let receipt = calculate_supply_concentration(
        CandidateId::new(candidate_id)?,
        &formula,
        &dataset,
        &source_document,
        aggregation_policy,
    )?;

    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
