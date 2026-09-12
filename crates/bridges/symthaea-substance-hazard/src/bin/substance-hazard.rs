// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, process};
use symthaea_discovery::CandidateId;
use symthaea_substance_hazard::{
    calculate_hazard_evidence_from_bytes, HazardBinding, HazardBindingBasis,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("substance-hazard: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let data_path = args.next().ok_or("missing <hazard-data.json>")?;
    let source_path = args.next().ok_or("missing <source-document>")?;
    let policy_path = args.next().ok_or("missing <policy.json>")?;
    let candidate_id = args.next().ok_or("missing <candidate-id>")?;
    let substance_id = args.next().ok_or("missing <substance-id>")?;
    let mapping_note = args.next();
    if args.next().is_some() {
        return Err("usage: substance-hazard <hazard-data.json> <source-document> <policy.json> <candidate-id> <substance-id> [explicit-mapping-note]".into());
    }

    let binding = HazardBinding {
        candidate_id: CandidateId::new(candidate_id)?,
        substance_id,
        basis: match mapping_note {
            Some(note) => HazardBindingBasis::ExplicitMapping { note },
            None => HazardBindingBasis::ExactSubstanceId,
        },
    };

    let dataset_json = fs::read(data_path)?;
    let source_document = fs::read(source_path)?;
    let policy_json = fs::read(policy_path)?;
    let receipt = calculate_hazard_evidence_from_bytes(
        &dataset_json,
        &source_document,
        &policy_json,
        binding,
    )?;

    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
