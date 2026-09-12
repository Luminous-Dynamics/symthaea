// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, process};
use symthaea_bandgap_discovery_evidence::predict_bandgap_evidence;
use symthaea_discovery::CandidateId;

fn main() {
    if let Err(error) = run() {
        eprintln!("bandgap-discovery-evidence: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let candidate_id = args.next().ok_or("missing <candidate-id>")?;
    let composition_path = args.next().ok_or("missing <composition.json>")?;
    if args.next().is_some() {
        return Err("usage: bandgap-discovery-evidence <candidate-id> <composition.json>".into());
    }

    let composition: Vec<(u8, f64)> = serde_json::from_slice(&fs::read(composition_path)?)?;
    let receipt = predict_bandgap_evidence(CandidateId::new(candidate_id)?, composition)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
