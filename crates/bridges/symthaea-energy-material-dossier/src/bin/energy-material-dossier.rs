// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::Deserialize;
use std::{env, fs, process};
use symthaea_discovery::CandidateId;
use symthaea_energy_material_dossier::{
    assemble_dossier, EvidenceContribution, IdentityAssertion,
};
use symthaea_energy_material_screening::EnergyMaterialScreeningPolicy;

#[derive(Debug, Deserialize)]
struct DossierInput {
    candidate_id: CandidateId,
    policy: EnergyMaterialScreeningPolicy,
    identity_assertions: Vec<IdentityAssertion>,
    contributions: Vec<EvidenceContribution>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("energy-material-dossier: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input_path = args.next().ok_or("missing <dossier-input.json>")?;
    if args.next().is_some() {
        return Err("usage: energy-material-dossier <dossier-input.json>".into());
    }

    let input: DossierInput = serde_json::from_slice(&fs::read(input_path)?)?;
    let dossier = assemble_dossier(
        input.candidate_id,
        &input.policy,
        input.identity_assertions,
        input.contributions,
    )?;
    let digest = dossier.sha256()?;
    let output = serde_json::json!({
        "dossier_sha256": digest,
        "dossier": dossier,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
