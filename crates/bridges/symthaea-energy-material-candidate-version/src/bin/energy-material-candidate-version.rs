// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};
use std::{env, fs, process};
use symthaea_discovery::Candidate;
use symthaea_energy_material_candidate_version::{
    bind_dossier_to_candidate_version, CandidateVersionBoundDossier, ReceiptVersionAttestation,
};
use symthaea_energy_material_dossier::EnergyMaterialDossier;

#[derive(Debug, Deserialize)]
struct Input {
    candidate: Candidate,
    dossier: EnergyMaterialDossier,
    receipt_version_attestations: Vec<ReceiptVersionAttestation>,
}

#[derive(Debug, Serialize)]
struct Output {
    candidate_bound_dossier_sha256: String,
    candidate_bound_dossier: CandidateVersionBoundDossier,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("energy-material-candidate-version: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let input_path = args
        .next()
        .ok_or("usage: energy-material-candidate-version <binding-input.json>")?;
    if args.next().is_some() {
        return Err("usage: energy-material-candidate-version <binding-input.json>".into());
    }

    let bytes = fs::read(input_path)?;
    let input: Input = serde_json::from_slice(&bytes)?;
    let bound = bind_dossier_to_candidate_version(
        input.candidate,
        input.dossier,
        input.receipt_version_attestations,
    )?;
    let output = Output {
        candidate_bound_dossier_sha256: bound.sha256()?,
        candidate_bound_dossier: bound,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
