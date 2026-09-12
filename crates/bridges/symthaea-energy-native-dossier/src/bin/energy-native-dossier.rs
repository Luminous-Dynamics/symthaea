// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};
use std::{env, fs, process};
use symthaea_energy_evidence_envelope::EnergyEvidenceEnvelope;
use symthaea_energy_material_campaign::Tier1CampaignManifest;
use symthaea_energy_material_dossier::IdentityAssertion;
use symthaea_energy_native_dossier::{
    assemble_native_envelope_dossier, NativeEnvelopeDossier,
};

#[derive(Debug, Deserialize)]
struct Input {
    manifest: Tier1CampaignManifest,
    identity_assertions: Vec<IdentityAssertion>,
    envelopes: Vec<EnergyEvidenceEnvelope>,
}

#[derive(Debug, Serialize)]
struct Output {
    native_dossier_sha256: String,
    native_dossier: NativeEnvelopeDossier,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("energy-native-dossier: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let input_path = args
        .next()
        .ok_or("usage: energy-native-dossier <input.json>")?;
    if args.next().is_some() {
        return Err("usage: energy-native-dossier <input.json>".into());
    }

    let input: Input = serde_json::from_slice(&fs::read(input_path)?)?;
    let native_dossier = assemble_native_envelope_dossier(
        &input.manifest,
        input.identity_assertions,
        input.envelopes,
    )?;
    let output = Output {
        native_dossier_sha256: native_dossier.sha256(&input.manifest)?,
        native_dossier,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
