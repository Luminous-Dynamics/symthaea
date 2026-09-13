// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};
use std::{env, fs, process};
use symthaea_energy_material_campaign::{AcquisitionDeclaration, Tier1CampaignManifest};
use symthaea_energy_native_campaign_admission::{
    admit_native_campaign_result, NativeCampaignAdmissionReceipt,
};
use symthaea_energy_native_dossier::NativeEnvelopeDossier;

#[derive(Debug, Deserialize)]
struct Input {
    manifest: Tier1CampaignManifest,
    native_dossier: NativeEnvelopeDossier,
    #[serde(default)]
    acquisition_declarations: Vec<AcquisitionDeclaration>,
}

#[derive(Debug, Serialize)]
struct Output {
    native_campaign_admission_sha256: String,
    admission: NativeCampaignAdmissionReceipt,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("energy-native-campaign-admission: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let input_path = args
        .next()
        .ok_or("usage: energy-native-campaign-admission <input.json>")?;
    if args.next().is_some() {
        return Err("usage: energy-native-campaign-admission <input.json>".into());
    }

    let input: Input = serde_json::from_slice(&fs::read(input_path)?)?;
    let admission = admit_native_campaign_result(
        &input.manifest,
        &input.native_dossier,
        input.acquisition_declarations,
    )?;
    let output = Output {
        native_campaign_admission_sha256: admission.sha256()?,
        admission,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
