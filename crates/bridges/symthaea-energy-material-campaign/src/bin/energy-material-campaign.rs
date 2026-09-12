// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};
use std::{env, fs, process};
use symthaea_energy_material_campaign::{
    admit_campaign_result, freeze_campaign_manifest, AcquisitionDeclaration, EvidenceLanePlan,
    Tier1CampaignManifest,
};
use symthaea_energy_material_candidate_version::{
    CandidateVersionAnchor, CandidateVersionBoundDossier,
};
use symthaea_energy_material_screening::EnergyMaterialScreeningPolicy;

#[derive(Debug, Deserialize)]
struct FreezeInput {
    campaign_id: String,
    candidate_anchor: CandidateVersionAnchor,
    screening_policy: EnergyMaterialScreeningPolicy,
    evidence_lanes: Vec<EvidenceLanePlan>,
    external_registration_reference: Option<String>,
    #[serde(default)]
    notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AdmitInput {
    manifest: Tier1CampaignManifest,
    candidate_bound_dossier: CandidateVersionBoundDossier,
    #[serde(default)]
    acquisition_declarations: Vec<AcquisitionDeclaration>,
}

#[derive(Debug, Serialize)]
struct FreezeOutput {
    campaign_manifest_sha256: String,
    manifest: Tier1CampaignManifest,
}

#[derive(Debug, Serialize)]
struct AdmitOutput {
    campaign_admission_sha256: String,
    admission: symthaea_energy_material_campaign::CampaignAdmissionReceipt,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("energy-material-campaign: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let command = args.next().ok_or(
        "usage: energy-material-campaign <freeze|admit> <input.json>",
    )?;
    let input_path = args.next().ok_or(
        "usage: energy-material-campaign <freeze|admit> <input.json>",
    )?;
    if args.next().is_some() {
        return Err("usage: energy-material-campaign <freeze|admit> <input.json>".into());
    }

    let bytes = fs::read(input_path)?;
    match command.to_string_lossy().as_ref() {
        "freeze" => {
            let input: FreezeInput = serde_json::from_slice(&bytes)?;
            let manifest = freeze_campaign_manifest(
                input.campaign_id,
                input.candidate_anchor,
                input.screening_policy,
                input.evidence_lanes,
                input.external_registration_reference,
                input.notes,
            )?;
            let output = FreezeOutput {
                campaign_manifest_sha256: manifest.sha256()?,
                manifest,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "admit" => {
            let input: AdmitInput = serde_json::from_slice(&bytes)?;
            let admission = admit_campaign_result(
                &input.manifest,
                &input.candidate_bound_dossier,
                input.acquisition_declarations,
            )?;
            let output = AdmitOutput {
                campaign_admission_sha256: admission.sha256()?,
                admission,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        _ => {
            return Err("usage: energy-material-campaign <freeze|admit> <input.json>".into());
        }
    }
    Ok(())
}
