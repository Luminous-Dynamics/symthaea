// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::Serialize;
use std::{env, fs, process};
use symthaea_discovery::Prediction;
use symthaea_energy_evidence_envelope::{
    wrap_evidence_payload_json, EnergyEvidenceEnvelope,
};
use symthaea_energy_material_campaign::Tier1CampaignManifest;
use symthaea_energy_material_screening::EvidenceDimension;

#[derive(Debug, Serialize)]
struct Output {
    evidence_envelope_sha256: String,
    envelope: EnergyEvidenceEnvelope,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("energy-evidence-envelope: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let usage = "usage: energy-evidence-envelope <manifest.json> <dimension> <prediction.json> <payload-type> <receipt.json>";
    let manifest_path = args.next().ok_or(usage)?;
    let dimension = args.next().ok_or(usage)?;
    let prediction_path = args.next().ok_or(usage)?;
    let payload_type = args.next().ok_or(usage)?;
    let receipt_path = args.next().ok_or(usage)?;
    if args.next().is_some() {
        return Err(usage.into());
    }

    let manifest: Tier1CampaignManifest =
        serde_json::from_slice(&fs::read(manifest_path)?)?;
    let prediction: Prediction = serde_json::from_slice(&fs::read(prediction_path)?)?;
    let payload_json = fs::read_to_string(receipt_path)?;
    let dimension = parse_dimension(&dimension.to_string_lossy())?;
    let envelope = wrap_evidence_payload_json(
        &manifest,
        dimension,
        prediction,
        payload_type.to_string_lossy().into_owned(),
        payload_json,
    )?;
    let output = Output {
        evidence_envelope_sha256: envelope.sha256()?,
        envelope,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn parse_dimension(value: &str) -> Result<EvidenceDimension, Box<dyn std::error::Error>> {
    Ok(match value {
        "functional_performance" => EvidenceDimension::FunctionalPerformance,
        "thermodynamic_stability" => EvidenceDimension::ThermodynamicStability,
        "critical_material_burden" => EvidenceDimension::CriticalMaterialBurden,
        "supply_resilience" => EvidenceDimension::SupplyResilience,
        "human_environmental_hazard" => EvidenceDimension::HumanEnvironmentalHazard,
        "circularity" => EvidenceDimension::Circularity,
        "manufacturability" => EvidenceDimension::Manufacturability,
        _ => {
            return Err(format!(
                "unknown dimension {value:?}; expected one of functional_performance, thermodynamic_stability, critical_material_burden, supply_resilience, human_environmental_hazard, circularity, manufacturability"
            )
            .into())
        }
    })
}
