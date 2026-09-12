// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::env;
use std::fs;
use symthaea_discovery::CandidateId;
use symthaea_materials_project_stability::{
    bind_stability_evidence, parse_summary_docs_json, BindingBasis, CaptureMetadata,
    StabilityBinding,
};

fn main() {
    if let Err(error) = run_cli() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run_cli() -> Result<(), String> {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "materials-project-stability-evidence".into());
    let summary_path = args.next().ok_or_else(|| usage(&program))?;
    let metadata_path = args.next().ok_or_else(|| usage(&program))?;
    let candidate_id = args.next().ok_or_else(|| usage(&program))?;
    let material_id = args.next().ok_or_else(|| usage(&program))?;
    let mapping_note = args.next();
    if args.next().is_some() {
        return Err(usage(&program));
    }

    let summary_bytes = fs::read(&summary_path)
        .map_err(|error| format!("failed to read SummaryDoc JSON capture: {error}"))?;
    let metadata_bytes = fs::read(&metadata_path)
        .map_err(|error| format!("failed to read capture metadata JSON: {error}"))?;
    let metadata: CaptureMetadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|error| format!("failed to parse capture metadata JSON: {error}"))?;
    let capture = parse_summary_docs_json(&summary_bytes, metadata)
        .map_err(|error| error.to_string())?;

    let basis = match mapping_note {
        Some(note) => BindingBasis::ExplicitMapping { note },
        None => BindingBasis::ExactMaterialId,
    };
    let receipt = bind_stability_evidence(
        &capture,
        StabilityBinding {
            candidate_id: CandidateId::new(candidate_id).map_err(|error| error.to_string())?,
            material_id,
            basis,
        },
    )
    .map_err(|error| error.to_string())?;

    // Host-local capture/metadata paths are deliberately absent from evidence.
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} <summary-docs.json> <capture-metadata.json> <candidate-id> <material-id> [explicit-mapping-note]"
    )
}
