// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validate a frozen speech cohort manifest without downloading or reading audio.

use clap::Parser;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use symthaea_stt::cohort_manifest::{
    CohortManifest, canonical_cohort_manifest_digest,
    require_external_processing_authorization, validate_cohort_manifest,
    verify_counterfactual_pair_manifest_artifacts,
};

#[derive(Debug, Parser)]
#[command(
    name = "stt-cohort-validate",
    about = "Fail-closed structural and provenance checks for a speech cohort manifest"
)]
struct Args {
    /// Path to the cohort_manifest_v0.1.json file.
    manifest: PathBuf,

    /// Also require explicit authorization to send every record to an external model/API.
    #[arg(long)]
    require_external_processing: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("STRUCTURAL_VALIDATION=FAIL");
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let bytes = fs::read(&args.manifest)?;
    let manifest: CohortManifest = serde_json::from_slice(&bytes)?;
    let summary = validate_cohort_manifest(&manifest)?;
    let manifest_directory = args.manifest.parent().unwrap_or_else(|| Path::new("."));
    let verified_pair_artifacts =
        verify_counterfactual_pair_manifest_artifacts(&manifest, manifest_directory)?;

    if args.require_external_processing {
        for record in &manifest.records {
            require_external_processing_authorization(record)?;
        }
    }

    let digest = canonical_cohort_manifest_digest(&manifest)?;
    println!("STRUCTURAL_VALIDATION=PASS");
    println!("cohort_id={}", manifest.cohort_id);
    println!("evidence_state={:?}", manifest.evidence_state);
    println!("manifest_blake3={digest}");
    println!("records={}", summary.record_count);
    println!("pairs={}", summary.pair_count);
    println!("pair_manifest_artifacts_verified={verified_pair_artifacts}");
    println!("preference_pairs={}", summary.preference_pair_count);
    println!("development_pairs={}", summary.development_pair_count);
    println!("pilot_pairs={}", summary.pilot_pair_count);
    println!("confirmatory_pairs={}", summary.confirmatory_pair_count);
    println!("unique_source_families={}", summary.unique_source_family_count);
    println!(
        "preference_evaluation_pair_overlap={}",
        summary.preference_evaluation_pair_overlap
    );
    println!("rights=DECLARED_BY_DATA_STEWARD_NOT_LEGALLY_VERIFIED");
    println!("audio=NOT_OPENED_OR_DECODED");
    println!("model_inference=NOT_RUN");
    Ok(())
}
