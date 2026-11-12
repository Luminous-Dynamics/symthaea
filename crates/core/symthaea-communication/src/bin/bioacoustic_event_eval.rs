// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Score a frozen set of bioacoustic event predictions against Test annotations.
//!
//! This does not run a model or verify audio bytes. Before using the report as
//! evidence, run bioacoustic-asset-audit and preserve the model-run provenance.

use clap::Parser;
use std::path::PathBuf;
use symthaea_communication::bioacoustic_corpus::BioacousticCorpusManifest;
use symthaea_communication::bioacoustic_eval::{
    BioacousticEventPredictionBundle, evaluate_bioacoustic_events,
};

#[derive(Debug, Parser)]
#[command(
    name = "bioacoustic-event-eval",
    about = "Score precomputed predictions against exhaustive Test-split event annotations"
)]
struct Args {
    /// Bioacoustic corpus manifest JSON (annotations and split lineage).
    corpus_manifest: PathBuf,
    /// JSON prediction bundle containing frozen model/threshold/matching metadata.
    predictions: PathBuf,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("BIOACOUSTIC_EVENT_EVALUATION=FAIL");
        eprintln!("REASON={error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let manifest_bytes = std::fs::read(&args.corpus_manifest).map_err(|error| {
        format!(
            "cannot read corpus manifest {}: {error}",
            args.corpus_manifest.display()
        )
    })?;
    let manifest: BioacousticCorpusManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("cannot parse corpus manifest: {error}"))?;

    let prediction_bytes = std::fs::read(&args.predictions).map_err(|error| {
        format!(
            "cannot read prediction bundle {}: {error}",
            args.predictions.display()
        )
    })?;
    let bundle: BioacousticEventPredictionBundle = serde_json::from_slice(&prediction_bytes)
        .map_err(|error| format!("cannot parse event prediction bundle: {error}"))?;

    let report = evaluate_bioacoustic_events(&manifest, &bundle)?;
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("cannot serialize event evaluation report: {error}"))?;
    println!("{json}");
    Ok(())
}
