// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metadata-only preflight for a versioned bioacoustic corpus manifest.
//!
//! This command does not read the referenced audio assets and cannot establish
//! that the supplied rights decision is legally correct. It checks the
//! manifest's declared metadata, computes its canonical identity, and states
//! its evidence ceiling explicitly.

use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;
use symthaea_communication::bioacoustic_corpus::{
    BioacousticCorpusManifest, BioacousticSplit, BioacousticSplitPolicy,
    validate_bioacoustic_corpus,
};

#[derive(Debug, Parser)]
#[command(
    name = "bioacoustic-corpus-preflight",
    about = "Validate declared metadata for a bird/dolphin/whale corpus manifest"
)]
struct Args {
    /// JSON BioacousticCorpusManifest file.
    manifest: PathBuf,
    /// Require stable IDs for original continuous/source recordings.
    #[arg(long)]
    require_source_recording_ids: bool,
    /// Require event/segment annotations on every recording.
    #[arg(long)]
    require_annotations: bool,
    /// Require sessions to be disjoint across splits.
    #[arg(long)]
    hold_out_sessions: bool,
    /// Require sites to be disjoint across splits.
    #[arg(long)]
    hold_out_sites: bool,
    /// Require known individual IDs to be disjoint across splits.
    #[arg(long)]
    hold_out_individuals: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("BIOACOUSTIC_METADATA_PREFLIGHT=FAIL");
        eprintln!("REASON={error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.manifest)
        .map_err(|error| format!("cannot read manifest {}: {error}", args.manifest.display()))?;
    let manifest: BioacousticCorpusManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse bioacoustic corpus manifest: {error}"))?;

    let policy = BioacousticSplitPolicy {
        require_source_recording_ids: args.require_source_recording_ids,
        require_annotations: args.require_annotations,
        hold_out_sessions: args.hold_out_sessions,
        hold_out_sites: args.hold_out_sites,
        hold_out_individuals: args.hold_out_individuals,
    };
    validate_bioacoustic_corpus(&manifest, policy)?;
    let manifest_identity = manifest.computed_manifest_blake3()?;

    let mut split_counts = BTreeMap::<BioacousticSplit, usize>::new();
    for recording in &manifest.recordings {
        *split_counts.entry(recording.split).or_default() += 1;
    }

    println!("BIOACOUSTIC_METADATA_PREFLIGHT=PASS");
    println!("CORPUS_SCHEMA_VERSION={}", manifest.schema_version);
    println!("DATASET_ID={}", manifest.dataset_id);
    println!("DATASET_REVISION={}", manifest.dataset_revision);
    println!("MANIFEST_BLAKE3={manifest_identity}");
    println!("RECORDINGS={}", manifest.recordings.len());
    for (split, count) in split_counts {
        println!("SPLIT_{split:?}={count}");
    }
    println!("AUDIO_BYTES_VERIFIED=false");
    println!("RIGHTS_DECISION_AUTHENTICATED=false");
    println!("MODEL_OR_DETECTION_PERFORMANCE_EVALUATED=false");
    Ok(())
}
