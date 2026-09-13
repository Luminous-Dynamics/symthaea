// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Build one content-bound reasoning capability matrix from portable lane bundles.
//!
//! Required environment:
//! - `SYMTHAEA_REASONING_LANE_BUNDLES`: comma-separated JSON bundle paths
//!
//! Optional environment:
//! - `SYMTHAEA_REASONING_MATRIX_PATH`: output path

use std::env;
use std::fs;
use std::path::PathBuf;
use symthaea::intelligence::{
    build_capability_matrix, CapabilityLaneBundle, ReasoningCapabilityArtifact,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("reasoning capability matrix build failed: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let raw_paths = env::var("SYMTHAEA_REASONING_LANE_BUNDLES")
        .map_err(|_| "SYMTHAEA_REASONING_LANE_BUNDLES is required".to_string())?;
    let paths: Vec<PathBuf> = raw_paths
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .collect();
    if paths.is_empty() {
        return Err("no reasoning lane bundle paths were supplied".into());
    }

    let mut lanes = Vec::with_capacity(paths.len());
    for path in &paths {
        let raw = fs::read(path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let bundle: CapabilityLaneBundle = serde_json::from_slice(&raw)
            .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
        let lane = bundle
            .qualify()
            .map_err(|err| format!("lane {} failed qualification: {err}", path.display()))?;
        println!(
            "lane={} domain={:?} episodes={} coverage={:.3} exact_accuracy={}",
            lane.descriptor.lane_id,
            lane.descriptor.domain,
            lane.capability.episodes,
            lane.capability.coverage,
            lane.capability
                .exact_accuracy
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".into())
        );
        lanes.push(lane);
    }

    let subject_revision = lanes[0].subject_revision.clone();
    let matrix = build_capability_matrix(subject_revision, lanes)
        .map_err(|err| format!("matrix qualification failed: {err}"))?;
    let artifact = ReasoningCapabilityArtifact::new(matrix)
        .map_err(|err| format!("artifact construction failed: {err}"))?;
    artifact
        .validate()
        .map_err(|err| format!("artifact self-validation failed: {err}"))?;

    let output = env::var("SYMTHAEA_REASONING_MATRIX_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/reasoning/capability-matrix.json"));
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let encoded = serde_json::to_vec_pretty(&artifact)
        .map_err(|err| format!("failed to encode matrix artifact: {err}"))?;
    fs::write(&output, encoded)
        .map_err(|err| format!("failed to write {}: {err}", output.display()))?;

    println!("subject: {}", artifact.matrix.subject_revision);
    println!("lanes:   {}", artifact.matrix.lanes.len());
    println!("digest:  {}", artifact.matrix_digest);
    println!("output:  {}", output.display());
    Ok(())
}
