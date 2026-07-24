// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verify the secret-free Series 21 hybrid and transparency-gossip bundle.
//!
//! Usage:
//!   cargo run -p symthaea-vocal-tract --example checkpoint_series21_hybrid_gossip_verifier -- \
//!     series21-public-bundle.postcard 1784548800

use std::{env, fs, process};

use symthaea_vocal_tract::{
    CheckpointOperationalTrustMetrics, CheckpointOperationalTrustRequirements,
    MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES, apply_series21_public_verifiability,
    assemble_checkpoint_operational_trust_evidence,
    decode_checkpoint_series21_public_verification_bundle,
};

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let bundle_path = arguments.next().ok_or_else(|| {
        "expected Series 21 public bundle path and verification Unix timestamp".to_owned()
    })?;
    let verification_time = arguments
        .next()
        .ok_or_else(|| "missing verification Unix timestamp".to_owned())?
        .parse::<u64>()
        .map_err(|_| "verification timestamp must be an unsigned integer".to_owned())?;
    if arguments.next().is_some() || verification_time == 0 {
        return Err("unexpected arguments or zero verification timestamp".to_owned());
    }

    let metadata = fs::metadata(&bundle_path)
        .map_err(|error| format!("failed to inspect {bundle_path}: {error}"))?;
    if !metadata.is_file() || metadata.len() as usize > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err("bundle is not a bounded regular file".to_owned());
    }
    let encoded =
        fs::read(&bundle_path).map_err(|error| format!("failed to read {bundle_path}: {error}"))?;
    let bundle = decode_checkpoint_series21_public_verification_bundle(&encoded)
        .map_err(|error| format!("bundle decode failed: {error}"))?;
    let summary = bundle
        .verify(verification_time)
        .map_err(|error| format!("Series 21 public verification failed: {error}"))?;

    let mut metrics = CheckpointOperationalTrustMetrics::default();
    apply_series21_public_verifiability(&mut metrics, &summary)
        .map_err(|error| format!("evidence application failed: {error}"))?;
    let report = assemble_checkpoint_operational_trust_evidence(
        metrics,
        CheckpointOperationalTrustRequirements::series_21_delta(),
    );
    for gate in &report.gates {
        if gate.required {
            println!("{}: {:?}", gate.name, gate.status);
        }
    }
    if !report.passed() {
        return Err("Series 21 hybrid/gossip gates did not pass".to_owned());
    }
    println!(
        "verified {} ML-DSA-65 signatures from {} organizations and {} independent gossip origins",
        summary.hybrid().valid_post_quantum_signatures,
        summary.hybrid().unique_organizations,
        summary.gossip().unique_origins,
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}
