// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trusted, non-executing verifier for a persisted direct GWT-1 evidence set.
//!
//! This binary never instantiates Symthaea's cognitive managers. It is intended
//! to run from a trusted evaluator revision in a separate job after candidate
//! code has finished. It recomputes the envelope resolution from raw bytes and
//! checks the persisted source/run/toolchain identity against independently
//! derived expectations supplied by the trusted workflow.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use symthaea_psych_bench::benchmarks::butlin::{
    GWT1_SPECIALISTS_V1, Gwt1EvidenceEnvelopeResolutionV1, Gwt1EvidenceEnvelopeV1,
    resolve_gwt1_evidence_envelope_v1,
};

#[derive(Debug, Deserialize)]
struct ExpectedSourceIdentity {
    source_commit_sha: String,
    source_tree_sha: String,
    specialist_blob_shas: BTreeMap<String, String>,
}

fn required_env(name: &str) -> Result<String, io::Error> {
    env::var(name).map_err(|_| io::Error::other(format!("missing required environment variable {name}")))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn main() {
    if let Err(error) = verify() {
        eprintln!("trusted GWT-1 evidence verification failed: {error}");
        std::process::exit(2);
    }
}

fn verify() -> Result<(), Box<dyn std::error::Error>> {
    let evidence_dir = PathBuf::from(required_env("SYMTHAEA_GWT1_EVIDENCE_DIR")?);
    let expected_identity_path = PathBuf::from(required_env("EXPECTED_GWT1_SOURCE_IDENTITY")?);
    let expected_toolchain_path = PathBuf::from(required_env("EXPECTED_GWT1_TOOLCHAIN")?);
    let expected_run_id = required_env("EXPECTED_GWT1_EXECUTION_RUN_ID")?;

    let raw_bytes = fs::read(evidence_dir.join("raw_observations.json"))?;
    let envelope: Gwt1EvidenceEnvelopeV1 =
        read_json(&evidence_dir.join("evidence_envelope.json"))?;
    let stored_resolution: Gwt1EvidenceEnvelopeResolutionV1 =
        read_json(&evidence_dir.join("resolution.json"))?;

    let recomputed = resolve_gwt1_evidence_envelope_v1(&envelope, &raw_bytes);
    if recomputed != stored_resolution {
        return Err(io::Error::other(format!(
            "stored resolution differs from trusted recomputation: stored={:?}, recomputed={:?}",
            stored_resolution.outcome, recomputed.outcome
        ))
        .into());
    }

    let expected: ExpectedSourceIdentity = read_json(&expected_identity_path)?;
    let receipt = &envelope.receipt;

    if receipt.source_commit_sha != expected.source_commit_sha {
        return Err(io::Error::other(format!(
            "source commit mismatch: receipt={}, expected={}",
            receipt.source_commit_sha, expected.source_commit_sha
        ))
        .into());
    }
    if receipt.source_tree_sha != expected.source_tree_sha {
        return Err(io::Error::other(format!(
            "source tree mismatch: receipt={}, expected={}",
            receipt.source_tree_sha, expected.source_tree_sha
        ))
        .into());
    }
    if receipt.execution_run_id != expected_run_id {
        return Err(io::Error::other(format!(
            "execution run mismatch: receipt={}, expected={expected_run_id}",
            receipt.execution_run_id
        ))
        .into());
    }

    let expected_toolchain = fs::read_to_string(expected_toolchain_path)?;
    if receipt.toolchain.trim_end() != expected_toolchain.trim_end() {
        return Err(io::Error::other("toolchain identity mismatch").into());
    }

    let expected_ids: BTreeSet<String> = GWT1_SPECIALISTS_V1
        .iter()
        .map(|id| (*id).to_string())
        .collect();
    let expected_blob_ids: BTreeSet<String> = expected.specialist_blob_shas.keys().cloned().collect();
    if expected_blob_ids != expected_ids {
        return Err(io::Error::other(format!(
            "trusted expected specialist set mismatch: observed={expected_blob_ids:?}, expected={expected_ids:?}"
        ))
        .into());
    }

    let receipt_blob_shas: BTreeMap<String, String> = receipt
        .specialists
        .iter()
        .map(|specialist| (specialist.id.clone(), specialist.source_blob_sha.clone()))
        .collect();
    if receipt_blob_shas.len() != receipt.specialists.len() {
        return Err(io::Error::other("duplicate specialist IDs in receipt").into());
    }
    if receipt_blob_shas != expected.specialist_blob_shas {
        return Err(io::Error::other(format!(
            "specialist blob identity mismatch: receipt={receipt_blob_shas:?}, expected={:?}",
            expected.specialist_blob_shas
        ))
        .into());
    }

    let outcome_text = fs::read_to_string(evidence_dir.join("outcome.txt"))?;
    let expected_outcome = format!("{:?}", recomputed.outcome);
    if outcome_text.trim() != expected_outcome {
        return Err(io::Error::other(format!(
            "outcome text mismatch: file={:?}, recomputed={expected_outcome:?}",
            outcome_text.trim()
        ))
        .into());
    }

    println!("Trusted GWT-1 evidence verified: {:?}", recomputed.outcome);
    Ok(())
}
