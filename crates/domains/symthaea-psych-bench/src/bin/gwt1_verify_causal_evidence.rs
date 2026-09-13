// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trusted, non-executing verifier for persisted causal GWT-1 evidence.
//!
//! This binary runs from the trusted evaluator revision. It never invokes the
//! four production specialist managers. It parses the candidate-produced raw
//! artifacts, independently recomputes the paired causal envelope resolution,
//! and binds the embedded execution identity to values derived by the trusted
//! workflow from Git, the pinned toolchain, and the workflow run itself.

#[cfg(not(feature = "symthaea-backend"))]
fn main() {
    eprintln!("gwt1_verify_causal_evidence requires --features symthaea-backend");
    std::process::exit(2);
}

#[cfg(feature = "symthaea-backend")]
fn main() {
    if let Err(error) = enabled::verify() {
        eprintln!("trusted GWT-1 causal evidence verification failed: {error}");
        std::process::exit(2);
    }
}

#[cfg(feature = "symthaea-backend")]
mod enabled {
    use std::collections::{BTreeMap, BTreeSet};
    use std::env;
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};

    use serde::Deserialize;
    use symthaea_psych_bench::benchmarks::butlin::{
        GWT1_SPECIALISTS_V1, Gwt1CausalEvidenceEnvelopeResolutionV1,
        Gwt1CausalEvidenceEnvelopeV1, resolve_gwt1_causal_envelope_v1,
    };

    #[derive(Debug, Deserialize)]
    struct ExpectedSourceIdentity {
        source_commit_sha: String,
        source_tree_sha: String,
        specialist_blob_shas: BTreeMap<String, String>,
    }

    fn required_env(name: &str) -> Result<String, io::Error> {
        env::var(name)
            .map_err(|_| io::Error::other(format!("missing required environment variable {name}")))
    }

    fn read_json<T: for<'de> Deserialize<'de>>(
        path: &Path,
    ) -> Result<T, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    pub fn verify() -> Result<(), Box<dyn std::error::Error>> {
        let evidence_dir = PathBuf::from(required_env("SYMTHAEA_GWT1_CAUSAL_EVIDENCE_DIR")?);
        let expected_identity_path = PathBuf::from(required_env("EXPECTED_GWT1_SOURCE_IDENTITY")?);
        let expected_toolchain_path = PathBuf::from(required_env("EXPECTED_GWT1_TOOLCHAIN")?);
        let expected_run_id = required_env("EXPECTED_GWT1_EXECUTION_RUN_ID")?;

        let causal_raw = fs::read(evidence_dir.join("causal_observations.json"))?;
        let matched_sham_raw = fs::read(evidence_dir.join("matched_sham_observations.json"))?;
        let envelope: Gwt1CausalEvidenceEnvelopeV1 =
            read_json(&evidence_dir.join("causal_evidence_envelope.json"))?;
        let stored_resolution: Gwt1CausalEvidenceEnvelopeResolutionV1 =
            read_json(&evidence_dir.join("resolution.json"))?;

        let recomputed =
            resolve_gwt1_causal_envelope_v1(&envelope, &causal_raw, &matched_sham_raw);
        if recomputed != stored_resolution {
            return Err(io::Error::other(format!(
                "stored causal resolution differs from trusted recomputation: stored={:?}, recomputed={:?}",
                stored_resolution.outcome, recomputed.outcome
            ))
            .into());
        }

        let expected: ExpectedSourceIdentity = read_json(&expected_identity_path)?;
        let identity = &envelope.execution_identity;

        if identity.source_commit_sha != expected.source_commit_sha {
            return Err(io::Error::other(format!(
                "source commit mismatch: envelope={}, expected={}",
                identity.source_commit_sha, expected.source_commit_sha
            ))
            .into());
        }
        if identity.source_tree_sha != expected.source_tree_sha {
            return Err(io::Error::other(format!(
                "source tree mismatch: envelope={}, expected={}",
                identity.source_tree_sha, expected.source_tree_sha
            ))
            .into());
        }
        if identity.execution_run_id != expected_run_id {
            return Err(io::Error::other(format!(
                "execution run mismatch: envelope={}, expected={expected_run_id}",
                identity.execution_run_id
            ))
            .into());
        }

        let expected_toolchain = fs::read_to_string(expected_toolchain_path)?;
        if identity.toolchain.trim_end() != expected_toolchain.trim_end() {
            return Err(io::Error::other("toolchain identity mismatch").into());
        }

        let expected_ids: BTreeSet<String> = GWT1_SPECIALISTS_V1
            .iter()
            .map(|id| (*id).to_string())
            .collect();
        let expected_blob_ids: BTreeSet<String> =
            expected.specialist_blob_shas.keys().cloned().collect();
        if expected_blob_ids != expected_ids {
            return Err(io::Error::other(format!(
                "trusted expected specialist set mismatch: observed={expected_blob_ids:?}, expected={expected_ids:?}"
            ))
            .into());
        }
        if identity.specialist_blob_shas != expected.specialist_blob_shas {
            return Err(io::Error::other(format!(
                "specialist blob identity mismatch: envelope={:?}, expected={:?}",
                identity.specialist_blob_shas, expected.specialist_blob_shas
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

        println!(
            "Trusted GWT-1 causal evidence verified: {:?}",
            recomputed.outcome
        );
        Ok(())
    }
}
