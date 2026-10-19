// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Content-addressed verification for source artifacts behind culinary outcomes.
//!
//! Evidence labels such as `qualified` are metadata declarations, not proof that
//! the referenced bytes exist or are unchanged. This module verifies an exact
//! manifest digest pinned by a separate trusted input, then verifies every
//! declared artifact's SHA-256 and byte length against supplied payload bytes.
//! It also binds experiment protocol identity, process-trace IDs, and outcome
//! evidence source/version pairs to the manifest.
//!
//! SHA-256 equality proves byte integrity against the expected digest, not issuer
//! identity, truth of a claim, calibration quality, or that randomization was
//! physically followed. The expected manifest digest must be obtained from a
//! separate trusted/pinned record; reading it from the same untrusted manifest
//! would not authenticate that manifest.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::paired_outcome_analysis::{
    PairedOutcomeAnalysisV1, analyze_paired_outcomes_v1,
};

const ARTIFACT_KINDS: &[&str] = &[
    "protocol",
    "process_trace",
    "raw_measurement",
    "derived_outcome",
    "instrument_calibration",
    "other",
];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceBundleWireV1 {
    schema_version: String,
    bundle_id: String,
    experiment_id: String,
    protocol_sha256: String,
    artifacts: Vec<EvidenceArtifactWireV1>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceArtifactWireV1 {
    artifact_id: String,
    artifact_kind: String,
    path: String,
    source_ref: String,
    source_version: String,
    sha256: String,
    byte_length: u64,
}

/// Capability object produced only by successful exact-byte bundle verification.
///
/// Fields are private to prevent callers from constructing a "verified" receipt
/// from status labels. The object binds one canonicalized in-memory experiment
/// value to the exact manifest bytes and all payloads checked by the verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCulinaryEvidenceBundleV1 {
    manifest_sha256: String,
    experiment_fingerprint_sha256: String,
    experiment_id: String,
    protocol_sha256: String,
    artifact_count: usize,
    verified_byte_count: u64,
    process_trace_count: usize,
    outcome_source_count: usize,
}

impl VerifiedCulinaryEvidenceBundleV1 {
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn experiment_id(&self) -> &str {
        &self.experiment_id
    }

    pub fn protocol_sha256(&self) -> &str {
        &self.protocol_sha256
    }

    pub fn artifact_count(&self) -> usize {
        self.artifact_count
    }

    pub fn verified_byte_count(&self) -> u64 {
        self.verified_byte_count
    }

    pub fn process_trace_count(&self) -> usize {
        self.process_trace_count
    }

    pub fn outcome_source_count(&self) -> usize {
        self.outcome_source_count
    }
}

fn non_empty<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str).filter(|text| !text.trim().is_empty())
}

fn strict_non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn experiment_fingerprint(experiment: &Value) -> Result<String, Vec<String>> {
    let bytes = serde_json::to_vec(experiment)
        .map_err(|error| vec![format!("failed to serialize experiment for binding: {error}")])?;
    Ok(sha256_hex(&bytes))
}

/// Validate a normalized, safe relative artifact path.
///
/// This is a lexical path check only. The caller that resolves the path on disk
/// must additionally reject symlinks and verify that canonical resolution stays
/// within the trusted repository/artifact root before reading its bytes.
fn safe_relative_path(path: &str) -> bool {
    if path.is_empty()
        || path.starts_with('/')
        || path.starts_with('~')
        || path.contains('\\')
        || path.contains(':')
        || path.chars().any(char::is_control)
    {
        return false;
    }
    path.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

/// Verify an experiment's content-addressed evidence bundle.
///
/// `expected_manifest_sha256` must come from a separate trusted record, such as
/// a signed run receipt or reviewed immutable reference. `artifact_bytes` maps
/// manifest artifact IDs to the exact bytes resolved by a trusted caller. Missing
/// and extra payloads are both rejected. Payload resolution and symlink-safe I/O
/// are the caller's responsibility; this function verifies bytes once supplied.
pub fn verify_culinary_evidence_bundle_v1(
    experiment: &Value,
    manifest_bytes: &[u8],
    expected_manifest_sha256: &str,
    artifact_bytes: &BTreeMap<String, Vec<u8>>,
) -> Result<VerifiedCulinaryEvidenceBundleV1, Vec<String>> {
    let mut errors = Vec::new();

    if !lower_sha256(expected_manifest_sha256) {
        errors.push("expected_manifest_sha256 must be 64 lowercase hexadecimal characters".to_owned());
    }
    let actual_manifest_sha256 = sha256_hex(manifest_bytes);
    if actual_manifest_sha256 != expected_manifest_sha256 {
        errors.push("exact evidence-manifest bytes do not match the separately pinned manifest SHA-256".to_owned());
    }

    let wire: EvidenceBundleWireV1 = match serde_json::from_slice(manifest_bytes) {
        Ok(value) => value,
        Err(error) => {
            return Err(vec![format!("evidence manifest is not valid strict JSON for the V1 schema: {error}")]);
        }
    };
    if wire.schema_version != "1.0.0" {
        errors.push("manifest schema_version must be 1.0.0".to_owned());
    }
    if !strict_non_empty(&wire.bundle_id) {
        errors.push("manifest bundle_id is required".to_owned());
    }
    if !strict_non_empty(&wire.experiment_id) {
        errors.push("manifest experiment_id is required".to_owned());
    }
    if !lower_sha256(&wire.protocol_sha256) {
        errors.push("manifest protocol_sha256 must be 64 lowercase hexadecimal characters".to_owned());
    }
    if wire.experiment_id != non_empty(experiment, "experiment_id").unwrap_or("") {
        errors.push("manifest experiment_id must exactly match experiment_id".to_owned());
    }
    if wire.protocol_sha256 != non_empty(experiment, "protocol_sha256").unwrap_or("") {
        errors.push("manifest protocol_sha256 must exactly match the preregistered protocol digest".to_owned());
    }

    if wire.artifacts.is_empty() {
        errors.push("manifest artifacts must not be empty".to_owned());
    }
    let mut artifact_ids = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut source_refs = BTreeSet::new();
    let mut by_id = BTreeMap::<String, EvidenceArtifactWireV1>::new();
    let mut by_source_ref = BTreeMap::<String, EvidenceArtifactWireV1>::new();
    let mut protocol_artifacts = Vec::new();

    for (index, artifact) in wire.artifacts.iter().enumerate() {
        let label = format!("artifacts[{index}]");
        if !strict_non_empty(&artifact.artifact_id) {
            errors.push(format!("{label}.artifact_id is required"));
        }
        if !strict_non_empty(&artifact.path) || !safe_relative_path(&artifact.path) {
            errors.push(format!("{label}.path must be a safe normalized relative path"));
        }
        if !strict_non_empty(&artifact.source_ref) {
            errors.push(format!("{label}.source_ref is required"));
        }
        if !strict_non_empty(&artifact.source_version) {
            errors.push(format!("{label}.source_version is required"));
        }
        if !ARTIFACT_KINDS.contains(&artifact.artifact_kind.as_str()) {
            errors.push(format!("{label}.artifact_kind is invalid"));
        }
        if !lower_sha256(&artifact.sha256) {
            errors.push(format!("{label}.sha256 must be 64 lowercase hexadecimal characters"));
        }
        if artifact.byte_length == 0 {
            errors.push(format!("{label}.byte_length must be positive"));
        }
        if !artifact_ids.insert(artifact.artifact_id.clone()) {
            errors.push(format!("duplicate artifact_id {:?}", artifact.artifact_id));
        }
        if !paths.insert(artifact.path.clone()) {
            errors.push(format!("duplicate artifact path {:?}", artifact.path));
        }
        if !source_refs.insert(artifact.source_ref.clone()) {
            errors.push(format!("duplicate artifact source_ref {:?}", artifact.source_ref));
        }
        match artifact.artifact_kind.as_str() {
            "protocol" => protocol_artifacts.push(artifact),
            _ => {}
        }
        by_id.insert(artifact.artifact_id.clone(), artifact.clone());
        by_source_ref.insert(artifact.source_ref.clone(), artifact.clone());
    }

    let mut referenced_artifact_ids = BTreeSet::new();
    if protocol_artifacts.len() != 1 {
        errors.push(format!(
            "manifest must contain exactly one protocol artifact, found {}",
            protocol_artifacts.len()
        ));
    } else {
        if protocol_artifacts[0].sha256 != wire.protocol_sha256 {
            errors.push("protocol artifact SHA-256 must equal manifest protocol_sha256".to_owned());
        }
        referenced_artifact_ids.insert(protocol_artifacts[0].artifact_id.clone());
    }

    let expected_ids: BTreeSet<String> = by_id.keys().cloned().collect();
    let supplied_ids: BTreeSet<String> = artifact_bytes.keys().cloned().collect();
    for missing in expected_ids.difference(&supplied_ids) {
        errors.push(format!("payload bytes missing for artifact_id {missing:?}"));
    }
    for extra in supplied_ids.difference(&expected_ids) {
        errors.push(format!("undeclared payload bytes supplied for artifact_id {extra:?}"));
    }

    let mut verified_byte_count = 0_u64;
    for (artifact_id, artifact) in &by_id {
        let Some(bytes) = artifact_bytes.get(artifact_id) else {
            continue;
        };
        if artifact.byte_length != bytes.len() as u64 {
            errors.push(format!(
                "artifact {artifact_id:?} byte_length mismatch: declared {}, actual {}",
                artifact.byte_length,
                bytes.len()
            ));
        }
        let actual_sha = sha256_hex(bytes);
        if actual_sha != artifact.sha256 {
            errors.push(format!("artifact {artifact_id:?} payload SHA-256 mismatch"));
        }
        verified_byte_count = match verified_byte_count.checked_add(bytes.len() as u64) {
            Some(value) => value,
            None => {
                errors.push("total verified payload byte count overflowed".to_owned());
                u64::MAX
            }
        };
    }

    let mut verified_process_trace_ids = BTreeSet::new();
    let mut seen_process_trace_ids = BTreeSet::new();
    let blocks: &[Value] = match experiment.get("blocks").and_then(Value::as_array) {
        Some(items) => items.as_slice(),
        None => {
            errors.push("experiment.blocks must be an array".to_owned());
            &[]
        }
    };
    let mut outcome_source_refs = BTreeSet::new();

    for (block_index, block) in blocks.iter().enumerate() {
        let runs: &[Value] = match block.get("runs").and_then(Value::as_array) {
            Some(items) => items.as_slice(),
            None => {
                errors.push(format!("experiment.blocks[{block_index}].runs must be an array"));
                continue;
            }
        };
        for (run_index, run) in runs.iter().enumerate() {
            let label = format!("blocks[{block_index}].runs[{run_index}]");
            let Some(trace_id) = non_empty(run, "process_trace_id") else {
                errors.push(format!("{label}.process_trace_id is required"));
                continue;
            };
            if !seen_process_trace_ids.insert(trace_id.to_owned()) {
                errors.push(format!("{label}.process_trace_id {trace_id:?} is reused by multiple runs"));
            }
            match by_id.get(trace_id) {
                Some(artifact) if artifact.artifact_kind == "process_trace" => {
                    verified_process_trace_ids.insert(trace_id.to_owned());
                    referenced_artifact_ids.insert(trace_id.to_owned());
                }
                _ => errors.push(format!(
                    "{label}.process_trace_id {trace_id:?} must resolve to a declared process_trace artifact with the same artifact_id"
                )),
            }

            let outcomes = match run.get("outcomes").and_then(Value::as_array) {
                Some(items) => items,
                None => {
                    errors.push(format!("{label}.outcomes must be an array"));
                    continue;
                }
            };
            for (outcome_index, outcome) in outcomes.iter().enumerate() {
                let outcome_label = format!("{label}.outcomes[{outcome_index}]");
                let evidence = outcome.get("evidence").unwrap_or(&Value::Null);
                let Some(source_ref) = non_empty(evidence, "source_ref") else {
                    errors.push(format!("{outcome_label}.evidence.source_ref is required"));
                    continue;
                };
                let Some(source_version) = non_empty(evidence, "source_version") else {
                    errors.push(format!("{outcome_label}.evidence.source_version is required"));
                    continue;
                };
                outcome_source_refs.insert(source_ref.to_owned());
                match by_source_ref.get(source_ref) {
                    Some(artifact) => {
                        if artifact.source_version != source_version {
                            errors.push(format!(
                                "{outcome_label} source_version does not match artifact manifest for {source_ref:?}"
                            ));
                        }
                        if !matches!(artifact.artifact_kind.as_str(), "raw_measurement" | "derived_outcome") {
                            errors.push(format!(
                                "{outcome_label} source_ref resolves to incompatible artifact_kind {:?}",
                                artifact.artifact_kind
                            ));
                        }
                        referenced_artifact_ids.insert(artifact.artifact_id.clone());
                        if non_empty(evidence, "evidence_status") == Some("qualified")
                            && source_ref.starts_with("urn:synthetic:")
                        {
                            errors.push(format!("{outcome_label} cannot be qualified from a synthetic source"));
                        }
                    }
                    None => errors.push(format!(
                        "{outcome_label}.evidence.source_ref {source_ref:?} has no manifest artifact"
                    )),
                }
            }
        }
    }

    for artifact_id in by_id.keys() {
        if !referenced_artifact_ids.contains(artifact_id) {
            errors.push(format!("manifest artifact {artifact_id:?} is not referenced by protocol, process trace, or outcome evidence"));
        }
    }

    let experiment_fp = match experiment_fingerprint(experiment) {
        Ok(value) => value,
        Err(nested) => {
            errors.extend(nested);
            String::new()
        }
    };
    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    Ok(VerifiedCulinaryEvidenceBundleV1 {
        manifest_sha256: actual_manifest_sha256,
        experiment_fingerprint_sha256: experiment_fp,
        experiment_id: wire.experiment_id,
        protocol_sha256: wire.protocol_sha256,
        artifact_count: by_id.len(),
        verified_byte_count,
        process_trace_count: verified_process_trace_ids.len(),
        outcome_source_count: outcome_source_refs.len(),
    })
}

/// Analyze only the exact experiment value whose artifact bundle was verified.
///
/// The verified object is not serializable and its fields are private, preventing
/// callers from manufacturing a successful receipt from evidence status labels.
/// This binds byte integrity to the in-memory experiment consumed by the analyzer.
pub fn analyze_paired_outcomes_with_verified_bundle_v1(
    experiment: &Value,
    verified: &VerifiedCulinaryEvidenceBundleV1,
) -> Result<PairedOutcomeAnalysisV1, Vec<String>> {
    let mut errors = Vec::new();
    if non_empty(experiment, "experiment_id") != Some(verified.experiment_id.as_str()) {
        errors.push("verified evidence bundle belongs to a different experiment_id".to_owned());
    }
    if non_empty(experiment, "protocol_sha256") != Some(verified.protocol_sha256.as_str()) {
        errors.push("verified evidence bundle belongs to a different protocol_sha256".to_owned());
    }
    match experiment_fingerprint(experiment) {
        Ok(actual) if actual == verified.experiment_fingerprint_sha256 => {}
        Ok(_) => errors.push("experiment contents changed after evidence bundle verification".to_owned()),
        Err(nested) => errors.extend(nested),
    }
    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }
    analyze_paired_outcomes_v1(experiment)
}
