// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Structural inspection of externally witnessed SAT trace-head receipts.
//!
//! This module deliberately stops before cryptographic trust. It verifies that
//! a receipt binds the exact trace head, that its canonical statement digest is
//! correct, that referenced transparency metadata is internally consistent,
//! and that the declared distinct-witness threshold is met. It does NOT verify
//! signatures, transparency inclusion/consistency proofs, key status,
//! revocation, witness independence, or the trustworthiness of referenced
//! artifacts. The report always makes those omissions explicit.

use super::{StrictJson, TraceVerificationReport, digest_json, is_digest};
use chrono::{DateTime, FixedOffset};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fmt;

const TRACE_HEAD_DOMAIN: &str = "SAT-TRACE-HEAD-v0.1\n";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointVerificationError {
    InvalidJson(String),
    DuplicateJsonKey(String),
    InvalidShape(String),
    InvalidDigest(String),
    InvalidTimestamp(String),
    TraceNotInternallyConsistent,
    RunIdMismatch,
    ManifestDigestMismatch,
    HeadSequenceMismatch,
    HeadEventDigestMismatch,
    StatementDigestMismatch,
    DuplicateWitnessId(String),
    WitnessPolicyDigestMismatch(String),
    WitnessDoesNotMatchTransparencyEntry(String),
    TreeSizeDoesNotContainEntry(String),
    MissingConsistencyProof(String),
    ConflictingCheckpointRoot {
        service_id: String,
        log_id: String,
        tree_size: u64,
    },
    WitnessThresholdNotMet {
        required: u64,
        found: usize,
    },
}

impl fmt::Display for CheckpointVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(message) => write!(f, "invalid checkpoint JSON: {message}"),
            Self::DuplicateJsonKey(key) => write!(f, "duplicate checkpoint JSON key: {key}"),
            Self::InvalidShape(message) => write!(f, "invalid checkpoint shape: {message}"),
            Self::InvalidDigest(field) => write!(f, "invalid digest in {field}"),
            Self::InvalidTimestamp(field) => write!(f, "invalid RFC 3339 timestamp in {field}"),
            Self::TraceNotInternallyConsistent => {
                write!(f, "trace must pass internal chain consistency before checkpoint inspection")
            }
            Self::RunIdMismatch => write!(f, "checkpoint receipt run_id differs from trace"),
            Self::ManifestDigestMismatch => {
                write!(f, "checkpoint receipt manifest_digest differs from trace")
            }
            Self::HeadSequenceMismatch => {
                write!(f, "checkpoint receipt head_sequence differs from trace")
            }
            Self::HeadEventDigestMismatch => {
                write!(f, "checkpoint receipt head_event_digest differs from trace")
            }
            Self::StatementDigestMismatch => {
                write!(f, "trace-head statement digest does not match canonical statement")
            }
            Self::DuplicateWitnessId(id) => write!(f, "duplicate witness identity: {id}"),
            Self::WitnessPolicyDigestMismatch(id) => {
                write!(f, "witness {id} used a different trust-policy digest")
            }
            Self::WitnessDoesNotMatchTransparencyEntry(id) => {
                write!(f, "witness {id} does not match a supplied transparency entry")
            }
            Self::TreeSizeDoesNotContainEntry(log_id) => {
                write!(f, "tree size does not contain the claimed entry for log {log_id}")
            }
            Self::MissingConsistencyProof(log_id) => {
                write!(f, "a prior checkpoint is referenced but no consistency proof is supplied for log {log_id}")
            }
            Self::ConflictingCheckpointRoot {
                service_id,
                log_id,
                tree_size,
            } => write!(
                f,
                "conflicting roots for service {service_id}, log {log_id}, tree size {tree_size}"
            ),
            Self::WitnessThresholdNotMet { required, found } => {
                write!(f, "declared witness threshold {required} not met by {found} distinct witness IDs")
            }
        }
    }
}

impl std::error::Error for CheckpointVerificationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointReceiptInspection {
    pub receipt_id: String,
    pub run_id: String,
    pub manifest_digest: String,
    pub head_sequence: u64,
    pub head_event_digest: String,
    pub statement_digest: String,
    pub transparency_entry_count: usize,
    pub distinct_witness_id_count: usize,
    pub required_witness_count: u64,
    pub declared_witness_count_meets_threshold: bool,
    pub statement_digest_verified: bool,
    pub receipt_matches_trace_head: bool,
    /// True means non-empty references were structurally present, not fetched or verified.
    pub artifact_references_present: bool,
    /// Always false: this structural inspector never verifies cryptographic signatures.
    pub signatures_verified: bool,
    /// Always false: inclusion proofs are referenced but not cryptographically checked here.
    pub inclusion_proofs_verified: bool,
    /// Always false: consistency proofs are referenced but not cryptographically checked here.
    pub consistency_proofs_verified: bool,
    /// Always false: identity uniqueness does not prove operational independence.
    pub witness_independence_verified: bool,
    /// Always false: this module does not authenticate trust roots or checkpoint custody.
    pub checkpoint_authenticated: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointReceipt {
    schema_version: String,
    receipt_id: String,
    issued_at: String,
    run_id: String,
    manifest_digest: String,
    head_sequence: u64,
    head_event_digest: String,
    trace_head_statement: TraceHeadStatement,
    statement_digest: String,
    transparency_entries: Vec<TransparencyEntry>,
    witness_attestations: Vec<WitnessAttestation>,
    trust_policy: TrustPolicy,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    limitations: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceHeadStatement {
    schema_version: String,
    run_id: String,
    manifest_digest: String,
    head_sequence: u64,
    head_event_digest: String,
    checkpointed_at: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransparencyEntry {
    service_profile: String,
    service_id: String,
    log_id: String,
    entry_index: u64,
    tree_size: u64,
    root_hash: String,
    root_hash_algorithm: String,
    root_hash_encoding: String,
    signed_checkpoint_ref: String,
    inclusion_proof_ref: String,
    consistency_proof_ref: Option<String>,
    signed_statement_ref: String,
    checkpoint_key_id: String,
    verification_policy_ref: String,
    previous_checkpoint_ref: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WitnessAttestation {
    witness_id: String,
    service_id: String,
    log_id: String,
    tree_size: u64,
    root_hash: String,
    root_hash_algorithm: String,
    root_hash_encoding: String,
    observed_at: String,
    signature_ref: String,
    key_id: String,
    trust_policy_digest: String,
    independence_basis_ref: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustPolicy {
    policy_id: String,
    version: String,
    policy_digest: String,
    minimum_independent_witnesses: u64,
    key_status_evidence_ref: String,
    revocation_snapshot_ref: String,
    governance_policy_ref: String,
}

fn valid_digest(value: &str, field: &str) -> Result<(), CheckpointVerificationError> {
    if is_digest(value) {
        Ok(())
    } else {
        Err(CheckpointVerificationError::InvalidDigest(field.to_owned()))
    }
}

fn valid_timestamp(value: &str, field: &str) -> Result<(), CheckpointVerificationError> {
    if DateTime::<FixedOffset>::parse_from_rfc3339(value).is_ok() {
        Ok(())
    } else {
        Err(CheckpointVerificationError::InvalidTimestamp(field.to_owned()))
    }
}

fn non_empty(value: &str, field: &str) -> Result<(), CheckpointVerificationError> {
    if value.trim().is_empty() {
        Err(CheckpointVerificationError::InvalidShape(format!(
            "{field} must be non-empty"
        )))
    } else {
        Ok(())
    }
}

fn map_json_error(error: serde_json::Error) -> CheckpointVerificationError {
    let message = error.to_string();
    if let Some(key) = message.split("duplicate JSON key: ").nth(1) {
        CheckpointVerificationError::DuplicateJsonKey(key.to_owned())
    } else {
        CheckpointVerificationError::InvalidJson(message)
    }
}

/// Inspect a checkpoint receipt against an internally consistent trace report.
///
/// This recomputes the canonical trace-head statement digest and performs
/// structural cross-checks. It does not fetch or verify referenced artifacts.
pub fn inspect_checkpoint_receipt(
    receipt_json: &str,
    trace: &TraceVerificationReport,
) -> Result<CheckpointReceiptInspection, CheckpointVerificationError> {
    if !trace.event_chain_consistent || trace.event_count == 0 || trace.head_digest.is_empty() {
        return Err(CheckpointVerificationError::TraceNotInternallyConsistent);
    }

    let raw: StrictJson = serde_json::from_str(receipt_json).map_err(map_json_error)?;
    if !raw.is_object() {
        return Err(CheckpointVerificationError::InvalidShape(
            "receipt must be a JSON object".to_owned(),
        ));
    }
    let receipt: CheckpointReceipt =
        serde_json::from_str(receipt_json).map_err(map_json_error)?;

    if receipt.schema_version != "sat-trace-checkpoint-receipt/v0.1" {
        return Err(CheckpointVerificationError::InvalidShape(
            "unsupported schema_version".to_owned(),
        ));
    }
    non_empty(&receipt.receipt_id, "receipt_id")?;
    valid_timestamp(&receipt.issued_at, "issued_at")?;
    valid_digest(&receipt.manifest_digest, "manifest_digest")?;
    valid_digest(&receipt.head_event_digest, "head_event_digest")?;
    valid_digest(&receipt.statement_digest, "statement_digest")?;
    if receipt.run_id != trace.run_id {
        return Err(CheckpointVerificationError::RunIdMismatch);
    }
    if receipt.manifest_digest != trace.manifest_digest {
        return Err(CheckpointVerificationError::ManifestDigestMismatch);
    }
    if receipt.head_sequence != trace.event_count as u64 {
        return Err(CheckpointVerificationError::HeadSequenceMismatch);
    }
    if receipt.head_event_digest != trace.head_digest {
        return Err(CheckpointVerificationError::HeadEventDigestMismatch);
    }

    let statement = &receipt.trace_head_statement;
    if statement.schema_version != "sat-trace-head-statement/v0.1"
        || statement.run_id != receipt.run_id
        || statement.manifest_digest != receipt.manifest_digest
        || statement.head_sequence != receipt.head_sequence
        || statement.head_event_digest != receipt.head_event_digest
    {
        return Err(CheckpointVerificationError::InvalidShape(
            "trace_head_statement does not match the outer receipt identity".to_owned(),
        ));
    }
    valid_timestamp(&statement.checkpointed_at, "trace_head_statement.checkpointed_at")?;

    let raw_statement = raw.get("trace_head_statement").ok_or_else(|| {
        CheckpointVerificationError::InvalidShape("missing trace_head_statement".to_owned())
    })?;
    let computed_statement_digest =
        digest_json(TRACE_HEAD_DOMAIN, raw_statement);
    if computed_statement_digest != receipt.statement_digest {
        return Err(CheckpointVerificationError::StatementDigestMismatch);
    }

    if receipt.transparency_entries.is_empty() {
        return Err(CheckpointVerificationError::InvalidShape(
            "at least one transparency entry is required".to_owned(),
        ));
    }
    if receipt.witness_attestations.is_empty() {
        return Err(CheckpointVerificationError::InvalidShape(
            "at least one witness attestation is required".to_owned(),
        ));
    }

    // Detect conflicting roots for one service/log/tree-size tuple.
    let mut roots: HashMap<(String, String, u64), (String, String, String)> = HashMap::new();
    for entry in &receipt.transparency_entries {
        if ![
            "scitt-rfc9943",
            "sigstore-rekor-v2",
            "signed-note",
            "private-transparency-service",
        ].contains(&entry.service_profile.as_str()) {
            return Err(CheckpointVerificationError::InvalidShape(
                "unsupported transparency service profile".to_owned(),
            ));
        }
        non_empty(&entry.service_id, "transparency_entries.service_id")?;
        non_empty(&entry.log_id, "transparency_entries.log_id")?;
        non_empty(&entry.root_hash, "transparency_entries.root_hash")?;
        non_empty(&entry.root_hash_algorithm, "transparency_entries.root_hash_algorithm")?;
        non_empty(&entry.root_hash_encoding, "transparency_entries.root_hash_encoding")?;
        non_empty(&entry.signed_checkpoint_ref, "transparency_entries.signed_checkpoint_ref")?;
        non_empty(&entry.inclusion_proof_ref, "transparency_entries.inclusion_proof_ref")?;
        if entry.tree_size == 0 || entry.entry_index >= entry.tree_size {
            return Err(CheckpointVerificationError::TreeSizeDoesNotContainEntry(
                entry.log_id.clone(),
            ));
        }
        if entry.previous_checkpoint_ref.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err(CheckpointVerificationError::InvalidShape(
                "previous_checkpoint_ref must be non-empty when present".to_owned(),
            ));
        }
        if entry.previous_checkpoint_ref.is_some()
            && entry.consistency_proof_ref.as_deref().is_none_or(|v| v.trim().is_empty())
        {
            return Err(CheckpointVerificationError::MissingConsistencyProof(
                entry.log_id.clone(),
            ));
        }
        non_empty(&entry.signed_statement_ref, "transparency_entries.signed_statement_ref")?;
        non_empty(&entry.checkpoint_key_id, "transparency_entries.checkpoint_key_id")?;
        non_empty(&entry.verification_policy_ref, "transparency_entries.verification_policy_ref")?;
        if !["hex", "base64", "base64url", "provider-defined"]
            .contains(&entry.root_hash_encoding.as_str())
        {
            return Err(CheckpointVerificationError::InvalidShape(
                "unsupported transparency root hash encoding".to_owned(),
            ));
        }
        let key = (
            entry.service_id.clone(),
            entry.log_id.clone(),
            entry.tree_size,
        );
        let root = (
            entry.root_hash.clone(),
            entry.root_hash_algorithm.clone(),
            entry.root_hash_encoding.clone(),
        );
        if let Some(prior) = roots.insert(key.clone(), root.clone()) {
            if prior != root {
                return Err(CheckpointVerificationError::ConflictingCheckpointRoot {
                    service_id: key.0,
                    log_id: key.1,
                    tree_size: key.2,
                });
            }
        }
    }

    valid_digest(&receipt.trust_policy.policy_digest, "trust_policy.policy_digest")?;
    non_empty(&receipt.trust_policy.policy_id, "trust_policy.policy_id")?;
    non_empty(&receipt.trust_policy.version, "trust_policy.version")?;
    non_empty(
        &receipt.trust_policy.key_status_evidence_ref,
        "trust_policy.key_status_evidence_ref",
    )?;
    non_empty(
        &receipt.trust_policy.revocation_snapshot_ref,
        "trust_policy.revocation_snapshot_ref",
    )?;
    non_empty(
        &receipt.trust_policy.governance_policy_ref,
        "trust_policy.governance_policy_ref",
    )?;
    if receipt.trust_policy.minimum_independent_witnesses == 0 {
        return Err(CheckpointVerificationError::InvalidShape(
            "minimum_independent_witnesses must be at least one".to_owned(),
        ));
    }

    let mut witness_ids = HashSet::new();
    for witness in &receipt.witness_attestations {
        non_empty(&witness.witness_id, "witness_attestations.witness_id")?;
        non_empty(&witness.service_id, "witness_attestations.service_id")?;
        non_empty(&witness.log_id, "witness_attestations.log_id")?;
        non_empty(&witness.root_hash, "witness_attestations.root_hash")?;
        non_empty(&witness.signature_ref, "witness_attestations.signature_ref")?;
        non_empty(&witness.key_id, "witness_attestations.key_id")?;
        valid_timestamp(&witness.observed_at, "witness_attestations.observed_at")?;
        valid_digest(&witness.trust_policy_digest, "witness_attestations.trust_policy_digest")?;
        if witness.trust_policy_digest != receipt.trust_policy.policy_digest {
            return Err(CheckpointVerificationError::WitnessPolicyDigestMismatch(
                witness.witness_id.clone(),
            ));
        }
        non_empty(
            &witness.independence_basis_ref,
            "witness_attestations.independence_basis_ref",
        )?;
        if !["hex", "base64", "base64url", "provider-defined"]
            .contains(&witness.root_hash_encoding.as_str())
        {
            return Err(CheckpointVerificationError::InvalidShape(
                "unsupported witness root hash encoding".to_owned(),
            ));
        }
        if !witness_ids.insert(witness.witness_id.clone()) {
            return Err(CheckpointVerificationError::DuplicateWitnessId(
                witness.witness_id.clone(),
            ));
        }
        let matching_entry = receipt.transparency_entries.iter().any(|entry| {
            entry.service_id == witness.service_id
                && entry.log_id == witness.log_id
                && entry.tree_size == witness.tree_size
                && entry.root_hash == witness.root_hash
                && entry.root_hash_algorithm == witness.root_hash_algorithm
                && entry.root_hash_encoding == witness.root_hash_encoding
        });
        if !matching_entry {
            return Err(CheckpointVerificationError::WitnessDoesNotMatchTransparencyEntry(
                witness.witness_id.clone(),
            ));
        }
    }

    let required = receipt.trust_policy.minimum_independent_witnesses;
    if witness_ids.len() < required as usize {
        return Err(CheckpointVerificationError::WitnessThresholdNotMet {
            required,
            found: witness_ids.len(),
        });
    }

    let references_present = receipt.transparency_entries.iter().all(|entry| {
        !entry.signed_checkpoint_ref.trim().is_empty()
            && !entry.inclusion_proof_ref.trim().is_empty()
            && !entry.signed_statement_ref.trim().is_empty()
            && !entry.verification_policy_ref.trim().is_empty()
            && entry.previous_checkpoint_ref.as_deref().is_none_or(|v| !v.trim().is_empty())
            && entry.consistency_proof_ref.as_deref().is_none_or(|v| !v.trim().is_empty())
    }) && receipt.witness_attestations.iter().all(|witness| {
        !witness.signature_ref.trim().is_empty()
            && !witness.key_id.trim().is_empty()
            && !witness.independence_basis_ref.trim().is_empty()
    }) && !receipt.trust_policy.key_status_evidence_ref.trim().is_empty()
        && !receipt.trust_policy.revocation_snapshot_ref.trim().is_empty()
        && !receipt.trust_policy.governance_policy_ref.trim().is_empty();

    // Read otherwise informational fields so future schema changes are visible.
    let _ = (&receipt.artifact_refs, &receipt.limitations);

    Ok(CheckpointReceiptInspection {
        receipt_id: receipt.receipt_id,
        run_id: receipt.run_id,
        manifest_digest: receipt.manifest_digest,
        head_sequence: receipt.head_sequence,
        head_event_digest: receipt.head_event_digest,
        statement_digest: receipt.statement_digest,
        transparency_entry_count: receipt.transparency_entries.len(),
        distinct_witness_id_count: witness_ids.len(),
        required_witness_count: required,
        declared_witness_count_meets_threshold: witness_ids.len() >= required as usize,
        statement_digest_verified: true,
        receipt_matches_trace_head: true,
        artifact_references_present: references_present,
        signatures_verified: false,
        inclusion_proofs_verified: false,
        consistency_proofs_verified: false,
        witness_independence_verified: false,
        checkpoint_authenticated: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TraceVerificationReport, verify_trace};

    const MANIFEST: &str = "sha256:0fa06d577488566420fea82078b2f30fd70280af9363e26f558bcac66bb32f2d";
    const HEAD: &str = "sha256:bb201acf3e03cd941e35e0db7d2e3a9e1a81f20378acce81f56412b142cc3fd0";
    const RUN: &str = "00000000-0000-4000-8000-000000000001";

    fn trace_report() -> TraceVerificationReport {
        let manifest = include_str!(
            "../../../../docs/research/examples/sat-trace-verifier-manifest-v0.1.json"
        );
        let events = include_str!(
            "../../../../docs/research/examples/sat-trace-verifier-trace-v0.1.jsonl"
        );
        let manifest_digest = crate::compute_manifest_digest(manifest).unwrap();
        verify_trace(events, RUN, &manifest_digest).unwrap()
    }

    fn example_receipt() -> &'static str {
        include_str!(
            "../../../../docs/research/examples/sat-trace-checkpoint-example-v0.1.json"
        )
    }

    fn parse_receipt() -> serde_json::Value {
        serde_json::from_str(example_receipt()).unwrap()
    }

    #[test]
    fn synthetic_receipt_matches_trace_but_is_not_authenticated() {
        let report = inspect_checkpoint_receipt(example_receipt(), &trace_report()).unwrap();
        assert_eq!(report.run_id, RUN);
        assert_eq!(report.manifest_digest, MANIFEST);
        assert_eq!(report.head_event_digest, HEAD);
        assert_eq!(report.head_sequence, 2);
        assert_eq!(report.distinct_witness_id_count, 2);
        assert_eq!(report.required_witness_count, 2);
        assert!(report.declared_witness_count_meets_threshold);
        assert!(report.statement_digest_verified);
        assert!(report.receipt_matches_trace_head);
        assert!(report.artifact_references_present);
        assert!(!report.signatures_verified);
        assert!(!report.inclusion_proofs_verified);
        assert!(!report.consistency_proofs_verified);
        assert!(!report.witness_independence_verified);
        assert!(!report.checkpoint_authenticated);
    }

    #[test]
    fn run_manifest_and_head_mismatch_are_rejected() {
        let trace = trace_report();
        let mut receipt = parse_receipt();
        receipt["run_id"] = serde_json::json!("00000000-0000-4000-8000-000000000002");
        let error = inspect_checkpoint_receipt(&receipt.to_string(), &trace).unwrap_err();
        assert_eq!(error, CheckpointVerificationError::RunIdMismatch);

        let mut receipt = parse_receipt();
        receipt["head_event_digest"] = serde_json::json!("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let error = inspect_checkpoint_receipt(&receipt.to_string(), &trace).unwrap_err();
        assert_eq!(error, CheckpointVerificationError::HeadEventDigestMismatch);
    }

    #[test]
    fn trace_head_statement_digest_is_recomputed() {
        let mut receipt = parse_receipt();
        receipt["trace_head_statement"]["checkpointed_at"] =
            serde_json::json!("2026-11-25T13:01:00Z");
        let error = inspect_checkpoint_receipt(&receipt.to_string(), &trace_report()).unwrap_err();
        assert_eq!(error, CheckpointVerificationError::StatementDigestMismatch);
    }

    #[test]
    fn duplicate_witness_identity_is_rejected() {
        let mut receipt = parse_receipt();
        receipt["witness_attestations"][1]["witness_id"] =
            receipt["witness_attestations"][0]["witness_id"].clone();
        let error = inspect_checkpoint_receipt(&receipt.to_string(), &trace_report()).unwrap_err();
        assert!(matches!(error, CheckpointVerificationError::DuplicateWitnessId(_)));
    }

    #[test]
    fn witness_root_must_match_transparency_entry() {
        let mut receipt = parse_receipt();
        receipt["witness_attestations"][0]["root_hash"] =
            serde_json::json!("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
        let error = inspect_checkpoint_receipt(&receipt.to_string(), &trace_report()).unwrap_err();
        assert!(matches!(error, CheckpointVerificationError::WitnessDoesNotMatchTransparencyEntry(_)));
    }

    #[test]
    fn duplicate_root_conflict_for_same_log_size_is_rejected() {
        let mut receipt = parse_receipt();
        let mut entry = receipt["transparency_entries"][0].clone();
        entry["root_hash"] =
            serde_json::json!("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
        receipt["transparency_entries"].as_array_mut().unwrap().push(entry);
        let error = inspect_checkpoint_receipt(&receipt.to_string(), &trace_report()).unwrap_err();
        assert!(matches!(error, CheckpointVerificationError::ConflictingCheckpointRoot { .. }));
    }

    #[test]
    fn missing_consistency_proof_after_prior_checkpoint_is_rejected() {
        let mut receipt = parse_receipt();
        receipt["transparency_entries"][0]["consistency_proof_ref"] = serde_json::Value::Null;
        let error = inspect_checkpoint_receipt(&receipt.to_string(), &trace_report()).unwrap_err();
        assert!(matches!(error, CheckpointVerificationError::MissingConsistencyProof(_)));
    }

    #[test]
    fn insufficient_declared_witness_count_is_rejected() {
        let mut receipt = parse_receipt();
        receipt["trust_policy"]["minimum_independent_witnesses"] = serde_json::json!(3);
        let error = inspect_checkpoint_receipt(&receipt.to_string(), &trace_report()).unwrap_err();
        assert!(matches!(error, CheckpointVerificationError::WitnessThresholdNotMet { .. }));
    }

    #[test]
    fn duplicate_json_keys_are_rejected() {
        let duplicate = r#"{"receipt_id":"one","receipt_id":"two"}"#;
        let error = inspect_checkpoint_receipt(duplicate, &trace_report()).unwrap_err();
        assert!(matches!(error, CheckpointVerificationError::DuplicateJsonKey(_)));
    }
}
