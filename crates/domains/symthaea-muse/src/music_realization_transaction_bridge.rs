// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binds capability-qualified realization attempts to exact EVAL-040 candidates.
//!
//! This is an adapter boundary, not a provider SDK or a musical-quality oracle.
//! Provider receipts and control observations are untrusted claims until bound
//! to the transaction, actual output bytes, and independently replayed EVAL-040
//! evidence. A backend cannot certify its own musical correctness.

use crate::music_edit_transaction_decision::{
    MusicEditAttemptOutcomeV1, MusicEditCandidateAttemptV1, MusicEditEvidenceKindV1,
    MusicEditEvidenceRecordV1, MusicEditEvidenceStateV1, MusicEditTransactionV1,
    validate_music_edit_transaction_v1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const MUSIC_REALIZATION_BRIDGE_SCHEMA_V1: &str =
    "mel-music-realization-transaction-bridge-v1";
const RECEIPT_DOMAIN: &[u8] = b"luminous.mel.music-realization-transaction-bridge.v1";
const BINDING_REPORT_DOMAIN: &[u8] = b"luminous.mel.music-realization-verified-binding-report.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicRealizationOutputModalityV1 {
    SymbolicScore,
    Midi,
    Audio,
    Stems,
    MusicXml,
    Project,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicRealizerControlRequirementV1 {
    Hard,
    Advisory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicRealizerDeclaredFidelityV1 {
    Exact,
    Bounded,
    Advisory,
    BestEffort,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicRealizerObservedDispositionV1 {
    Applied,
    BestEffort,
    Ignored,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRealizerControlBindingV1 {
    pub control_id: String,
    /// Commitment to this exact requested control in the frozen realization request.
    pub request_control_sha256: String,
    pub requirement: MusicRealizerControlRequirementV1,
    pub declared_fidelity: MusicRealizerDeclaredFidelityV1,
    /// Required for Bounded controls; commits the explicit permitted bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fidelity_bound_sha256: Option<String>,
    /// Provider/adapter-reported observation, not independent proof of musical success.
    pub observed_disposition: MusicRealizerObservedDispositionV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicRealizerExecutionEventStateV1 {
    Submitted,
    Completed,
    Failed,
    OutcomeUnknown,
    ReconciledNotExecuted,
    ReconciledCompleted,
    ReconciledFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRealizerExecutionEventV1 {
    /// One-based contiguous event position.
    pub sequence: u64,
    pub state: MusicRealizerExecutionEventStateV1,
    /// Provider request ID or adapter-generated immutable submission ID.
    /// The verifier requires this correlation identity on every event so a
    /// completion/reconciliation cannot be attached to a different request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Reconciliation events must name the earlier OutcomeUnknown sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconciles_sequence: Option<u64>,
    /// Digest of an independently checkable provider status/idempotency record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconciliation_receipt_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRealizationBridgeReceiptPayloadV1 {
    pub schema_version: String,
    pub transaction_id: String,
    pub attempt_id: String,
    pub realization_request_id: String,
    pub realization_request_sha256: String,
    pub capability_manifest_sha256: String,
    /// Digest of the upstream MEL-REAL-001 realization receipt payload.
    pub upstream_realization_receipt_sha256: String,
    pub backend_id: String,
    pub backend_version: String,
    pub adapter_id: String,
    pub adapter_version: String,
    pub source_lineage_sha256: String,
    pub candidate_id: String,
    pub output_modality: MusicRealizationOutputModalityV1,
    /// Zero and None are required when no completed output is bound.
    pub output_byte_length: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_artifact_sha256: Option<String>,
    /// Required when an OutcomeUnknown event is followed by a retry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key_sha256: Option<String>,
    /// Ordered execution history; the final state is derived from the last event.
    pub execution_events: Vec<MusicRealizerExecutionEventV1>,
    /// Canonical order by control_id.
    pub controls: Vec<MusicRealizerControlBindingV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicRealizationBridgeReceiptV1 {
    pub payload: MusicRealizationBridgeReceiptPayloadV1,
    pub receipt_sha256: String,
}

/// One control from the independently verified frozen request and capability
/// manifest. The provider receipt must contain this exact control; omission,
/// addition, or substitution is rejected before a candidate can be verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MusicRealizationExpectedControlV1 {
    pub control_id: String,
    pub request_control_sha256: String,
    pub requirement: MusicRealizerControlRequirementV1,
    pub declared_fidelity: MusicRealizerDeclaredFidelityV1,
    pub fidelity_bound_sha256: Option<String>,
}

/// Trusted identity values obtained from the upstream realization verifier and
/// the caller's frozen request/manifest. A self-sealed receipt cannot choose
/// these expected values for itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MusicRealizationExpectedBindingV1 {
    pub transaction_id: String,
    pub attempt_id: String,
    pub candidate_id: String,
    pub source_lineage_sha256: String,
    pub realization_request_id: String,
    pub realization_request_sha256: String,
    pub capability_manifest_sha256: String,
    pub upstream_realization_receipt_sha256: String,
    pub backend_id: String,
    pub backend_version: String,
    pub adapter_id: String,
    pub adapter_version: String,
    pub output_modality: MusicRealizationOutputModalityV1,
    /// Complete canonical control set from the frozen request/manifest.
    /// This is not reconstructed from provider-reported receipt contents.
    pub controls: Vec<MusicRealizationExpectedControlV1>,
}

/// Opaque proof token returned only by the verifier below. It intentionally
/// has private fields and no Deserialize implementation: downstream code
/// cannot construct, mutate, or deserialize a self-asserted "verified" value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MusicRealizationVerifiedBindingV1 {
    transaction_id: String,
    attempt_id: String,
    candidate_id: String,
    source_lineage_sha256: String,
    realization_request_id: String,
    request_sha256: String,
    capability_manifest_sha256: String,
    upstream_realization_receipt_sha256: String,
    realization_receipt_sha256: String,
    backend_id: String,
    backend_version: String,
    adapter_id: String,
    adapter_version: String,
    output_modality: MusicRealizationOutputModalityV1,
    final_execution_state: MusicRealizerExecutionEventStateV1,
    submission_count: u64,
    observed_output_sha256: Option<String>,
    observed_output_byte_length: u64,
    report_sha256: String,
}

impl MusicRealizationVerifiedBindingV1 {
    pub fn transaction_id(&self) -> &str { &self.transaction_id }
    pub fn attempt_id(&self) -> &str { &self.attempt_id }
    pub fn candidate_id(&self) -> &str { &self.candidate_id }
    pub fn source_lineage_sha256(&self) -> &str { &self.source_lineage_sha256 }
    pub fn realization_request_id(&self) -> &str { &self.realization_request_id }
    pub fn request_sha256(&self) -> &str { &self.request_sha256 }
    pub fn capability_manifest_sha256(&self) -> &str { &self.capability_manifest_sha256 }
    pub fn upstream_realization_receipt_sha256(&self) -> &str { &self.upstream_realization_receipt_sha256 }
    pub fn realization_receipt_sha256(&self) -> &str { &self.realization_receipt_sha256 }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn backend_version(&self) -> &str { &self.backend_version }
    pub fn adapter_id(&self) -> &str { &self.adapter_id }
    pub fn adapter_version(&self) -> &str { &self.adapter_version }
    pub fn output_modality(&self) -> MusicRealizationOutputModalityV1 { self.output_modality }
    pub fn final_execution_state(&self) -> MusicRealizerExecutionEventStateV1 { self.final_execution_state }
    pub fn submission_count(&self) -> u64 { self.submission_count }
    pub fn observed_output_sha256(&self) -> Option<&str> {
        self.observed_output_sha256.as_deref()
    }
    pub fn observed_output_byte_length(&self) -> u64 { self.observed_output_byte_length }
    pub fn report_sha256(&self) -> &str { &self.report_sha256 }
}

impl MusicRealizationVerifiedBindingV1 {
    /// Convert a completed, independently byte-verified binding into the
    /// EVAL-040 evidence record. The caller supplies the qualified verifier
    /// build digest and must include this record in the final transaction.
    pub fn to_evidence_record(
        &self,
        replay_verifier_sha256: &str,
    ) -> Result<MusicEditEvidenceRecordV1, MusicRealizationBridgeIssueV1> {
        if !is_sha256(replay_verifier_sha256) {
            return Err(MusicRealizationBridgeIssueV1::InvalidVerifierDigest);
        }
        if !matches!(
            self.final_execution_state,
            MusicRealizerExecutionEventStateV1::Completed
                | MusicRealizerExecutionEventStateV1::ReconciledCompleted
        ) {
            return Err(MusicRealizationBridgeIssueV1::UnresolvedExecutionCannotBeVerified);
        }
        let Some(candidate_artifact_sha256) = self.observed_output_sha256.as_ref() else {
            return Err(MusicRealizationBridgeIssueV1::UnresolvedExecutionCannotBeVerified);
        };
        Ok(MusicEditEvidenceRecordV1 {
            kind: MusicEditEvidenceKindV1::RealizationExecution,
            state: MusicEditEvidenceStateV1::Verified,
            source_lineage_sha256: Some(self.source_lineage_sha256.clone()),
            candidate_artifact_sha256: Some(candidate_artifact_sha256.clone()),
            receipt_sha256: Some(self.realization_receipt_sha256.clone()),
            report_sha256: Some(self.report_sha256.clone()),
            replay_verifier_sha256: Some(replay_verifier_sha256.to_owned()),
            failure_class: None,
            reason_code: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct MusicRealizationVerifiedBindingReportV1 {
    transaction_id: String,
    attempt_id: String,
    candidate_id: String,
    source_lineage_sha256: String,
    realization_request_id: String,
    realization_request_sha256: String,
    capability_manifest_sha256: String,
    upstream_realization_receipt_sha256: String,
    realization_receipt_sha256: String,
    backend_id: String,
    backend_version: String,
    adapter_id: String,
    adapter_version: String,
    output_modality: MusicRealizationOutputModalityV1,
    final_execution_state: MusicRealizerExecutionEventStateV1,
    submission_count: u64,
    observed_output_sha256: Option<String>,
    observed_output_byte_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicRealizationBridgeIssueV1 {
    WrongSchema,
    EmptyField(&'static str),
    InvalidDigest(&'static str),
    ReceiptDigestMismatch,
    TransactionInvalid,
    TransactionMismatch,
    AttemptNotFound,
    CandidateMismatch,
    SourceLineageMismatch,
    InvalidExecutionTrace,
    NonCanonicalExecutionOrder,
    InvalidExecutionTransition,
    InvalidReconciliationBinding,
    ProviderRequestIdentityMismatch,
    ExpectedBindingMismatch(&'static str),
    RealizationEvidenceNotRequiredByPolicy,
    InvalidVerifierDigest,
    UnresolvedExecutionCannotBeVerified,
    MissingIdempotencyKeyForRetry,
    SubmissionBudgetUnderCounted,
    ExecutionOutcomeMismatch,
    CompletedOutputMissing,
    UnexpectedOutputBytes,
    OutputLengthMismatch,
    OutputDigestMismatch,
    MissingByteMaterializationEvidence,
    ByteEvidenceMismatch,
    NonCanonicalControlOrder,
    ExpectedControlSetMismatch,
    ExpectedControlBindingMismatch(String),
    DuplicateControl(String),
    HardControlNotHonored(String),
    InvalidControl(String),
    SerializationError,
}

impl std::fmt::Display for MusicRealizationBridgeIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MusicRealizationBridgeIssueV1 {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn domain_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, MusicRealizationBridgeIssueV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| MusicRealizationBridgeIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(b"\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn receipt_digest(
    payload: &MusicRealizationBridgeReceiptPayloadV1,
) -> Result<String, MusicRealizationBridgeIssueV1> {
    let bytes = serde_json::to_vec(payload)
        .map_err(|_| MusicRealizationBridgeIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(b"\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Seals the bridge payload. This is a content digest, not a provider signature
/// or proof of provider honesty.
pub fn seal_music_realization_bridge_receipt_v1(
    payload: MusicRealizationBridgeReceiptPayloadV1,
) -> Result<MusicRealizationBridgeReceiptV1, MusicRealizationBridgeIssueV1> {
    let receipt_sha256 = receipt_digest(&payload)?;
    Ok(MusicRealizationBridgeReceiptV1 {
        payload,
        receipt_sha256,
    })
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn final_state(
    events: &[MusicRealizerExecutionEventV1],
) -> Option<MusicRealizerExecutionEventStateV1> {
    events.last().map(|event| event.state)
}

fn validate_execution_events(
    payload: &MusicRealizationBridgeReceiptPayloadV1,
) -> Result<u64, Vec<MusicRealizationBridgeIssueV1>> {
    use MusicRealizerExecutionEventStateV1 as S;

    if payload.execution_events.is_empty() {
        return Err(vec![MusicRealizationBridgeIssueV1::InvalidExecutionTrace]);
    }

    let mut issues = Vec::new();
    let mut unknown_sequences = BTreeSet::new();
    let mut submissions = 0_u64;

    for (index, event) in payload.execution_events.iter().enumerate() {
        let expected = u64::try_from(index + 1).unwrap_or(u64::MAX);
        if event.sequence != expected {
            issues.push(MusicRealizationBridgeIssueV1::NonCanonicalExecutionOrder);
        }
        if !event.provider_request_id.as_deref().is_some_and(nonempty) {
            issues.push(MusicRealizationBridgeIssueV1::ProviderRequestIdentityMismatch);
        }

        match event.state {
            S::Submitted => {
                submissions = match submissions.checked_add(1) {
                    Some(value) => value,
                    None => {
                        issues.push(MusicRealizationBridgeIssueV1::InvalidExecutionTrace);
                        u64::MAX
                    }
                };
                if event.reconciles_sequence.is_some()
                    || event.reconciliation_receipt_sha256.is_some()
                    || event.reason_code.is_some()
                {
                    issues.push(MusicRealizationBridgeIssueV1::InvalidExecutionTrace);
                }
            }
            S::Completed => {
                let matches_submission = index > 0
                    && payload.execution_events[index - 1].state == S::Submitted
                    && payload.execution_events[index - 1].provider_request_id
                        == event.provider_request_id;
                if !matches_submission {
                    issues.push(MusicRealizationBridgeIssueV1::ProviderRequestIdentityMismatch);
                }
                if event.reconciles_sequence.is_some()
                    || event.reconciliation_receipt_sha256.is_some()
                    || event.reason_code.is_some()
                {
                    issues.push(MusicRealizationBridgeIssueV1::InvalidExecutionTrace);
                }
            }
            S::Failed | S::OutcomeUnknown => {
                let matches_submission = index > 0
                    && payload.execution_events[index - 1].state == S::Submitted
                    && payload.execution_events[index - 1].provider_request_id
                        == event.provider_request_id;
                if !matches_submission {
                    issues.push(MusicRealizationBridgeIssueV1::ProviderRequestIdentityMismatch);
                }
                if !event.reason_code.as_deref().is_some_and(nonempty)
                    || event.reconciles_sequence.is_some()
                    || event.reconciliation_receipt_sha256.is_some()
                {
                    issues.push(MusicRealizationBridgeIssueV1::InvalidExecutionTrace);
                }
                if event.state == S::OutcomeUnknown {
                    unknown_sequences.insert(event.sequence);
                }
            }
            S::ReconciledNotExecuted | S::ReconciledCompleted | S::ReconciledFailed => {
                let reference = event.reconciles_sequence;
                let previous_unknown = index > 0
                    && payload.execution_events[index - 1].state == S::OutcomeUnknown
                    && reference == Some(payload.execution_events[index - 1].sequence);
                let receipt_valid = event
                    .reconciliation_receipt_sha256
                    .as_deref()
                    .is_some_and(is_sha256);
                let reason_valid = match event.state {
                    S::ReconciledNotExecuted | S::ReconciledFailed => {
                        event.reason_code.as_deref().is_some_and(nonempty)
                    }
                    S::ReconciledCompleted => event
                        .reason_code
                        .as_deref()
                        .is_none_or(|reason| reason.trim().is_empty()),
                    _ => false,
                };
                let request_identity_matches = index > 0
                    && payload.execution_events[index - 1].provider_request_id
                        == event.provider_request_id;
                if !previous_unknown
                    || !request_identity_matches
                    || !reference.is_some_and(|sequence| unknown_sequences.contains(&sequence))
                    || !receipt_valid
                    || !reason_valid
                {
                    issues.push(MusicRealizationBridgeIssueV1::InvalidReconciliationBinding);
                }
            }
        }

        if index > 0 {
            let previous = payload.execution_events[index - 1].state;
            let allowed = matches!(
                (previous, event.state),
                (S::Submitted, S::Completed | S::Failed | S::OutcomeUnknown)
                    | (
                        S::OutcomeUnknown,
                        S::ReconciledNotExecuted
                            | S::ReconciledCompleted
                            | S::ReconciledFailed
                    )
                    | (S::ReconciledNotExecuted, S::Submitted)
            );
            if !allowed {
                issues.push(MusicRealizationBridgeIssueV1::InvalidExecutionTransition);
            }
        } else if event.state != S::Submitted {
            issues.push(MusicRealizationBridgeIssueV1::InvalidExecutionTransition);
        }
    }

    if payload.execution_events.iter().any(|event| {
        event.state == S::ReconciledNotExecuted
    }) && !payload.idempotency_key_sha256.as_deref().is_some_and(is_sha256)
    {
        issues.push(MusicRealizationBridgeIssueV1::MissingIdempotencyKeyForRetry);
    }

    if let Some(key) = payload.idempotency_key_sha256.as_deref()
        && !is_sha256(key)
    {
        issues.push(MusicRealizationBridgeIssueV1::InvalidDigest(
            "idempotency_key_sha256",
        ));
    }

    if issues.is_empty() {
        Ok(submissions)
    } else {
        Err(issues)
    }
}

fn validate_controls(
    expected_controls: &[MusicRealizationExpectedControlV1],
    controls: &[MusicRealizerControlBindingV1],
) -> Vec<MusicRealizationBridgeIssueV1> {
    use MusicRealizerControlRequirementV1::Hard;
    use MusicRealizerDeclaredFidelityV1 as F;
    use MusicRealizerObservedDispositionV1 as O;

    let mut issues = Vec::new();
    if !expected_controls
        .windows(2)
        .all(|pair| pair[0].control_id.as_str() < pair[1].control_id.as_str())
        || !controls
            .windows(2)
            .all(|pair| pair[0].control_id.as_str() < pair[1].control_id.as_str())
    {
        issues.push(MusicRealizationBridgeIssueV1::NonCanonicalControlOrder);
    }

    if expected_controls.len() != controls.len() {
        issues.push(MusicRealizationBridgeIssueV1::ExpectedControlSetMismatch);
    }
    for (expected, observed) in expected_controls.iter().zip(controls) {
        if expected.control_id != observed.control_id
            || expected.request_control_sha256 != observed.request_control_sha256
            || expected.requirement != observed.requirement
            || expected.declared_fidelity != observed.declared_fidelity
            || expected.fidelity_bound_sha256 != observed.fidelity_bound_sha256
        {
            issues.push(MusicRealizationBridgeIssueV1::ExpectedControlBindingMismatch(
                expected.control_id.clone(),
            ));
        }
    }

    let mut expected_seen = BTreeSet::new();
    for control in expected_controls {
        if !nonempty(&control.control_id)
            || !expected_seen.insert(control.control_id.clone())
            || !is_sha256(&control.request_control_sha256)
            || (control.declared_fidelity == F::Bounded
                && !control.fidelity_bound_sha256.as_deref().is_some_and(is_sha256))
            || (control.declared_fidelity != F::Bounded
                && control.fidelity_bound_sha256.is_some())
            || (control.requirement == Hard
                && !matches!(control.declared_fidelity, F::Exact | F::Bounded))
        {
            issues.push(MusicRealizationBridgeIssueV1::InvalidControl(
                control.control_id.clone(),
            ));
        }
    }

    let mut seen = BTreeSet::new();
    for control in controls {
        if !nonempty(&control.control_id) {
            issues.push(MusicRealizationBridgeIssueV1::InvalidControl(
                control.control_id.clone(),
            ));
            continue;
        }
        if !seen.insert(control.control_id.clone()) {
            issues.push(MusicRealizationBridgeIssueV1::DuplicateControl(
                control.control_id.clone(),
            ));
        }
        if !is_sha256(&control.request_control_sha256) {
            issues.push(MusicRealizationBridgeIssueV1::InvalidControl(
                control.control_id.clone(),
            ));
        }
        if control.declared_fidelity == F::Bounded
            && !control.fidelity_bound_sha256.as_deref().is_some_and(is_sha256)
        {
            issues.push(MusicRealizationBridgeIssueV1::InvalidControl(
                control.control_id.clone(),
            ));
        }
        if control.declared_fidelity != F::Bounded
            && control.fidelity_bound_sha256.is_some()
        {
            issues.push(MusicRealizationBridgeIssueV1::InvalidControl(
                control.control_id.clone(),
            ));
        }
        if control.requirement == Hard
            && (!matches!(control.declared_fidelity, F::Exact | F::Bounded)
                || control.observed_disposition != O::Applied)
        {
            issues.push(MusicRealizationBridgeIssueV1::HardControlNotHonored(
                control.control_id.clone(),
            ));
        }
        // Advisory controls may be best-effort, ignored, or unavailable. This
        // observation is retained for audit; it never upgrades musical evidence.
    }
    issues
}

fn validate_payload_shape(
    payload: &MusicRealizationBridgeReceiptPayloadV1,
    expected_controls: &[MusicRealizationExpectedControlV1],
) -> Vec<MusicRealizationBridgeIssueV1> {
    let mut issues = Vec::new();
    if payload.schema_version != MUSIC_REALIZATION_BRIDGE_SCHEMA_V1 {
        issues.push(MusicRealizationBridgeIssueV1::WrongSchema);
    }
    for (field, value) in [
        ("transaction_id", payload.transaction_id.as_str()),
        ("attempt_id", payload.attempt_id.as_str()),
        ("realization_request_id", payload.realization_request_id.as_str()),
        ("backend_id", payload.backend_id.as_str()),
        ("backend_version", payload.backend_version.as_str()),
        ("adapter_id", payload.adapter_id.as_str()),
        ("adapter_version", payload.adapter_version.as_str()),
        ("candidate_id", payload.candidate_id.as_str()),
    ] {
        if !nonempty(value) {
            issues.push(MusicRealizationBridgeIssueV1::EmptyField(field));
        }
    }
    for (field, value) in [
        ("realization_request_sha256", payload.realization_request_sha256.as_str()),
        ("capability_manifest_sha256", payload.capability_manifest_sha256.as_str()),
        (
            "upstream_realization_receipt_sha256",
            payload.upstream_realization_receipt_sha256.as_str(),
        ),
        ("source_lineage_sha256", payload.source_lineage_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            issues.push(MusicRealizationBridgeIssueV1::InvalidDigest(field));
        }
    }
    issues.extend(validate_controls(expected_controls, &payload.controls));
    issues
}

fn find_attempt<'a>(
    transaction: &'a MusicEditTransactionV1,
    attempt_id: &str,
) -> Option<&'a MusicEditCandidateAttemptV1> {
    transaction
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == attempt_id)
}

/// Verifies the bridge receipt against one exact EVAL-040 attempt and, when
/// completed, the actual output bytes. The provider's hash is never trusted
/// without recomputing the digest over those bytes.
pub fn verify_music_realization_transaction_binding_v1(
    transaction: &MusicEditTransactionV1,
    expected: &MusicRealizationExpectedBindingV1,
    receipt: &MusicRealizationBridgeReceiptV1,
    actual_output_bytes: Option<&[u8]>,
) -> Result<MusicRealizationVerifiedBindingV1, Vec<MusicRealizationBridgeIssueV1>> {
    use MusicEditAttemptOutcomeV1 as A;
    use MusicEditEvidenceKindV1 as K;
    use MusicEditEvidenceStateV1 as E;
    use MusicRealizerExecutionEventStateV1 as S;

    let mut issues = validate_payload_shape(&receipt.payload, &expected.controls);

    // These expected identities must come from the upstream realization
    // request/manifest verifier and the caller's frozen execution context.
    for (field, value) in [
        ("expected_transaction_id", expected.transaction_id.as_str()),
        ("expected_attempt_id", expected.attempt_id.as_str()),
        ("expected_candidate_id", expected.candidate_id.as_str()),
        ("expected_source_lineage", expected.source_lineage_sha256.as_str()),
        ("expected_realization_request_id", expected.realization_request_id.as_str()),
        ("expected_backend_id", expected.backend_id.as_str()),
        ("expected_backend_version", expected.backend_version.as_str()),
        ("expected_adapter_id", expected.adapter_id.as_str()),
        ("expected_adapter_version", expected.adapter_version.as_str()),
    ] {
        if !nonempty(value) {
            issues.push(MusicRealizationBridgeIssueV1::EmptyField(field));
        }
    }
    for (field, value) in [
        ("expected_request_sha256", expected.realization_request_sha256.as_str()),
        ("expected_capability_manifest_sha256", expected.capability_manifest_sha256.as_str()),
        (
            "expected_upstream_receipt_sha256",
            expected.upstream_realization_receipt_sha256.as_str(),
        ),
        ("expected_source_lineage_sha256", expected.source_lineage_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            issues.push(MusicRealizationBridgeIssueV1::InvalidDigest(field));
        }
    }

    let payload = &receipt.payload;
    for (field, matches) in [
        ("transaction_id", expected.transaction_id == payload.transaction_id),
        ("attempt_id", expected.attempt_id == payload.attempt_id),
        ("candidate_id", expected.candidate_id == payload.candidate_id),
        (
            "source_lineage_sha256",
            expected.source_lineage_sha256 == payload.source_lineage_sha256,
        ),
        (
            "realization_request_id",
            expected.realization_request_id == payload.realization_request_id,
        ),
        (
            "realization_request_sha256",
            expected.realization_request_sha256 == payload.realization_request_sha256,
        ),
        (
            "capability_manifest_sha256",
            expected.capability_manifest_sha256 == payload.capability_manifest_sha256,
        ),
        (
            "upstream_realization_receipt_sha256",
            expected.upstream_realization_receipt_sha256
                == payload.upstream_realization_receipt_sha256,
        ),
        ("backend_id", expected.backend_id == payload.backend_id),
        ("backend_version", expected.backend_version == payload.backend_version),
        ("adapter_id", expected.adapter_id == payload.adapter_id),
        ("adapter_version", expected.adapter_version == payload.adapter_version),
        ("output_modality", expected.output_modality == payload.output_modality),
    ] {
        if !matches {
            issues.push(MusicRealizationBridgeIssueV1::ExpectedBindingMismatch(field));
        }
    }

    if !is_sha256(&receipt.receipt_sha256) {
        issues.push(MusicRealizationBridgeIssueV1::InvalidDigest("receipt_sha256"));
    } else {
        match receipt_digest(&receipt.payload) {
            Ok(expected) if expected != receipt.receipt_sha256 => {
                issues.push(MusicRealizationBridgeIssueV1::ReceiptDigestMismatch);
            }
            Err(_) => issues.push(MusicRealizationBridgeIssueV1::SerializationError),
            _ => {}
        }
    }

    if !validate_music_edit_transaction_v1(transaction).is_empty() {
        issues.push(MusicRealizationBridgeIssueV1::TransactionInvalid);
    }
    if !transaction
        .policy
        .required_evidence
        .contains(&K::RealizationExecution)
    {
        issues.push(MusicRealizationBridgeIssueV1::RealizationEvidenceNotRequiredByPolicy);
    }
    if receipt.payload.transaction_id != transaction.transaction_id {
        issues.push(MusicRealizationBridgeIssueV1::TransactionMismatch);
    }
    if receipt.payload.source_lineage_sha256 != transaction.source_lineage_sha256 {
        issues.push(MusicRealizationBridgeIssueV1::SourceLineageMismatch);
    }

    let Some(attempt) = find_attempt(transaction, &receipt.payload.attempt_id) else {
        issues.push(MusicRealizationBridgeIssueV1::AttemptNotFound);
        return Err(issues);
    };
    if attempt.candidate_id != receipt.payload.candidate_id {
        issues.push(MusicRealizationBridgeIssueV1::CandidateMismatch);
    }
    if attempt.source_lineage_sha256 != receipt.payload.source_lineage_sha256 {
        issues.push(MusicRealizationBridgeIssueV1::SourceLineageMismatch);
    }

    // Count every submitted provider call independently of trace validity.
    // A malformed later transition must not erase earlier/current submissions
    // from the budget denominator.
    let observed_submission_count = u64::try_from(
        receipt
            .payload
            .execution_events
            .iter()
            .filter(|event| event.state == S::Submitted)
            .count(),
    )
    .unwrap_or(u64::MAX);
    let submissions = match validate_execution_events(&receipt.payload) {
        Ok(count) => count,
        Err(trace_issues) => {
            issues.extend(trace_issues);
            observed_submission_count
        }
    };
    if attempt.render_calls < observed_submission_count {
        issues.push(MusicRealizationBridgeIssueV1::SubmissionBudgetUnderCounted);
    }

    let state = final_state(&receipt.payload.execution_events);
    let completed = matches!(state, Some(S::Completed | S::ReconciledCompleted));
    let unresolved = matches!(state, Some(S::Submitted | S::OutcomeUnknown));
    let failed = matches!(
        state,
        Some(S::Failed | S::ReconciledFailed | S::ReconciledNotExecuted)
    );

    if completed {
        let Some(bytes) = actual_output_bytes else {
            issues.push(MusicRealizationBridgeIssueV1::CompletedOutputMissing);
            return Err(issues);
        };
        let observed_sha256 = sha256_bytes(bytes);
        let observed_length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if observed_length != receipt.payload.output_byte_length {
            issues.push(MusicRealizationBridgeIssueV1::OutputLengthMismatch);
        }
        if attempt.materialized_bytes < observed_length {
            issues.push(MusicRealizationBridgeIssueV1::ByteEvidenceMismatch);
        }
        if receipt.payload.output_artifact_sha256.as_deref() != Some(observed_sha256.as_str())
            || attempt.candidate_artifact_sha256.as_deref() != Some(observed_sha256.as_str())
        {
            issues.push(MusicRealizationBridgeIssueV1::OutputDigestMismatch);
        }
        if attempt.outcome != A::CandidateReady {
            issues.push(MusicRealizationBridgeIssueV1::ExecutionOutcomeMismatch);
        }
        let materialization = attempt
            .evidence
            .iter()
            .find(|record| record.kind == K::ActualByteMaterialization);
        match materialization {
            None => issues.push(MusicRealizationBridgeIssueV1::MissingByteMaterializationEvidence),
            Some(record)
                if record.state != E::Verified
                    || record.source_lineage_sha256.as_deref()
                        != Some(transaction.source_lineage_sha256.as_str())
                    || record.candidate_artifact_sha256.as_deref()
                        != Some(observed_sha256.as_str()) =>
            {
                issues.push(MusicRealizationBridgeIssueV1::ByteEvidenceMismatch);
            }
            Some(_) => {}
        }
    } else {
        if actual_output_bytes.is_some()
            || receipt.payload.output_artifact_sha256.is_some()
            || receipt.payload.output_byte_length != 0
            || attempt.candidate_artifact_sha256.is_some()
        {
            issues.push(MusicRealizationBridgeIssueV1::UnexpectedOutputBytes);
        }
        if unresolved {
            if attempt.outcome != A::OutcomeUnknown {
                issues.push(MusicRealizationBridgeIssueV1::ExecutionOutcomeMismatch);
            }
        } else if failed && !matches!(
            attempt.outcome,
            A::Failed | A::RepairableFailure | A::HardInvalid
        ) {
            issues.push(MusicRealizationBridgeIssueV1::ExecutionOutcomeMismatch);
        } else if !unresolved && !failed {
            issues.push(MusicRealizationBridgeIssueV1::InvalidExecutionTrace);
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let observed_output_sha256 = if completed {
        actual_output_bytes.map(sha256_bytes)
    } else {
        None
    };
    let final_execution_state = state.expect("execution trace was checked as non-empty");
    let report_payload = MusicRealizationVerifiedBindingReportV1 {
        transaction_id: transaction.transaction_id.clone(),
        attempt_id: attempt.attempt_id.clone(),
        candidate_id: attempt.candidate_id.clone(),
        source_lineage_sha256: transaction.source_lineage_sha256.clone(),
        realization_request_id: receipt.payload.realization_request_id.clone(),
        realization_request_sha256: receipt.payload.realization_request_sha256.clone(),
        capability_manifest_sha256: receipt.payload.capability_manifest_sha256.clone(),
        upstream_realization_receipt_sha256: receipt
            .payload
            .upstream_realization_receipt_sha256
            .clone(),
        realization_receipt_sha256: receipt.receipt_sha256.clone(),
        backend_id: receipt.payload.backend_id.clone(),
        backend_version: receipt.payload.backend_version.clone(),
        adapter_id: receipt.payload.adapter_id.clone(),
        adapter_version: receipt.payload.adapter_version.clone(),
        output_modality: receipt.payload.output_modality,
        final_execution_state,
        submission_count: submissions,
        observed_output_sha256: observed_output_sha256.clone(),
        observed_output_byte_length: receipt.payload.output_byte_length,
    };
    let report_sha256 = domain_digest(BINDING_REPORT_DOMAIN, &report_payload)
        .map_err(|issue| vec![issue])?;
    Ok(MusicRealizationVerifiedBindingV1 {
        transaction_id: transaction.transaction_id.clone(),
        attempt_id: attempt.attempt_id.clone(),
        candidate_id: attempt.candidate_id.clone(),
        source_lineage_sha256: transaction.source_lineage_sha256.clone(),
        realization_request_id: receipt.payload.realization_request_id.clone(),
        request_sha256: receipt.payload.realization_request_sha256.clone(),
        capability_manifest_sha256: receipt.payload.capability_manifest_sha256.clone(),
        upstream_realization_receipt_sha256: receipt
            .payload
            .upstream_realization_receipt_sha256
            .clone(),
        realization_receipt_sha256: receipt.receipt_sha256.clone(),
        backend_id: receipt.payload.backend_id.clone(),
        backend_version: receipt.payload.backend_version.clone(),
        adapter_id: receipt.payload.adapter_id.clone(),
        adapter_version: receipt.payload.adapter_version.clone(),
        output_modality: receipt.payload.output_modality,
        final_execution_state,
        submission_count: submissions,
        observed_output_sha256,
        observed_output_byte_length: receipt.payload.output_byte_length,
        report_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music_edit_transaction_decision::{
        MusicEditAttemptKindV1, MusicEditDecisionActionV1, MusicEditDecisionReasonV1,
        MusicEditEvidenceRecordV1, MusicEditTransactionBudgetV1,
        MusicEditTransactionPolicyV1, evaluate_music_edit_transaction_v1,
        music_edit_transaction_policy_digest_v1,
    };

    fn d(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn required() -> Vec<MusicEditEvidenceKindV1> {
        vec![
            MusicEditEvidenceKindV1::SymbolicObligation,
            MusicEditEvidenceKindV1::ActualByteMaterialization,
            MusicEditEvidenceKindV1::RendererExecution,
            MusicEditEvidenceKindV1::TargetAcoustic,
            MusicEditEvidenceKindV1::ProtectedPreservation,
            MusicEditEvidenceKindV1::Metamorphic,
            MusicEditEvidenceKindV1::RightsAdmission,
            MusicEditEvidenceKindV1::ReferenceAdjudication,
            MusicEditEvidenceKindV1::RealizationExecution,
        ]
    }

    fn evidence_record(
        kind: MusicEditEvidenceKindV1,
        candidate_sha: Option<&str>,
        verified: bool,
    ) -> MusicEditEvidenceRecordV1 {
        if verified {
            MusicEditEvidenceRecordV1 {
                kind,
                state: MusicEditEvidenceStateV1::Verified,
                source_lineage_sha256: Some(d(1)),
                candidate_artifact_sha256: candidate_sha.map(str::to_owned),
                receipt_sha256: Some(d(10)),
                report_sha256: Some(d(11)),
                replay_verifier_sha256: Some(d(12)),
                failure_class: None,
                reason_code: None,
            }
        } else {
            MusicEditEvidenceRecordV1 {
                kind,
                state: MusicEditEvidenceStateV1::Pending,
                source_lineage_sha256: None,
                candidate_artifact_sha256: None,
                receipt_sha256: None,
                report_sha256: None,
                replay_verifier_sha256: None,
                failure_class: None,
                reason_code: None,
            }
        }
    }

    fn transaction(
        outcome: MusicEditAttemptOutcomeV1,
        output_sha256: Option<&str>,
        output_length: u64,
        render_calls: u64,
    ) -> MusicEditTransactionV1 {
        let policy = MusicEditTransactionPolicyV1 {
            schema_version:
                crate::music_edit_transaction_decision::MUSIC_EDIT_TRANSACTION_POLICY_SCHEMA_V1
                    .to_owned(),
            policy_id: "bridge-test-policy-v1".to_owned(),
            required_evidence: required(),
        };
        let policy_sha256 = music_edit_transaction_policy_digest_v1(&policy).unwrap();
        let completed = outcome == MusicEditAttemptOutcomeV1::CandidateReady;
        let attempt = MusicEditCandidateAttemptV1 {
            attempt_index: 1,
            attempt_id: "attempt-001".to_owned(),
            candidate_id: "candidate-001".to_owned(),
            kind: MusicEditAttemptKindV1::Initial,
            outcome,
            source_lineage_sha256: d(1),
            candidate_symbolic_sha256: Some(d(3)),
            candidate_artifact_sha256: output_sha256.map(str::to_owned),
            outcome_reason_code: if completed {
                None
            } else {
                Some("provider_timeout_or_failure".to_owned())
            },
            render_calls,
            elapsed_ms: 200,
            materialized_bytes: if completed { output_length } else { 0 },
            evidence: required()
                .into_iter()
                .map(|kind| {
                    evidence_record(
                        kind,
                        output_sha256,
                        completed && kind != MusicEditEvidenceKindV1::RealizationExecution,
                    )
                })
                .collect(),
        };
        MusicEditTransactionV1 {
            schema_version:
                crate::music_edit_transaction_decision::MUSIC_EDIT_TRANSACTION_SCHEMA_V1
                    .to_owned(),
            transaction_id: "transaction-001".to_owned(),
            source_lineage_sha256: d(1),
            validity_envelope_sha256: d(2),
            policy_sha256,
            policy,
            budget: MusicEditTransactionBudgetV1 {
                maximum_render_calls: 10,
                maximum_repair_iterations: 4,
                maximum_elapsed_ms: 10_000,
                maximum_materialized_bytes: 1_000_000,
            },
            attempts: vec![attempt],
        }
    }

    fn expected_binding_for(
        request_id: &str,
        backend_id: &str,
        output_modality: MusicRealizationOutputModalityV1,
    ) -> MusicRealizationExpectedBindingV1 {
        MusicRealizationExpectedBindingV1 {
            transaction_id: "transaction-001".to_owned(),
            attempt_id: "attempt-001".to_owned(),
            candidate_id: "candidate-001".to_owned(),
            source_lineage_sha256: d(1),
            realization_request_id: request_id.to_owned(),
            realization_request_sha256: d(21),
            capability_manifest_sha256: d(22),
            upstream_realization_receipt_sha256: d(23),
            backend_id: backend_id.to_owned(),
            backend_version: "1.0.0".to_owned(),
            adapter_id: "mock-adapter".to_owned(),
            adapter_version: "1.0.0".to_owned(),
            output_modality,
            controls: Vec::new(),
        }
    }

    fn expected_binding() -> MusicRealizationExpectedBindingV1 {
        expected_binding_for(
            "request-001",
            "mock-midi-realizer",
            MusicRealizationOutputModalityV1::Midi,
        )
    }

    fn expected_control(
        control_id: &str,
        request_control_sha256: &str,
        requirement: MusicRealizerControlRequirementV1,
        declared_fidelity: MusicRealizerDeclaredFidelityV1,
        fidelity_bound_sha256: Option<&str>,
    ) -> MusicRealizationExpectedControlV1 {
        MusicRealizationExpectedControlV1 {
            control_id: control_id.to_owned(),
            request_control_sha256: request_control_sha256.to_owned(),
            requirement,
            declared_fidelity,
            fidelity_bound_sha256: fidelity_bound_sha256.map(str::to_owned),
        }
    }

    fn expected_binding_with_control(
        control: MusicRealizationExpectedControlV1,
    ) -> MusicRealizationExpectedBindingV1 {
        let mut expected = expected_binding();
        expected.controls.push(control);
        expected
    }

    fn attach_realization_evidence(
        tx: &mut MusicEditTransactionV1,
        evidence: MusicEditEvidenceRecordV1,
    ) {
        let record = tx.attempts[0]
            .evidence
            .iter_mut()
            .find(|record| record.kind == MusicEditEvidenceKindV1::RealizationExecution)
            .expect("realization evidence gate is present in the policy");
        *record = evidence;
    }

    fn completed_receipt(
        tx: &MusicEditTransactionV1,
        bytes: &[u8],
    ) -> MusicRealizationBridgeReceiptV1 {
        let payload = MusicRealizationBridgeReceiptPayloadV1 {
            schema_version: MUSIC_REALIZATION_BRIDGE_SCHEMA_V1.to_owned(),
            transaction_id: tx.transaction_id.clone(),
            attempt_id: "attempt-001".to_owned(),
            realization_request_id: "request-001".to_owned(),
            realization_request_sha256: d(21),
            capability_manifest_sha256: d(22),
            upstream_realization_receipt_sha256: d(23),
            backend_id: "mock-midi-realizer".to_owned(),
            backend_version: "1.0.0".to_owned(),
            adapter_id: "mock-adapter".to_owned(),
            adapter_version: "1.0.0".to_owned(),
            source_lineage_sha256: tx.source_lineage_sha256.clone(),
            candidate_id: "candidate-001".to_owned(),
            output_modality: MusicRealizationOutputModalityV1::Midi,
            output_byte_length: bytes.len() as u64,
            output_artifact_sha256: Some(sha256_bytes(bytes)),
            idempotency_key_sha256: Some(d(24)),
            execution_events: vec![
                MusicRealizerExecutionEventV1 {
                    sequence: 1,
                    state: MusicRealizerExecutionEventStateV1::Submitted,
                    provider_request_id: Some("provider-request-001".to_owned()),
                    reconciles_sequence: None,
                    reconciliation_receipt_sha256: None,
                    reason_code: None,
                },
                MusicRealizerExecutionEventV1 {
                    sequence: 2,
                    state: MusicRealizerExecutionEventStateV1::Completed,
                    provider_request_id: Some("provider-request-001".to_owned()),
                    reconciles_sequence: None,
                    reconciliation_receipt_sha256: None,
                    reason_code: None,
                },
            ],
            controls: Vec::new(),
        };
        seal_music_realization_bridge_receipt_v1(payload).unwrap()
    }

    #[test]
    fn completed_realization_binds_actual_bytes_to_exact_candidate() {
        let bytes = b"MThd\0\0\0\x06";
        let output_sha = sha256_bytes(bytes);
        let mut tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let receipt = completed_receipt(&tx, bytes);
        let binding = verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes),
        )
        .unwrap();
        assert_eq!(binding.candidate_id, "candidate-001");
        assert_eq!(binding.submission_count, 1);
        assert_eq!(binding.observed_output_sha256.as_deref(), Some(output_sha.as_str()));
        assert_eq!(binding.final_execution_state, MusicRealizerExecutionEventStateV1::Completed);
        let evidence = binding.to_evidence_record(&d(0x77)).unwrap();
        attach_realization_evidence(&mut tx, evidence);
        let decision = evaluate_music_edit_transaction_v1(&tx, &tx.policy_sha256).unwrap();
        assert_eq!(decision.payload.action, MusicEditDecisionActionV1::Accept);
    }

    #[test]
    fn bridge_refuses_when_transaction_policy_omits_realization_evidence() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let mut tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        tx.policy
            .required_evidence
            .retain(|kind| *kind != MusicEditEvidenceKindV1::RealizationExecution);
        tx.attempts[0]
            .evidence
            .retain(|record| record.kind != MusicEditEvidenceKindV1::RealizationExecution);
        tx.policy_sha256 = music_edit_transaction_policy_digest_v1(&tx.policy).unwrap();
        let receipt = completed_receipt(&tx, bytes);
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::RealizationEvidenceNotRequiredByPolicy));
    }

    #[test]
    fn transaction_abstains_until_realization_evidence_is_verified() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let decision = evaluate_music_edit_transaction_v1(&tx, &tx.policy_sha256).unwrap();
        assert_eq!(decision.payload.action, MusicEditDecisionActionV1::Abstain);
        assert!(decision
            .payload
            .reasons
            .contains(&MusicEditDecisionReasonV1::EvidencePending));
    }

    #[test]
    fn mutated_output_bytes_are_rejected() {
        let bytes = b"original-midi";
        let altered = b"altered-midi";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let receipt = completed_receipt(&tx, bytes);
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(altered)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::OutputDigestMismatch));
    }

    #[test]
    fn candidate_and_source_substitution_are_rejected() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.candidate_id = "candidate-other".to_owned();
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::CandidateMismatch));

        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.source_lineage_sha256 = d(99);
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::SourceLineageMismatch));
    }

    #[test]
    fn hard_control_cannot_silently_downgrade_to_best_effort() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let expected = expected_binding_with_control(expected_control(
            "preserve_melody_pitch",
            &d(61),
            MusicRealizerControlRequirementV1::Hard,
            MusicRealizerDeclaredFidelityV1::Exact,
            None,
        ));
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.controls.push(MusicRealizerControlBindingV1 {
            control_id: "preserve_melody_pitch".to_owned(),
            request_control_sha256: d(61),
            requirement: MusicRealizerControlRequirementV1::Hard,
            declared_fidelity: MusicRealizerDeclaredFidelityV1::BestEffort,
            fidelity_bound_sha256: None,
            observed_disposition: MusicRealizerObservedDispositionV1::BestEffort,
        });
        receipt.payload.controls.sort_by(|a, b| a.control_id.cmp(&b.control_id));
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected,
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::ExpectedControlBindingMismatch(
            "preserve_melody_pitch".to_owned()
        )));
    }

    #[test]
    fn bounded_hard_control_requires_a_bound_commitment() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let expected = expected_binding_with_control(expected_control(
            "maximum_pitch_deviation",
            &d(63),
            MusicRealizerControlRequirementV1::Hard,
            MusicRealizerDeclaredFidelityV1::Bounded,
            Some(&d(64)),
        ));
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.controls.push(MusicRealizerControlBindingV1 {
            control_id: "maximum_pitch_deviation".to_owned(),
            request_control_sha256: d(63),
            requirement: MusicRealizerControlRequirementV1::Hard,
            declared_fidelity: MusicRealizerDeclaredFidelityV1::Bounded,
            fidelity_bound_sha256: None,
            observed_disposition: MusicRealizerObservedDispositionV1::Applied,
        });
        receipt.payload.controls.sort_by(|a, b| a.control_id.cmp(&b.control_id));
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected,
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::InvalidControl(
            "maximum_pitch_deviation".to_owned()
        )));

        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.controls.push(MusicRealizerControlBindingV1 {
            control_id: "maximum_pitch_deviation".to_owned(),
            request_control_sha256: d(63),
            requirement: MusicRealizerControlRequirementV1::Hard,
            declared_fidelity: MusicRealizerDeclaredFidelityV1::Bounded,
            fidelity_bound_sha256: Some(d(64)),
            observed_disposition: MusicRealizerObservedDispositionV1::Applied,
        });
        receipt.payload.controls.sort_by(|a, b| a.control_id.cmp(&b.control_id));
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected,
            &receipt,
            Some(bytes)
        )
        .is_ok());
    }

    #[test]
    fn omitted_or_substituted_requested_controls_are_rejected() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let expected = expected_binding_with_control(expected_control(
            "preserve_melody_pitch",
            &d(61),
            MusicRealizerControlRequirementV1::Hard,
            MusicRealizerDeclaredFidelityV1::Exact,
            None,
        ));

        // The provider cannot omit a hard control from its receipt.
        let receipt = completed_receipt(&tx, bytes);
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected,
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::ExpectedControlSetMismatch));

        // Nor can it present the same control ID with a different request digest.
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.controls.push(MusicRealizerControlBindingV1 {
            control_id: "preserve_melody_pitch".to_owned(),
            request_control_sha256: d(62),
            requirement: MusicRealizerControlRequirementV1::Hard,
            declared_fidelity: MusicRealizerDeclaredFidelityV1::Exact,
            fidelity_bound_sha256: None,
            observed_disposition: MusicRealizerObservedDispositionV1::Applied,
        });
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected,
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::ExpectedControlBindingMismatch(
            "preserve_melody_pitch".to_owned()
        )));
    }

    #[test]
    fn completion_cannot_be_bound_to_a_different_provider_request() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.execution_events[1].provider_request_id =
            Some("different-provider-request".to_owned());
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::ProviderRequestIdentityMismatch));
    }

    #[test]
    fn best_effort_advisory_control_is_audited_but_not_a_hard_failure() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let expected = expected_binding_with_control(expected_control(
            "suggest_sparse_texture",
            &d(62),
            MusicRealizerControlRequirementV1::Advisory,
            MusicRealizerDeclaredFidelityV1::BestEffort,
            None,
        ));
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.controls.push(MusicRealizerControlBindingV1 {
            control_id: "suggest_sparse_texture".to_owned(),
            request_control_sha256: d(62),
            requirement: MusicRealizerControlRequirementV1::Advisory,
            declared_fidelity: MusicRealizerDeclaredFidelityV1::BestEffort,
            fidelity_bound_sha256: None,
            observed_disposition: MusicRealizerObservedDispositionV1::BestEffort,
        });
        receipt.payload.controls.sort_by(|a, b| a.control_id.cmp(&b.control_id));
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected,
            &receipt,
            Some(bytes)
        )
        .is_ok());
    }

    #[test]
    fn unknown_provider_outcome_abstains_and_binds_no_output() {
        let tx = transaction(
            MusicEditAttemptOutcomeV1::OutcomeUnknown,
            None,
            0,
            1,
        );
        let payload = MusicRealizationBridgeReceiptPayloadV1 {
            schema_version: MUSIC_REALIZATION_BRIDGE_SCHEMA_V1.to_owned(),
            transaction_id: tx.transaction_id.clone(),
            attempt_id: "attempt-001".to_owned(),
            realization_request_id: "request-unknown".to_owned(),
            realization_request_sha256: d(21),
            capability_manifest_sha256: d(22),
            upstream_realization_receipt_sha256: d(23),
            backend_id: "mock-streaming-realizer".to_owned(),
            backend_version: "1.0.0".to_owned(),
            adapter_id: "mock-adapter".to_owned(),
            adapter_version: "1.0.0".to_owned(),
            source_lineage_sha256: tx.source_lineage_sha256.clone(),
            candidate_id: "candidate-001".to_owned(),
            output_modality: MusicRealizationOutputModalityV1::Audio,
            output_byte_length: 0,
            output_artifact_sha256: None,
            idempotency_key_sha256: None,
            execution_events: vec![
                MusicRealizerExecutionEventV1 {
                    sequence: 1,
                    state: MusicRealizerExecutionEventStateV1::Submitted,
                    provider_request_id: Some("provider-request-unknown".to_owned()),
                    reconciles_sequence: None,
                    reconciliation_receipt_sha256: None,
                    reason_code: None,
                },
                MusicRealizerExecutionEventV1 {
                    sequence: 2,
                    state: MusicRealizerExecutionEventStateV1::OutcomeUnknown,
                    provider_request_id: Some("provider-request-unknown".to_owned()),
                    reconciles_sequence: None,
                    reconciliation_receipt_sha256: None,
                    reason_code: Some("timeout_after_submission".to_owned()),
                },
            ],
            controls: Vec::new(),
        };
        let receipt = seal_music_realization_bridge_receipt_v1(payload).unwrap();
        let binding = verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding_for(
                "request-unknown",
                "mock-streaming-realizer",
                MusicRealizationOutputModalityV1::Audio,
            ),
            &receipt,
            None,
        )
        .unwrap();
        assert_eq!(
            binding.final_execution_state,
            MusicRealizerExecutionEventStateV1::OutcomeUnknown
        );
        assert_eq!(
            binding.to_evidence_record(&d(0x77)),
            Err(MusicRealizationBridgeIssueV1::UnresolvedExecutionCannotBeVerified)
        );
        let decision = evaluate_music_edit_transaction_v1(&tx, &tx.policy_sha256).unwrap();
        assert_eq!(decision.payload.action, MusicEditDecisionActionV1::Abstain);
    }

    #[test]
    fn blind_retry_after_unknown_outcome_is_rejected() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            2,
        );
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.execution_events = vec![
            MusicRealizerExecutionEventV1 {
                sequence: 1,
                state: MusicRealizerExecutionEventStateV1::Submitted,
                provider_request_id: Some("request-1".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
            MusicRealizerExecutionEventV1 {
                sequence: 2,
                state: MusicRealizerExecutionEventStateV1::OutcomeUnknown,
                provider_request_id: Some("request-1".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: Some("timeout".to_owned()),
            },
            MusicRealizerExecutionEventV1 {
                sequence: 3,
                state: MusicRealizerExecutionEventStateV1::Submitted,
                provider_request_id: Some("request-2".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
            MusicRealizerExecutionEventV1 {
                sequence: 4,
                state: MusicRealizerExecutionEventStateV1::Completed,
                provider_request_id: Some("request-2".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
        ];
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::InvalidExecutionTransition));
    }

    #[test]
    fn retry_is_permitted_only_after_reconciliation_and_consumes_each_submission() {
        let bytes = b"candidate-after-reconcile";
        let output_sha = sha256_bytes(bytes);
        let mut tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            2,
        );
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.idempotency_key_sha256 = Some(d(41));
        receipt.payload.execution_events = vec![
            MusicRealizerExecutionEventV1 {
                sequence: 1,
                state: MusicRealizerExecutionEventStateV1::Submitted,
                provider_request_id: Some("request-1".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
            MusicRealizerExecutionEventV1 {
                sequence: 2,
                state: MusicRealizerExecutionEventStateV1::OutcomeUnknown,
                provider_request_id: Some("request-1".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: Some("timeout".to_owned()),
            },
            MusicRealizerExecutionEventV1 {
                sequence: 3,
                state: MusicRealizerExecutionEventStateV1::ReconciledNotExecuted,
                provider_request_id: Some("request-1".to_owned()),
                reconciles_sequence: Some(2),
                reconciliation_receipt_sha256: Some(d(42)),
                reason_code: Some("provider_status_not_executed".to_owned()),
            },
            MusicRealizerExecutionEventV1 {
                sequence: 4,
                state: MusicRealizerExecutionEventStateV1::Submitted,
                provider_request_id: Some("request-2".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
            MusicRealizerExecutionEventV1 {
                sequence: 5,
                state: MusicRealizerExecutionEventStateV1::Completed,
                provider_request_id: Some("request-2".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
        ];
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        let binding = verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes),
        )
        .unwrap();
        assert_eq!(binding.submission_count, 2);
        let evidence = binding.to_evidence_record(&d(0x77)).unwrap();
        attach_realization_evidence(&mut tx, evidence);
        let decision = evaluate_music_edit_transaction_v1(&tx, &tx.policy_sha256).unwrap();
        assert_eq!(decision.payload.action, MusicEditDecisionActionV1::Accept);
    }

    #[test]
    fn submission_count_cannot_exceed_transaction_render_budget() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.execution_events.push(MusicRealizerExecutionEventV1 {
            sequence: 3,
            state: MusicRealizerExecutionEventStateV1::Submitted,
            provider_request_id: Some("unaccounted-request".to_owned()),
            reconciles_sequence: None,
            reconciliation_receipt_sha256: None,
            reason_code: None,
        });
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert_eq!(tx.attempts[0].render_calls, 1);
        assert_eq!(
            receipt.payload.execution_events
                .iter()
                .filter(|event| event.state == MusicRealizerExecutionEventStateV1::Submitted)
                .count(),
            2
        );
        let issues = verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes),
        )
        .unwrap_err();
        assert!(
            issues.contains(&MusicRealizationBridgeIssueV1::SubmissionBudgetUnderCounted),
            "expected submission under-count; got: {issues:?}"
        );
        let issues = verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes),
        )
        .unwrap_err();
        assert!(issues.contains(&MusicRealizationBridgeIssueV1::InvalidExecutionTransition));
        assert!(issues.contains(&MusicRealizationBridgeIssueV1::SubmissionBudgetUnderCounted));
    }

    #[test]
    fn resealed_receipt_cannot_substitute_expected_capability_manifest() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.capability_manifest_sha256 = d(88);
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::ExpectedBindingMismatch(
            "capability_manifest_sha256"
        )));
    }

    #[test]
    fn reconciled_completed_call_becomes_verified_without_a_second_submission() {
        let bytes = b"candidate-resolved-by-status";
        let output_sha = sha256_bytes(bytes);
        let mut tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.execution_events = vec![
            MusicRealizerExecutionEventV1 {
                sequence: 1,
                state: MusicRealizerExecutionEventStateV1::Submitted,
                provider_request_id: Some("provider-request-001".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
            MusicRealizerExecutionEventV1 {
                sequence: 2,
                state: MusicRealizerExecutionEventStateV1::OutcomeUnknown,
                provider_request_id: Some("provider-request-001".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: Some("client_timeout".to_owned()),
            },
            MusicRealizerExecutionEventV1 {
                sequence: 3,
                state: MusicRealizerExecutionEventStateV1::ReconciledCompleted,
                provider_request_id: Some("provider-request-001".to_owned()),
                reconciles_sequence: Some(2),
                reconciliation_receipt_sha256: Some(d(89)),
                reason_code: None,
            },
        ];
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        let binding = verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes),
        )
        .unwrap();
        assert_eq!(binding.submission_count, 1);
        assert_eq!(
            binding.final_execution_state,
            MusicRealizerExecutionEventStateV1::ReconciledCompleted
        );
        let evidence = binding.to_evidence_record(&d(0x77)).unwrap();
        attach_realization_evidence(&mut tx, evidence);
        let decision = evaluate_music_edit_transaction_v1(&tx, &tx.policy_sha256).unwrap();
        assert_eq!(decision.payload.action, MusicEditDecisionActionV1::Accept);
    }

    #[test]
    fn bridge_receipt_tampering_fails_digest_replay() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.backend_version = "2.0.0".to_owned();
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::ReceiptDigestMismatch));
    }

    #[test]
    fn missing_verified_byte_receipt_blocks_realization_binding() {
        let bytes = b"candidate";
        let output_sha = sha256_bytes(bytes);
        let mut tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            1,
        );
        tx.attempts[0]
            .evidence
            .retain(|item| item.kind != MusicEditEvidenceKindV1::ActualByteMaterialization);
        let receipt = completed_receipt(&tx, bytes);
        assert!(verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes)
        )
        .unwrap_err()
        .contains(&MusicRealizationBridgeIssueV1::MissingByteMaterializationEvidence));
    }

    #[test]
    fn retry_budget_is_not_refunded_after_reconciled_non_execution() {
        let bytes = b"candidate-after-reconcile";
        let output_sha = sha256_bytes(bytes);
        let mut tx = transaction(
            MusicEditAttemptOutcomeV1::CandidateReady,
            Some(&output_sha),
            bytes.len() as u64,
            2,
        );
        tx.budget.maximum_render_calls = 1;
        let mut receipt = completed_receipt(&tx, bytes);
        receipt.payload.idempotency_key_sha256 = Some(d(51));
        receipt.payload.execution_events = vec![
            MusicRealizerExecutionEventV1 {
                sequence: 1,
                state: MusicRealizerExecutionEventStateV1::Submitted,
                provider_request_id: Some("request-1".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
            MusicRealizerExecutionEventV1 {
                sequence: 2,
                state: MusicRealizerExecutionEventStateV1::OutcomeUnknown,
                provider_request_id: Some("request-1".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: Some("timeout".to_owned()),
            },
            MusicRealizerExecutionEventV1 {
                sequence: 3,
                state: MusicRealizerExecutionEventStateV1::ReconciledNotExecuted,
                provider_request_id: Some("request-1".to_owned()),
                reconciles_sequence: Some(2),
                reconciliation_receipt_sha256: Some(d(52)),
                reason_code: Some("provider_status_not_executed".to_owned()),
            },
            MusicRealizerExecutionEventV1 {
                sequence: 4,
                state: MusicRealizerExecutionEventStateV1::Submitted,
                provider_request_id: Some("request-2".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
            MusicRealizerExecutionEventV1 {
                sequence: 5,
                state: MusicRealizerExecutionEventStateV1::Completed,
                provider_request_id: Some("request-2".to_owned()),
                reconciles_sequence: None,
                reconciliation_receipt_sha256: None,
                reason_code: None,
            },
        ];
        receipt = seal_music_realization_bridge_receipt_v1(receipt.payload).unwrap();
        let binding = verify_music_realization_transaction_binding_v1(
            &tx,
            &expected_binding(),
            &receipt,
            Some(bytes),
        )
        .unwrap();
        assert_eq!(binding.submission_count, 2);
        let decision = evaluate_music_edit_transaction_v1(&tx, &tx.policy_sha256).unwrap();
        assert_eq!(decision.payload.action, MusicEditDecisionActionV1::Reject);
        assert!(decision
            .payload
            .reasons
            .contains(&MusicEditDecisionReasonV1::BudgetExceeded));
    }
}
