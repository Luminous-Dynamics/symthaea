// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replayable, budgeted terminal decisions for one exact musical edit transaction.
//!
//! This module decides whether an immutable candidate attempt may be accepted,
//! repaired, rejected, or must be left unresolved. It does not render audio,
//! infer musical quality, authorize publication, or replace upstream verifiers.
//! Evidence receipts must be replayed by their owning verifier before they are
//! represented here as Verified.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const MUSIC_EDIT_TRANSACTION_SCHEMA_V1: &str = "mel-music-edit-transaction-v1";
pub const MUSIC_EDIT_DECISION_RECEIPT_SCHEMA_V1: &str =
    "mel-music-edit-transaction-decision-receipt-v1";

const TRANSACTION_DOMAIN: &[u8] = b"luminous.mel.music-edit-transaction.v1";
const DECISION_DOMAIN: &[u8] = b"luminous.mel.music-edit-transaction-decision.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditEvidenceKindV1 {
    SymbolicObligation,
    RendererExecution,
    TargetAcoustic,
    ProtectedPreservation,
    Metamorphic,
    RightsAdmission,
    ReferenceAdjudication,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditEvidenceStateV1 {
    NotRun,
    Pending,
    Verified,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditEvidenceFailureClassV1 {
    Repairable,
    HardGate,
    Integrity,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditAttemptKindV1 {
    Initial,
    Repair,
    Alternative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditAttemptOutcomeV1 {
    CandidateReady,
    RepairableFailure,
    HardInvalid,
    OutcomeUnknown,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditDecisionActionV1 {
    Accept,
    Repair,
    Abstain,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditDecisionReasonV1 {
    AllRequiredEvidenceVerified,
    BudgetExceeded,
    BudgetExhausted,
    CandidateHardInvalid,
    CandidateRepairable,
    EvidenceMissing,
    EvidencePending,
    EvidenceRejected,
    ExecutionFailed,
    IntegrityViolation,
    NoCandidateAttempt,
    OutcomeUnknown,
    RequiredEvidenceUnresolved,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditTransactionBudgetV1 {
    pub maximum_render_calls: u64,
    pub maximum_repair_iterations: u64,
    pub maximum_elapsed_ms: u64,
    pub maximum_materialized_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditTransactionUsageV1 {
    pub render_calls: u64,
    pub repair_iterations: u64,
    pub elapsed_ms: u64,
    pub materialized_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditEvidenceRecordV1 {
    pub kind: MusicEditEvidenceKindV1,
    pub state: MusicEditEvidenceStateV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_lineage_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_artifact_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_verifier_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_class: Option<MusicEditEvidenceFailureClassV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditCandidateAttemptV1 {
    /// One-based, contiguous position in the immutable transaction ledger.
    pub attempt_index: u64,
    pub attempt_id: String,
    pub candidate_id: String,
    pub kind: MusicEditAttemptKindV1,
    pub outcome: MusicEditAttemptOutcomeV1,
    pub source_lineage_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_symbolic_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_artifact_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_reason_code: Option<String>,
    /// Actual calls, including failed or unusable renders.
    pub render_calls: u64,
    /// Actual wall-clock duration attributed to this attempt.
    pub elapsed_ms: u64,
    /// Actual source and candidate bytes materialized by this attempt.
    pub materialized_bytes: u64,
    /// Canonically ordered, candidate-scoped evidence observations.
    pub evidence: Vec<MusicEditEvidenceRecordV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditTransactionV1 {
    pub schema_version: String,
    pub transaction_id: String,
    pub source_lineage_sha256: String,
    pub validity_envelope_sha256: String,
    pub policy_sha256: String,
    /// Canonical ascending enum order; each evidence kind appears once.
    pub required_evidence: Vec<MusicEditEvidenceKindV1>,
    pub budget: MusicEditTransactionBudgetV1,
    /// Append-only, one-based attempt ledger. Candidate IDs are never reused.
    pub attempts: Vec<MusicEditCandidateAttemptV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicEditTransactionIssueV1 {
    WrongSchema,
    EmptyField(&'static str),
    InvalidDigest(&'static str),
    NonCanonicalRequiredEvidence,
    DuplicateRequiredEvidence(MusicEditEvidenceKindV1),
    NonCanonicalAttemptOrder,
    FirstAttemptMustBeInitial,
    LaterAttemptCannotBeInitial,
    EmptyOrDuplicateAttemptId(String),
    EmptyOrDuplicateCandidateId(String),
    InvalidAttemptSource(String),
    CandidateReadyWithoutArtifact(String),
    InvalidCandidateDigest(String),
    InvalidOutcomeReason(String),
    UnknownOutcomeNotTerminal(String),
    NonCanonicalEvidenceOrder(String),
    DuplicateEvidenceKind {
        attempt_id: String,
        kind: MusicEditEvidenceKindV1,
    },
    InvalidEvidenceRecord {
        attempt_id: String,
        kind: MusicEditEvidenceKindV1,
    },
    BudgetUsageOverflow,
    SerializationError,
    InvalidTransaction,
    DecisionReceiptMismatch,
}

impl std::fmt::Display for MusicEditTransactionIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MusicEditTransactionIssueV1 {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditTransactionDecisionPayloadV1 {
    pub schema_version: String,
    pub transaction_id: String,
    pub transaction_sha256: String,
    pub policy_sha256: String,
    pub action: MusicEditDecisionActionV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_attempt_id: Option<String>,
    pub reasons: Vec<MusicEditDecisionReasonV1>,
    pub usage: MusicEditTransactionUsageV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditTransactionDecisionReceiptV1 {
    pub payload: MusicEditTransactionDecisionPayloadV1,
    pub receipt_sha256: String,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, MusicEditTransactionIssueV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| MusicEditTransactionIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(b"\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_evidence_record(
    attempt_id: &str,
    record: &MusicEditEvidenceRecordV1,
    issues: &mut Vec<MusicEditTransactionIssueV1>,
) {
    let bound_digests = [
        record.source_lineage_sha256.as_deref(),
        record.candidate_artifact_sha256.as_deref(),
        record.receipt_sha256.as_deref(),
        record.report_sha256.as_deref(),
        record.replay_verifier_sha256.as_deref(),
    ];
    let all_bound = bound_digests.iter().all(|value| value.is_some());
    let all_empty = bound_digests.iter().all(|value| value.is_none());

    let state_shape_valid = match record.state {
        MusicEditEvidenceStateV1::NotRun | MusicEditEvidenceStateV1::Pending => {
            all_empty && record.failure_class.is_none() && record.reason_code.is_none()
        }
        MusicEditEvidenceStateV1::Verified => {
            all_bound && record.failure_class.is_none() && record.reason_code.is_none()
        }
        MusicEditEvidenceStateV1::Rejected => {
            all_bound
                && record.failure_class.is_some()
                && record.reason_code.as_ref().is_some_and(|code| !code.trim().is_empty())
        }
    };

    let digest_fields_valid = bound_digests.iter().flatten().all(|value| is_sha256(value));
    if !state_shape_valid || !digest_fields_valid {
        issues.push(MusicEditTransactionIssueV1::InvalidEvidenceRecord {
            attempt_id: attempt_id.to_owned(),
            kind: record.kind,
        });
    }
}

fn checked_usage(
    transaction: &MusicEditTransactionV1,
) -> Result<MusicEditTransactionUsageV1, MusicEditTransactionIssueV1> {
    let mut usage = MusicEditTransactionUsageV1::default();
    for attempt in &transaction.attempts {
        usage.render_calls = usage
            .render_calls
            .checked_add(attempt.render_calls)
            .ok_or(MusicEditTransactionIssueV1::BudgetUsageOverflow)?;
        usage.elapsed_ms = usage
            .elapsed_ms
            .checked_add(attempt.elapsed_ms)
            .ok_or(MusicEditTransactionIssueV1::BudgetUsageOverflow)?;
        usage.materialized_bytes = usage
            .materialized_bytes
            .checked_add(attempt.materialized_bytes)
            .ok_or(MusicEditTransactionIssueV1::BudgetUsageOverflow)?;
        if attempt.kind == MusicEditAttemptKindV1::Repair {
            usage.repair_iterations = usage
                .repair_iterations
                .checked_add(1)
                .ok_or(MusicEditTransactionIssueV1::BudgetUsageOverflow)?;
        }
    }
    Ok(usage)
}

/// Validates the append-only ledger's structure and canonical ordering.
/// Evidence-to-candidate binding is evaluated as a terminal integrity decision.
pub fn validate_music_edit_transaction_v1(
    transaction: &MusicEditTransactionV1,
) -> Vec<MusicEditTransactionIssueV1> {
    let mut issues = Vec::new();

    if transaction.schema_version != MUSIC_EDIT_TRANSACTION_SCHEMA_V1 {
        issues.push(MusicEditTransactionIssueV1::WrongSchema);
    }
    for (field, value) in [
        ("transaction_id", transaction.transaction_id.as_str()),
        ("source_lineage_sha256", transaction.source_lineage_sha256.as_str()),
        ("validity_envelope_sha256", transaction.validity_envelope_sha256.as_str()),
        ("policy_sha256", transaction.policy_sha256.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(MusicEditTransactionIssueV1::EmptyField(field));
        }
    }
    for (field, value) in [
        ("source_lineage_sha256", transaction.source_lineage_sha256.as_str()),
        ("validity_envelope_sha256", transaction.validity_envelope_sha256.as_str()),
        ("policy_sha256", transaction.policy_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            issues.push(MusicEditTransactionIssueV1::InvalidDigest(field));
        }
    }

    if !transaction
        .required_evidence
        .windows(2)
        .all(|pair| pair[0] < pair[1])
    {
        issues.push(MusicEditTransactionIssueV1::NonCanonicalRequiredEvidence);
    }
    let mut required = BTreeSet::new();
    for kind in &transaction.required_evidence {
        if !required.insert(*kind) {
            issues.push(MusicEditTransactionIssueV1::DuplicateRequiredEvidence(*kind));
        }
    }

    let mut attempt_ids = BTreeSet::new();
    let mut candidate_ids = BTreeSet::new();
    for (index, attempt) in transaction.attempts.iter().enumerate() {
        let expected_index = u64::try_from(index + 1).unwrap_or(u64::MAX);
        if attempt.attempt_index != expected_index {
            issues.push(MusicEditTransactionIssueV1::NonCanonicalAttemptOrder);
        }
        if attempt.attempt_id.trim().is_empty() || !attempt_ids.insert(attempt.attempt_id.clone()) {
            issues.push(MusicEditTransactionIssueV1::EmptyOrDuplicateAttemptId(
                attempt.attempt_id.clone(),
            ));
        }
        if attempt.candidate_id.trim().is_empty()
            || !candidate_ids.insert(attempt.candidate_id.clone())
        {
            issues.push(MusicEditTransactionIssueV1::EmptyOrDuplicateCandidateId(
                attempt.candidate_id.clone(),
            ));
        }
        if index == 0 && attempt.kind != MusicEditAttemptKindV1::Initial {
            issues.push(MusicEditTransactionIssueV1::FirstAttemptMustBeInitial);
        }
        if index > 0 && attempt.kind == MusicEditAttemptKindV1::Initial {
            issues.push(MusicEditTransactionIssueV1::LaterAttemptCannotBeInitial);
        }
        if !is_sha256(&attempt.source_lineage_sha256) {
            issues.push(MusicEditTransactionIssueV1::InvalidAttemptSource(
                attempt.attempt_id.clone(),
            ));
        }
        if attempt.outcome == MusicEditAttemptOutcomeV1::CandidateReady
            && attempt.candidate_artifact_sha256.is_none()
        {
            issues.push(MusicEditTransactionIssueV1::CandidateReadyWithoutArtifact(
                attempt.attempt_id.clone(),
            ));
        }
        for candidate_digest in [
            attempt.candidate_symbolic_sha256.as_deref(),
            attempt.candidate_artifact_sha256.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if !is_sha256(candidate_digest) {
                issues.push(MusicEditTransactionIssueV1::InvalidCandidateDigest(
                    attempt.attempt_id.clone(),
                ));
            }
        }

        let outcome_requires_reason = attempt.outcome != MusicEditAttemptOutcomeV1::CandidateReady;
        let reason_present = attempt
            .outcome_reason_code
            .as_ref()
            .is_some_and(|reason| !reason.trim().is_empty());
        if outcome_requires_reason != reason_present {
            issues.push(MusicEditTransactionIssueV1::InvalidOutcomeReason(
                attempt.attempt_id.clone(),
            ));
        }
        if attempt.outcome == MusicEditAttemptOutcomeV1::OutcomeUnknown
            && index + 1 != transaction.attempts.len()
        {
            issues.push(MusicEditTransactionIssueV1::UnknownOutcomeNotTerminal(
                attempt.attempt_id.clone(),
            ));
        }

        if !attempt.evidence.windows(2).all(|pair| pair[0].kind < pair[1].kind) {
            issues.push(MusicEditTransactionIssueV1::NonCanonicalEvidenceOrder(
                attempt.attempt_id.clone(),
            ));
        }
        let mut evidence_kinds = BTreeSet::new();
        for record in &attempt.evidence {
            if !evidence_kinds.insert(record.kind) {
                issues.push(MusicEditTransactionIssueV1::DuplicateEvidenceKind {
                    attempt_id: attempt.attempt_id.clone(),
                    kind: record.kind,
                });
            }
            validate_evidence_record(&attempt.attempt_id, record, &mut issues);
        }
    }

    if checked_usage(transaction).is_err() {
        issues.push(MusicEditTransactionIssueV1::BudgetUsageOverflow);
    }
    issues
}

fn required_record<'a>(
    attempt: &'a MusicEditCandidateAttemptV1,
    kind: MusicEditEvidenceKindV1,
) -> Option<&'a MusicEditEvidenceRecordV1> {
    attempt.evidence.iter().find(|record| record.kind == kind)
}

fn budget_exceeded(
    usage: &MusicEditTransactionUsageV1,
    budget: &MusicEditTransactionBudgetV1,
) -> bool {
    usage.render_calls > budget.maximum_render_calls
        || usage.repair_iterations > budget.maximum_repair_iterations
        || usage.elapsed_ms > budget.maximum_elapsed_ms
        || usage.materialized_bytes > budget.maximum_materialized_bytes
}

fn budget_allows_repair(
    usage: &MusicEditTransactionUsageV1,
    budget: &MusicEditTransactionBudgetV1,
) -> bool {
    usage.render_calls < budget.maximum_render_calls
        && usage.repair_iterations < budget.maximum_repair_iterations
        && usage.elapsed_ms < budget.maximum_elapsed_ms
        && usage.materialized_bytes < budget.maximum_materialized_bytes
}

fn make_receipt(
    transaction: &MusicEditTransactionV1,
    action: MusicEditDecisionActionV1,
    selected_attempt_id: Option<String>,
    mut reasons: Vec<MusicEditDecisionReasonV1>,
    usage: MusicEditTransactionUsageV1,
) -> Result<MusicEditTransactionDecisionReceiptV1, MusicEditTransactionIssueV1> {
    reasons.sort();
    reasons.dedup();
    let transaction_sha256 = digest(TRANSACTION_DOMAIN, transaction)?;
    let payload = MusicEditTransactionDecisionPayloadV1 {
        schema_version: MUSIC_EDIT_DECISION_RECEIPT_SCHEMA_V1.to_owned(),
        transaction_id: transaction.transaction_id.clone(),
        transaction_sha256,
        policy_sha256: transaction.policy_sha256.clone(),
        action,
        selected_attempt_id,
        reasons,
        usage,
    };
    let receipt_sha256 = digest(DECISION_DOMAIN, &payload)?;
    Ok(MusicEditTransactionDecisionReceiptV1 {
        payload,
        receipt_sha256,
    })
}

/// Evaluates one immutable transaction. Structural errors are returned as issues;
/// semantic integrity, hard-gate and budget outcomes become explicit decisions.
pub fn evaluate_music_edit_transaction_v1(
    transaction: &MusicEditTransactionV1,
) -> Result<MusicEditTransactionDecisionReceiptV1, Vec<MusicEditTransactionIssueV1>> {
    let issues = validate_music_edit_transaction_v1(transaction);
    if !issues.is_empty() {
        return Err(issues);
    }
    let usage = checked_usage(transaction)
        .map_err(|issue| vec![issue])?;

    let last_attempt = transaction.attempts.last();
    let selected_attempt_id = last_attempt.map(|attempt| attempt.attempt_id.clone());

    if transaction.attempts.iter().any(|attempt| {
        attempt.source_lineage_sha256 != transaction.source_lineage_sha256
            || attempt.evidence.iter().any(|record| {
                matches!(
                    record.state,
                    MusicEditEvidenceStateV1::Verified | MusicEditEvidenceStateV1::Rejected
                ) && (record.source_lineage_sha256.as_deref()
                    != Some(transaction.source_lineage_sha256.as_str())
                    || record.candidate_artifact_sha256.as_deref()
                        != attempt.candidate_artifact_sha256.as_deref())
            })
            || attempt.evidence.iter().any(|record| {
                record.state == MusicEditEvidenceStateV1::Rejected
                    && record.failure_class == Some(MusicEditEvidenceFailureClassV1::Integrity)
            })
    }) {
        return make_receipt(
            transaction,
            MusicEditDecisionActionV1::Reject,
            selected_attempt_id,
            vec![MusicEditDecisionReasonV1::IntegrityViolation],
            usage,
        )
        .map_err(|issue| vec![issue]);
    }

    if budget_exceeded(&usage, &transaction.budget) {
        return make_receipt(
            transaction,
            MusicEditDecisionActionV1::Reject,
            selected_attempt_id,
            vec![MusicEditDecisionReasonV1::BudgetExceeded],
            usage,
        )
        .map_err(|issue| vec![issue]);
    }

    // Earlier failed candidates remain in the ledger, but a later exact candidate
    // may still pass. The first valid candidate in append order is deterministic.
    for attempt in &transaction.attempts {
        if attempt.outcome != MusicEditAttemptOutcomeV1::CandidateReady {
            continue;
        }
        let mut all_verified = true;
        let mut hard_rejection = false;
        for kind in &transaction.required_evidence {
            match required_record(attempt, *kind).map(|record| record.state) {
                Some(MusicEditEvidenceStateV1::Verified) => {}
                Some(MusicEditEvidenceStateV1::Rejected) => {
                    let record = required_record(attempt, *kind).expect("record was matched");
                    match record.failure_class {
                        Some(MusicEditEvidenceFailureClassV1::HardGate) => {
                            hard_rejection = true;
                        }
                        Some(MusicEditEvidenceFailureClassV1::Integrity) => {
                            hard_rejection = true;
                        }
                        _ => all_verified = false,
                    }
                }
                Some(MusicEditEvidenceStateV1::NotRun)
                | Some(MusicEditEvidenceStateV1::Pending)
                | None => all_verified = false,
            }
        }
        if all_verified {
            return make_receipt(
                transaction,
                MusicEditDecisionActionV1::Accept,
                Some(attempt.attempt_id.clone()),
                vec![MusicEditDecisionReasonV1::AllRequiredEvidenceVerified],
                usage,
            )
            .map_err(|issue| vec![issue]);
        }
        // This does not terminate evaluation: a later candidate may be valid.
        let _ = hard_rejection;
    }

    let Some(last) = last_attempt else {
        return make_receipt(
            transaction,
            MusicEditDecisionActionV1::Abstain,
            None,
            vec![MusicEditDecisionReasonV1::NoCandidateAttempt],
            usage,
        )
        .map_err(|issue| vec![issue]);
    };

    let mut hard_rejection = last.outcome == MusicEditAttemptOutcomeV1::HardInvalid;
    let mut repairable = last.outcome == MusicEditAttemptOutcomeV1::RepairableFailure;
    let mut unresolved = matches!(
        last.outcome,
        MusicEditAttemptOutcomeV1::OutcomeUnknown | MusicEditAttemptOutcomeV1::Failed
    );
    let mut reasons = Vec::new();

    if hard_rejection {
        reasons.push(MusicEditDecisionReasonV1::CandidateHardInvalid);
    }
    for kind in &transaction.required_evidence {
        match required_record(last, *kind) {
            None | Some(MusicEditEvidenceRecordV1 {
                state: MusicEditEvidenceStateV1::NotRun,
                ..
            }) => {
                unresolved = true;
                reasons.push(MusicEditDecisionReasonV1::EvidenceMissing);
            }
            Some(MusicEditEvidenceRecordV1 {
                state: MusicEditEvidenceStateV1::Pending,
                ..
            }) => {
                unresolved = true;
                reasons.push(MusicEditDecisionReasonV1::EvidencePending);
            }
            Some(MusicEditEvidenceRecordV1 {
                state: MusicEditEvidenceStateV1::Rejected,
                failure_class: Some(MusicEditEvidenceFailureClassV1::HardGate),
                ..
            }) => {
                hard_rejection = true;
                reasons.push(MusicEditDecisionReasonV1::EvidenceRejected);
            }
            Some(MusicEditEvidenceRecordV1 {
                state: MusicEditEvidenceStateV1::Rejected,
                failure_class: Some(MusicEditEvidenceFailureClassV1::Integrity),
                ..
            }) => {
                hard_rejection = true;
                reasons.push(MusicEditDecisionReasonV1::IntegrityViolation);
            }
            Some(MusicEditEvidenceRecordV1 {
                state: MusicEditEvidenceStateV1::Rejected,
                failure_class: Some(MusicEditEvidenceFailureClassV1::Repairable),
                ..
            }) => {
                repairable = true;
                reasons.push(MusicEditDecisionReasonV1::CandidateRepairable);
            }
            Some(MusicEditEvidenceRecordV1 {
                state: MusicEditEvidenceStateV1::Rejected,
                failure_class: Some(MusicEditEvidenceFailureClassV1::Unresolved),
                ..
            }) => {
                unresolved = true;
                reasons.push(MusicEditDecisionReasonV1::RequiredEvidenceUnresolved);
            }
            Some(MusicEditEvidenceRecordV1 {
                state: MusicEditEvidenceStateV1::Verified,
                ..
            }) => {}
            Some(MusicEditEvidenceRecordV1 {
                state: MusicEditEvidenceStateV1::Rejected,
                failure_class: None,
                ..
            }) => {
                unresolved = true;
                reasons.push(MusicEditDecisionReasonV1::RequiredEvidenceUnresolved);
            }
        }
    }

    let (action, reason) = if hard_rejection {
        (
            MusicEditDecisionActionV1::Reject,
            MusicEditDecisionReasonV1::CandidateHardInvalid,
        )
    } else if repairable && budget_allows_repair(&usage, &transaction.budget) {
        (
            MusicEditDecisionActionV1::Repair,
            MusicEditDecisionReasonV1::CandidateRepairable,
        )
    } else {
        if repairable {
            reasons.push(MusicEditDecisionReasonV1::BudgetExhausted);
        }
        let fallback = if last.outcome == MusicEditAttemptOutcomeV1::OutcomeUnknown {
            MusicEditDecisionReasonV1::OutcomeUnknown
        } else if last.outcome == MusicEditAttemptOutcomeV1::Failed {
            MusicEditDecisionReasonV1::ExecutionFailed
        } else if !reasons.is_empty() {
            reasons[0]
        } else {
            MusicEditDecisionReasonV1::EvidenceMissing
        };
        let _ = unresolved;
        (MusicEditDecisionActionV1::Abstain, fallback)
    };
    reasons.push(reason);

    make_receipt(transaction, action, selected_attempt_id, reasons, usage)
        .map_err(|issue| vec![issue])
}

/// Replays the transaction and verifies the decision payload and its digest.
/// A receipt from another source, candidate, policy, ordering or budget cannot pass.
pub fn verify_music_edit_transaction_decision_v1(
    transaction: &MusicEditTransactionV1,
    receipt: &MusicEditTransactionDecisionReceiptV1,
) -> Result<(), MusicEditTransactionIssueV1> {
    let expected = evaluate_music_edit_transaction_v1(transaction)
        .map_err(|_| MusicEditTransactionIssueV1::InvalidTransaction)?;
    if expected != *receipt {
        return Err(MusicEditTransactionIssueV1::DecisionReceiptMismatch);
    }
    let actual_digest = digest(DECISION_DOMAIN, &receipt.payload)?;
    if actual_digest != receipt.receipt_sha256 {
        return Err(MusicEditTransactionIssueV1::DecisionReceiptMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn required() -> Vec<MusicEditEvidenceKindV1> {
        vec![
            MusicEditEvidenceKindV1::SymbolicObligation,
            MusicEditEvidenceKindV1::RendererExecution,
            MusicEditEvidenceKindV1::TargetAcoustic,
            MusicEditEvidenceKindV1::ProtectedPreservation,
            MusicEditEvidenceKindV1::Metamorphic,
            MusicEditEvidenceKindV1::RightsAdmission,
            MusicEditEvidenceKindV1::ReferenceAdjudication,
        ]
    }

    fn verified(kind: MusicEditEvidenceKindV1, byte: u8) -> MusicEditEvidenceRecordV1 {
        MusicEditEvidenceRecordV1 {
            kind,
            state: MusicEditEvidenceStateV1::Verified,
            source_lineage_sha256: Some(d(1)),
            candidate_artifact_sha256: Some(d(4)),
            receipt_sha256: Some(d(byte)),
            report_sha256: Some(d(byte.wrapping_add(1))),
            replay_verifier_sha256: Some(d(0x55)),
            failure_class: None,
            reason_code: None,
        }
    }

    fn ready_attempt() -> MusicEditCandidateAttemptV1 {
        MusicEditCandidateAttemptV1 {
            attempt_index: 1,
            attempt_id: "attempt-001".to_owned(),
            candidate_id: "candidate-001".to_owned(),
            kind: MusicEditAttemptKindV1::Initial,
            outcome: MusicEditAttemptOutcomeV1::CandidateReady,
            source_lineage_sha256: d(1),
            candidate_symbolic_sha256: Some(d(3)),
            candidate_artifact_sha256: Some(d(4)),
            outcome_reason_code: None,
            render_calls: 1,
            elapsed_ms: 250,
            materialized_bytes: 64,
            evidence: required()
                .into_iter()
                .enumerate()
                .map(|(index, kind)| verified(kind, 10 + index as u8))
                .collect(),
        }
    }

    fn transaction() -> MusicEditTransactionV1 {
        MusicEditTransactionV1 {
            schema_version: MUSIC_EDIT_TRANSACTION_SCHEMA_V1.to_owned(),
            transaction_id: "tx-001".to_owned(),
            source_lineage_sha256: d(1),
            validity_envelope_sha256: d(2),
            policy_sha256: d(8),
            required_evidence: required(),
            budget: MusicEditTransactionBudgetV1 {
                maximum_render_calls: 4,
                maximum_repair_iterations: 2,
                maximum_elapsed_ms: 10_000,
                maximum_materialized_bytes: 1_024,
            },
            attempts: vec![ready_attempt()],
        }
    }

    fn pending(kind: MusicEditEvidenceKindV1) -> MusicEditEvidenceRecordV1 {
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

    #[test]
    fn accepts_only_when_every_required_gate_is_verified_for_exact_candidate() {
        let tx = transaction();
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Accept);
        assert_eq!(
            receipt.payload.selected_attempt_id.as_deref(),
            Some("attempt-001")
        );
        assert!(verify_music_edit_transaction_decision_v1(&tx, &receipt).is_ok());
    }

    #[test]
    fn pending_required_evidence_abstains() {
        let mut tx = transaction();
        let record = tx.attempts[0]
            .evidence
            .iter_mut()
            .find(|record| record.kind == MusicEditEvidenceKindV1::RendererExecution)
            .unwrap();
        *record = pending(MusicEditEvidenceKindV1::RendererExecution);
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Abstain);
        assert!(receipt
            .payload
            .reasons
            .contains(&MusicEditDecisionReasonV1::EvidencePending));
    }

    #[test]
    fn candidate_evidence_from_another_output_rejects() {
        let mut tx = transaction();
        tx.attempts[0].evidence[0].candidate_artifact_sha256 = Some(d(0x99));
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Reject);
        assert!(receipt
            .payload
            .reasons
            .contains(&MusicEditDecisionReasonV1::IntegrityViolation));
    }

    #[test]
    fn protected_content_hard_failure_rejects() {
        let mut tx = transaction();
        let record = tx.attempts[0]
            .evidence
            .iter_mut()
            .find(|record| record.kind == MusicEditEvidenceKindV1::ProtectedPreservation)
            .unwrap();
        record.state = MusicEditEvidenceStateV1::Rejected;
        record.failure_class = Some(MusicEditEvidenceFailureClassV1::HardGate);
        record.reason_code = Some("protected-event-changed".to_owned());
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Reject);
    }

    #[test]
    fn repairable_failure_repairs_only_with_remaining_budget() {
        let mut tx = transaction();
        tx.attempts[0].outcome = MusicEditAttemptOutcomeV1::RepairableFailure;
        tx.attempts[0].outcome_reason_code = Some("local-artifact-repairable".to_owned());
        tx.attempts[0].candidate_artifact_sha256 = None;
        tx.attempts[0].candidate_symbolic_sha256 = None;
        tx.attempts[0].evidence.clear();
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Repair);
    }

    #[test]
    fn exhausted_render_budget_turns_repair_into_abstention() {
        let mut tx = transaction();
        tx.budget.maximum_render_calls = 1;
        tx.attempts[0].outcome = MusicEditAttemptOutcomeV1::RepairableFailure;
        tx.attempts[0].outcome_reason_code = Some("render-artifact-repairable".to_owned());
        tx.attempts[0].candidate_artifact_sha256 = None;
        tx.attempts[0].candidate_symbolic_sha256 = None;
        tx.attempts[0].evidence.clear();
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Abstain);
        assert!(receipt
            .payload
            .reasons
            .contains(&MusicEditDecisionReasonV1::BudgetExhausted));
    }

    #[test]
    fn unknown_outcome_abstains_and_blocks_later_attempts() {
        let mut tx = transaction();
        tx.attempts[0].outcome = MusicEditAttemptOutcomeV1::OutcomeUnknown;
        tx.attempts[0].outcome_reason_code = Some("provider-timeout-after-submit".to_owned());
        tx.attempts[0].candidate_artifact_sha256 = None;
        tx.attempts[0].candidate_symbolic_sha256 = None;
        tx.attempts[0].evidence.clear();
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Abstain);

        let mut later = ready_attempt();
        later.attempt_index = 2;
        later.attempt_id = "attempt-002".to_owned();
        later.candidate_id = "candidate-002".to_owned();
        tx.attempts.push(later);
        assert!(validate_music_edit_transaction_v1(&tx)
            .contains(&MusicEditTransactionIssueV1::UnknownOutcomeNotTerminal(
                "attempt-001".to_owned()
            )));
    }

    #[test]
    fn failed_render_calls_remain_in_consumed_budget() {
        let mut tx = transaction();
        tx.attempts[0].outcome = MusicEditAttemptOutcomeV1::Failed;
        tx.attempts[0].outcome_reason_code = Some("renderer-process-failed".to_owned());
        tx.attempts[0].candidate_artifact_sha256 = None;
        tx.attempts[0].candidate_symbolic_sha256 = None;
        tx.attempts[0].evidence.clear();
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Abstain);
        assert_eq!(receipt.payload.usage.render_calls, 1);
    }

    #[test]
    fn over_budget_ledger_is_rejected_instead_of_hidden() {
        let mut tx = transaction();
        tx.budget.maximum_render_calls = 0;
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Reject);
        assert!(receipt
            .payload
            .reasons
            .contains(&MusicEditDecisionReasonV1::BudgetExceeded));
    }

    #[test]
    fn duplicate_candidate_identity_is_a_structural_error() {
        let mut tx = transaction();
        let mut second = ready_attempt();
        second.attempt_index = 2;
        second.attempt_id = "attempt-002".to_owned();
        tx.attempts.push(second);
        assert!(validate_music_edit_transaction_v1(&tx).contains(
            &MusicEditTransactionIssueV1::EmptyOrDuplicateCandidateId("candidate-001".to_owned())
        ));
    }

    #[test]
    fn malformed_or_reordered_required_evidence_is_rejected() {
        let mut tx = transaction();
        tx.required_evidence.reverse();
        assert!(validate_music_edit_transaction_v1(&tx)
            .contains(&MusicEditTransactionIssueV1::NonCanonicalRequiredEvidence));
    }

    #[test]
    fn budget_arithmetic_overflow_is_reported() {
        let mut tx = transaction();
        tx.attempts[0].render_calls = u64::MAX;
        let mut second = ready_attempt();
        second.attempt_index = 2;
        second.attempt_id = "attempt-002".to_owned();
        second.candidate_id = "candidate-002".to_owned();
        second.render_calls = 1;
        tx.attempts.push(second);
        assert!(validate_music_edit_transaction_v1(&tx)
            .contains(&MusicEditTransactionIssueV1::BudgetUsageOverflow));
    }

    #[test]
    fn tampered_decision_receipt_fails_replay() {
        let tx = transaction();
        let mut receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        receipt.payload.action = MusicEditDecisionActionV1::Repair;
        assert_eq!(
            verify_music_edit_transaction_decision_v1(&tx, &receipt),
            Err(MusicEditTransactionIssueV1::DecisionReceiptMismatch)
        );
    }

    #[test]
    fn required_rights_evidence_cannot_be_omitted() {
        let mut tx = transaction();
        tx.attempts[0]
            .evidence
            .retain(|record| record.kind != MusicEditEvidenceKindV1::RightsAdmission);
        let receipt = evaluate_music_edit_transaction_v1(&tx).unwrap();
        assert_eq!(receipt.payload.action, MusicEditDecisionActionV1::Abstain);
        assert!(receipt
            .payload
            .reasons
            .contains(&MusicEditDecisionReasonV1::EvidenceMissing));
    }
}
