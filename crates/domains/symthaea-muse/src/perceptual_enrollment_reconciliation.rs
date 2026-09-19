// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1ENR-B: reconcile every allocated enrollment slot against
//! independently witnessed enrollment and authenticated P1E/P1EWR evidence.
//!
//! This module intentionally stops short of manufacturing a complete
//! `FrozenPerceptualEnrollmentPartitionV1`. Historical P1E authenticates the
//! screening-failure count, but it contains no pre-enrollment-withdrawal count.
//! That missing recruitment-side quantity belongs to P1RIR. Here we establish
//! the stronger part already supported by evidence: every consumed enrollment
//! slot has exactly one terminal study-data disposition and historical P1E's
//! ambiguous `unused_enrollment_slots` is decomposed into truly unallocated
//! capacity plus `EnrolledNoScoredSession`.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_collection_authenticity::{
        participant_token_commitment, validate_collection_authenticity_bundle,
        FrozenPerceptualCollectionAuthenticityBundleV1,
        FrozenPerceptualCollectionAuthenticityPolicyV1, FrozenPerceptualWithdrawalPolicyV1,
    },
    perceptual_collection_evidence::{
        FrozenPerceptualCollectionAuthorityV1, PerceptualCollectionCloseV1,
        RawPerceptualCollectionV1, SessionStatusV1,
    },
    perceptual_enrollment_lifecycle::{
        participant_allocation_reference_commitment, validate_enrollment_allocation_ledger,
        EnrolledParticipantDispositionV1, FrozenPerceptualEnrollmentPolicyV1,
    },
    perceptual_enrollment_store::{state_commitment, DurableEnrollmentAllocationStateV1},
    perceptual_enrollment_witness::{
        validate_enrollment_witness_bundle, FrozenPerceptualEnrollmentWitnessBundleV1,
        FrozenPerceptualEnrollmentWitnessPolicyV1,
    },
    perceptual_participant_identity::{
        FrozenParticipantIdentityBoundaryPolicyV1,
        FrozenPerceptualParticipantTokenGenerationReceiptV1,
    },
    perceptual_participant_schedule::{
        PerceptualCohortSlotsV1, PerceptualParticipantScheduleBookV1,
    },
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ALLOCATED_ENROLLMENT_RECONCILIATION_VERSION: &str =
    "mel003-allocated-enrollment-reconciliation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocatedEnrollmentDispositionEvidenceKindV1 {
    RetainedCompleteSession,
    RetainedAbortedSession,
    AuthenticatedWithdrawalTombstone,
    NoAuthenticatedScoredArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllocatedEnrollmentDispositionRecordV1 {
    pub sequence: u32,
    /// P1ENR-domain participant allocation commitment. The distinct P1EIR
    /// participant commitment is derived transiently during validation and is
    /// not copied into this cross-domain reconciliation artifact.
    pub enrollment_participant_commitment_sha256: String,
    pub enrollment_witness_receipt_sha256: String,
    pub disposition: EnrolledParticipantDispositionV1,
    pub evidence_kind: AllocatedEnrollmentDispositionEvidenceKindV1,
    /// Session SHA-256 or withdrawal-tombstone SHA-256 when such an artifact
    /// exists. `None` for `EnrolledNoScoredSession`, where the evidence is the
    /// complete validated absence of a session/tombstone in the authenticated
    /// collection bundle.
    pub source_artifact_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenAllocatedEnrollmentReconciliationV1 {
    pub reconciliation_version: String,
    pub enrollment_policy_sha256: String,
    pub enrollment_allocation_ledger_sha256: String,
    pub enrollment_witness_bundle_sha256: String,
    pub collection_authenticity_bundle_sha256: String,
    pub collection_close_sha256: String,
    /// Allocation order is evidence. Do not sort this vector.
    pub allocated_dispositions: Vec<AllocatedEnrollmentDispositionRecordV1>,
    /// Capacity never consumed by P1ENR allocation.
    pub unused_unallocated_slots: usize,
    /// Authenticated historical P1E descriptive quantity. It remains outside
    /// the registered enrolled-cap equation.
    pub historical_screening_failure_count: usize,
    /// Historical P1E's ambiguous value. Validation requires this to equal
    /// `unused_unallocated_slots + EnrolledNoScoredSession count`.
    pub historical_unused_enrollment_slots: usize,
    pub reconciliation_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocatedEnrollmentReconciliationIssueV1 {
    WrongReconciliationVersion,
    InvalidEnrollmentLedger,
    InvalidEnrollmentStateDigest,
    EnrollmentStateIdentityMismatch { field: String },
    InvalidEnrollmentWitnessBundle,
    InvalidCollectionAuthenticityBundle,
    EnrollmentPolicyMismatch,
    EnrollmentLedgerMismatch,
    EnrollmentWitnessBundleMismatch,
    CollectionAuthenticityBundleMismatch,
    CollectionCloseMismatch,
    MissingAllocatedToken { index: usize },
    EnrollmentParticipantCommitmentMismatch { index: usize },
    ParticipantCommitmentDerivationFailed { index: usize },
    MissingEnrollmentWitness { index: usize },
    EnrollmentWitnessMismatch { index: usize },
    DuplicateCollectionParticipant { commitment: String },
    UnexpectedRetainedSession { commitment: String },
    UnexpectedWithdrawalTombstone { commitment: String },
    DispositionMismatch { index: usize },
    WrongDispositionCount { found: usize, expected: usize },
    CompleteCountMismatch { found: usize, expected: usize },
    AbortedCountMismatch { found: usize, expected: usize },
    WithdrawalCountMismatch { found: usize, expected: usize },
    UnusedUnallocatedMismatch { found: usize, expected: usize },
    HistoricalUnusedDecompositionMismatch { found: usize, expected: usize },
    ScreeningFailureCountMismatch { found: usize, expected: usize },
    ReconciliationDigestMismatch,
    SerializationFailed,
}

#[derive(Debug)]
struct DerivedReconciliationV1 {
    records: Vec<AllocatedEnrollmentDispositionRecordV1>,
    unused_unallocated_slots: usize,
    allocated_collection_commitments: BTreeSet<String>,
}

#[allow(clippy::too_many_arguments)]
pub fn reconcile_allocated_enrollment_dispositions(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    enrollment_state: &DurableEnrollmentAllocationStateV1,
    enrollment_witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    enrollment_witness_bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
    collection_authority: &FrozenPerceptualCollectionAuthorityV1,
    collection: &RawPerceptualCollectionV1,
    close: &PerceptualCollectionCloseV1,
    withdrawal_policy: &FrozenPerceptualWithdrawalPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    authenticity_bundle: &FrozenPerceptualCollectionAuthenticityBundleV1,
) -> Result<FrozenAllocatedEnrollmentReconciliationV1, Vec<AllocatedEnrollmentReconciliationIssueV1>> {
    let mut issues = validate_prerequisites(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        enrollment_state,
        enrollment_witness_policy,
        enrollment_witness_bundle,
        collection_authority,
        collection,
        close,
        withdrawal_policy,
        authenticity_policy,
        authenticity_bundle,
    );
    let derived = derive_records(
        protocol,
        cohort,
        enrollment_state,
        enrollment_witness_bundle,
        collection,
        authenticity_bundle,
        &mut issues,
    );
    issues.extend(validate_cross_collection_membership(
        collection,
        authenticity_bundle,
        &derived.allocated_collection_commitments,
    ));
    issues.extend(validate_close_accounting(close, &derived));
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut result = FrozenAllocatedEnrollmentReconciliationV1 {
        reconciliation_version: ALLOCATED_ENROLLMENT_RECONCILIATION_VERSION.into(),
        enrollment_policy_sha256: enrollment_policy.policy_sha256.clone(),
        enrollment_allocation_ledger_sha256: enrollment_state.ledger.ledger_sha256.clone(),
        enrollment_witness_bundle_sha256: enrollment_witness_bundle.bundle_sha256.clone(),
        collection_authenticity_bundle_sha256: authenticity_bundle.bundle_sha256.clone(),
        collection_close_sha256: close.close_sha256.clone(),
        allocated_dispositions: derived.records,
        unused_unallocated_slots: derived.unused_unallocated_slots,
        historical_screening_failure_count: close.screening_failure_count,
        historical_unused_enrollment_slots: close.unused_enrollment_slots,
        reconciliation_sha256: String::new(),
    };
    result.reconciliation_sha256 = allocated_enrollment_reconciliation_commitment(&result)
        .map_err(|_| vec![AllocatedEnrollmentReconciliationIssueV1::SerializationFailed])?;
    let validation = validate_allocated_enrollment_reconciliation(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        enrollment_state,
        enrollment_witness_policy,
        enrollment_witness_bundle,
        collection_authority,
        collection,
        close,
        withdrawal_policy,
        authenticity_policy,
        authenticity_bundle,
        &result,
    );
    if validation.is_empty() {
        Ok(result)
    } else {
        Err(validation)
    }
}

pub fn allocated_enrollment_reconciliation_commitment(
    reconciliation: &FrozenAllocatedEnrollmentReconciliationV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = reconciliation.clone();
    unsigned.reconciliation_sha256.clear();
    canonical_json_sha256(&unsigned)
}

#[allow(clippy::too_many_arguments)]
pub fn validate_allocated_enrollment_reconciliation(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    enrollment_state: &DurableEnrollmentAllocationStateV1,
    enrollment_witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    enrollment_witness_bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
    collection_authority: &FrozenPerceptualCollectionAuthorityV1,
    collection: &RawPerceptualCollectionV1,
    close: &PerceptualCollectionCloseV1,
    withdrawal_policy: &FrozenPerceptualWithdrawalPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    authenticity_bundle: &FrozenPerceptualCollectionAuthenticityBundleV1,
    reconciliation: &FrozenAllocatedEnrollmentReconciliationV1,
) -> Vec<AllocatedEnrollmentReconciliationIssueV1> {
    let mut issues = validate_prerequisites(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        enrollment_state,
        enrollment_witness_policy,
        enrollment_witness_bundle,
        collection_authority,
        collection,
        close,
        withdrawal_policy,
        authenticity_policy,
        authenticity_bundle,
    );

    if reconciliation.reconciliation_version != ALLOCATED_ENROLLMENT_RECONCILIATION_VERSION {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::WrongReconciliationVersion);
    }
    if reconciliation.enrollment_policy_sha256 != enrollment_policy.policy_sha256 {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::EnrollmentPolicyMismatch);
    }
    if reconciliation.enrollment_allocation_ledger_sha256
        != enrollment_state.ledger.ledger_sha256
    {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::EnrollmentLedgerMismatch);
    }
    if reconciliation.enrollment_witness_bundle_sha256 != enrollment_witness_bundle.bundle_sha256 {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::EnrollmentWitnessBundleMismatch);
    }
    if reconciliation.collection_authenticity_bundle_sha256 != authenticity_bundle.bundle_sha256 {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::CollectionAuthenticityBundleMismatch);
    }
    if reconciliation.collection_close_sha256 != close.close_sha256 {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::CollectionCloseMismatch);
    }

    let derived = derive_records(
        protocol,
        cohort,
        enrollment_state,
        enrollment_witness_bundle,
        collection,
        authenticity_bundle,
        &mut issues,
    );
    if reconciliation.allocated_dispositions.len() != derived.records.len() {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::WrongDispositionCount {
            found: reconciliation.allocated_dispositions.len(),
            expected: derived.records.len(),
        });
    }
    for (index, expected) in derived.records.iter().enumerate() {
        match reconciliation.allocated_dispositions.get(index) {
            Some(found) if found == expected => {}
            _ => issues.push(AllocatedEnrollmentReconciliationIssueV1::DispositionMismatch {
                index,
            }),
        }
    }
    issues.extend(validate_cross_collection_membership(
        collection,
        authenticity_bundle,
        &derived.allocated_collection_commitments,
    ));
    issues.extend(validate_close_accounting(close, &derived));

    if reconciliation.unused_unallocated_slots != derived.unused_unallocated_slots {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::UnusedUnallocatedMismatch {
            found: reconciliation.unused_unallocated_slots,
            expected: derived.unused_unallocated_slots,
        });
    }
    if reconciliation.historical_screening_failure_count != close.screening_failure_count {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::ScreeningFailureCountMismatch {
            found: reconciliation.historical_screening_failure_count,
            expected: close.screening_failure_count,
        });
    }
    if reconciliation.historical_unused_enrollment_slots != close.unused_enrollment_slots {
        issues.push(
            AllocatedEnrollmentReconciliationIssueV1::HistoricalUnusedDecompositionMismatch {
                found: reconciliation.historical_unused_enrollment_slots,
                expected: close.unused_enrollment_slots,
            },
        );
    }
    match allocated_enrollment_reconciliation_commitment(reconciliation) {
        Ok(value) if value == reconciliation.reconciliation_sha256 => {}
        Ok(_) => issues.push(AllocatedEnrollmentReconciliationIssueV1::ReconciliationDigestMismatch),
        Err(_) => issues.push(AllocatedEnrollmentReconciliationIssueV1::SerializationFailed),
    }
    issues
}

#[allow(clippy::too_many_arguments)]
fn validate_prerequisites(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    enrollment_state: &DurableEnrollmentAllocationStateV1,
    enrollment_witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    enrollment_witness_bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
    collection_authority: &FrozenPerceptualCollectionAuthorityV1,
    collection: &RawPerceptualCollectionV1,
    close: &PerceptualCollectionCloseV1,
    withdrawal_policy: &FrozenPerceptualWithdrawalPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    authenticity_bundle: &FrozenPerceptualCollectionAuthenticityBundleV1,
) -> Vec<AllocatedEnrollmentReconciliationIssueV1> {
    let mut issues = Vec::new();
    if state_commitment(enrollment_state)
        .map(|value| value != enrollment_state.state_sha256)
        .unwrap_or(true)
    {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::InvalidEnrollmentStateDigest);
    }
    if enrollment_state.enrollment_policy_sha256 != enrollment_policy.policy_sha256 {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::EnrollmentStateIdentityMismatch {
            field: "enrollment_policy_sha256".into(),
        });
    }
    if enrollment_state.token_generation_receipt_sha256 != token_receipt.receipt_sha256 {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::EnrollmentStateIdentityMismatch {
            field: "token_generation_receipt_sha256".into(),
        });
    }
    match canonical_json_sha256(schedule) {
        Ok(value) if value == enrollment_state.participant_schedule_sha256 => {}
        _ => issues.push(AllocatedEnrollmentReconciliationIssueV1::EnrollmentStateIdentityMismatch {
            field: "participant_schedule_sha256".into(),
        }),
    }
    if !validate_enrollment_allocation_ledger(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        &enrollment_state.eligibility_gates,
        &enrollment_state.ledger,
    )
    .is_empty()
    {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::InvalidEnrollmentLedger);
    }
    if !validate_enrollment_witness_bundle(
        enrollment_policy,
        authenticity_policy,
        enrollment_witness_policy,
        &enrollment_state.ledger,
        enrollment_witness_bundle,
    )
    .is_empty()
    {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::InvalidEnrollmentWitnessBundle);
    }
    if !validate_collection_authenticity_bundle(
        protocol,
        stimulus_pack,
        render_binding,
        schedule,
        collection_authority,
        collection,
        close,
        withdrawal_policy,
        authenticity_policy,
        authenticity_bundle,
    )
    .is_empty()
    {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::InvalidCollectionAuthenticityBundle);
    }
    issues
}

fn derive_records(
    protocol: &FrozenPerceptualStudyProtocolV1,
    cohort: &PerceptualCohortSlotsV1,
    enrollment_state: &DurableEnrollmentAllocationStateV1,
    enrollment_witness_bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
    collection: &RawPerceptualCollectionV1,
    authenticity_bundle: &FrozenPerceptualCollectionAuthenticityBundleV1,
    issues: &mut Vec<AllocatedEnrollmentReconciliationIssueV1>,
) -> DerivedReconciliationV1 {
    let mut sessions = BTreeMap::new();
    for session in &collection.sessions {
        match participant_token_commitment(&session.participant_token) {
            Ok(commitment) => {
                if sessions.insert(commitment.clone(), session).is_some() {
                    issues.push(
                        AllocatedEnrollmentReconciliationIssueV1::DuplicateCollectionParticipant {
                            commitment,
                        },
                    );
                }
            }
            Err(_) => issues.push(
                AllocatedEnrollmentReconciliationIssueV1::ParticipantCommitmentDerivationFailed {
                    index: usize::MAX,
                },
            ),
        }
    }
    let mut tombstones = BTreeMap::new();
    for tombstone in &authenticity_bundle.withdrawal_tombstones {
        if tombstones
            .insert(
                tombstone.participant_token_commitment_sha256.clone(),
                tombstone,
            )
            .is_some()
        {
            issues.push(
                AllocatedEnrollmentReconciliationIssueV1::DuplicateCollectionParticipant {
                    commitment: tombstone.participant_token_commitment_sha256.clone(),
                },
            );
        }
    }

    let mut canonical_tokens = cohort.participant_tokens.clone();
    canonical_tokens.sort();
    let mut records = Vec::with_capacity(enrollment_state.ledger.allocations.len());
    let mut allocated_collection_commitments = BTreeSet::new();

    for (index, allocation) in enrollment_state.ledger.allocations.iter().enumerate() {
        let Some(token) = canonical_tokens.get(index) else {
            issues.push(AllocatedEnrollmentReconciliationIssueV1::MissingAllocatedToken { index });
            continue;
        };
        match participant_allocation_reference_commitment(token) {
            Ok(commitment) if commitment == allocation.participant_token_commitment_sha256 => {}
            _ => issues.push(
                AllocatedEnrollmentReconciliationIssueV1::EnrollmentParticipantCommitmentMismatch {
                    index,
                },
            ),
        }
        let collection_commitment = match participant_token_commitment(token) {
            Ok(value) => value,
            Err(_) => {
                issues.push(
                    AllocatedEnrollmentReconciliationIssueV1::ParticipantCommitmentDerivationFailed {
                        index,
                    },
                );
                continue;
            }
        };
        allocated_collection_commitments.insert(collection_commitment.clone());

        let witness = enrollment_witness_bundle.entries.get(index);
        let witness_receipt_sha256 = match witness {
            Some(entry)
                if entry.sequence == index as u32
                    && entry.enrollment_participant_commitment_sha256
                        == allocation.participant_token_commitment_sha256 =>
            {
                entry.receipt_sha256.clone()
            }
            Some(_) => {
                issues.push(AllocatedEnrollmentReconciliationIssueV1::EnrollmentWitnessMismatch {
                    index,
                });
                String::new()
            }
            None => {
                issues.push(AllocatedEnrollmentReconciliationIssueV1::MissingEnrollmentWitness {
                    index,
                });
                String::new()
            }
        };

        let (disposition, evidence_kind, source_artifact_sha256) =
            if let Some(tombstone) = tombstones.get(&collection_commitment) {
                (
                    EnrolledParticipantDispositionV1::WithdrawnAndDeleted,
                    AllocatedEnrollmentDispositionEvidenceKindV1::AuthenticatedWithdrawalTombstone,
                    Some(tombstone.tombstone_sha256.clone()),
                )
            } else if let Some(session) = sessions.get(&collection_commitment) {
                match session.status {
                    SessionStatusV1::Complete => (
                        EnrolledParticipantDispositionV1::CompleteSession,
                        AllocatedEnrollmentDispositionEvidenceKindV1::RetainedCompleteSession,
                        Some(session.session_sha256.clone()),
                    ),
                    SessionStatusV1::Aborted(_) => (
                        EnrolledParticipantDispositionV1::AbortedSession,
                        AllocatedEnrollmentDispositionEvidenceKindV1::RetainedAbortedSession,
                        Some(session.session_sha256.clone()),
                    ),
                }
            } else {
                (
                    EnrolledParticipantDispositionV1::EnrolledNoScoredSession,
                    AllocatedEnrollmentDispositionEvidenceKindV1::NoAuthenticatedScoredArtifact,
                    None,
                )
            };

        records.push(AllocatedEnrollmentDispositionRecordV1 {
            sequence: index as u32,
            enrollment_participant_commitment_sha256: allocation
                .participant_token_commitment_sha256
                .clone(),
            enrollment_witness_receipt_sha256: witness_receipt_sha256,
            disposition,
            evidence_kind,
            source_artifact_sha256,
        });
    }

    let unused_unallocated_slots = protocol
        .sample_size
        .maximum_enrolled_participants
        .saturating_sub(enrollment_state.ledger.allocations.len());
    DerivedReconciliationV1 {
        records,
        unused_unallocated_slots,
        allocated_collection_commitments,
    }
}

fn validate_cross_collection_membership(
    collection: &RawPerceptualCollectionV1,
    authenticity_bundle: &FrozenPerceptualCollectionAuthenticityBundleV1,
    allocated_collection_commitments: &BTreeSet<String>,
) -> Vec<AllocatedEnrollmentReconciliationIssueV1> {
    let mut issues = Vec::new();
    for session in &collection.sessions {
        if let Ok(commitment) = participant_token_commitment(&session.participant_token) {
            if !allocated_collection_commitments.contains(&commitment) {
                issues.push(AllocatedEnrollmentReconciliationIssueV1::UnexpectedRetainedSession {
                    commitment,
                });
            }
        }
    }
    for tombstone in &authenticity_bundle.withdrawal_tombstones {
        if !allocated_collection_commitments.contains(&tombstone.participant_token_commitment_sha256)
        {
            issues.push(
                AllocatedEnrollmentReconciliationIssueV1::UnexpectedWithdrawalTombstone {
                    commitment: tombstone.participant_token_commitment_sha256.clone(),
                },
            );
        }
    }
    issues
}

fn validate_close_accounting(
    close: &PerceptualCollectionCloseV1,
    derived: &DerivedReconciliationV1,
) -> Vec<AllocatedEnrollmentReconciliationIssueV1> {
    let mut issues = Vec::new();
    let complete = derived
        .records
        .iter()
        .filter(|record| record.disposition == EnrolledParticipantDispositionV1::CompleteSession)
        .count();
    let aborted = derived
        .records
        .iter()
        .filter(|record| record.disposition == EnrolledParticipantDispositionV1::AbortedSession)
        .count();
    let withdrawn = derived
        .records
        .iter()
        .filter(|record| {
            record.disposition == EnrolledParticipantDispositionV1::WithdrawnAndDeleted
        })
        .count();
    let no_scored = derived
        .records
        .iter()
        .filter(|record| {
            record.disposition == EnrolledParticipantDispositionV1::EnrolledNoScoredSession
        })
        .count();

    if complete != close.completed_sessions {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::CompleteCountMismatch {
            found: complete,
            expected: close.completed_sessions,
        });
    }
    if aborted != close.aborted_sessions {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::AbortedCountMismatch {
            found: aborted,
            expected: close.aborted_sessions,
        });
    }
    if withdrawn != close.withdrawn_and_deleted_sessions {
        issues.push(AllocatedEnrollmentReconciliationIssueV1::WithdrawalCountMismatch {
            found: withdrawn,
            expected: close.withdrawn_and_deleted_sessions,
        });
    }
    let expected_historical_unused = derived.unused_unallocated_slots.saturating_add(no_scored);
    if close.unused_enrollment_slots != expected_historical_unused {
        issues.push(
            AllocatedEnrollmentReconciliationIssueV1::HistoricalUnusedDecompositionMismatch {
                found: close.unused_enrollment_slots,
                expected: expected_historical_unused,
            },
        );
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_unused_decomposes_into_unallocated_plus_enrolled_no_session() {
        let derived = DerivedReconciliationV1 {
            records: vec![
                AllocatedEnrollmentDispositionRecordV1 {
                    sequence: 0,
                    enrollment_participant_commitment_sha256: "a".repeat(64),
                    enrollment_witness_receipt_sha256: "b".repeat(64),
                    disposition: EnrolledParticipantDispositionV1::EnrolledNoScoredSession,
                    evidence_kind:
                        AllocatedEnrollmentDispositionEvidenceKindV1::NoAuthenticatedScoredArtifact,
                    source_artifact_sha256: None,
                },
                AllocatedEnrollmentDispositionRecordV1 {
                    sequence: 1,
                    enrollment_participant_commitment_sha256: "c".repeat(64),
                    enrollment_witness_receipt_sha256: "d".repeat(64),
                    disposition: EnrolledParticipantDispositionV1::CompleteSession,
                    evidence_kind:
                        AllocatedEnrollmentDispositionEvidenceKindV1::RetainedCompleteSession,
                    source_artifact_sha256: Some("e".repeat(64)),
                },
            ],
            unused_unallocated_slots: 3,
            allocated_collection_commitments: BTreeSet::new(),
        };
        let close = PerceptualCollectionCloseV1 {
            close_version: "test".into(),
            collection_authority_sha256: "1".repeat(64),
            raw_dataset_sha256: "2".repeat(64),
            participant_schedule_sha256: "3".repeat(64),
            close_reason: crate::evidence_digest::perceptual_collection_evidence::PerceptualCollectionCloseReasonV1::FrozenDeadlineReached,
            completed_sessions: 1,
            aborted_sessions: 0,
            withdrawn_and_deleted_sessions: 0,
            unused_enrollment_slots: 4,
            screening_failure_count: 0,
            closed_at_utc: "2026-09-19T12:00:00Z".into(),
            outcome_monitoring_performed_before_close: false,
            private_audit_accessed_before_close: false,
            randomization_key_revealed_before_close: false,
            close_sha256: "4".repeat(64),
        };
        assert!(validate_close_accounting(&close, &derived).is_empty());
    }
}
