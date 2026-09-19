// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1EILR-C: crash-safe coordinator state between durable allocation
//! and independent enrollment witnessing.
//!
//! This module does not replace either P1ENR allocation durability or P1EIR-
//! bound enrollment witnessing. It linearizes their handoff. The coordinator
//! persists only exact confirmed ledger/witness evidence and may observe at
//! most one durable-but-not-yet-witnessed allocation during recovery.
//!
//! A caller performing the production enrollment transition must hold the
//! coordinator mutation lock across:
//!
//! ```text
//! inspect/reconcile -> allocate or recover pending -> witness -> confirm
//! ```
//!
//! so another cooperating process cannot allocate N+1 while N remains pending.
//! The low-level allocator/witness APIs remain available for qualification and
//! explicit recovery, but are not equivalent to this orchestration theorem.

use crate::evidence_digest::{
    canonical_json_bytes, canonical_json_sha256,
    perceptual_collection_authenticity::FrozenPerceptualCollectionAuthenticityPolicyV1,
    perceptual_enrollment_lifecycle::{
        enrollment_allocation_ledger_commitment, validate_enrollment_allocation_ledger,
        FrozenPerceptualEnrollmentAllocationLedgerV1, FrozenPerceptualEnrollmentPolicyV1,
        ZERO_SHA256,
    },
    perceptual_enrollment_store::{state_commitment, DurableEnrollmentAllocationStateV1},
    perceptual_enrollment_witness::{
        validate_enrollment_witness_bundle, FrozenPerceptualEnrollmentWitnessBundleV1,
        FrozenPerceptualEnrollmentWitnessPolicyV1, VerifiedWitnessedEnrollmentAllocationV1,
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
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const PERCEPTUAL_ENROLLMENT_COORDINATOR_STATE_VERSION: &str =
    "mel003-perceptual-enrollment-coordinator-state-v1";
const COORDINATOR_STATE_FILE_NAME: &str = "mel003-enrollment-witness-coordinator.state.json";
const COORDINATOR_LOCK_FILE_NAME: &str = ".mel003-enrollment-witness-coordinator.lock";
const MAX_COORDINATOR_STATE_BYTES: u64 = 32 * 1024 * 1024;
const PENDING_WITNESS_REQUEST_DOMAIN: &str =
    "symthaea.mel003.p1.enrollment-witness-coordinator.v1/pending-request";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurablePerceptualEnrollmentCoordinatorStateV1 {
    pub state_version: String,
    pub enrollment_policy_sha256: String,
    pub enrollment_witness_policy_sha256: String,
    /// Exact last allocation ledger for which every allocation has a verified
    /// witness receipt. The ledger contains only commitment-level participant
    /// references, never raw participant pseudonyms.
    pub confirmed_enrollment_ledger: FrozenPerceptualEnrollmentAllocationLedgerV1,
    /// Exact witnessed prefix corresponding one-for-one with the confirmed
    /// enrollment ledger.
    pub confirmed_witness_bundle: FrozenPerceptualEnrollmentWitnessBundleV1,
    pub coordinator_state_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingEnrollmentWitnessRequestV1 {
    pub sequence: u32,
    pub enrollment_witness_policy_sha256: String,
    pub witness_log_id: String,
    pub enrollment_ledger_sha256: String,
    pub durable_state_sha256: String,
    pub allocation_head_sha256: String,
    pub enrollment_participant_commitment_sha256: String,
    pub participant_schedule_projection_sha256: String,
    pub previous_witness_head_sha256: String,
    /// Stable retry identity. A production witness adapter should use this as
    /// the idempotency identity for an uncertain/retried external witness call.
    pub request_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnrollmentCoordinatorRecoveryDispositionV1 {
    Synchronized,
    ExactlyOnePending(PendingEnrollmentWitnessRequestV1),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualEnrollmentCoordinatorIssueV1 {
    WrongStateVersion,
    EnrollmentPolicyMismatch,
    EnrollmentWitnessPolicyMismatch,
    InvalidCoordinatorStateDigest,
    CoordinatorStateDigestMismatch,
    InvalidCurrentDurableState,
    CurrentStateIdentityMismatch { field: String },
    InvalidCurrentEnrollmentLedger,
    ConfirmedGateCardinalityMismatch,
    InvalidConfirmedEnrollmentLedger,
    InvalidConfirmedWitnessBundle,
    ConfirmedPrefixMismatch,
    LocalAllocationRollback { confirmed: usize, current: usize },
    ImpossibleAllocationGap { confirmed: usize, current: usize },
    SynchronizedLedgerMismatch,
    MissingPendingAllocation,
    PendingRequestSerializationFailed,
    TransitionRequiresPendingAllocation,
    WitnessBundleDoesNotExtendConfirmedPrefix,
    VerifiedWitnessReceiptMismatch,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PendingWitnessRequestCommitmentV1<'a> {
    domain: &'static str,
    sequence: u32,
    enrollment_witness_policy_sha256: &'a str,
    witness_log_id: &'a str,
    enrollment_ledger_sha256: &'a str,
    durable_state_sha256: &'a str,
    allocation_head_sha256: &'a str,
    enrollment_participant_commitment_sha256: &'a str,
    participant_schedule_projection_sha256: &'a str,
    previous_witness_head_sha256: &'a str,
}

pub fn coordinator_state_commitment(
    state: &DurablePerceptualEnrollmentCoordinatorStateV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = state.clone();
    unsigned.coordinator_state_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn seal_coordinator_state(
    state: &mut DurablePerceptualEnrollmentCoordinatorStateV1,
) -> Result<(), serde_json::Error> {
    state.coordinator_state_sha256 = coordinator_state_commitment(state)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn new_confirmed_enrollment_coordinator_state(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    current_state: &DurableEnrollmentAllocationStateV1,
    confirmed_witness_bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
) -> Result<DurablePerceptualEnrollmentCoordinatorStateV1, Vec<PerceptualEnrollmentCoordinatorIssueV1>> {
    let mut candidate = DurablePerceptualEnrollmentCoordinatorStateV1 {
        state_version: PERCEPTUAL_ENROLLMENT_COORDINATOR_STATE_VERSION.into(),
        enrollment_policy_sha256: enrollment_policy.policy_sha256.clone(),
        enrollment_witness_policy_sha256: witness_policy.policy_sha256.clone(),
        confirmed_enrollment_ledger: current_state.ledger.clone(),
        confirmed_witness_bundle: confirmed_witness_bundle.clone(),
        coordinator_state_sha256: String::new(),
    };
    seal_coordinator_state(&mut candidate)
        .map_err(|_| vec![PerceptualEnrollmentCoordinatorIssueV1::SerializationFailed])?;
    let issues = validate_enrollment_coordinator_state(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        witness_policy,
        authenticity_policy,
        current_state,
        &candidate,
    );
    if issues.is_empty() {
        Ok(candidate)
    } else {
        Err(issues)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn classify_enrollment_coordinator_recovery(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    current_state: &DurableEnrollmentAllocationStateV1,
    coordinator: &DurablePerceptualEnrollmentCoordinatorStateV1,
) -> Result<EnrollmentCoordinatorRecoveryDispositionV1, Vec<PerceptualEnrollmentCoordinatorIssueV1>> {
    let issues = validate_enrollment_coordinator_state(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        witness_policy,
        authenticity_policy,
        current_state,
        coordinator,
    );
    if !issues.is_empty() {
        return Err(issues);
    }

    let confirmed = coordinator.confirmed_enrollment_ledger.allocations.len();
    let current = current_state.ledger.allocations.len();
    match relative_count_state(confirmed, current) {
        RelativeCountStateV1::Synchronized => Ok(EnrollmentCoordinatorRecoveryDispositionV1::Synchronized),
        RelativeCountStateV1::OnePending => {
            let allocation = current_state
                .ledger
                .allocations
                .last()
                .ok_or_else(|| vec![PerceptualEnrollmentCoordinatorIssueV1::MissingPendingAllocation])?;
            let previous_witness_head_sha256 = coordinator
                .confirmed_witness_bundle
                .entries
                .last()
                .map(|entry| entry.witness_head_sha256.clone())
                .unwrap_or_else(|| ZERO_SHA256.into());
            let mut request = PendingEnrollmentWitnessRequestV1 {
                sequence: allocation.sequence,
                enrollment_witness_policy_sha256: witness_policy.policy_sha256.clone(),
                witness_log_id: witness_policy.witness_log_id.clone(),
                enrollment_ledger_sha256: current_state.ledger.ledger_sha256.clone(),
                durable_state_sha256: current_state.state_sha256.clone(),
                allocation_head_sha256: allocation.allocation_head_sha256.clone(),
                enrollment_participant_commitment_sha256: allocation
                    .participant_token_commitment_sha256
                    .clone(),
                participant_schedule_projection_sha256: allocation
                    .participant_schedule_projection_sha256
                    .clone(),
                previous_witness_head_sha256,
                request_sha256: String::new(),
            };
            request.request_sha256 = pending_witness_request_commitment(&request).map_err(|_| {
                vec![PerceptualEnrollmentCoordinatorIssueV1::PendingRequestSerializationFailed]
            })?;
            Ok(EnrollmentCoordinatorRecoveryDispositionV1::ExactlyOnePending(request))
        }
        RelativeCountStateV1::Rollback => Err(vec![
            PerceptualEnrollmentCoordinatorIssueV1::LocalAllocationRollback {
                confirmed,
                current,
            },
        ]),
        RelativeCountStateV1::ImpossibleGap => Err(vec![
            PerceptualEnrollmentCoordinatorIssueV1::ImpossibleAllocationGap {
                confirmed,
                current,
            },
        ]),
    }
}

pub fn pending_witness_request_commitment(
    request: &PendingEnrollmentWitnessRequestV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PendingWitnessRequestCommitmentV1 {
        domain: PENDING_WITNESS_REQUEST_DOMAIN,
        sequence: request.sequence,
        enrollment_witness_policy_sha256: &request.enrollment_witness_policy_sha256,
        witness_log_id: &request.witness_log_id,
        enrollment_ledger_sha256: &request.enrollment_ledger_sha256,
        durable_state_sha256: &request.durable_state_sha256,
        allocation_head_sha256: &request.allocation_head_sha256,
        enrollment_participant_commitment_sha256: &request
            .enrollment_participant_commitment_sha256,
        participant_schedule_projection_sha256: &request
            .participant_schedule_projection_sha256,
        previous_witness_head_sha256: &request.previous_witness_head_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn confirm_pending_enrollment_witness(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    current_state: &DurableEnrollmentAllocationStateV1,
    prior: &DurablePerceptualEnrollmentCoordinatorStateV1,
    verified: &VerifiedWitnessedEnrollmentAllocationV1,
    new_witness_bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
) -> Result<DurablePerceptualEnrollmentCoordinatorStateV1, Vec<PerceptualEnrollmentCoordinatorIssueV1>> {
    let pending = match classify_enrollment_coordinator_recovery(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        witness_policy,
        authenticity_policy,
        current_state,
        prior,
    )? {
        EnrollmentCoordinatorRecoveryDispositionV1::Synchronized => {
            return Err(vec![
                PerceptualEnrollmentCoordinatorIssueV1::TransitionRequiresPendingAllocation,
            ]);
        }
        EnrollmentCoordinatorRecoveryDispositionV1::ExactlyOnePending(pending) => pending,
    };

    if !validate_enrollment_witness_bundle(
        enrollment_policy,
        authenticity_policy,
        witness_policy,
        &current_state.ledger,
        new_witness_bundle,
    )
    .is_empty()
    {
        return Err(vec![
            PerceptualEnrollmentCoordinatorIssueV1::InvalidConfirmedWitnessBundle,
        ]);
    }
    let old_len = prior.confirmed_witness_bundle.entries.len();
    if new_witness_bundle.entries.len() != old_len.saturating_add(1)
        || new_witness_bundle.entries.get(..old_len)
            != Some(prior.confirmed_witness_bundle.entries.as_slice())
    {
        return Err(vec![
            PerceptualEnrollmentCoordinatorIssueV1::WitnessBundleDoesNotExtendConfirmedPrefix,
        ]);
    }
    let Some(last) = new_witness_bundle.entries.last() else {
        return Err(vec![
            PerceptualEnrollmentCoordinatorIssueV1::VerifiedWitnessReceiptMismatch,
        ]);
    };
    if last != verified.receipt()
        || last.sequence != pending.sequence
        || last.enrollment_ledger_sha256 != pending.enrollment_ledger_sha256
        || last.durable_state_sha256 != pending.durable_state_sha256
        || last.allocation_head_sha256 != pending.allocation_head_sha256
        || last.enrollment_participant_commitment_sha256
            != pending.enrollment_participant_commitment_sha256
        || last.participant_schedule_projection_sha256
            != pending.participant_schedule_projection_sha256
        || last.previous_witness_head_sha256 != pending.previous_witness_head_sha256
    {
        return Err(vec![
            PerceptualEnrollmentCoordinatorIssueV1::VerifiedWitnessReceiptMismatch,
        ]);
    }

    let mut next = DurablePerceptualEnrollmentCoordinatorStateV1 {
        state_version: PERCEPTUAL_ENROLLMENT_COORDINATOR_STATE_VERSION.into(),
        enrollment_policy_sha256: enrollment_policy.policy_sha256.clone(),
        enrollment_witness_policy_sha256: witness_policy.policy_sha256.clone(),
        confirmed_enrollment_ledger: current_state.ledger.clone(),
        confirmed_witness_bundle: new_witness_bundle.clone(),
        coordinator_state_sha256: String::new(),
    };
    seal_coordinator_state(&mut next)
        .map_err(|_| vec![PerceptualEnrollmentCoordinatorIssueV1::SerializationFailed])?;
    let validation = validate_enrollment_coordinator_state(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        witness_policy,
        authenticity_policy,
        current_state,
        &next,
    );
    if validation.is_empty() {
        Ok(next)
    } else {
        Err(validation)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn validate_enrollment_coordinator_state(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    current_state: &DurableEnrollmentAllocationStateV1,
    coordinator: &DurablePerceptualEnrollmentCoordinatorStateV1,
) -> Vec<PerceptualEnrollmentCoordinatorIssueV1> {
    let mut issues = Vec::new();
    if coordinator.state_version != PERCEPTUAL_ENROLLMENT_COORDINATOR_STATE_VERSION {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::WrongStateVersion);
    }
    if coordinator.enrollment_policy_sha256 != enrollment_policy.policy_sha256 {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::EnrollmentPolicyMismatch);
    }
    if coordinator.enrollment_witness_policy_sha256 != witness_policy.policy_sha256 {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::EnrollmentWitnessPolicyMismatch);
    }
    match coordinator_state_commitment(coordinator) {
        Ok(value) if value == coordinator.coordinator_state_sha256 => {}
        Ok(_) => issues.push(PerceptualEnrollmentCoordinatorIssueV1::CoordinatorStateDigestMismatch),
        Err(_) => issues.push(PerceptualEnrollmentCoordinatorIssueV1::InvalidCoordinatorStateDigest),
    }
    match state_commitment(current_state) {
        Ok(value) if value == current_state.state_sha256 => {}
        _ => issues.push(PerceptualEnrollmentCoordinatorIssueV1::InvalidCurrentDurableState),
    }
    if current_state.enrollment_policy_sha256 != enrollment_policy.policy_sha256 {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::CurrentStateIdentityMismatch {
            field: "enrollment_policy_sha256".into(),
        });
    }
    if current_state.token_generation_receipt_sha256 != token_receipt.receipt_sha256 {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::CurrentStateIdentityMismatch {
            field: "token_generation_receipt_sha256".into(),
        });
    }
    match canonical_json_sha256(schedule) {
        Ok(value) if value == current_state.participant_schedule_sha256 => {}
        _ => issues.push(PerceptualEnrollmentCoordinatorIssueV1::CurrentStateIdentityMismatch {
            field: "participant_schedule_sha256".into(),
        }),
    }
    if current_state.eligibility_gates.len() != current_state.ledger.allocations.len() {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::InvalidCurrentEnrollmentLedger);
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
        &current_state.eligibility_gates,
        &current_state.ledger,
    )
    .is_empty()
    {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::InvalidCurrentEnrollmentLedger);
    }

    let confirmed = coordinator.confirmed_enrollment_ledger.allocations.len();
    let current = current_state.ledger.allocations.len();
    if confirmed > current_state.eligibility_gates.len() {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::ConfirmedGateCardinalityMismatch);
        return issues;
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
        &current_state.eligibility_gates[..confirmed],
        &coordinator.confirmed_enrollment_ledger,
    )
    .is_empty()
    {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::InvalidConfirmedEnrollmentLedger);
    }
    if !validate_enrollment_witness_bundle(
        enrollment_policy,
        authenticity_policy,
        witness_policy,
        &coordinator.confirmed_enrollment_ledger,
        &coordinator.confirmed_witness_bundle,
    )
    .is_empty()
    {
        issues.push(PerceptualEnrollmentCoordinatorIssueV1::InvalidConfirmedWitnessBundle);
    }
    match enrollment_ledger_prefix(&current_state.ledger, confirmed) {
        Ok(prefix) if prefix == coordinator.confirmed_enrollment_ledger => {}
        _ => issues.push(PerceptualEnrollmentCoordinatorIssueV1::ConfirmedPrefixMismatch),
    }

    match relative_count_state(confirmed, current) {
        RelativeCountStateV1::Synchronized => {
            if current_state.ledger != coordinator.confirmed_enrollment_ledger {
                issues.push(PerceptualEnrollmentCoordinatorIssueV1::SynchronizedLedgerMismatch);
            }
        }
        RelativeCountStateV1::OnePending => {}
        RelativeCountStateV1::Rollback => {
            issues.push(PerceptualEnrollmentCoordinatorIssueV1::LocalAllocationRollback {
                confirmed,
                current,
            });
        }
        RelativeCountStateV1::ImpossibleGap => {
            issues.push(PerceptualEnrollmentCoordinatorIssueV1::ImpossibleAllocationGap {
                confirmed,
                current,
            });
        }
    }
    issues
}

fn enrollment_ledger_prefix(
    ledger: &FrozenPerceptualEnrollmentAllocationLedgerV1,
    count: usize,
) -> Result<FrozenPerceptualEnrollmentAllocationLedgerV1, serde_json::Error> {
    let mut prefix = ledger.clone();
    prefix.allocations.truncate(count);
    prefix.final_allocation_head_sha256 = prefix
        .allocations
        .last()
        .map(|entry| entry.allocation_head_sha256.clone())
        .unwrap_or_else(|| ZERO_SHA256.into());
    prefix.ledger_sha256 = enrollment_allocation_ledger_commitment(&prefix)?;
    Ok(prefix)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelativeCountStateV1 {
    Synchronized,
    OnePending,
    Rollback,
    ImpossibleGap,
}

fn relative_count_state(confirmed: usize, current: usize) -> RelativeCountStateV1 {
    if current == confirmed {
        RelativeCountStateV1::Synchronized
    } else if current == confirmed.saturating_add(1) {
        RelativeCountStateV1::OnePending
    } else if current < confirmed {
        RelativeCountStateV1::Rollback
    } else {
        RelativeCountStateV1::ImpossibleGap
    }
}

#[derive(Debug)]
pub enum DurableEnrollmentCoordinatorErrorV1 {
    UnsupportedPlatform,
    RootUnavailable,
    StateAlreadyExists,
    StateMissing,
    StateTooLarge,
    StateMalformed,
    StateDigestMismatch,
    ExpectedCoordinatorHeadMismatch { expected: String, found: String },
    ReadBackMismatch,
    Serialization,
    EntropyUnavailable,
    LocalLockPoisoned,
    KernelLockUnavailable,
    Domain(Vec<PerceptualEnrollmentCoordinatorIssueV1>),
    Io(std::io::Error),
}

impl std::fmt::Display for DurableEnrollmentCoordinatorErrorV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(formatter, "enrollment coordinator requires Linux"),
            Self::RootUnavailable => write!(formatter, "enrollment coordinator root unavailable"),
            Self::StateAlreadyExists => write!(formatter, "enrollment coordinator state already exists"),
            Self::StateMissing => write!(formatter, "enrollment coordinator state missing"),
            Self::StateTooLarge => write!(formatter, "enrollment coordinator state exceeds bound"),
            Self::StateMalformed => write!(formatter, "enrollment coordinator state malformed"),
            Self::StateDigestMismatch => write!(formatter, "enrollment coordinator state digest mismatch"),
            Self::ExpectedCoordinatorHeadMismatch { expected, found } => write!(
                formatter,
                "coordinator head mismatch: expected {expected}, found {found}"
            ),
            Self::ReadBackMismatch => write!(formatter, "enrollment coordinator read-back mismatch"),
            Self::Serialization => write!(formatter, "enrollment coordinator serialization failed"),
            Self::EntropyUnavailable => write!(formatter, "enrollment coordinator temporary-file entropy unavailable"),
            Self::LocalLockPoisoned => write!(formatter, "enrollment coordinator local lock poisoned"),
            Self::KernelLockUnavailable => write!(formatter, "enrollment coordinator kernel lock unavailable"),
            Self::Domain(issues) => write!(formatter, "enrollment coordinator domain validation failed with {} issue(s)", issues.len()),
            Self::Io(error) => write!(formatter, "enrollment coordinator I/O failed: {error}"),
        }
    }
}

impl std::error::Error for DurableEnrollmentCoordinatorErrorV1 {}

impl From<std::io::Error> for DurableEnrollmentCoordinatorErrorV1 {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<Vec<PerceptualEnrollmentCoordinatorIssueV1>> for DurableEnrollmentCoordinatorErrorV1 {
    fn from(issues: Vec<PerceptualEnrollmentCoordinatorIssueV1>) -> Self {
        Self::Domain(issues)
    }
}

/// Durable CAS journal for the allocation->witness handoff.
///
/// `transition` holds the process-local mutex and kernel `flock` for the entire
/// closure. Production code should perform allocation/recovery and independent
/// witnessing inside that closure, then return the newly sealed confirmed
/// coordinator state. If the closure errors or the process dies first, the old
/// coordinator state remains authoritative and restart sees the pending durable
/// allocation through `classify_enrollment_coordinator_recovery`.
pub struct DurableEnrollmentCoordinatorStoreV1 {
    root: PathBuf,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableEnrollmentCoordinatorStoreV1 {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        }
    }

    pub fn initialize(
        &self,
        state: &DurablePerceptualEnrollmentCoordinatorStateV1,
    ) -> Result<DurablePerceptualEnrollmentCoordinatorStateV1, DurableEnrollmentCoordinatorErrorV1> {
        let _local = self.local_lock.lock().map_err(|_| DurableEnrollmentCoordinatorErrorV1::LocalLockPoisoned)?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelCoordinatorLock::exclusive(&lock_file)?;
        if self.state_path()?.exists() {
            return Err(DurableEnrollmentCoordinatorErrorV1::StateAlreadyExists);
        }
        require_valid_state_digest(state)?;
        self.write_state_locked(state)?;
        let read_back = self.read_state_locked()?;
        if read_back != *state {
            return Err(DurableEnrollmentCoordinatorErrorV1::ReadBackMismatch);
        }
        Ok(read_back)
    }

    pub fn inspect(
        &self,
    ) -> Result<DurablePerceptualEnrollmentCoordinatorStateV1, DurableEnrollmentCoordinatorErrorV1> {
        let _local = self.local_lock.lock().map_err(|_| DurableEnrollmentCoordinatorErrorV1::LocalLockPoisoned)?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelCoordinatorLock::exclusive(&lock_file)?;
        self.read_state_locked()
    }

    pub fn transition<R, F>(
        &self,
        expected_coordinator_state_sha256: &str,
        operation: F,
    ) -> Result<R, DurableEnrollmentCoordinatorErrorV1>
    where
        F: FnOnce(
            &DurablePerceptualEnrollmentCoordinatorStateV1,
        ) -> Result<
            (DurablePerceptualEnrollmentCoordinatorStateV1, R),
            DurableEnrollmentCoordinatorErrorV1,
        >,
    {
        let _local = self.local_lock.lock().map_err(|_| DurableEnrollmentCoordinatorErrorV1::LocalLockPoisoned)?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelCoordinatorLock::exclusive(&lock_file)?;
        let current = self.read_state_locked()?;
        if current.coordinator_state_sha256 != expected_coordinator_state_sha256 {
            return Err(DurableEnrollmentCoordinatorErrorV1::ExpectedCoordinatorHeadMismatch {
                expected: expected_coordinator_state_sha256.into(),
                found: current.coordinator_state_sha256,
            });
        }
        let (next, output) = operation(&current)?;
        require_valid_state_digest(&next)?;
        self.write_state_locked(&next)?;
        let read_back = self.read_state_locked()?;
        if read_back != next {
            return Err(DurableEnrollmentCoordinatorErrorV1::ReadBackMismatch);
        }
        Ok(output)
    }

    fn read_state_locked(
        &self,
    ) -> Result<DurablePerceptualEnrollmentCoordinatorStateV1, DurableEnrollmentCoordinatorErrorV1> {
        let path = self.state_path()?;
        let file = match open_private_regular_file(&path, false, false) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(DurableEnrollmentCoordinatorErrorV1::StateMissing);
            }
            Err(error) => return Err(error.into()),
        };
        let metadata = file.metadata()?;
        if metadata.len() == 0 || metadata.len() > MAX_COORDINATOR_STATE_BYTES {
            return Err(DurableEnrollmentCoordinatorErrorV1::StateTooLarge);
        }
        let mut encoded = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_COORDINATOR_STATE_BYTES.saturating_add(1))
            .read_to_end(&mut encoded)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_COORDINATOR_STATE_BYTES {
            return Err(DurableEnrollmentCoordinatorErrorV1::StateTooLarge);
        }
        let state: DurablePerceptualEnrollmentCoordinatorStateV1 = serde_json::from_slice(&encoded)
            .map_err(|_| DurableEnrollmentCoordinatorErrorV1::StateMalformed)?;
        require_valid_state_digest(&state)?;
        Ok(state)
    }

    fn write_state_locked(
        &self,
        state: &DurablePerceptualEnrollmentCoordinatorStateV1,
    ) -> Result<(), DurableEnrollmentCoordinatorErrorV1> {
        require_valid_state_digest(state)?;
        let encoded = canonical_json_bytes(state)
            .map_err(|_| DurableEnrollmentCoordinatorErrorV1::Serialization)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_COORDINATOR_STATE_BYTES {
            return Err(DurableEnrollmentCoordinatorErrorV1::StateTooLarge);
        }
        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let mut nonce = [0u8; 16];
        OsRng
            .try_fill_bytes(&mut nonce)
            .map_err(|_| DurableEnrollmentCoordinatorErrorV1::EntropyUnavailable)?;
        let suffix = nonce.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        let temp = operation_root.join(format!(
            ".mel003-enrollment-witness-coordinator-{}-{suffix}.tmp",
            std::process::id()
        ));
        let target = operation_root.join(COORDINATOR_STATE_FILE_NAME);
        let result = (|| {
            let mut file = open_private_regular_file(&temp, true, true)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
            fs::rename(&temp, &target)?;
            root.sync_all()?;
            Ok::<(), DurableEnrollmentCoordinatorErrorV1>(())
        })();
        let _ = fs::remove_file(&temp);
        result
    }

    fn open_lock_file(&self) -> Result<File, DurableEnrollmentCoordinatorErrorV1> {
        let path = self.operation_root_path()?.join(COORDINATOR_LOCK_FILE_NAME);
        open_private_regular_file(&path, true, false).map_err(Into::into)
    }

    fn state_path(&self) -> Result<PathBuf, DurableEnrollmentCoordinatorErrorV1> {
        Ok(self.operation_root_path()?.join(COORDINATOR_STATE_FILE_NAME))
    }

    fn ensure_root(&self) -> Result<Arc<File>, DurableEnrollmentCoordinatorErrorV1> {
        #[cfg(not(target_os = "linux"))]
        {
            return Err(DurableEnrollmentCoordinatorErrorV1::UnsupportedPlatform);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

            let mut pinned = self.pinned_root.lock().map_err(|_| DurableEnrollmentCoordinatorErrorV1::LocalLockPoisoned)?;
            if let Some(root) = pinned.as_ref() {
                return Ok(Arc::clone(root));
            }
            fs::create_dir_all(&self.root)?;
            let metadata = fs::symlink_metadata(&self.root)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(DurableEnrollmentCoordinatorErrorV1::RootUnavailable);
            }
            fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))?;
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
            let root = Arc::new(options.open(&self.root)?);
            if !root.metadata()?.is_dir() {
                return Err(DurableEnrollmentCoordinatorErrorV1::RootUnavailable);
            }
            *pinned = Some(Arc::clone(&root));
            Ok(root)
        }
    }

    fn operation_root_path(&self) -> Result<PathBuf, DurableEnrollmentCoordinatorErrorV1> {
        #[cfg(not(target_os = "linux"))]
        {
            return Err(DurableEnrollmentCoordinatorErrorV1::UnsupportedPlatform);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let root = self.ensure_root()?;
            let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
            if !path.is_dir() {
                return Err(DurableEnrollmentCoordinatorErrorV1::RootUnavailable);
            }
            Ok(path)
        }
    }
}

fn require_valid_state_digest(
    state: &DurablePerceptualEnrollmentCoordinatorStateV1,
) -> Result<(), DurableEnrollmentCoordinatorErrorV1> {
    match coordinator_state_commitment(state) {
        Ok(value) if value == state.coordinator_state_sha256 => Ok(()),
        Ok(_) => Err(DurableEnrollmentCoordinatorErrorV1::StateDigestMismatch),
        Err(_) => Err(DurableEnrollmentCoordinatorErrorV1::Serialization),
    }
}

fn open_private_regular_file(path: &Path, create: bool, create_new: bool) -> std::io::Result<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut options = OpenOptions::new();
        options.read(true).write(create || create_new);
        if create_new {
            options.create_new(true);
        } else if create {
            options.create(true);
        }
        options
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let file = options.open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o077 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "coordinator state/lock must be a private regular file",
            ));
        }
        Ok(file)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, create, create_new);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "enrollment coordinator requires Linux",
        ))
    }
}

struct KernelCoordinatorLock<'a> {
    file: &'a File,
}

impl<'a> KernelCoordinatorLock<'a> {
    fn exclusive(file: &'a File) -> Result<Self, DurableEnrollmentCoordinatorErrorV1> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: the borrowed File owns a live descriptor for the guard lifetime.
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
            if result != 0 {
                return Err(DurableEnrollmentCoordinatorErrorV1::KernelLockUnavailable);
            }
            Ok(Self { file })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = file;
            Err(DurableEnrollmentCoordinatorErrorV1::UnsupportedPlatform)
        }
    }
}

impl Drop for KernelCoordinatorLock<'_> {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: unlock the same live descriptor retained by this guard.
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_count_state_accepts_only_zero_or_one_pending() {
        assert_eq!(relative_count_state(4, 4), RelativeCountStateV1::Synchronized);
        assert_eq!(relative_count_state(4, 5), RelativeCountStateV1::OnePending);
        assert_eq!(relative_count_state(4, 3), RelativeCountStateV1::Rollback);
        assert_eq!(relative_count_state(4, 6), RelativeCountStateV1::ImpossibleGap);
    }

    #[test]
    fn pending_request_identity_is_stable_and_order_sensitive() {
        let mut request = PendingEnrollmentWitnessRequestV1 {
            sequence: 2,
            enrollment_witness_policy_sha256: "a".repeat(64),
            witness_log_id: "enrollment-witness-log".into(),
            enrollment_ledger_sha256: "b".repeat(64),
            durable_state_sha256: "c".repeat(64),
            allocation_head_sha256: "d".repeat(64),
            enrollment_participant_commitment_sha256: "e".repeat(64),
            participant_schedule_projection_sha256: "f".repeat(64),
            previous_witness_head_sha256: "1".repeat(64),
            request_sha256: String::new(),
        };
        let first = pending_witness_request_commitment(&request).unwrap();
        let second = pending_witness_request_commitment(&request).unwrap();
        assert_eq!(first, second);
        request.sequence = 3;
        assert_ne!(first, pending_witness_request_commitment(&request).unwrap());
    }
}
