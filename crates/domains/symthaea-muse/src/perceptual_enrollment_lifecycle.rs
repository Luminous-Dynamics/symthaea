// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1ENR-A: pre-enrollment eligibility, deterministic slot allocation,
//! and final enrollment-capacity accounting.
//!
//! This module is intentionally layered on P1DTR rather than rewriting P1D or
//! P1E. It freezes when a participant consumes one of the preregistered maximum-
//! enrollment slots and prevents an operator from selecting whichever remaining
//! randomized schedule they prefer after seeing participant information.
//!
//! Screening failures and pre-enrollment withdrawals remain outside the
//! registered enrollment cap. Once an eligibility-satisfied participant is
//! deterministically allocated a P1DTR slot, that slot remains consumed and must
//! end in exactly one enrolled disposition: complete, aborted, withdrawn/deleted,
//! or enrolled-without-a-scored-session.
//!
//! Absolute-time ordering is not asserted here. Eligibility/allocation events
//! carry opaque chronology commitments for later P1ER reconciliation.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_participant_identity::{
        FrozenParticipantIdentityBoundaryPolicyV1, FrozenParticipantScheduleProjectionV1,
        FrozenPerceptualParticipantTokenGenerationReceiptV1, project_participant_schedule,
        validate_participant_identity_boundary_policy,
        validate_participant_token_generation_receipt,
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

pub const PERCEPTUAL_ENROLLMENT_POLICY_VERSION: &str =
    "mel003-perceptual-enrollment-policy-v1";
pub const PERCEPTUAL_ELIGIBILITY_GATE_VERSION: &str =
    "mel003-perceptual-eligibility-gate-v1";
pub const PERCEPTUAL_ENROLLMENT_ALLOCATION_LEDGER_VERSION: &str =
    "mel003-perceptual-enrollment-allocation-ledger-v1";
pub const PERCEPTUAL_ENROLLMENT_PARTITION_VERSION: &str =
    "mel003-perceptual-enrollment-partition-v1";
pub const ZERO_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
const PARTICIPANT_ALLOCATION_REFERENCE_DOMAIN: &str =
    "symthaea.mel003.p1.enrollment.v1/participant-allocation-reference";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnrollmentSlotAllocationOrderV1 {
    /// Canonically sort the complete P1DTR token set and consume the next
    /// previously unallocated token. Token values are random pseudonyms; the
    /// ordering rule is deterministic and operator-independent.
    LexicographicParticipantToken,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualEnrollmentPolicyV1 {
    pub policy_version: String,
    pub protocol_sha256: String,
    pub token_generation_receipt_sha256: String,
    pub participant_identity_boundary_policy_sha256: String,
    pub participant_schedule_sha256: String,
    /// Exact participant-facing consent artifact to which the eligibility gate
    /// attests. This is reconciled with P1E collection authority after restack.
    pub consent_form_sha256: String,
    pub participant_information_sha256: String,
    /// Exact screening/eligibility flow or policy artifact.
    pub eligibility_policy_sha256: String,
    pub allocation_order: EnrollmentSlotAllocationOrderV1,
    pub screening_failure_is_pre_enrollment: bool,
    pub pre_enrollment_withdrawal_is_pre_allocation: bool,
    pub eligibility_required_before_allocation: bool,
    pub operator_schedule_choice_prohibited: bool,
    pub one_slot_per_eligibility_gate: bool,
    pub scored_access_before_allocation_prohibited: bool,
    pub policy_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualEligibilityGateReceiptV1 {
    pub gate_version: String,
    pub enrollment_policy_sha256: String,
    pub protocol_sha256: String,
    pub consent_form_sha256: String,
    pub participant_information_sha256: String,
    /// Opaque commitment to the restricted recruitment/screening transaction.
    /// No name, email, phone number, or other direct identity belongs here.
    pub eligibility_attempt_sha256: String,
    /// Opaque chronology-event commitment for later P1ER reconciliation.
    pub eligibility_chronology_event_sha256: String,
    pub informed_consent_confirmed: bool,
    pub minimum_age_eligibility_confirmed: bool,
    pub stereo_playback_check_passed: bool,
    pub comprehension_practice_passed: bool,
    pub eligible_for_enrollment: bool,
    pub gate_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerceptualEnrollmentAllocationReceiptV1 {
    pub sequence: u32,
    pub enrollment_policy_sha256: String,
    pub eligibility_gate_sha256: String,
    /// Domain-separated commitment to the P1DTR participant pseudonym. The raw
    /// token is intentionally not duplicated into the allocation ledger.
    pub participant_token_commitment_sha256: String,
    pub participant_schedule_projection_sha256: String,
    pub previous_allocation_head_sha256: String,
    pub allocation_head_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualEnrollmentAllocationLedgerV1 {
    pub ledger_version: String,
    pub enrollment_policy_sha256: String,
    pub token_generation_receipt_sha256: String,
    pub participant_schedule_sha256: String,
    /// Append order is evidence and therefore must not be sorted after creation.
    pub allocations: Vec<PerceptualEnrollmentAllocationReceiptV1>,
    pub final_allocation_head_sha256: String,
    pub ledger_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EnrolledParticipantDispositionV1 {
    CompleteSession,
    AbortedSession,
    WithdrawnAndDeleted,
    EnrolledNoScoredSession,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnrolledParticipantDispositionRecordV1 {
    pub participant_token_commitment_sha256: String,
    pub disposition: EnrolledParticipantDispositionV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualEnrollmentPartitionV1 {
    pub partition_version: String,
    pub enrollment_allocation_ledger_sha256: String,
    pub maximum_enrolled_participants: usize,
    /// Exactly one record for every consumed enrollment slot. Canonically sorted
    /// by participant-token commitment at seal time.
    pub enrolled_dispositions: Vec<EnrolledParticipantDispositionRecordV1>,
    /// Capacity never allocated to an eligibility-satisfied participant.
    pub unused_unallocated_slots: usize,
    /// Pre-enrollment operational quantities. They do not consume the registered
    /// maximum-enrollment cap under this v1 lifecycle.
    pub screening_failure_count: usize,
    pub pre_enrollment_withdrawal_count: usize,
    pub partition_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualEnrollmentLifecycleIssueV1 {
    InvalidProtocol,
    InvalidTokenGenerationReceipt,
    InvalidIdentityBoundaryPolicy,
    InvalidParticipantSchedule,
    WrongPolicyVersion,
    PolicyDigestMismatch,
    InvalidDigest { field: String },
    MissingPolicyProtection { field: String },
    WrongAllocationOrder,
    WrongEligibilityGateVersion,
    EligibilityPolicyMismatch { field: String },
    EligibilityRequirementNotSatisfied { field: String },
    EligibilityGateDigestMismatch,
    WrongLedgerVersion,
    LedgerPolicyMismatch,
    TokenReceiptMismatch,
    ParticipantScheduleDigestMismatch,
    TooManyAllocations { found: usize, maximum: usize },
    AllocationSequenceMismatch { index: usize },
    PreviousAllocationHeadMismatch { index: usize },
    AllocationHeadMismatch { index: usize },
    MissingEligibilityGate { index: usize },
    DuplicateEligibilityGate { digest: String },
    EligibilityGateNotValid { index: usize },
    WrongAllocatedParticipant { index: usize },
    WrongScheduleProjection { index: usize },
    DuplicateParticipantAllocation { commitment: String },
    LedgerFinalHeadMismatch,
    LedgerDigestMismatch,
    EnrollmentCapacityExhausted,
    WrongPartitionVersion,
    PartitionLedgerMismatch,
    MaximumEnrollmentMismatch { found: usize, expected: usize },
    DispositionNotCanonical { index: usize },
    DuplicateDisposition { commitment: String },
    MissingDisposition { commitment: String },
    UnexpectedDisposition { commitment: String },
    InvalidDispositionCommitment { index: usize },
    UnusedSlotCountMismatch { found: usize, expected: usize },
    EnrollmentPartitionMismatch { found: usize, expected: usize },
    PartitionDigestMismatch,
    SerializationFailed,
}

#[derive(Serialize)]
struct EnrollmentPolicyCommitment<'a> {
    policy_version: &'a str,
    protocol_sha256: &'a str,
    token_generation_receipt_sha256: &'a str,
    participant_identity_boundary_policy_sha256: &'a str,
    participant_schedule_sha256: &'a str,
    consent_form_sha256: &'a str,
    participant_information_sha256: &'a str,
    eligibility_policy_sha256: &'a str,
    allocation_order: EnrollmentSlotAllocationOrderV1,
    screening_failure_is_pre_enrollment: bool,
    pre_enrollment_withdrawal_is_pre_allocation: bool,
    eligibility_required_before_allocation: bool,
    operator_schedule_choice_prohibited: bool,
    one_slot_per_eligibility_gate: bool,
    scored_access_before_allocation_prohibited: bool,
}

#[derive(Serialize)]
struct EligibilityGateCommitment<'a> {
    gate_version: &'a str,
    enrollment_policy_sha256: &'a str,
    protocol_sha256: &'a str,
    consent_form_sha256: &'a str,
    participant_information_sha256: &'a str,
    eligibility_attempt_sha256: &'a str,
    eligibility_chronology_event_sha256: &'a str,
    informed_consent_confirmed: bool,
    minimum_age_eligibility_confirmed: bool,
    stereo_playback_check_passed: bool,
    comprehension_practice_passed: bool,
    eligible_for_enrollment: bool,
}

#[derive(Serialize)]
struct ParticipantAllocationReference<'a> {
    domain: &'static str,
    participant_token: &'a str,
}

#[derive(Serialize)]
struct AllocationHeadCommitment<'a> {
    sequence: u32,
    enrollment_policy_sha256: &'a str,
    eligibility_gate_sha256: &'a str,
    participant_token_commitment_sha256: &'a str,
    participant_schedule_projection_sha256: &'a str,
    previous_allocation_head_sha256: &'a str,
}

pub fn enrollment_policy_commitment(
    policy: &FrozenPerceptualEnrollmentPolicyV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&EnrollmentPolicyCommitment {
        policy_version: &policy.policy_version,
        protocol_sha256: &policy.protocol_sha256,
        token_generation_receipt_sha256: &policy.token_generation_receipt_sha256,
        participant_identity_boundary_policy_sha256: &policy
            .participant_identity_boundary_policy_sha256,
        participant_schedule_sha256: &policy.participant_schedule_sha256,
        consent_form_sha256: &policy.consent_form_sha256,
        participant_information_sha256: &policy.participant_information_sha256,
        eligibility_policy_sha256: &policy.eligibility_policy_sha256,
        allocation_order: policy.allocation_order,
        screening_failure_is_pre_enrollment: policy.screening_failure_is_pre_enrollment,
        pre_enrollment_withdrawal_is_pre_allocation: policy
            .pre_enrollment_withdrawal_is_pre_allocation,
        eligibility_required_before_allocation: policy.eligibility_required_before_allocation,
        operator_schedule_choice_prohibited: policy.operator_schedule_choice_prohibited,
        one_slot_per_eligibility_gate: policy.one_slot_per_eligibility_gate,
        scored_access_before_allocation_prohibited: policy
            .scored_access_before_allocation_prohibited,
    })
}

pub fn seal_enrollment_policy(
    policy: &mut FrozenPerceptualEnrollmentPolicyV1,
) -> Result<(), serde_json::Error> {
    policy.policy_sha256 = enrollment_policy_commitment(policy)?;
    Ok(())
}

pub fn eligibility_gate_commitment(
    gate: &FrozenPerceptualEligibilityGateReceiptV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&EligibilityGateCommitment {
        gate_version: &gate.gate_version,
        enrollment_policy_sha256: &gate.enrollment_policy_sha256,
        protocol_sha256: &gate.protocol_sha256,
        consent_form_sha256: &gate.consent_form_sha256,
        participant_information_sha256: &gate.participant_information_sha256,
        eligibility_attempt_sha256: &gate.eligibility_attempt_sha256,
        eligibility_chronology_event_sha256: &gate.eligibility_chronology_event_sha256,
        informed_consent_confirmed: gate.informed_consent_confirmed,
        minimum_age_eligibility_confirmed: gate.minimum_age_eligibility_confirmed,
        stereo_playback_check_passed: gate.stereo_playback_check_passed,
        comprehension_practice_passed: gate.comprehension_practice_passed,
        eligible_for_enrollment: gate.eligible_for_enrollment,
    })
}

pub fn seal_eligibility_gate(
    gate: &mut FrozenPerceptualEligibilityGateReceiptV1,
) -> Result<(), serde_json::Error> {
    gate.gate_sha256 = eligibility_gate_commitment(gate)?;
    Ok(())
}

pub fn participant_allocation_reference_commitment(
    participant_token: &str,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ParticipantAllocationReference {
        domain: PARTICIPANT_ALLOCATION_REFERENCE_DOMAIN,
        participant_token,
    })
}

fn allocation_head_commitment(
    allocation: &PerceptualEnrollmentAllocationReceiptV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&AllocationHeadCommitment {
        sequence: allocation.sequence,
        enrollment_policy_sha256: &allocation.enrollment_policy_sha256,
        eligibility_gate_sha256: &allocation.eligibility_gate_sha256,
        participant_token_commitment_sha256: &allocation.participant_token_commitment_sha256,
        participant_schedule_projection_sha256: &allocation.participant_schedule_projection_sha256,
        previous_allocation_head_sha256: &allocation.previous_allocation_head_sha256,
    })
}

pub fn enrollment_allocation_ledger_commitment(
    ledger: &FrozenPerceptualEnrollmentAllocationLedgerV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = ledger.clone();
    unsigned.ledger_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn enrollment_partition_commitment(
    partition: &FrozenPerceptualEnrollmentPartitionV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = partition.clone();
    unsigned.partition_sha256.clear();
    canonical_json_sha256(&unsigned)
}

#[allow(clippy::too_many_arguments)]
pub fn validate_enrollment_policy(
    protocol: &FrozenPerceptualStudyProtocolV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    policy: &FrozenPerceptualEnrollmentPolicyV1,
) -> Vec<PerceptualEnrollmentLifecycleIssueV1> {
    let mut issues = Vec::new();
    if !protocol.validate().is_empty() {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::InvalidProtocol);
    }
    if !validate_participant_token_generation_receipt(protocol, cohort, token_receipt).is_empty() {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::InvalidTokenGenerationReceipt);
    }
    if !validate_participant_identity_boundary_policy(token_receipt, identity_policy).is_empty() {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::InvalidIdentityBoundaryPolicy);
    }
    if policy.policy_version != PERCEPTUAL_ENROLLMENT_POLICY_VERSION {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::WrongPolicyVersion);
    }
    verify_digest(
        "protocol_sha256",
        canonical_json_sha256(protocol),
        &policy.protocol_sha256,
        &mut issues,
    );
    if policy.token_generation_receipt_sha256 != token_receipt.receipt_sha256 {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::TokenReceiptMismatch);
    }
    if policy.participant_identity_boundary_policy_sha256 != identity_policy.policy_sha256 {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::PolicyDigestMismatch);
    }
    verify_digest(
        "participant_schedule_sha256",
        canonical_json_sha256(schedule),
        &policy.participant_schedule_sha256,
        &mut issues,
    );
    for (field, digest) in [
        ("consent_form_sha256", policy.consent_form_sha256.as_str()),
        (
            "participant_information_sha256",
            policy.participant_information_sha256.as_str(),
        ),
        (
            "eligibility_policy_sha256",
            policy.eligibility_policy_sha256.as_str(),
        ),
        ("policy_sha256", policy.policy_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if policy.allocation_order != EnrollmentSlotAllocationOrderV1::LexicographicParticipantToken {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::WrongAllocationOrder);
    }
    for (field, required) in [
        (
            "screening_failure_is_pre_enrollment",
            policy.screening_failure_is_pre_enrollment,
        ),
        (
            "pre_enrollment_withdrawal_is_pre_allocation",
            policy.pre_enrollment_withdrawal_is_pre_allocation,
        ),
        (
            "eligibility_required_before_allocation",
            policy.eligibility_required_before_allocation,
        ),
        (
            "operator_schedule_choice_prohibited",
            policy.operator_schedule_choice_prohibited,
        ),
        ("one_slot_per_eligibility_gate", policy.one_slot_per_eligibility_gate),
        (
            "scored_access_before_allocation_prohibited",
            policy.scored_access_before_allocation_prohibited,
        ),
    ] {
        if !required {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::MissingPolicyProtection {
                field: field.into(),
            });
        }
    }
    match enrollment_policy_commitment(policy) {
        Ok(found) if found == policy.policy_sha256 => {}
        Ok(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::PolicyDigestMismatch),
        Err(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::SerializationFailed),
    }
    issues
}

pub fn validate_eligibility_gate(
    protocol: &FrozenPerceptualStudyProtocolV1,
    policy: &FrozenPerceptualEnrollmentPolicyV1,
    gate: &FrozenPerceptualEligibilityGateReceiptV1,
) -> Vec<PerceptualEnrollmentLifecycleIssueV1> {
    let mut issues = Vec::new();
    if gate.gate_version != PERCEPTUAL_ELIGIBILITY_GATE_VERSION {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::WrongEligibilityGateVersion);
    }
    for (field, found, expected) in [
        (
            "enrollment_policy_sha256",
            gate.enrollment_policy_sha256.as_str(),
            policy.policy_sha256.as_str(),
        ),
        (
            "protocol_sha256",
            gate.protocol_sha256.as_str(),
            policy.protocol_sha256.as_str(),
        ),
        (
            "consent_form_sha256",
            gate.consent_form_sha256.as_str(),
            policy.consent_form_sha256.as_str(),
        ),
        (
            "participant_information_sha256",
            gate.participant_information_sha256.as_str(),
            policy.participant_information_sha256.as_str(),
        ),
    ] {
        if found != expected {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::EligibilityPolicyMismatch {
                field: field.into(),
            });
        }
    }
    match canonical_json_sha256(protocol) {
        Ok(found) if found == gate.protocol_sha256 => {}
        Ok(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::EligibilityPolicyMismatch {
            field: "protocol_sha256".into(),
        }),
        Err(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::SerializationFailed),
    }
    for (field, digest) in [
        ("eligibility_attempt_sha256", gate.eligibility_attempt_sha256.as_str()),
        (
            "eligibility_chronology_event_sha256",
            gate.eligibility_chronology_event_sha256.as_str(),
        ),
        ("gate_sha256", gate.gate_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for (field, satisfied) in [
        ("informed_consent_confirmed", gate.informed_consent_confirmed),
        (
            "minimum_age_eligibility_confirmed",
            gate.minimum_age_eligibility_confirmed,
        ),
        ("stereo_playback_check_passed", gate.stereo_playback_check_passed),
        (
            "comprehension_practice_passed",
            gate.comprehension_practice_passed,
        ),
        ("eligible_for_enrollment", gate.eligible_for_enrollment),
    ] {
        if !satisfied {
            issues.push(
                PerceptualEnrollmentLifecycleIssueV1::EligibilityRequirementNotSatisfied {
                    field: field.into(),
                },
            );
        }
    }
    match eligibility_gate_commitment(gate) {
        Ok(found) if found == gate.gate_sha256 => {}
        Ok(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::EligibilityGateDigestMismatch),
        Err(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::SerializationFailed),
    }
    issues
}

pub fn new_enrollment_allocation_ledger(
    policy: &FrozenPerceptualEnrollmentPolicyV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    schedule: &PerceptualParticipantScheduleBookV1,
) -> Result<FrozenPerceptualEnrollmentAllocationLedgerV1, serde_json::Error> {
    let mut ledger = FrozenPerceptualEnrollmentAllocationLedgerV1 {
        ledger_version: PERCEPTUAL_ENROLLMENT_ALLOCATION_LEDGER_VERSION.into(),
        enrollment_policy_sha256: policy.policy_sha256.clone(),
        token_generation_receipt_sha256: token_receipt.receipt_sha256.clone(),
        participant_schedule_sha256: canonical_json_sha256(schedule)?,
        allocations: Vec::new(),
        final_allocation_head_sha256: ZERO_SHA256.into(),
        ledger_sha256: String::new(),
    };
    ledger.ledger_sha256 = enrollment_allocation_ledger_commitment(&ledger)?;
    Ok(ledger)
}

#[allow(clippy::too_many_arguments)]
pub fn allocate_next_enrollment_slot(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    gate: &FrozenPerceptualEligibilityGateReceiptV1,
    prior_gates: &[FrozenPerceptualEligibilityGateReceiptV1],
    ledger: &mut FrozenPerceptualEnrollmentAllocationLedgerV1,
) -> Result<FrozenParticipantScheduleProjectionV1, Vec<PerceptualEnrollmentLifecycleIssueV1>> {
    let mut all_gates = prior_gates.to_vec();
    all_gates.push(gate.clone());
    let mut issues = validate_enrollment_allocation_ledger(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        prior_gates,
        ledger,
    );
    issues.extend(validate_eligibility_gate(protocol, enrollment_policy, gate));
    if ledger
        .allocations
        .iter()
        .any(|entry| entry.eligibility_gate_sha256 == gate.gate_sha256)
    {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::DuplicateEligibilityGate {
            digest: gate.gate_sha256.clone(),
        });
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut canonical_tokens = cohort.participant_tokens.clone();
    canonical_tokens.sort();
    let index = ledger.allocations.len();
    let Some(participant_token) = canonical_tokens.get(index) else {
        return Err(vec![PerceptualEnrollmentLifecycleIssueV1::EnrollmentCapacityExhausted]);
    };
    let projection = project_participant_schedule(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        participant_token,
    )
    .map_err(|_| vec![PerceptualEnrollmentLifecycleIssueV1::InvalidParticipantSchedule])?;

    let previous = ledger
        .allocations
        .last()
        .map(|entry| entry.allocation_head_sha256.clone())
        .unwrap_or_else(|| ZERO_SHA256.into());
    let mut allocation = PerceptualEnrollmentAllocationReceiptV1 {
        sequence: index as u32,
        enrollment_policy_sha256: enrollment_policy.policy_sha256.clone(),
        eligibility_gate_sha256: gate.gate_sha256.clone(),
        participant_token_commitment_sha256: participant_allocation_reference_commitment(
            participant_token,
        )
        .map_err(|_| vec![PerceptualEnrollmentLifecycleIssueV1::SerializationFailed])?,
        participant_schedule_projection_sha256: projection.projection_sha256.clone(),
        previous_allocation_head_sha256: previous,
        allocation_head_sha256: String::new(),
    };
    allocation.allocation_head_sha256 = allocation_head_commitment(&allocation)
        .map_err(|_| vec![PerceptualEnrollmentLifecycleIssueV1::SerializationFailed])?;
    ledger.allocations.push(allocation);
    ledger.final_allocation_head_sha256 = ledger
        .allocations
        .last()
        .map(|entry| entry.allocation_head_sha256.clone())
        .unwrap_or_else(|| ZERO_SHA256.into());
    ledger.ledger_sha256 = enrollment_allocation_ledger_commitment(ledger)
        .map_err(|_| vec![PerceptualEnrollmentLifecycleIssueV1::SerializationFailed])?;

    let post = validate_enrollment_allocation_ledger(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        &all_gates,
        ledger,
    );
    if post.is_empty() {
        Ok(projection)
    } else {
        Err(post)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn validate_enrollment_allocation_ledger(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    eligibility_gates: &[FrozenPerceptualEligibilityGateReceiptV1],
    ledger: &FrozenPerceptualEnrollmentAllocationLedgerV1,
) -> Vec<PerceptualEnrollmentLifecycleIssueV1> {
    let mut issues = validate_enrollment_policy(
        protocol,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
    );
    if ledger.ledger_version != PERCEPTUAL_ENROLLMENT_ALLOCATION_LEDGER_VERSION {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::WrongLedgerVersion);
    }
    if ledger.enrollment_policy_sha256 != enrollment_policy.policy_sha256 {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::LedgerPolicyMismatch);
    }
    if ledger.token_generation_receipt_sha256 != token_receipt.receipt_sha256 {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::TokenReceiptMismatch);
    }
    verify_digest(
        "participant_schedule_sha256",
        canonical_json_sha256(schedule),
        &ledger.participant_schedule_sha256,
        &mut issues,
    );
    let maximum = protocol.sample_size.maximum_enrolled_participants;
    if ledger.allocations.len() > maximum {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::TooManyAllocations {
            found: ledger.allocations.len(),
            maximum,
        });
    }

    let gates_by_digest: BTreeMap<_, _> = eligibility_gates
        .iter()
        .map(|gate| (gate.gate_sha256.as_str(), gate))
        .collect();
    let mut seen_gate_digests = BTreeSet::new();
    let mut seen_participants = BTreeSet::new();
    let mut canonical_tokens = cohort.participant_tokens.clone();
    canonical_tokens.sort();
    let mut previous = ZERO_SHA256.to_string();

    for (index, allocation) in ledger.allocations.iter().enumerate() {
        if allocation.sequence != index as u32 {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::AllocationSequenceMismatch {
                index,
            });
        }
        if allocation.enrollment_policy_sha256 != enrollment_policy.policy_sha256 {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::LedgerPolicyMismatch);
        }
        if allocation.previous_allocation_head_sha256 != previous {
            issues.push(
                PerceptualEnrollmentLifecycleIssueV1::PreviousAllocationHeadMismatch { index },
            );
        }
        match allocation_head_commitment(allocation) {
            Ok(found) if found == allocation.allocation_head_sha256 => {}
            _ => issues.push(PerceptualEnrollmentLifecycleIssueV1::AllocationHeadMismatch {
                index,
            }),
        }
        if !seen_gate_digests.insert(allocation.eligibility_gate_sha256.clone()) {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::DuplicateEligibilityGate {
                digest: allocation.eligibility_gate_sha256.clone(),
            });
        }
        match gates_by_digest.get(allocation.eligibility_gate_sha256.as_str()) {
            Some(gate) => {
                if !validate_eligibility_gate(protocol, enrollment_policy, gate).is_empty() {
                    issues.push(PerceptualEnrollmentLifecycleIssueV1::EligibilityGateNotValid {
                        index,
                    });
                }
            }
            None => issues.push(PerceptualEnrollmentLifecycleIssueV1::MissingEligibilityGate {
                index,
            }),
        }

        let Some(expected_token) = canonical_tokens.get(index) else {
            continue;
        };
        let expected_commitment = participant_allocation_reference_commitment(expected_token);
        match expected_commitment {
            Ok(found) if found == allocation.participant_token_commitment_sha256 => {}
            _ => issues.push(PerceptualEnrollmentLifecycleIssueV1::WrongAllocatedParticipant {
                index,
            }),
        }
        if !seen_participants.insert(allocation.participant_token_commitment_sha256.clone()) {
            issues.push(
                PerceptualEnrollmentLifecycleIssueV1::DuplicateParticipantAllocation {
                    commitment: allocation.participant_token_commitment_sha256.clone(),
                },
            );
        }
        match project_participant_schedule(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            expected_token,
        ) {
            Ok(projection)
                if projection.projection_sha256
                    == allocation.participant_schedule_projection_sha256 => {}
            _ => issues.push(PerceptualEnrollmentLifecycleIssueV1::WrongScheduleProjection {
                index,
            }),
        }
        previous = allocation.allocation_head_sha256.clone();
    }
    if ledger.final_allocation_head_sha256 != previous {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::LedgerFinalHeadMismatch);
    }
    match enrollment_allocation_ledger_commitment(ledger) {
        Ok(found) if found == ledger.ledger_sha256 => {}
        Ok(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::LedgerDigestMismatch),
        Err(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::SerializationFailed),
    }
    issues
}

pub fn seal_enrollment_partition(
    partition: &mut FrozenPerceptualEnrollmentPartitionV1,
) -> Result<(), serde_json::Error> {
    partition.enrolled_dispositions.sort_by(|left, right| {
        left.participant_token_commitment_sha256
            .cmp(&right.participant_token_commitment_sha256)
    });
    partition.partition_sha256 = enrollment_partition_commitment(partition)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn validate_enrollment_partition(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    eligibility_gates: &[FrozenPerceptualEligibilityGateReceiptV1],
    ledger: &FrozenPerceptualEnrollmentAllocationLedgerV1,
    partition: &FrozenPerceptualEnrollmentPartitionV1,
) -> Vec<PerceptualEnrollmentLifecycleIssueV1> {
    let mut issues = validate_enrollment_allocation_ledger(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        token_receipt,
        identity_policy,
        schedule,
        enrollment_policy,
        eligibility_gates,
        ledger,
    );
    if partition.partition_version != PERCEPTUAL_ENROLLMENT_PARTITION_VERSION {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::WrongPartitionVersion);
    }
    if partition.enrollment_allocation_ledger_sha256 != ledger.ledger_sha256 {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::PartitionLedgerMismatch);
    }
    let expected_maximum = protocol.sample_size.maximum_enrolled_participants;
    if partition.maximum_enrolled_participants != expected_maximum {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::MaximumEnrollmentMismatch {
            found: partition.maximum_enrolled_participants,
            expected: expected_maximum,
        });
    }
    for (index, pair) in partition.enrolled_dispositions.windows(2).enumerate() {
        if pair[0].participant_token_commitment_sha256
            >= pair[1].participant_token_commitment_sha256
        {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::DispositionNotCanonical {
                index: index + 1,
            });
        }
    }
    let allocated: BTreeSet<_> = ledger
        .allocations
        .iter()
        .map(|entry| entry.participant_token_commitment_sha256.as_str())
        .collect();
    let mut dispositions = BTreeSet::new();
    for (index, disposition) in partition.enrolled_dispositions.iter().enumerate() {
        if !is_sha256(&disposition.participant_token_commitment_sha256) {
            issues.push(
                PerceptualEnrollmentLifecycleIssueV1::InvalidDispositionCommitment { index },
            );
        }
        if !dispositions.insert(disposition.participant_token_commitment_sha256.as_str()) {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::DuplicateDisposition {
                commitment: disposition.participant_token_commitment_sha256.clone(),
            });
        }
        if !allocated.contains(disposition.participant_token_commitment_sha256.as_str()) {
            issues.push(PerceptualEnrollmentLifecycleIssueV1::UnexpectedDisposition {
                commitment: disposition.participant_token_commitment_sha256.clone(),
            });
        }
    }
    for commitment in allocated.difference(&dispositions) {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::MissingDisposition {
            commitment: (*commitment).into(),
        });
    }
    let expected_unused = expected_maximum.saturating_sub(ledger.allocations.len());
    if partition.unused_unallocated_slots != expected_unused {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::UnusedSlotCountMismatch {
            found: partition.unused_unallocated_slots,
            expected: expected_unused,
        });
    }
    let found_partition = partition
        .enrolled_dispositions
        .len()
        .saturating_add(partition.unused_unallocated_slots);
    if found_partition != expected_maximum {
        issues.push(PerceptualEnrollmentLifecycleIssueV1::EnrollmentPartitionMismatch {
            found: found_partition,
            expected: expected_maximum,
        });
    }
    match enrollment_partition_commitment(partition) {
        Ok(found) if found == partition.partition_sha256 => {}
        Ok(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::PartitionDigestMismatch),
        Err(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::SerializationFailed),
    }
    issues
}

pub fn disposition_counts(
    partition: &FrozenPerceptualEnrollmentPartitionV1,
) -> BTreeMap<EnrolledParticipantDispositionV1, usize> {
    let mut counts = BTreeMap::new();
    for record in &partition.enrolled_dispositions {
        *counts.entry(record.disposition).or_insert(0) += 1;
    }
    counts
}

fn verify_digest(
    field: &str,
    result: Result<String, serde_json::Error>,
    expected: &str,
    issues: &mut Vec<PerceptualEnrollmentLifecycleIssueV1>,
) {
    match result {
        Ok(found) if found == expected => {}
        Ok(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::InvalidDigest {
            field: field.into(),
        }),
        Err(_) => issues.push(PerceptualEnrollmentLifecycleIssueV1::SerializationFailed),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn participant_allocation_reference_is_domain_bound() {
        let a = participant_allocation_reference_commitment("0123456789abcdef0123456789abcdef")
            .unwrap();
        let b = participant_allocation_reference_commitment("fedcba9876543210fedcba9876543210")
            .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn allocation_head_binds_sequence_and_previous_head() {
        let mut allocation = PerceptualEnrollmentAllocationReceiptV1 {
            sequence: 0,
            enrollment_policy_sha256: "a".repeat(64),
            eligibility_gate_sha256: "b".repeat(64),
            participant_token_commitment_sha256: "c".repeat(64),
            participant_schedule_projection_sha256: "d".repeat(64),
            previous_allocation_head_sha256: ZERO_SHA256.into(),
            allocation_head_sha256: String::new(),
        };
        let first = allocation_head_commitment(&allocation).unwrap();
        allocation.sequence = 1;
        assert_ne!(first, allocation_head_commitment(&allocation).unwrap());
        allocation.sequence = 0;
        allocation.previous_allocation_head_sha256 = "e".repeat(64);
        assert_ne!(first, allocation_head_commitment(&allocation).unwrap());
    }

    #[test]
    fn screening_failure_is_not_an_enrolled_disposition() {
        let dispositions = [
            EnrolledParticipantDispositionV1::CompleteSession,
            EnrolledParticipantDispositionV1::AbortedSession,
            EnrolledParticipantDispositionV1::WithdrawnAndDeleted,
            EnrolledParticipantDispositionV1::EnrolledNoScoredSession,
        ];
        assert_eq!(dispositions.len(), 4);
    }

    #[test]
    fn eligibility_commitment_binds_gate_outcome() {
        let mut gate = FrozenPerceptualEligibilityGateReceiptV1 {
            gate_version: PERCEPTUAL_ELIGIBILITY_GATE_VERSION.into(),
            enrollment_policy_sha256: "a".repeat(64),
            protocol_sha256: "b".repeat(64),
            consent_form_sha256: "c".repeat(64),
            participant_information_sha256: "d".repeat(64),
            eligibility_attempt_sha256: "e".repeat(64),
            eligibility_chronology_event_sha256: "f".repeat(64),
            informed_consent_confirmed: true,
            minimum_age_eligibility_confirmed: true,
            stereo_playback_check_passed: true,
            comprehension_practice_passed: true,
            eligible_for_enrollment: true,
            gate_sha256: String::new(),
        };
        let passing = eligibility_gate_commitment(&gate).unwrap();
        gate.eligible_for_enrollment = false;
        assert_ne!(passing, eligibility_gate_commitment(&gate).unwrap());
    }
}
