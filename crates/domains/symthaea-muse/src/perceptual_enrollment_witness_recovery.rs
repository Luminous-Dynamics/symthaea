// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1EILR-D: reconstruct verified witnessed-enrollment authority after
//! process loss without serializing runtime authority or replaying builder state.
//!
//! The recovery path reopens the real P1ENR durable store at the exact ledger
//! named by a recovered signed witness bundle, validates that bundle under the
//! frozen P1EIR-bound witness policy, re-derives the deterministic participant
//! slot/projection, and only then constructs a new non-serializable runtime
//! authority object.

use crate::evidence_digest::{
    perceptual_collection_authenticity::{
        participant_token_commitment, FrozenPerceptualCollectionAuthenticityPolicyV1,
    },
    perceptual_enrollment_coordinator::{
        classify_enrollment_coordinator_recovery, seal_coordinator_state,
        validate_enrollment_coordinator_state, DurablePerceptualEnrollmentCoordinatorStateV1,
        EnrollmentCoordinatorRecoveryDispositionV1,
    },
    perceptual_enrollment_lifecycle::{
        participant_allocation_reference_commitment, FrozenPerceptualEnrollmentPolicyV1,
    },
    perceptual_enrollment_store::DurableEnrollmentAllocationStoreV1,
    perceptual_enrollment_witness::{
        validate_enrollment_witness_bundle, FrozenPerceptualEnrollmentWitnessBundleV1,
        FrozenPerceptualEnrollmentWitnessPolicyV1,
        VerifiedWitnessedEnrollmentAllocationV1, WitnessedEnrollmentAllocationReceiptV1,
    },
    perceptual_participant_identity::{
        project_participant_schedule, FrozenParticipantIdentityBoundaryPolicyV1,
        FrozenParticipantScheduleProjectionV1,
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

mod sealed {
    pub trait Sealed {}
}

/// Common scored-authority precursor for both the live #4582 witness path and
/// the crash-recovered path in this module.
///
/// The trait is sealed: external crates cannot implement it for caller-created
/// data and thereby manufacture a verified enrollment authority.
pub trait VerifiedEnrollmentWitnessAuthorityV1: sealed::Sealed {
    fn witnessed_receipt(&self) -> &WitnessedEnrollmentAllocationReceiptV1;
    fn participant_schedule_projection(&self) -> &FrozenParticipantScheduleProjectionV1;
    fn collection_participant_commitment_sha256(&self) -> &str;
    fn current_enrollment_ledger_sha256(&self) -> &str;
}

impl sealed::Sealed for VerifiedWitnessedEnrollmentAllocationV1 {}

impl VerifiedEnrollmentWitnessAuthorityV1 for VerifiedWitnessedEnrollmentAllocationV1 {
    fn witnessed_receipt(&self) -> &WitnessedEnrollmentAllocationReceiptV1 {
        self.receipt()
    }

    fn participant_schedule_projection(&self) -> &FrozenParticipantScheduleProjectionV1 {
        self.participant_schedule_projection()
    }

    fn collection_participant_commitment_sha256(&self) -> &str {
        self.collection_participant_commitment_sha256()
    }

    fn current_enrollment_ledger_sha256(&self) -> &str {
        self.current_enrollment_ledger_sha256()
    }
}

/// Runtime-only authority reconstructed from durable P1ENR state plus a fully
/// validated recovered witness bundle. Private fields and no serde traits are
/// intentional.
#[derive(Debug)]
pub struct RecoveredVerifiedWitnessedEnrollmentAllocationV1 {
    receipt: WitnessedEnrollmentAllocationReceiptV1,
    participant_schedule_projection: FrozenParticipantScheduleProjectionV1,
    collection_participant_commitment_sha256: String,
}

impl RecoveredVerifiedWitnessedEnrollmentAllocationV1 {
    pub fn receipt(&self) -> &WitnessedEnrollmentAllocationReceiptV1 {
        &self.receipt
    }

    pub fn participant_schedule_projection(&self) -> &FrozenParticipantScheduleProjectionV1 {
        &self.participant_schedule_projection
    }

    pub fn collection_participant_commitment_sha256(&self) -> &str {
        &self.collection_participant_commitment_sha256
    }

    pub fn current_enrollment_ledger_sha256(&self) -> &str {
        &self.receipt.enrollment_ledger_sha256
    }
}

impl sealed::Sealed for RecoveredVerifiedWitnessedEnrollmentAllocationV1 {}

impl VerifiedEnrollmentWitnessAuthorityV1 for RecoveredVerifiedWitnessedEnrollmentAllocationV1 {
    fn witnessed_receipt(&self) -> &WitnessedEnrollmentAllocationReceiptV1 {
        &self.receipt
    }

    fn participant_schedule_projection(&self) -> &FrozenParticipantScheduleProjectionV1 {
        &self.participant_schedule_projection
    }

    fn collection_participant_commitment_sha256(&self) -> &str {
        &self.collection_participant_commitment_sha256
    }

    fn current_enrollment_ledger_sha256(&self) -> &str {
        &self.receipt.enrollment_ledger_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerceptualEnrollmentWitnessRecoveryIssueV1 {
    EmptyRecoveredWitnessBundle,
    DurableAllocationStateUnavailable,
    InvalidRecoveredWitnessBundle,
    MissingCurrentAllocationTail,
    MissingDeterministicParticipantSlot,
    EnrollmentParticipantCommitmentMismatch,
    ParticipantProjectionMismatch,
    CollectionParticipantCommitmentFailed,
    CoordinatorNotExactlyOnePending,
    CoordinatorValidationFailed,
    WitnessBundleDoesNotExtendConfirmedPrefix,
    RecoveredAuthorityDoesNotMatchPendingAllocation,
    SerializationFailed,
}

#[allow(clippy::too_many_arguments)]
pub fn recover_verified_witnessed_enrollment_allocation(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    store: &DurableEnrollmentAllocationStoreV1,
    recovered_bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
) -> Result<RecoveredVerifiedWitnessedEnrollmentAllocationV1, Vec<PerceptualEnrollmentWitnessRecoveryIssueV1>> {
    let Some(last_receipt) = recovered_bundle.entries.last() else {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::EmptyRecoveredWitnessBundle,
        ]);
    };

    // Reopen the real durable allocator state at the exact ledger named by the
    // recovered signed receipt. Caller-created in-memory state is insufficient.
    let state = store
        .inspect_current(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            &last_receipt.enrollment_ledger_sha256,
        )
        .map_err(|_| {
            vec![PerceptualEnrollmentWitnessRecoveryIssueV1::DurableAllocationStateUnavailable]
        })?;

    if !validate_enrollment_witness_bundle(
        enrollment_policy,
        authenticity_policy,
        witness_policy,
        &state.ledger,
        recovered_bundle,
    )
    .is_empty()
    {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::InvalidRecoveredWitnessBundle,
        ]);
    }

    let Some(allocation) = state.ledger.allocations.last() else {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::MissingCurrentAllocationTail,
        ]);
    };
    if allocation.sequence != last_receipt.sequence
        || allocation.allocation_head_sha256 != last_receipt.allocation_head_sha256
        || allocation.participant_token_commitment_sha256
            != last_receipt.enrollment_participant_commitment_sha256
        || allocation.participant_schedule_projection_sha256
            != last_receipt.participant_schedule_projection_sha256
    {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::InvalidRecoveredWitnessBundle,
        ]);
    }

    let mut canonical_tokens = cohort.participant_tokens.clone();
    canonical_tokens.sort();
    let Some(participant_token) = canonical_tokens.get(allocation.sequence as usize) else {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::MissingDeterministicParticipantSlot,
        ]);
    };
    match participant_allocation_reference_commitment(participant_token) {
        Ok(value) if value == allocation.participant_token_commitment_sha256 => {}
        _ => {
            return Err(vec![
                PerceptualEnrollmentWitnessRecoveryIssueV1::EnrollmentParticipantCommitmentMismatch,
            ]);
        }
    }

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
    .map_err(|_| {
        vec![PerceptualEnrollmentWitnessRecoveryIssueV1::ParticipantProjectionMismatch]
    })?;
    if projection.projection_sha256 != allocation.participant_schedule_projection_sha256
        || projection.projection_sha256 != last_receipt.participant_schedule_projection_sha256
    {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::ParticipantProjectionMismatch,
        ]);
    }

    let collection_participant_commitment_sha256 = participant_token_commitment(participant_token)
        .map_err(|_| {
            vec![PerceptualEnrollmentWitnessRecoveryIssueV1::CollectionParticipantCommitmentFailed]
        })?;

    Ok(RecoveredVerifiedWitnessedEnrollmentAllocationV1 {
        receipt: last_receipt.clone(),
        participant_schedule_projection: projection,
        collection_participant_commitment_sha256,
    })
}

/// Recovery-safe coordinator confirmation. This is generic only over the sealed
/// verified-authority surface, so both live and recovered paths use the same
/// confirmation theorem without admitting caller-defined implementations.
#[allow(clippy::too_many_arguments)]
pub fn confirm_pending_enrollment_witness_authority<A: VerifiedEnrollmentWitnessAuthorityV1>(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
    identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    store: &DurableEnrollmentAllocationStoreV1,
    prior: &DurablePerceptualEnrollmentCoordinatorStateV1,
    authority: &A,
    new_witness_bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
) -> Result<DurablePerceptualEnrollmentCoordinatorStateV1, Vec<PerceptualEnrollmentWitnessRecoveryIssueV1>> {
    let state = store
        .inspect_current(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            authority.current_enrollment_ledger_sha256(),
        )
        .map_err(|_| {
            vec![PerceptualEnrollmentWitnessRecoveryIssueV1::DurableAllocationStateUnavailable]
        })?;

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
        &state,
        prior,
    ) {
        Ok(EnrollmentCoordinatorRecoveryDispositionV1::ExactlyOnePending(value)) => value,
        Ok(EnrollmentCoordinatorRecoveryDispositionV1::Synchronized) => {
            return Err(vec![
                PerceptualEnrollmentWitnessRecoveryIssueV1::CoordinatorNotExactlyOnePending,
            ]);
        }
        Err(_) => {
            return Err(vec![
                PerceptualEnrollmentWitnessRecoveryIssueV1::CoordinatorValidationFailed,
            ]);
        }
    };

    if !validate_enrollment_witness_bundle(
        enrollment_policy,
        authenticity_policy,
        witness_policy,
        &state.ledger,
        new_witness_bundle,
    )
    .is_empty()
    {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::InvalidRecoveredWitnessBundle,
        ]);
    }
    let old_len = prior.confirmed_witness_bundle.entries.len();
    if new_witness_bundle.entries.len() != old_len.saturating_add(1)
        || new_witness_bundle.entries.get(..old_len)
            != Some(prior.confirmed_witness_bundle.entries.as_slice())
    {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::WitnessBundleDoesNotExtendConfirmedPrefix,
        ]);
    }
    let Some(last) = new_witness_bundle.entries.last() else {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::InvalidRecoveredWitnessBundle,
        ]);
    };
    let authority_receipt = authority.witnessed_receipt();
    if last != authority_receipt
        || last.sequence != pending.sequence
        || last.enrollment_ledger_sha256 != pending.enrollment_ledger_sha256
        || last.durable_state_sha256 != pending.durable_state_sha256
        || last.allocation_head_sha256 != pending.allocation_head_sha256
        || last.enrollment_participant_commitment_sha256
            != pending.enrollment_participant_commitment_sha256
        || last.participant_schedule_projection_sha256
            != pending.participant_schedule_projection_sha256
        || last.previous_witness_head_sha256 != pending.previous_witness_head_sha256
        || authority.participant_schedule_projection().projection_sha256
            != last.participant_schedule_projection_sha256
    {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::RecoveredAuthorityDoesNotMatchPendingAllocation,
        ]);
    }

    let mut next = DurablePerceptualEnrollmentCoordinatorStateV1 {
        state_version: prior.state_version.clone(),
        enrollment_policy_sha256: enrollment_policy.policy_sha256.clone(),
        enrollment_witness_policy_sha256: witness_policy.policy_sha256.clone(),
        confirmed_enrollment_ledger: state.ledger.clone(),
        confirmed_witness_bundle: new_witness_bundle.clone(),
        coordinator_state_sha256: String::new(),
    };
    seal_coordinator_state(&mut next).map_err(|_| {
        vec![PerceptualEnrollmentWitnessRecoveryIssueV1::SerializationFailed]
    })?;
    if !validate_enrollment_coordinator_state(
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
        &state,
        &next,
    )
    .is_empty()
    {
        return Err(vec![
            PerceptualEnrollmentWitnessRecoveryIssueV1::CoordinatorValidationFailed,
        ]);
    }
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_verified_authority<T: VerifiedEnrollmentWitnessAuthorityV1>() {}

    #[test]
    fn live_and_recovered_authority_share_one_sealed_surface() {
        assert_verified_authority::<VerifiedWitnessedEnrollmentAllocationV1>();
        assert_verified_authority::<RecoveredVerifiedWitnessedEnrollmentAllocationV1>();
    }
}
