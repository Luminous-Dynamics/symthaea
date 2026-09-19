// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! EKM-083 — durable, read-only epoch-issuance evidence-chain validation.
//!
//! EKM-078 proves receipt-cursor and epoch-sequence continuity but deliberately
//! does not prove that an epoch was actually issued. EKM-079 describes one
//! passive activation/epoch record, EKM-080 binds all nine activation phases to
//! one stable verifier profile, EKM-081 checks local verifier policy and
//! checkpoint-relative trust continuity, and EKM-082 independently attests the
//! current verifier trust head.
//!
//! This module binds those layers to the exact EKM-077 receipt segment while the
//! EKM-082 currentness proof is still fresh, then chains the durable binding
//! receipts across EKM-078. It still does NOT claim verifier correctness or full
//! epoch issuance. No active epoch, mutation authority, or activation authority
//! is constructed here.

use crate::knowledge::epistemic_restart_continuity::{
    activation_verifier_currentness::{
        ActivationVerifierCurrentnessError, VerifiedActivationVerifierCurrentnessV1,
    },
    authority_epoch_contract::issuance_record::{
        runtime_validation::{
            trust_review::{
                ActivationVerifierTrustReviewReceiptV1, CanonicalVerifierTrustProfileDigestV1,
            },
            VerifiedActivationEvidenceReceiptV1,
        },
        RestartAuthorityEpochIssuanceRecordError, RestartAuthorityEpochIssuanceRecordV1,
    },
    epoch_bound_receipt_segment::{
        multi_epoch_history_chain::{
            MultiEpochRevisionHistoryChainError, MultiEpochRevisionHistoryChainV2,
        },
        EpochBoundRevisionReceiptSegmentError, EpochBoundRevisionReceiptSegmentV2,
    },
};
use crate::knowledge::epistemic_restart_revision_audit_restoration::ImmutableRevisionAuditRestorationV1;
use std::error::Error;
use std::fmt;

const MAX_EPOCH_BINDINGS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochIssuanceEvidenceBindingVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpochIssuanceEvidenceBindingDigestV1([u8; 32]);

impl EpochIssuanceEvidenceBindingDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] { self.0 }
    pub fn to_hex(self) -> String { hex32(self.0) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochIssuanceEvidenceChainVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpochIssuanceEvidenceChainDigestV1([u8; 32]);

impl EpochIssuanceEvidenceChainDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] { self.0 }
    pub fn to_hex(self) -> String { hex32(self.0) }
}

/// Durable audit receipt binding one EKM-079 epoch record to one EKM-077 segment
/// plus the exact EKM-080/081/082 verifier evidence stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedEpochIssuanceEvidenceBindingV1 {
    version: EpochIssuanceEvidenceBindingVersion,
    validated_at_cycle: u64,
    epoch_sequence: u64,
    previous_epoch_digest: Option<[u8; 32]>,
    authority_epoch_digest: [u8; 32],
    issuance_record_digest: [u8; 32],
    activation_commit_receipt_digest: [u8; 32],
    segment_digest: [u8; 32],
    activation_evidence_receipt_digest: [u8; 32],
    trust_review_digest: [u8; 32],
    verifier_currentness_receipt_digest: [u8; 32],
    canonical_verifier_profile_digest: CanonicalVerifierTrustProfileDigestV1,
    expected_live_generation: u64,
    committed_live_generation: u64,
    activated_at_cycle: u64,
    currentness_attested_at_cycle: u64,
    currentness_expires_at_cycle: u64,
    all_activation_phases_provider_accepted: bool,
    verifier_local_policy_accepted: bool,
    verifier_checkpoint_continuity_proven: bool,
    verifier_current_head_proven: bool,
    evidence_binding_consistent: bool,
    verifier_correctness_independently_proven: bool,
    epoch_issuance_verified: bool,
    active_epoch_installed: bool,
    mutation_authority: bool,
    activation_authorized: bool,
    binding_digest: EpochIssuanceEvidenceBindingDigestV1,
}

impl VerifiedEpochIssuanceEvidenceBindingV1 {
    pub fn version(&self) -> EpochIssuanceEvidenceBindingVersion { self.version }
    pub fn validated_at_cycle(&self) -> u64 { self.validated_at_cycle }
    pub fn epoch_sequence(&self) -> u64 { self.epoch_sequence }
    pub fn previous_epoch_digest(&self) -> Option<[u8; 32]> { self.previous_epoch_digest }
    pub fn authority_epoch_digest(&self) -> [u8; 32] { self.authority_epoch_digest }
    pub fn issuance_record_digest(&self) -> [u8; 32] { self.issuance_record_digest }
    pub fn activation_commit_receipt_digest(&self) -> [u8; 32] { self.activation_commit_receipt_digest }
    pub fn segment_digest(&self) -> [u8; 32] { self.segment_digest }
    pub fn activation_evidence_receipt_digest(&self) -> [u8; 32] { self.activation_evidence_receipt_digest }
    pub fn trust_review_digest(&self) -> [u8; 32] { self.trust_review_digest }
    pub fn verifier_currentness_receipt_digest(&self) -> [u8; 32] { self.verifier_currentness_receipt_digest }
    pub fn canonical_verifier_profile_digest(&self) -> CanonicalVerifierTrustProfileDigestV1 { self.canonical_verifier_profile_digest }
    pub fn expected_live_generation(&self) -> u64 { self.expected_live_generation }
    pub fn committed_live_generation(&self) -> u64 { self.committed_live_generation }
    pub fn activated_at_cycle(&self) -> u64 { self.activated_at_cycle }
    pub fn currentness_attested_at_cycle(&self) -> u64 { self.currentness_attested_at_cycle }
    pub fn currentness_expires_at_cycle(&self) -> u64 { self.currentness_expires_at_cycle }
    pub fn all_activation_phases_provider_accepted(&self) -> bool { self.all_activation_phases_provider_accepted }
    pub fn verifier_local_policy_accepted(&self) -> bool { self.verifier_local_policy_accepted }
    pub fn verifier_checkpoint_continuity_proven(&self) -> bool { self.verifier_checkpoint_continuity_proven }
    pub fn verifier_current_head_proven(&self) -> bool { self.verifier_current_head_proven }
    pub fn evidence_binding_consistent(&self) -> bool { self.evidence_binding_consistent }
    pub fn verifier_correctness_independently_proven(&self) -> bool { self.verifier_correctness_independently_proven }
    pub fn epoch_issuance_verified(&self) -> bool { self.epoch_issuance_verified }
    pub fn active_epoch_installed(&self) -> bool { self.active_epoch_installed }
    pub fn mutation_authority(&self) -> bool { self.mutation_authority }
    pub fn activation_authorized(&self) -> bool { self.activation_authorized }
    pub fn binding_digest(&self) -> EpochIssuanceEvidenceBindingDigestV1 { self.binding_digest }

    pub fn verify_internal(&self) -> Result<(), EpochIssuanceEvidenceChainError> {
        validate_binding_shape(self)?;
        if digest_binding(self)? != self.binding_digest {
            return Err(EpochIssuanceEvidenceChainError::BindingDigestMismatch);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bind_epoch_issuance_evidence(
    record: &RestartAuthorityEpochIssuanceRecordV1,
    segment: &EpochBoundRevisionReceiptSegmentV2,
    activation_evidence: &VerifiedActivationEvidenceReceiptV1,
    trust_review: &ActivationVerifierTrustReviewReceiptV1,
    currentness: &VerifiedActivationVerifierCurrentnessV1,
    validated_at_cycle: u64,
) -> Result<VerifiedEpochIssuanceEvidenceBindingV1, EpochIssuanceEvidenceChainError> {
    record.verify().map_err(EpochIssuanceEvidenceChainError::IssuanceRecordInvalid)?;
    segment.verify().map_err(EpochIssuanceEvidenceChainError::SegmentInvalid)?;
    currentness.verify_internal().map_err(EpochIssuanceEvidenceChainError::CurrentnessInvalid)?;

    if activation_evidence.issuance_record_digest() != record.record_digest()
        || activation_evidence.activation_commit_receipt_digest() != record.activation_commit_receipt_digest()
        || activation_evidence.authority_epoch_digest() != record.authority_epoch_digest()
    {
        return Err(EpochIssuanceEvidenceChainError::ActivationEvidenceRecordMismatch);
    }
    if !activation_evidence.all_phase_evidence_provider_accepted()
        || !activation_evidence.verifier_profile_stable_during_validation()
        || activation_evidence.provider_trust_independently_established()
        || activation_evidence.epoch_issuance_chain_verified()
        || activation_evidence.epoch_issuance_authorized()
        || activation_evidence.mutation_authority()
        || activation_evidence.activation_authorized()
    {
        return Err(EpochIssuanceEvidenceChainError::UnexpectedActivationEvidenceAuthority);
    }

    if trust_review.activation_evidence_receipt_digest() != activation_evidence.receipt_digest()
        || trust_review.issuance_record_digest() != record.record_digest()
    {
        return Err(EpochIssuanceEvidenceChainError::TrustReviewEvidenceMismatch);
    }
    if !trust_review.local_policy_accepted()
        || !trust_review.continuity_relative_to_caller_checkpoint_proven()
        || !trust_review.profile_fresh_at_review_cycle()
        || trust_review.provider_trust_independently_established()
        || trust_review.global_current_head_independently_proven()
        || trust_review.epoch_issuance_chain_verified()
        || trust_review.epoch_issuance_authorized()
        || trust_review.mutation_authority()
        || trust_review.activation_authorized()
    {
        return Err(EpochIssuanceEvidenceChainError::UnexpectedTrustReviewAuthority);
    }

    let statement = currentness.statement();
    if statement.trust_review_digest() != trust_review.review_digest()
        || statement.activation_evidence_receipt_digest() != activation_evidence.receipt_digest().as_bytes()
        || statement.issuance_record_digest() != record.record_digest().as_bytes()
        || statement.verifier_profile_digest() != trust_review.candidate_profile_digest()
        || statement.trust_policy_digest() != trust_review.trust_policy_digest().as_bytes()
        || statement.reference_checkpoint_digest() != trust_review.reference_checkpoint_digest().as_bytes()
    {
        return Err(EpochIssuanceEvidenceChainError::CurrentnessBindingMismatch);
    }
    if !currentness.global_current_head_independently_proven()
        || currentness.verifier_correctness_independently_proven()
        || currentness.trusted_state_mutated()
        || currentness.epoch_issuance_chain_verified()
        || currentness.epoch_issuance_authorized()
        || currentness.mutation_authority()
        || currentness.activation_authorized()
    {
        return Err(EpochIssuanceEvidenceChainError::UnexpectedCurrentnessAuthority);
    }

    if validated_at_cycle < currentness.verified_at_cycle()
        || validated_at_cycle < trust_review.reviewed_at_cycle()
        || validated_at_cycle < activation_evidence.verified_at_cycle()
    {
        return Err(EpochIssuanceEvidenceChainError::BindingPredatesSourceEvidence);
    }
    if validated_at_cycle >= statement.expires_at_cycle() {
        return Err(EpochIssuanceEvidenceChainError::CurrentnessExpiredBeforeBinding {
            validated_at_cycle,
            expires_at_cycle: statement.expires_at_cycle(),
        });
    }

    if segment.authority_epoch_sequence() != record.epoch_sequence()
        || segment.authority_epoch_digest() != record.authority_epoch_digest().as_bytes()
    {
        return Err(EpochIssuanceEvidenceChainError::SegmentEpochMismatch);
    }
    if segment.captured_at_cycle() < record.activated_at_cycle() {
        return Err(EpochIssuanceEvidenceChainError::SegmentPredatesActivation {
            segment_cycle: segment.captured_at_cycle(),
            activated_at_cycle: record.activated_at_cycle(),
        });
    }
    for receipt in segment.records() {
        if receipt.evaluated_at_cycle() < record.activated_at_cycle() {
            return Err(EpochIssuanceEvidenceChainError::ReceiptPredatesAuthorityEpoch {
                receipt_id: receipt.receipt_id().0,
                evaluated_at_cycle: receipt.evaluated_at_cycle(),
                activated_at_cycle: record.activated_at_cycle(),
            });
        }
    }

    let mut out = VerifiedEpochIssuanceEvidenceBindingV1 {
        version: EpochIssuanceEvidenceBindingVersion::V1,
        validated_at_cycle,
        epoch_sequence: record.epoch_sequence(),
        previous_epoch_digest: record.previous_epoch_digest().map(|digest| digest.as_bytes()),
        authority_epoch_digest: record.authority_epoch_digest().as_bytes(),
        issuance_record_digest: record.record_digest().as_bytes(),
        activation_commit_receipt_digest: record.activation_commit_receipt_digest().as_bytes(),
        segment_digest: segment.segment_digest().as_bytes(),
        activation_evidence_receipt_digest: activation_evidence.receipt_digest().as_bytes(),
        trust_review_digest: trust_review.review_digest().as_bytes(),
        verifier_currentness_receipt_digest: currentness.receipt_digest().as_bytes(),
        canonical_verifier_profile_digest: trust_review.candidate_profile_digest(),
        expected_live_generation: record.expected_live_generation(),
        committed_live_generation: record.committed_live_generation(),
        activated_at_cycle: record.activated_at_cycle(),
        currentness_attested_at_cycle: statement.attested_at_cycle(),
        currentness_expires_at_cycle: statement.expires_at_cycle(),
        all_activation_phases_provider_accepted: true,
        verifier_local_policy_accepted: true,
        verifier_checkpoint_continuity_proven: true,
        verifier_current_head_proven: true,
        evidence_binding_consistent: true,
        verifier_correctness_independently_proven: false,
        epoch_issuance_verified: false,
        active_epoch_installed: false,
        mutation_authority: false,
        activation_authorized: false,
        binding_digest: EpochIssuanceEvidenceBindingDigestV1([0; 32]),
    };
    out.binding_digest = digest_binding(&out)?;
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedEpochIssuanceEvidenceChainV1 {
    version: EpochIssuanceEvidenceChainVersion,
    captured_at_cycle: u64,
    multi_epoch_history_chain_digest: [u8; 32],
    legacy_audit_restoration_digest: [u8; 32],
    bindings: Vec<VerifiedEpochIssuanceEvidenceBindingV1>,
    final_next_receipt_id: u64,
    latest_epoch_sequence: Option<u64>,
    latest_authority_epoch_digest: Option<[u8; 32]>,
    receipt_cursor_continuity_proven: bool,
    epoch_sequence_continuity_proven: bool,
    epoch_predecessor_continuity_proven: bool,
    live_generation_continuity_proven: bool,
    activation_cycle_monotonicity_proven: bool,
    issuance_evidence_chain_consistent: bool,
    verifier_current_head_evidence_bound: bool,
    verifier_correctness_independently_proven: bool,
    epoch_issuance_chain_verified: bool,
    active_epoch_registry_constructed: bool,
    operational_history_constructed: bool,
    mutation_authority: bool,
    activation_authorized: bool,
    chain_digest: EpochIssuanceEvidenceChainDigestV1,
}

impl VerifiedEpochIssuanceEvidenceChainV1 {
    pub fn version(&self) -> EpochIssuanceEvidenceChainVersion { self.version }
    pub fn captured_at_cycle(&self) -> u64 { self.captured_at_cycle }
    pub fn multi_epoch_history_chain_digest(&self) -> [u8; 32] { self.multi_epoch_history_chain_digest }
    pub fn legacy_audit_restoration_digest(&self) -> [u8; 32] { self.legacy_audit_restoration_digest }
    pub fn bindings(&self) -> &[VerifiedEpochIssuanceEvidenceBindingV1] { &self.bindings }
    pub fn final_next_receipt_id(&self) -> u64 { self.final_next_receipt_id }
    pub fn latest_epoch_sequence(&self) -> Option<u64> { self.latest_epoch_sequence }
    pub fn latest_authority_epoch_digest(&self) -> Option<[u8; 32]> { self.latest_authority_epoch_digest }
    pub fn receipt_cursor_continuity_proven(&self) -> bool { self.receipt_cursor_continuity_proven }
    pub fn epoch_sequence_continuity_proven(&self) -> bool { self.epoch_sequence_continuity_proven }
    pub fn epoch_predecessor_continuity_proven(&self) -> bool { self.epoch_predecessor_continuity_proven }
    pub fn live_generation_continuity_proven(&self) -> bool { self.live_generation_continuity_proven }
    pub fn activation_cycle_monotonicity_proven(&self) -> bool { self.activation_cycle_monotonicity_proven }
    pub fn issuance_evidence_chain_consistent(&self) -> bool { self.issuance_evidence_chain_consistent }
    pub fn verifier_current_head_evidence_bound(&self) -> bool { self.verifier_current_head_evidence_bound }
    pub fn verifier_correctness_independently_proven(&self) -> bool { self.verifier_correctness_independently_proven }
    pub fn epoch_issuance_chain_verified(&self) -> bool { self.epoch_issuance_chain_verified }
    pub fn active_epoch_registry_constructed(&self) -> bool { self.active_epoch_registry_constructed }
    pub fn operational_history_constructed(&self) -> bool { self.operational_history_constructed }
    pub fn mutation_authority(&self) -> bool { self.mutation_authority }
    pub fn activation_authorized(&self) -> bool { self.activation_authorized }
    pub fn chain_digest(&self) -> EpochIssuanceEvidenceChainDigestV1 { self.chain_digest }

    pub fn verify_against(
        &self,
        legacy_audit: &ImmutableRevisionAuditRestorationV1,
        history_chain: &MultiEpochRevisionHistoryChainV2,
    ) -> Result<(), EpochIssuanceEvidenceChainError> {
        history_chain.verify_against(legacy_audit)
            .map_err(EpochIssuanceEvidenceChainError::HistoryChainInvalid)?;
        validate_stored_chain(history_chain, &self.bindings, self.captured_at_cycle)?;
        validate_chain_receipt_shape(self, history_chain)?;
        if self.multi_epoch_history_chain_digest != history_chain.chain_digest().as_bytes()
            || self.legacy_audit_restoration_digest != legacy_audit.restoration_digest().as_bytes()
            || self.final_next_receipt_id != history_chain.final_next_receipt_id().0
        {
            return Err(EpochIssuanceEvidenceChainError::StoredChainBindingMismatch);
        }
        if digest_chain_receipt(self)? != self.chain_digest {
            return Err(EpochIssuanceEvidenceChainError::ChainReceiptDigestMismatch);
        }
        Ok(())
    }
}

pub fn validate_epoch_issuance_evidence_chain(
    legacy_audit: &ImmutableRevisionAuditRestorationV1,
    history_chain: &MultiEpochRevisionHistoryChainV2,
    bindings: Vec<VerifiedEpochIssuanceEvidenceBindingV1>,
    captured_at_cycle: u64,
) -> Result<VerifiedEpochIssuanceEvidenceChainV1, EpochIssuanceEvidenceChainError> {
    history_chain.verify_against(legacy_audit)
        .map_err(EpochIssuanceEvidenceChainError::HistoryChainInvalid)?;
    if !history_chain.receipt_cursor_continuity_proven()
        || !history_chain.epoch_sequence_continuity_proven()
        || history_chain.epoch_issuance_chain_verified()
        || history_chain.operational_history_constructed()
        || history_chain.mutation_authority()
        || history_chain.activation_authorized()
    {
        return Err(EpochIssuanceEvidenceChainError::UnexpectedHistoryChainAuthority);
    }
    validate_stored_chain(history_chain, &bindings, captured_at_cycle)?;

    let latest_epoch_sequence = bindings.last().map(VerifiedEpochIssuanceEvidenceBindingV1::epoch_sequence);
    let latest_authority_epoch_digest = bindings.last().map(VerifiedEpochIssuanceEvidenceBindingV1::authority_epoch_digest);
    let has_epoch_bindings = !bindings.is_empty();
    let mut out = VerifiedEpochIssuanceEvidenceChainV1 {
        version: EpochIssuanceEvidenceChainVersion::V1,
        captured_at_cycle,
        multi_epoch_history_chain_digest: history_chain.chain_digest().as_bytes(),
        legacy_audit_restoration_digest: legacy_audit.restoration_digest().as_bytes(),
        bindings,
        final_next_receipt_id: history_chain.final_next_receipt_id().0,
        latest_epoch_sequence,
        latest_authority_epoch_digest,
        receipt_cursor_continuity_proven: true,
        epoch_sequence_continuity_proven: true,
        epoch_predecessor_continuity_proven: true,
        live_generation_continuity_proven: true,
        activation_cycle_monotonicity_proven: true,
        issuance_evidence_chain_consistent: true,
        verifier_current_head_evidence_bound: has_epoch_bindings,
        verifier_correctness_independently_proven: false,
        epoch_issuance_chain_verified: false,
        active_epoch_registry_constructed: false,
        operational_history_constructed: false,
        mutation_authority: false,
        activation_authorized: false,
        chain_digest: EpochIssuanceEvidenceChainDigestV1([0; 32]),
    };
    validate_chain_receipt_shape(&out, history_chain)?;
    out.chain_digest = digest_chain_receipt(&out)?;
    Ok(out)
}

fn validate_binding_shape(
    binding: &VerifiedEpochIssuanceEvidenceBindingV1,
) -> Result<(), EpochIssuanceEvidenceChainError> {
    if binding.version != EpochIssuanceEvidenceBindingVersion::V1
        || binding.epoch_sequence == 0
        || binding.authority_epoch_digest == [0; 32]
        || binding.issuance_record_digest == [0; 32]
        || binding.activation_commit_receipt_digest == [0; 32]
        || binding.segment_digest == [0; 32]
        || binding.activation_evidence_receipt_digest == [0; 32]
        || binding.trust_review_digest == [0; 32]
        || binding.verifier_currentness_receipt_digest == [0; 32]
    {
        return Err(EpochIssuanceEvidenceChainError::MalformedBinding);
    }
    match (binding.epoch_sequence, binding.previous_epoch_digest) {
        (1, None) => {}
        (1, Some(_)) => return Err(EpochIssuanceEvidenceChainError::GenesisHasPredecessor),
        (_, Some(digest)) if digest != [0; 32] => {}
        _ => return Err(EpochIssuanceEvidenceChainError::MissingEpochPredecessor),
    }
    let successor = binding.expected_live_generation.checked_add(1)
        .ok_or(EpochIssuanceEvidenceChainError::GenerationOverflow)?;
    if binding.committed_live_generation != successor {
        return Err(EpochIssuanceEvidenceChainError::InvalidGenerationTransition {
            expected: successor,
            actual: binding.committed_live_generation,
        });
    }
    if binding.validated_at_cycle < binding.currentness_attested_at_cycle
        || binding.validated_at_cycle >= binding.currentness_expires_at_cycle
    {
        return Err(EpochIssuanceEvidenceChainError::InvalidBindingCurrentnessWindow);
    }
    if !binding.all_activation_phases_provider_accepted
        || !binding.verifier_local_policy_accepted
        || !binding.verifier_checkpoint_continuity_proven
        || !binding.verifier_current_head_proven
        || !binding.evidence_binding_consistent
        || binding.verifier_correctness_independently_proven
        || binding.epoch_issuance_verified
        || binding.active_epoch_installed
        || binding.mutation_authority
        || binding.activation_authorized
    {
        return Err(EpochIssuanceEvidenceChainError::UnexpectedBindingAuthority);
    }
    Ok(())
}

fn validate_stored_chain(
    history_chain: &MultiEpochRevisionHistoryChainV2,
    bindings: &[VerifiedEpochIssuanceEvidenceBindingV1],
    captured_at_cycle: u64,
) -> Result<(), EpochIssuanceEvidenceChainError> {
    if bindings.len() > MAX_EPOCH_BINDINGS {
        return Err(EpochIssuanceEvidenceChainError::TooManyBindings {
            actual: bindings.len(),
            maximum: MAX_EPOCH_BINDINGS,
        });
    }
    if bindings.len() != history_chain.segments().len() {
        return Err(EpochIssuanceEvidenceChainError::BindingCountMismatch {
            expected: history_chain.segments().len(),
            actual: bindings.len(),
        });
    }
    if captured_at_cycle < history_chain.captured_at_cycle() {
        return Err(EpochIssuanceEvidenceChainError::CapturePredatesHistoryChain {
            captured_at_cycle,
            history_chain_cycle: history_chain.captured_at_cycle(),
        });
    }

    let mut previous_epoch_digest = None;
    let mut previous_committed_generation = None;
    let mut previous_activation_cycle = None;
    for (index, (segment, binding)) in history_chain.segments().iter().zip(bindings).enumerate() {
        segment.verify().map_err(EpochIssuanceEvidenceChainError::SegmentInvalid)?;
        binding.verify_internal()?;
        if binding.validated_at_cycle > captured_at_cycle {
            return Err(EpochIssuanceEvidenceChainError::BindingAfterChainCapture {
                binding_cycle: binding.validated_at_cycle,
                chain_cycle: captured_at_cycle,
            });
        }
        let expected_sequence = u64::try_from(index)
            .map_err(|_| EpochIssuanceEvidenceChainError::LengthOverflow)?
            .checked_add(1)
            .ok_or(EpochIssuanceEvidenceChainError::EpochSequenceOverflow)?;
        if binding.epoch_sequence != expected_sequence
            || segment.authority_epoch_sequence() != expected_sequence
        {
            return Err(EpochIssuanceEvidenceChainError::EpochSequenceMismatch {
                expected: expected_sequence,
                binding: binding.epoch_sequence,
                segment: segment.authority_epoch_sequence(),
            });
        }
        if binding.segment_digest != segment.segment_digest().as_bytes()
            || binding.authority_epoch_digest != segment.authority_epoch_digest()
        {
            return Err(EpochIssuanceEvidenceChainError::StoredSegmentBindingMismatch);
        }
        if binding.previous_epoch_digest != previous_epoch_digest {
            return Err(EpochIssuanceEvidenceChainError::EpochPredecessorMismatch);
        }
        if let Some(previous_generation) = previous_committed_generation {
            if binding.expected_live_generation != previous_generation {
                return Err(EpochIssuanceEvidenceChainError::LiveGenerationDiscontinuity {
                    previous_committed: previous_generation,
                    next_expected: binding.expected_live_generation,
                });
            }
        }
        if let Some(previous_cycle) = previous_activation_cycle {
            if binding.activated_at_cycle <= previous_cycle {
                return Err(EpochIssuanceEvidenceChainError::ActivationCycleNotIncreasing {
                    previous: previous_cycle,
                    current: binding.activated_at_cycle,
                });
            }
        }
        previous_epoch_digest = Some(binding.authority_epoch_digest);
        previous_committed_generation = Some(binding.committed_live_generation);
        previous_activation_cycle = Some(binding.activated_at_cycle);
    }
    Ok(())
}

fn validate_chain_receipt_shape(
    receipt: &VerifiedEpochIssuanceEvidenceChainV1,
    history_chain: &MultiEpochRevisionHistoryChainV2,
) -> Result<(), EpochIssuanceEvidenceChainError> {
    if receipt.version != EpochIssuanceEvidenceChainVersion::V1
        || !receipt.receipt_cursor_continuity_proven
        || !receipt.epoch_sequence_continuity_proven
        || !receipt.epoch_predecessor_continuity_proven
        || !receipt.live_generation_continuity_proven
        || !receipt.activation_cycle_monotonicity_proven
        || !receipt.issuance_evidence_chain_consistent
        || receipt.verifier_correctness_independently_proven
        || receipt.epoch_issuance_chain_verified
        || receipt.active_epoch_registry_constructed
        || receipt.operational_history_constructed
        || receipt.mutation_authority
        || receipt.activation_authorized
    {
        return Err(EpochIssuanceEvidenceChainError::UnexpectedChainAuthority);
    }
    let has_bindings = !receipt.bindings.is_empty();
    if receipt.verifier_current_head_evidence_bound != has_bindings {
        return Err(EpochIssuanceEvidenceChainError::VerifierCurrentnessClaimMismatch);
    }
    if receipt.latest_epoch_sequence != receipt.bindings.last().map(VerifiedEpochIssuanceEvidenceBindingV1::epoch_sequence)
        || receipt.latest_authority_epoch_digest != receipt.bindings.last().map(VerifiedEpochIssuanceEvidenceBindingV1::authority_epoch_digest)
        || receipt.bindings.len() != history_chain.segments().len()
    {
        return Err(EpochIssuanceEvidenceChainError::LatestEpochIdentityMismatch);
    }
    Ok(())
}

fn digest_binding(
    binding: &VerifiedEpochIssuanceEvidenceBindingV1,
) -> Result<EpochIssuanceEvidenceBindingDigestV1, EpochIssuanceEvidenceChainError> {
    validate_binding_shape(binding)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-epoch-issuance-evidence-binding-v1");
    hasher.update(&[1]);
    hasher.update(&binding.validated_at_cycle.to_le_bytes());
    hasher.update(&binding.epoch_sequence.to_le_bytes());
    match binding.previous_epoch_digest {
        Some(digest) => { hasher.update(&[1]); hasher.update(&digest); }
        None => { hasher.update(&[0]); }
    }
    hasher.update(&binding.authority_epoch_digest);
    hasher.update(&binding.issuance_record_digest);
    hasher.update(&binding.activation_commit_receipt_digest);
    hasher.update(&binding.segment_digest);
    hasher.update(&binding.activation_evidence_receipt_digest);
    hasher.update(&binding.trust_review_digest);
    hasher.update(&binding.verifier_currentness_receipt_digest);
    hasher.update(&binding.canonical_verifier_profile_digest.as_bytes());
    hasher.update(&binding.expected_live_generation.to_le_bytes());
    hasher.update(&binding.committed_live_generation.to_le_bytes());
    hasher.update(&binding.activated_at_cycle.to_le_bytes());
    hasher.update(&binding.currentness_attested_at_cycle.to_le_bytes());
    hasher.update(&binding.currentness_expires_at_cycle.to_le_bytes());
    for value in [
        binding.all_activation_phases_provider_accepted,
        binding.verifier_local_policy_accepted,
        binding.verifier_checkpoint_continuity_proven,
        binding.verifier_current_head_proven,
        binding.evidence_binding_consistent,
        binding.verifier_correctness_independently_proven,
        binding.epoch_issuance_verified,
        binding.active_epoch_installed,
        binding.mutation_authority,
        binding.activation_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    Ok(EpochIssuanceEvidenceBindingDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_chain_receipt(
    receipt: &VerifiedEpochIssuanceEvidenceChainV1,
) -> Result<EpochIssuanceEvidenceChainDigestV1, EpochIssuanceEvidenceChainError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-epoch-issuance-evidence-chain-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.captured_at_cycle.to_le_bytes());
    hasher.update(&receipt.multi_epoch_history_chain_digest);
    hasher.update(&receipt.legacy_audit_restoration_digest);
    let count = u64::try_from(receipt.bindings.len())
        .map_err(|_| EpochIssuanceEvidenceChainError::LengthOverflow)?;
    hasher.update(&count.to_le_bytes());
    for binding in &receipt.bindings {
        binding.verify_internal()?;
        hasher.update(&binding.binding_digest.as_bytes());
    }
    hasher.update(&receipt.final_next_receipt_id.to_le_bytes());
    match receipt.latest_epoch_sequence {
        Some(sequence) => { hasher.update(&[1]); hasher.update(&sequence.to_le_bytes()); }
        None => { hasher.update(&[0]); }
    }
    match receipt.latest_authority_epoch_digest {
        Some(digest) => { hasher.update(&[1]); hasher.update(&digest); }
        None => { hasher.update(&[0]); }
    }
    for value in [
        receipt.receipt_cursor_continuity_proven,
        receipt.epoch_sequence_continuity_proven,
        receipt.epoch_predecessor_continuity_proven,
        receipt.live_generation_continuity_proven,
        receipt.activation_cycle_monotonicity_proven,
        receipt.issuance_evidence_chain_consistent,
        receipt.verifier_current_head_evidence_bound,
        receipt.verifier_correctness_independently_proven,
        receipt.epoch_issuance_chain_verified,
        receipt.active_epoch_registry_constructed,
        receipt.operational_history_constructed,
        receipt.mutation_authority,
        receipt.activation_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    Ok(EpochIssuanceEvidenceChainDigestV1(*hasher.finalize().as_bytes()))
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[derive(Debug)]
pub enum EpochIssuanceEvidenceChainError {
    IssuanceRecordInvalid(RestartAuthorityEpochIssuanceRecordError),
    SegmentInvalid(EpochBoundRevisionReceiptSegmentError),
    CurrentnessInvalid(ActivationVerifierCurrentnessError),
    HistoryChainInvalid(MultiEpochRevisionHistoryChainError),
    ActivationEvidenceRecordMismatch,
    UnexpectedActivationEvidenceAuthority,
    TrustReviewEvidenceMismatch,
    UnexpectedTrustReviewAuthority,
    CurrentnessBindingMismatch,
    UnexpectedCurrentnessAuthority,
    BindingPredatesSourceEvidence,
    CurrentnessExpiredBeforeBinding { validated_at_cycle: u64, expires_at_cycle: u64 },
    SegmentEpochMismatch,
    SegmentPredatesActivation { segment_cycle: u64, activated_at_cycle: u64 },
    ReceiptPredatesAuthorityEpoch { receipt_id: u64, evaluated_at_cycle: u64, activated_at_cycle: u64 },
    MalformedBinding,
    GenesisHasPredecessor,
    MissingEpochPredecessor,
    GenerationOverflow,
    InvalidGenerationTransition { expected: u64, actual: u64 },
    InvalidBindingCurrentnessWindow,
    UnexpectedBindingAuthority,
    BindingDigestMismatch,
    UnexpectedHistoryChainAuthority,
    TooManyBindings { actual: usize, maximum: usize },
    BindingCountMismatch { expected: usize, actual: usize },
    CapturePredatesHistoryChain { captured_at_cycle: u64, history_chain_cycle: u64 },
    BindingAfterChainCapture { binding_cycle: u64, chain_cycle: u64 },
    LengthOverflow,
    EpochSequenceOverflow,
    EpochSequenceMismatch { expected: u64, binding: u64, segment: u64 },
    StoredSegmentBindingMismatch,
    EpochPredecessorMismatch,
    LiveGenerationDiscontinuity { previous_committed: u64, next_expected: u64 },
    ActivationCycleNotIncreasing { previous: u64, current: u64 },
    UnexpectedChainAuthority,
    VerifierCurrentnessClaimMismatch,
    LatestEpochIdentityMismatch,
    StoredChainBindingMismatch,
    ChainReceiptDigestMismatch,
}

impl fmt::Display for EpochIssuanceEvidenceChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epoch issuance evidence chain validation failed: {self:?}")
    }
}
impl Error for EpochIssuanceEvidenceChainError {}

#[cfg(test)]
mod tests {
    #[test]
    fn generation_successor_is_checked_without_wrapping() {
        assert_eq!(41u64.checked_add(1), Some(42));
        assert_eq!(u64::MAX.checked_add(1), None);
    }
}
