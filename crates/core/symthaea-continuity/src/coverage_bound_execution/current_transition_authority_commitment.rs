// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rollback-resistant commitment that the exact fresh owner/operator authority head
//! gated one exact already-journaled A -> B execution world.
//!
//! Fresh transition-authority currentness is an in-process proof. A crash-time auditor
//! additionally needs durable evidence that this exact current authority proof was
//! consumed before physical mutation. This module commits that decision under the
//! execution journal's rollback-resistant root without collapsing owner-authority,
//! verifier-currentness, or execution-journal trust roots.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::current_transition_authority::{
    CurrentAuthorizedTransitionAuthorityId, CurrentAuthorizedTransitionAuthorityV1,
    QualifiedTransitionAuthorityCurrentnessId,
};
use super::current_verifier_commitment::{
    CurrentVerifierExecutionBindingV1, QualifiedCurrentVerifierExecutionCommitmentId,
    QualifiedCurrentVerifierExecutionCommitmentV1,
};
use super::{QualifiedEffectCoverageCommitmentId, QualifiedEffectCoverageCommitmentV1};
use crate::commit_eligibility::{CommitEligibleTransitionId, CommitEligibleTransitionV1};
use crate::distributed_state::DistributedStateContextId;
use crate::execution_capability::ExecutionAttemptId;
use crate::execution_journal_anchor::{
    ExecutionJournalAnchorError, ExecutionJournalAnchorProfileId,
    ExecutionJournalAnchorProfileV1, QualifiedExecutionJournalAnchorId,
    QualifiedExecutionJournalAnchorV1,
};
use crate::no_effects_execution::{
    QualifiedNoEffectsExecutionCommitmentId, QualifiedNoEffectsExecutionCommitmentV1,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_authority::{
    AuthenticatedTransitionAuthorityId, TransitionAuthorityClaimId, TransitionAuthorityPolicyId,
    TransitionAuthorityProfileId,
};
use crate::transition_lineage::{
    KnownGoodBoundExecutionAttemptIntentV1, KnownGoodExecutionIntentId,
    KnownGoodTransitionLineageError,
};
use crate::trusted_commit_epoch::QualifiedTrustedCommitEpochId;
use crate::witness::TargetRealizationId;

pub const CURRENT_TRANSITION_AUTHORITY_EXECUTION_COMMITMENT_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-current-transition-authority-execution-commitment-claim-v1";
pub const CURRENT_TRANSITION_AUTHORITY_EXECUTION_COMMITMENT_AUTH_PURPOSE: &str =
    "symthaea.continuity.current-transition-authority-execution-commitment.v1";

const CLAIM_DOMAIN: &[u8] =
    b"symthaea.continuity.current-transition-authority-execution-commitment-claim.v1\0";
const WIRE_DOMAIN: &[u8] =
    b"symthaea.continuity.current-transition-authority-execution-commitment-wire.v1\0";
const AUTH_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-current-transition-authority-execution-commitment.v1\0";
const QUALIFIED_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-current-transition-authority-execution-commitment.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}

digest_id!(CurrentTransitionAuthorityExecutionCommitmentClaimId);
digest_id!(AuthenticatedCurrentTransitionAuthorityExecutionCommitmentId);
digest_id!(QualifiedCurrentTransitionAuthorityExecutionCommitmentId);

/// Lane-specific historical protected world. Both variants bind the exact physical
/// attempt, not merely a transition subject or target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurrentTransitionAuthorityExecutionBindingV1 {
    Effectful {
        historical_commitment_id: QualifiedEffectCoverageCommitmentId,
        attempt_id: ExecutionAttemptId,
    },
    NoEffects {
        historical_commitment_id: QualifiedNoEffectsExecutionCommitmentId,
        attempt_id: ExecutionAttemptId,
    },
}

impl CurrentTransitionAuthorityExecutionBindingV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Effectful { .. } => 1,
            Self::NoEffects { .. } => 2,
        }
    }

    pub fn attempt_id(self) -> ExecutionAttemptId {
        match self {
            Self::Effectful { attempt_id, .. } | Self::NoEffects { attempt_id, .. } => attempt_id,
        }
    }
}

/// Serializable claim over the exact current owner-authority decision consumed by one
/// protected A -> B execution world. This record is audit material, not execution
/// authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentTransitionAuthorityExecutionCommitmentClaimV1 {
    schema_version: String,
    execution_profile_id: ExecutionJournalAnchorProfileId,
    execution_root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    binding: CurrentTransitionAuthorityExecutionBindingV1,
    current_verifier_commitment_id: QualifiedCurrentVerifierExecutionCommitmentId,
    known_good_intent_id: KnownGoodExecutionIntentId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    current_authority_id: CurrentAuthorizedTransitionAuthorityId,
    authority_currentness_id: QualifiedTransitionAuthorityCurrentnessId,
    commit_eligibility_id: CommitEligibleTransitionId,
    authority_id: AuthenticatedTransitionAuthorityId,
    authority_claim_id: TransitionAuthorityClaimId,
    authority_policy_id: TransitionAuthorityPolicyId,
    authority_policy_generation: u64,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    authority_basis_digest: [u8; 32],
    decision_time_unix_ms: u64,
    raw_commitment_evidence_digest: [u8; 32],
    claim_id: CurrentTransitionAuthorityExecutionCommitmentClaimId,
}

impl CurrentTransitionAuthorityExecutionCommitmentClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn for_effectful(
        profile: &ExecutionJournalAnchorProfileV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical: &QualifiedEffectCoverageCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        eligibility: &CommitEligibleTransitionV1,
        current_authority: &CurrentAuthorizedTransitionAuthorityV1,
        raw_commitment_evidence_digest: [u8; 32],
    ) -> Result<Self, CurrentTransitionAuthorityExecutionCommitmentError> {
        let binding = CurrentTransitionAuthorityExecutionBindingV1::Effectful {
            historical_commitment_id: historical.id(),
            attempt_id: historical.attempt_id(),
        };
        require_current_verifier_effectful(current_verifier_commitment, historical)?;
        Self::new_common(
            profile,
            journal_anchor,
            binding,
            historical.journal_anchor_id(),
            current_verifier_commitment,
            intent,
            eligibility,
            current_authority,
            raw_commitment_evidence_digest,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn for_no_effects(
        profile: &ExecutionJournalAnchorProfileV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical: &QualifiedNoEffectsExecutionCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        eligibility: &CommitEligibleTransitionV1,
        current_authority: &CurrentAuthorizedTransitionAuthorityV1,
        raw_commitment_evidence_digest: [u8; 32],
    ) -> Result<Self, CurrentTransitionAuthorityExecutionCommitmentError> {
        let binding = CurrentTransitionAuthorityExecutionBindingV1::NoEffects {
            historical_commitment_id: historical.id(),
            attempt_id: historical.attempt_id(),
        };
        require_current_verifier_no_effects(current_verifier_commitment, historical)?;
        Self::new_common(
            profile,
            journal_anchor,
            binding,
            historical.journal_anchor_id(),
            current_verifier_commitment,
            intent,
            eligibility,
            current_authority,
            raw_commitment_evidence_digest,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_common(
        profile: &ExecutionJournalAnchorProfileV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        binding: CurrentTransitionAuthorityExecutionBindingV1,
        historical_journal_anchor_id: QualifiedExecutionJournalAnchorId,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        eligibility: &CommitEligibleTransitionV1,
        current_authority: &CurrentAuthorizedTransitionAuthorityV1,
        raw_commitment_evidence_digest: [u8; 32],
    ) -> Result<Self, CurrentTransitionAuthorityExecutionCommitmentError> {
        profile.validate()?;
        intent.validate()?;
        if raw_commitment_evidence_digest == [0; 32] || eligibility.authority_basis_digest() == [0; 32] {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::ZeroDigest);
        }
        if profile.id() != journal_anchor.profile_id()
            || profile.root_epoch() != journal_anchor.root_epoch()
            || historical_journal_anchor_id != journal_anchor.id()
            || current_verifier_commitment.journal_anchor_id() != journal_anchor.id()
        {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::ProtectedWorldMismatch);
        }
        require_authority_matches_eligibility(current_authority, eligibility)?;
        require_intent_matches_eligibility(intent, eligibility, binding)?;
        if eligibility.commit_time_unix_ms() != journal_anchor.anchored_at_unix_ms()
            || eligibility.commit_time_unix_ms() != current_verifier_commitment.decision_time_unix_ms()
        {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::DecisionBoundaryMismatch);
        }

        let lineage = intent.lineage();
        let claim_id = CurrentTransitionAuthorityExecutionCommitmentClaimId(hash_claim(
            profile.id(),
            profile.root_epoch(),
            journal_anchor.id(),
            journal_anchor.trusted_epoch_id(),
            binding,
            current_verifier_commitment.id(),
            intent.id(),
            lineage.subject_id(),
            lineage.source_realization_id(),
            lineage.target_realization_id(),
            lineage.distributed_context_id(),
            current_authority.id(),
            current_authority.currentness_id(),
            eligibility.id(),
            eligibility.authority_id(),
            eligibility.authority_claim_id(),
            eligibility.authority_policy_id(),
            eligibility.authority_policy_generation(),
            eligibility.authority_profile_id(),
            eligibility.authority_root_epoch(),
            eligibility.authority_basis_digest(),
            eligibility.commit_time_unix_ms(),
            raw_commitment_evidence_digest,
        ));
        Ok(Self {
            schema_version:
                CURRENT_TRANSITION_AUTHORITY_EXECUTION_COMMITMENT_CLAIM_SCHEMA_V1.to_owned(),
            execution_profile_id: profile.id(),
            execution_root_epoch: profile.root_epoch(),
            journal_anchor_id: journal_anchor.id(),
            trusted_epoch_id: journal_anchor.trusted_epoch_id(),
            binding,
            current_verifier_commitment_id: current_verifier_commitment.id(),
            known_good_intent_id: intent.id(),
            subject_id: lineage.subject_id(),
            source_realization_id: lineage.source_realization_id(),
            target_realization_id: lineage.target_realization_id(),
            distributed_context_id: lineage.distributed_context_id(),
            current_authority_id: current_authority.id(),
            authority_currentness_id: current_authority.currentness_id(),
            commit_eligibility_id: eligibility.id(),
            authority_id: eligibility.authority_id(),
            authority_claim_id: eligibility.authority_claim_id(),
            authority_policy_id: eligibility.authority_policy_id(),
            authority_policy_generation: eligibility.authority_policy_generation(),
            authority_profile_id: eligibility.authority_profile_id(),
            authority_root_epoch: eligibility.authority_root_epoch(),
            authority_basis_digest: eligibility.authority_basis_digest(),
            decision_time_unix_ms: eligibility.commit_time_unix_ms(),
            raw_commitment_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), CurrentTransitionAuthorityExecutionCommitmentError> {
        if self.schema_version
            != CURRENT_TRANSITION_AUTHORITY_EXECUTION_COMMITMENT_CLAIM_SCHEMA_V1
        {
            return Err(
                CurrentTransitionAuthorityExecutionCommitmentError::UnsupportedClaimSchema(
                    self.schema_version.clone(),
                ),
            );
        }
        if self.execution_root_epoch == 0
            || self.authority_policy_generation == 0
            || self.authority_root_epoch == 0
            || self.decision_time_unix_ms == 0
        {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::ZeroGenerationOrTime);
        }
        if self.authority_basis_digest == [0; 32]
            || self.raw_commitment_evidence_digest == [0; 32]
        {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::ZeroDigest);
        }
        if self.source_realization_id == self.target_realization_id {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::IntentMismatch);
        }
        let expected = CurrentTransitionAuthorityExecutionCommitmentClaimId(hash_claim(
            self.execution_profile_id,
            self.execution_root_epoch,
            self.journal_anchor_id,
            self.trusted_epoch_id,
            self.binding,
            self.current_verifier_commitment_id,
            self.known_good_intent_id,
            self.subject_id,
            self.source_realization_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.current_authority_id,
            self.authority_currentness_id,
            self.commit_eligibility_id,
            self.authority_id,
            self.authority_claim_id,
            self.authority_policy_id,
            self.authority_policy_generation,
            self.authority_profile_id,
            self.authority_root_epoch,
            self.authority_basis_digest,
            self.decision_time_unix_ms,
            self.raw_commitment_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> CurrentTransitionAuthorityExecutionCommitmentClaimId { self.claim_id }
    pub fn binding(&self) -> CurrentTransitionAuthorityExecutionBindingV1 { self.binding }
    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId { self.journal_anchor_id }
    pub fn current_verifier_commitment_id(&self) -> QualifiedCurrentVerifierExecutionCommitmentId {
        self.current_verifier_commitment_id
    }
    pub fn current_authority_id(&self) -> CurrentAuthorizedTransitionAuthorityId {
        self.current_authority_id
    }
    pub fn authority_currentness_id(&self) -> QualifiedTransitionAuthorityCurrentnessId {
        self.authority_currentness_id
    }
    pub fn decision_time_unix_ms(&self) -> u64 { self.decision_time_unix_ms }
}

pub fn canonical_current_transition_authority_execution_commitment_claim_bytes(
    claim: &CurrentTransitionAuthorityExecutionCommitmentClaimV1,
) -> Result<Vec<u8>, CurrentTransitionAuthorityExecutionCommitmentError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(1024);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.execution_profile_id.as_bytes());
    out.extend_from_slice(&claim.execution_root_epoch.to_le_bytes());
    out.extend_from_slice(claim.journal_anchor_id.as_bytes());
    out.extend_from_slice(claim.trusted_epoch_id.as_bytes());
    encode_binding(&mut out, claim.binding);
    out.extend_from_slice(claim.current_verifier_commitment_id.as_bytes());
    out.extend_from_slice(claim.known_good_intent_id.as_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.source_realization_id.as_bytes());
    out.extend_from_slice(claim.target_realization_id.as_bytes());
    out.extend_from_slice(claim.distributed_context_id.as_bytes());
    out.extend_from_slice(claim.current_authority_id.as_bytes());
    out.extend_from_slice(claim.authority_currentness_id.as_bytes());
    out.extend_from_slice(claim.commit_eligibility_id.as_bytes());
    out.extend_from_slice(claim.authority_id.as_bytes());
    out.extend_from_slice(claim.authority_claim_id.as_bytes());
    out.extend_from_slice(claim.authority_policy_id.as_bytes());
    out.extend_from_slice(&claim.authority_policy_generation.to_le_bytes());
    out.extend_from_slice(claim.authority_profile_id.as_bytes());
    out.extend_from_slice(&claim.authority_root_epoch.to_le_bytes());
    out.extend_from_slice(&claim.authority_basis_digest);
    out.extend_from_slice(&claim.decision_time_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_commitment_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_current_transition_authority_execution_commitment_claim_digest(
    claim: &CurrentTransitionAuthorityExecutionCommitmentClaimV1,
) -> Result<[u8; 32], CurrentTransitionAuthorityExecutionCommitmentError> {
    Ok(*blake3::hash(
        &canonical_current_transition_authority_execution_commitment_claim_bytes(claim)?,
    )
    .as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedCurrentTransitionAuthorityExecutionCommitmentV1 {
    claim: CurrentTransitionAuthorityExecutionCommitmentClaimV1,
    profile: ExecutionJournalAnchorProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedCurrentTransitionAuthorityExecutionCommitmentId,
}

impl AuthenticatedCurrentTransitionAuthorityExecutionCommitmentV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: CurrentTransitionAuthorityExecutionCommitmentClaimV1,
        profile: ExecutionJournalAnchorProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, CurrentTransitionAuthorityExecutionCommitmentError> {
        claim.validate()?;
        profile.validate()?;
        if authentication_evidence_digest == [0; 32] {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::ZeroDigest);
        }
        if claim.execution_profile_id != profile.id()
            || claim.execution_root_epoch != profile.root_epoch()
        {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::CommitmentRootMismatch);
        }
        let evidence_id = AuthenticatedCurrentTransitionAuthorityExecutionCommitmentId(
            domain_hash_parts(
                AUTH_DOMAIN,
                &[
                    claim.id().as_bytes(),
                    profile.id().as_bytes(),
                    &profile.root_epoch().to_le_bytes(),
                    &authentication_evidence_digest,
                ],
            ),
        );
        Ok(Self { claim, profile, authentication_evidence_digest, evidence_id })
    }
}

/// Non-Serde proof that the exact fresh current owner-authority decision was durably
/// bound to the same A -> B protected world as the current-verifier decision.
#[derive(Debug, Clone)]
pub struct QualifiedCurrentTransitionAuthorityExecutionCommitmentV1 {
    commitment_id: QualifiedCurrentTransitionAuthorityExecutionCommitmentId,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    binding: CurrentTransitionAuthorityExecutionBindingV1,
    current_verifier_commitment_id: QualifiedCurrentVerifierExecutionCommitmentId,
    known_good_intent_id: KnownGoodExecutionIntentId,
    current_authority_id: CurrentAuthorizedTransitionAuthorityId,
    authority_currentness_id: QualifiedTransitionAuthorityCurrentnessId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    decision_time_unix_ms: u64,
}

impl QualifiedCurrentTransitionAuthorityExecutionCommitmentV1 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qualify_effectful(
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical: &QualifiedEffectCoverageCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        eligibility: &CommitEligibleTransitionV1,
        current_authority: &CurrentAuthorizedTransitionAuthorityV1,
        authenticated: &AuthenticatedCurrentTransitionAuthorityExecutionCommitmentV1,
    ) -> Result<Self, CurrentTransitionAuthorityExecutionCommitmentError> {
        require_current_verifier_effectful(current_verifier_commitment, historical)?;
        let binding = CurrentTransitionAuthorityExecutionBindingV1::Effectful {
            historical_commitment_id: historical.id(),
            attempt_id: historical.attempt_id(),
        };
        Self::qualify_common(
            journal_anchor,
            historical.journal_anchor_id(),
            binding,
            current_verifier_commitment,
            intent,
            eligibility,
            current_authority,
            authenticated,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qualify_no_effects(
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical: &QualifiedNoEffectsExecutionCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        eligibility: &CommitEligibleTransitionV1,
        current_authority: &CurrentAuthorizedTransitionAuthorityV1,
        authenticated: &AuthenticatedCurrentTransitionAuthorityExecutionCommitmentV1,
    ) -> Result<Self, CurrentTransitionAuthorityExecutionCommitmentError> {
        require_current_verifier_no_effects(current_verifier_commitment, historical)?;
        let binding = CurrentTransitionAuthorityExecutionBindingV1::NoEffects {
            historical_commitment_id: historical.id(),
            attempt_id: historical.attempt_id(),
        };
        Self::qualify_common(
            journal_anchor,
            historical.journal_anchor_id(),
            binding,
            current_verifier_commitment,
            intent,
            eligibility,
            current_authority,
            authenticated,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn qualify_common(
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        historical_journal_anchor_id: QualifiedExecutionJournalAnchorId,
        expected_binding: CurrentTransitionAuthorityExecutionBindingV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        eligibility: &CommitEligibleTransitionV1,
        current_authority: &CurrentAuthorizedTransitionAuthorityV1,
        authenticated: &AuthenticatedCurrentTransitionAuthorityExecutionCommitmentV1,
    ) -> Result<Self, CurrentTransitionAuthorityExecutionCommitmentError> {
        authenticated.claim.validate()?;
        authenticated.profile.validate()?;
        intent.validate()?;
        require_authority_matches_eligibility(current_authority, eligibility)?;
        require_intent_matches_eligibility(intent, eligibility, expected_binding)?;
        let claim = &authenticated.claim;
        if authenticated.profile.id() != journal_anchor.profile_id()
            || authenticated.profile.root_epoch() != journal_anchor.root_epoch()
            || claim.execution_profile_id != journal_anchor.profile_id()
            || claim.execution_root_epoch != journal_anchor.root_epoch()
            || claim.journal_anchor_id != journal_anchor.id()
            || claim.trusted_epoch_id != journal_anchor.trusted_epoch_id()
            || historical_journal_anchor_id != journal_anchor.id()
        {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::ProtectedWorldMismatch);
        }
        let lineage = intent.lineage();
        if claim.binding != expected_binding
            || claim.current_verifier_commitment_id != current_verifier_commitment.id()
            || claim.known_good_intent_id != intent.id()
            || claim.subject_id != lineage.subject_id()
            || claim.source_realization_id != lineage.source_realization_id()
            || claim.target_realization_id != lineage.target_realization_id()
            || claim.distributed_context_id != lineage.distributed_context_id()
            || claim.current_authority_id != current_authority.id()
            || claim.authority_currentness_id != current_authority.currentness_id()
            || claim.commit_eligibility_id != eligibility.id()
            || claim.authority_id != eligibility.authority_id()
            || claim.authority_claim_id != eligibility.authority_claim_id()
            || claim.authority_policy_id != eligibility.authority_policy_id()
            || claim.authority_policy_generation != eligibility.authority_policy_generation()
            || claim.authority_profile_id != eligibility.authority_profile_id()
            || claim.authority_root_epoch != eligibility.authority_root_epoch()
            || claim.authority_basis_digest != eligibility.authority_basis_digest()
        {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::AuthorityWorldMismatch);
        }
        if claim.decision_time_unix_ms != eligibility.commit_time_unix_ms()
            || claim.decision_time_unix_ms != journal_anchor.anchored_at_unix_ms()
            || claim.decision_time_unix_ms != current_verifier_commitment.decision_time_unix_ms()
        {
            return Err(CurrentTransitionAuthorityExecutionCommitmentError::DecisionBoundaryMismatch);
        }
        let commitment_id = QualifiedCurrentTransitionAuthorityExecutionCommitmentId(
            domain_hash_parts(
                QUALIFIED_DOMAIN,
                &[
                    claim.id().as_bytes(),
                    journal_anchor.id().as_bytes(),
                    authenticated.evidence_id.as_bytes(),
                    current_verifier_commitment.id().as_bytes(),
                    current_authority.id().as_bytes(),
                    intent.id().as_bytes(),
                ],
            ),
        );
        Ok(Self {
            commitment_id,
            journal_anchor_id: journal_anchor.id(),
            binding: expected_binding,
            current_verifier_commitment_id: current_verifier_commitment.id(),
            known_good_intent_id: intent.id(),
            current_authority_id: current_authority.id(),
            authority_currentness_id: current_authority.currentness_id(),
            subject_id: lineage.subject_id(),
            source_realization_id: lineage.source_realization_id(),
            target_realization_id: lineage.target_realization_id(),
            decision_time_unix_ms: eligibility.commit_time_unix_ms(),
        })
    }

    pub fn id(&self) -> QualifiedCurrentTransitionAuthorityExecutionCommitmentId {
        self.commitment_id
    }
    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId { self.journal_anchor_id }
    pub fn binding(&self) -> CurrentTransitionAuthorityExecutionBindingV1 { self.binding }
    pub fn current_verifier_commitment_id(&self) -> QualifiedCurrentVerifierExecutionCommitmentId {
        self.current_verifier_commitment_id
    }
    pub fn known_good_intent_id(&self) -> KnownGoodExecutionIntentId { self.known_good_intent_id }
    pub fn current_authority_id(&self) -> CurrentAuthorizedTransitionAuthorityId {
        self.current_authority_id
    }
    pub fn authority_currentness_id(&self) -> QualifiedTransitionAuthorityCurrentnessId {
        self.authority_currentness_id
    }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.source_realization_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn decision_time_unix_ms(&self) -> u64 { self.decision_time_unix_ms }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CurrentTransitionAuthorityExecutionCommitmentError {
    #[error(transparent)]
    JournalAnchor(#[from] ExecutionJournalAnchorError),
    #[error(transparent)]
    KnownGoodLineage(#[from] KnownGoodTransitionLineageError),
    #[error("unsupported current-transition-authority execution commitment schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("current-transition-authority execution commitment digest must be non-zero")]
    ZeroDigest,
    #[error("current-transition-authority execution commitment generation/time must be non-zero")]
    ZeroGenerationOrTime,
    #[error("current-transition-authority execution commitment changed protected journal world")]
    ProtectedWorldMismatch,
    #[error("current-verifier commitment does not bind exact historical execution lane")]
    CurrentVerifierWorldMismatch,
    #[error("fresh current transition authority does not match exact commit eligibility")]
    CurrentAuthorityMismatch,
    #[error("durable A -> B intent does not match exact commit eligibility/historical attempt")]
    IntentMismatch,
    #[error("authenticated commitment fields differ from exact current authority world")]
    AuthorityWorldMismatch,
    #[error("current-transition-authority decision boundary is not exact")]
    DecisionBoundaryMismatch,
    #[error("current-transition-authority execution commitment identity mismatch")]
    ClaimIdentityMismatch,
    #[error("current-transition-authority execution commitment changed execution root")]
    CommitmentRootMismatch,
}

fn require_authority_matches_eligibility(
    current_authority: &CurrentAuthorizedTransitionAuthorityV1,
    eligibility: &CommitEligibleTransitionV1,
) -> Result<(), CurrentTransitionAuthorityExecutionCommitmentError> {
    if current_authority.commit_eligibility_id() != eligibility.id()
        || current_authority.authority_claim_id() != eligibility.authority_claim_id()
        || current_authority.authority_profile_id() != eligibility.authority_profile_id()
        || current_authority.subject_id() != eligibility.subject_id()
        || current_authority.target_realization_id() != eligibility.target_realization_id()
        || current_authority.distributed_context_id() != eligibility.distributed_context_id()
        || current_authority.decision_time_unix_ms() != eligibility.commit_time_unix_ms()
    {
        return Err(CurrentTransitionAuthorityExecutionCommitmentError::CurrentAuthorityMismatch);
    }
    Ok(())
}

fn require_intent_matches_eligibility(
    intent: &KnownGoodBoundExecutionAttemptIntentV1,
    eligibility: &CommitEligibleTransitionV1,
    binding: CurrentTransitionAuthorityExecutionBindingV1,
) -> Result<(), CurrentTransitionAuthorityExecutionCommitmentError> {
    let lineage = intent.lineage();
    if intent.attempt_id() != binding.attempt_id()
        || lineage.subject_id() != eligibility.subject_id()
        || lineage.target_realization_id() != eligibility.target_realization_id()
        || lineage.distributed_context_id() != eligibility.distributed_context_id()
        || lineage.commit_time_unix_ms() != eligibility.commit_time_unix_ms()
    {
        return Err(CurrentTransitionAuthorityExecutionCommitmentError::IntentMismatch);
    }
    Ok(())
}

fn require_current_verifier_effectful(
    current: &QualifiedCurrentVerifierExecutionCommitmentV1,
    historical: &QualifiedEffectCoverageCommitmentV1,
) -> Result<(), CurrentTransitionAuthorityExecutionCommitmentError> {
    match current.binding() {
        CurrentVerifierExecutionBindingV1::Effectful {
            historical_commitment_id,
            historical_coverage_id,
            ..
        } if historical_commitment_id == historical.id()
            && historical_coverage_id == historical.coverage_id() => Ok(()),
        _ => Err(CurrentTransitionAuthorityExecutionCommitmentError::CurrentVerifierWorldMismatch),
    }
}

fn require_current_verifier_no_effects(
    current: &QualifiedCurrentVerifierExecutionCommitmentV1,
    historical: &QualifiedNoEffectsExecutionCommitmentV1,
) -> Result<(), CurrentTransitionAuthorityExecutionCommitmentError> {
    match current.binding() {
        CurrentVerifierExecutionBindingV1::NoEffects {
            historical_commitment_id,
            historical_coverage_id,
            ..
        } if historical_commitment_id == historical.id()
            && historical_coverage_id == historical.coverage_id() => Ok(()),
        _ => Err(CurrentTransitionAuthorityExecutionCommitmentError::CurrentVerifierWorldMismatch),
    }
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    execution_profile_id: ExecutionJournalAnchorProfileId,
    execution_root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    binding: CurrentTransitionAuthorityExecutionBindingV1,
    current_verifier_commitment_id: QualifiedCurrentVerifierExecutionCommitmentId,
    known_good_intent_id: KnownGoodExecutionIntentId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    current_authority_id: CurrentAuthorizedTransitionAuthorityId,
    authority_currentness_id: QualifiedTransitionAuthorityCurrentnessId,
    commit_eligibility_id: CommitEligibleTransitionId,
    authority_id: AuthenticatedTransitionAuthorityId,
    authority_claim_id: TransitionAuthorityClaimId,
    authority_policy_id: TransitionAuthorityPolicyId,
    authority_policy_generation: u64,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    authority_basis_digest: [u8; 32],
    decision_time_unix_ms: u64,
    raw_commitment_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(CLAIM_DOMAIN);
    h.update(execution_profile_id.as_bytes());
    h.update(&execution_root_epoch.to_le_bytes());
    h.update(journal_anchor_id.as_bytes());
    h.update(trusted_epoch_id.as_bytes());
    hash_binding(&mut h, binding);
    h.update(current_verifier_commitment_id.as_bytes());
    h.update(known_good_intent_id.as_bytes());
    h.update(subject_id.as_bytes());
    h.update(source_realization_id.as_bytes());
    h.update(target_realization_id.as_bytes());
    h.update(distributed_context_id.as_bytes());
    h.update(current_authority_id.as_bytes());
    h.update(authority_currentness_id.as_bytes());
    h.update(commit_eligibility_id.as_bytes());
    h.update(authority_id.as_bytes());
    h.update(authority_claim_id.as_bytes());
    h.update(authority_policy_id.as_bytes());
    h.update(&authority_policy_generation.to_le_bytes());
    h.update(authority_profile_id.as_bytes());
    h.update(&authority_root_epoch.to_le_bytes());
    h.update(&authority_basis_digest);
    h.update(&decision_time_unix_ms.to_le_bytes());
    h.update(&raw_commitment_evidence_digest);
    *h.finalize().as_bytes()
}

fn hash_binding(h: &mut blake3::Hasher, binding: CurrentTransitionAuthorityExecutionBindingV1) {
    h.update(&[binding.tag()]);
    match binding {
        CurrentTransitionAuthorityExecutionBindingV1::Effectful {
            historical_commitment_id,
            attempt_id,
        } => {
            h.update(historical_commitment_id.as_bytes());
            h.update(attempt_id.as_bytes());
        }
        CurrentTransitionAuthorityExecutionBindingV1::NoEffects {
            historical_commitment_id,
            attempt_id,
        } => {
            h.update(historical_commitment_id.as_bytes());
            h.update(attempt_id.as_bytes());
        }
    }
}

fn encode_binding(out: &mut Vec<u8>, binding: CurrentTransitionAuthorityExecutionBindingV1) {
    out.push(binding.tag());
    match binding {
        CurrentTransitionAuthorityExecutionBindingV1::Effectful {
            historical_commitment_id,
            attempt_id,
        } => {
            out.extend_from_slice(historical_commitment_id.as_bytes());
            out.extend_from_slice(attempt_id.as_bytes());
        }
        CurrentTransitionAuthorityExecutionBindingV1::NoEffects {
            historical_commitment_id,
            attempt_id,
        } => {
            out.extend_from_slice(historical_commitment_id.as_bytes());
            out.extend_from_slice(attempt_id.as_bytes());
        }
    }
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    for part in parts {
        h.update(&((*part).len() as u64).to_le_bytes());
        h.update(part);
    }
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_transition_authority_commitment_domains_are_distinct() {
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
        assert_ne!(WIRE_DOMAIN, AUTH_DOMAIN);
        assert_ne!(AUTH_DOMAIN, QUALIFIED_DOMAIN);
    }
}
