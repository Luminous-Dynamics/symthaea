// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh transition-authority currentness at the final physical decision boundary.
//!
//! A historically authenticated owner/operator grant can later be revoked or
//! superseded. Grant validity therefore does not prove current authority.
//!
//! `AuthenticatedGrant != ValidAtCommit != CurrentGrantHead != PhysicalAuthority`.
//!
//! V1 deliberately has no freshness TTL. The exact grant head must be attested with
//! a fresh challenge at the same exact decision time already used by continuity,
//! verifier-currentness, and the protected execution world.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::current_authorized::{
    CurrentAuthorizedCoverageError, CurrentAuthorizedEffectExecutionEligibilityV1,
    CurrentAuthorizedNoEffectsExecutionEligibilityV1,
    PendingCurrentAuthorizedEffectExecutionV1, PendingCurrentAuthorizedNoEffectsExecutionV1,
    ReadyCurrentAuthorizedEffectExecutionV1, ReadyCurrentAuthorizedNoEffectsExecutionV1,
    prepare_current_authorized_effect_execution,
    prepare_current_authorized_no_effects_execution,
};
use super::current_verifier_commitment::QualifiedCurrentVerifierExecutionCommitmentV1;
use super::QualifiedEffectCoverageCommitmentV1;
use crate::backend_bound_execution::QualifiedBackendEffectCommitmentV1;
use crate::commit_eligibility::{CommitEligibleTransitionId, CommitEligibleTransitionV1};
use crate::distributed_state::DistributedStateContextId;
use crate::effect_scoped_execution::QualifiedEffectScopeCommitmentV1;
use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptOutcomeV1, ExecutionAttemptReceiptV1,
    ExecutionBackendId, ExecutionEpochAnchorModeV1,
};
use crate::execution_journal::ReconstructedExecutionJournalV1;
use crate::execution_journal_anchor::QualifiedExecutionJournalAnchorV1;
use crate::external_effect_authority::{ExternalEffectAuthorityError, ExternalEffectPlanV1};
use crate::external_effects::ExternalEffectContractV1;
use crate::no_effects_execution::QualifiedNoEffectsExecutionCommitmentV1;
use crate::no_external_effects::{NoExternalEffectsDeclarationV1, NoExternalEffectsError};
use crate::scope::ContinuitySubjectId;
use crate::transition_authority::{
    AuthenticatedTransitionAuthorityId, TransitionAuthorityClaimId, TransitionAuthorityError,
    TransitionAuthorityPolicyId, TransitionAuthorityProfileId, TransitionAuthorityProfileV1,
};
use crate::trusted_commit_epoch::QualifiedTrustedCommitEpochV1;
use crate::witness::TargetRealizationId;

pub const TRANSITION_AUTHORITY_CURRENTNESS_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-transition-authority-currentness-claim-v1";
pub const TRANSITION_AUTHORITY_CURRENTNESS_AUTH_PURPOSE: &str =
    "symthaea.continuity.transition-authority-currentness.v1";

const TRUSTED_ROOT_DOMAIN: &[u8] =
    b"symthaea.continuity.trusted-transition-authority-currentness-root.v1\0";
const CLAIM_DOMAIN: &[u8] =
    b"symthaea.continuity.transition-authority-currentness-claim.v1\0";
const WIRE_DOMAIN: &[u8] =
    b"symthaea.continuity.transition-authority-currentness-wire.v1\0";
const AUTH_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-transition-authority-currentness.v1\0";
const QUALIFIED_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-transition-authority-currentness.v1\0";
const CURRENT_DOMAIN: &[u8] =
    b"symthaea.continuity.current-authorized-transition-authority.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}

digest_id!(TrustedTransitionAuthorityCurrentnessRootId);
digest_id!(TransitionAuthorityCurrentnessClaimId);
digest_id!(AuthenticatedTransitionAuthorityCurrentnessId);
digest_id!(QualifiedTransitionAuthorityCurrentnessId);
digest_id!(CurrentAuthorizedTransitionAuthorityId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionAuthorityDispositionV1 {
    Active,
    Revoked,
    Superseded { successor_claim_id: TransitionAuthorityClaimId },
}

impl TransitionAuthorityDispositionV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Active => 1,
            Self::Revoked => 2,
            Self::Superseded { .. } => 3,
        }
    }

    fn terminal(self) -> bool {
        !matches!(self, Self::Active)
    }
}

/// Provisioned trust root for grant-currentness authentication. It has no public V1
/// constructor: a serializable authority profile is configuration, not trust.
#[derive(Debug)]
pub struct TrustedTransitionAuthorityCurrentnessRootV1 {
    root_id: TrustedTransitionAuthorityCurrentnessRootId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_digest: [u8; 32],
    authority_root_epoch: u64,
}

impl TrustedTransitionAuthorityCurrentnessRootV1 {
    #[cfg(test)]
    pub(crate) fn provision_for_test(
        profile: &TransitionAuthorityProfileV1,
    ) -> Result<Self, TransitionAuthorityCurrentnessError> {
        profile.validate()?;
        let root_id = TrustedTransitionAuthorityCurrentnessRootId(domain_hash_parts(
            TRUSTED_ROOT_DOMAIN,
            &[
                profile.id().as_bytes(),
                &profile.root_digest(),
                &profile.root_epoch().to_le_bytes(),
            ],
        ));
        Ok(Self {
            root_id,
            authority_profile_id: profile.id(),
            authority_root_digest: profile.root_digest(),
            authority_root_epoch: profile.root_epoch(),
        })
    }

    pub fn id(&self) -> TrustedTransitionAuthorityCurrentnessRootId { self.root_id }
    pub fn authority_profile_id(&self) -> TransitionAuthorityProfileId { self.authority_profile_id }
    pub fn authority_root_epoch(&self) -> u64 { self.authority_root_epoch }
}

/// Transportable current-head claim for one exact already-authenticated transition
/// grant. The claim itself has no authority until authenticated against the provisioned
/// currentness root and qualified against exact eligibility lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionAuthorityCurrentnessClaimV1 {
    schema_version: String,
    commit_eligibility_id: CommitEligibleTransitionId,
    authority_id: AuthenticatedTransitionAuthorityId,
    authority_claim_id: TransitionAuthorityClaimId,
    authority_policy_id: TransitionAuthorityPolicyId,
    authority_policy_generation: u64,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    authority_basis_digest: [u8; 32],
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    sequence: u64,
    predecessor_currentness_id: Option<QualifiedTransitionAuthorityCurrentnessId>,
    disposition: TransitionAuthorityDispositionV1,
    freshness_challenge_digest: [u8; 32],
    monotonic_counter: u64,
    observed_at_unix_ms: u64,
    raw_evidence_digest: [u8; 32],
    claim_id: TransitionAuthorityCurrentnessClaimId,
}

impl TransitionAuthorityCurrentnessClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eligibility: &CommitEligibleTransitionV1,
        sequence: u64,
        predecessor_currentness_id: Option<QualifiedTransitionAuthorityCurrentnessId>,
        disposition: TransitionAuthorityDispositionV1,
        freshness_challenge_digest: [u8; 32],
        monotonic_counter: u64,
        observed_at_unix_ms: u64,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, TransitionAuthorityCurrentnessError> {
        validate_material(
            sequence,
            freshness_challenge_digest,
            monotonic_counter,
            observed_at_unix_ms,
            raw_evidence_digest,
        )?;
        validate_disposition(disposition, eligibility.authority_claim_id())?;
        let claim_id = TransitionAuthorityCurrentnessClaimId(hash_claim(
            eligibility.id(),
            eligibility.authority_id(),
            eligibility.authority_claim_id(),
            eligibility.authority_policy_id(),
            eligibility.authority_policy_generation(),
            eligibility.authority_profile_id(),
            eligibility.authority_root_epoch(),
            eligibility.authority_basis_digest(),
            eligibility.subject_id(),
            eligibility.target_realization_id(),
            eligibility.distributed_context_id(),
            sequence,
            predecessor_currentness_id,
            disposition,
            freshness_challenge_digest,
            monotonic_counter,
            observed_at_unix_ms,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: TRANSITION_AUTHORITY_CURRENTNESS_CLAIM_SCHEMA_V1.to_owned(),
            commit_eligibility_id: eligibility.id(),
            authority_id: eligibility.authority_id(),
            authority_claim_id: eligibility.authority_claim_id(),
            authority_policy_id: eligibility.authority_policy_id(),
            authority_policy_generation: eligibility.authority_policy_generation(),
            authority_profile_id: eligibility.authority_profile_id(),
            authority_root_epoch: eligibility.authority_root_epoch(),
            authority_basis_digest: eligibility.authority_basis_digest(),
            subject_id: eligibility.subject_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            sequence,
            predecessor_currentness_id,
            disposition,
            freshness_challenge_digest,
            monotonic_counter,
            observed_at_unix_ms,
            raw_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), TransitionAuthorityCurrentnessError> {
        if self.schema_version != TRANSITION_AUTHORITY_CURRENTNESS_CLAIM_SCHEMA_V1 {
            return Err(TransitionAuthorityCurrentnessError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        if self.authority_policy_generation == 0 || self.authority_root_epoch == 0 {
            return Err(TransitionAuthorityCurrentnessError::ZeroGeneration);
        }
        if self.authority_basis_digest == [0; 32] {
            return Err(TransitionAuthorityCurrentnessError::ZeroDigest);
        }
        validate_material(
            self.sequence,
            self.freshness_challenge_digest,
            self.monotonic_counter,
            self.observed_at_unix_ms,
            self.raw_evidence_digest,
        )?;
        validate_disposition(self.disposition, self.authority_claim_id)?;
        let expected = TransitionAuthorityCurrentnessClaimId(hash_claim(
            self.commit_eligibility_id,
            self.authority_id,
            self.authority_claim_id,
            self.authority_policy_id,
            self.authority_policy_generation,
            self.authority_profile_id,
            self.authority_root_epoch,
            self.authority_basis_digest,
            self.subject_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.sequence,
            self.predecessor_currentness_id,
            self.disposition,
            self.freshness_challenge_digest,
            self.monotonic_counter,
            self.observed_at_unix_ms,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(TransitionAuthorityCurrentnessError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> TransitionAuthorityCurrentnessClaimId { self.claim_id }
}

pub fn canonical_transition_authority_currentness_claim_bytes(
    claim: &TransitionAuthorityCurrentnessClaimV1,
) -> Result<Vec<u8>, TransitionAuthorityCurrentnessError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(704);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.commit_eligibility_id.as_bytes());
    out.extend_from_slice(claim.authority_id.as_bytes());
    out.extend_from_slice(claim.authority_claim_id.as_bytes());
    out.extend_from_slice(claim.authority_policy_id.as_bytes());
    out.extend_from_slice(&claim.authority_policy_generation.to_le_bytes());
    out.extend_from_slice(claim.authority_profile_id.as_bytes());
    out.extend_from_slice(&claim.authority_root_epoch.to_le_bytes());
    out.extend_from_slice(&claim.authority_basis_digest);
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.target_realization_id.as_bytes());
    out.extend_from_slice(claim.distributed_context_id.as_bytes());
    out.extend_from_slice(&claim.sequence.to_le_bytes());
    encode_optional_id(&mut out, claim.predecessor_currentness_id.map(|id| *id.as_bytes()));
    encode_disposition(&mut out, claim.disposition);
    out.extend_from_slice(&claim.freshness_challenge_digest);
    out.extend_from_slice(&claim.monotonic_counter.to_le_bytes());
    out.extend_from_slice(&claim.observed_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_transition_authority_currentness_claim_digest(
    claim: &TransitionAuthorityCurrentnessClaimV1,
) -> Result<[u8; 32], TransitionAuthorityCurrentnessError> {
    Ok(*blake3::hash(&canonical_transition_authority_currentness_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedTransitionAuthorityCurrentnessV1 {
    claim: TransitionAuthorityCurrentnessClaimV1,
    trusted_root_id: TrustedTransitionAuthorityCurrentnessRootId,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedTransitionAuthorityCurrentnessId,
}

impl AuthenticatedTransitionAuthorityCurrentnessV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: TransitionAuthorityCurrentnessClaimV1,
        profile: &TransitionAuthorityProfileV1,
        trusted_root: &TrustedTransitionAuthorityCurrentnessRootV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, TransitionAuthorityCurrentnessError> {
        claim.validate()?;
        require_root_matches_profile(trusted_root, profile)?;
        if claim.authority_profile_id != profile.id()
            || claim.authority_root_epoch != profile.root_epoch()
        {
            return Err(TransitionAuthorityCurrentnessError::AuthorityRootMismatch);
        }
        if authentication_evidence_digest == [0; 32] {
            return Err(TransitionAuthorityCurrentnessError::ZeroDigest);
        }
        let payload_digest = canonical_transition_authority_currentness_claim_digest(&claim)?;
        let evidence_id = AuthenticatedTransitionAuthorityCurrentnessId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                claim.id().as_bytes(),
                trusted_root.id().as_bytes(),
                &payload_digest,
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self {
            claim,
            trusted_root_id: trusted_root.id(),
            authentication_evidence_digest,
            evidence_id,
        })
    }
}

#[derive(Debug, Clone)]
pub struct QualifiedTransitionAuthorityCurrentnessV1 {
    currentness_id: QualifiedTransitionAuthorityCurrentnessId,
    commit_eligibility_id: CommitEligibleTransitionId,
    authority_claim_id: TransitionAuthorityClaimId,
    authority_profile_id: TransitionAuthorityProfileId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    sequence: u64,
    disposition: TransitionAuthorityDispositionV1,
    freshness_challenge_digest: [u8; 32],
    monotonic_counter: u64,
    observed_at_unix_ms: u64,
}

impl QualifiedTransitionAuthorityCurrentnessV1 {
    pub(crate) fn qualify(
        eligibility: &CommitEligibleTransitionV1,
        profile: &TransitionAuthorityProfileV1,
        trusted_root: &TrustedTransitionAuthorityCurrentnessRootV1,
        authenticated: &AuthenticatedTransitionAuthorityCurrentnessV1,
        expected_freshness_challenge: [u8; 32],
        previous: Option<&QualifiedTransitionAuthorityCurrentnessV1>,
    ) -> Result<Self, TransitionAuthorityCurrentnessError> {
        profile.validate()?;
        authenticated.claim.validate()?;
        require_root_matches_profile(trusted_root, profile)?;
        if authenticated.trusted_root_id != trusted_root.id() {
            return Err(TransitionAuthorityCurrentnessError::AuthorityRootMismatch);
        }
        if expected_freshness_challenge == [0; 32]
            || authenticated.claim.freshness_challenge_digest != expected_freshness_challenge
        {
            return Err(TransitionAuthorityCurrentnessError::ChallengeMismatch);
        }
        require_claim_matches_eligibility(&authenticated.claim, eligibility, profile)?;
        if authenticated.claim.observed_at_unix_ms != eligibility.commit_time_unix_ms() {
            return Err(TransitionAuthorityCurrentnessError::NotAtDecisionBoundary);
        }
        validate_progression(previous, &authenticated.claim)?;
        let currentness_id = QualifiedTransitionAuthorityCurrentnessId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[
                authenticated.claim.id().as_bytes(),
                authenticated.evidence_id.as_bytes(),
                trusted_root.id().as_bytes(),
                eligibility.id().as_bytes(),
                &expected_freshness_challenge,
            ],
        ));
        Ok(Self {
            currentness_id,
            commit_eligibility_id: eligibility.id(),
            authority_claim_id: eligibility.authority_claim_id(),
            authority_profile_id: eligibility.authority_profile_id(),
            subject_id: eligibility.subject_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            sequence: authenticated.claim.sequence,
            disposition: authenticated.claim.disposition,
            freshness_challenge_digest: authenticated.claim.freshness_challenge_digest,
            monotonic_counter: authenticated.claim.monotonic_counter,
            observed_at_unix_ms: authenticated.claim.observed_at_unix_ms,
        })
    }

    pub fn id(&self) -> QualifiedTransitionAuthorityCurrentnessId { self.currentness_id }
    pub fn disposition(&self) -> TransitionAuthorityDispositionV1 { self.disposition }
    pub fn observed_at_unix_ms(&self) -> u64 { self.observed_at_unix_ms }
}

/// Non-Serde proof that the exact transition grant is the currently active authority
/// head at the exact decision boundary.
#[derive(Debug)]
pub struct CurrentAuthorizedTransitionAuthorityV1 {
    current_id: CurrentAuthorizedTransitionAuthorityId,
    qualified_currentness_id: QualifiedTransitionAuthorityCurrentnessId,
    commit_eligibility_id: CommitEligibleTransitionId,
    authority_claim_id: TransitionAuthorityClaimId,
    authority_profile_id: TransitionAuthorityProfileId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    decision_time_unix_ms: u64,
}

impl CurrentAuthorizedTransitionAuthorityV1 {
    pub fn bind(
        eligibility: &CommitEligibleTransitionV1,
        qualified: QualifiedTransitionAuthorityCurrentnessV1,
    ) -> Result<Self, TransitionAuthorityCurrentnessError> {
        if qualified.disposition != TransitionAuthorityDispositionV1::Active {
            return Err(TransitionAuthorityCurrentnessError::AuthorityNotActive);
        }
        if qualified.commit_eligibility_id != eligibility.id()
            || qualified.authority_claim_id != eligibility.authority_claim_id()
            || qualified.authority_profile_id != eligibility.authority_profile_id()
            || qualified.subject_id != eligibility.subject_id()
            || qualified.target_realization_id != eligibility.target_realization_id()
            || qualified.distributed_context_id != eligibility.distributed_context_id()
            || qualified.observed_at_unix_ms != eligibility.commit_time_unix_ms()
        {
            return Err(TransitionAuthorityCurrentnessError::EligibilityMismatch);
        }
        let current_id = CurrentAuthorizedTransitionAuthorityId(domain_hash_parts(
            CURRENT_DOMAIN,
            &[
                eligibility.id().as_bytes(),
                qualified.id().as_bytes(),
                eligibility.authority_claim_id().as_bytes(),
                eligibility.authority_profile_id().as_bytes(),
                &eligibility.commit_time_unix_ms().to_le_bytes(),
            ],
        ));
        Ok(Self {
            current_id,
            qualified_currentness_id: qualified.id(),
            commit_eligibility_id: eligibility.id(),
            authority_claim_id: eligibility.authority_claim_id(),
            authority_profile_id: eligibility.authority_profile_id(),
            subject_id: eligibility.subject_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            decision_time_unix_ms: eligibility.commit_time_unix_ms(),
        })
    }

    pub fn id(&self) -> CurrentAuthorizedTransitionAuthorityId { self.current_id }
    pub fn currentness_id(&self) -> QualifiedTransitionAuthorityCurrentnessId {
        self.qualified_currentness_id
    }
    pub fn commit_eligibility_id(&self) -> CommitEligibleTransitionId { self.commit_eligibility_id }
    pub fn authority_claim_id(&self) -> TransitionAuthorityClaimId { self.authority_claim_id }
    pub fn authority_profile_id(&self) -> TransitionAuthorityProfileId { self.authority_profile_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn decision_time_unix_ms(&self) -> u64 { self.decision_time_unix_ms }
}

/// Effectful physical eligibility additionally requiring a fresh active owner/operator
/// grant head for the exact authorized effect plan.
#[derive(Debug)]
pub struct CurrentAuthorityEffectExecutionEligibilityV1 {
    inner: CurrentAuthorizedEffectExecutionEligibilityV1,
    current_authority: CurrentAuthorizedTransitionAuthorityV1,
}

impl CurrentAuthorityEffectExecutionEligibilityV1 {
    pub fn bind(
        inner: CurrentAuthorizedEffectExecutionEligibilityV1,
        plan: &ExternalEffectPlanV1,
        current_authority: CurrentAuthorizedTransitionAuthorityV1,
    ) -> Result<Self, CurrentAuthorityExecutionError> {
        plan.validate()?;
        let coverage = inner.current_coverage();
        if coverage.coverage().plan_id() != plan.id()
            || current_authority.commit_eligibility_id() != plan.commit_eligibility_id()
            || current_authority.subject_id() != plan.subject_id()
            || current_authority.target_realization_id() != plan.target_realization_id()
            || current_authority.distributed_context_id() != plan.distributed_context_id()
            || current_authority.decision_time_unix_ms() != plan.commit_time_unix_ms()
            || current_authority.decision_time_unix_ms() != coverage.decision_time_unix_ms()
        {
            return Err(CurrentAuthorityExecutionError::EffectTransitionMismatch);
        }
        Ok(Self { inner, current_authority })
    }
}

/// Proven-no-effects sibling requiring the fresh active grant head for the exact
/// no-effects declaration.
#[derive(Debug)]
pub struct CurrentAuthorityNoEffectsExecutionEligibilityV1 {
    inner: CurrentAuthorizedNoEffectsExecutionEligibilityV1,
    current_authority: CurrentAuthorizedTransitionAuthorityV1,
}

impl CurrentAuthorityNoEffectsExecutionEligibilityV1 {
    pub fn bind(
        inner: CurrentAuthorizedNoEffectsExecutionEligibilityV1,
        declaration: &NoExternalEffectsDeclarationV1,
        current_authority: CurrentAuthorizedTransitionAuthorityV1,
    ) -> Result<Self, CurrentAuthorityExecutionError> {
        declaration.validate()?;
        let coverage = inner.current_coverage();
        if coverage.coverage().declaration_id() != declaration.id()
            || current_authority.commit_eligibility_id() != declaration.commit_eligibility_id()
            || current_authority.subject_id() != declaration.subject_id()
            || current_authority.target_realization_id() != declaration.target_realization_id()
            || current_authority.distributed_context_id() != declaration.distributed_context_id()
            || current_authority.decision_time_unix_ms() != declaration.commit_time_unix_ms()
            || current_authority.decision_time_unix_ms() != coverage.decision_time_unix_ms()
        {
            return Err(CurrentAuthorityExecutionError::NoEffectsTransitionMismatch);
        }
        Ok(Self { inner, current_authority })
    }
}

#[derive(Debug)]
pub struct PendingCurrentAuthorityEffectExecutionV1 {
    inner: PendingCurrentAuthorizedEffectExecutionV1,
    current_authority: CurrentAuthorizedTransitionAuthorityV1,
}

impl PendingCurrentAuthorityEffectExecutionV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn release_after_durable_current_authority(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        effect_scope: &QualifiedEffectScopeCommitmentV1,
        backend_commitment: &QualifiedBackendEffectCommitmentV1,
        coverage_commitment: &QualifiedEffectCoverageCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
    ) -> Result<ReadyCurrentAuthorityEffectExecutionV1, CurrentAuthorityExecutionError> {
        if self.current_authority.decision_time_unix_ms() != journal_anchor.anchored_at_unix_ms() {
            return Err(CurrentAuthorityExecutionError::DecisionBoundaryMismatch);
        }
        let ready = self.inner.release_after_durable_current_coverage(
            journal,
            journal_anchor,
            effect_scope,
            backend_commitment,
            coverage_commitment,
            current_verifier_commitment,
        )?;
        if ready.subject_id() != self.current_authority.subject_id()
            || ready.target_realization_id() != self.current_authority.target_realization_id()
        {
            return Err(CurrentAuthorityExecutionError::ReadyTransitionMismatch);
        }
        Ok(ReadyCurrentAuthorityEffectExecutionV1 {
            inner: ready,
            current_authority_id: self.current_authority.id(),
            authority_currentness_id: self.current_authority.currentness_id(),
        })
    }
}

#[derive(Debug)]
pub struct ReadyCurrentAuthorityEffectExecutionV1 {
    inner: ReadyCurrentAuthorizedEffectExecutionV1,
    current_authority_id: CurrentAuthorizedTransitionAuthorityId,
    authority_currentness_id: QualifiedTransitionAuthorityCurrentnessId,
}

impl ReadyCurrentAuthorityEffectExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.inner.attempt_id() }
    pub fn backend_id(&self) -> ExecutionBackendId { self.inner.backend_id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.inner.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.inner.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.inner.target_realization_id() }
    pub fn effect_contract(&self) -> &ExternalEffectContractV1 { self.inner.effect_contract() }
    pub fn current_authority_id(&self) -> CurrentAuthorizedTransitionAuthorityId {
        self.current_authority_id
    }
    pub fn authority_currentness_id(&self) -> QualifiedTransitionAuthorityCurrentnessId {
        self.authority_currentness_id
    }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, CurrentAuthorityExecutionError> {
        Ok(self.inner.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[derive(Debug)]
pub struct PendingCurrentAuthorityNoEffectsExecutionV1 {
    inner: PendingCurrentAuthorizedNoEffectsExecutionV1,
    current_authority: CurrentAuthorizedTransitionAuthorityV1,
}

impl PendingCurrentAuthorityNoEffectsExecutionV1 {
    pub fn release_after_durable_current_authority(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        commitment: &QualifiedNoEffectsExecutionCommitmentV1,
        current_verifier_commitment: &QualifiedCurrentVerifierExecutionCommitmentV1,
    ) -> Result<ReadyCurrentAuthorityNoEffectsExecutionV1, CurrentAuthorityExecutionError> {
        if self.current_authority.decision_time_unix_ms() != journal_anchor.anchored_at_unix_ms() {
            return Err(CurrentAuthorityExecutionError::DecisionBoundaryMismatch);
        }
        let ready = self.inner.release_after_durable_current_no_effects(
            journal,
            journal_anchor,
            commitment,
            current_verifier_commitment,
        )?;
        if ready.subject_id() != self.current_authority.subject_id()
            || ready.target_realization_id() != self.current_authority.target_realization_id()
        {
            return Err(CurrentAuthorityExecutionError::ReadyTransitionMismatch);
        }
        Ok(ReadyCurrentAuthorityNoEffectsExecutionV1 {
            inner: ready,
            current_authority_id: self.current_authority.id(),
            authority_currentness_id: self.current_authority.currentness_id(),
        })
    }
}

#[derive(Debug)]
pub struct ReadyCurrentAuthorityNoEffectsExecutionV1 {
    inner: ReadyCurrentAuthorizedNoEffectsExecutionV1,
    current_authority_id: CurrentAuthorizedTransitionAuthorityId,
    authority_currentness_id: QualifiedTransitionAuthorityCurrentnessId,
}

impl ReadyCurrentAuthorityNoEffectsExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.inner.attempt_id() }
    pub fn backend_id(&self) -> ExecutionBackendId { self.inner.backend_id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.inner.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.inner.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.inner.target_realization_id() }
    pub fn current_authority_id(&self) -> CurrentAuthorizedTransitionAuthorityId {
        self.current_authority_id
    }
    pub fn authority_currentness_id(&self) -> QualifiedTransitionAuthorityCurrentnessId {
        self.authority_currentness_id
    }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, CurrentAuthorityExecutionError> {
        Ok(self.inner.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_current_authority_effect_execution(
    bound: CurrentAuthorityEffectExecutionEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingCurrentAuthorityEffectExecutionV1, CurrentAuthorityExecutionError> {
    let CurrentAuthorityEffectExecutionEligibilityV1 { inner, current_authority } = bound;
    let pending = prepare_current_authorized_effect_execution(
        inner,
        current_epoch,
        previous_epoch,
        epoch_anchor_mode,
        predecessor_journal_anchor,
        session_generation,
        session_nonce,
    )?;
    Ok(PendingCurrentAuthorityEffectExecutionV1 { inner: pending, current_authority })
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_current_authority_no_effects_execution(
    bound: CurrentAuthorityNoEffectsExecutionEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingCurrentAuthorityNoEffectsExecutionV1, CurrentAuthorityExecutionError> {
    let CurrentAuthorityNoEffectsExecutionEligibilityV1 { inner, current_authority } = bound;
    let pending = prepare_current_authorized_no_effects_execution(
        inner,
        current_epoch,
        previous_epoch,
        epoch_anchor_mode,
        predecessor_journal_anchor,
        session_generation,
        session_nonce,
    )?;
    Ok(PendingCurrentAuthorityNoEffectsExecutionV1 { inner: pending, current_authority })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TransitionAuthorityCurrentnessError {
    #[error(transparent)]
    Authority(#[from] TransitionAuthorityError),
    #[error("unsupported transition-authority currentness schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("transition-authority currentness digest must be non-zero")]
    ZeroDigest,
    #[error("transition-authority currentness generation must be non-zero")]
    ZeroGeneration,
    #[error("transition-authority currentness sequence/counter/time must be non-zero")]
    ZeroCurrentnessMaterial,
    #[error("trusted transition-authority currentness root does not match authority profile")]
    AuthorityRootMismatch,
    #[error("transition-authority currentness claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("transition-authority currentness claim differs from exact commit eligibility")]
    EligibilityMismatch,
    #[error("transition-authority currentness freshness challenge mismatch or replay")]
    ChallengeMismatch,
    #[error("initial transition-authority currentness must start at sequence 1 without predecessor")]
    InvalidInitialCurrentness,
    #[error("transition-authority currentness sequence/predecessor is not exact successor")]
    CurrentnessSequenceMismatch,
    #[error("transition-authority currentness monotonic counter did not strictly advance")]
    MonotonicCounterRollback,
    #[error("transition-authority currentness observation time rolled back")]
    ObservationTimeRollback,
    #[error("terminal transition-authority disposition cannot change")]
    TerminalDispositionChanged,
    #[error("superseded transition authority must name a different successor claim")]
    InvalidSupersedingClaim,
    #[error("transition-authority currentness was not observed at the exact decision boundary")]
    NotAtDecisionBoundary,
    #[error("transition authority is revoked or superseded")]
    AuthorityNotActive,
}

#[derive(Debug, Error)]
pub enum CurrentAuthorityExecutionError {
    #[error(transparent)]
    Currentness(#[from] TransitionAuthorityCurrentnessError),
    #[error(transparent)]
    EffectPlan(#[from] ExternalEffectAuthorityError),
    #[error(transparent)]
    NoEffects(#[from] NoExternalEffectsError),
    #[error(transparent)]
    CurrentAuthorized(#[from] CurrentAuthorizedCoverageError),
    #[error("fresh transition-authority head differs from exact effectful transition")]
    EffectTransitionMismatch,
    #[error("fresh transition-authority head differs from exact no-effects transition")]
    NoEffectsTransitionMismatch,
    #[error("fresh transition-authority decision differs from protected journal boundary")]
    DecisionBoundaryMismatch,
    #[error("ready physical transition differs from fresh transition-authority head")]
    ReadyTransitionMismatch,
}

fn require_root_matches_profile(
    root: &TrustedTransitionAuthorityCurrentnessRootV1,
    profile: &TransitionAuthorityProfileV1,
) -> Result<(), TransitionAuthorityCurrentnessError> {
    profile.validate()?;
    if root.authority_profile_id != profile.id()
        || root.authority_root_digest != profile.root_digest()
        || root.authority_root_epoch != profile.root_epoch()
    {
        return Err(TransitionAuthorityCurrentnessError::AuthorityRootMismatch);
    }
    Ok(())
}

fn require_claim_matches_eligibility(
    claim: &TransitionAuthorityCurrentnessClaimV1,
    eligibility: &CommitEligibleTransitionV1,
    profile: &TransitionAuthorityProfileV1,
) -> Result<(), TransitionAuthorityCurrentnessError> {
    if claim.commit_eligibility_id != eligibility.id()
        || claim.authority_id != eligibility.authority_id()
        || claim.authority_claim_id != eligibility.authority_claim_id()
        || claim.authority_policy_id != eligibility.authority_policy_id()
        || claim.authority_policy_generation != eligibility.authority_policy_generation()
        || claim.authority_profile_id != eligibility.authority_profile_id()
        || claim.authority_profile_id != profile.id()
        || claim.authority_root_epoch != eligibility.authority_root_epoch()
        || claim.authority_root_epoch != profile.root_epoch()
        || claim.authority_basis_digest != eligibility.authority_basis_digest()
        || claim.subject_id != eligibility.subject_id()
        || claim.target_realization_id != eligibility.target_realization_id()
        || claim.distributed_context_id != eligibility.distributed_context_id()
    {
        return Err(TransitionAuthorityCurrentnessError::EligibilityMismatch);
    }
    Ok(())
}

fn validate_progression(
    previous: Option<&QualifiedTransitionAuthorityCurrentnessV1>,
    claim: &TransitionAuthorityCurrentnessClaimV1,
) -> Result<(), TransitionAuthorityCurrentnessError> {
    match previous {
        None => {
            if claim.sequence != 1 || claim.predecessor_currentness_id.is_some() {
                return Err(TransitionAuthorityCurrentnessError::InvalidInitialCurrentness);
            }
        }
        Some(previous) => {
            let expected_sequence = previous.sequence.checked_add(1)
                .ok_or(TransitionAuthorityCurrentnessError::CurrentnessSequenceMismatch)?;
            if claim.sequence != expected_sequence
                || claim.predecessor_currentness_id != Some(previous.id())
            {
                return Err(TransitionAuthorityCurrentnessError::CurrentnessSequenceMismatch);
            }
            if claim.commit_eligibility_id != previous.commit_eligibility_id
                || claim.authority_claim_id != previous.authority_claim_id
                || claim.authority_profile_id != previous.authority_profile_id
                || claim.subject_id != previous.subject_id
                || claim.target_realization_id != previous.target_realization_id
                || claim.distributed_context_id != previous.distributed_context_id
            {
                return Err(TransitionAuthorityCurrentnessError::EligibilityMismatch);
            }
            if claim.freshness_challenge_digest == previous.freshness_challenge_digest {
                return Err(TransitionAuthorityCurrentnessError::ChallengeMismatch);
            }
            if claim.monotonic_counter <= previous.monotonic_counter {
                return Err(TransitionAuthorityCurrentnessError::MonotonicCounterRollback);
            }
            if claim.observed_at_unix_ms < previous.observed_at_unix_ms {
                return Err(TransitionAuthorityCurrentnessError::ObservationTimeRollback);
            }
            if previous.disposition.terminal() && claim.disposition != previous.disposition {
                return Err(TransitionAuthorityCurrentnessError::TerminalDispositionChanged);
            }
        }
    }
    Ok(())
}

fn validate_material(
    sequence: u64,
    freshness_challenge_digest: [u8; 32],
    monotonic_counter: u64,
    observed_at_unix_ms: u64,
    raw_evidence_digest: [u8; 32],
) -> Result<(), TransitionAuthorityCurrentnessError> {
    if sequence == 0 || monotonic_counter == 0 || observed_at_unix_ms == 0 {
        return Err(TransitionAuthorityCurrentnessError::ZeroCurrentnessMaterial);
    }
    if freshness_challenge_digest == [0; 32] || raw_evidence_digest == [0; 32] {
        return Err(TransitionAuthorityCurrentnessError::ZeroDigest);
    }
    Ok(())
}

fn validate_disposition(
    disposition: TransitionAuthorityDispositionV1,
    authority_claim_id: TransitionAuthorityClaimId,
) -> Result<(), TransitionAuthorityCurrentnessError> {
    if let TransitionAuthorityDispositionV1::Superseded { successor_claim_id } = disposition {
        if successor_claim_id == authority_claim_id {
            return Err(TransitionAuthorityCurrentnessError::InvalidSupersedingClaim);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    commit_eligibility_id: CommitEligibleTransitionId,
    authority_id: AuthenticatedTransitionAuthorityId,
    authority_claim_id: TransitionAuthorityClaimId,
    authority_policy_id: TransitionAuthorityPolicyId,
    authority_policy_generation: u64,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    authority_basis_digest: [u8; 32],
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    sequence: u64,
    predecessor_currentness_id: Option<QualifiedTransitionAuthorityCurrentnessId>,
    disposition: TransitionAuthorityDispositionV1,
    freshness_challenge_digest: [u8; 32],
    monotonic_counter: u64,
    observed_at_unix_ms: u64,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(CLAIM_DOMAIN);
    h.update(commit_eligibility_id.as_bytes());
    h.update(authority_id.as_bytes());
    h.update(authority_claim_id.as_bytes());
    h.update(authority_policy_id.as_bytes());
    h.update(&authority_policy_generation.to_le_bytes());
    h.update(authority_profile_id.as_bytes());
    h.update(&authority_root_epoch.to_le_bytes());
    h.update(&authority_basis_digest);
    h.update(subject_id.as_bytes());
    h.update(target_realization_id.as_bytes());
    h.update(distributed_context_id.as_bytes());
    h.update(&sequence.to_le_bytes());
    hash_optional_id(&mut h, predecessor_currentness_id.map(|id| *id.as_bytes()));
    hash_disposition(&mut h, disposition);
    h.update(&freshness_challenge_digest);
    h.update(&monotonic_counter.to_le_bytes());
    h.update(&observed_at_unix_ms.to_le_bytes());
    h.update(&raw_evidence_digest);
    *h.finalize().as_bytes()
}

fn hash_disposition(h: &mut blake3::Hasher, disposition: TransitionAuthorityDispositionV1) {
    h.update(&[disposition.tag()]);
    if let TransitionAuthorityDispositionV1::Superseded { successor_claim_id } = disposition {
        h.update(successor_claim_id.as_bytes());
    }
}

fn encode_disposition(out: &mut Vec<u8>, disposition: TransitionAuthorityDispositionV1) {
    out.push(disposition.tag());
    if let TransitionAuthorityDispositionV1::Superseded { successor_claim_id } = disposition {
        out.extend_from_slice(successor_claim_id.as_bytes());
    }
}

fn hash_optional_id(h: &mut blake3::Hasher, id: Option<[u8; 32]>) {
    match id {
        Some(id) => { h.update(&[1]); h.update(&id); }
        None => { h.update(&[0]); }
    }
}

fn encode_optional_id(out: &mut Vec<u8>, id: Option<[u8; 32]>) {
    match id {
        Some(id) => { out.push(1); out.extend_from_slice(&id); }
        None => out.push(0),
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
    fn currentness_domains_are_distinct() {
        assert_ne!(TRUSTED_ROOT_DOMAIN, CLAIM_DOMAIN);
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
        assert_ne!(WIRE_DOMAIN, AUTH_DOMAIN);
        assert_ne!(AUTH_DOMAIN, QUALIFIED_DOMAIN);
        assert_ne!(QUALIFIED_DOMAIN, CURRENT_DOMAIN);
    }
}
