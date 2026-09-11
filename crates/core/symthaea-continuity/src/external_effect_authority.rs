// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-execution authorization for the exact declared external-effect surface.
//!
//! A transition authority over subject/target/context must not silently acquire a
//! broader externally visible effect surface later. This module gives the exact same
//! authority root an additive, domain-separated authorization over one exact effect
//! plan before trusted eligibility can enter the known-good execution boundary.
//!
//! `TransitionAuthority != EffectPlan != AuthenticatedEffectAuthorization != ExecutionCapability`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::active_lkg::ActiveKnownGoodSelectionV1;
use crate::commit_eligibility::{CommitEligibleTransitionId, CommitEligibleTransitionV1};
use crate::distributed_state::DistributedStateContextId;
use crate::external_effects::{
    ExternalEffectError, ExternalEffectObligationV1,
};
use crate::known_good::QualifiedKnownGoodCheckpointV1;
use crate::scope::ContinuitySubjectId;
use crate::transition_authority::{
    TransitionAuthorityClaimId, TransitionAuthorityError, TransitionAuthorityProfileId,
    TransitionAuthorityProfileV1,
};
use crate::transition_lineage::{
    KnownGoodBoundTrustedCommitEligibilityV1, KnownGoodTransitionLineageError,
    KnownGoodTransitionLineageId,
};
use crate::trusted_commit_epoch::{TrustedCommitEligibilityId, TrustedCommitEligibilityV1};
use crate::witness::TargetRealizationId;

pub const EXTERNAL_EFFECT_PLAN_SCHEMA_V1: &str =
    "symthaea-continuity-external-effect-plan-v1";
pub const EXTERNAL_EFFECT_AUTHORITY_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-external-effect-authority-claim-v1";
pub const EXTERNAL_EFFECT_AUTHORITY_PURPOSE: &str =
    "symthaea.continuity.external-effect-authority.v1";

const PLAN_DOMAIN: &[u8] = b"symthaea.continuity.external-effect-plan.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.external-effect-authority-claim.v1\0";
const CLAIM_WIRE_DOMAIN: &[u8] = b"symthaea.continuity.external-effect-authority-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-external-effect-authority.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-external-effect-authority.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}

digest_id!(ExternalEffectPlanId);
digest_id!(ExternalEffectAuthorityClaimId);
digest_id!(AuthenticatedExternalEffectAuthorityId);
digest_id!(QualifiedExternalEffectAuthorizationId);

/// Stable, serializable side-effect plan established at commit-eligibility time,
/// before any one-use execution capability exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalEffectPlanV1 {
    schema_version: String,
    commit_eligibility_id: CommitEligibleTransitionId,
    original_authority_claim_id: TransitionAuthorityClaimId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    commit_time_unix_ms: u64,
    coverage_manifest_digest: [u8; 32],
    obligations: Vec<ExternalEffectObligationV1>,
    plan_id: ExternalEffectPlanId,
}

impl ExternalEffectPlanV1 {
    pub fn new(
        eligibility: &CommitEligibleTransitionV1,
        coverage_manifest_digest: [u8; 32],
        mut obligations: Vec<ExternalEffectObligationV1>,
    ) -> Result<Self, ExternalEffectAuthorityError> {
        if coverage_manifest_digest == [0; 32] {
            return Err(ExternalEffectAuthorityError::ZeroCoverageManifestDigest);
        }
        if obligations.is_empty() {
            return Err(ExternalEffectAuthorityError::EmptyPlanRequiresNoEffectsProof);
        }
        for obligation in &obligations {
            obligation.validate()?;
        }
        obligations.sort_by_key(ExternalEffectObligationV1::id);
        if obligations.windows(2).any(|pair| pair[0].id() == pair[1].id()) {
            return Err(ExternalEffectAuthorityError::DuplicateObligation);
        }

        let plan_id = ExternalEffectPlanId(hash_plan(
            eligibility.id(),
            eligibility.authority_claim_id(),
            eligibility.authority_profile_id(),
            eligibility.authority_root_epoch(),
            eligibility.subject_id(),
            eligibility.target_realization_id(),
            eligibility.distributed_context_id(),
            eligibility.commit_time_unix_ms(),
            coverage_manifest_digest,
            &obligations,
        ));
        Ok(Self {
            schema_version: EXTERNAL_EFFECT_PLAN_SCHEMA_V1.to_owned(),
            commit_eligibility_id: eligibility.id(),
            original_authority_claim_id: eligibility.authority_claim_id(),
            authority_profile_id: eligibility.authority_profile_id(),
            authority_root_epoch: eligibility.authority_root_epoch(),
            subject_id: eligibility.subject_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            commit_time_unix_ms: eligibility.commit_time_unix_ms(),
            coverage_manifest_digest,
            obligations,
            plan_id,
        })
    }

    pub fn validate(&self) -> Result<(), ExternalEffectAuthorityError> {
        if self.schema_version != EXTERNAL_EFFECT_PLAN_SCHEMA_V1 {
            return Err(ExternalEffectAuthorityError::UnsupportedPlanSchema(
                self.schema_version.clone(),
            ));
        }
        if self.authority_root_epoch == 0 {
            return Err(ExternalEffectAuthorityError::ZeroAuthorityRootEpoch);
        }
        if self.commit_time_unix_ms == 0 {
            return Err(ExternalEffectAuthorityError::ZeroCommitTime);
        }
        if self.coverage_manifest_digest == [0; 32] {
            return Err(ExternalEffectAuthorityError::ZeroCoverageManifestDigest);
        }
        if self.obligations.is_empty() {
            return Err(ExternalEffectAuthorityError::EmptyPlanRequiresNoEffectsProof);
        }
        for obligation in &self.obligations {
            obligation.validate()?;
        }
        if self.obligations.windows(2).any(|pair| pair[0].id() >= pair[1].id()) {
            return Err(ExternalEffectAuthorityError::NonCanonicalObligations);
        }
        let expected = ExternalEffectPlanId(hash_plan(
            self.commit_eligibility_id,
            self.original_authority_claim_id,
            self.authority_profile_id,
            self.authority_root_epoch,
            self.subject_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.commit_time_unix_ms,
            self.coverage_manifest_digest,
            &self.obligations,
        ));
        if expected != self.plan_id {
            return Err(ExternalEffectAuthorityError::PlanIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ExternalEffectPlanId { self.plan_id }
    pub fn commit_eligibility_id(&self) -> CommitEligibleTransitionId { self.commit_eligibility_id }
    pub fn original_authority_claim_id(&self) -> TransitionAuthorityClaimId { self.original_authority_claim_id }
    pub fn authority_profile_id(&self) -> TransitionAuthorityProfileId { self.authority_profile_id }
    pub fn authority_root_epoch(&self) -> u64 { self.authority_root_epoch }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn commit_time_unix_ms(&self) -> u64 { self.commit_time_unix_ms }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn obligations(&self) -> &[ExternalEffectObligationV1] { &self.obligations }
}

/// Additive authorization claim over one exact predeclared effect plan. It must be
/// authenticated by the same exact authority profile/root already admitted for the
/// underlying transition eligibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalEffectAuthorityClaimV1 {
    schema_version: String,
    plan_id: ExternalEffectPlanId,
    commit_eligibility_id: CommitEligibleTransitionId,
    original_authority_claim_id: TransitionAuthorityClaimId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    coverage_manifest_digest: [u8; 32],
    authorized_at_unix_ms: u64,
    claim_id: ExternalEffectAuthorityClaimId,
}

impl ExternalEffectAuthorityClaimV1 {
    pub fn new(
        eligibility: &CommitEligibleTransitionV1,
        plan: &ExternalEffectPlanV1,
        profile: &TransitionAuthorityProfileV1,
    ) -> Result<Self, ExternalEffectAuthorityError> {
        plan.validate()?;
        profile.validate()?;
        require_plan_matches_eligibility(plan, eligibility)?;
        if profile.id() != eligibility.authority_profile_id()
            || profile.root_epoch() != eligibility.authority_root_epoch()
            || profile.id() != plan.authority_profile_id()
            || profile.root_epoch() != plan.authority_root_epoch()
        {
            return Err(ExternalEffectAuthorityError::AuthorityRootMismatch);
        }
        let authorized_at_unix_ms = eligibility.commit_time_unix_ms();
        let claim_id = ExternalEffectAuthorityClaimId(hash_claim(
            plan.id(),
            eligibility.id(),
            eligibility.authority_claim_id(),
            profile.id(),
            profile.root_epoch(),
            eligibility.subject_id(),
            eligibility.target_realization_id(),
            eligibility.distributed_context_id(),
            plan.coverage_manifest_digest(),
            authorized_at_unix_ms,
        ));
        Ok(Self {
            schema_version: EXTERNAL_EFFECT_AUTHORITY_CLAIM_SCHEMA_V1.to_owned(),
            plan_id: plan.id(),
            commit_eligibility_id: eligibility.id(),
            original_authority_claim_id: eligibility.authority_claim_id(),
            authority_profile_id: profile.id(),
            authority_root_epoch: profile.root_epoch(),
            subject_id: eligibility.subject_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            coverage_manifest_digest: plan.coverage_manifest_digest(),
            authorized_at_unix_ms,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), ExternalEffectAuthorityError> {
        if self.schema_version != EXTERNAL_EFFECT_AUTHORITY_CLAIM_SCHEMA_V1 {
            return Err(ExternalEffectAuthorityError::UnsupportedAuthorityClaimSchema(
                self.schema_version.clone(),
            ));
        }
        if self.authority_root_epoch == 0 {
            return Err(ExternalEffectAuthorityError::ZeroAuthorityRootEpoch);
        }
        if self.authorized_at_unix_ms == 0 {
            return Err(ExternalEffectAuthorityError::ZeroAuthorizationTime);
        }
        if self.coverage_manifest_digest == [0; 32] {
            return Err(ExternalEffectAuthorityError::ZeroCoverageManifestDigest);
        }
        let expected = ExternalEffectAuthorityClaimId(hash_claim(
            self.plan_id,
            self.commit_eligibility_id,
            self.original_authority_claim_id,
            self.authority_profile_id,
            self.authority_root_epoch,
            self.subject_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.coverage_manifest_digest,
            self.authorized_at_unix_ms,
        ));
        if expected != self.claim_id {
            return Err(ExternalEffectAuthorityError::AuthorityClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ExternalEffectAuthorityClaimId { self.claim_id }
}

pub fn canonical_external_effect_authority_claim_bytes(
    claim: &ExternalEffectAuthorityClaimV1,
) -> Result<Vec<u8>, ExternalEffectAuthorityError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(416);
    out.extend_from_slice(CLAIM_WIRE_DOMAIN);
    out.extend_from_slice(claim.plan_id.as_bytes());
    out.extend_from_slice(claim.commit_eligibility_id.as_bytes());
    out.extend_from_slice(claim.original_authority_claim_id.as_bytes());
    out.extend_from_slice(claim.authority_profile_id.as_bytes());
    out.extend_from_slice(&claim.authority_root_epoch.to_le_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.target_realization_id.as_bytes());
    out.extend_from_slice(claim.distributed_context_id.as_bytes());
    out.extend_from_slice(&claim.coverage_manifest_digest);
    out.extend_from_slice(&claim.authorized_at_unix_ms.to_le_bytes());
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_external_effect_authority_claim_digest(
    claim: &ExternalEffectAuthorityClaimV1,
) -> Result<[u8; 32], ExternalEffectAuthorityError> {
    Ok(*blake3::hash(&canonical_external_effect_authority_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedExternalEffectAuthorityV1 {
    claim: ExternalEffectAuthorityClaimV1,
    profile: TransitionAuthorityProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedExternalEffectAuthorityId,
}

impl AuthenticatedExternalEffectAuthorityV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: ExternalEffectAuthorityClaimV1,
        profile: TransitionAuthorityProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, ExternalEffectAuthorityError> {
        claim.validate()?;
        profile.validate()?;
        if claim.authority_profile_id != profile.id()
            || claim.authority_root_epoch != profile.root_epoch()
        {
            return Err(ExternalEffectAuthorityError::AuthorityRootMismatch);
        }
        if authentication_evidence_digest == [0; 32] {
            return Err(ExternalEffectAuthorityError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedExternalEffectAuthorityId(domain_hash_parts(
            AUTH_DOMAIN,
            &[
                claim.id().as_bytes(),
                profile.id().as_bytes(),
                &profile.root_epoch().to_le_bytes(),
                &authentication_evidence_digest,
            ],
        ));
        Ok(Self { claim, profile, authentication_evidence_digest, evidence_id })
    }
}

/// Non-Serde proof that the exact same transition authority root authenticated the
/// exact predeclared external-effect plan for this commit transaction.
#[derive(Debug, Clone)]
pub struct QualifiedExternalEffectAuthorizationV1 {
    authorization_id: QualifiedExternalEffectAuthorizationId,
    plan_id: ExternalEffectPlanId,
    commit_eligibility_id: CommitEligibleTransitionId,
    original_authority_claim_id: TransitionAuthorityClaimId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    coverage_manifest_digest: [u8; 32],
    authorized_at_unix_ms: u64,
    authenticated_evidence_id: AuthenticatedExternalEffectAuthorityId,
}

impl QualifiedExternalEffectAuthorizationV1 {
    pub(crate) fn qualify(
        eligibility: &CommitEligibleTransitionV1,
        plan: &ExternalEffectPlanV1,
        evidence: &AuthenticatedExternalEffectAuthorityV1,
    ) -> Result<Self, ExternalEffectAuthorityError> {
        plan.validate()?;
        evidence.claim.validate()?;
        evidence.profile.validate()?;
        require_plan_matches_eligibility(plan, eligibility)?;
        let claim = &evidence.claim;
        if claim.plan_id != plan.id()
            || claim.commit_eligibility_id != eligibility.id()
            || claim.original_authority_claim_id != eligibility.authority_claim_id()
            || claim.authority_profile_id != eligibility.authority_profile_id()
            || claim.authority_root_epoch != eligibility.authority_root_epoch()
            || claim.subject_id != eligibility.subject_id()
            || claim.target_realization_id != eligibility.target_realization_id()
            || claim.distributed_context_id != eligibility.distributed_context_id()
            || claim.coverage_manifest_digest != plan.coverage_manifest_digest()
            || claim.authorized_at_unix_ms != eligibility.commit_time_unix_ms()
        {
            return Err(ExternalEffectAuthorityError::AuthorizationContextMismatch);
        }
        if evidence.profile.id() != eligibility.authority_profile_id()
            || evidence.profile.root_epoch() != eligibility.authority_root_epoch()
        {
            return Err(ExternalEffectAuthorityError::AuthorityRootMismatch);
        }
        let authorization_id = QualifiedExternalEffectAuthorizationId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[
                plan.id().as_bytes(),
                eligibility.id().as_bytes(),
                claim.id().as_bytes(),
                evidence.profile.id().as_bytes(),
                &evidence.profile.root_epoch().to_le_bytes(),
                evidence.evidence_id.as_bytes(),
                &eligibility.commit_time_unix_ms().to_le_bytes(),
            ],
        ));
        Ok(Self {
            authorization_id,
            plan_id: plan.id(),
            commit_eligibility_id: eligibility.id(),
            original_authority_claim_id: eligibility.authority_claim_id(),
            authority_profile_id: eligibility.authority_profile_id(),
            authority_root_epoch: eligibility.authority_root_epoch(),
            subject_id: eligibility.subject_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            coverage_manifest_digest: plan.coverage_manifest_digest(),
            authorized_at_unix_ms: eligibility.commit_time_unix_ms(),
            authenticated_evidence_id: evidence.evidence_id,
        })
    }

    pub fn id(&self) -> QualifiedExternalEffectAuthorizationId { self.authorization_id }
    pub fn plan_id(&self) -> ExternalEffectPlanId { self.plan_id }
    pub fn commit_eligibility_id(&self) -> CommitEligibleTransitionId { self.commit_eligibility_id }
    pub fn authority_profile_id(&self) -> TransitionAuthorityProfileId { self.authority_profile_id }
    pub fn authority_root_epoch(&self) -> u64 { self.authority_root_epoch }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn authorized_at_unix_ms(&self) -> u64 { self.authorized_at_unix_ms }
}

/// Non-Clone trusted commit eligibility plus its independently authenticated external
/// effect authorization. This prevents later execution code from dropping the plan.
#[derive(Debug)]
pub struct EffectAuthorizedTrustedCommitEligibilityV1 {
    trusted: TrustedCommitEligibilityV1,
    plan: ExternalEffectPlanV1,
    authorization: QualifiedExternalEffectAuthorizationV1,
}

impl EffectAuthorizedTrustedCommitEligibilityV1 {
    pub fn bind(
        trusted: TrustedCommitEligibilityV1,
        plan: ExternalEffectPlanV1,
        authorization: QualifiedExternalEffectAuthorizationV1,
    ) -> Result<Self, ExternalEffectAuthorityError> {
        plan.validate()?;
        if trusted.eligibility_id() != plan.commit_eligibility_id()
            || trusted.eligibility_id() != authorization.commit_eligibility_id()
            || trusted.subject_id() != plan.subject_id()
            || trusted.subject_id() != authorization.subject_id()
            || trusted.target_realization_id() != plan.target_realization_id()
            || trusted.target_realization_id() != authorization.target_realization_id()
            || trusted.distributed_context_id() != plan.distributed_context_id()
            || trusted.distributed_context_id() != authorization.distributed_context_id()
            || trusted.commit_time_unix_ms() != plan.commit_time_unix_ms()
            || trusted.commit_time_unix_ms() != authorization.authorized_at_unix_ms()
            || plan.id() != authorization.plan_id()
            || plan.coverage_manifest_digest() != authorization.coverage_manifest_digest()
        {
            return Err(ExternalEffectAuthorityError::TrustedEligibilityMismatch);
        }
        Ok(Self { trusted, plan, authorization })
    }

    pub fn plan(&self) -> &ExternalEffectPlanV1 { &self.plan }
    pub fn authorization(&self) -> &QualifiedExternalEffectAuthorizationV1 { &self.authorization }

    fn into_parts(
        self,
    ) -> (
        TrustedCommitEligibilityV1,
        ExternalEffectPlanV1,
        QualifiedExternalEffectAuthorizationV1,
    ) {
        (self.trusted, self.plan, self.authorization)
    }
}

/// Exact active-A binding that retains the authorized effect plan while preserving
/// the existing V1 A->B lineage identity.
#[derive(Debug)]
pub struct EffectScopedKnownGoodBoundEligibilityV1 {
    bound: KnownGoodBoundTrustedCommitEligibilityV1,
    plan: ExternalEffectPlanV1,
    authorization: QualifiedExternalEffectAuthorizationV1,
}

impl EffectScopedKnownGoodBoundEligibilityV1 {
    pub fn bind(
        active: &ActiveKnownGoodSelectionV1,
        checkpoint: &QualifiedKnownGoodCheckpointV1,
        authorized: EffectAuthorizedTrustedCommitEligibilityV1,
    ) -> Result<Self, ExternalEffectAuthorityError> {
        let (trusted, plan, authorization) = authorized.into_parts();
        let bound = KnownGoodBoundTrustedCommitEligibilityV1::bind(active, checkpoint, trusted)?;
        let lineage = bound.lineage();
        if lineage.subject_id() != plan.subject_id()
            || lineage.target_realization_id() != plan.target_realization_id()
            || lineage.distributed_context_id() != plan.distributed_context_id()
            || lineage.commit_time_unix_ms() != plan.commit_time_unix_ms()
            || lineage.trusted_eligibility_id() == TrustedCommitEligibilityId::from_bytes_forbidden_placeholder()
        {
            return Err(ExternalEffectAuthorityError::KnownGoodLineageMismatch);
        }
        Ok(Self { bound, plan, authorization })
    }

    pub fn lineage(&self) -> &crate::transition_lineage::KnownGoodTransitionLineageV1 {
        self.bound.lineage()
    }
    pub fn plan(&self) -> &ExternalEffectPlanV1 { &self.plan }
    pub fn authorization(&self) -> &QualifiedExternalEffectAuthorizationV1 { &self.authorization }

    pub(crate) fn into_parts(
        self,
    ) -> (
        KnownGoodBoundTrustedCommitEligibilityV1,
        ExternalEffectPlanV1,
        QualifiedExternalEffectAuthorizationV1,
    ) {
        (self.bound, self.plan, self.authorization)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExternalEffectAuthorityError {
    #[error(transparent)]
    ExternalEffect(#[from] ExternalEffectError),
    #[error(transparent)]
    Authority(#[from] TransitionAuthorityError),
    #[error(transparent)]
    KnownGood(#[from] KnownGoodTransitionLineageError),
    #[error("unsupported external-effect plan schema: {0}")]
    UnsupportedPlanSchema(String),
    #[error("unsupported external-effect authority claim schema: {0}")]
    UnsupportedAuthorityClaimSchema(String),
    #[error("external-effect coverage manifest digest must be non-zero")]
    ZeroCoverageManifestDigest,
    #[error("external-effect plan requires at least one declared effect; use a separate no-effects proof")]
    EmptyPlanRequiresNoEffectsProof,
    #[error("duplicate external-effect obligation")]
    DuplicateObligation,
    #[error("external-effect obligations are not in strict canonical id order")]
    NonCanonicalObligations,
    #[error("external-effect authority root epoch must be non-zero")]
    ZeroAuthorityRootEpoch,
    #[error("external-effect plan commit time must be non-zero")]
    ZeroCommitTime,
    #[error("external-effect plan identity mismatch")]
    PlanIdentityMismatch,
    #[error("external-effect authority profile/root differs from the original transition authority")]
    AuthorityRootMismatch,
    #[error("external-effect authorization time must be non-zero")]
    ZeroAuthorizationTime,
    #[error("external-effect authority claim identity mismatch")]
    AuthorityClaimIdentityMismatch,
    #[error("external-effect authorization authentication digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("external-effect plan does not match exact commit eligibility")]
    PlanEligibilityMismatch,
    #[error("authenticated external-effect authorization does not match exact plan/eligibility")]
    AuthorizationContextMismatch,
    #[error("trusted commit eligibility does not retain exact authorized external-effect plan")]
    TrustedEligibilityMismatch,
    #[error("active known-good A->B lineage does not retain exact authorized external-effect plan")]
    KnownGoodLineageMismatch,
}

fn require_plan_matches_eligibility(
    plan: &ExternalEffectPlanV1,
    eligibility: &CommitEligibleTransitionV1,
) -> Result<(), ExternalEffectAuthorityError> {
    if plan.commit_eligibility_id() != eligibility.id()
        || plan.original_authority_claim_id() != eligibility.authority_claim_id()
        || plan.authority_profile_id() != eligibility.authority_profile_id()
        || plan.authority_root_epoch() != eligibility.authority_root_epoch()
        || plan.subject_id() != eligibility.subject_id()
        || plan.target_realization_id() != eligibility.target_realization_id()
        || plan.distributed_context_id() != eligibility.distributed_context_id()
        || plan.commit_time_unix_ms() != eligibility.commit_time_unix_ms()
    {
        return Err(ExternalEffectAuthorityError::PlanEligibilityMismatch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_plan(
    eligibility_id: CommitEligibleTransitionId,
    authority_claim_id: TransitionAuthorityClaimId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    subject_id: ContinuitySubjectId,
    target_id: TargetRealizationId,
    context_id: DistributedStateContextId,
    commit_time_unix_ms: u64,
    coverage_manifest_digest: [u8; 32],
    obligations: &[ExternalEffectObligationV1],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PLAN_DOMAIN);
    hasher.update(eligibility_id.as_bytes());
    hasher.update(authority_claim_id.as_bytes());
    hasher.update(authority_profile_id.as_bytes());
    hasher.update(&authority_root_epoch.to_le_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(target_id.as_bytes());
    hasher.update(context_id.as_bytes());
    hasher.update(&commit_time_unix_ms.to_le_bytes());
    hasher.update(&coverage_manifest_digest);
    hasher.update(&(obligations.len() as u64).to_le_bytes());
    for obligation in obligations {
        hasher.update(obligation.id().as_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    plan_id: ExternalEffectPlanId,
    eligibility_id: CommitEligibleTransitionId,
    original_authority_claim_id: TransitionAuthorityClaimId,
    profile_id: TransitionAuthorityProfileId,
    root_epoch: u64,
    subject_id: ContinuitySubjectId,
    target_id: TargetRealizationId,
    context_id: DistributedStateContextId,
    coverage_manifest_digest: [u8; 32],
    authorized_at_unix_ms: u64,
) -> [u8; 32] {
    domain_hash_parts(
        CLAIM_DOMAIN,
        &[
            plan_id.as_bytes(),
            eligibility_id.as_bytes(),
            original_authority_claim_id.as_bytes(),
            profile_id.as_bytes(),
            &root_epoch.to_le_bytes(),
            subject_id.as_bytes(),
            target_id.as_bytes(),
            context_id.as_bytes(),
            &coverage_manifest_digest,
            &authorized_at_unix_ms.to_le_bytes(),
        ],
    )
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for part in parts {
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_wire_domain_is_distinct_from_plan_domain() {
        assert_ne!(PLAN_DOMAIN, CLAIM_WIRE_DOMAIN);
    }

    #[test]
    fn authority_claim_domain_is_distinct_from_transition_authority_domain() {
        assert_ne!(CLAIM_DOMAIN, b"symthaea.continuity.transition-authority-claim.v1\0");
    }
}
