// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit, backend-relative proof that one transition has no external effects.
//!
//! An empty obligation vector is never evidence. The owner first authorizes an exact
//! no-effects constraint for one backend implementation; an independent verifier then
//! proves the reachable external-effect set is the canonical empty set under an exact
//! analysis/boundary/taxonomy profile.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::active_lkg::ActiveKnownGoodSelectionV1;
use crate::commit_eligibility::{CommitEligibleTransitionId, CommitEligibleTransitionV1};
use crate::distributed_state::DistributedStateContextId;
use crate::effect_coverage::{EffectCoverageError, EffectCoverageProfileId, EffectCoverageProfileV1};
use crate::execution_capability::{ExecutionBackendId, ExecutionBackendProfileV1, ExecutionCapabilityError};
use crate::known_good::QualifiedKnownGoodCheckpointV1;
use crate::scope::ContinuitySubjectId;
use crate::transition_authority::{
    TransitionAuthorityClaimId, TransitionAuthorityError, TransitionAuthorityProfileId,
    TransitionAuthorityProfileV1,
};
use crate::transition_lineage::{
    KnownGoodBoundTrustedCommitEligibilityV1, KnownGoodTransitionLineageError,
};
use crate::trusted_commit_epoch::{TrustedCommitEligibilityId, TrustedCommitEligibilityV1};
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};
use crate::witness::TargetRealizationId;

pub const NO_EXTERNAL_EFFECTS_DECLARATION_SCHEMA_V1: &str =
    "symthaea-continuity-no-external-effects-declaration-v1";
pub const NO_EXTERNAL_EFFECTS_AUTHORITY_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-no-external-effects-authority-claim-v1";
pub const NO_EXTERNAL_EFFECTS_COVERAGE_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-no-external-effects-coverage-claim-v1";
pub const NO_EXTERNAL_EFFECTS_AUTHORITY_PURPOSE: &str =
    "symthaea.continuity.no-external-effects-authority.v1";
pub const NO_EXTERNAL_EFFECTS_COVERAGE_AUTH_PURPOSE: &str =
    "symthaea.continuity.no-external-effects-coverage.v1";

const DECLARATION_DOMAIN: &[u8] = b"symthaea.continuity.no-external-effects-declaration.v1\0";
const AUTH_CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.no-external-effects-authority-claim.v1\0";
const AUTH_WIRE_DOMAIN: &[u8] = b"symthaea.continuity.no-external-effects-authority-wire.v1\0";
const AUTH_EVIDENCE_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-no-external-effects-authority.v1\0";
const QUALIFIED_AUTH_DOMAIN: &[u8] = b"symthaea.continuity.qualified-no-external-effects-authority.v1\0";
const COVERAGE_CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.no-external-effects-coverage-claim.v1\0";
const COVERAGE_WIRE_DOMAIN: &[u8] = b"symthaea.continuity.no-external-effects-coverage-wire.v1\0";
const COVERAGE_AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-no-external-effects-coverage.v1\0";
const QUALIFIED_COVERAGE_DOMAIN: &[u8] = b"symthaea.continuity.qualified-no-external-effects-coverage.v1\0";
const EMPTY_EFFECT_SET_DOMAIN: &[u8] = b"symthaea.continuity.empty-external-effect-set.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }
    };
}

digest_id!(NoExternalEffectsDeclarationId);
digest_id!(NoExternalEffectsAuthorityClaimId);
digest_id!(AuthenticatedNoExternalEffectsAuthorityId);
digest_id!(QualifiedNoExternalEffectsAuthorizationId);
digest_id!(NoExternalEffectsCoverageClaimId);
digest_id!(AuthenticatedNoExternalEffectsCoverageId);
digest_id!(QualifiedNoExternalEffectsCoverageId);

pub fn canonical_empty_external_effect_set_digest() -> [u8; 32] {
    *blake3::hash(EMPTY_EFFECT_SET_DOMAIN).as_bytes()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoExternalEffectsDeclarationV1 {
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
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    declaration_id: NoExternalEffectsDeclarationId,
}

impl NoExternalEffectsDeclarationV1 {
    pub fn new(
        eligibility: &CommitEligibleTransitionV1,
        backend: &ExecutionBackendProfileV1,
        coverage_manifest_digest: [u8; 32],
    ) -> Result<Self, NoExternalEffectsError> {
        backend.validate()?;
        require_nonzero(coverage_manifest_digest)?;
        let declaration_id = NoExternalEffectsDeclarationId(hash_declaration(
            eligibility.id(), eligibility.authority_claim_id(), eligibility.authority_profile_id(),
            eligibility.authority_root_epoch(), eligibility.subject_id(),
            eligibility.target_realization_id(), eligibility.distributed_context_id(),
            eligibility.commit_time_unix_ms(), coverage_manifest_digest, backend.id(),
            backend.implementation_digest(), backend.backend_generation(),
        ));
        Ok(Self {
            schema_version: NO_EXTERNAL_EFFECTS_DECLARATION_SCHEMA_V1.to_owned(),
            commit_eligibility_id: eligibility.id(),
            original_authority_claim_id: eligibility.authority_claim_id(),
            authority_profile_id: eligibility.authority_profile_id(),
            authority_root_epoch: eligibility.authority_root_epoch(),
            subject_id: eligibility.subject_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            commit_time_unix_ms: eligibility.commit_time_unix_ms(), coverage_manifest_digest,
            backend_id: backend.id(), backend_implementation_digest: backend.implementation_digest(),
            backend_generation: backend.backend_generation(), declaration_id,
        })
    }

    pub fn validate(&self) -> Result<(), NoExternalEffectsError> {
        if self.schema_version != NO_EXTERNAL_EFFECTS_DECLARATION_SCHEMA_V1 {
            return Err(NoExternalEffectsError::UnsupportedDeclarationSchema(self.schema_version.clone()));
        }
        if self.authority_root_epoch == 0 || self.backend_generation == 0 || self.commit_time_unix_ms == 0 {
            return Err(NoExternalEffectsError::ZeroGenerationOrTime);
        }
        require_nonzero(self.coverage_manifest_digest)?;
        require_nonzero(self.backend_implementation_digest)?;
        let expected = NoExternalEffectsDeclarationId(hash_declaration(
            self.commit_eligibility_id, self.original_authority_claim_id,
            self.authority_profile_id, self.authority_root_epoch, self.subject_id,
            self.target_realization_id, self.distributed_context_id, self.commit_time_unix_ms,
            self.coverage_manifest_digest, self.backend_id, self.backend_implementation_digest,
            self.backend_generation,
        ));
        if expected != self.declaration_id { return Err(NoExternalEffectsError::DeclarationIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> NoExternalEffectsDeclarationId { self.declaration_id }
    pub fn commit_eligibility_id(&self) -> CommitEligibleTransitionId { self.commit_eligibility_id }
    pub fn original_authority_claim_id(&self) -> TransitionAuthorityClaimId { self.original_authority_claim_id }
    pub fn authority_profile_id(&self) -> TransitionAuthorityProfileId { self.authority_profile_id }
    pub fn authority_root_epoch(&self) -> u64 { self.authority_root_epoch }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn commit_time_unix_ms(&self) -> u64 { self.commit_time_unix_ms }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn backend_implementation_digest(&self) -> [u8; 32] { self.backend_implementation_digest }
    pub fn backend_generation(&self) -> u64 { self.backend_generation }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoExternalEffectsAuthorityClaimV1 {
    schema_version: String,
    declaration_id: NoExternalEffectsDeclarationId,
    original_authority_claim_id: TransitionAuthorityClaimId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    authorized_at_unix_ms: u64,
    claim_id: NoExternalEffectsAuthorityClaimId,
}

impl NoExternalEffectsAuthorityClaimV1 {
    pub fn new(
        declaration: &NoExternalEffectsDeclarationV1,
        profile: &TransitionAuthorityProfileV1,
    ) -> Result<Self, NoExternalEffectsError> {
        declaration.validate()?; profile.validate()?;
        if profile.id() != declaration.authority_profile_id()
            || profile.root_epoch() != declaration.authority_root_epoch()
        { return Err(NoExternalEffectsError::AuthorityRootMismatch); }
        let authorized_at_unix_ms = declaration.commit_time_unix_ms();
        let claim_id = NoExternalEffectsAuthorityClaimId(hash_authority_claim(
            declaration.id(), declaration.original_authority_claim_id(), profile.id(),
            profile.root_epoch(), authorized_at_unix_ms,
        ));
        Ok(Self {
            schema_version: NO_EXTERNAL_EFFECTS_AUTHORITY_CLAIM_SCHEMA_V1.to_owned(),
            declaration_id: declaration.id(), original_authority_claim_id: declaration.original_authority_claim_id(),
            authority_profile_id: profile.id(), authority_root_epoch: profile.root_epoch(),
            authorized_at_unix_ms, claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), NoExternalEffectsError> {
        if self.schema_version != NO_EXTERNAL_EFFECTS_AUTHORITY_CLAIM_SCHEMA_V1 {
            return Err(NoExternalEffectsError::UnsupportedAuthoritySchema(self.schema_version.clone()));
        }
        if self.authority_root_epoch == 0 || self.authorized_at_unix_ms == 0 {
            return Err(NoExternalEffectsError::ZeroGenerationOrTime);
        }
        let expected = NoExternalEffectsAuthorityClaimId(hash_authority_claim(
            self.declaration_id, self.original_authority_claim_id, self.authority_profile_id,
            self.authority_root_epoch, self.authorized_at_unix_ms,
        ));
        if expected != self.claim_id { return Err(NoExternalEffectsError::AuthorityClaimIdentityMismatch); }
        Ok(())
    }
    pub fn id(&self) -> NoExternalEffectsAuthorityClaimId { self.claim_id }
}

pub fn canonical_no_external_effects_authority_claim_bytes(
    claim: &NoExternalEffectsAuthorityClaimV1,
) -> Result<Vec<u8>, NoExternalEffectsError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(256); out.extend_from_slice(AUTH_WIRE_DOMAIN);
    out.extend_from_slice(claim.declaration_id.as_bytes());
    out.extend_from_slice(claim.original_authority_claim_id.as_bytes());
    out.extend_from_slice(claim.authority_profile_id.as_bytes());
    out.extend_from_slice(&claim.authority_root_epoch.to_le_bytes());
    out.extend_from_slice(&claim.authorized_at_unix_ms.to_le_bytes());
    out.extend_from_slice(claim.claim_id.as_bytes()); Ok(out)
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedNoExternalEffectsAuthorityV1 {
    claim: NoExternalEffectsAuthorityClaimV1,
    profile: TransitionAuthorityProfileV1,
    evidence_digest: [u8; 32],
    evidence_id: AuthenticatedNoExternalEffectsAuthorityId,
}

impl AuthenticatedNoExternalEffectsAuthorityV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: NoExternalEffectsAuthorityClaimV1,
        profile: TransitionAuthorityProfileV1,
        evidence_digest: [u8; 32],
    ) -> Result<Self, NoExternalEffectsError> {
        claim.validate()?; profile.validate()?; require_nonzero(evidence_digest)?;
        if claim.authority_profile_id != profile.id() || claim.authority_root_epoch != profile.root_epoch() {
            return Err(NoExternalEffectsError::AuthorityRootMismatch);
        }
        let evidence_id = AuthenticatedNoExternalEffectsAuthorityId(domain_hash_parts(
            AUTH_EVIDENCE_DOMAIN,
            &[claim.id().as_bytes(), profile.id().as_bytes(), &profile.root_epoch().to_le_bytes(), &evidence_digest],
        ));
        Ok(Self { claim, profile, evidence_digest, evidence_id })
    }
}

#[derive(Debug, Clone)]
pub struct QualifiedNoExternalEffectsAuthorizationV1 {
    authorization_id: QualifiedNoExternalEffectsAuthorizationId,
    declaration_id: NoExternalEffectsDeclarationId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
}

impl QualifiedNoExternalEffectsAuthorizationV1 {
    pub(crate) fn qualify(
        declaration: &NoExternalEffectsDeclarationV1,
        evidence: &AuthenticatedNoExternalEffectsAuthorityV1,
    ) -> Result<Self, NoExternalEffectsError> {
        declaration.validate()?; evidence.claim.validate()?; evidence.profile.validate()?;
        if evidence.claim.declaration_id != declaration.id()
            || evidence.claim.original_authority_claim_id != declaration.original_authority_claim_id()
            || evidence.profile.id() != declaration.authority_profile_id()
            || evidence.profile.root_epoch() != declaration.authority_root_epoch()
            || evidence.claim.authorized_at_unix_ms != declaration.commit_time_unix_ms()
        { return Err(NoExternalEffectsError::AuthorityContextMismatch); }
        let authorization_id = QualifiedNoExternalEffectsAuthorizationId(domain_hash_parts(
            QUALIFIED_AUTH_DOMAIN,
            &[declaration.id().as_bytes(), evidence.claim.id().as_bytes(), evidence.evidence_id.as_bytes()],
        ));
        Ok(Self { authorization_id, declaration_id: declaration.id(),
            authority_profile_id: evidence.profile.id(), authority_root_epoch: evidence.profile.root_epoch() })
    }
    pub fn id(&self) -> QualifiedNoExternalEffectsAuthorizationId { self.authorization_id }
    pub fn declaration_id(&self) -> NoExternalEffectsDeclarationId { self.declaration_id }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoExternalEffectsCoverageOutcomeV1 { NoExternalEffects, EffectsFound, Unknown }
impl NoExternalEffectsCoverageOutcomeV1 { fn tag(self) -> u8 { match self { Self::NoExternalEffects=>1, Self::EffectsFound=>2, Self::Unknown=>3 } } }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoExternalEffectsCoverageClaimV1 {
    schema_version: String,
    declaration_id: NoExternalEffectsDeclarationId,
    coverage_profile_id: EffectCoverageProfileId,
    verifier_profile_id: VerifierProfileId,
    analyzed_at_unix_ms: u64,
    transaction_challenge: [u8; 32],
    outcome: NoExternalEffectsCoverageOutcomeV1,
    discovered_effect_set_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
    claim_id: NoExternalEffectsCoverageClaimId,
}

impl NoExternalEffectsCoverageClaimV1 {
    pub fn new(
        declaration: &NoExternalEffectsDeclarationV1,
        coverage_profile: &EffectCoverageProfileV1,
        analyzed_at_unix_ms: u64,
        transaction_challenge: [u8; 32],
        outcome: NoExternalEffectsCoverageOutcomeV1,
        discovered_effect_set_digest: Option<[u8; 32]>,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, NoExternalEffectsError> {
        declaration.validate()?; if analyzed_at_unix_ms == 0 { return Err(NoExternalEffectsError::ZeroGenerationOrTime); }
        require_nonzero(transaction_challenge)?; require_nonzero(raw_evidence_digest)?;
        validate_coverage_outcome(outcome, discovered_effect_set_digest)?;
        let claim_id = NoExternalEffectsCoverageClaimId(hash_coverage_claim(
            declaration.id(), coverage_profile.id(), coverage_profile.verifier_profile_id(),
            analyzed_at_unix_ms, transaction_challenge, outcome, discovered_effect_set_digest,
            raw_evidence_digest,
        ));
        Ok(Self { schema_version: NO_EXTERNAL_EFFECTS_COVERAGE_CLAIM_SCHEMA_V1.to_owned(),
            declaration_id: declaration.id(), coverage_profile_id: coverage_profile.id(),
            verifier_profile_id: coverage_profile.verifier_profile_id(), analyzed_at_unix_ms,
            transaction_challenge, outcome, discovered_effect_set_digest, raw_evidence_digest, claim_id })
    }
    pub fn validate(&self, declaration: &NoExternalEffectsDeclarationV1) -> Result<(), NoExternalEffectsError> {
        if self.schema_version != NO_EXTERNAL_EFFECTS_COVERAGE_CLAIM_SCHEMA_V1 {
            return Err(NoExternalEffectsError::UnsupportedCoverageSchema(self.schema_version.clone()));
        }
        if self.declaration_id != declaration.id() { return Err(NoExternalEffectsError::CoverageContextMismatch); }
        if self.analyzed_at_unix_ms == 0 { return Err(NoExternalEffectsError::ZeroGenerationOrTime); }
        require_nonzero(self.transaction_challenge)?; require_nonzero(self.raw_evidence_digest)?;
        validate_coverage_outcome(self.outcome, self.discovered_effect_set_digest)?;
        let expected = NoExternalEffectsCoverageClaimId(hash_coverage_claim(
            self.declaration_id, self.coverage_profile_id, self.verifier_profile_id,
            self.analyzed_at_unix_ms, self.transaction_challenge, self.outcome,
            self.discovered_effect_set_digest, self.raw_evidence_digest,
        ));
        if expected != self.claim_id { return Err(NoExternalEffectsError::CoverageClaimIdentityMismatch); }
        Ok(())
    }
    pub fn id(&self) -> NoExternalEffectsCoverageClaimId { self.claim_id }
}

pub fn canonical_no_external_effects_coverage_claim_bytes(
    claim: &NoExternalEffectsCoverageClaimV1,
    declaration: &NoExternalEffectsDeclarationV1,
) -> Result<Vec<u8>, NoExternalEffectsError> {
    claim.validate(declaration)?;
    let mut out = Vec::with_capacity(384); out.extend_from_slice(COVERAGE_WIRE_DOMAIN);
    out.extend_from_slice(claim.declaration_id.as_bytes()); out.extend_from_slice(claim.coverage_profile_id.as_bytes());
    out.extend_from_slice(claim.verifier_profile_id.as_bytes()); out.extend_from_slice(&claim.analyzed_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.transaction_challenge); out.push(claim.outcome.tag());
    match claim.discovered_effect_set_digest { Some(d)=>{out.push(1);out.extend_from_slice(&d);} None=>out.push(0) }
    out.extend_from_slice(&claim.raw_evidence_digest); out.extend_from_slice(claim.claim_id.as_bytes()); Ok(out)
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedNoExternalEffectsCoverageV1 {
    claim: NoExternalEffectsCoverageClaimV1,
    coverage_profile: EffectCoverageProfileV1,
    verifier: VerifierProfileV1,
    evidence_digest: [u8; 32],
    evidence_id: AuthenticatedNoExternalEffectsCoverageId,
}

impl AuthenticatedNoExternalEffectsCoverageV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: NoExternalEffectsCoverageClaimV1,
        declaration: &NoExternalEffectsDeclarationV1,
        coverage_profile: EffectCoverageProfileV1,
        verifier: VerifierProfileV1,
        evidence_digest: [u8; 32],
    ) -> Result<Self, NoExternalEffectsError> {
        declaration.validate()?; coverage_profile.validate_against(&verifier)?; claim.validate(declaration)?; require_nonzero(evidence_digest)?;
        if claim.coverage_profile_id != coverage_profile.id() || claim.verifier_profile_id != verifier.id() {
            return Err(NoExternalEffectsError::CoverageContextMismatch);
        }
        let evidence_id = AuthenticatedNoExternalEffectsCoverageId(domain_hash_parts(
            COVERAGE_AUTH_DOMAIN,
            &[claim.id().as_bytes(), coverage_profile.id().as_bytes(), verifier.id().as_bytes(),
                &verifier.root_epoch().to_le_bytes(), &evidence_digest],
        ));
        Ok(Self { claim, coverage_profile, verifier, evidence_digest, evidence_id })
    }
}

#[derive(Debug, Clone)]
pub struct QualifiedNoExternalEffectsCoverageV1 {
    coverage_id: QualifiedNoExternalEffectsCoverageId,
    declaration_id: NoExternalEffectsDeclarationId,
    coverage_profile_id: EffectCoverageProfileId,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    analyzed_at_unix_ms: u64,
    transaction_challenge: [u8; 32],
}

impl QualifiedNoExternalEffectsCoverageV1 {
    pub(crate) fn qualify(
        declaration: &NoExternalEffectsDeclarationV1,
        evidence: &AuthenticatedNoExternalEffectsCoverageV1,
        expected_challenge: [u8; 32],
    ) -> Result<Self, NoExternalEffectsError> {
        declaration.validate()?; evidence.coverage_profile.validate_against(&evidence.verifier)?;
        evidence.claim.validate(declaration)?; require_nonzero(expected_challenge)?;
        if evidence.claim.transaction_challenge != expected_challenge { return Err(NoExternalEffectsError::ChallengeMismatch); }
        if evidence.claim.outcome != NoExternalEffectsCoverageOutcomeV1::NoExternalEffects
            || evidence.claim.discovered_effect_set_digest != Some(canonical_empty_external_effect_set_digest())
        { return Err(NoExternalEffectsError::NoEffectsNotEstablished); }
        let coverage_id = QualifiedNoExternalEffectsCoverageId(domain_hash_parts(
            QUALIFIED_COVERAGE_DOMAIN,
            &[declaration.id().as_bytes(), evidence.claim.id().as_bytes(),
                evidence.coverage_profile.id().as_bytes(), evidence.verifier.id().as_bytes(),
                evidence.evidence_id.as_bytes(), &expected_challenge],
        ));
        Ok(Self { coverage_id, declaration_id: declaration.id(), coverage_profile_id: evidence.coverage_profile.id(),
            verifier_profile_id: evidence.verifier.id(), verifier_root_epoch: evidence.verifier.root_epoch(),
            analyzed_at_unix_ms: evidence.claim.analyzed_at_unix_ms, transaction_challenge: expected_challenge })
    }
    pub fn id(&self) -> QualifiedNoExternalEffectsCoverageId { self.coverage_id }
    pub fn declaration_id(&self) -> NoExternalEffectsDeclarationId { self.declaration_id }
    pub fn analyzed_at_unix_ms(&self) -> u64 { self.analyzed_at_unix_ms }
    pub fn transaction_challenge(&self) -> [u8; 32] { self.transaction_challenge }
}

/// Non-Clone exact no-effects constraint bound to trusted eligibility and one backend.
#[derive(Debug)]
pub struct NoEffectsAuthorizedTrustedEligibilityV1 {
    eligibility: TrustedCommitEligibilityV1,
    declaration: NoExternalEffectsDeclarationV1,
    authorization: QualifiedNoExternalEffectsAuthorizationV1,
    coverage: QualifiedNoExternalEffectsCoverageV1,
    backend: ExecutionBackendProfileV1,
}

impl NoEffectsAuthorizedTrustedEligibilityV1 {
    pub fn bind(
        eligibility: TrustedCommitEligibilityV1,
        declaration: NoExternalEffectsDeclarationV1,
        authorization: QualifiedNoExternalEffectsAuthorizationV1,
        coverage: QualifiedNoExternalEffectsCoverageV1,
        backend: ExecutionBackendProfileV1,
    ) -> Result<Self, NoExternalEffectsError> {
        declaration.validate()?; backend.validate()?;
        if declaration.commit_eligibility_id() != eligibility.eligibility_id()
            || declaration.subject_id() != eligibility.subject_id()
            || declaration.target_realization_id() != eligibility.target_realization_id()
            || declaration.distributed_context_id() != eligibility.distributed_context_id()
            || declaration.commit_time_unix_ms() != eligibility.commit_time_unix_ms()
            || declaration.backend_id() != backend.id()
            || declaration.backend_implementation_digest() != backend.implementation_digest()
            || declaration.backend_generation() != backend.backend_generation()
            || authorization.declaration_id() != declaration.id()
            || coverage.declaration_id() != declaration.id()
        { return Err(NoExternalEffectsError::TrustedEligibilityMismatch); }
        Ok(Self { eligibility, declaration, authorization, coverage, backend })
    }
    pub fn declaration(&self) -> &NoExternalEffectsDeclarationV1 { &self.declaration }
    pub fn authorization(&self) -> &QualifiedNoExternalEffectsAuthorizationV1 { &self.authorization }
    pub fn coverage(&self) -> &QualifiedNoExternalEffectsCoverageV1 { &self.coverage }
    pub fn backend(&self) -> &ExecutionBackendProfileV1 { &self.backend }
}

/// Active-A-bound no-effects eligibility. This is the input to the no-effects execution coordinator.
#[derive(Debug)]
pub struct NoEffectsKnownGoodBoundEligibilityV1 {
    bound: KnownGoodBoundTrustedCommitEligibilityV1,
    declaration: NoExternalEffectsDeclarationV1,
    authorization: QualifiedNoExternalEffectsAuthorizationV1,
    coverage: QualifiedNoExternalEffectsCoverageV1,
    backend: ExecutionBackendProfileV1,
}

impl NoEffectsKnownGoodBoundEligibilityV1 {
    pub fn bind_active_known_good(
        active: &ActiveKnownGoodSelectionV1,
        checkpoint: &QualifiedKnownGoodCheckpointV1,
        authorized: NoEffectsAuthorizedTrustedEligibilityV1,
    ) -> Result<Self, NoExternalEffectsError> {
        let NoEffectsAuthorizedTrustedEligibilityV1 { eligibility, declaration, authorization, coverage, backend } = authorized;
        let bound = KnownGoodBoundTrustedCommitEligibilityV1::bind(active, checkpoint, eligibility)?;
        if bound.lineage().subject_id() != declaration.subject_id()
            || bound.lineage().target_realization_id() != declaration.target_realization_id()
            || bound.lineage().distributed_context_id() != declaration.distributed_context_id()
            || bound.lineage().commit_time_unix_ms() != declaration.commit_time_unix_ms()
        { return Err(NoExternalEffectsError::KnownGoodLineageMismatch); }
        Ok(Self { bound, declaration, authorization, coverage, backend })
    }
    pub fn declaration(&self) -> &NoExternalEffectsDeclarationV1 { &self.declaration }
    pub fn authorization(&self) -> &QualifiedNoExternalEffectsAuthorizationV1 { &self.authorization }
    pub fn coverage(&self) -> &QualifiedNoExternalEffectsCoverageV1 { &self.coverage }
    pub fn backend(&self) -> &ExecutionBackendProfileV1 { &self.backend }
    pub(crate) fn into_parts(self) -> (KnownGoodBoundTrustedCommitEligibilityV1, NoExternalEffectsDeclarationV1,
        QualifiedNoExternalEffectsAuthorizationV1, QualifiedNoExternalEffectsCoverageV1, ExecutionBackendProfileV1) {
        (self.bound, self.declaration, self.authorization, self.coverage, self.backend)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NoExternalEffectsError {
    #[error(transparent)] Execution(#[from] ExecutionCapabilityError),
    #[error(transparent)] Authority(#[from] TransitionAuthorityError),
    #[error(transparent)] Verification(#[from] VerificationAdmissionError),
    #[error(transparent)] Coverage(#[from] EffectCoverageError),
    #[error(transparent)] Lineage(#[from] KnownGoodTransitionLineageError),
    #[error("unsupported no-effects declaration schema: {0}")] UnsupportedDeclarationSchema(String),
    #[error("unsupported no-effects authority schema: {0}")] UnsupportedAuthoritySchema(String),
    #[error("unsupported no-effects coverage schema: {0}")] UnsupportedCoverageSchema(String),
    #[error("no-effects digest must be non-zero")] ZeroDigest,
    #[error("no-effects generation/time must be non-zero")] ZeroGenerationOrTime,
    #[error("no-effects declaration identity mismatch")] DeclarationIdentityMismatch,
    #[error("no-effects authority root mismatch")] AuthorityRootMismatch,
    #[error("no-effects authority claim identity mismatch")] AuthorityClaimIdentityMismatch,
    #[error("no-effects authority context mismatch")] AuthorityContextMismatch,
    #[error("no-effects coverage context mismatch")] CoverageContextMismatch,
    #[error("no-effects coverage claim material is inconsistent")] CoverageOutcomeMismatch,
    #[error("no-effects coverage claim identity mismatch")] CoverageClaimIdentityMismatch,
    #[error("no-effects coverage challenge mismatch")] ChallengeMismatch,
    #[error("no-effects analysis did not establish the canonical empty effect set")]
    NoEffectsNotEstablished,
    #[error("no-effects trusted eligibility/backend/authority/coverage mismatch")]
    TrustedEligibilityMismatch,
    #[error("no-effects active known-good lineage mismatch")]
    KnownGoodLineageMismatch,
}

fn validate_coverage_outcome(
    outcome: NoExternalEffectsCoverageOutcomeV1,
    discovered: Option<[u8; 32]>,
) -> Result<(), NoExternalEffectsError> {
    match outcome {
        NoExternalEffectsCoverageOutcomeV1::NoExternalEffects => {
            if discovered != Some(canonical_empty_external_effect_set_digest()) {
                return Err(NoExternalEffectsError::CoverageOutcomeMismatch);
            }
        }
        NoExternalEffectsCoverageOutcomeV1::EffectsFound => {
            let Some(digest) = discovered else { return Err(NoExternalEffectsError::CoverageOutcomeMismatch); };
            require_nonzero(digest)?;
            if digest == canonical_empty_external_effect_set_digest() {
                return Err(NoExternalEffectsError::CoverageOutcomeMismatch);
            }
        }
        NoExternalEffectsCoverageOutcomeV1::Unknown => {
            if discovered.is_some() { return Err(NoExternalEffectsError::CoverageOutcomeMismatch); }
        }
    }
    Ok(())
}

fn require_nonzero(digest: [u8; 32]) -> Result<(), NoExternalEffectsError> {
    if digest == [0; 32] { return Err(NoExternalEffectsError::ZeroDigest); } Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_declaration(
    eligibility_id: CommitEligibleTransitionId, authority_claim_id: TransitionAuthorityClaimId,
    authority_profile_id: TransitionAuthorityProfileId, authority_root_epoch: u64,
    subject_id: ContinuitySubjectId, target_id: TargetRealizationId,
    context_id: DistributedStateContextId, commit_time: u64, manifest: [u8;32],
    backend_id: ExecutionBackendId, implementation: [u8;32], backend_generation: u64,
) -> [u8;32] {
    domain_hash_parts(DECLARATION_DOMAIN, &[eligibility_id.as_bytes(), authority_claim_id.as_bytes(),
        authority_profile_id.as_bytes(), &authority_root_epoch.to_le_bytes(), subject_id.as_bytes(),
        target_id.as_bytes(), context_id.as_bytes(), &commit_time.to_le_bytes(), &manifest,
        backend_id.as_bytes(), &implementation, &backend_generation.to_le_bytes()])
}
fn hash_authority_claim(declaration_id: NoExternalEffectsDeclarationId, authority_claim_id: TransitionAuthorityClaimId,
    profile_id: TransitionAuthorityProfileId, root_epoch: u64, authorized_at: u64) -> [u8;32] {
    domain_hash_parts(AUTH_CLAIM_DOMAIN, &[declaration_id.as_bytes(), authority_claim_id.as_bytes(),
        profile_id.as_bytes(), &root_epoch.to_le_bytes(), &authorized_at.to_le_bytes()])
}
#[allow(clippy::too_many_arguments)]
fn hash_coverage_claim(declaration_id: NoExternalEffectsDeclarationId, profile_id: EffectCoverageProfileId,
    verifier_id: VerifierProfileId, analyzed_at: u64, challenge:[u8;32], outcome:NoExternalEffectsCoverageOutcomeV1,
    discovered:Option<[u8;32]>, raw:[u8;32]) -> [u8;32] {
    let mut h=blake3::Hasher::new(); h.update(COVERAGE_CLAIM_DOMAIN); h.update(declaration_id.as_bytes());
    h.update(profile_id.as_bytes()); h.update(verifier_id.as_bytes()); h.update(&analyzed_at.to_le_bytes());
    h.update(&challenge); h.update(&[outcome.tag()]); match discovered { Some(d)=>{h.update(&[1]);h.update(&d);} None=>{h.update(&[0]);} }
    h.update(&raw); *h.finalize().as_bytes()
}
fn domain_hash_parts(domain:&[u8], parts:&[&[u8]])->[u8;32]{let mut h=blake3::Hasher::new();h.update(domain);for p in parts{h.update(&((*p).len() as u64).to_le_bytes());h.update(p);}*h.finalize().as_bytes()}

#[cfg(test)] mod tests { use super::*; #[test] fn empty_set_digest_is_not_zero(){assert_ne!(canonical_empty_external_effect_set_digest(),[0;32]);} }
