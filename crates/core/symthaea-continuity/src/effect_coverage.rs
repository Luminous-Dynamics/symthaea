// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Backend-relative completeness for declared external-effect coverage.
//!
//! “Complete” is deliberately relative to an exact executor implementation, exact
//! effect taxonomy, exact external-boundary definition, and exact analysis model.
//! It never means omniscience about arbitrary real-world effects.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backend_effect_authority::{
    BackendEffectAuthorityError, BackendEffectBindingId, BackendEffectBindingV1,
};
use crate::execution_capability::{ExecutionBackendId, ExecutionBackendProfileV1, ExecutionCapabilityError};
use crate::external_effect_authority::{ExternalEffectAuthorityError, ExternalEffectPlanId, ExternalEffectPlanV1};
use crate::external_effects::{ExternalEffectError, ExternalEffectObligationId};
use crate::scope::ContinuitySubjectId;
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};
use crate::witness::TargetRealizationId;

pub const EFFECT_COVERAGE_PROFILE_SCHEMA_V1: &str =
    "symthaea-continuity-effect-coverage-profile-v1";
pub const EFFECT_COVERAGE_SUBJECT_SCHEMA_V1: &str =
    "symthaea-continuity-effect-coverage-subject-v1";
pub const EFFECT_COVERAGE_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-effect-coverage-claim-v1";
pub const EFFECT_COVERAGE_AUTH_PURPOSE: &str =
    "symthaea.continuity.effect-coverage.v1";

const PROFILE_DOMAIN: &[u8] = b"symthaea.continuity.effect-coverage-profile.v1\0";
const SUBJECT_DOMAIN: &[u8] = b"symthaea.continuity.effect-coverage-subject.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.effect-coverage-claim.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.continuity.effect-coverage-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-effect-coverage.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-effect-coverage.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }
    };
}

digest_id!(EffectCoverageProfileId);
digest_id!(EffectCoverageSubjectId);
digest_id!(EffectCoverageClaimId);
digest_id!(AuthenticatedEffectCoverageId);
digest_id!(QualifiedExternalEffectCoverageId);

/// Exact verifier-owned interpretation of “all effects inside this boundary.”
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectCoverageProfileV1 {
    schema_version: String,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    analysis_model_digest: [u8; 32],
    external_boundary_definition_digest: [u8; 32],
    effect_taxonomy_digest: [u8; 32],
    profile_generation: u64,
    profile_id: EffectCoverageProfileId,
}

impl EffectCoverageProfileV1 {
    pub fn new(
        verifier: &VerifierProfileV1,
        analysis_model_digest: [u8; 32],
        external_boundary_definition_digest: [u8; 32],
        effect_taxonomy_digest: [u8; 32],
        profile_generation: u64,
    ) -> Result<Self, EffectCoverageError> {
        verifier.validate()?;
        validate_nonzero_digest(analysis_model_digest)?;
        validate_nonzero_digest(external_boundary_definition_digest)?;
        validate_nonzero_digest(effect_taxonomy_digest)?;
        if profile_generation == 0 { return Err(EffectCoverageError::ZeroGeneration); }
        let profile_id = EffectCoverageProfileId(hash_profile(
            verifier.id(), verifier.root_epoch(), analysis_model_digest,
            external_boundary_definition_digest, effect_taxonomy_digest, profile_generation,
        ));
        Ok(Self {
            schema_version: EFFECT_COVERAGE_PROFILE_SCHEMA_V1.to_owned(),
            verifier_profile_id: verifier.id(), verifier_root_epoch: verifier.root_epoch(),
            analysis_model_digest, external_boundary_definition_digest, effect_taxonomy_digest,
            profile_generation, profile_id,
        })
    }

    pub fn validate_against(&self, verifier: &VerifierProfileV1) -> Result<(), EffectCoverageError> {
        if self.schema_version != EFFECT_COVERAGE_PROFILE_SCHEMA_V1 {
            return Err(EffectCoverageError::UnsupportedProfileSchema(self.schema_version.clone()));
        }
        verifier.validate()?;
        if self.verifier_profile_id != verifier.id() || self.verifier_root_epoch != verifier.root_epoch() {
            return Err(EffectCoverageError::VerifierProfileMismatch);
        }
        validate_nonzero_digest(self.analysis_model_digest)?;
        validate_nonzero_digest(self.external_boundary_definition_digest)?;
        validate_nonzero_digest(self.effect_taxonomy_digest)?;
        if self.profile_generation == 0 { return Err(EffectCoverageError::ZeroGeneration); }
        let expected = EffectCoverageProfileId(hash_profile(
            self.verifier_profile_id, self.verifier_root_epoch, self.analysis_model_digest,
            self.external_boundary_definition_digest, self.effect_taxonomy_digest,
            self.profile_generation,
        ));
        if expected != self.profile_id { return Err(EffectCoverageError::ProfileIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> EffectCoverageProfileId { self.profile_id }
    pub fn verifier_profile_id(&self) -> VerifierProfileId { self.verifier_profile_id }
    pub fn verifier_root_epoch(&self) -> u64 { self.verifier_root_epoch }
    pub fn analysis_model_digest(&self) -> [u8; 32] { self.analysis_model_digest }
    pub fn external_boundary_definition_digest(&self) -> [u8; 32] { self.external_boundary_definition_digest }
    pub fn effect_taxonomy_digest(&self) -> [u8; 32] { self.effect_taxonomy_digest }
    pub fn generation(&self) -> u64 { self.profile_generation }
}

/// Exact backend + plan world whose declared obligations are being analyzed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectCoverageSubjectV1 {
    schema_version: String,
    plan_id: ExternalEffectPlanId,
    backend_binding_id: BackendEffectBindingId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    coverage_manifest_digest: [u8; 32],
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    declared_obligation_set_digest: [u8; 32],
    declared_obligation_count: u32,
    subject_digest: EffectCoverageSubjectId,
}

impl EffectCoverageSubjectV1 {
    pub fn new(
        plan: &ExternalEffectPlanV1,
        binding: &BackendEffectBindingV1,
        backend: &ExecutionBackendProfileV1,
    ) -> Result<Self, EffectCoverageError> {
        plan.validate()?;
        binding.validate()?;
        backend.validate()?;
        if binding.plan_id() != plan.id()
            || binding.subject_id() != plan.subject_id()
            || binding.target_realization_id() != plan.target_realization_id()
            || binding.coverage_manifest_digest() != plan.coverage_manifest_digest()
            || binding.backend_id() != backend.id()
            || binding.backend_implementation_digest() != backend.implementation_digest()
            || binding.backend_generation() != backend.backend_generation()
        {
            return Err(EffectCoverageError::BackendPlanMismatch);
        }
        if plan.obligations().is_empty() {
            return Err(EffectCoverageError::EmptyDeclaredSetRequiresNoEffectsProof);
        }
        let declared_obligation_set_digest = hash_obligation_set(plan.obligations().iter().map(|o| o.id()));
        let declared_obligation_count = u32::try_from(plan.obligations().len())
            .map_err(|_| EffectCoverageError::TooManyObligations)?;
        let subject_digest = EffectCoverageSubjectId(hash_subject(
            plan.id(), binding.id(), plan.subject_id(), plan.target_realization_id(),
            plan.coverage_manifest_digest(), backend.id(), backend.implementation_digest(),
            backend.backend_generation(), declared_obligation_set_digest, declared_obligation_count,
        ));
        Ok(Self {
            schema_version: EFFECT_COVERAGE_SUBJECT_SCHEMA_V1.to_owned(),
            plan_id: plan.id(), backend_binding_id: binding.id(), subject_id: plan.subject_id(),
            target_realization_id: plan.target_realization_id(),
            coverage_manifest_digest: plan.coverage_manifest_digest(), backend_id: backend.id(),
            backend_implementation_digest: backend.implementation_digest(),
            backend_generation: backend.backend_generation(), declared_obligation_set_digest,
            declared_obligation_count, subject_digest,
        })
    }

    pub fn validate(&self) -> Result<(), EffectCoverageError> {
        if self.schema_version != EFFECT_COVERAGE_SUBJECT_SCHEMA_V1 {
            return Err(EffectCoverageError::UnsupportedSubjectSchema(self.schema_version.clone()));
        }
        validate_nonzero_digest(self.coverage_manifest_digest)?;
        validate_nonzero_digest(self.backend_implementation_digest)?;
        validate_nonzero_digest(self.declared_obligation_set_digest)?;
        if self.backend_generation == 0 || self.declared_obligation_count == 0 {
            return Err(EffectCoverageError::ZeroGeneration);
        }
        let expected = EffectCoverageSubjectId(hash_subject(
            self.plan_id, self.backend_binding_id, self.subject_id, self.target_realization_id,
            self.coverage_manifest_digest, self.backend_id, self.backend_implementation_digest,
            self.backend_generation, self.declared_obligation_set_digest,
            self.declared_obligation_count,
        ));
        if expected != self.subject_digest { return Err(EffectCoverageError::SubjectIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> EffectCoverageSubjectId { self.subject_digest }
    pub fn plan_id(&self) -> ExternalEffectPlanId { self.plan_id }
    pub fn backend_binding_id(&self) -> BackendEffectBindingId { self.backend_binding_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn backend_implementation_digest(&self) -> [u8; 32] { self.backend_implementation_digest }
    pub fn backend_generation(&self) -> u64 { self.backend_generation }
    pub fn declared_obligation_set_digest(&self) -> [u8; 32] { self.declared_obligation_set_digest }
    pub fn declared_obligation_count(&self) -> u32 { self.declared_obligation_count }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectCoverageOutcomeV1 {
    CompleteDeclaredEffects,
    Incomplete,
    Unknown,
}
impl EffectCoverageOutcomeV1 {
    fn tag(self) -> u8 { match self { Self::CompleteDeclaredEffects => 1, Self::Incomplete => 2, Self::Unknown => 3 } }
}

/// Transportable verifier claim. Evidence strength belongs to the provisioned verifier,
/// never to this raw claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectCoverageClaimV1 {
    schema_version: String,
    coverage_subject_id: EffectCoverageSubjectId,
    coverage_profile_id: EffectCoverageProfileId,
    verifier_profile_id: VerifierProfileId,
    analyzed_at_unix_ms: u64,
    transaction_challenge: [u8; 32],
    outcome: EffectCoverageOutcomeV1,
    discovered_effect_set_digest: Option<[u8; 32]>,
    uncovered_effect_set_digest: Option<[u8; 32]>,
    raw_evidence_digest: [u8; 32],
    claim_id: EffectCoverageClaimId,
}

impl EffectCoverageClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &EffectCoverageSubjectV1,
        profile: &EffectCoverageProfileV1,
        analyzed_at_unix_ms: u64,
        transaction_challenge: [u8; 32],
        outcome: EffectCoverageOutcomeV1,
        discovered_effect_set_digest: Option<[u8; 32]>,
        uncovered_effect_set_digest: Option<[u8; 32]>,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, EffectCoverageError> {
        subject.validate()?;
        if analyzed_at_unix_ms == 0 { return Err(EffectCoverageError::ZeroAnalysisTime); }
        validate_nonzero_digest(transaction_challenge)?;
        validate_nonzero_digest(raw_evidence_digest)?;
        validate_outcome_material(subject, outcome, discovered_effect_set_digest, uncovered_effect_set_digest)?;
        let claim_id = EffectCoverageClaimId(hash_claim(
            subject.id(), profile.id(), profile.verifier_profile_id(), analyzed_at_unix_ms,
            transaction_challenge, outcome, discovered_effect_set_digest,
            uncovered_effect_set_digest, raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: EFFECT_COVERAGE_CLAIM_SCHEMA_V1.to_owned(),
            coverage_subject_id: subject.id(), coverage_profile_id: profile.id(),
            verifier_profile_id: profile.verifier_profile_id(), analyzed_at_unix_ms,
            transaction_challenge, outcome, discovered_effect_set_digest,
            uncovered_effect_set_digest, raw_evidence_digest, claim_id,
        })
    }

    pub fn validate(&self, subject: &EffectCoverageSubjectV1) -> Result<(), EffectCoverageError> {
        if self.schema_version != EFFECT_COVERAGE_CLAIM_SCHEMA_V1 {
            return Err(EffectCoverageError::UnsupportedClaimSchema(self.schema_version.clone()));
        }
        if self.coverage_subject_id != subject.id() { return Err(EffectCoverageError::ClaimSubjectMismatch); }
        if self.analyzed_at_unix_ms == 0 { return Err(EffectCoverageError::ZeroAnalysisTime); }
        validate_nonzero_digest(self.transaction_challenge)?;
        validate_nonzero_digest(self.raw_evidence_digest)?;
        validate_outcome_material(subject, self.outcome, self.discovered_effect_set_digest, self.uncovered_effect_set_digest)?;
        let expected = EffectCoverageClaimId(hash_claim(
            self.coverage_subject_id, self.coverage_profile_id, self.verifier_profile_id,
            self.analyzed_at_unix_ms, self.transaction_challenge, self.outcome,
            self.discovered_effect_set_digest, self.uncovered_effect_set_digest,
            self.raw_evidence_digest,
        ));
        if expected != self.claim_id { return Err(EffectCoverageError::ClaimIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> EffectCoverageClaimId { self.claim_id }
}

pub fn canonical_effect_coverage_claim_bytes(
    claim: &EffectCoverageClaimV1,
    subject: &EffectCoverageSubjectV1,
) -> Result<Vec<u8>, EffectCoverageError> {
    claim.validate(subject)?;
    let mut out = Vec::with_capacity(512);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.coverage_subject_id.as_bytes());
    out.extend_from_slice(claim.coverage_profile_id.as_bytes());
    out.extend_from_slice(claim.verifier_profile_id.as_bytes());
    out.extend_from_slice(&claim.analyzed_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.transaction_challenge);
    out.push(claim.outcome.tag());
    encode_optional_digest(&mut out, claim.discovered_effect_set_digest);
    encode_optional_digest(&mut out, claim.uncovered_effect_set_digest);
    out.extend_from_slice(&claim.raw_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_effect_coverage_claim_digest(
    claim: &EffectCoverageClaimV1,
    subject: &EffectCoverageSubjectV1,
) -> Result<[u8; 32], EffectCoverageError> {
    Ok(*blake3::hash(&canonical_effect_coverage_claim_bytes(claim, subject)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedEffectCoverageV1 {
    claim: EffectCoverageClaimV1,
    profile: EffectCoverageProfileV1,
    verifier: VerifierProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedEffectCoverageId,
}

impl AuthenticatedEffectCoverageV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: EffectCoverageClaimV1,
        subject: &EffectCoverageSubjectV1,
        profile: EffectCoverageProfileV1,
        verifier: VerifierProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, EffectCoverageError> {
        subject.validate()?;
        profile.validate_against(&verifier)?;
        claim.validate(subject)?;
        if claim.coverage_profile_id != profile.id()
            || claim.verifier_profile_id != verifier.id()
        {
            return Err(EffectCoverageError::VerifierProfileMismatch);
        }
        validate_nonzero_digest(authentication_evidence_digest)?;
        let evidence_id = AuthenticatedEffectCoverageId(domain_hash_parts(
            AUTH_DOMAIN,
            &[claim.id().as_bytes(), profile.id().as_bytes(), verifier.id().as_bytes(),
                &verifier.root_epoch().to_le_bytes(), &authentication_evidence_digest],
        ));
        Ok(Self { claim, profile, verifier, authentication_evidence_digest, evidence_id })
    }
}

/// Non-Serde proof that, under one exact analysis/boundary/taxonomy profile, the
/// verifier found the declared obligation set to be complete for the exact backend.
#[derive(Debug, Clone)]
pub struct QualifiedExternalEffectCoverageV1 {
    coverage_id: QualifiedExternalEffectCoverageId,
    coverage_subject_id: EffectCoverageSubjectId,
    plan_id: ExternalEffectPlanId,
    backend_binding_id: BackendEffectBindingId,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    coverage_manifest_digest: [u8; 32],
    declared_obligation_set_digest: [u8; 32],
    profile_id: EffectCoverageProfileId,
    verifier_profile_id: VerifierProfileId,
    verifier_root_epoch: u64,
    analysis_model_digest: [u8; 32],
    external_boundary_definition_digest: [u8; 32],
    effect_taxonomy_digest: [u8; 32],
    analyzed_at_unix_ms: u64,
    transaction_challenge: [u8; 32],
}

impl QualifiedExternalEffectCoverageV1 {
    pub(crate) fn qualify(
        subject: &EffectCoverageSubjectV1,
        authenticated: &AuthenticatedEffectCoverageV1,
        expected_transaction_challenge: [u8; 32],
    ) -> Result<Self, EffectCoverageError> {
        subject.validate()?;
        authenticated.profile.validate_against(&authenticated.verifier)?;
        authenticated.claim.validate(subject)?;
        validate_nonzero_digest(expected_transaction_challenge)?;
        let claim = &authenticated.claim;
        if claim.transaction_challenge != expected_transaction_challenge {
            return Err(EffectCoverageError::ChallengeMismatch);
        }
        if claim.outcome != EffectCoverageOutcomeV1::CompleteDeclaredEffects
            || claim.discovered_effect_set_digest != Some(subject.declared_obligation_set_digest())
            || claim.uncovered_effect_set_digest.is_some()
        {
            return Err(EffectCoverageError::CoverageNotComplete);
        }
        let coverage_id = QualifiedExternalEffectCoverageId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[subject.id().as_bytes(), claim.id().as_bytes(), authenticated.profile.id().as_bytes(),
                authenticated.verifier.id().as_bytes(), authenticated.evidence_id.as_bytes(),
                &expected_transaction_challenge],
        ));
        Ok(Self {
            coverage_id, coverage_subject_id: subject.id(), plan_id: subject.plan_id(),
            backend_binding_id: subject.backend_binding_id(), backend_id: subject.backend_id(),
            backend_implementation_digest: subject.backend_implementation_digest(),
            backend_generation: subject.backend_generation(),
            coverage_manifest_digest: subject.coverage_manifest_digest(),
            declared_obligation_set_digest: subject.declared_obligation_set_digest(),
            profile_id: authenticated.profile.id(), verifier_profile_id: authenticated.verifier.id(),
            verifier_root_epoch: authenticated.verifier.root_epoch(),
            analysis_model_digest: authenticated.profile.analysis_model_digest(),
            external_boundary_definition_digest: authenticated.profile.external_boundary_definition_digest(),
            effect_taxonomy_digest: authenticated.profile.effect_taxonomy_digest(),
            analyzed_at_unix_ms: claim.analyzed_at_unix_ms,
            transaction_challenge: claim.transaction_challenge,
        })
    }

    pub fn id(&self) -> QualifiedExternalEffectCoverageId { self.coverage_id }
    pub fn coverage_subject_id(&self) -> EffectCoverageSubjectId { self.coverage_subject_id }
    pub fn plan_id(&self) -> ExternalEffectPlanId { self.plan_id }
    pub fn backend_binding_id(&self) -> BackendEffectBindingId { self.backend_binding_id }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn backend_implementation_digest(&self) -> [u8; 32] { self.backend_implementation_digest }
    pub fn backend_generation(&self) -> u64 { self.backend_generation }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn declared_obligation_set_digest(&self) -> [u8; 32] { self.declared_obligation_set_digest }
    pub fn analyzed_at_unix_ms(&self) -> u64 { self.analyzed_at_unix_ms }
    pub fn transaction_challenge(&self) -> [u8; 32] { self.transaction_challenge }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EffectCoverageError {
    #[error(transparent)] Verification(#[from] VerificationAdmissionError),
    #[error(transparent)] BackendAuthority(#[from] BackendEffectAuthorityError),
    #[error(transparent)] Execution(#[from] ExecutionCapabilityError),
    #[error(transparent)] EffectAuthority(#[from] ExternalEffectAuthorityError),
    #[error(transparent)] ExternalEffect(#[from] ExternalEffectError),
    #[error("unsupported effect-coverage profile schema: {0}")] UnsupportedProfileSchema(String),
    #[error("unsupported effect-coverage subject schema: {0}")] UnsupportedSubjectSchema(String),
    #[error("unsupported effect-coverage claim schema: {0}")] UnsupportedClaimSchema(String),
    #[error("effect-coverage digest must be non-zero")] ZeroDigest,
    #[error("effect-coverage generation/count must be non-zero")] ZeroGeneration,
    #[error("effect-coverage verifier profile/root mismatch")] VerifierProfileMismatch,
    #[error("effect-coverage profile identity mismatch")] ProfileIdentityMismatch,
    #[error("effect-coverage backend/plan/binding mismatch")] BackendPlanMismatch,
    #[error("empty declared effect sets require the separate no-effects theorem")]
    EmptyDeclaredSetRequiresNoEffectsProof,
    #[error("too many effect obligations for V1")]
    TooManyObligations,
    #[error("effect-coverage subject identity mismatch")] SubjectIdentityMismatch,
    #[error("effect-coverage analysis time must be non-zero")] ZeroAnalysisTime,
    #[error("effect-coverage claim belongs to another subject")] ClaimSubjectMismatch,
    #[error("effect-coverage outcome material is inconsistent")]
    InvalidOutcomeMaterial,
    #[error("effect-coverage claim identity mismatch")] ClaimIdentityMismatch,
    #[error("effect-coverage transaction challenge mismatch")] ChallengeMismatch,
    #[error("effect-coverage analysis did not establish complete declared effects")]
    CoverageNotComplete,
}

fn validate_outcome_material(
    subject: &EffectCoverageSubjectV1,
    outcome: EffectCoverageOutcomeV1,
    discovered: Option<[u8; 32]>,
    uncovered: Option<[u8; 32]>,
) -> Result<(), EffectCoverageError> {
    match outcome {
        EffectCoverageOutcomeV1::CompleteDeclaredEffects => {
            if discovered != Some(subject.declared_obligation_set_digest()) || uncovered.is_some() {
                return Err(EffectCoverageError::InvalidOutcomeMaterial);
            }
        }
        EffectCoverageOutcomeV1::Incomplete => {
            let Some(uncovered) = uncovered else { return Err(EffectCoverageError::InvalidOutcomeMaterial); };
            validate_nonzero_digest(uncovered)?;
            if let Some(discovered) = discovered { validate_nonzero_digest(discovered)?; }
        }
        EffectCoverageOutcomeV1::Unknown => {
            if discovered.is_some() || uncovered.is_some() {
                return Err(EffectCoverageError::InvalidOutcomeMaterial);
            }
        }
    }
    Ok(())
}

fn validate_nonzero_digest(digest: [u8; 32]) -> Result<(), EffectCoverageError> {
    if digest == [0; 32] { return Err(EffectCoverageError::ZeroDigest); }
    Ok(())
}

fn hash_obligation_set(ids: impl Iterator<Item = ExternalEffectObligationId>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.continuity.effect-obligation-set.v1\0");
    let ids = ids.collect::<Vec<_>>();
    hasher.update(&(ids.len() as u64).to_le_bytes());
    for id in ids { hasher.update(id.as_bytes()); }
    *hasher.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn hash_profile(
    verifier_profile_id: VerifierProfileId, verifier_root_epoch: u64,
    analysis_model_digest: [u8; 32], boundary_digest: [u8; 32], taxonomy_digest: [u8; 32],
    generation: u64,
) -> [u8; 32] {
    domain_hash_parts(PROFILE_DOMAIN, &[
        verifier_profile_id.as_bytes(), &verifier_root_epoch.to_le_bytes(), &analysis_model_digest,
        &boundary_digest, &taxonomy_digest, &generation.to_le_bytes(),
    ])
}

#[allow(clippy::too_many_arguments)]
fn hash_subject(
    plan_id: ExternalEffectPlanId, binding_id: BackendEffectBindingId, subject_id: ContinuitySubjectId,
    target_id: TargetRealizationId, manifest_digest: [u8; 32], backend_id: ExecutionBackendId,
    implementation_digest: [u8; 32], backend_generation: u64,
    declared_set_digest: [u8; 32], declared_count: u32,
) -> [u8; 32] {
    domain_hash_parts(SUBJECT_DOMAIN, &[
        plan_id.as_bytes(), binding_id.as_bytes(), subject_id.as_bytes(), target_id.as_bytes(),
        &manifest_digest, backend_id.as_bytes(), &implementation_digest,
        &backend_generation.to_le_bytes(), &declared_set_digest, &declared_count.to_le_bytes(),
    ])
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    subject_id: EffectCoverageSubjectId, profile_id: EffectCoverageProfileId,
    verifier_profile_id: VerifierProfileId, analyzed_at: u64, challenge: [u8; 32],
    outcome: EffectCoverageOutcomeV1, discovered: Option<[u8; 32]>,
    uncovered: Option<[u8; 32]>, raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CLAIM_DOMAIN);
    hasher.update(subject_id.as_bytes());
    hasher.update(profile_id.as_bytes());
    hasher.update(verifier_profile_id.as_bytes());
    hasher.update(&analyzed_at.to_le_bytes());
    hasher.update(&challenge);
    hasher.update(&[outcome.tag()]);
    hash_optional_digest(&mut hasher, discovered);
    hash_optional_digest(&mut hasher, uncovered);
    hasher.update(&raw_evidence_digest);
    *hasher.finalize().as_bytes()
}

fn hash_optional_digest(hasher: &mut blake3::Hasher, digest: Option<[u8; 32]>) {
    match digest {
        Some(digest) => { hasher.update(&[1]); hasher.update(&digest); }
        None => { hasher.update(&[0]); }
    }
}

fn encode_optional_digest(out: &mut Vec<u8>, digest: Option<[u8; 32]>) {
    match digest {
        Some(digest) => { out.push(1); out.extend_from_slice(&digest); }
        None => out.push(0),
    }
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for part in parts {
        hasher.update(&((*part).len() as u64).to_le_bytes());
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coverage_domains_are_distinct() {
        assert_ne!(PROFILE_DOMAIN, SUBJECT_DOMAIN);
        assert_ne!(SUBJECT_DOMAIN, CLAIM_DOMAIN);
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
        assert_ne!(WIRE_DOMAIN, AUTH_DOMAIN);
        assert_ne!(AUTH_DOMAIN, QUALIFIED_DOMAIN);
    }
}
