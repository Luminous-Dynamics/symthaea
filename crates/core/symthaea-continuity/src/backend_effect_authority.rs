// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind an already-authorized external-effect plan to one exact execution backend.
//!
//! A plan that authorizes an effect surface for A -> B must not be reusable with a
//! different backend implementation whose real side effects may differ. This layer
//! adds a second, domain-separated authorization under the same transition authority
//! root over the exact backend identity, implementation digest, and generation.
//!
//! `EffectAuthorization != BackendBinding != BackendEffectAuthorization != Capability`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution_capability::{
    ExecutionBackendId, ExecutionBackendProfileV1, ExecutionCapabilityError,
};
use crate::external_effect_authority::{
    EffectScopedKnownGoodBoundEligibilityV1, ExternalEffectAuthorityError,
    ExternalEffectPlanId, QualifiedExternalEffectAuthorizationId,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_authority::{
    TransitionAuthorityError, TransitionAuthorityProfileId, TransitionAuthorityProfileV1,
};
use crate::witness::TargetRealizationId;

pub const BACKEND_EFFECT_BINDING_SCHEMA_V1: &str =
    "symthaea-continuity-backend-effect-binding-v1";
pub const BACKEND_EFFECT_AUTHORITY_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-backend-effect-authority-claim-v1";
pub const BACKEND_EFFECT_AUTHORITY_PURPOSE: &str =
    "symthaea.continuity.backend-effect-authority.v1";

const BINDING_DOMAIN: &[u8] = b"symthaea.continuity.backend-effect-binding.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.backend-effect-authority-claim.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.continuity.backend-effect-authority-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-backend-effect-authority.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-backend-effect-authority.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }
    };
}

digest_id!(BackendEffectBindingId);
digest_id!(BackendEffectAuthorityClaimId);
digest_id!(AuthenticatedBackendEffectAuthorityId);
digest_id!(QualifiedBackendEffectAuthorizationId);

/// Serializable descriptive binding between one exact authorized effect plan and one
/// exact executor implementation. It is not authority by itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendEffectBindingV1 {
    schema_version: String,
    plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    coverage_manifest_digest: [u8; 32],
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    binding_id: BackendEffectBindingId,
}

impl BackendEffectBindingV1 {
    pub fn new(
        scoped: &EffectScopedKnownGoodBoundEligibilityV1,
        backend: &ExecutionBackendProfileV1,
    ) -> Result<Self, BackendEffectAuthorityError> {
        backend.validate()?;
        scoped.plan().validate()?;
        let authorization = scoped.authorization();
        let plan = scoped.plan();
        if authorization.plan_id() != plan.id()
            || authorization.subject_id() != plan.subject_id()
            || authorization.target_realization_id() != plan.target_realization_id()
            || authorization.coverage_manifest_digest() != plan.coverage_manifest_digest()
        {
            return Err(BackendEffectAuthorityError::EffectScopeMismatch);
        }
        let binding_id = BackendEffectBindingId(hash_binding(
            plan.id(), authorization.id(), authorization.authority_profile_id(),
            authorization.authority_root_epoch(), plan.subject_id(),
            plan.target_realization_id(), plan.coverage_manifest_digest(), backend.id(),
            backend.implementation_digest(), backend.backend_generation(),
        ));
        Ok(Self {
            schema_version: BACKEND_EFFECT_BINDING_SCHEMA_V1.to_owned(),
            plan_id: plan.id(),
            effect_authorization_id: authorization.id(),
            authority_profile_id: authorization.authority_profile_id(),
            authority_root_epoch: authorization.authority_root_epoch(),
            subject_id: plan.subject_id(),
            target_realization_id: plan.target_realization_id(),
            coverage_manifest_digest: plan.coverage_manifest_digest(),
            backend_id: backend.id(),
            backend_implementation_digest: backend.implementation_digest(),
            backend_generation: backend.backend_generation(),
            binding_id,
        })
    }

    pub fn validate(&self) -> Result<(), BackendEffectAuthorityError> {
        if self.schema_version != BACKEND_EFFECT_BINDING_SCHEMA_V1 {
            return Err(BackendEffectAuthorityError::UnsupportedBindingSchema(
                self.schema_version.clone(),
            ));
        }
        if self.authority_root_epoch == 0 || self.backend_generation == 0 {
            return Err(BackendEffectAuthorityError::ZeroGeneration);
        }
        if self.coverage_manifest_digest == [0; 32]
            || self.backend_implementation_digest == [0; 32]
        {
            return Err(BackendEffectAuthorityError::ZeroDigest);
        }
        let expected = BackendEffectBindingId(hash_binding(
            self.plan_id, self.effect_authorization_id, self.authority_profile_id,
            self.authority_root_epoch, self.subject_id, self.target_realization_id,
            self.coverage_manifest_digest, self.backend_id,
            self.backend_implementation_digest, self.backend_generation,
        ));
        if expected != self.binding_id {
            return Err(BackendEffectAuthorityError::BindingIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> BackendEffectBindingId { self.binding_id }
    pub fn plan_id(&self) -> ExternalEffectPlanId { self.plan_id }
    pub fn effect_authorization_id(&self) -> QualifiedExternalEffectAuthorizationId { self.effect_authorization_id }
    pub fn authority_profile_id(&self) -> TransitionAuthorityProfileId { self.authority_profile_id }
    pub fn authority_root_epoch(&self) -> u64 { self.authority_root_epoch }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn backend_implementation_digest(&self) -> [u8; 32] { self.backend_implementation_digest }
    pub fn backend_generation(&self) -> u64 { self.backend_generation }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendEffectAuthorityClaimV1 {
    schema_version: String,
    binding_id: BackendEffectBindingId,
    plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    claim_id: BackendEffectAuthorityClaimId,
}

impl BackendEffectAuthorityClaimV1 {
    pub fn new(
        binding: &BackendEffectBindingV1,
        profile: &TransitionAuthorityProfileV1,
    ) -> Result<Self, BackendEffectAuthorityError> {
        binding.validate()?;
        profile.validate()?;
        if profile.id() != binding.authority_profile_id()
            || profile.root_epoch() != binding.authority_root_epoch()
        {
            return Err(BackendEffectAuthorityError::AuthorityRootMismatch);
        }
        let claim_id = BackendEffectAuthorityClaimId(hash_claim(
            binding.id(), binding.plan_id(), binding.effect_authorization_id(),
            profile.id(), profile.root_epoch(), binding.backend_id(),
            binding.backend_implementation_digest(), binding.backend_generation(),
        ));
        Ok(Self {
            schema_version: BACKEND_EFFECT_AUTHORITY_CLAIM_SCHEMA_V1.to_owned(),
            binding_id: binding.id(),
            plan_id: binding.plan_id(),
            effect_authorization_id: binding.effect_authorization_id(),
            authority_profile_id: profile.id(),
            authority_root_epoch: profile.root_epoch(),
            backend_id: binding.backend_id(),
            backend_implementation_digest: binding.backend_implementation_digest(),
            backend_generation: binding.backend_generation(),
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), BackendEffectAuthorityError> {
        if self.schema_version != BACKEND_EFFECT_AUTHORITY_CLAIM_SCHEMA_V1 {
            return Err(BackendEffectAuthorityError::UnsupportedClaimSchema(
                self.schema_version.clone(),
            ));
        }
        if self.authority_root_epoch == 0 || self.backend_generation == 0 {
            return Err(BackendEffectAuthorityError::ZeroGeneration);
        }
        if self.backend_implementation_digest == [0; 32] {
            return Err(BackendEffectAuthorityError::ZeroDigest);
        }
        let expected = BackendEffectAuthorityClaimId(hash_claim(
            self.binding_id, self.plan_id, self.effect_authorization_id,
            self.authority_profile_id, self.authority_root_epoch, self.backend_id,
            self.backend_implementation_digest, self.backend_generation,
        ));
        if expected != self.claim_id {
            return Err(BackendEffectAuthorityError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> BackendEffectAuthorityClaimId { self.claim_id }
}

pub fn canonical_backend_effect_authority_claim_bytes(
    claim: &BackendEffectAuthorityClaimV1,
) -> Result<Vec<u8>, BackendEffectAuthorityError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(384);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.binding_id.as_bytes());
    out.extend_from_slice(claim.plan_id.as_bytes());
    out.extend_from_slice(claim.effect_authorization_id.as_bytes());
    out.extend_from_slice(claim.authority_profile_id.as_bytes());
    out.extend_from_slice(&claim.authority_root_epoch.to_le_bytes());
    out.extend_from_slice(claim.backend_id.as_bytes());
    out.extend_from_slice(&claim.backend_implementation_digest);
    out.extend_from_slice(&claim.backend_generation.to_le_bytes());
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_backend_effect_authority_claim_digest(
    claim: &BackendEffectAuthorityClaimV1,
) -> Result<[u8; 32], BackendEffectAuthorityError> {
    Ok(*blake3::hash(&canonical_backend_effect_authority_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedBackendEffectAuthorityV1 {
    claim: BackendEffectAuthorityClaimV1,
    profile: TransitionAuthorityProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedBackendEffectAuthorityId,
}

impl AuthenticatedBackendEffectAuthorityV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: BackendEffectAuthorityClaimV1,
        profile: TransitionAuthorityProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, BackendEffectAuthorityError> {
        claim.validate()?;
        profile.validate()?;
        if claim.authority_profile_id != profile.id()
            || claim.authority_root_epoch != profile.root_epoch()
        {
            return Err(BackendEffectAuthorityError::AuthorityRootMismatch);
        }
        if authentication_evidence_digest == [0; 32] {
            return Err(BackendEffectAuthorityError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedBackendEffectAuthorityId(domain_hash_parts(
            AUTH_DOMAIN,
            &[claim.id().as_bytes(), profile.id().as_bytes(),
                &profile.root_epoch().to_le_bytes(), &authentication_evidence_digest],
        ));
        Ok(Self { claim, profile, authentication_evidence_digest, evidence_id })
    }
}

#[derive(Debug, Clone)]
pub struct QualifiedBackendEffectAuthorizationV1 {
    authorization_id: QualifiedBackendEffectAuthorizationId,
    binding_id: BackendEffectBindingId,
    plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
}

impl QualifiedBackendEffectAuthorizationV1 {
    pub(crate) fn qualify(
        binding: &BackendEffectBindingV1,
        authenticated: &AuthenticatedBackendEffectAuthorityV1,
    ) -> Result<Self, BackendEffectAuthorityError> {
        binding.validate()?;
        authenticated.claim.validate()?;
        authenticated.profile.validate()?;
        let claim = &authenticated.claim;
        if claim.binding_id != binding.id()
            || claim.plan_id != binding.plan_id()
            || claim.effect_authorization_id != binding.effect_authorization_id()
            || claim.authority_profile_id != binding.authority_profile_id()
            || claim.authority_root_epoch != binding.authority_root_epoch()
            || claim.backend_id != binding.backend_id()
            || claim.backend_implementation_digest != binding.backend_implementation_digest()
            || claim.backend_generation != binding.backend_generation()
            || authenticated.profile.id() != binding.authority_profile_id()
            || authenticated.profile.root_epoch() != binding.authority_root_epoch()
        {
            return Err(BackendEffectAuthorityError::AuthorizationContextMismatch);
        }
        let authorization_id = QualifiedBackendEffectAuthorizationId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[binding.id().as_bytes(), claim.id().as_bytes(), authenticated.evidence_id.as_bytes(),
                binding.backend_id().as_bytes(), &binding.backend_implementation_digest(),
                &binding.backend_generation().to_le_bytes()],
        ));
        Ok(Self {
            authorization_id,
            binding_id: binding.id(),
            plan_id: binding.plan_id(),
            effect_authorization_id: binding.effect_authorization_id(),
            authority_profile_id: binding.authority_profile_id(),
            authority_root_epoch: binding.authority_root_epoch(),
            backend_id: binding.backend_id(),
            backend_implementation_digest: binding.backend_implementation_digest(),
            backend_generation: binding.backend_generation(),
        })
    }

    pub fn id(&self) -> QualifiedBackendEffectAuthorizationId { self.authorization_id }
    pub fn binding_id(&self) -> BackendEffectBindingId { self.binding_id }
    pub fn plan_id(&self) -> ExternalEffectPlanId { self.plan_id }
    pub fn effect_authorization_id(&self) -> QualifiedExternalEffectAuthorizationId { self.effect_authorization_id }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn backend_implementation_digest(&self) -> [u8; 32] { self.backend_implementation_digest }
    pub fn backend_generation(&self) -> u64 { self.backend_generation }
}

/// Non-Clone wrapper used by the only public effect-scoped physical preparation path.
#[derive(Debug)]
pub struct BackendBoundEffectScopedEligibilityV1 {
    scoped: EffectScopedKnownGoodBoundEligibilityV1,
    backend: ExecutionBackendProfileV1,
    binding: BackendEffectBindingV1,
    backend_authorization: QualifiedBackendEffectAuthorizationV1,
}

impl BackendBoundEffectScopedEligibilityV1 {
    pub fn bind(
        scoped: EffectScopedKnownGoodBoundEligibilityV1,
        backend: ExecutionBackendProfileV1,
        binding: BackendEffectBindingV1,
        backend_authorization: QualifiedBackendEffectAuthorizationV1,
    ) -> Result<Self, BackendEffectAuthorityError> {
        backend.validate()?;
        binding.validate()?;
        if binding.plan_id() != scoped.plan().id()
            || binding.effect_authorization_id() != scoped.authorization().id()
            || binding.subject_id() != scoped.plan().subject_id()
            || binding.target_realization_id() != scoped.plan().target_realization_id()
            || binding.coverage_manifest_digest() != scoped.plan().coverage_manifest_digest()
            || binding.backend_id() != backend.id()
            || binding.backend_implementation_digest() != backend.implementation_digest()
            || binding.backend_generation() != backend.backend_generation()
            || backend_authorization.binding_id() != binding.id()
            || backend_authorization.plan_id() != binding.plan_id()
            || backend_authorization.effect_authorization_id() != binding.effect_authorization_id()
            || backend_authorization.backend_id() != backend.id()
            || backend_authorization.backend_implementation_digest() != backend.implementation_digest()
            || backend_authorization.backend_generation() != backend.backend_generation()
        {
            return Err(BackendEffectAuthorityError::BackendBindingMismatch);
        }
        Ok(Self { scoped, backend, binding, backend_authorization })
    }

    pub fn backend(&self) -> &ExecutionBackendProfileV1 { &self.backend }
    pub fn binding(&self) -> &BackendEffectBindingV1 { &self.binding }
    pub fn backend_authorization(&self) -> &QualifiedBackendEffectAuthorizationV1 {
        &self.backend_authorization
    }
    pub fn scoped(&self) -> &EffectScopedKnownGoodBoundEligibilityV1 { &self.scoped }

    pub(crate) fn into_parts(self) -> (
        EffectScopedKnownGoodBoundEligibilityV1,
        ExecutionBackendProfileV1,
        BackendEffectBindingV1,
        QualifiedBackendEffectAuthorizationV1,
    ) {
        (self.scoped, self.backend, self.binding, self.backend_authorization)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BackendEffectAuthorityError {
    #[error(transparent)]
    Execution(#[from] ExecutionCapabilityError),
    #[error(transparent)]
    EffectAuthority(#[from] ExternalEffectAuthorityError),
    #[error(transparent)]
    TransitionAuthority(#[from] TransitionAuthorityError),
    #[error("unsupported backend-effect binding schema: {0}")]
    UnsupportedBindingSchema(String),
    #[error("unsupported backend-effect authority claim schema: {0}")]
    UnsupportedClaimSchema(String),
    #[error("backend-effect generation/root epoch must be non-zero")]
    ZeroGeneration,
    #[error("backend-effect digests must be non-zero")]
    ZeroDigest,
    #[error("effect plan and qualified effect authorization disagree")]
    EffectScopeMismatch,
    #[error("backend-effect binding identity mismatch")]
    BindingIdentityMismatch,
    #[error("backend-effect authorization must use the exact original transition authority root")]
    AuthorityRootMismatch,
    #[error("backend-effect authority claim identity mismatch")]
    ClaimIdentityMismatch,
    #[error("backend-effect authority authentication digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("authenticated backend-effect authorization differs from exact binding")]
    AuthorizationContextMismatch,
    #[error("backend-bound effect eligibility differs from exact backend/binding/authorization")]
    BackendBindingMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_binding(
    plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    subject_id: ContinuitySubjectId,
    target_id: TargetRealizationId,
    coverage_manifest_digest: [u8; 32],
    backend_id: ExecutionBackendId,
    implementation_digest: [u8; 32],
    backend_generation: u64,
) -> [u8; 32] {
    domain_hash_parts(BINDING_DOMAIN, &[
        plan_id.as_bytes(), effect_authorization_id.as_bytes(), authority_profile_id.as_bytes(),
        &authority_root_epoch.to_le_bytes(), subject_id.as_bytes(), target_id.as_bytes(),
        &coverage_manifest_digest, backend_id.as_bytes(), &implementation_digest,
        &backend_generation.to_le_bytes(),
    ])
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    binding_id: BackendEffectBindingId,
    plan_id: ExternalEffectPlanId,
    effect_authorization_id: QualifiedExternalEffectAuthorizationId,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    backend_id: ExecutionBackendId,
    implementation_digest: [u8; 32],
    backend_generation: u64,
) -> [u8; 32] {
    domain_hash_parts(CLAIM_DOMAIN, &[
        binding_id.as_bytes(), plan_id.as_bytes(), effect_authorization_id.as_bytes(),
        authority_profile_id.as_bytes(), &authority_root_epoch.to_le_bytes(), backend_id.as_bytes(),
        &implementation_digest, &backend_generation.to_le_bytes(),
    ])
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
    fn backend_effect_domains_are_distinct() {
        assert_ne!(BINDING_DOMAIN, CLAIM_DOMAIN);
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
        assert_ne!(WIRE_DOMAIN, AUTH_DOMAIN);
        assert_ne!(AUTH_DOMAIN, QUALIFIED_DOMAIN);
    }
}
