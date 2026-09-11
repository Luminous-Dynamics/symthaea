// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Final backend-bound gate for effect-scoped physical execution.
//!
//! An authorized external-effect plan is not portable across executor implementations.
//! This module retains the exact backend binding through preparation and requires a
//! same-root protected backend commitment tied to the exact effect-scope commitment
//! and post-intent journal anchor before a physical token is released.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backend_effect_authority::{
    BackendBoundEffectScopedEligibilityV1, BackendEffectAuthorityError, BackendEffectBindingId,
    BackendEffectBindingV1, QualifiedBackendEffectAuthorizationId,
    QualifiedBackendEffectAuthorizationV1,
};
use crate::effect_scoped_execution::{
    EffectScopedExecutionEnvelopeId, EffectScopedExecutionError,
    PendingEffectScopedExecutionV1, QualifiedEffectScopeCommitmentId,
    QualifiedEffectScopeCommitmentV1, ReadyEffectScopedExecutionV1,
    prepare_effect_scoped_execution,
};
use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptOutcomeV1, ExecutionAttemptReceiptV1,
    ExecutionBackendId, ExecutionBackendProfileV1, ExecutionEpochAnchorModeV1,
};
use crate::execution_journal::ReconstructedExecutionJournalV1;
use crate::execution_journal_anchor::{
    ExecutionJournalAnchorError, ExecutionJournalAnchorProfileId,
    ExecutionJournalAnchorProfileV1, QualifiedExecutionJournalAnchorId,
    QualifiedExecutionJournalAnchorV1,
};
use crate::external_effects::ExternalEffectContractV1;
use crate::scope::ContinuitySubjectId;
use crate::trusted_commit_epoch::{QualifiedTrustedCommitEpochId, QualifiedTrustedCommitEpochV1};
use crate::witness::TargetRealizationId;

pub const BACKEND_EFFECT_COMMITMENT_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-backend-effect-commitment-claim-v1";
pub const BACKEND_EFFECT_COMMITMENT_AUTH_PURPOSE: &str =
    "symthaea.continuity.backend-effect-commitment.v1";

const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.backend-effect-commitment-claim.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.continuity.backend-effect-commitment-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-backend-effect-commitment.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-backend-effect-commitment.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }
    };
}

digest_id!(BackendEffectCommitmentClaimId);
digest_id!(AuthenticatedBackendEffectCommitmentId);
digest_id!(QualifiedBackendEffectCommitmentId);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendEffectCommitmentClaimV1 {
    schema_version: String,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    effect_scope_commitment_id: QualifiedEffectScopeCommitmentId,
    envelope_id: EffectScopedExecutionEnvelopeId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    backend_binding_id: BackendEffectBindingId,
    backend_authorization_id: QualifiedBackendEffectAuthorizationId,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    committed_at_unix_ms: u64,
    raw_commitment_evidence_digest: [u8; 32],
    claim_id: BackendEffectCommitmentClaimId,
}

impl BackendEffectCommitmentClaimV1 {
    pub fn new(
        profile: &ExecutionJournalAnchorProfileV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        effect_scope: &QualifiedEffectScopeCommitmentV1,
        binding: &BackendEffectBindingV1,
        backend_authorization: &QualifiedBackendEffectAuthorizationV1,
        raw_commitment_evidence_digest: [u8; 32],
    ) -> Result<Self, BackendBoundExecutionError> {
        profile.validate()?;
        binding.validate()?;
        if raw_commitment_evidence_digest == [0; 32] {
            return Err(BackendBoundExecutionError::ZeroCommitmentEvidenceDigest);
        }
        if profile.id() != journal_anchor.profile_id()
            || profile.root_epoch() != journal_anchor.root_epoch()
            || effect_scope.journal_anchor_id() != journal_anchor.id()
            || effect_scope.subject_id() != binding.subject_id()
            || effect_scope.effect_plan_id() != binding.plan_id()
            || effect_scope.effect_authorization_id() != binding.effect_authorization_id()
            || backend_authorization.binding_id() != binding.id()
            || backend_authorization.plan_id() != binding.plan_id()
            || backend_authorization.effect_authorization_id() != binding.effect_authorization_id()
            || backend_authorization.backend_id() != binding.backend_id()
            || backend_authorization.backend_implementation_digest()
                != binding.backend_implementation_digest()
            || backend_authorization.backend_generation() != binding.backend_generation()
        {
            return Err(BackendBoundExecutionError::CommitmentContextMismatch);
        }
        let committed_at_unix_ms = effect_scope.committed_at_unix_ms();
        if committed_at_unix_ms == 0
            || committed_at_unix_ms != journal_anchor.anchored_at_unix_ms()
        {
            return Err(BackendBoundExecutionError::CommitmentTimeMismatch);
        }
        let claim_id = BackendEffectCommitmentClaimId(hash_claim(
            profile.id(), profile.root_epoch(), journal_anchor.id(),
            journal_anchor.trusted_epoch_id(), effect_scope.id(), effect_scope.envelope_id(),
            effect_scope.attempt_id(), effect_scope.subject_id(), binding.id(),
            backend_authorization.id(), binding.backend_id(),
            binding.backend_implementation_digest(), binding.backend_generation(),
            committed_at_unix_ms, raw_commitment_evidence_digest,
        ));
        Ok(Self {
            schema_version: BACKEND_EFFECT_COMMITMENT_CLAIM_SCHEMA_V1.to_owned(),
            profile_id: profile.id(),
            root_epoch: profile.root_epoch(),
            journal_anchor_id: journal_anchor.id(),
            trusted_epoch_id: journal_anchor.trusted_epoch_id(),
            effect_scope_commitment_id: effect_scope.id(),
            envelope_id: effect_scope.envelope_id(),
            attempt_id: effect_scope.attempt_id(),
            subject_id: effect_scope.subject_id(),
            backend_binding_id: binding.id(),
            backend_authorization_id: backend_authorization.id(),
            backend_id: binding.backend_id(),
            backend_implementation_digest: binding.backend_implementation_digest(),
            backend_generation: binding.backend_generation(),
            committed_at_unix_ms,
            raw_commitment_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), BackendBoundExecutionError> {
        if self.schema_version != BACKEND_EFFECT_COMMITMENT_CLAIM_SCHEMA_V1 {
            return Err(BackendBoundExecutionError::UnsupportedCommitmentSchema(
                self.schema_version.clone(),
            ));
        }
        if self.root_epoch == 0 || self.backend_generation == 0 {
            return Err(BackendBoundExecutionError::ZeroGeneration);
        }
        if self.backend_implementation_digest == [0; 32]
            || self.raw_commitment_evidence_digest == [0; 32]
        {
            return Err(BackendBoundExecutionError::ZeroCommitmentEvidenceDigest);
        }
        if self.committed_at_unix_ms == 0 {
            return Err(BackendBoundExecutionError::CommitmentTimeMismatch);
        }
        let expected = BackendEffectCommitmentClaimId(hash_claim(
            self.profile_id, self.root_epoch, self.journal_anchor_id, self.trusted_epoch_id,
            self.effect_scope_commitment_id, self.envelope_id, self.attempt_id, self.subject_id,
            self.backend_binding_id, self.backend_authorization_id, self.backend_id,
            self.backend_implementation_digest, self.backend_generation,
            self.committed_at_unix_ms, self.raw_commitment_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(BackendBoundExecutionError::CommitmentIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> BackendEffectCommitmentClaimId { self.claim_id }
}

pub fn canonical_backend_effect_commitment_claim_bytes(
    claim: &BackendEffectCommitmentClaimV1,
) -> Result<Vec<u8>, BackendBoundExecutionError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(640);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.profile_id.as_bytes());
    out.extend_from_slice(&claim.root_epoch.to_le_bytes());
    out.extend_from_slice(claim.journal_anchor_id.as_bytes());
    out.extend_from_slice(claim.trusted_epoch_id.as_bytes());
    out.extend_from_slice(claim.effect_scope_commitment_id.as_bytes());
    out.extend_from_slice(claim.envelope_id.as_bytes());
    out.extend_from_slice(claim.attempt_id.as_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.backend_binding_id.as_bytes());
    out.extend_from_slice(claim.backend_authorization_id.as_bytes());
    out.extend_from_slice(claim.backend_id.as_bytes());
    out.extend_from_slice(&claim.backend_implementation_digest);
    out.extend_from_slice(&claim.backend_generation.to_le_bytes());
    out.extend_from_slice(&claim.committed_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_commitment_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_backend_effect_commitment_claim_digest(
    claim: &BackendEffectCommitmentClaimV1,
) -> Result<[u8; 32], BackendBoundExecutionError> {
    Ok(*blake3::hash(&canonical_backend_effect_commitment_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedBackendEffectCommitmentV1 {
    claim: BackendEffectCommitmentClaimV1,
    profile: ExecutionJournalAnchorProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedBackendEffectCommitmentId,
}

impl AuthenticatedBackendEffectCommitmentV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: BackendEffectCommitmentClaimV1,
        profile: ExecutionJournalAnchorProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, BackendBoundExecutionError> {
        claim.validate()?;
        profile.validate()?;
        if claim.profile_id != profile.id() || claim.root_epoch != profile.root_epoch() {
            return Err(BackendBoundExecutionError::CommitmentRootMismatch);
        }
        if authentication_evidence_digest == [0; 32] {
            return Err(BackendBoundExecutionError::ZeroAuthenticationEvidenceDigest);
        }
        let evidence_id = AuthenticatedBackendEffectCommitmentId(domain_hash_parts(
            AUTH_DOMAIN,
            &[claim.id().as_bytes(), profile.id().as_bytes(),
                &profile.root_epoch().to_le_bytes(), &authentication_evidence_digest],
        ));
        Ok(Self { claim, profile, authentication_evidence_digest, evidence_id })
    }
}

#[derive(Debug, Clone)]
pub struct QualifiedBackendEffectCommitmentV1 {
    commitment_id: QualifiedBackendEffectCommitmentId,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    effect_scope_commitment_id: QualifiedEffectScopeCommitmentId,
    envelope_id: EffectScopedExecutionEnvelopeId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    backend_binding_id: BackendEffectBindingId,
    backend_authorization_id: QualifiedBackendEffectAuthorizationId,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
}

impl QualifiedBackendEffectCommitmentV1 {
    pub(crate) fn qualify(
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        effect_scope: &QualifiedEffectScopeCommitmentV1,
        binding: &BackendEffectBindingV1,
        backend_authorization: &QualifiedBackendEffectAuthorizationV1,
        authenticated: &AuthenticatedBackendEffectCommitmentV1,
    ) -> Result<Self, BackendBoundExecutionError> {
        binding.validate()?;
        authenticated.claim.validate()?;
        authenticated.profile.validate()?;
        let claim = &authenticated.claim;
        if authenticated.profile.id() != journal_anchor.profile_id()
            || authenticated.profile.root_epoch() != journal_anchor.root_epoch()
            || claim.journal_anchor_id != journal_anchor.id()
            || claim.trusted_epoch_id != journal_anchor.trusted_epoch_id()
            || claim.effect_scope_commitment_id != effect_scope.id()
            || effect_scope.journal_anchor_id() != journal_anchor.id()
            || claim.envelope_id != effect_scope.envelope_id()
            || claim.attempt_id != effect_scope.attempt_id()
            || claim.subject_id != effect_scope.subject_id()
            || claim.backend_binding_id != binding.id()
            || claim.backend_authorization_id != backend_authorization.id()
            || claim.backend_id != binding.backend_id()
            || claim.backend_implementation_digest != binding.backend_implementation_digest()
            || claim.backend_generation != binding.backend_generation()
            || claim.committed_at_unix_ms != effect_scope.committed_at_unix_ms()
        {
            return Err(BackendBoundExecutionError::CommitmentContextMismatch);
        }
        let commitment_id = QualifiedBackendEffectCommitmentId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[claim.id().as_bytes(), journal_anchor.id().as_bytes(), effect_scope.id().as_bytes(),
                binding.id().as_bytes(), backend_authorization.id().as_bytes(),
                authenticated.evidence_id.as_bytes()],
        ));
        Ok(Self {
            commitment_id,
            journal_anchor_id: journal_anchor.id(),
            effect_scope_commitment_id: effect_scope.id(),
            envelope_id: effect_scope.envelope_id(),
            attempt_id: effect_scope.attempt_id(),
            subject_id: effect_scope.subject_id(),
            backend_binding_id: binding.id(),
            backend_authorization_id: backend_authorization.id(),
            backend_id: binding.backend_id(),
            backend_implementation_digest: binding.backend_implementation_digest(),
            backend_generation: binding.backend_generation(),
        })
    }

    pub fn id(&self) -> QualifiedBackendEffectCommitmentId { self.commitment_id }
    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId { self.journal_anchor_id }
    pub fn effect_scope_commitment_id(&self) -> QualifiedEffectScopeCommitmentId { self.effect_scope_commitment_id }
    pub fn envelope_id(&self) -> EffectScopedExecutionEnvelopeId { self.envelope_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn backend_binding_id(&self) -> BackendEffectBindingId { self.backend_binding_id }
    pub fn backend_authorization_id(&self) -> QualifiedBackendEffectAuthorizationId { self.backend_authorization_id }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
}

#[derive(Debug)]
pub struct PendingBackendBoundEffectExecutionV1 {
    inner: PendingEffectScopedExecutionV1,
    backend: ExecutionBackendProfileV1,
    binding: BackendEffectBindingV1,
    backend_authorization: QualifiedBackendEffectAuthorizationV1,
}

impl PendingBackendBoundEffectExecutionV1 {
    pub fn effect_scope(&self) -> &PendingEffectScopedExecutionV1 { &self.inner }
    pub fn backend(&self) -> &ExecutionBackendProfileV1 { &self.backend }
    pub fn binding(&self) -> &BackendEffectBindingV1 { &self.binding }
    pub fn backend_authorization(&self) -> &QualifiedBackendEffectAuthorizationV1 { &self.backend_authorization }

    pub fn release_after_durable_backend_scope(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        effect_scope: &QualifiedEffectScopeCommitmentV1,
        backend_commitment: &QualifiedBackendEffectCommitmentV1,
    ) -> Result<ReadyBackendBoundEffectExecutionV1, BackendBoundExecutionError> {
        if backend_commitment.journal_anchor_id() != journal_anchor.id()
            || backend_commitment.effect_scope_commitment_id() != effect_scope.id()
            || backend_commitment.envelope_id() != self.inner.envelope().id()
            || backend_commitment.attempt_id() != self.inner.intent().attempt_id()
            || backend_commitment.subject_id() != self.inner.intent().lineage().subject_id()
            || backend_commitment.backend_binding_id() != self.binding.id()
            || backend_commitment.backend_authorization_id() != self.backend_authorization.id()
            || backend_commitment.backend_id() != self.backend.id()
        {
            return Err(BackendBoundExecutionError::ReleaseContextMismatch);
        }
        let ready = self.inner.release_after_durable_scope(journal, journal_anchor, effect_scope)?;
        if ready.backend_id() != self.backend.id()
            || ready.attempt_id() != backend_commitment.attempt_id()
        {
            return Err(BackendBoundExecutionError::ReadyBackendMismatch);
        }
        Ok(ReadyBackendBoundEffectExecutionV1 {
            inner: ready,
            backend: self.backend,
            binding: self.binding,
            backend_authorization: self.backend_authorization,
            backend_commitment_id: backend_commitment.id(),
        })
    }
}

#[derive(Debug)]
pub struct ReadyBackendBoundEffectExecutionV1 {
    inner: ReadyEffectScopedExecutionV1,
    backend: ExecutionBackendProfileV1,
    binding: BackendEffectBindingV1,
    backend_authorization: QualifiedBackendEffectAuthorizationV1,
    backend_commitment_id: QualifiedBackendEffectCommitmentId,
}

impl ReadyBackendBoundEffectExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.inner.attempt_id() }
    pub fn backend(&self) -> &ExecutionBackendProfileV1 { &self.backend }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend.id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.inner.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.inner.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.inner.target_realization_id() }
    pub fn effect_contract(&self) -> &ExternalEffectContractV1 { self.inner.effect_contract() }
    pub fn backend_binding(&self) -> &BackendEffectBindingV1 { &self.binding }
    pub fn backend_authorization_id(&self) -> QualifiedBackendEffectAuthorizationId { self.backend_authorization.id() }
    pub fn backend_commitment_id(&self) -> QualifiedBackendEffectCommitmentId { self.backend_commitment_id }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, BackendBoundExecutionError> {
        Ok(self.inner.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_backend_bound_effect_execution(
    bound: BackendBoundEffectScopedEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingBackendBoundEffectExecutionV1, BackendBoundExecutionError> {
    let (scoped, backend, binding, backend_authorization) = bound.into_parts();
    let inner = prepare_effect_scoped_execution(
        scoped, current_epoch, previous_epoch, epoch_anchor_mode,
        predecessor_journal_anchor, &backend, session_generation, session_nonce,
    )?;
    if inner.authorization().id() != binding.effect_authorization_id()
        || inner.plan().id() != binding.plan_id()
        || inner.intent().lineage().subject_id() != binding.subject_id()
        || inner.intent().lineage().target_realization_id() != binding.target_realization_id()
    {
        return Err(BackendBoundExecutionError::PreparationContextMismatch);
    }
    Ok(PendingBackendBoundEffectExecutionV1 {
        inner, backend, binding, backend_authorization,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BackendBoundExecutionError {
    #[error(transparent)]
    BackendAuthority(#[from] BackendEffectAuthorityError),
    #[error(transparent)]
    EffectExecution(#[from] EffectScopedExecutionError),
    #[error(transparent)]
    JournalAnchor(#[from] ExecutionJournalAnchorError),
    #[error("unsupported backend-effect commitment schema: {0}")]
    UnsupportedCommitmentSchema(String),
    #[error("backend-effect commitment generation/root epoch must be non-zero")]
    ZeroGeneration,
    #[error("backend-effect commitment evidence/implementation digest must be non-zero")]
    ZeroCommitmentEvidenceDigest,
    #[error("backend-effect commitment time differs from exact protected effect scope")]
    CommitmentTimeMismatch,
    #[error("backend-effect commitment context does not match exact journal/effect/backend binding")]
    CommitmentContextMismatch,
    #[error("backend-effect commitment identity mismatch")]
    CommitmentIdentityMismatch,
    #[error("backend-effect commitment changed rollback-resistant root")]
    CommitmentRootMismatch,
    #[error("backend-effect commitment authentication digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("backend-bound preparation differs from exact authorized plan/backend")]
    PreparationContextMismatch,
    #[error("backend-bound physical release supplied a different protected backend world")]
    ReleaseContextMismatch,
    #[error("ready effect-scoped attempt uses a different backend")]
    ReadyBackendMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    effect_scope_commitment_id: QualifiedEffectScopeCommitmentId,
    envelope_id: EffectScopedExecutionEnvelopeId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    backend_binding_id: BackendEffectBindingId,
    backend_authorization_id: QualifiedBackendEffectAuthorizationId,
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    committed_at_unix_ms: u64,
    raw_commitment_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CLAIM_DOMAIN);
    hasher.update(profile_id.as_bytes());
    hasher.update(&root_epoch.to_le_bytes());
    hasher.update(journal_anchor_id.as_bytes());
    hasher.update(trusted_epoch_id.as_bytes());
    hasher.update(effect_scope_commitment_id.as_bytes());
    hasher.update(envelope_id.as_bytes());
    hasher.update(attempt_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(backend_binding_id.as_bytes());
    hasher.update(backend_authorization_id.as_bytes());
    hasher.update(backend_id.as_bytes());
    hasher.update(&backend_implementation_digest);
    hasher.update(&backend_generation.to_le_bytes());
    hasher.update(&committed_at_unix_ms.to_le_bytes());
    hasher.update(&raw_commitment_evidence_digest);
    *hasher.finalize().as_bytes()
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
    fn backend_commitment_domains_are_distinct() {
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
        assert_ne!(WIRE_DOMAIN, AUTH_DOMAIN);
        assert_ne!(AUTH_DOMAIN, QUALIFIED_DOMAIN);
    }
}
