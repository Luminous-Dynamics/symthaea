// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Carry backend-relative effect-coverage qualification into the pre-mutation world.
//!
//! A verifier proof that the declared effect set is complete must exist before
//! mutation and must itself be committed into the exact protected post-intent world.

pub mod current_authorized;
pub mod current_transition_authority;
pub mod current_transition_authority_commitment;
pub mod current_verifier_commitment;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backend_bound_execution::{
    BackendBoundExecutionError, PendingBackendBoundEffectExecutionV1,
    QualifiedBackendEffectCommitmentId, QualifiedBackendEffectCommitmentV1,
    ReadyBackendBoundEffectExecutionV1, prepare_backend_bound_effect_execution,
};
use crate::backend_effect_authority::{
    BackendBoundEffectScopedEligibilityV1, BackendEffectBindingId,
};
use crate::effect_coverage::{
    EffectCoverageError, QualifiedExternalEffectCoverageId, QualifiedExternalEffectCoverageV1,
};
use crate::effect_scoped_execution::{
    EffectScopedExecutionEnvelopeId, QualifiedEffectScopeCommitmentId,
    QualifiedEffectScopeCommitmentV1,
};
use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptOutcomeV1, ExecutionAttemptReceiptV1,
    ExecutionBackendId, ExecutionEpochAnchorModeV1,
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

pub const EFFECT_COVERAGE_COMMITMENT_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-effect-coverage-commitment-claim-v1";
pub const EFFECT_COVERAGE_COMMITMENT_AUTH_PURPOSE: &str =
    "symthaea.continuity.effect-coverage-commitment.v1";

const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.effect-coverage-commitment-claim.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.continuity.effect-coverage-commitment-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-effect-coverage-commitment.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-effect-coverage-commitment.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }
    };
}

digest_id!(EffectCoverageCommitmentClaimId);
digest_id!(AuthenticatedEffectCoverageCommitmentId);
digest_id!(QualifiedEffectCoverageCommitmentId);

/// Non-Clone pre-execution value requiring one exact complete coverage proof for the
/// exact backend-bound effect eligibility.
#[derive(Debug)]
pub struct CoverageQualifiedBackendEffectEligibilityV1 {
    bound: BackendBoundEffectScopedEligibilityV1,
    coverage: QualifiedExternalEffectCoverageV1,
}

impl CoverageQualifiedBackendEffectEligibilityV1 {
    pub fn bind(
        bound: BackendBoundEffectScopedEligibilityV1,
        coverage: QualifiedExternalEffectCoverageV1,
    ) -> Result<Self, CoverageBoundExecutionError> {
        let binding = bound.binding();
        let backend = bound.backend();
        let plan = bound.scoped().plan();
        if coverage.plan_id() != plan.id()
            || coverage.backend_binding_id() != binding.id()
            || coverage.backend_id() != backend.id()
            || coverage.backend_implementation_digest() != backend.implementation_digest()
            || coverage.backend_generation() != backend.backend_generation()
            || coverage.coverage_manifest_digest() != plan.coverage_manifest_digest()
        {
            return Err(CoverageBoundExecutionError::CoverageBindingMismatch);
        }
        Ok(Self { bound, coverage })
    }

    pub fn coverage(&self) -> &QualifiedExternalEffectCoverageV1 { &self.coverage }
    pub fn bound(&self) -> &BackendBoundEffectScopedEligibilityV1 { &self.bound }
    pub(crate) fn into_parts(self) -> (
        BackendBoundEffectScopedEligibilityV1,
        QualifiedExternalEffectCoverageV1,
    ) { (self.bound, self.coverage) }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectCoverageCommitmentClaimV1 {
    schema_version: String,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    effect_scope_commitment_id: QualifiedEffectScopeCommitmentId,
    backend_commitment_id: QualifiedBackendEffectCommitmentId,
    envelope_id: EffectScopedExecutionEnvelopeId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    backend_binding_id: BackendEffectBindingId,
    backend_id: ExecutionBackendId,
    coverage_id: QualifiedExternalEffectCoverageId,
    coverage_manifest_digest: [u8; 32],
    analyzed_at_unix_ms: u64,
    transaction_challenge: [u8; 32],
    committed_at_unix_ms: u64,
    raw_commitment_evidence_digest: [u8; 32],
    claim_id: EffectCoverageCommitmentClaimId,
}

impl EffectCoverageCommitmentClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile: &ExecutionJournalAnchorProfileV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        effect_scope: &QualifiedEffectScopeCommitmentV1,
        backend_commitment: &QualifiedBackendEffectCommitmentV1,
        coverage: &QualifiedExternalEffectCoverageV1,
        raw_commitment_evidence_digest: [u8; 32],
    ) -> Result<Self, CoverageBoundExecutionError> {
        profile.validate()?;
        if raw_commitment_evidence_digest == [0; 32] || coverage.transaction_challenge() == [0; 32] {
            return Err(CoverageBoundExecutionError::ZeroDigest);
        }
        if profile.id() != journal_anchor.profile_id()
            || profile.root_epoch() != journal_anchor.root_epoch()
            || effect_scope.journal_anchor_id() != journal_anchor.id()
            || backend_commitment.journal_anchor_id() != journal_anchor.id()
            || backend_commitment.effect_scope_commitment_id() != effect_scope.id()
            || coverage.backend_binding_id() != backend_commitment.backend_binding_id()
            || coverage.backend_id() != backend_commitment.backend_id()
            || coverage.coverage_manifest_digest() != effect_scope.coverage_manifest_digest()
            || effect_scope.envelope_id() != backend_commitment.envelope_id()
            || effect_scope.attempt_id() != backend_commitment.attempt_id()
            || effect_scope.subject_id() != backend_commitment.subject_id()
        {
            return Err(CoverageBoundExecutionError::CommitmentContextMismatch);
        }
        if coverage.analyzed_at_unix_ms() > journal_anchor.anchored_at_unix_ms() {
            return Err(CoverageBoundExecutionError::CoverageAnalysisAfterCommitment);
        }
        let committed_at_unix_ms = journal_anchor.anchored_at_unix_ms();
        let claim_id = EffectCoverageCommitmentClaimId(hash_claim(
            profile.id(), profile.root_epoch(), journal_anchor.id(), journal_anchor.trusted_epoch_id(),
            effect_scope.id(), backend_commitment.id(), effect_scope.envelope_id(),
            effect_scope.attempt_id(), effect_scope.subject_id(), backend_commitment.backend_binding_id(),
            backend_commitment.backend_id(), coverage.id(), coverage.coverage_manifest_digest(),
            coverage.analyzed_at_unix_ms(), coverage.transaction_challenge(), committed_at_unix_ms,
            raw_commitment_evidence_digest,
        ));
        Ok(Self {
            schema_version: EFFECT_COVERAGE_COMMITMENT_CLAIM_SCHEMA_V1.to_owned(),
            profile_id: profile.id(), root_epoch: profile.root_epoch(),
            journal_anchor_id: journal_anchor.id(), trusted_epoch_id: journal_anchor.trusted_epoch_id(),
            effect_scope_commitment_id: effect_scope.id(), backend_commitment_id: backend_commitment.id(),
            envelope_id: effect_scope.envelope_id(), attempt_id: effect_scope.attempt_id(),
            subject_id: effect_scope.subject_id(), backend_binding_id: backend_commitment.backend_binding_id(),
            backend_id: backend_commitment.backend_id(), coverage_id: coverage.id(),
            coverage_manifest_digest: coverage.coverage_manifest_digest(),
            analyzed_at_unix_ms: coverage.analyzed_at_unix_ms(),
            transaction_challenge: coverage.transaction_challenge(), committed_at_unix_ms,
            raw_commitment_evidence_digest, claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), CoverageBoundExecutionError> {
        if self.schema_version != EFFECT_COVERAGE_COMMITMENT_CLAIM_SCHEMA_V1 {
            return Err(CoverageBoundExecutionError::UnsupportedCommitmentSchema(self.schema_version.clone()));
        }
        if self.root_epoch == 0 || self.analyzed_at_unix_ms == 0 || self.committed_at_unix_ms == 0 {
            return Err(CoverageBoundExecutionError::ZeroTimeOrGeneration);
        }
        if self.coverage_manifest_digest == [0; 32]
            || self.transaction_challenge == [0; 32]
            || self.raw_commitment_evidence_digest == [0; 32]
        {
            return Err(CoverageBoundExecutionError::ZeroDigest);
        }
        if self.analyzed_at_unix_ms > self.committed_at_unix_ms {
            return Err(CoverageBoundExecutionError::CoverageAnalysisAfterCommitment);
        }
        let expected = EffectCoverageCommitmentClaimId(hash_claim(
            self.profile_id, self.root_epoch, self.journal_anchor_id, self.trusted_epoch_id,
            self.effect_scope_commitment_id, self.backend_commitment_id, self.envelope_id,
            self.attempt_id, self.subject_id, self.backend_binding_id, self.backend_id,
            self.coverage_id, self.coverage_manifest_digest, self.analyzed_at_unix_ms,
            self.transaction_challenge, self.committed_at_unix_ms,
            self.raw_commitment_evidence_digest,
        ));
        if expected != self.claim_id { return Err(CoverageBoundExecutionError::CommitmentIdentityMismatch); }
        Ok(())
    }

    pub fn id(&self) -> EffectCoverageCommitmentClaimId { self.claim_id }
}

pub fn canonical_effect_coverage_commitment_claim_bytes(
    claim: &EffectCoverageCommitmentClaimV1,
) -> Result<Vec<u8>, CoverageBoundExecutionError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(768);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.profile_id.as_bytes());
    out.extend_from_slice(&claim.root_epoch.to_le_bytes());
    out.extend_from_slice(claim.journal_anchor_id.as_bytes());
    out.extend_from_slice(claim.trusted_epoch_id.as_bytes());
    out.extend_from_slice(claim.effect_scope_commitment_id.as_bytes());
    out.extend_from_slice(claim.backend_commitment_id.as_bytes());
    out.extend_from_slice(claim.envelope_id.as_bytes());
    out.extend_from_slice(claim.attempt_id.as_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.backend_binding_id.as_bytes());
    out.extend_from_slice(claim.backend_id.as_bytes());
    out.extend_from_slice(claim.coverage_id.as_bytes());
    out.extend_from_slice(&claim.coverage_manifest_digest);
    out.extend_from_slice(&claim.analyzed_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.transaction_challenge);
    out.extend_from_slice(&claim.committed_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_commitment_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_effect_coverage_commitment_claim_digest(
    claim: &EffectCoverageCommitmentClaimV1,
) -> Result<[u8; 32], CoverageBoundExecutionError> {
    Ok(*blake3::hash(&canonical_effect_coverage_commitment_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedEffectCoverageCommitmentV1 {
    claim: EffectCoverageCommitmentClaimV1,
    profile: ExecutionJournalAnchorProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedEffectCoverageCommitmentId,
}

impl AuthenticatedEffectCoverageCommitmentV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: EffectCoverageCommitmentClaimV1,
        profile: ExecutionJournalAnchorProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, CoverageBoundExecutionError> {
        claim.validate()?;
        profile.validate()?;
        if claim.profile_id != profile.id() || claim.root_epoch != profile.root_epoch() {
            return Err(CoverageBoundExecutionError::CommitmentRootMismatch);
        }
        if authentication_evidence_digest == [0; 32] {
            return Err(CoverageBoundExecutionError::ZeroDigest);
        }
        let evidence_id = AuthenticatedEffectCoverageCommitmentId(domain_hash_parts(
            AUTH_DOMAIN,
            &[claim.id().as_bytes(), profile.id().as_bytes(), &profile.root_epoch().to_le_bytes(),
                &authentication_evidence_digest],
        ));
        Ok(Self { claim, profile, authentication_evidence_digest, evidence_id })
    }
}

#[derive(Debug, Clone)]
pub struct QualifiedEffectCoverageCommitmentV1 {
    commitment_id: QualifiedEffectCoverageCommitmentId,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    effect_scope_commitment_id: QualifiedEffectScopeCommitmentId,
    backend_commitment_id: QualifiedBackendEffectCommitmentId,
    envelope_id: EffectScopedExecutionEnvelopeId,
    attempt_id: ExecutionAttemptId,
    coverage_id: QualifiedExternalEffectCoverageId,
    backend_binding_id: BackendEffectBindingId,
    backend_id: ExecutionBackendId,
}

impl QualifiedEffectCoverageCommitmentV1 {
    pub(crate) fn qualify(
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        effect_scope: &QualifiedEffectScopeCommitmentV1,
        backend_commitment: &QualifiedBackendEffectCommitmentV1,
        coverage: &QualifiedExternalEffectCoverageV1,
        authenticated: &AuthenticatedEffectCoverageCommitmentV1,
    ) -> Result<Self, CoverageBoundExecutionError> {
        authenticated.claim.validate()?;
        authenticated.profile.validate()?;
        let claim = &authenticated.claim;
        if authenticated.profile.id() != journal_anchor.profile_id()
            || authenticated.profile.root_epoch() != journal_anchor.root_epoch()
            || claim.journal_anchor_id != journal_anchor.id()
            || claim.trusted_epoch_id != journal_anchor.trusted_epoch_id()
            || claim.effect_scope_commitment_id != effect_scope.id()
            || claim.backend_commitment_id != backend_commitment.id()
            || claim.envelope_id != effect_scope.envelope_id()
            || claim.attempt_id != effect_scope.attempt_id()
            || claim.subject_id != effect_scope.subject_id()
            || claim.backend_binding_id != backend_commitment.backend_binding_id()
            || claim.backend_id != backend_commitment.backend_id()
            || claim.coverage_id != coverage.id()
            || claim.coverage_manifest_digest != coverage.coverage_manifest_digest()
            || claim.analyzed_at_unix_ms != coverage.analyzed_at_unix_ms()
            || claim.transaction_challenge != coverage.transaction_challenge()
            || claim.committed_at_unix_ms != journal_anchor.anchored_at_unix_ms()
        {
            return Err(CoverageBoundExecutionError::CommitmentContextMismatch);
        }
        let commitment_id = QualifiedEffectCoverageCommitmentId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[claim.id().as_bytes(), journal_anchor.id().as_bytes(), effect_scope.id().as_bytes(),
                backend_commitment.id().as_bytes(), coverage.id().as_bytes(),
                authenticated.evidence_id.as_bytes()],
        ));
        Ok(Self {
            commitment_id, journal_anchor_id: journal_anchor.id(),
            effect_scope_commitment_id: effect_scope.id(), backend_commitment_id: backend_commitment.id(),
            envelope_id: effect_scope.envelope_id(), attempt_id: effect_scope.attempt_id(),
            coverage_id: coverage.id(), backend_binding_id: backend_commitment.backend_binding_id(),
            backend_id: backend_commitment.backend_id(),
        })
    }

    pub fn id(&self) -> QualifiedEffectCoverageCommitmentId { self.commitment_id }
    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId { self.journal_anchor_id }
    pub fn effect_scope_commitment_id(&self) -> QualifiedEffectScopeCommitmentId { self.effect_scope_commitment_id }
    pub fn backend_commitment_id(&self) -> QualifiedBackendEffectCommitmentId { self.backend_commitment_id }
    pub fn envelope_id(&self) -> EffectScopedExecutionEnvelopeId { self.envelope_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn coverage_id(&self) -> QualifiedExternalEffectCoverageId { self.coverage_id }
}

#[derive(Debug)]
pub struct PendingCoverageQualifiedEffectExecutionV1 {
    inner: PendingBackendBoundEffectExecutionV1,
    coverage: QualifiedExternalEffectCoverageV1,
}

impl PendingCoverageQualifiedEffectExecutionV1 {
    pub fn inner(&self) -> &PendingBackendBoundEffectExecutionV1 { &self.inner }
    pub fn coverage(&self) -> &QualifiedExternalEffectCoverageV1 { &self.coverage }

    pub fn release_after_durable_coverage(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        effect_scope: &QualifiedEffectScopeCommitmentV1,
        backend_commitment: &QualifiedBackendEffectCommitmentV1,
        coverage_commitment: &QualifiedEffectCoverageCommitmentV1,
    ) -> Result<ReadyCoverageQualifiedEffectExecutionV1, CoverageBoundExecutionError> {
        if coverage_commitment.journal_anchor_id() != journal_anchor.id()
            || coverage_commitment.effect_scope_commitment_id() != effect_scope.id()
            || coverage_commitment.backend_commitment_id() != backend_commitment.id()
            || coverage_commitment.coverage_id() != self.coverage.id()
            || coverage_commitment.attempt_id() != self.inner.effect_scope().attempt_id()
        {
            return Err(CoverageBoundExecutionError::ReleaseContextMismatch);
        }
        let ready = self.inner.release_after_durable_backend_scope(
            journal, journal_anchor, effect_scope, backend_commitment,
        )?;
        if ready.backend_id() != self.coverage.backend_id()
            || ready.backend_binding().id() != self.coverage.backend_binding_id()
            || ready.effect_contract().coverage_manifest_digest()
                != self.coverage.coverage_manifest_digest()
        {
            return Err(CoverageBoundExecutionError::ReadyCoverageMismatch);
        }
        Ok(ReadyCoverageQualifiedEffectExecutionV1 {
            inner: ready,
            coverage: self.coverage,
            coverage_commitment_id: coverage_commitment.id(),
        })
    }
}

#[derive(Debug)]
pub struct ReadyCoverageQualifiedEffectExecutionV1 {
    inner: ReadyBackendBoundEffectExecutionV1,
    coverage: QualifiedExternalEffectCoverageV1,
    coverage_commitment_id: QualifiedEffectCoverageCommitmentId,
}

impl ReadyCoverageQualifiedEffectExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.inner.attempt_id() }
    pub fn backend_id(&self) -> ExecutionBackendId { self.inner.backend_id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.inner.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.inner.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.inner.target_realization_id() }
    pub fn effect_contract(&self) -> &ExternalEffectContractV1 { self.inner.effect_contract() }
    pub fn coverage(&self) -> &QualifiedExternalEffectCoverageV1 { &self.coverage }
    pub fn coverage_commitment_id(&self) -> QualifiedEffectCoverageCommitmentId { self.coverage_commitment_id }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, CoverageBoundExecutionError> {
        Ok(self.inner.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_coverage_qualified_effect_execution(
    bound: CoverageQualifiedBackendEffectEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingCoverageQualifiedEffectExecutionV1, CoverageBoundExecutionError> {
    let (backend_bound, coverage) = bound.into_parts();
    let inner = prepare_backend_bound_effect_execution(
        backend_bound, current_epoch, previous_epoch, epoch_anchor_mode,
        predecessor_journal_anchor, session_generation, session_nonce,
    )?;
    if inner.binding().id() != coverage.backend_binding_id()
        || inner.backend().id() != coverage.backend_id()
        || inner.binding().coverage_manifest_digest() != coverage.coverage_manifest_digest()
    {
        return Err(CoverageBoundExecutionError::CoverageBindingMismatch);
    }
    Ok(PendingCoverageQualifiedEffectExecutionV1 { inner, coverage })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CoverageBoundExecutionError {
    #[error(transparent)] Coverage(#[from] EffectCoverageError),
    #[error(transparent)] Backend(#[from] BackendBoundExecutionError),
    #[error(transparent)] JournalAnchor(#[from] ExecutionJournalAnchorError),
    #[error("complete effect-coverage proof does not bind exact backend effect eligibility")]
    CoverageBindingMismatch,
    #[error("unsupported effect-coverage commitment schema: {0}")]
    UnsupportedCommitmentSchema(String),
    #[error("effect-coverage commitment digest must be non-zero")]
    ZeroDigest,
    #[error("effect-coverage commitment time/root epoch must be non-zero")]
    ZeroTimeOrGeneration,
    #[error("effect-coverage analysis occurred after the protected pre-mutation commitment")]
    CoverageAnalysisAfterCommitment,
    #[error("effect-coverage commitment changed rollback-resistant root")]
    CommitmentRootMismatch,
    #[error("effect-coverage commitment context mismatch")]
    CommitmentContextMismatch,
    #[error("effect-coverage commitment identity mismatch")]
    CommitmentIdentityMismatch,
    #[error("physical release supplied a different protected coverage world")]
    ReleaseContextMismatch,
    #[error("ready backend/effect world does not match complete coverage proof")]
    ReadyCoverageMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_claim(
    profile_id: ExecutionJournalAnchorProfileId, root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    effect_scope_commitment_id: QualifiedEffectScopeCommitmentId,
    backend_commitment_id: QualifiedBackendEffectCommitmentId,
    envelope_id: EffectScopedExecutionEnvelopeId, attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId, backend_binding_id: BackendEffectBindingId,
    backend_id: ExecutionBackendId, coverage_id: QualifiedExternalEffectCoverageId,
    manifest_digest: [u8; 32], analyzed_at: u64, challenge: [u8; 32],
    committed_at: u64, raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CLAIM_DOMAIN);
    hasher.update(profile_id.as_bytes()); hasher.update(&root_epoch.to_le_bytes());
    hasher.update(journal_anchor_id.as_bytes()); hasher.update(trusted_epoch_id.as_bytes());
    hasher.update(effect_scope_commitment_id.as_bytes()); hasher.update(backend_commitment_id.as_bytes());
    hasher.update(envelope_id.as_bytes()); hasher.update(attempt_id.as_bytes());
    hasher.update(subject_id.as_bytes()); hasher.update(backend_binding_id.as_bytes());
    hasher.update(backend_id.as_bytes()); hasher.update(coverage_id.as_bytes());
    hasher.update(&manifest_digest); hasher.update(&analyzed_at.to_le_bytes());
    hasher.update(&challenge); hasher.update(&committed_at.to_le_bytes());
    hasher.update(&raw_evidence_digest);
    *hasher.finalize().as_bytes()
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new(); hasher.update(domain);
    for part in parts { hasher.update(&((*part).len() as u64).to_le_bytes()); hasher.update(part); }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coverage_commitment_domains_are_distinct() {
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
        assert_ne!(WIRE_DOMAIN, AUTH_DOMAIN);
        assert_ne!(AUTH_DOMAIN, QUALIFIED_DOMAIN);
    }
}
