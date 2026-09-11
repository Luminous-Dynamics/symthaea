// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-safe physical execution for transitions independently proven to have no
//! external effects inside one exact backend-relative external boundary.
//!
//! An empty obligation vector is never an input to this path. The owner-authorized
//! no-effects declaration and verifier-qualified canonical empty effect set must both
//! pre-exist capability minting, and that exact proof world must be protected under
//! the same rollback-resistant root as the exact post-intent journal anchor before a
//! physical token can escape.
//!
//! `NoEffectsProof != DurableIntent != ProtectedNoEffectsWorld != PhysicalAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptOutcomeV1, ExecutionAttemptReceiptV1,
    ExecutionBackendId, ExecutionEpochAnchorModeV1,
};
use crate::execution_coordinator::{
    KnownGoodExecutionCoordinatorError, PendingAnchoredKnownGoodExecutionV1,
    ReadyKnownGoodExecutionAttemptV1, prepare_known_good_execution,
};
use crate::execution_journal::ReconstructedExecutionJournalV1;
use crate::execution_journal_anchor::{
    ExecutionJournalAnchorError, ExecutionJournalAnchorProfileId,
    ExecutionJournalAnchorProfileV1, QualifiedExecutionJournalAnchorId,
    QualifiedExecutionJournalAnchorV1,
};
use crate::no_external_effects::{
    NoEffectsKnownGoodBoundEligibilityV1, NoExternalEffectsDeclarationId,
    NoExternalEffectsDeclarationV1, NoExternalEffectsError,
    QualifiedNoExternalEffectsAuthorizationId, QualifiedNoExternalEffectsAuthorizationV1,
    QualifiedNoExternalEffectsCoverageId, QualifiedNoExternalEffectsCoverageV1,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::KnownGoodExecutionIntentId;
use crate::trusted_commit_epoch::{QualifiedTrustedCommitEpochId, QualifiedTrustedCommitEpochV1};
use crate::witness::TargetRealizationId;

pub const NO_EFFECTS_EXECUTION_ENVELOPE_SCHEMA_V1: &str =
    "symthaea-continuity-no-effects-execution-envelope-v1";
pub const NO_EFFECTS_EXECUTION_COMMITMENT_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-no-effects-execution-commitment-claim-v1";
pub const NO_EFFECTS_EXECUTION_COMMITMENT_AUTH_PURPOSE: &str =
    "symthaea.continuity.no-effects-execution-commitment.v1";

const ENVELOPE_DOMAIN: &[u8] = b"symthaea.continuity.no-effects-execution-envelope.v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea.continuity.no-effects-execution-commitment-claim.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.continuity.no-effects-execution-commitment-wire.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.continuity.authenticated-no-effects-execution-commitment.v1\0";
const QUALIFIED_DOMAIN: &[u8] = b"symthaea.continuity.qualified-no-effects-execution-commitment.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }
    };
}

digest_id!(NoEffectsExecutionEnvelopeId);
digest_id!(NoEffectsExecutionCommitmentClaimId);
digest_id!(AuthenticatedNoEffectsExecutionCommitmentId);
digest_id!(QualifiedNoEffectsExecutionCommitmentId);

/// Persistent semantic description of the exact no-effects execution world.
///
/// This is audit/recovery material only. It is not an execution capability and a
/// self-consistent serialized envelope cannot recreate any qualified parent proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoEffectsExecutionEnvelopeV1 {
    schema_version: String,
    known_good_intent_id: KnownGoodExecutionIntentId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    declaration_id: NoExternalEffectsDeclarationId,
    authorization_id: QualifiedNoExternalEffectsAuthorizationId,
    coverage_id: QualifiedNoExternalEffectsCoverageId,
    coverage_manifest_digest: [u8; 32],
    coverage_analyzed_at_unix_ms: u64,
    coverage_challenge: [u8; 32],
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    envelope_id: NoEffectsExecutionEnvelopeId,
}

impl NoEffectsExecutionEnvelopeV1 {
    fn new(
        pending: &PendingAnchoredKnownGoodExecutionV1,
        declaration: &NoExternalEffectsDeclarationV1,
        authorization: &QualifiedNoExternalEffectsAuthorizationV1,
        coverage: &QualifiedNoExternalEffectsCoverageV1,
        backend_id: ExecutionBackendId,
        backend_implementation_digest: [u8; 32],
        backend_generation: u64,
    ) -> Result<Self, NoEffectsExecutionError> {
        pending.intent().validate()?;
        declaration.validate()?;
        let lineage = pending.intent().lineage();
        let execution_intent = pending.intent().execution_intent();
        if declaration.subject_id() != lineage.subject_id()
            || declaration.target_realization_id() != lineage.target_realization_id()
            || declaration.distributed_context_id() != lineage.distributed_context_id()
            || declaration.commit_time_unix_ms() != lineage.commit_time_unix_ms()
            || authorization.declaration_id() != declaration.id()
            || coverage.declaration_id() != declaration.id()
            || coverage.analyzed_at_unix_ms() != declaration.commit_time_unix_ms()
            || execution_intent.backend_id() != backend_id
            || declaration.backend_id() != backend_id
            || declaration.backend_implementation_digest() != backend_implementation_digest
            || declaration.backend_generation() != backend_generation
        {
            return Err(NoEffectsExecutionError::EnvelopeContextMismatch);
        }
        if backend_implementation_digest == [0; 32]
            || coverage.transaction_challenge() == [0; 32]
            || declaration.coverage_manifest_digest() == [0; 32]
            || backend_generation == 0
        {
            return Err(NoEffectsExecutionError::ZeroDigestOrGeneration);
        }
        let envelope_id = NoEffectsExecutionEnvelopeId(hash_envelope(
            pending.intent().id(), pending.attempt_id(), lineage.subject_id(),
            lineage.source_realization_id(), lineage.target_realization_id(),
            declaration.id(), authorization.id(), coverage.id(),
            declaration.coverage_manifest_digest(), coverage.analyzed_at_unix_ms(),
            coverage.transaction_challenge(), backend_id, backend_implementation_digest,
            backend_generation,
        ));
        Ok(Self {
            schema_version: NO_EFFECTS_EXECUTION_ENVELOPE_SCHEMA_V1.to_owned(),
            known_good_intent_id: pending.intent().id(), attempt_id: pending.attempt_id(),
            subject_id: lineage.subject_id(), source_realization_id: lineage.source_realization_id(),
            target_realization_id: lineage.target_realization_id(), declaration_id: declaration.id(),
            authorization_id: authorization.id(), coverage_id: coverage.id(),
            coverage_manifest_digest: declaration.coverage_manifest_digest(),
            coverage_analyzed_at_unix_ms: coverage.analyzed_at_unix_ms(),
            coverage_challenge: coverage.transaction_challenge(), backend_id,
            backend_implementation_digest, backend_generation, envelope_id,
        })
    }

    pub fn validate(&self) -> Result<(), NoEffectsExecutionError> {
        if self.schema_version != NO_EFFECTS_EXECUTION_ENVELOPE_SCHEMA_V1 {
            return Err(NoEffectsExecutionError::UnsupportedEnvelopeSchema(self.schema_version.clone()));
        }
        if self.source_realization_id == self.target_realization_id {
            return Err(NoEffectsExecutionError::SourceEqualsTarget);
        }
        if self.coverage_manifest_digest == [0; 32]
            || self.coverage_challenge == [0; 32]
            || self.backend_implementation_digest == [0; 32]
            || self.coverage_analyzed_at_unix_ms == 0
            || self.backend_generation == 0
        {
            return Err(NoEffectsExecutionError::ZeroDigestOrGeneration);
        }
        let expected = NoEffectsExecutionEnvelopeId(hash_envelope(
            self.known_good_intent_id, self.attempt_id, self.subject_id,
            self.source_realization_id, self.target_realization_id, self.declaration_id,
            self.authorization_id, self.coverage_id, self.coverage_manifest_digest,
            self.coverage_analyzed_at_unix_ms, self.coverage_challenge, self.backend_id,
            self.backend_implementation_digest, self.backend_generation,
        ));
        if expected != self.envelope_id {
            return Err(NoEffectsExecutionError::EnvelopeIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> NoEffectsExecutionEnvelopeId { self.envelope_id }
    pub fn known_good_intent_id(&self) -> KnownGoodExecutionIntentId { self.known_good_intent_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.source_realization_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn declaration_id(&self) -> NoExternalEffectsDeclarationId { self.declaration_id }
    pub fn authorization_id(&self) -> QualifiedNoExternalEffectsAuthorizationId { self.authorization_id }
    pub fn coverage_id(&self) -> QualifiedNoExternalEffectsCoverageId { self.coverage_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn coverage_analyzed_at_unix_ms(&self) -> u64 { self.coverage_analyzed_at_unix_ms }
    pub fn coverage_challenge(&self) -> [u8; 32] { self.coverage_challenge }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
}

/// Protected platform claim over the exact no-effects world and exact already-
/// qualified post-intent journal anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoEffectsExecutionCommitmentClaimV1 {
    schema_version: String,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    envelope_id: NoEffectsExecutionEnvelopeId,
    known_good_intent_id: KnownGoodExecutionIntentId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    declaration_id: NoExternalEffectsDeclarationId,
    authorization_id: QualifiedNoExternalEffectsAuthorizationId,
    coverage_id: QualifiedNoExternalEffectsCoverageId,
    coverage_manifest_digest: [u8; 32],
    coverage_challenge: [u8; 32],
    backend_id: ExecutionBackendId,
    backend_implementation_digest: [u8; 32],
    backend_generation: u64,
    committed_at_unix_ms: u64,
    raw_commitment_evidence_digest: [u8; 32],
    claim_id: NoEffectsExecutionCommitmentClaimId,
}

impl NoEffectsExecutionCommitmentClaimV1 {
    pub fn new(
        profile: &ExecutionJournalAnchorProfileV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        envelope: &NoEffectsExecutionEnvelopeV1,
        raw_commitment_evidence_digest: [u8; 32],
    ) -> Result<Self, NoEffectsExecutionError> {
        profile.validate()?;
        envelope.validate()?;
        if raw_commitment_evidence_digest == [0; 32] {
            return Err(NoEffectsExecutionError::ZeroDigestOrGeneration);
        }
        if profile.id() != journal_anchor.profile_id()
            || profile.root_epoch() != journal_anchor.root_epoch()
            || envelope.subject_id() != journal_anchor.subject_id()
            || envelope.coverage_analyzed_at_unix_ms() != journal_anchor.anchored_at_unix_ms()
        {
            return Err(NoEffectsExecutionError::CommitmentAnchorMismatch);
        }
        let committed_at_unix_ms = journal_anchor.anchored_at_unix_ms();
        let claim_id = NoEffectsExecutionCommitmentClaimId(hash_commitment_claim(
            profile.id(), profile.root_epoch(), journal_anchor.id(), journal_anchor.trusted_epoch_id(),
            envelope.id(), envelope.known_good_intent_id(), envelope.attempt_id(), envelope.subject_id(),
            envelope.declaration_id(), envelope.authorization_id(), envelope.coverage_id(),
            envelope.coverage_manifest_digest(), envelope.coverage_challenge(), envelope.backend_id(),
            envelope.backend_implementation_digest, envelope.backend_generation,
            committed_at_unix_ms, raw_commitment_evidence_digest,
        ));
        Ok(Self {
            schema_version: NO_EFFECTS_EXECUTION_COMMITMENT_CLAIM_SCHEMA_V1.to_owned(),
            profile_id: profile.id(), root_epoch: profile.root_epoch(),
            journal_anchor_id: journal_anchor.id(), trusted_epoch_id: journal_anchor.trusted_epoch_id(),
            envelope_id: envelope.id(), known_good_intent_id: envelope.known_good_intent_id(),
            attempt_id: envelope.attempt_id(), subject_id: envelope.subject_id(),
            declaration_id: envelope.declaration_id(), authorization_id: envelope.authorization_id(),
            coverage_id: envelope.coverage_id(), coverage_manifest_digest: envelope.coverage_manifest_digest(),
            coverage_challenge: envelope.coverage_challenge(), backend_id: envelope.backend_id(),
            backend_implementation_digest: envelope.backend_implementation_digest,
            backend_generation: envelope.backend_generation, committed_at_unix_ms,
            raw_commitment_evidence_digest, claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), NoEffectsExecutionError> {
        if self.schema_version != NO_EFFECTS_EXECUTION_COMMITMENT_CLAIM_SCHEMA_V1 {
            return Err(NoEffectsExecutionError::UnsupportedCommitmentSchema(self.schema_version.clone()));
        }
        if self.root_epoch == 0 || self.backend_generation == 0 || self.committed_at_unix_ms == 0
            || self.coverage_manifest_digest == [0; 32] || self.coverage_challenge == [0; 32]
            || self.backend_implementation_digest == [0; 32]
            || self.raw_commitment_evidence_digest == [0; 32]
        {
            return Err(NoEffectsExecutionError::ZeroDigestOrGeneration);
        }
        let expected = NoEffectsExecutionCommitmentClaimId(hash_commitment_claim(
            self.profile_id, self.root_epoch, self.journal_anchor_id, self.trusted_epoch_id,
            self.envelope_id, self.known_good_intent_id, self.attempt_id, self.subject_id,
            self.declaration_id, self.authorization_id, self.coverage_id,
            self.coverage_manifest_digest, self.coverage_challenge, self.backend_id,
            self.backend_implementation_digest, self.backend_generation, self.committed_at_unix_ms,
            self.raw_commitment_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(NoEffectsExecutionError::CommitmentIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> NoEffectsExecutionCommitmentClaimId { self.claim_id }
}

pub fn canonical_no_effects_execution_commitment_claim_bytes(
    claim: &NoEffectsExecutionCommitmentClaimV1,
) -> Result<Vec<u8>, NoEffectsExecutionError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(704);
    out.extend_from_slice(WIRE_DOMAIN);
    out.extend_from_slice(claim.profile_id.as_bytes());
    out.extend_from_slice(&claim.root_epoch.to_le_bytes());
    out.extend_from_slice(claim.journal_anchor_id.as_bytes());
    out.extend_from_slice(claim.trusted_epoch_id.as_bytes());
    out.extend_from_slice(claim.envelope_id.as_bytes());
    out.extend_from_slice(claim.known_good_intent_id.as_bytes());
    out.extend_from_slice(claim.attempt_id.as_bytes());
    out.extend_from_slice(claim.subject_id.as_bytes());
    out.extend_from_slice(claim.declaration_id.as_bytes());
    out.extend_from_slice(claim.authorization_id.as_bytes());
    out.extend_from_slice(claim.coverage_id.as_bytes());
    out.extend_from_slice(&claim.coverage_manifest_digest);
    out.extend_from_slice(&claim.coverage_challenge);
    out.extend_from_slice(claim.backend_id.as_bytes());
    out.extend_from_slice(&claim.backend_implementation_digest);
    out.extend_from_slice(&claim.backend_generation.to_le_bytes());
    out.extend_from_slice(&claim.committed_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_commitment_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_no_effects_execution_commitment_claim_digest(
    claim: &NoEffectsExecutionCommitmentClaimV1,
) -> Result<[u8; 32], NoEffectsExecutionError> {
    Ok(*blake3::hash(&canonical_no_effects_execution_commitment_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedNoEffectsExecutionCommitmentV1 {
    claim: NoEffectsExecutionCommitmentClaimV1,
    profile: ExecutionJournalAnchorProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedNoEffectsExecutionCommitmentId,
}

impl AuthenticatedNoEffectsExecutionCommitmentV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: NoEffectsExecutionCommitmentClaimV1,
        profile: ExecutionJournalAnchorProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, NoEffectsExecutionError> {
        claim.validate()?;
        profile.validate()?;
        if claim.profile_id != profile.id() || claim.root_epoch != profile.root_epoch() {
            return Err(NoEffectsExecutionError::CommitmentRootMismatch);
        }
        if authentication_evidence_digest == [0; 32] {
            return Err(NoEffectsExecutionError::ZeroDigestOrGeneration);
        }
        let evidence_id = AuthenticatedNoEffectsExecutionCommitmentId(domain_hash_parts(
            AUTH_DOMAIN,
            &[claim.id().as_bytes(), profile.id().as_bytes(), &profile.root_epoch().to_le_bytes(),
                &authentication_evidence_digest],
        ));
        Ok(Self { claim, profile, authentication_evidence_digest, evidence_id })
    }
}

#[derive(Debug, Clone)]
pub struct QualifiedNoEffectsExecutionCommitmentV1 {
    commitment_id: QualifiedNoEffectsExecutionCommitmentId,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    trusted_epoch_id: QualifiedTrustedCommitEpochId,
    envelope_id: NoEffectsExecutionEnvelopeId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    declaration_id: NoExternalEffectsDeclarationId,
    authorization_id: QualifiedNoExternalEffectsAuthorizationId,
    coverage_id: QualifiedNoExternalEffectsCoverageId,
    backend_id: ExecutionBackendId,
}

impl QualifiedNoEffectsExecutionCommitmentV1 {
    pub(crate) fn qualify(
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        envelope: &NoEffectsExecutionEnvelopeV1,
        authenticated: &AuthenticatedNoEffectsExecutionCommitmentV1,
    ) -> Result<Self, NoEffectsExecutionError> {
        envelope.validate()?;
        authenticated.claim.validate()?;
        authenticated.profile.validate()?;
        let claim = &authenticated.claim;
        if authenticated.profile.id() != journal_anchor.profile_id()
            || authenticated.profile.root_epoch() != journal_anchor.root_epoch()
            || claim.journal_anchor_id != journal_anchor.id()
            || claim.trusted_epoch_id != journal_anchor.trusted_epoch_id()
            || claim.committed_at_unix_ms != journal_anchor.anchored_at_unix_ms()
            || claim.envelope_id != envelope.id()
            || claim.known_good_intent_id != envelope.known_good_intent_id()
            || claim.attempt_id != envelope.attempt_id()
            || claim.subject_id != envelope.subject_id()
            || claim.declaration_id != envelope.declaration_id()
            || claim.authorization_id != envelope.authorization_id()
            || claim.coverage_id != envelope.coverage_id()
            || claim.coverage_manifest_digest != envelope.coverage_manifest_digest()
            || claim.coverage_challenge != envelope.coverage_challenge()
            || claim.backend_id != envelope.backend_id()
            || claim.backend_implementation_digest != envelope.backend_implementation_digest
            || claim.backend_generation != envelope.backend_generation
        {
            return Err(NoEffectsExecutionError::CommitmentContextMismatch);
        }
        let commitment_id = QualifiedNoEffectsExecutionCommitmentId(domain_hash_parts(
            QUALIFIED_DOMAIN,
            &[claim.id().as_bytes(), journal_anchor.id().as_bytes(), envelope.id().as_bytes(),
                authenticated.evidence_id.as_bytes()],
        ));
        Ok(Self {
            commitment_id, journal_anchor_id: journal_anchor.id(),
            trusted_epoch_id: journal_anchor.trusted_epoch_id(), envelope_id: envelope.id(),
            attempt_id: envelope.attempt_id(), subject_id: envelope.subject_id(),
            declaration_id: envelope.declaration_id(), authorization_id: envelope.authorization_id(),
            coverage_id: envelope.coverage_id(), backend_id: envelope.backend_id(),
        })
    }

    pub fn id(&self) -> QualifiedNoEffectsExecutionCommitmentId { self.commitment_id }
    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId { self.journal_anchor_id }
    pub fn envelope_id(&self) -> NoEffectsExecutionEnvelopeId { self.envelope_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.attempt_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn declaration_id(&self) -> NoExternalEffectsDeclarationId { self.declaration_id }
    pub fn authorization_id(&self) -> QualifiedNoExternalEffectsAuthorizationId { self.authorization_id }
    pub fn coverage_id(&self) -> QualifiedNoExternalEffectsCoverageId { self.coverage_id }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
}

#[derive(Debug)]
pub struct PendingProvenNoEffectsExecutionV1 {
    pending: PendingAnchoredKnownGoodExecutionV1,
    declaration: NoExternalEffectsDeclarationV1,
    authorization: QualifiedNoExternalEffectsAuthorizationV1,
    coverage: QualifiedNoExternalEffectsCoverageV1,
    envelope: NoEffectsExecutionEnvelopeV1,
}

impl PendingProvenNoEffectsExecutionV1 {
    pub fn intent(&self) -> &crate::transition_lineage::KnownGoodBoundExecutionAttemptIntentV1 {
        self.pending.intent()
    }
    pub fn envelope(&self) -> &NoEffectsExecutionEnvelopeV1 { &self.envelope }

    pub fn release_after_protected_no_effects(
        self,
        journal: &ReconstructedExecutionJournalV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
        commitment: &QualifiedNoEffectsExecutionCommitmentV1,
    ) -> Result<ReadyProvenNoEffectsExecutionV1, NoEffectsExecutionError> {
        self.envelope.validate()?;
        self.declaration.validate()?;
        if commitment.journal_anchor_id() != journal_anchor.id()
            || commitment.envelope_id() != self.envelope.id()
            || commitment.attempt_id() != self.envelope.attempt_id()
            || commitment.subject_id() != self.envelope.subject_id()
            || commitment.declaration_id() != self.declaration.id()
            || commitment.authorization_id() != self.authorization.id()
            || commitment.coverage_id() != self.coverage.id()
            || commitment.backend_id() != self.envelope.backend_id()
        {
            return Err(NoEffectsExecutionError::ReleaseContextMismatch);
        }
        let ready = self.pending.release_after_durable_anchor(journal, journal_anchor)?;
        if ready.attempt_id() != self.envelope.attempt_id()
            || ready.known_good_intent_id() != self.envelope.known_good_intent_id()
            || ready.backend_id() != self.envelope.backend_id()
            || ready.subject_id() != self.envelope.subject_id()
            || ready.source_realization_id() != self.envelope.source_realization_id()
            || ready.target_realization_id() != self.envelope.target_realization_id()
        {
            return Err(NoEffectsExecutionError::ReadyAttemptMismatch);
        }
        Ok(ReadyProvenNoEffectsExecutionV1 {
            ready, envelope: self.envelope, commitment_id: commitment.id(),
        })
    }
}

#[derive(Debug)]
pub struct ReadyProvenNoEffectsExecutionV1 {
    ready: ReadyKnownGoodExecutionAttemptV1,
    envelope: NoEffectsExecutionEnvelopeV1,
    commitment_id: QualifiedNoEffectsExecutionCommitmentId,
}

impl ReadyProvenNoEffectsExecutionV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.ready.attempt_id() }
    pub fn backend_id(&self) -> ExecutionBackendId { self.ready.backend_id() }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.ready.subject_id() }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.ready.source_realization_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.ready.target_realization_id() }
    pub fn envelope(&self) -> &NoEffectsExecutionEnvelopeV1 { &self.envelope }
    pub fn commitment_id(&self) -> QualifiedNoEffectsExecutionCommitmentId { self.commitment_id }

    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, NoEffectsExecutionError> {
        Ok(self.ready.finish(outcome, backend_evidence_digest, result_digest)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_proven_no_effects_execution(
    bound: NoEffectsKnownGoodBoundEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingProvenNoEffectsExecutionV1, NoEffectsExecutionError> {
    let (known_good_bound, declaration, authorization, coverage, backend) = bound.into_parts();
    if coverage.analyzed_at_unix_ms() != declaration.commit_time_unix_ms() {
        return Err(NoEffectsExecutionError::CoverageNotAtCommitBoundary);
    }
    let backend_id = backend.id();
    let backend_implementation_digest = backend.implementation_digest();
    let backend_generation = backend.backend_generation();
    let pending = prepare_known_good_execution(
        known_good_bound, current_epoch, previous_epoch, epoch_anchor_mode,
        predecessor_journal_anchor, &backend, session_generation, session_nonce,
    )?;
    let envelope = NoEffectsExecutionEnvelopeV1::new(
        &pending, &declaration, &authorization, &coverage, backend_id,
        backend_implementation_digest, backend_generation,
    )?;
    Ok(PendingProvenNoEffectsExecutionV1 {
        pending, declaration, authorization, coverage, envelope,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NoEffectsExecutionError {
    #[error(transparent)] Coordinator(#[from] KnownGoodExecutionCoordinatorError),
    #[error(transparent)] NoEffects(#[from] NoExternalEffectsError),
    #[error(transparent)] JournalAnchor(#[from] ExecutionJournalAnchorError),
    #[error("unsupported no-effects execution envelope schema: {0}")]
    UnsupportedEnvelopeSchema(String),
    #[error("unsupported no-effects execution commitment schema: {0}")]
    UnsupportedCommitmentSchema(String),
    #[error("no-effects execution contains a zero digest, generation, or time")]
    ZeroDigestOrGeneration,
    #[error("no-effects execution source equals target")]
    SourceEqualsTarget,
    #[error("no-effects execution envelope identity mismatch")]
    EnvelopeIdentityMismatch,
    #[error("no-effects eligibility / backend / coverage does not match exact A -> B intent")]
    EnvelopeContextMismatch,
    #[error("no-effects coverage was not established at the exact trusted commit boundary")]
    CoverageNotAtCommitBoundary,
    #[error("no-effects protected commitment root/anchor/time does not match exact journal world")]
    CommitmentAnchorMismatch,
    #[error("no-effects protected commitment identity mismatch")]
    CommitmentIdentityMismatch,
    #[error("no-effects protected commitment authentication root mismatch")]
    CommitmentRootMismatch,
    #[error("no-effects protected commitment does not bind exact envelope/journal world")]
    CommitmentContextMismatch,
    #[error("physical release supplied a different no-effects protected world")]
    ReleaseContextMismatch,
    #[error("ready physical attempt differs from exact protected no-effects world")]
    ReadyAttemptMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_envelope(
    intent_id: KnownGoodExecutionIntentId, attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId, source_id: TargetRealizationId,
    target_id: TargetRealizationId, declaration_id: NoExternalEffectsDeclarationId,
    authorization_id: QualifiedNoExternalEffectsAuthorizationId,
    coverage_id: QualifiedNoExternalEffectsCoverageId, manifest: [u8; 32],
    analyzed_at: u64, challenge: [u8; 32], backend_id: ExecutionBackendId,
    implementation: [u8; 32], backend_generation: u64,
) -> [u8; 32] {
    domain_hash_parts(ENVELOPE_DOMAIN, &[
        intent_id.as_bytes(), attempt_id.as_bytes(), subject_id.as_bytes(), source_id.as_bytes(),
        target_id.as_bytes(), declaration_id.as_bytes(), authorization_id.as_bytes(),
        coverage_id.as_bytes(), &manifest, &analyzed_at.to_le_bytes(), &challenge,
        backend_id.as_bytes(), &implementation, &backend_generation.to_le_bytes(),
    ])
}

#[allow(clippy::too_many_arguments)]
fn hash_commitment_claim(
    profile_id: ExecutionJournalAnchorProfileId, root_epoch: u64,
    anchor_id: QualifiedExecutionJournalAnchorId, epoch_id: QualifiedTrustedCommitEpochId,
    envelope_id: NoEffectsExecutionEnvelopeId, intent_id: KnownGoodExecutionIntentId,
    attempt_id: ExecutionAttemptId, subject_id: ContinuitySubjectId,
    declaration_id: NoExternalEffectsDeclarationId,
    authorization_id: QualifiedNoExternalEffectsAuthorizationId,
    coverage_id: QualifiedNoExternalEffectsCoverageId, manifest: [u8; 32],
    challenge: [u8; 32], backend_id: ExecutionBackendId, implementation: [u8; 32],
    backend_generation: u64, committed_at: u64, raw: [u8; 32],
) -> [u8; 32] {
    domain_hash_parts(CLAIM_DOMAIN, &[
        profile_id.as_bytes(), &root_epoch.to_le_bytes(), anchor_id.as_bytes(), epoch_id.as_bytes(),
        envelope_id.as_bytes(), intent_id.as_bytes(), attempt_id.as_bytes(), subject_id.as_bytes(),
        declaration_id.as_bytes(), authorization_id.as_bytes(), coverage_id.as_bytes(), &manifest,
        &challenge, backend_id.as_bytes(), &implementation, &backend_generation.to_le_bytes(),
        &committed_at.to_le_bytes(), &raw,
    ])
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
    fn commitment_wire_domain_is_distinct() {
        assert_ne!(CLAIM_DOMAIN, WIRE_DOMAIN);
    }

    #[test]
    fn no_effects_execution_domain_is_not_generic_attempt_domain() {
        assert_ne!(ENVELOPE_DOMAIN, b"symthaea.continuity.execution-attempt.v1\0");
    }
}
