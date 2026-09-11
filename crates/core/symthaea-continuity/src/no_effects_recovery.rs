// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scope-bounded crash recovery for transitions whose exact backend-relative external
//! effect surface was independently proven empty before physical mutation.
//!
//! This theorem is deliberately narrower than “nothing happened anywhere.” It means
//! exact source A is recovered and still current, while the exact pre-mutation
//! protected no-effects proof established an empty reachable effect set inside one
//! exact declared external boundary/model/taxonomy/backend world.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::crash_recovery_qualification::{
    CrashRecoveryQualificationError, QualifiedCrashRecoveryToActiveKnownGoodId,
    QualifiedCrashRecoveryToActiveKnownGoodV1,
};
use crate::current_crash_recovery::{
    CurrentCrashRecoveryError, QualifiedCurrentCrashRecoveryId,
    QualifiedCurrentCrashRecoveryV1,
};
use crate::execution_capability::{ExecutionAttemptId, ExecutionBackendId};
use crate::execution_journal_anchor::{
    QualifiedExecutionJournalAnchorId, QualifiedExecutionJournalAnchorV1,
};
use crate::no_effects_execution::{
    NoEffectsExecutionError, QualifiedNoEffectsExecutionCommitmentId,
    QualifiedNoEffectsExecutionCommitmentV1,
};
use crate::no_effects_rebind::{
    NoEffectsRebindError, ReboundNoEffectsExecutionId, ReboundNoEffectsExecutionV1,
};
use crate::no_external_effects::{
    NoExternalEffectsDeclarationId, QualifiedNoExternalEffectsAuthorizationId,
    QualifiedNoExternalEffectsCoverageId,
};
use crate::scope::ContinuitySubjectId;
use crate::witness::TargetRealizationId;

pub const NO_EFFECTS_CURRENT_RECOVERY_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-no-effects-current-recovery-record-v1";

const RECOVERY_DOMAIN: &[u8] = b"symthaea.continuity.no-effects-current-recovery.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedNoEffectsCurrentRecoveryId([u8; 32]);
impl QualifiedNoEffectsCurrentRecoveryId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Persistent audit record. It describes the exact proof composition but cannot
/// recreate any qualified parent after restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoEffectsCurrentRecoveryRecordV1 {
    schema_version: String,
    current_crash_recovery_id: QualifiedCurrentCrashRecoveryId,
    crash_recovery_id: QualifiedCrashRecoveryToActiveKnownGoodId,
    rebound_no_effects_id: ReboundNoEffectsExecutionId,
    protected_no_effects_commitment_id: QualifiedNoEffectsExecutionCommitmentId,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    original_attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    recovered_realization_id: TargetRealizationId,
    failed_or_abandoned_target_id: TargetRealizationId,
    declaration_id: NoExternalEffectsDeclarationId,
    authorization_id: QualifiedNoExternalEffectsAuthorizationId,
    coverage_id: QualifiedNoExternalEffectsCoverageId,
    backend_id: ExecutionBackendId,
    coverage_manifest_digest: [u8; 32],
    protected_at_unix_ms: u64,
    recovered_at_unix_ms: u64,
    currentness_at_unix_ms: u64,
    recovery_id: QualifiedNoEffectsCurrentRecoveryId,
}

impl NoEffectsCurrentRecoveryRecordV1 {
    pub fn validate(&self) -> Result<(), NoEffectsCurrentRecoveryError> {
        if self.schema_version != NO_EFFECTS_CURRENT_RECOVERY_RECORD_SCHEMA_V1 {
            return Err(NoEffectsCurrentRecoveryError::UnsupportedSchema(self.schema_version.clone()));
        }
        if self.coverage_manifest_digest == [0; 32] {
            return Err(NoEffectsCurrentRecoveryError::ZeroCoverageManifestDigest);
        }
        if self.protected_at_unix_ms == 0
            || self.recovered_at_unix_ms == 0
            || self.currentness_at_unix_ms == 0
        {
            return Err(NoEffectsCurrentRecoveryError::ZeroTime);
        }
        if self.recovered_realization_id == self.failed_or_abandoned_target_id {
            return Err(NoEffectsCurrentRecoveryError::SourceEqualsTarget);
        }
        if self.protected_at_unix_ms >= self.recovered_at_unix_ms {
            return Err(NoEffectsCurrentRecoveryError::ProtectedWorldNotBeforeRecovery);
        }
        if self.currentness_at_unix_ms < self.recovered_at_unix_ms {
            return Err(NoEffectsCurrentRecoveryError::CurrentnessPredatesRecovery);
        }
        let expected = QualifiedNoEffectsCurrentRecoveryId(hash_recovery(
            self.current_crash_recovery_id,
            self.crash_recovery_id,
            self.rebound_no_effects_id,
            self.protected_no_effects_commitment_id,
            self.journal_anchor_id,
            self.original_attempt_id,
            self.subject_id,
            self.recovered_realization_id,
            self.failed_or_abandoned_target_id,
            self.declaration_id,
            self.authorization_id,
            self.coverage_id,
            self.backend_id,
            self.coverage_manifest_digest,
            self.protected_at_unix_ms,
            self.recovered_at_unix_ms,
            self.currentness_at_unix_ms,
        ));
        if expected != self.recovery_id {
            return Err(NoEffectsCurrentRecoveryError::RecoveryIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> QualifiedNoEffectsCurrentRecoveryId { self.recovery_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn recovered_realization_id(&self) -> TargetRealizationId { self.recovered_realization_id }
    pub fn failed_or_abandoned_target_id(&self) -> TargetRealizationId { self.failed_or_abandoned_target_id }
    pub fn original_attempt_id(&self) -> ExecutionAttemptId { self.original_attempt_id }
    pub fn coverage_manifest_digest(&self) -> [u8; 32] { self.coverage_manifest_digest }
    pub fn currentness_at_unix_ms(&self) -> u64 { self.currentness_at_unix_ms }
}

/// Non-Serde proof that exact A is recovered and still current after an ambiguous
/// A -> B crash, while the exact backend-relative boundary was independently proven
/// to have no external effects and that proof was protected before mutation.
#[derive(Debug, Clone)]
pub struct QualifiedNoEffectsCurrentRecoveryV1 {
    record: NoEffectsCurrentRecoveryRecordV1,
}

impl QualifiedNoEffectsCurrentRecoveryV1 {
    pub fn qualify(
        current: &QualifiedCurrentCrashRecoveryV1,
        crash: &QualifiedCrashRecoveryToActiveKnownGoodV1,
        rebound: &ReboundNoEffectsExecutionV1,
        commitment: &QualifiedNoEffectsExecutionCommitmentV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
    ) -> Result<Self, NoEffectsCurrentRecoveryError> {
        let current_record = current.record();
        let crash_record = crash.record();
        let envelope = rebound.envelope();

        if current_record.crash_recovery_id() != crash.id()
            || current_record.subject_id() != crash_record.subject_id()
            || current_record.recovered_realization_id() != crash_record.recovered_realization_id()
        {
            return Err(NoEffectsCurrentRecoveryError::CurrentRecoveryParentMismatch);
        }
        if rebound.known_good_intent_id() != envelope.known_good_intent_id()
            || envelope.attempt_id() != crash_record.original_attempt_id()
            || rebound.subject_id() != crash_record.subject_id()
            || rebound.source_realization_id() != crash_record.recovered_realization_id()
            || rebound.target_realization_id() != crash_record.failed_or_abandoned_target_id()
        {
            return Err(NoEffectsCurrentRecoveryError::NoEffectsLineageMismatch);
        }
        if commitment.journal_anchor_id() != journal_anchor.id()
            || commitment.envelope_id() != rebound.envelope_id()
            || commitment.attempt_id() != crash_record.original_attempt_id()
            || commitment.subject_id() != crash_record.subject_id()
            || commitment.declaration_id() != rebound.declaration_id()
            || commitment.authorization_id() != rebound.authorization_id()
            || commitment.coverage_id() != rebound.coverage_id()
            || commitment.backend_id() != rebound.backend_id()
        {
            return Err(NoEffectsCurrentRecoveryError::ProtectedNoEffectsWorldMismatch);
        }
        if journal_anchor.subject_id() != crash_record.subject_id()
            || journal_anchor.anchored_at_unix_ms() != envelope.coverage_analyzed_at_unix_ms()
        {
            return Err(NoEffectsCurrentRecoveryError::ProtectedNoEffectsWorldMismatch);
        }

        let protected_at_unix_ms = journal_anchor.anchored_at_unix_ms();
        let recovered_at_unix_ms = crash_record.recovered_at_unix_ms();
        let currentness_at_unix_ms = current_record.currentness_anchored_at_unix_ms();
        if protected_at_unix_ms >= recovered_at_unix_ms {
            return Err(NoEffectsCurrentRecoveryError::ProtectedWorldNotBeforeRecovery);
        }
        if currentness_at_unix_ms < recovered_at_unix_ms {
            return Err(NoEffectsCurrentRecoveryError::CurrentnessPredatesRecovery);
        }

        let recovery_id = QualifiedNoEffectsCurrentRecoveryId(hash_recovery(
            current.id(),
            crash.id(),
            rebound.id(),
            commitment.id(),
            journal_anchor.id(),
            crash_record.original_attempt_id(),
            crash_record.subject_id(),
            crash_record.recovered_realization_id(),
            crash_record.failed_or_abandoned_target_id(),
            rebound.declaration_id(),
            rebound.authorization_id(),
            rebound.coverage_id(),
            rebound.backend_id(),
            envelope.coverage_manifest_digest(),
            protected_at_unix_ms,
            recovered_at_unix_ms,
            currentness_at_unix_ms,
        ));
        let record = NoEffectsCurrentRecoveryRecordV1 {
            schema_version: NO_EFFECTS_CURRENT_RECOVERY_RECORD_SCHEMA_V1.to_owned(),
            current_crash_recovery_id: current.id(),
            crash_recovery_id: crash.id(),
            rebound_no_effects_id: rebound.id(),
            protected_no_effects_commitment_id: commitment.id(),
            journal_anchor_id: journal_anchor.id(),
            original_attempt_id: crash_record.original_attempt_id(),
            subject_id: crash_record.subject_id(),
            recovered_realization_id: crash_record.recovered_realization_id(),
            failed_or_abandoned_target_id: crash_record.failed_or_abandoned_target_id(),
            declaration_id: rebound.declaration_id(),
            authorization_id: rebound.authorization_id(),
            coverage_id: rebound.coverage_id(),
            backend_id: rebound.backend_id(),
            coverage_manifest_digest: envelope.coverage_manifest_digest(),
            protected_at_unix_ms,
            recovered_at_unix_ms,
            currentness_at_unix_ms,
            recovery_id,
        };
        record.validate()?;
        Ok(Self { record })
    }

    pub fn rebind(
        record: NoEffectsCurrentRecoveryRecordV1,
        current: &QualifiedCurrentCrashRecoveryV1,
        crash: &QualifiedCrashRecoveryToActiveKnownGoodV1,
        rebound: &ReboundNoEffectsExecutionV1,
        commitment: &QualifiedNoEffectsExecutionCommitmentV1,
        journal_anchor: &QualifiedExecutionJournalAnchorV1,
    ) -> Result<Self, NoEffectsCurrentRecoveryError> {
        record.validate()?;
        let fresh = Self::qualify(current, crash, rebound, commitment, journal_anchor)?;
        if fresh.record != record {
            return Err(NoEffectsCurrentRecoveryError::RecoveryLineageMismatch);
        }
        Ok(fresh)
    }

    pub fn id(&self) -> QualifiedNoEffectsCurrentRecoveryId { self.record.id() }
    pub fn record(&self) -> &NoEffectsCurrentRecoveryRecordV1 { &self.record }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NoEffectsCurrentRecoveryError {
    #[error(transparent)]
    CurrentRecovery(#[from] CurrentCrashRecoveryError),
    #[error(transparent)]
    CrashRecovery(#[from] CrashRecoveryQualificationError),
    #[error(transparent)]
    Rebind(#[from] NoEffectsRebindError),
    #[error(transparent)]
    NoEffectsExecution(#[from] NoEffectsExecutionError),
    #[error("unsupported no-effects current-recovery record schema: {0}")]
    UnsupportedSchema(String),
    #[error("no-effects current-recovery coverage-manifest digest must be non-zero")]
    ZeroCoverageManifestDigest,
    #[error("no-effects current-recovery times must be non-zero")]
    ZeroTime,
    #[error("no-effects current-recovery source equals abandoned target")]
    SourceEqualsTarget,
    #[error("current crash recovery does not name the exact supplied crash-recovery parent")]
    CurrentRecoveryParentMismatch,
    #[error("rebound no-effects world does not match exact original A -> B crash lineage")]
    NoEffectsLineageMismatch,
    #[error("protected no-effects commitment/journal anchor does not match exact rebound world")]
    ProtectedNoEffectsWorldMismatch,
    #[error("protected no-effects world was not established strictly before recovered state")]
    ProtectedWorldNotBeforeRecovery,
    #[error("fresh active-LKG currentness predates recovered state")]
    CurrentnessPredatesRecovery,
    #[error("no-effects current-recovery identity does not match canonical fields")]
    RecoveryIdentityMismatch,
    #[error("persisted no-effects current-recovery record does not match exact live parent proofs")]
    RecoveryLineageMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_recovery(
    current_id: QualifiedCurrentCrashRecoveryId,
    crash_id: QualifiedCrashRecoveryToActiveKnownGoodId,
    rebound_id: ReboundNoEffectsExecutionId,
    commitment_id: QualifiedNoEffectsExecutionCommitmentId,
    anchor_id: QualifiedExecutionJournalAnchorId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    recovered_id: TargetRealizationId,
    target_id: TargetRealizationId,
    declaration_id: NoExternalEffectsDeclarationId,
    authorization_id: QualifiedNoExternalEffectsAuthorizationId,
    coverage_id: QualifiedNoExternalEffectsCoverageId,
    backend_id: ExecutionBackendId,
    manifest: [u8; 32],
    protected_at: u64,
    recovered_at: u64,
    currentness_at: u64,
) -> [u8; 32] {
    let protected_at_bytes = protected_at.to_le_bytes();
    let recovered_at_bytes = recovered_at.to_le_bytes();
    let currentness_at_bytes = currentness_at.to_le_bytes();
    domain_hash_parts(
        RECOVERY_DOMAIN,
        &[
            current_id.as_bytes(), crash_id.as_bytes(), rebound_id.as_bytes(),
            commitment_id.as_bytes(), anchor_id.as_bytes(), attempt_id.as_bytes(),
            subject_id.as_bytes(), recovered_id.as_bytes(), target_id.as_bytes(),
            declaration_id.as_bytes(), authorization_id.as_bytes(), coverage_id.as_bytes(),
            backend_id.as_bytes(), &manifest, &protected_at_bytes, &recovered_at_bytes,
            &currentness_at_bytes,
        ],
    )
}

fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    for part in parts {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_domain_is_distinct_from_effectful_scope_bound_recovery() {
        assert_ne!(RECOVERY_DOMAIN, b"symthaea.continuity.scope-bound-recovery.v1\0");
    }
}
