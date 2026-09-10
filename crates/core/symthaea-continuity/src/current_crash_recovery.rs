// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind crash-recovered source A to a fresh proof that A is still the active LKG head.
//!
//! Crash recovery proves the subject/distributed world is healthy at exact source A.
//! Active-LKG currentness separately proves which selection a fresh rollback-resistant
//! platform attestation names as current. This module composes the two without adding
//! execution, retry, promotion, or bootstrap authority.
//!
//! `CrashRecoveredA + FreshCurrentHeadA -> CurrentCrashRecoveredA`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::active_lkg::{ActiveKnownGoodSelectionId, ActiveKnownGoodSelectionError};
use crate::active_lkg_currentness::{
    ActiveLkgCurrentnessError, QualifiedActiveLkgCurrentnessId,
    QualifiedActiveLkgCurrentnessV1,
};
use crate::crash_recovery_qualification::{
    CrashRecoveryQualificationError, QualifiedCrashRecoveryToActiveKnownGoodId,
    QualifiedCrashRecoveryToActiveKnownGoodV1,
};
use crate::execution_journal_anchor::ExecutionJournalAnchorProfileId;
use crate::known_good::KnownGoodCheckpointId;
use crate::scope::ContinuitySubjectId;
use crate::witness::TargetRealizationId;

pub const CURRENT_CRASH_RECOVERY_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-current-crash-recovery-record-v1";

const CURRENT_RECOVERY_DOMAIN: &[u8] =
    b"symthaea.continuity.current-crash-recovery.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedCurrentCrashRecoveryId([u8; 32]);
impl QualifiedCurrentCrashRecoveryId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Persistent audit record for the composition. It cannot recreate either qualified
/// parent proof after restart; `rebind()` requires both exact live objects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentCrashRecoveryRecordV1 {
    schema_version: String,
    crash_recovery_id: QualifiedCrashRecoveryToActiveKnownGoodId,
    currentness_id: QualifiedActiveLkgCurrentnessId,
    currentness_profile_id: ExecutionJournalAnchorProfileId,
    currentness_root_epoch: u64,
    currentness_anchor_sequence: u64,
    active_selection_id: ActiveKnownGoodSelectionId,
    selection_generation: u64,
    source_checkpoint_id: KnownGoodCheckpointId,
    subject_id: ContinuitySubjectId,
    recovered_realization_id: TargetRealizationId,
    freshness_challenge_digest: [u8; 32],
    recovered_at_unix_ms: u64,
    currentness_anchored_at_unix_ms: u64,
    binding_id: QualifiedCurrentCrashRecoveryId,
}

impl CurrentCrashRecoveryRecordV1 {
    pub fn validate(&self) -> Result<(), CurrentCrashRecoveryError> {
        if self.schema_version != CURRENT_CRASH_RECOVERY_RECORD_SCHEMA_V1 {
            return Err(CurrentCrashRecoveryError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.currentness_root_epoch == 0 {
            return Err(CurrentCrashRecoveryError::ZeroCurrentnessRootEpoch);
        }
        if self.currentness_anchor_sequence == 0 {
            return Err(CurrentCrashRecoveryError::ZeroCurrentnessAnchorSequence);
        }
        if self.selection_generation == 0 {
            return Err(CurrentCrashRecoveryError::ZeroSelectionGeneration);
        }
        if self.freshness_challenge_digest == [0; 32] {
            return Err(CurrentCrashRecoveryError::ZeroFreshnessChallengeDigest);
        }
        if self.recovered_at_unix_ms == 0 || self.currentness_anchored_at_unix_ms == 0 {
            return Err(CurrentCrashRecoveryError::ZeroTime);
        }
        if self.currentness_anchored_at_unix_ms < self.recovered_at_unix_ms {
            return Err(CurrentCrashRecoveryError::CurrentnessPredatesRecovery);
        }
        let expected = QualifiedCurrentCrashRecoveryId(hash_binding(
            self.crash_recovery_id,
            self.currentness_id,
            self.currentness_profile_id,
            self.currentness_root_epoch,
            self.currentness_anchor_sequence,
            self.active_selection_id,
            self.selection_generation,
            self.source_checkpoint_id,
            self.subject_id,
            self.recovered_realization_id,
            self.freshness_challenge_digest,
            self.recovered_at_unix_ms,
            self.currentness_anchored_at_unix_ms,
        ));
        if expected != self.binding_id {
            return Err(CurrentCrashRecoveryError::BindingIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> QualifiedCurrentCrashRecoveryId { self.binding_id }
    pub fn crash_recovery_id(&self) -> QualifiedCrashRecoveryToActiveKnownGoodId {
        self.crash_recovery_id
    }
    pub fn currentness_id(&self) -> QualifiedActiveLkgCurrentnessId { self.currentness_id }
    pub fn active_selection_id(&self) -> ActiveKnownGoodSelectionId { self.active_selection_id }
    pub fn source_checkpoint_id(&self) -> KnownGoodCheckpointId { self.source_checkpoint_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn recovered_realization_id(&self) -> TargetRealizationId {
        self.recovered_realization_id
    }
    pub fn freshness_challenge_digest(&self) -> [u8; 32] {
        self.freshness_challenge_digest
    }
    pub fn currentness_anchored_at_unix_ms(&self) -> u64 {
        self.currentness_anchored_at_unix_ms
    }
}

/// Non-Serde proof that crash-recovered A is still the freshly attested current
/// active-LKG head after recovery completed.
#[derive(Debug, Clone)]
pub struct QualifiedCurrentCrashRecoveryV1 {
    record: CurrentCrashRecoveryRecordV1,
}

impl QualifiedCurrentCrashRecoveryV1 {
    pub fn qualify(
        recovery: &QualifiedCrashRecoveryToActiveKnownGoodV1,
        currentness: &QualifiedActiveLkgCurrentnessV1,
    ) -> Result<Self, CurrentCrashRecoveryError> {
        let recovered = recovery.record();
        if currentness.active_selection_id() != recovered.active_selection_id()
            || currentness.checkpoint_id() != recovered.source_checkpoint_id()
            || currentness.subject_id() != recovered.subject_id()
            || currentness.realization_id() != recovered.recovered_realization_id()
        {
            return Err(CurrentCrashRecoveryError::CurrentHeadMismatch);
        }
        if currentness.anchored_at_unix_ms() < recovered.recovered_at_unix_ms() {
            return Err(CurrentCrashRecoveryError::CurrentnessPredatesRecovery);
        }
        if currentness.freshness_challenge_digest() == [0; 32] {
            return Err(CurrentCrashRecoveryError::ZeroFreshnessChallengeDigest);
        }

        let binding_id = QualifiedCurrentCrashRecoveryId(hash_binding(
            recovery.id(),
            currentness.id(),
            currentness.profile_id(),
            currentness.root_epoch(),
            currentness.anchor_sequence(),
            currentness.active_selection_id(),
            currentness.selection_generation(),
            currentness.checkpoint_id(),
            currentness.subject_id(),
            currentness.realization_id(),
            currentness.freshness_challenge_digest(),
            recovered.recovered_at_unix_ms(),
            currentness.anchored_at_unix_ms(),
        ));
        let record = CurrentCrashRecoveryRecordV1 {
            schema_version: CURRENT_CRASH_RECOVERY_RECORD_SCHEMA_V1.to_owned(),
            crash_recovery_id: recovery.id(),
            currentness_id: currentness.id(),
            currentness_profile_id: currentness.profile_id(),
            currentness_root_epoch: currentness.root_epoch(),
            currentness_anchor_sequence: currentness.anchor_sequence(),
            active_selection_id: currentness.active_selection_id(),
            selection_generation: currentness.selection_generation(),
            source_checkpoint_id: currentness.checkpoint_id(),
            subject_id: currentness.subject_id(),
            recovered_realization_id: currentness.realization_id(),
            freshness_challenge_digest: currentness.freshness_challenge_digest(),
            recovered_at_unix_ms: recovered.recovered_at_unix_ms(),
            currentness_anchored_at_unix_ms: currentness.anchored_at_unix_ms(),
            binding_id,
        };
        record.validate()?;
        Ok(Self { record })
    }

    pub fn rebind(
        record: CurrentCrashRecoveryRecordV1,
        recovery: &QualifiedCrashRecoveryToActiveKnownGoodV1,
        currentness: &QualifiedActiveLkgCurrentnessV1,
    ) -> Result<Self, CurrentCrashRecoveryError> {
        record.validate()?;
        let fresh = Self::qualify(recovery, currentness)?;
        if fresh.record != record {
            return Err(CurrentCrashRecoveryError::BindingLineageMismatch);
        }
        Ok(fresh)
    }

    pub fn id(&self) -> QualifiedCurrentCrashRecoveryId { self.record.id() }
    pub fn record(&self) -> &CurrentCrashRecoveryRecordV1 { &self.record }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CurrentCrashRecoveryError {
    #[error(transparent)]
    ActiveSelection(#[from] ActiveKnownGoodSelectionError),
    #[error(transparent)]
    Currentness(#[from] ActiveLkgCurrentnessError),
    #[error(transparent)]
    Recovery(#[from] CrashRecoveryQualificationError),
    #[error("unsupported current crash-recovery record schema: {0}")]
    UnsupportedSchema(String),
    #[error("current crash-recovery root epoch must be non-zero")]
    ZeroCurrentnessRootEpoch,
    #[error("current crash-recovery anchor sequence must be non-zero")]
    ZeroCurrentnessAnchorSequence,
    #[error("current crash-recovery selection generation must be non-zero")]
    ZeroSelectionGeneration,
    #[error("current crash-recovery freshness challenge must be non-zero")]
    ZeroFreshnessChallengeDigest,
    #[error("current crash-recovery times must be non-zero")]
    ZeroTime,
    #[error("fresh active-LKG currentness does not name the exact crash-recovered A head")]
    CurrentHeadMismatch,
    #[error("active-LKG currentness was anchored before crash recovery completed")]
    CurrentnessPredatesRecovery,
    #[error("current crash-recovery identity does not match canonical fields")]
    BindingIdentityMismatch,
    #[error("persisted current crash-recovery record does not match exact live proofs")]
    BindingLineageMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_binding(
    crash_recovery_id: QualifiedCrashRecoveryToActiveKnownGoodId,
    currentness_id: QualifiedActiveLkgCurrentnessId,
    profile_id: ExecutionJournalAnchorProfileId,
    root_epoch: u64,
    anchor_sequence: u64,
    active_selection_id: ActiveKnownGoodSelectionId,
    selection_generation: u64,
    source_checkpoint_id: KnownGoodCheckpointId,
    subject_id: ContinuitySubjectId,
    recovered_realization_id: TargetRealizationId,
    freshness_challenge_digest: [u8; 32],
    recovered_at_unix_ms: u64,
    currentness_anchored_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CURRENT_RECOVERY_DOMAIN);
    hasher.update(crash_recovery_id.as_bytes());
    hasher.update(currentness_id.as_bytes());
    hasher.update(profile_id.as_bytes());
    hasher.update(&root_epoch.to_le_bytes());
    hasher.update(&anchor_sequence.to_le_bytes());
    hasher.update(active_selection_id.as_bytes());
    hasher.update(&selection_generation.to_le_bytes());
    hasher.update(source_checkpoint_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(recovered_realization_id.as_bytes());
    hasher.update(&freshness_challenge_digest);
    hasher.update(&recovered_at_unix_ms.to_le_bytes());
    hasher.update(&currentness_anchored_at_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_domain_is_distinct_from_parent_domains() {
        assert_ne!(
            CURRENT_RECOVERY_DOMAIN,
            b"symthaea.continuity.crash-recovered-active-known-good.v1\0"
        );
        assert_ne!(
            CURRENT_RECOVERY_DOMAIN,
            b"symthaea.continuity.qualified-active-lkg-currentness.v1\0"
        );
    }
}
