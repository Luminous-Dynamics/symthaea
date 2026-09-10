// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent evidence for realizations that have earned known-good checkpoint status.
//!
//! A qualified checkpoint is still not the active Last Known Good selection.
//! Selection/promotion is a separate authority boundary so health evidence cannot
//! silently redefine what recovery is authorized to restore.
//!
//! `PostTransitionHealth != QualifiedCheckpoint != ActiveLastKnownGood`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed::RecoveryPathClassV1;
use crate::distributed_state::DistributedStateContextId;
use crate::post_execution_health::{
    PostExecutionHealthOutcomeV1, QualifiedPostExecutionHealthId,
    QualifiedPostExecutionHealthV1,
};
use crate::post_transition_distributed_health::{
    PostTransitionDistributedStateDigest, QualifiedPostTransitionDistributedHealthId,
    QualifiedPostTransitionDistributedHealthV1,
};
use crate::scope::ContinuitySubjectId;
use crate::witness::TargetRealizationId;

pub const KNOWN_GOOD_CHECKPOINT_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-known-good-checkpoint-record-v1";

const CHECKPOINT_DOMAIN: &[u8] = b"symthaea.continuity.known-good-checkpoint.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct KnownGoodCheckpointId([u8; 32]);

impl KnownGoodCheckpointId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable description of one exact independently qualified recovery path.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KnownGoodRecoveryPathV1 {
    recovery_path_class: RecoveryPathClassV1,
    recovery_path_identity_digest: [u8; 32],
}

impl KnownGoodRecoveryPathV1 {
    pub fn recovery_path_class(&self) -> &RecoveryPathClassV1 {
        &self.recovery_path_class
    }

    pub fn recovery_path_identity_digest(&self) -> [u8; 32] {
        self.recovery_path_identity_digest
    }
}

/// Persistent checkpoint evidence. This record is intentionally serializable and
/// therefore does not itself grant recovery or execution authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnownGoodCheckpointRecordV1 {
    schema_version: String,
    checkpoint_generation: u64,
    predecessor_checkpoint_id: Option<KnownGoodCheckpointId>,
    subject_id: ContinuitySubjectId,
    realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    local_health_id: QualifiedPostExecutionHealthId,
    distributed_health_id: QualifiedPostTransitionDistributedHealthId,
    distributed_state_digest: PostTransitionDistributedStateDigest,
    health_profile_digest: [u8; 32],
    recovery_paths: Vec<KnownGoodRecoveryPathV1>,
    established_at_unix_ms: u64,
    checkpoint_id: KnownGoodCheckpointId,
}

impl KnownGoodCheckpointRecordV1 {
    pub fn validate(&self) -> Result<(), KnownGoodCheckpointError> {
        if self.schema_version != KNOWN_GOOD_CHECKPOINT_RECORD_SCHEMA_V1 {
            return Err(KnownGoodCheckpointError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.checkpoint_generation == 0 {
            return Err(KnownGoodCheckpointError::ZeroGeneration);
        }
        if self.checkpoint_generation == 1 && self.predecessor_checkpoint_id.is_some() {
            return Err(KnownGoodCheckpointError::UnexpectedPredecessorForFirstGeneration);
        }
        if self.checkpoint_generation > 1 && self.predecessor_checkpoint_id.is_none() {
            return Err(KnownGoodCheckpointError::MissingPredecessor);
        }
        if self.health_profile_digest == [0; 32] {
            return Err(KnownGoodCheckpointError::ZeroHealthProfileDigest);
        }
        if self.established_at_unix_ms == 0 {
            return Err(KnownGoodCheckpointError::ZeroEstablishedTime);
        }
        validate_recovery_paths(&self.recovery_paths)?;
        let expected = KnownGoodCheckpointId(hash_checkpoint(
            self.checkpoint_generation,
            self.predecessor_checkpoint_id,
            self.subject_id,
            self.realization_id,
            self.distributed_context_id,
            self.local_health_id,
            self.distributed_health_id,
            self.distributed_state_digest,
            self.health_profile_digest,
            &self.recovery_paths,
            self.established_at_unix_ms,
        ));
        if expected != self.checkpoint_id {
            return Err(KnownGoodCheckpointError::CheckpointIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> KnownGoodCheckpointId {
        self.checkpoint_id
    }

    pub fn generation(&self) -> u64 {
        self.checkpoint_generation
    }

    pub fn predecessor_checkpoint_id(&self) -> Option<KnownGoodCheckpointId> {
        self.predecessor_checkpoint_id
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.subject_id
    }

    pub fn realization_id(&self) -> TargetRealizationId {
        self.realization_id
    }

    pub fn distributed_context_id(&self) -> DistributedStateContextId {
        self.distributed_context_id
    }

    pub fn distributed_state_digest(&self) -> PostTransitionDistributedStateDigest {
        self.distributed_state_digest
    }

    pub fn health_profile_digest(&self) -> [u8; 32] {
        self.health_profile_digest
    }

    pub fn recovery_paths(&self) -> &[KnownGoodRecoveryPathV1] {
        &self.recovery_paths
    }

    pub fn established_at_unix_ms(&self) -> u64 {
        self.established_at_unix_ms
    }
}

/// Non-Serde checkpoint re-bound to the exact live qualification objects that
/// established it. This is evidence that a realization earned checkpoint status;
/// it is still not an active LKG pointer or recovery capability.
#[derive(Debug, Clone)]
pub struct QualifiedKnownGoodCheckpointV1 {
    record: KnownGoodCheckpointRecordV1,
}

impl QualifiedKnownGoodCheckpointV1 {
    pub fn establish(
        local_health: &QualifiedPostExecutionHealthV1,
        distributed_health: &QualifiedPostTransitionDistributedHealthV1,
        predecessor: Option<&QualifiedKnownGoodCheckpointV1>,
    ) -> Result<Self, KnownGoodCheckpointError> {
        if local_health.outcome() != PostExecutionHealthOutcomeV1::Healthy {
            return Err(KnownGoodCheckpointError::LocalHealthNotHealthy);
        }
        require_exact_post_state(local_health, distributed_health)?;

        let checkpoint_generation = match predecessor {
            Some(previous) => {
                previous.record.validate()?;
                if previous.subject_id() != local_health.subject_id() {
                    return Err(KnownGoodCheckpointError::PredecessorSubjectMismatch);
                }
                previous
                    .generation()
                    .checked_add(1)
                    .ok_or(KnownGoodCheckpointError::GenerationOverflow)?
            }
            None => 1,
        };
        let predecessor_checkpoint_id = predecessor.map(Self::id);
        let recovery_paths = distributed_health
            .recovery_paths()
            .iter()
            .map(|path| KnownGoodRecoveryPathV1 {
                recovery_path_class: path.recovery_path_class().clone(),
                recovery_path_identity_digest: path.recovery_path_identity_digest(),
            })
            .collect::<Vec<_>>();
        validate_recovery_paths(&recovery_paths)?;

        let subject_id = local_health.subject_id();
        let realization_id = local_health.target_realization_id();
        let distributed_context_id = local_health.distributed_context_id();
        let local_health_id = local_health.id();
        let distributed_health_id = distributed_health.id();
        let distributed_state_digest = distributed_health.current_state_digest();
        let health_profile_digest = local_health.health_profile_digest();
        let established_at_unix_ms = distributed_health.evaluated_at_unix_ms();
        let checkpoint_id = KnownGoodCheckpointId(hash_checkpoint(
            checkpoint_generation,
            predecessor_checkpoint_id,
            subject_id,
            realization_id,
            distributed_context_id,
            local_health_id,
            distributed_health_id,
            distributed_state_digest,
            health_profile_digest,
            &recovery_paths,
            established_at_unix_ms,
        ));
        let record = KnownGoodCheckpointRecordV1 {
            schema_version: KNOWN_GOOD_CHECKPOINT_RECORD_SCHEMA_V1.to_owned(),
            checkpoint_generation,
            predecessor_checkpoint_id,
            subject_id,
            realization_id,
            distributed_context_id,
            local_health_id,
            distributed_health_id,
            distributed_state_digest,
            health_profile_digest,
            recovery_paths,
            established_at_unix_ms,
            checkpoint_id,
        };
        record.validate()?;
        Ok(Self { record })
    }

    /// Rebind a serialized record to the exact live qualification objects. This is
    /// intentionally stricter than calling `record.validate()` because a self-hash
    /// cannot establish that the referenced proof objects actually exist.
    pub fn rebind(
        record: KnownGoodCheckpointRecordV1,
        local_health: &QualifiedPostExecutionHealthV1,
        distributed_health: &QualifiedPostTransitionDistributedHealthV1,
    ) -> Result<Self, KnownGoodCheckpointError> {
        record.validate()?;
        require_exact_post_state(local_health, distributed_health)?;
        if record.subject_id != local_health.subject_id()
            || record.realization_id != local_health.target_realization_id()
            || record.distributed_context_id != local_health.distributed_context_id()
            || record.local_health_id != local_health.id()
            || record.distributed_health_id != distributed_health.id()
            || record.distributed_state_digest != distributed_health.current_state_digest()
            || record.health_profile_digest != local_health.health_profile_digest()
            || record.established_at_unix_ms != distributed_health.evaluated_at_unix_ms()
        {
            return Err(KnownGoodCheckpointError::QualificationLineageMismatch);
        }
        let expected_paths = distributed_health
            .recovery_paths()
            .iter()
            .map(|path| KnownGoodRecoveryPathV1 {
                recovery_path_class: path.recovery_path_class().clone(),
                recovery_path_identity_digest: path.recovery_path_identity_digest(),
            })
            .collect::<Vec<_>>();
        if record.recovery_paths != expected_paths {
            return Err(KnownGoodCheckpointError::RecoveryPathLineageMismatch);
        }
        Ok(Self { record })
    }

    pub fn id(&self) -> KnownGoodCheckpointId {
        self.record.id()
    }

    pub fn generation(&self) -> u64 {
        self.record.generation()
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.record.subject_id()
    }

    pub fn realization_id(&self) -> TargetRealizationId {
        self.record.realization_id()
    }

    pub fn distributed_context_id(&self) -> DistributedStateContextId {
        self.record.distributed_context_id()
    }

    pub fn recovery_paths(&self) -> &[KnownGoodRecoveryPathV1] {
        self.record.recovery_paths()
    }

    pub fn record(&self) -> &KnownGoodCheckpointRecordV1 {
        &self.record
    }
}

fn require_exact_post_state(
    local_health: &QualifiedPostExecutionHealthV1,
    distributed_health: &QualifiedPostTransitionDistributedHealthV1,
) -> Result<(), KnownGoodCheckpointError> {
    if distributed_health.local_health_id() != local_health.id()
        || distributed_health.transitioned_subject_id() != local_health.subject_id()
        || distributed_health.context_id() != local_health.distributed_context_id()
        || distributed_health.evaluated_at_unix_ms() != local_health.qualified_at_unix_ms()
    {
        return Err(KnownGoodCheckpointError::QualificationLineageMismatch);
    }
    if distributed_health.recovery_paths().is_empty() {
        return Err(KnownGoodCheckpointError::NoQualifiedRecoveryPath);
    }
    Ok(())
}

fn validate_recovery_paths(paths: &[KnownGoodRecoveryPathV1]) -> Result<(), KnownGoodCheckpointError> {
    if paths.is_empty() {
        return Err(KnownGoodCheckpointError::NoQualifiedRecoveryPath);
    }
    for path in paths {
        if path.recovery_path_identity_digest == [0; 32] {
            return Err(KnownGoodCheckpointError::ZeroRecoveryPathIdentity);
        }
    }
    if paths.windows(2).any(|pair| {
        pair[0].recovery_path_class >= pair[1].recovery_path_class
    }) {
        return Err(KnownGoodCheckpointError::NonCanonicalRecoveryPaths);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KnownGoodCheckpointError {
    #[error("unsupported known-good checkpoint schema: {0}")]
    UnsupportedSchema(String),
    #[error("known-good checkpoint generation must be non-zero")]
    ZeroGeneration,
    #[error("first known-good checkpoint generation may not name a predecessor")]
    UnexpectedPredecessorForFirstGeneration,
    #[error("known-good checkpoint generation after 1 requires an exact predecessor")]
    MissingPredecessor,
    #[error("known-good checkpoint generation overflow")]
    GenerationOverflow,
    #[error("known-good checkpoint predecessor belongs to another continuity subject")]
    PredecessorSubjectMismatch,
    #[error("local post-execution health is not Healthy")]
    LocalHealthNotHealthy,
    #[error("local and distributed post-transition qualification lineage does not match")]
    QualificationLineageMismatch,
    #[error("known-good checkpoint health profile digest must be non-zero")]
    ZeroHealthProfileDigest,
    #[error("known-good checkpoint established time must be non-zero")]
    ZeroEstablishedTime,
    #[error("known-good checkpoint requires at least one independently qualified recovery path")]
    NoQualifiedRecoveryPath,
    #[error("known-good recovery path identity must be non-zero")]
    ZeroRecoveryPathIdentity,
    #[error("known-good recovery paths must be in canonical sorted unique class order")]
    NonCanonicalRecoveryPaths,
    #[error("persisted recovery-path lineage differs from the live qualified distributed state")]
    RecoveryPathLineageMismatch,
    #[error("known-good checkpoint identity does not match canonical fields")]
    CheckpointIdentityMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_checkpoint(
    generation: u64,
    predecessor: Option<KnownGoodCheckpointId>,
    subject_id: ContinuitySubjectId,
    realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    local_health_id: QualifiedPostExecutionHealthId,
    distributed_health_id: QualifiedPostTransitionDistributedHealthId,
    distributed_state_digest: PostTransitionDistributedStateDigest,
    health_profile_digest: [u8; 32],
    recovery_paths: &[KnownGoodRecoveryPathV1],
    established_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CHECKPOINT_DOMAIN);
    hasher.update(&generation.to_le_bytes());
    match predecessor {
        Some(id) => {
            hasher.update(&[1]);
            hasher.update(id.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(subject_id.as_bytes());
    hasher.update(realization_id.as_bytes());
    hasher.update(distributed_context_id.as_bytes());
    hasher.update(local_health_id.as_bytes());
    hasher.update(distributed_health_id.as_bytes());
    hasher.update(distributed_state_digest.as_bytes());
    hasher.update(&health_profile_digest);
    hasher.update(&(recovery_paths.len() as u64).to_le_bytes());
    for path in recovery_paths {
        encode_recovery_class(&mut hasher, &path.recovery_path_class);
        hasher.update(&path.recovery_path_identity_digest);
    }
    hasher.update(&established_at_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn encode_recovery_class(hasher: &mut blake3::Hasher, class: &RecoveryPathClassV1) {
    match class {
        RecoveryPathClassV1::DeviceLocalAutomaticRollback => {
            hasher.update(&[1]);
        }
        RecoveryPathClassV1::OutOfBandManagement => {
            hasher.update(&[2]);
        }
        RecoveryPathClassV1::IndependentNetworkPath => {
            hasher.update(&[3]);
        }
        RecoveryPathClassV1::LocalPhysicalIntervention => {
            hasher.update(&[4]);
        }
        RecoveryPathClassV1::Custom { kind_id } => {
            hasher.update(&[255]);
            hasher.update(&(kind_id.len() as u64).to_le_bytes());
            hasher.update(kind_id.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_paths_require_canonical_unique_classes() {
        let duplicate = vec![
            KnownGoodRecoveryPathV1 {
                recovery_path_class: RecoveryPathClassV1::OutOfBandManagement,
                recovery_path_identity_digest: [1; 32],
            },
            KnownGoodRecoveryPathV1 {
                recovery_path_class: RecoveryPathClassV1::OutOfBandManagement,
                recovery_path_identity_digest: [2; 32],
            },
        ];
        assert_eq!(
            validate_recovery_paths(&duplicate).unwrap_err(),
            KnownGoodCheckpointError::NonCanonicalRecoveryPaths
        );
    }

    #[test]
    fn first_generation_cannot_claim_predecessor() {
        assert!(matches!(
            KnownGoodCheckpointError::UnexpectedPredecessorForFirstGeneration,
            KnownGoodCheckpointError::UnexpectedPredecessorForFirstGeneration
        ));
    }
}