// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Recovery qualification when crash reconciliation discovers the exact active
//! known-good source realization A already present after a durable A -> B intent.
//!
//! This is intentionally distinct from `QualifiedRecoveryToActiveKnownGoodV1`, which
//! requires a separate explicit physical recovery attempt targeting A. Here no fake
//! recovery execution is invented: independent crash reconciliation establishes A's
//! identity, source-health qualification establishes A as Healthy, and exact
//! distributed-health V2 establishes the surrounding world.
//!
//! `ObservedA != HealthyA != DistributedHealthyA != CrashRecoveredA`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::active_lkg::{ActiveKnownGoodSelectionId, ActiveKnownGoodSelectionV1};
use crate::crash_reconciliation::{
    CrashReconciliationClassificationV1, CrashReconciliationId,
    QualifiedCrashReconciliationV1,
};
use crate::distributed_state::DistributedStateContextId;
use crate::exact_distributed_health::{
    ExactDistributedStateDigestV2, QualifiedExactDistributedHealthIdV2,
    QualifiedExactDistributedHealthV2,
};
use crate::exact_local_health::{
    HealthyLocalSnapshotBasisV1, QualifiedCrashSourceHealthId,
    QualifiedHealthyLocalSnapshotId, QualifiedHealthyLocalSnapshotV1,
};
use crate::execution_capability::ExecutionAttemptId;
use crate::known_good::{KnownGoodCheckpointId, QualifiedKnownGoodCheckpointV1};
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::{
    KnownGoodBoundExecutionAttemptIntentV1, KnownGoodTransitionLineageError,
    KnownGoodTransitionLineageId,
};
use crate::witness::TargetRealizationId;

pub const CRASH_RECOVERED_ACTIVE_KNOWN_GOOD_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-crash-recovered-active-known-good-record-v1";

const CRASH_RECOVERY_DOMAIN: &[u8] =
    b"symthaea.continuity.crash-recovered-active-known-good.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct QualifiedCrashRecoveryToActiveKnownGoodId([u8; 32]);

impl QualifiedCrashRecoveryToActiveKnownGoodId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable audit record for one exact crash-recovery qualification. The record
/// cannot recreate the non-Serde qualified proofs it names; callers must `rebind()`
/// against the exact live objects before treating it as qualified state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashRecoveredActiveKnownGoodRecordV1 {
    schema_version: String,
    original_lineage_id: KnownGoodTransitionLineageId,
    original_attempt_id: ExecutionAttemptId,
    active_selection_id: ActiveKnownGoodSelectionId,
    source_checkpoint_id: KnownGoodCheckpointId,
    subject_id: ContinuitySubjectId,
    recovered_realization_id: TargetRealizationId,
    failed_or_abandoned_target_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    reconciliation_id: CrashReconciliationId,
    source_health_id: QualifiedCrashSourceHealthId,
    local_snapshot_id: QualifiedHealthyLocalSnapshotId,
    distributed_health_id: QualifiedExactDistributedHealthIdV2,
    distributed_state_digest: ExactDistributedStateDigestV2,
    recovered_at_unix_ms: u64,
    recovery_id: QualifiedCrashRecoveryToActiveKnownGoodId,
}

impl CrashRecoveredActiveKnownGoodRecordV1 {
    pub fn validate(&self) -> Result<(), CrashRecoveryQualificationError> {
        if self.schema_version != CRASH_RECOVERED_ACTIVE_KNOWN_GOOD_RECORD_SCHEMA_V1 {
            return Err(CrashRecoveryQualificationError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.recovered_realization_id == self.failed_or_abandoned_target_id {
            return Err(CrashRecoveryQualificationError::SourceEqualsTarget);
        }
        if self.recovered_at_unix_ms == 0 {
            return Err(CrashRecoveryQualificationError::ZeroRecoveryTime);
        }
        let expected = QualifiedCrashRecoveryToActiveKnownGoodId(hash_recovery(
            self.original_lineage_id,
            self.original_attempt_id,
            self.active_selection_id,
            self.source_checkpoint_id,
            self.subject_id,
            self.recovered_realization_id,
            self.failed_or_abandoned_target_id,
            self.distributed_context_id,
            self.reconciliation_id,
            self.source_health_id,
            self.local_snapshot_id,
            self.distributed_health_id,
            self.distributed_state_digest,
            self.recovered_at_unix_ms,
        ));
        if expected != self.recovery_id {
            return Err(CrashRecoveryQualificationError::RecoveryIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> QualifiedCrashRecoveryToActiveKnownGoodId {
        self.recovery_id
    }

    pub fn original_attempt_id(&self) -> ExecutionAttemptId {
        self.original_attempt_id
    }

    pub fn active_selection_id(&self) -> ActiveKnownGoodSelectionId {
        self.active_selection_id
    }

    pub fn source_checkpoint_id(&self) -> KnownGoodCheckpointId {
        self.source_checkpoint_id
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.subject_id
    }

    pub fn recovered_realization_id(&self) -> TargetRealizationId {
        self.recovered_realization_id
    }

    pub fn failed_or_abandoned_target_id(&self) -> TargetRealizationId {
        self.failed_or_abandoned_target_id
    }

    pub fn reconciliation_id(&self) -> CrashReconciliationId {
        self.reconciliation_id
    }

    pub fn local_snapshot_id(&self) -> QualifiedHealthyLocalSnapshotId {
        self.local_snapshot_id
    }

    pub fn distributed_health_id(&self) -> QualifiedExactDistributedHealthIdV2 {
        self.distributed_health_id
    }

    pub fn distributed_state_digest(&self) -> ExactDistributedStateDigestV2 {
        self.distributed_state_digest
    }

    pub fn recovered_at_unix_ms(&self) -> u64 {
        self.recovered_at_unix_ms
    }
}

/// Non-Serde proof that a durable intent-only A -> B crash world has been restored to
/// an independently Healthy exact A with a freshly Healthy distributed world.
///
/// This proves recovery state, not a recovery action. It grants no execution, retry,
/// checkpoint promotion, or external transactional compensation authority.
#[derive(Debug, Clone)]
pub struct QualifiedCrashRecoveryToActiveKnownGoodV1 {
    record: CrashRecoveredActiveKnownGoodRecordV1,
}

impl QualifiedCrashRecoveryToActiveKnownGoodV1 {
    pub fn qualify(
        original_attempt: &KnownGoodBoundExecutionAttemptIntentV1,
        active: &ActiveKnownGoodSelectionV1,
        checkpoint: &QualifiedKnownGoodCheckpointV1,
        reconciliation: &QualifiedCrashReconciliationV1,
        local_snapshot: &QualifiedHealthyLocalSnapshotV1,
        distributed_health: &QualifiedExactDistributedHealthV2,
    ) -> Result<Self, CrashRecoveryQualificationError> {
        original_attempt.validate()?;
        active.record().validate()?;
        checkpoint.record().validate()?;

        let lineage = original_attempt.lineage();
        if lineage.active_selection_id() != active.id()
            || lineage.source_checkpoint_id() != checkpoint.id()
            || active.checkpoint_id() != checkpoint.id()
            || active.checkpoint_generation() != checkpoint.generation()
            || active.subject_id() != checkpoint.subject_id()
            || active.realization_id() != checkpoint.realization_id()
            || lineage.subject_id() != active.subject_id()
            || lineage.source_realization_id() != active.realization_id()
        {
            return Err(CrashRecoveryQualificationError::OriginalKnownGoodLineageMismatch);
        }

        if reconciliation.classification()
            != CrashReconciliationClassificationV1::SourceKnownGoodObserved
            || reconciliation.record().attempt_id() != original_attempt.attempt_id()
            || reconciliation.record().observed_realization_id()
                != Some(active.realization_id())
        {
            return Err(CrashRecoveryQualificationError::CrashReconciliationMismatch);
        }

        let source_health_id = match local_snapshot.basis() {
            HealthyLocalSnapshotBasisV1::CrashReconciledSource {
                health_id,
                reconciliation_id,
                attempt_id,
            } if reconciliation_id == reconciliation.id()
                && attempt_id == original_attempt.attempt_id() => health_id,
            _ => return Err(CrashRecoveryQualificationError::LocalSnapshotBasisMismatch),
        };

        if local_snapshot.subject_id() != active.subject_id()
            || local_snapshot.realization_id() != active.realization_id()
            || local_snapshot.distributed_context_id() != lineage.distributed_context_id()
            || local_snapshot.qualified_at_unix_ms()
                < reconciliation.record().reconciled_at_unix_ms()
        {
            return Err(CrashRecoveryQualificationError::LocalSnapshotContextMismatch);
        }

        if distributed_health.local_snapshot_id() != local_snapshot.id()
            || distributed_health.local_snapshot_basis() != local_snapshot.basis()
            || distributed_health.subject_id() != local_snapshot.subject_id()
            || distributed_health.realization_id() != local_snapshot.realization_id()
            || distributed_health.health_profile_digest() != local_snapshot.health_profile_digest()
            || distributed_health.context_id() != local_snapshot.distributed_context_id()
            || distributed_health.evaluated_at_unix_ms()
                != local_snapshot.qualified_at_unix_ms()
        {
            return Err(CrashRecoveryQualificationError::DistributedHealthMismatch);
        }
        if distributed_health.recovery_paths().is_empty() {
            return Err(CrashRecoveryQualificationError::NoCurrentRecoveryPath);
        }

        let recovered_at_unix_ms = distributed_health.evaluated_at_unix_ms();
        if recovered_at_unix_ms <= lineage.commit_time_unix_ms()
            || recovered_at_unix_ms < reconciliation.record().reconciled_at_unix_ms()
        {
            return Err(CrashRecoveryQualificationError::RecoveryPredatesCrashWorld);
        }

        let recovery_id = QualifiedCrashRecoveryToActiveKnownGoodId(hash_recovery(
            lineage.id(),
            original_attempt.attempt_id(),
            active.id(),
            checkpoint.id(),
            active.subject_id(),
            active.realization_id(),
            lineage.target_realization_id(),
            lineage.distributed_context_id(),
            reconciliation.id(),
            source_health_id,
            local_snapshot.id(),
            distributed_health.id(),
            distributed_health.current_state_digest(),
            recovered_at_unix_ms,
        ));
        let record = CrashRecoveredActiveKnownGoodRecordV1 {
            schema_version: CRASH_RECOVERED_ACTIVE_KNOWN_GOOD_RECORD_SCHEMA_V1.to_owned(),
            original_lineage_id: lineage.id(),
            original_attempt_id: original_attempt.attempt_id(),
            active_selection_id: active.id(),
            source_checkpoint_id: checkpoint.id(),
            subject_id: active.subject_id(),
            recovered_realization_id: active.realization_id(),
            failed_or_abandoned_target_id: lineage.target_realization_id(),
            distributed_context_id: lineage.distributed_context_id(),
            reconciliation_id: reconciliation.id(),
            source_health_id,
            local_snapshot_id: local_snapshot.id(),
            distributed_health_id: distributed_health.id(),
            distributed_state_digest: distributed_health.current_state_digest(),
            recovered_at_unix_ms,
            recovery_id,
        };
        record.validate()?;
        Ok(Self { record })
    }

    /// Rebind a persisted audit record to the exact live proof objects. A valid
    /// record self-hash is not sufficient to recreate qualification after restart.
    pub fn rebind(
        record: CrashRecoveredActiveKnownGoodRecordV1,
        original_attempt: &KnownGoodBoundExecutionAttemptIntentV1,
        active: &ActiveKnownGoodSelectionV1,
        checkpoint: &QualifiedKnownGoodCheckpointV1,
        reconciliation: &QualifiedCrashReconciliationV1,
        local_snapshot: &QualifiedHealthyLocalSnapshotV1,
        distributed_health: &QualifiedExactDistributedHealthV2,
    ) -> Result<Self, CrashRecoveryQualificationError> {
        record.validate()?;
        let fresh = Self::qualify(
            original_attempt,
            active,
            checkpoint,
            reconciliation,
            local_snapshot,
            distributed_health,
        )?;
        if fresh.record != record {
            return Err(CrashRecoveryQualificationError::RecoveryLineageMismatch);
        }
        Ok(fresh)
    }

    pub fn id(&self) -> QualifiedCrashRecoveryToActiveKnownGoodId {
        self.record.id()
    }

    pub fn record(&self) -> &CrashRecoveredActiveKnownGoodRecordV1 {
        &self.record
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CrashRecoveryQualificationError {
    #[error(transparent)]
    TransitionLineage(#[from] KnownGoodTransitionLineageError),
    #[error(transparent)]
    ActiveSelection(#[from] crate::active_lkg::ActiveKnownGoodSelectionError),
    #[error(transparent)]
    Checkpoint(#[from] crate::known_good::KnownGoodCheckpointError),
    #[error("unsupported crash-recovered active-known-good record schema: {0}")]
    UnsupportedSchema(String),
    #[error("crash recovery source and failed/abandoned target must differ")]
    SourceEqualsTarget,
    #[error("crash recovery time must be non-zero")]
    ZeroRecoveryTime,
    #[error("original A -> B attempt does not bind the exact supplied active known-good checkpoint")]
    OriginalKnownGoodLineageMismatch,
    #[error("crash reconciliation does not establish exact source A for the original durable attempt")]
    CrashReconciliationMismatch,
    #[error("healthy local snapshot is not the exact crash-reconciled source-health basis")]
    LocalSnapshotBasisMismatch,
    #[error("healthy local snapshot does not bind exact source A / subject / distributed context")]
    LocalSnapshotContextMismatch,
    #[error("exact distributed health does not bind the exact crash-source healthy snapshot")]
    DistributedHealthMismatch,
    #[error("recovered distributed world has no current independently qualified recovery path")]
    NoCurrentRecoveryPath,
    #[error("crash recovery qualification does not follow the original crash/reconciliation world")]
    RecoveryPredatesCrashWorld,
    #[error("crash recovery identity does not match canonical fields")]
    RecoveryIdentityMismatch,
    #[error("persisted crash recovery record does not match exact live proof lineage")]
    RecoveryLineageMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_recovery(
    original_lineage_id: KnownGoodTransitionLineageId,
    original_attempt_id: ExecutionAttemptId,
    active_selection_id: ActiveKnownGoodSelectionId,
    source_checkpoint_id: KnownGoodCheckpointId,
    subject_id: ContinuitySubjectId,
    recovered_realization_id: TargetRealizationId,
    failed_or_abandoned_target_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    reconciliation_id: CrashReconciliationId,
    source_health_id: QualifiedCrashSourceHealthId,
    local_snapshot_id: QualifiedHealthyLocalSnapshotId,
    distributed_health_id: QualifiedExactDistributedHealthIdV2,
    distributed_state_digest: ExactDistributedStateDigestV2,
    recovered_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CRASH_RECOVERY_DOMAIN);
    hasher.update(original_lineage_id.as_bytes());
    hasher.update(original_attempt_id.as_bytes());
    hasher.update(active_selection_id.as_bytes());
    hasher.update(source_checkpoint_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(recovered_realization_id.as_bytes());
    hasher.update(failed_or_abandoned_target_id.as_bytes());
    hasher.update(distributed_context_id.as_bytes());
    hasher.update(reconciliation_id.as_bytes());
    hasher.update(source_health_id.as_bytes());
    hasher.update(local_snapshot_id.as_bytes());
    hasher.update(distributed_health_id.as_bytes());
    hasher.update(distributed_state_digest.as_bytes());
    hasher.update(&recovered_at_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crash_recovery_domain_is_distinct_from_explicit_recovery_domain() {
        assert_ne!(
            CRASH_RECOVERY_DOMAIN,
            b"symthaea.continuity.recovered-active-known-good.v1\0"
        );
    }
}
