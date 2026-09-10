// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent recovery qualification back to the exact active known-good realization.
//!
//! A backend rollback result is never sufficient. Recovery is established only when
//! the exact active source realization A is independently observed healthy again and
//! the larger distributed world is freshly qualified after the recovery attempt.
//!
//! `RollbackReceipt != RecoveredActiveKnownGood != RecoveryAuthority`.

use thiserror::Error;

use crate::active_lkg::{ActiveKnownGoodSelectionId, ActiveKnownGoodSelectionV1};
use crate::distributed_state::DistributedStateContextId;
use crate::execution_capability::{ExecutionAttemptId, ExecutionAttemptIntentV1};
use crate::known_good::{KnownGoodCheckpointId, QualifiedKnownGoodCheckpointV1};
use crate::post_execution_health::{
    PostExecutionHealthOutcomeV1, QualifiedPostExecutionHealthId,
    QualifiedPostExecutionHealthV1,
};
use crate::post_transition_distributed_health::{
    PostTransitionDistributedStateDigest, QualifiedPostTransitionDistributedHealthId,
    QualifiedPostTransitionDistributedHealthV1,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::{
    KnownGoodBoundExecutionAttemptIntentV1, KnownGoodTransitionLineageId,
};
use crate::witness::TargetRealizationId;

const RECOVERY_DOMAIN: &[u8] = b"symthaea.continuity.recovered-active-known-good.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QualifiedRecoveryToActiveKnownGoodId([u8; 32]);
impl QualifiedRecoveryToActiveKnownGoodId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Non-Serde proof that one exact original A -> B transition has been recovered
/// back to the still-active A and that local + distributed health were freshly
/// re-established after the recovery attempt.
#[derive(Debug, Clone)]
pub struct QualifiedRecoveryToActiveKnownGoodV1 {
    recovery_id: QualifiedRecoveryToActiveKnownGoodId,
    original_lineage_id: KnownGoodTransitionLineageId,
    original_attempt_id: ExecutionAttemptId,
    active_selection_id: ActiveKnownGoodSelectionId,
    source_checkpoint_id: KnownGoodCheckpointId,
    subject_id: ContinuitySubjectId,
    recovered_realization_id: TargetRealizationId,
    failed_or_abandoned_target_id: TargetRealizationId,
    recovery_attempt_id: ExecutionAttemptId,
    recovery_context_id: DistributedStateContextId,
    local_health_id: QualifiedPostExecutionHealthId,
    distributed_health_id: QualifiedPostTransitionDistributedHealthId,
    distributed_state_digest: PostTransitionDistributedStateDigest,
    recovered_at_unix_ms: u64,
}

impl QualifiedRecoveryToActiveKnownGoodV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn qualify(
        original_attempt: &KnownGoodBoundExecutionAttemptIntentV1,
        active: &ActiveKnownGoodSelectionV1,
        checkpoint: &QualifiedKnownGoodCheckpointV1,
        recovery_intent: &ExecutionAttemptIntentV1,
        recovery_health: &QualifiedPostExecutionHealthV1,
        recovery_distributed_health: &QualifiedPostTransitionDistributedHealthV1,
    ) -> Result<Self, RecoveryQualificationError> {
        original_attempt.validate()?;
        active.record().validate()?;
        checkpoint.record().validate()?;
        recovery_intent.validate()?;

        let lineage = original_attempt.lineage();
        if lineage.active_selection_id() != active.id()
            || lineage.source_checkpoint_id() != checkpoint.id()
            || active.checkpoint_id() != checkpoint.id()
            || active.checkpoint_generation() != checkpoint.generation()
            || active.subject_id() != checkpoint.subject_id()
            || active.realization_id() != checkpoint.realization_id()
        {
            return Err(RecoveryQualificationError::OriginalKnownGoodLineageMismatch);
        }
        if lineage.subject_id() != active.subject_id()
            || lineage.source_realization_id() != active.realization_id()
        {
            return Err(RecoveryQualificationError::OriginalKnownGoodLineageMismatch);
        }

        // Recovery must be a distinct physical attempt targeting exactly A.
        if recovery_intent.id() == original_attempt.attempt_id() {
            return Err(RecoveryQualificationError::RecoveryAttemptEqualsOriginalAttempt);
        }
        if recovery_intent.subject_id() != active.subject_id() {
            return Err(RecoveryQualificationError::RecoverySubjectMismatch);
        }
        if recovery_intent.target_realization_id() != active.realization_id() {
            return Err(RecoveryQualificationError::RecoveryTargetIsNotActiveKnownGood);
        }

        if recovery_health.outcome() != PostExecutionHealthOutcomeV1::Healthy {
            return Err(RecoveryQualificationError::RecoveredTargetNotHealthy);
        }
        if recovery_health.attempt_id() != recovery_intent.id()
            || recovery_health.subject_id() != recovery_intent.subject_id()
            || recovery_health.target_realization_id() != recovery_intent.target_realization_id()
            || recovery_health.distributed_context_id() != recovery_intent.distributed_context_id()
        {
            return Err(RecoveryQualificationError::RecoveryHealthIntentMismatch);
        }

        if recovery_distributed_health.local_health_id() != recovery_health.id()
            || recovery_distributed_health.transitioned_subject_id() != recovery_health.subject_id()
            || recovery_distributed_health.context_id() != recovery_health.distributed_context_id()
            || recovery_distributed_health.evaluated_at_unix_ms()
                != recovery_health.qualified_at_unix_ms()
        {
            return Err(RecoveryQualificationError::RecoveryDistributedHealthMismatch);
        }

        let recovered_at_unix_ms = recovery_distributed_health.evaluated_at_unix_ms();
        if recovered_at_unix_ms <= lineage.commit_time_unix_ms() {
            return Err(RecoveryQualificationError::RecoveryDoesNotFollowOriginalTransition);
        }

        let recovery_id = QualifiedRecoveryToActiveKnownGoodId(hash_recovery(
            lineage.id(),
            original_attempt.attempt_id(),
            active.id(),
            checkpoint.id(),
            active.subject_id(),
            active.realization_id(),
            lineage.target_realization_id(),
            recovery_intent.id(),
            recovery_intent.distributed_context_id(),
            recovery_health.id(),
            recovery_distributed_health.id(),
            recovery_distributed_health.current_state_digest(),
            recovered_at_unix_ms,
        ));

        Ok(Self {
            recovery_id,
            original_lineage_id: lineage.id(),
            original_attempt_id: original_attempt.attempt_id(),
            active_selection_id: active.id(),
            source_checkpoint_id: checkpoint.id(),
            subject_id: active.subject_id(),
            recovered_realization_id: active.realization_id(),
            failed_or_abandoned_target_id: lineage.target_realization_id(),
            recovery_attempt_id: recovery_intent.id(),
            recovery_context_id: recovery_intent.distributed_context_id(),
            local_health_id: recovery_health.id(),
            distributed_health_id: recovery_distributed_health.id(),
            distributed_state_digest: recovery_distributed_health.current_state_digest(),
            recovered_at_unix_ms,
        })
    }

    pub fn id(&self) -> QualifiedRecoveryToActiveKnownGoodId { self.recovery_id }
    pub fn original_lineage_id(&self) -> KnownGoodTransitionLineageId { self.original_lineage_id }
    pub fn original_attempt_id(&self) -> ExecutionAttemptId { self.original_attempt_id }
    pub fn active_selection_id(&self) -> ActiveKnownGoodSelectionId { self.active_selection_id }
    pub fn source_checkpoint_id(&self) -> KnownGoodCheckpointId { self.source_checkpoint_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn recovered_realization_id(&self) -> TargetRealizationId { self.recovered_realization_id }
    pub fn failed_or_abandoned_target_id(&self) -> TargetRealizationId { self.failed_or_abandoned_target_id }
    pub fn recovery_attempt_id(&self) -> ExecutionAttemptId { self.recovery_attempt_id }
    pub fn recovery_context_id(&self) -> DistributedStateContextId { self.recovery_context_id }
    pub fn local_health_id(&self) -> QualifiedPostExecutionHealthId { self.local_health_id }
    pub fn distributed_health_id(&self) -> QualifiedPostTransitionDistributedHealthId { self.distributed_health_id }
    pub fn distributed_state_digest(&self) -> PostTransitionDistributedStateDigest { self.distributed_state_digest }
    pub fn recovered_at_unix_ms(&self) -> u64 { self.recovered_at_unix_ms }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RecoveryQualificationError {
    #[error(transparent)]
    TransitionLineage(#[from] crate::transition_lineage::KnownGoodTransitionLineageError),
    #[error(transparent)]
    ActiveSelection(#[from] crate::active_lkg::ActiveKnownGoodSelectionError),
    #[error(transparent)]
    Checkpoint(#[from] crate::known_good::KnownGoodCheckpointError),
    #[error(transparent)]
    Execution(#[from] crate::execution_capability::ExecutionCapabilityError),
    #[error("original A -> B attempt does not bind the exact still-active known-good checkpoint")]
    OriginalKnownGoodLineageMismatch,
    #[error("recovery attempt must be distinct from the original transition attempt")]
    RecoveryAttemptEqualsOriginalAttempt,
    #[error("recovery attempt belongs to another continuity subject")]
    RecoverySubjectMismatch,
    #[error("recovery attempt does not target the exact active known-good realization")]
    RecoveryTargetIsNotActiveKnownGood,
    #[error("independently observed recovered target is not Healthy")]
    RecoveredTargetNotHealthy,
    #[error("post-recovery health does not bind the exact recovery attempt")]
    RecoveryHealthIntentMismatch,
    #[error("fresh post-recovery distributed health does not bind the exact recovered local health")]
    RecoveryDistributedHealthMismatch,
    #[error("recovery qualification does not occur after the original transition commit")]
    RecoveryDoesNotFollowOriginalTransition,
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
    recovery_attempt_id: ExecutionAttemptId,
    recovery_context_id: DistributedStateContextId,
    local_health_id: QualifiedPostExecutionHealthId,
    distributed_health_id: QualifiedPostTransitionDistributedHealthId,
    distributed_state_digest: PostTransitionDistributedStateDigest,
    recovered_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECOVERY_DOMAIN);
    hasher.update(original_lineage_id.as_bytes());
    hasher.update(original_attempt_id.as_bytes());
    hasher.update(active_selection_id.as_bytes());
    hasher.update(source_checkpoint_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(recovered_realization_id.as_bytes());
    hasher.update(failed_or_abandoned_target_id.as_bytes());
    hasher.update(recovery_attempt_id.as_bytes());
    hasher.update(recovery_context_id.as_bytes());
    hasher.update(local_health_id.as_bytes());
    hasher.update(distributed_health_id.as_bytes());
    hasher.update(distributed_state_digest.as_bytes());
    hasher.update(&recovered_at_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_domain_is_not_transition_lineage_domain() {
        assert_ne!(RECOVERY_DOMAIN, b"symthaea.continuity.known-good-transition-lineage.v1\0");
    }
}