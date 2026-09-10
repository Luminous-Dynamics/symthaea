// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Promotion eligibility for advancing active Last Known Good from A to B.
//!
//! A checkpoint B may be independently healthy yet unrelated to the A -> B attempt
//! under consideration. Promotion therefore binds B's checkpoint lineage and live
//! qualification back to the exact durable transition attempt from active A.
//!
//! `HealthyCheckpoint(B) != PromotionEligible(A->B) != ActiveLkgSelection(B)`.

use thiserror::Error;

use crate::active_lkg::{ActiveKnownGoodSelectionId, ActiveKnownGoodSelectionV1};
use crate::execution_capability::ExecutionAttemptId;
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

const PROMOTION_DOMAIN: &[u8] = b"symthaea.continuity.lkg-promotion-eligibility.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LkgPromotionEligibilityId([u8; 32]);
impl LkgPromotionEligibilityId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Non-Serde proof that candidate checkpoint B is the exact healthy result of the
/// durable A -> B transition originating from the currently active checkpoint A.
#[derive(Debug)]
pub struct LkgPromotionEligibilityV1 {
    eligibility_id: LkgPromotionEligibilityId,
    original_lineage_id: KnownGoodTransitionLineageId,
    original_attempt_id: ExecutionAttemptId,
    active_selection_id: ActiveKnownGoodSelectionId,
    source_checkpoint_id: KnownGoodCheckpointId,
    candidate_checkpoint_id: KnownGoodCheckpointId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    candidate_realization_id: TargetRealizationId,
    candidate_local_health_id: QualifiedPostExecutionHealthId,
    candidate_distributed_health_id: QualifiedPostTransitionDistributedHealthId,
    candidate_distributed_state_digest: PostTransitionDistributedStateDigest,
    eligible_at_unix_ms: u64,
}

impl LkgPromotionEligibilityV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn qualify(
        original_attempt: &KnownGoodBoundExecutionAttemptIntentV1,
        active_a: &ActiveKnownGoodSelectionV1,
        checkpoint_a: &QualifiedKnownGoodCheckpointV1,
        checkpoint_b: &QualifiedKnownGoodCheckpointV1,
        b_local_health: &QualifiedPostExecutionHealthV1,
        b_distributed_health: &QualifiedPostTransitionDistributedHealthV1,
    ) -> Result<Self, LkgPromotionError> {
        original_attempt.validate()?;
        active_a.record().validate()?;
        checkpoint_a.record().validate()?;
        checkpoint_b.record().validate()?;

        let lineage = original_attempt.lineage();
        if lineage.active_selection_id() != active_a.id()
            || lineage.source_checkpoint_id() != checkpoint_a.id()
            || active_a.checkpoint_id() != checkpoint_a.id()
            || active_a.checkpoint_generation() != checkpoint_a.generation()
            || active_a.subject_id() != checkpoint_a.subject_id()
            || active_a.realization_id() != checkpoint_a.realization_id()
        {
            return Err(LkgPromotionError::ActiveSourceMismatch);
        }
        if lineage.subject_id() != checkpoint_a.subject_id()
            || lineage.source_realization_id() != checkpoint_a.realization_id()
        {
            return Err(LkgPromotionError::OriginalLineageMismatch);
        }

        let expected_candidate_generation = checkpoint_a
            .generation()
            .checked_add(1)
            .ok_or(LkgPromotionError::GenerationOverflow)?;
        if checkpoint_b.record().predecessor_checkpoint_id() != Some(checkpoint_a.id())
            || checkpoint_b.generation() != expected_candidate_generation
            || checkpoint_b.subject_id() != checkpoint_a.subject_id()
            || checkpoint_b.realization_id() != lineage.target_realization_id()
        {
            return Err(LkgPromotionError::CandidateCheckpointLineageMismatch);
        }

        // Rebind candidate B to the exact local + distributed post-transition proof.
        QualifiedKnownGoodCheckpointV1::rebind(
            checkpoint_b.record().clone(),
            b_local_health,
            b_distributed_health,
        )?;
        if b_local_health.outcome() != PostExecutionHealthOutcomeV1::Healthy {
            return Err(LkgPromotionError::CandidateNotHealthy);
        }
        if b_local_health.attempt_id() != original_attempt.attempt_id()
            || b_local_health.subject_id() != lineage.subject_id()
            || b_local_health.target_realization_id() != lineage.target_realization_id()
        {
            return Err(LkgPromotionError::CandidateHealthNotFromOriginalAttempt);
        }
        if b_distributed_health.local_health_id() != b_local_health.id()
            || b_distributed_health.transitioned_subject_id() != b_local_health.subject_id()
            || b_distributed_health.context_id() != b_local_health.distributed_context_id()
        {
            return Err(LkgPromotionError::CandidateDistributedHealthMismatch);
        }

        let eligible_at_unix_ms = b_distributed_health.evaluated_at_unix_ms();
        if eligible_at_unix_ms <= lineage.commit_time_unix_ms() {
            return Err(LkgPromotionError::PromotionDoesNotFollowTransition);
        }

        let eligibility_id = LkgPromotionEligibilityId(hash_promotion(
            lineage.id(),
            original_attempt.attempt_id(),
            active_a.id(),
            checkpoint_a.id(),
            checkpoint_b.id(),
            checkpoint_a.subject_id(),
            checkpoint_a.realization_id(),
            checkpoint_b.realization_id(),
            b_local_health.id(),
            b_distributed_health.id(),
            b_distributed_health.current_state_digest(),
            eligible_at_unix_ms,
        ));

        Ok(Self {
            eligibility_id,
            original_lineage_id: lineage.id(),
            original_attempt_id: original_attempt.attempt_id(),
            active_selection_id: active_a.id(),
            source_checkpoint_id: checkpoint_a.id(),
            candidate_checkpoint_id: checkpoint_b.id(),
            subject_id: checkpoint_a.subject_id(),
            source_realization_id: checkpoint_a.realization_id(),
            candidate_realization_id: checkpoint_b.realization_id(),
            candidate_local_health_id: b_local_health.id(),
            candidate_distributed_health_id: b_distributed_health.id(),
            candidate_distributed_state_digest: b_distributed_health.current_state_digest(),
            eligible_at_unix_ms,
        })
    }

    /// Consume promotion eligibility into a new active LKG selection. Selection is
    /// still policy state, not a physical execution capability.
    pub fn promote(
        self,
        active_a: &ActiveKnownGoodSelectionV1,
        checkpoint_b: &QualifiedKnownGoodCheckpointV1,
        selected_at_unix_ms: u64,
        selection_basis_digest: [u8; 32],
    ) -> Result<ActiveKnownGoodSelectionV1, LkgPromotionError> {
        if active_a.id() != self.active_selection_id
            || checkpoint_b.id() != self.candidate_checkpoint_id
            || active_a.subject_id() != self.subject_id
            || checkpoint_b.subject_id() != self.subject_id
            || active_a.realization_id() != self.source_realization_id
            || checkpoint_b.realization_id() != self.candidate_realization_id
        {
            return Err(LkgPromotionError::PromotionContextMismatch);
        }
        if selected_at_unix_ms < self.eligible_at_unix_ms {
            return Err(LkgPromotionError::SelectionPredatesEligibility);
        }
        ActiveKnownGoodSelectionV1::select(
            checkpoint_b,
            Some(active_a),
            selected_at_unix_ms,
            selection_basis_digest,
        )
        .map_err(Into::into)
    }

    pub fn id(&self) -> LkgPromotionEligibilityId { self.eligibility_id }
    pub fn original_lineage_id(&self) -> KnownGoodTransitionLineageId { self.original_lineage_id }
    pub fn original_attempt_id(&self) -> ExecutionAttemptId { self.original_attempt_id }
    pub fn active_selection_id(&self) -> ActiveKnownGoodSelectionId { self.active_selection_id }
    pub fn source_checkpoint_id(&self) -> KnownGoodCheckpointId { self.source_checkpoint_id }
    pub fn candidate_checkpoint_id(&self) -> KnownGoodCheckpointId { self.candidate_checkpoint_id }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.source_realization_id }
    pub fn candidate_realization_id(&self) -> TargetRealizationId { self.candidate_realization_id }
    pub fn eligible_at_unix_ms(&self) -> u64 { self.eligible_at_unix_ms }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LkgPromotionError {
    #[error(transparent)]
    TransitionLineage(#[from] crate::transition_lineage::KnownGoodTransitionLineageError),
    #[error(transparent)]
    ActiveSelection(#[from] crate::active_lkg::ActiveKnownGoodSelectionError),
    #[error(transparent)]
    Checkpoint(#[from] crate::known_good::KnownGoodCheckpointError),
    #[error("active LKG selection/checkpoint does not match the original A source")]
    ActiveSourceMismatch,
    #[error("original durable transition lineage does not match source checkpoint A")]
    OriginalLineageMismatch,
    #[error("candidate checkpoint B is not the exact successor/result of A -> B")]
    CandidateCheckpointLineageMismatch,
    #[error("candidate checkpoint generation overflow")]
    GenerationOverflow,
    #[error("candidate B is not independently Healthy")]
    CandidateNotHealthy,
    #[error("candidate B health was not established for the exact original transition attempt")]
    CandidateHealthNotFromOriginalAttempt,
    #[error("candidate B distributed health does not bind the exact candidate local health")]
    CandidateDistributedHealthMismatch,
    #[error("promotion eligibility does not occur after the original transition commit")]
    PromotionDoesNotFollowTransition,
    #[error("promotion inputs no longer match the exact qualified eligibility")]
    PromotionContextMismatch,
    #[error("active LKG selection time predates promotion eligibility")]
    SelectionPredatesEligibility,
}

#[allow(clippy::too_many_arguments)]
fn hash_promotion(
    original_lineage_id: KnownGoodTransitionLineageId,
    original_attempt_id: ExecutionAttemptId,
    active_selection_id: ActiveKnownGoodSelectionId,
    source_checkpoint_id: KnownGoodCheckpointId,
    candidate_checkpoint_id: KnownGoodCheckpointId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    candidate_realization_id: TargetRealizationId,
    local_health_id: QualifiedPostExecutionHealthId,
    distributed_health_id: QualifiedPostTransitionDistributedHealthId,
    distributed_state_digest: PostTransitionDistributedStateDigest,
    eligible_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PROMOTION_DOMAIN);
    hasher.update(original_lineage_id.as_bytes());
    hasher.update(original_attempt_id.as_bytes());
    hasher.update(active_selection_id.as_bytes());
    hasher.update(source_checkpoint_id.as_bytes());
    hasher.update(candidate_checkpoint_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(source_realization_id.as_bytes());
    hasher.update(candidate_realization_id.as_bytes());
    hasher.update(local_health_id.as_bytes());
    hasher.update(distributed_health_id.as_bytes());
    hasher.update(distributed_state_digest.as_bytes());
    hasher.update(&eligible_at_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_domain_is_distinct_from_recovery_domain() {
        assert_ne!(PROMOTION_DOMAIN, b"symthaea.continuity.recovered-active-known-good.v1\0");
    }
}