// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind one exact active known-good source realization A to one exact intended target B.
//!
//! The serializable lineage is descriptive recovery context, not execution authority.
//! The non-Serde wrapper owns trusted commit eligibility so future executor code can
//! consume an A-bound value rather than an unscoped target-only eligibility.
//!
//! `ActiveLkg(A) + TrustedEligibility(B) -> AtoBLineage != ExecutionCapability`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::active_lkg::{ActiveKnownGoodSelectionId, ActiveKnownGoodSelectionV1};
use crate::distributed_state::DistributedStateContextId;
use crate::execution_capability::{ExecutionAttemptId, ExecutionAttemptIntentV1};
use crate::known_good::{KnownGoodCheckpointId, QualifiedKnownGoodCheckpointV1};
use crate::scope::ContinuitySubjectId;
use crate::trusted_commit_epoch::{TrustedCommitEligibilityId, TrustedCommitEligibilityV1};
use crate::witness::TargetRealizationId;

pub const KNOWN_GOOD_TRANSITION_LINEAGE_SCHEMA_V1: &str =
    "symthaea-continuity-known-good-transition-lineage-v1";
pub const KNOWN_GOOD_EXECUTION_INTENT_SCHEMA_V1: &str =
    "symthaea-continuity-known-good-execution-intent-v1";

const LINEAGE_DOMAIN: &[u8] = b"symthaea.continuity.known-good-transition-lineage.v1\0";
const INTENT_DOMAIN: &[u8] = b"symthaea.continuity.known-good-execution-intent.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct KnownGoodTransitionLineageId([u8; 32]);
impl KnownGoodTransitionLineageId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct KnownGoodExecutionIntentId([u8; 32]);
impl KnownGoodExecutionIntentId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

/// Serializable exact A -> B recovery lineage. This cannot recreate trusted
/// eligibility or an execution capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnownGoodTransitionLineageV1 {
    schema_version: String,
    active_selection_id: ActiveKnownGoodSelectionId,
    active_selection_generation: u64,
    source_checkpoint_id: KnownGoodCheckpointId,
    source_checkpoint_generation: u64,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    trusted_eligibility_id: TrustedCommitEligibilityId,
    commit_time_unix_ms: u64,
    lineage_id: KnownGoodTransitionLineageId,
}

impl KnownGoodTransitionLineageV1 {
    pub fn validate(&self) -> Result<(), KnownGoodTransitionLineageError> {
        if self.schema_version != KNOWN_GOOD_TRANSITION_LINEAGE_SCHEMA_V1 {
            return Err(KnownGoodTransitionLineageError::UnsupportedLineageSchema(self.schema_version.clone()));
        }
        if self.active_selection_generation == 0 || self.source_checkpoint_generation == 0 {
            return Err(KnownGoodTransitionLineageError::ZeroGeneration);
        }
        if self.commit_time_unix_ms == 0 {
            return Err(KnownGoodTransitionLineageError::ZeroCommitTime);
        }
        if self.source_realization_id == self.target_realization_id {
            return Err(KnownGoodTransitionLineageError::SourceEqualsTarget);
        }
        let expected = KnownGoodTransitionLineageId(hash_lineage(
            self.active_selection_id,
            self.active_selection_generation,
            self.source_checkpoint_id,
            self.source_checkpoint_generation,
            self.subject_id,
            self.source_realization_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.trusted_eligibility_id,
            self.commit_time_unix_ms,
        ));
        if expected != self.lineage_id {
            return Err(KnownGoodTransitionLineageError::LineageIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> KnownGoodTransitionLineageId { self.lineage_id }
    pub fn active_selection_id(&self) -> ActiveKnownGoodSelectionId { self.active_selection_id }
    pub fn source_checkpoint_id(&self) -> KnownGoodCheckpointId { self.source_checkpoint_id }
    pub fn subject_id(&self) -> ContinuitySubjectId { self.subject_id }
    pub fn source_realization_id(&self) -> TargetRealizationId { self.source_realization_id }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.target_realization_id }
    pub fn distributed_context_id(&self) -> DistributedStateContextId { self.distributed_context_id }
    pub fn trusted_eligibility_id(&self) -> TrustedCommitEligibilityId { self.trusted_eligibility_id }
    pub fn commit_time_unix_ms(&self) -> u64 { self.commit_time_unix_ms }
}

/// Non-Serde, non-Clone trusted eligibility bound to one exact active known-good A.
#[derive(Debug)]
pub struct KnownGoodBoundTrustedCommitEligibilityV1 {
    lineage: KnownGoodTransitionLineageV1,
    eligibility: TrustedCommitEligibilityV1,
}

impl KnownGoodBoundTrustedCommitEligibilityV1 {
    pub fn bind(
        active: &ActiveKnownGoodSelectionV1,
        checkpoint: &QualifiedKnownGoodCheckpointV1,
        eligibility: TrustedCommitEligibilityV1,
    ) -> Result<Self, KnownGoodTransitionLineageError> {
        active.record().validate()?;
        checkpoint.record().validate()?;
        if active.checkpoint_id() != checkpoint.id()
            || active.checkpoint_generation() != checkpoint.generation()
            || active.subject_id() != checkpoint.subject_id()
            || active.realization_id() != checkpoint.realization_id()
        {
            return Err(KnownGoodTransitionLineageError::ActiveSelectionCheckpointMismatch);
        }
        if eligibility.subject_id() != checkpoint.subject_id() {
            return Err(KnownGoodTransitionLineageError::SubjectMismatch);
        }
        if eligibility.target_realization_id() == checkpoint.realization_id() {
            return Err(KnownGoodTransitionLineageError::SourceEqualsTarget);
        }
        if eligibility.commit_time_unix_ms() <= active.selected_at_unix_ms() {
            return Err(KnownGoodTransitionLineageError::CommitDoesNotFollowActiveSelection);
        }

        let lineage = KnownGoodTransitionLineageV1 {
            schema_version: KNOWN_GOOD_TRANSITION_LINEAGE_SCHEMA_V1.to_owned(),
            active_selection_id: active.id(),
            active_selection_generation: active.generation(),
            source_checkpoint_id: checkpoint.id(),
            source_checkpoint_generation: checkpoint.generation(),
            subject_id: eligibility.subject_id(),
            source_realization_id: checkpoint.realization_id(),
            target_realization_id: eligibility.target_realization_id(),
            distributed_context_id: eligibility.distributed_context_id(),
            trusted_eligibility_id: eligibility.id(),
            commit_time_unix_ms: eligibility.commit_time_unix_ms(),
            lineage_id: KnownGoodTransitionLineageId([0; 32]),
        };
        let lineage_id = KnownGoodTransitionLineageId(hash_lineage(
            lineage.active_selection_id,
            lineage.active_selection_generation,
            lineage.source_checkpoint_id,
            lineage.source_checkpoint_generation,
            lineage.subject_id,
            lineage.source_realization_id,
            lineage.target_realization_id,
            lineage.distributed_context_id,
            lineage.trusted_eligibility_id,
            lineage.commit_time_unix_ms,
        ));
        let lineage = KnownGoodTransitionLineageV1 { lineage_id, ..lineage };
        lineage.validate()?;
        Ok(Self { lineage, eligibility })
    }

    pub fn lineage(&self) -> &KnownGoodTransitionLineageV1 { &self.lineage }
    pub fn id(&self) -> KnownGoodTransitionLineageId { self.lineage.id() }

    pub(crate) fn into_parts(self) -> (KnownGoodTransitionLineageV1, TrustedCommitEligibilityV1) {
        (self.lineage, self.eligibility)
    }
}

/// Durable pre-mutation intent with exact active-LKG source lineage. Adapters should
/// journal this artifact (or an equivalent atomic superset) before physical mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnownGoodBoundExecutionAttemptIntentV1 {
    schema_version: String,
    lineage: KnownGoodTransitionLineageV1,
    execution_intent: ExecutionAttemptIntentV1,
    known_good_intent_id: KnownGoodExecutionIntentId,
}

impl KnownGoodBoundExecutionAttemptIntentV1 {
    pub fn bind(
        lineage: KnownGoodTransitionLineageV1,
        execution_intent: ExecutionAttemptIntentV1,
    ) -> Result<Self, KnownGoodTransitionLineageError> {
        lineage.validate()?;
        execution_intent.validate()?;
        if execution_intent.trusted_eligibility_id() != lineage.trusted_eligibility_id
            || execution_intent.subject_id() != lineage.subject_id
            || execution_intent.target_realization_id() != lineage.target_realization_id
            || execution_intent.distributed_context_id() != lineage.distributed_context_id
        {
            return Err(KnownGoodTransitionLineageError::ExecutionIntentLineageMismatch);
        }
        let known_good_intent_id = KnownGoodExecutionIntentId(hash_intent(
            lineage.id(),
            execution_intent.id(),
        ));
        Ok(Self {
            schema_version: KNOWN_GOOD_EXECUTION_INTENT_SCHEMA_V1.to_owned(),
            lineage,
            execution_intent,
            known_good_intent_id,
        })
    }

    pub fn validate(&self) -> Result<(), KnownGoodTransitionLineageError> {
        if self.schema_version != KNOWN_GOOD_EXECUTION_INTENT_SCHEMA_V1 {
            return Err(KnownGoodTransitionLineageError::UnsupportedIntentSchema(self.schema_version.clone()));
        }
        self.lineage.validate()?;
        self.execution_intent.validate()?;
        if self.execution_intent.trusted_eligibility_id() != self.lineage.trusted_eligibility_id
            || self.execution_intent.subject_id() != self.lineage.subject_id
            || self.execution_intent.target_realization_id() != self.lineage.target_realization_id
            || self.execution_intent.distributed_context_id() != self.lineage.distributed_context_id
        {
            return Err(KnownGoodTransitionLineageError::ExecutionIntentLineageMismatch);
        }
        let expected = KnownGoodExecutionIntentId(hash_intent(
            self.lineage.id(),
            self.execution_intent.id(),
        ));
        if expected != self.known_good_intent_id {
            return Err(KnownGoodTransitionLineageError::ExecutionIntentIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> KnownGoodExecutionIntentId { self.known_good_intent_id }
    pub fn attempt_id(&self) -> ExecutionAttemptId { self.execution_intent.id() }
    pub fn lineage(&self) -> &KnownGoodTransitionLineageV1 { &self.lineage }
    pub fn execution_intent(&self) -> &ExecutionAttemptIntentV1 { &self.execution_intent }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KnownGoodTransitionLineageError {
    #[error(transparent)]
    ActiveSelection(#[from] crate::active_lkg::ActiveKnownGoodSelectionError),
    #[error(transparent)]
    Checkpoint(#[from] crate::known_good::KnownGoodCheckpointError),
    #[error(transparent)]
    Execution(#[from] crate::execution_capability::ExecutionCapabilityError),
    #[error("unsupported known-good transition lineage schema: {0}")]
    UnsupportedLineageSchema(String),
    #[error("unsupported known-good execution intent schema: {0}")]
    UnsupportedIntentSchema(String),
    #[error("known-good transition lineage generation must be non-zero")]
    ZeroGeneration,
    #[error("known-good transition commit time must be non-zero")]
    ZeroCommitTime,
    #[error("active LKG selection does not bind the exact qualified checkpoint")]
    ActiveSelectionCheckpointMismatch,
    #[error("active LKG subject differs from transition subject")]
    SubjectMismatch,
    #[error("known-good source realization equals target realization")]
    SourceEqualsTarget,
    #[error("transition commit time must strictly follow active LKG selection")]
    CommitDoesNotFollowActiveSelection,
    #[error("known-good transition lineage identity does not match canonical fields")]
    LineageIdentityMismatch,
    #[error("execution attempt intent does not match the exact A -> B lineage")]
    ExecutionIntentLineageMismatch,
    #[error("known-good execution intent identity does not match canonical fields")]
    ExecutionIntentIdentityMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_lineage(
    active_selection_id: ActiveKnownGoodSelectionId,
    active_selection_generation: u64,
    source_checkpoint_id: KnownGoodCheckpointId,
    source_checkpoint_generation: u64,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    trusted_eligibility_id: TrustedCommitEligibilityId,
    commit_time_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(LINEAGE_DOMAIN);
    hasher.update(active_selection_id.as_bytes());
    hasher.update(&active_selection_generation.to_le_bytes());
    hasher.update(source_checkpoint_id.as_bytes());
    hasher.update(&source_checkpoint_generation.to_le_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(source_realization_id.as_bytes());
    hasher.update(target_realization_id.as_bytes());
    hasher.update(distributed_context_id.as_bytes());
    hasher.update(trusted_eligibility_id.as_bytes());
    hasher.update(&commit_time_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn hash_intent(
    lineage_id: KnownGoodTransitionLineageId,
    attempt_id: ExecutionAttemptId,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(INTENT_DOMAIN);
    hasher.update(lineage_id.as_bytes());
    hasher.update(attempt_id.as_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lineage_domain_is_distinct_from_execution_intent_domain() {
        assert_ne!(LINEAGE_DOMAIN, INTENT_DOMAIN);
    }
}