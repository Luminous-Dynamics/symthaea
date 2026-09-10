// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent reconciliation of a durable A -> B attempt whose trustworthy
//! backend result is absent.
//!
//! The executor journal deliberately treats intent-without-receipt as spent and
//! awaiting reconciliation. This module binds that durable fact to an independently
//! qualified post-execution observation and classifies the actual world without
//! creating retry, recovery, promotion, or health authority.
//!
//! `IntentWithoutReceipt != RetryPermission`.
//! `ObservedAorBorOther != Healthy != Recovered != Promoted`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed_state::DistributedStateContextId;
use crate::execution_capability::ExecutionAttemptId;
use crate::execution_journal::{
    ExecutionJournalDigest, JournalAttemptDispositionV1, ReconstructedExecutionJournalV1,
};
use crate::execution_journal_anchor::{
    QualifiedExecutionJournalAnchorId, QualifiedExecutionJournalAnchorV1,
};
use crate::post_execution_observation::{
    PostExecutionObservedStateV1, QualifiedPostExecutionObservationId,
    QualifiedPostExecutionObservationV1,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::{
    KnownGoodBoundExecutionAttemptIntentV1, KnownGoodExecutionIntentId,
    KnownGoodTransitionLineageError, KnownGoodTransitionLineageId,
};
use crate::witness::TargetRealizationId;

pub const CRASH_RECONCILIATION_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-crash-reconciliation-record-v1";

const RECONCILIATION_DOMAIN: &[u8] = b"symthaea.continuity.crash-reconciliation.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CrashReconciliationId([u8; 32]);

impl CrashReconciliationId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// What independent observation established about the physical realization after
/// an A -> B attempt became durable but no trustworthy result settled the attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrashReconciliationClassificationV1 {
    /// The exact active known-good source A is observed. This does not yet prove A
    /// is healthy; source-health and distributed-health requalification remain due.
    SourceKnownGoodObserved,
    /// The exact intended target B is observed. Target health must still qualify.
    TargetObserved,
    /// A different exact known realization C was observed. Automatic continuation
    /// is forbidden until explicit policy/manual reconciliation handles that world.
    OtherKnownRealizationObserved,
    /// The subject could not be reached by the qualified observer.
    Unreachable,
    /// The observer could not establish an exact physical state.
    Unknown,
}

impl CrashReconciliationClassificationV1 {
    fn tag(self) -> u8 {
        match self {
            Self::SourceKnownGoodObserved => 1,
            Self::TargetObserved => 2,
            Self::OtherKnownRealizationObserved => 3,
            Self::Unreachable => 4,
            Self::Unknown => 5,
        }
    }

    /// Describes the next proof boundary. This is not an action authorization and
    /// deliberately contains no automatic retry state.
    pub fn next_proof(self) -> CrashReconciliationNextProofV1 {
        match self {
            Self::SourceKnownGoodObserved => CrashReconciliationNextProofV1::SourceHealth,
            Self::TargetObserved => CrashReconciliationNextProofV1::TargetHealth,
            Self::OtherKnownRealizationObserved => {
                CrashReconciliationNextProofV1::ExplicitIntervention
            }
            Self::Unreachable | Self::Unknown => {
                CrashReconciliationNextProofV1::FreshPhysicalObservation
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashReconciliationNextProofV1 {
    SourceHealth,
    TargetHealth,
    ExplicitIntervention,
    FreshPhysicalObservation,
}

/// Persistent description of one exact crash reconciliation classification.
/// Serialization preserves audit evidence only; it cannot recreate the qualified
/// observation or grant any physical capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashReconciliationRecordV1 {
    schema_version: String,
    known_good_intent_id: KnownGoodExecutionIntentId,
    transition_lineage_id: KnownGoodTransitionLineageId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    journal_digest: ExecutionJournalDigest,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    observation_id: QualifiedPostExecutionObservationId,
    classification: CrashReconciliationClassificationV1,
    observed_realization_id: Option<TargetRealizationId>,
    observed_state_digest: Option<[u8; 32]>,
    reconciled_at_unix_ms: u64,
    reconciliation_id: CrashReconciliationId,
}

impl CrashReconciliationRecordV1 {
    pub fn validate(&self) -> Result<(), CrashReconciliationError> {
        if self.schema_version != CRASH_RECONCILIATION_RECORD_SCHEMA_V1 {
            return Err(CrashReconciliationError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.source_realization_id == self.target_realization_id {
            return Err(CrashReconciliationError::SourceEqualsTarget);
        }
        if self.reconciled_at_unix_ms == 0 {
            return Err(CrashReconciliationError::ZeroReconciliationTime);
        }
        validate_classification_material(
            self.classification,
            self.source_realization_id,
            self.target_realization_id,
            self.observed_realization_id,
            self.observed_state_digest,
        )?;
        let expected = CrashReconciliationId(hash_reconciliation(
            self.known_good_intent_id,
            self.transition_lineage_id,
            self.attempt_id,
            self.subject_id,
            self.source_realization_id,
            self.target_realization_id,
            self.distributed_context_id,
            self.journal_digest,
            self.journal_anchor_id,
            self.observation_id,
            self.classification,
            self.observed_realization_id,
            self.observed_state_digest,
            self.reconciled_at_unix_ms,
        ));
        if expected != self.reconciliation_id {
            return Err(CrashReconciliationError::ReconciliationIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> CrashReconciliationId {
        self.reconciliation_id
    }

    pub fn attempt_id(&self) -> ExecutionAttemptId {
        self.attempt_id
    }

    pub fn classification(&self) -> CrashReconciliationClassificationV1 {
        self.classification
    }

    pub fn observed_realization_id(&self) -> Option<TargetRealizationId> {
        self.observed_realization_id
    }

    pub fn next_proof(&self) -> CrashReconciliationNextProofV1 {
        self.classification.next_proof()
    }

    pub fn reconciled_at_unix_ms(&self) -> u64 {
        self.reconciled_at_unix_ms
    }
}

/// Non-Serde reconciliation proof rebound to the exact live independent observation.
/// It classifies the current realization only. It cannot clear health obligations,
/// recreate execution authority, or promote/recover Last Known Good.
#[derive(Debug, Clone)]
pub struct QualifiedCrashReconciliationV1 {
    record: CrashReconciliationRecordV1,
}

impl QualifiedCrashReconciliationV1 {
    pub fn qualify(
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        journal: &ReconstructedExecutionJournalV1,
        intent_anchor: &QualifiedExecutionJournalAnchorV1,
        observation: &QualifiedPostExecutionObservationV1,
    ) -> Result<Self, CrashReconciliationError> {
        intent.validate()?;
        let execution_intent = intent.execution_intent();
        let lineage = intent.lineage();
        let attempt_id = execution_intent.id();
        let eligibility_id = execution_intent.trusted_eligibility_id();

        let entry = journal
            .entry(attempt_id)
            .ok_or(CrashReconciliationError::IntentMissingFromJournal { attempt_id })?;
        if entry.trusted_eligibility_id() != eligibility_id
            || journal.attempt_for_eligibility(eligibility_id) != Some(attempt_id)
            || !journal.eligibility_is_spent(eligibility_id)
        {
            return Err(CrashReconciliationError::JournalEligibilityMismatch {
                attempt_id,
            });
        }
        if entry.receipt_id().is_some()
            || entry.disposition() != JournalAttemptDispositionV1::AwaitingReconciliation
        {
            return Err(CrashReconciliationError::AttemptNotPendingReconciliation {
                attempt_id,
            });
        }

        if intent_anchor.journal_digest() != journal.digest()
            || intent_anchor.journal_entry_count() != journal.len() as u64
            || intent_anchor.subject_id() != execution_intent.subject_id()
            || intent_anchor.trusted_epoch_id() != execution_intent.trusted_epoch_id()
        {
            return Err(CrashReconciliationError::AnchorDoesNotCoverPendingJournal);
        }

        if observation.attempt_id() != attempt_id
            || observation.subject_id() != execution_intent.subject_id()
            || observation.expected_target_realization_id()
                != execution_intent.target_realization_id()
            || observation.distributed_context_id()
                != execution_intent.distributed_context_id()
        {
            return Err(CrashReconciliationError::ObservationIntentMismatch);
        }
        if observation.qualified_at_unix_ms() <= lineage.commit_time_unix_ms()
            || observation.qualified_at_unix_ms() < intent_anchor.anchored_at_unix_ms()
        {
            return Err(CrashReconciliationError::ObservationPredatesReconciliationWorld);
        }

        let classification = classify_observation(lineage.source_realization_id(), observation)?;
        let observed_realization_id = observation.observed_realization_id();
        let observed_state_digest = observation.observed_state_digest();
        let reconciled_at_unix_ms = observation.qualified_at_unix_ms();
        validate_classification_material(
            classification,
            lineage.source_realization_id(),
            lineage.target_realization_id(),
            observed_realization_id,
            observed_state_digest,
        )?;

        let reconciliation_id = CrashReconciliationId(hash_reconciliation(
            intent.id(),
            lineage.id(),
            attempt_id,
            lineage.subject_id(),
            lineage.source_realization_id(),
            lineage.target_realization_id(),
            lineage.distributed_context_id(),
            journal.digest(),
            intent_anchor.id(),
            observation.id(),
            classification,
            observed_realization_id,
            observed_state_digest,
            reconciled_at_unix_ms,
        ));
        let record = CrashReconciliationRecordV1 {
            schema_version: CRASH_RECONCILIATION_RECORD_SCHEMA_V1.to_owned(),
            known_good_intent_id: intent.id(),
            transition_lineage_id: lineage.id(),
            attempt_id,
            subject_id: lineage.subject_id(),
            source_realization_id: lineage.source_realization_id(),
            target_realization_id: lineage.target_realization_id(),
            distributed_context_id: lineage.distributed_context_id(),
            journal_digest: journal.digest(),
            journal_anchor_id: intent_anchor.id(),
            observation_id: observation.id(),
            classification,
            observed_realization_id,
            observed_state_digest,
            reconciled_at_unix_ms,
            reconciliation_id,
        };
        record.validate()?;
        Ok(Self { record })
    }

    /// Rebind persisted audit state to the exact current intent/journal/anchor and
    /// independently qualified observation. A record self-hash is not sufficient.
    pub fn rebind(
        record: CrashReconciliationRecordV1,
        intent: &KnownGoodBoundExecutionAttemptIntentV1,
        journal: &ReconstructedExecutionJournalV1,
        intent_anchor: &QualifiedExecutionJournalAnchorV1,
        observation: &QualifiedPostExecutionObservationV1,
    ) -> Result<Self, CrashReconciliationError> {
        record.validate()?;
        let fresh = Self::qualify(intent, journal, intent_anchor, observation)?;
        if fresh.record != record {
            return Err(CrashReconciliationError::ReconciliationLineageMismatch);
        }
        Ok(fresh)
    }

    pub fn id(&self) -> CrashReconciliationId {
        self.record.id()
    }

    pub fn classification(&self) -> CrashReconciliationClassificationV1 {
        self.record.classification()
    }

    pub fn next_proof(&self) -> CrashReconciliationNextProofV1 {
        self.record.next_proof()
    }

    pub fn record(&self) -> &CrashReconciliationRecordV1 {
        &self.record
    }
}

fn classify_observation(
    source_realization_id: TargetRealizationId,
    observation: &QualifiedPostExecutionObservationV1,
) -> Result<CrashReconciliationClassificationV1, CrashReconciliationError> {
    match observation.observed_state() {
        PostExecutionObservedStateV1::ExpectedTargetObserved => {
            Ok(CrashReconciliationClassificationV1::TargetObserved)
        }
        PostExecutionObservedStateV1::DifferentKnownRealization => {
            let observed = observation
                .observed_realization_id()
                .ok_or(CrashReconciliationError::KnownStateMissingRealization)?;
            if observed == observation.expected_target_realization_id() {
                return Err(CrashReconciliationError::DifferentStateClaimsTarget);
            }
            if observed == source_realization_id {
                Ok(CrashReconciliationClassificationV1::SourceKnownGoodObserved)
            } else {
                Ok(CrashReconciliationClassificationV1::OtherKnownRealizationObserved)
            }
        }
        PostExecutionObservedStateV1::Unreachable => {
            Ok(CrashReconciliationClassificationV1::Unreachable)
        }
        PostExecutionObservedStateV1::Unknown => {
            Ok(CrashReconciliationClassificationV1::Unknown)
        }
    }
}

fn validate_classification_material(
    classification: CrashReconciliationClassificationV1,
    source: TargetRealizationId,
    target: TargetRealizationId,
    observed_realization: Option<TargetRealizationId>,
    observed_state_digest: Option<[u8; 32]>,
) -> Result<(), CrashReconciliationError> {
    match classification {
        CrashReconciliationClassificationV1::SourceKnownGoodObserved => {
            if observed_realization != Some(source) {
                return Err(CrashReconciliationError::SourceClassificationMismatch);
            }
            require_known_digest(observed_state_digest)?;
        }
        CrashReconciliationClassificationV1::TargetObserved => {
            if observed_realization != Some(target) {
                return Err(CrashReconciliationError::TargetClassificationMismatch);
            }
            require_known_digest(observed_state_digest)?;
        }
        CrashReconciliationClassificationV1::OtherKnownRealizationObserved => {
            let Some(observed) = observed_realization else {
                return Err(CrashReconciliationError::KnownStateMissingRealization);
            };
            if observed == source || observed == target {
                return Err(CrashReconciliationError::OtherClassificationMismatch);
            }
            require_known_digest(observed_state_digest)?;
        }
        CrashReconciliationClassificationV1::Unreachable
        | CrashReconciliationClassificationV1::Unknown => {
            if observed_realization.is_some() || observed_state_digest.is_some() {
                return Err(CrashReconciliationError::UnknownStateFabricatesIdentity);
            }
        }
    }
    Ok(())
}

fn require_known_digest(digest: Option<[u8; 32]>) -> Result<(), CrashReconciliationError> {
    if digest.is_none() || digest == Some([0; 32]) {
        return Err(CrashReconciliationError::KnownStateMissingDigest);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CrashReconciliationError {
    #[error(transparent)]
    Lineage(#[from] KnownGoodTransitionLineageError),
    #[error("unsupported crash reconciliation record schema: {0}")]
    UnsupportedSchema(String),
    #[error("crash reconciliation source and target realizations must differ")]
    SourceEqualsTarget,
    #[error("crash reconciliation time must be non-zero")]
    ZeroReconciliationTime,
    #[error("durable execution intent {attempt_id:?} is absent from reconstructed journal")]
    IntentMissingFromJournal { attempt_id: ExecutionAttemptId },
    #[error("journal spent-eligibility mapping does not match attempt {attempt_id:?}")]
    JournalEligibilityMismatch { attempt_id: ExecutionAttemptId },
    #[error("attempt {attempt_id:?} is not an intent-only pending reconciliation world")]
    AttemptNotPendingReconciliation { attempt_id: ExecutionAttemptId },
    #[error("qualified anchor does not cover the exact pending journal/subject/trusted-epoch world")]
    AnchorDoesNotCoverPendingJournal,
    #[error("qualified post-execution observation does not bind the exact durable A -> B attempt")]
    ObservationIntentMismatch,
    #[error("qualified observation does not occur after the durable anchored transition world")]
    ObservationPredatesReconciliationWorld,
    #[error("known physical state is missing an exact realization identity")]
    KnownStateMissingRealization,
    #[error("known physical state is missing a non-zero state digest")]
    KnownStateMissingDigest,
    #[error("different-known-realization state illegally names the expected target")]
    DifferentStateClaimsTarget,
    #[error("source-known-good classification does not identify exact source A")]
    SourceClassificationMismatch,
    #[error("target-observed classification does not identify exact target B")]
    TargetClassificationMismatch,
    #[error("other-known-realization classification aliases source A or target B")]
    OtherClassificationMismatch,
    #[error("unknown/unreachable classification fabricates a known realization or state digest")]
    UnknownStateFabricatesIdentity,
    #[error("crash reconciliation identity does not match canonical fields")]
    ReconciliationIdentityMismatch,
    #[error("persisted crash reconciliation does not match exact live proof lineage")]
    ReconciliationLineageMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_reconciliation(
    known_good_intent_id: KnownGoodExecutionIntentId,
    transition_lineage_id: KnownGoodTransitionLineageId,
    attempt_id: ExecutionAttemptId,
    subject_id: ContinuitySubjectId,
    source_realization_id: TargetRealizationId,
    target_realization_id: TargetRealizationId,
    distributed_context_id: DistributedStateContextId,
    journal_digest: ExecutionJournalDigest,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
    observation_id: QualifiedPostExecutionObservationId,
    classification: CrashReconciliationClassificationV1,
    observed_realization_id: Option<TargetRealizationId>,
    observed_state_digest: Option<[u8; 32]>,
    reconciled_at_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECONCILIATION_DOMAIN);
    hasher.update(known_good_intent_id.as_bytes());
    hasher.update(transition_lineage_id.as_bytes());
    hasher.update(attempt_id.as_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(source_realization_id.as_bytes());
    hasher.update(target_realization_id.as_bytes());
    hasher.update(distributed_context_id.as_bytes());
    hasher.update(journal_digest.as_bytes());
    hasher.update(journal_anchor_id.as_bytes());
    hasher.update(observation_id.as_bytes());
    hasher.update(&[classification.tag()]);
    match observed_realization_id {
        Some(id) => {
            hasher.update(&[1]);
            hasher.update(id.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match observed_state_digest {
        Some(digest) => {
            hasher.update(&[1]);
            hasher.update(&digest);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&reconciled_at_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_never_maps_to_retry() {
        assert_eq!(
            CrashReconciliationClassificationV1::Unknown.next_proof(),
            CrashReconciliationNextProofV1::FreshPhysicalObservation
        );
    }

    #[test]
    fn source_and_target_require_health_next() {
        assert_eq!(
            CrashReconciliationClassificationV1::SourceKnownGoodObserved.next_proof(),
            CrashReconciliationNextProofV1::SourceHealth
        );
        assert_eq!(
            CrashReconciliationClassificationV1::TargetObserved.next_proof(),
            CrashReconciliationNextProofV1::TargetHealth
        );
    }
}
