// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Type-level sequencing for known-good-bound physical execution.
//!
//! A physical adapter must never receive an executable attempt merely because a
//! one-use capability was minted. The exact A -> B intent must first be durably
//! visible in the reconstructed execution journal and that updated journal must be
//! committed by a newly qualified rollback-resistant anchor which directly extends
//! the exact anchor that preceded preparation.
//!
//! `PreparedIntent != DurableIntent != AnchoredIntent != ReadyForPhysicalExecution`.

use thiserror::Error;

use crate::execution_capability::{
    ExecutionAttemptId, ExecutionAttemptOutcomeV1, ExecutionAttemptReceiptV1,
    ExecutionBackendId, ExecutionBackendProfileV1, ExecutionCapabilityError,
    ExecutionEpochAnchorModeV1, ExecutionSessionV1, PreparedExecutionAttemptV1,
    mint_one_use_execution_capability,
};
use crate::execution_journal::{JournalAttemptDispositionV1, ReconstructedExecutionJournalV1};
use crate::execution_journal_anchor::{
    ExecutionJournalAnchorProfileId, QualifiedExecutionJournalAnchorId,
    QualifiedExecutionJournalAnchorV1,
};
use crate::scope::ContinuitySubjectId;
use crate::transition_lineage::{
    KnownGoodBoundExecutionAttemptIntentV1, KnownGoodBoundTrustedCommitEligibilityV1,
    KnownGoodExecutionIntentId, KnownGoodTransitionLineageError,
};
use crate::trusted_commit_epoch::QualifiedTrustedCommitEpochV1;
use crate::witness::TargetRealizationId;

/// Non-Serde, non-Clone state after one exact A -> B capability has been consumed
/// into an intent and prepared attempt, but before durability has been established.
///
/// The prepared attempt is intentionally private and has no accessor. The only way
/// to advance this value is to present the exact reconstructed journal plus a newly
/// qualified anchor that directly extends the exact predecessor anchor captured at
/// preparation time.
#[derive(Debug)]
pub struct PendingAnchoredKnownGoodExecutionV1 {
    intent: KnownGoodBoundExecutionAttemptIntentV1,
    prepared: PreparedExecutionAttemptV1,
    session: ExecutionSessionV1,
    predecessor_anchor_id: QualifiedExecutionJournalAnchorId,
    predecessor_anchor_profile_id: ExecutionJournalAnchorProfileId,
    predecessor_anchor_root_epoch: u64,
    predecessor_anchor_sequence: u64,
    predecessor_journal_entry_count: u64,
    predecessor_anchor_time_unix_ms: u64,
}

impl PendingAnchoredKnownGoodExecutionV1 {
    /// The only artifact exposed before the durability gate. Persist this exact
    /// known-good-bound intent (or an atomic superset containing it), reconstruct
    /// the journal, and qualify a fresh rollback-resistant anchor over that world.
    pub fn intent(&self) -> &KnownGoodBoundExecutionAttemptIntentV1 {
        &self.intent
    }

    pub fn attempt_id(&self) -> ExecutionAttemptId {
        self.intent.attempt_id()
    }

    pub fn predecessor_anchor_id(&self) -> QualifiedExecutionJournalAnchorId {
        self.predecessor_anchor_id
    }

    /// Consume the pending state only after the exact intent is durably represented
    /// in `journal` and `anchor` directly extends the captured predecessor anchor.
    ///
    /// A receipt must not already exist. If one does, the world is no longer the
    /// pre-mutation world this pending token represents and reconciliation is
    /// required instead of releasing the prepared attempt.
    pub fn release_after_durable_anchor(
        self,
        journal: &ReconstructedExecutionJournalV1,
        anchor: &QualifiedExecutionJournalAnchorV1,
    ) -> Result<ReadyKnownGoodExecutionAttemptV1, KnownGoodExecutionCoordinatorError> {
        self.intent.validate()?;

        let execution_intent = self.intent.execution_intent();
        let attempt_id = execution_intent.id();
        let eligibility_id = execution_intent.trusted_eligibility_id();
        let entry = journal
            .entry(attempt_id)
            .ok_or(KnownGoodExecutionCoordinatorError::IntentMissingFromJournal {
                attempt_id,
            })?;

        if entry.trusted_eligibility_id() != eligibility_id
            || journal.attempt_for_eligibility(eligibility_id) != Some(attempt_id)
            || !journal.eligibility_is_spent(eligibility_id)
        {
            return Err(KnownGoodExecutionCoordinatorError::JournalEligibilityMismatch {
                attempt_id,
            });
        }
        if entry.receipt_id().is_some()
            || entry.disposition() != JournalAttemptDispositionV1::AwaitingReconciliation
        {
            return Err(KnownGoodExecutionCoordinatorError::AttemptAlreadyHasResult {
                attempt_id,
            });
        }

        if anchor.journal_digest() != journal.digest()
            || anchor.journal_entry_count() != journal.len() as u64
        {
            return Err(KnownGoodExecutionCoordinatorError::AnchorDoesNotCoverJournal);
        }
        if anchor.subject_id() != execution_intent.subject_id()
            || anchor.trusted_epoch_id() != self.session.trusted_epoch_id()
        {
            return Err(KnownGoodExecutionCoordinatorError::AnchorExecutionContextMismatch);
        }
        if anchor.profile_id() != self.predecessor_anchor_profile_id
            || anchor.root_epoch() != self.predecessor_anchor_root_epoch
        {
            return Err(KnownGoodExecutionCoordinatorError::AnchorRootLineageMismatch);
        }
        let expected_sequence = self
            .predecessor_anchor_sequence
            .checked_add(1)
            .ok_or(KnownGoodExecutionCoordinatorError::AnchorSequenceOverflow)?;
        if anchor.anchor_sequence() != expected_sequence
            || anchor.predecessor_anchor_id() != Some(self.predecessor_anchor_id)
        {
            return Err(KnownGoodExecutionCoordinatorError::AnchorDoesNotExtendPredecessor);
        }
        if anchor.journal_entry_count() <= self.predecessor_journal_entry_count {
            return Err(KnownGoodExecutionCoordinatorError::JournalDidNotAdvance);
        }
        if anchor.anchored_at_unix_ms() < self.predecessor_anchor_time_unix_ms
            || anchor.anchored_at_unix_ms() != self.intent.lineage().commit_time_unix_ms()
        {
            return Err(KnownGoodExecutionCoordinatorError::AnchorTimeMismatch);
        }
        if self.prepared.id() != attempt_id
            || self.prepared.backend_id() != execution_intent.backend_id()
            || self.prepared.subject_id() != execution_intent.subject_id()
            || self.prepared.target_realization_id() != execution_intent.target_realization_id()
            || self.prepared.distributed_context_id() != execution_intent.distributed_context_id()
        {
            return Err(KnownGoodExecutionCoordinatorError::PreparedAttemptLineageMismatch);
        }

        Ok(ReadyKnownGoodExecutionAttemptV1 {
            intent: self.intent,
            prepared: self.prepared,
            session: self.session,
            journal_anchor_id: anchor.id(),
        })
    }
}

/// The first externally useful execution boundary for an adapter: an exact A -> B
/// attempt whose durable intent has already become spent anti-replay state and whose
/// journal world has been protected by a fresh rollback-resistant anchor.
///
/// This value is non-Serde and non-Clone. A backend may inspect its exact identifiers
/// to perform the physical operation, then must consume it to produce the backend
/// receipt. The receipt remains audit material rather than independent proof of the
/// resulting physical state.
#[derive(Debug)]
pub struct ReadyKnownGoodExecutionAttemptV1 {
    intent: KnownGoodBoundExecutionAttemptIntentV1,
    prepared: PreparedExecutionAttemptV1,
    session: ExecutionSessionV1,
    journal_anchor_id: QualifiedExecutionJournalAnchorId,
}

impl ReadyKnownGoodExecutionAttemptV1 {
    pub fn attempt_id(&self) -> ExecutionAttemptId {
        self.prepared.id()
    }

    pub fn known_good_intent_id(&self) -> KnownGoodExecutionIntentId {
        self.intent.id()
    }

    pub fn journal_anchor_id(&self) -> QualifiedExecutionJournalAnchorId {
        self.journal_anchor_id
    }

    pub fn backend_id(&self) -> ExecutionBackendId {
        self.prepared.backend_id()
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.prepared.subject_id()
    }

    pub fn source_realization_id(&self) -> TargetRealizationId {
        self.intent.lineage().source_realization_id()
    }

    pub fn target_realization_id(&self) -> TargetRealizationId {
        self.prepared.target_realization_id()
    }

    pub fn intent(&self) -> &KnownGoodBoundExecutionAttemptIntentV1 {
        &self.intent
    }

    /// Consume the ready execution token after the physical adapter has attempted
    /// the mutation. The backend result remains a claim; independent post-execution
    /// observation and health qualification are still required downstream.
    pub fn finish(
        self,
        outcome: ExecutionAttemptOutcomeV1,
        backend_evidence_digest: [u8; 32],
        result_digest: [u8; 32],
    ) -> Result<ExecutionAttemptReceiptV1, KnownGoodExecutionCoordinatorError> {
        Ok(self.prepared.finish(
            &self.session,
            outcome,
            backend_evidence_digest,
            result_digest,
        )?)
    }
}

/// Consume one exact active-known-good-bound eligibility into a pending execution
/// state. The caller receives the durable intent but not the prepared physical
/// attempt until `release_after_durable_anchor()` proves the crash boundary.
#[allow(clippy::too_many_arguments)]
pub fn prepare_known_good_execution(
    bound: KnownGoodBoundTrustedCommitEligibilityV1,
    current_epoch: &QualifiedTrustedCommitEpochV1,
    previous_epoch: Option<&QualifiedTrustedCommitEpochV1>,
    epoch_anchor_mode: ExecutionEpochAnchorModeV1,
    predecessor_journal_anchor: &QualifiedExecutionJournalAnchorV1,
    backend: &ExecutionBackendProfileV1,
    session_generation: u64,
    session_nonce: [u8; 32],
) -> Result<PendingAnchoredKnownGoodExecutionV1, KnownGoodExecutionCoordinatorError> {
    let (lineage, eligibility) = bound.into_parts();
    lineage.validate()?;

    if predecessor_journal_anchor.subject_id() != lineage.subject_id() {
        return Err(KnownGoodExecutionCoordinatorError::PredecessorAnchorSubjectMismatch);
    }
    if predecessor_journal_anchor.anchored_at_unix_ms() > lineage.commit_time_unix_ms() {
        return Err(KnownGoodExecutionCoordinatorError::PredecessorAnchorFromFuture);
    }

    let mut session = ExecutionSessionV1::open(
        &eligibility,
        current_epoch,
        previous_epoch,
        epoch_anchor_mode,
        backend,
        session_generation,
        session_nonce,
    )?;
    let capability = mint_one_use_execution_capability(&mut session, eligibility)?;
    let (execution_intent, prepared) = capability.begin_attempt(&session)?;
    let intent = KnownGoodBoundExecutionAttemptIntentV1::bind(lineage, execution_intent)?;

    if prepared.id() != intent.attempt_id() {
        return Err(KnownGoodExecutionCoordinatorError::PreparedAttemptLineageMismatch);
    }

    Ok(PendingAnchoredKnownGoodExecutionV1 {
        intent,
        prepared,
        session,
        predecessor_anchor_id: predecessor_journal_anchor.id(),
        predecessor_anchor_profile_id: predecessor_journal_anchor.profile_id(),
        predecessor_anchor_root_epoch: predecessor_journal_anchor.root_epoch(),
        predecessor_anchor_sequence: predecessor_journal_anchor.anchor_sequence(),
        predecessor_journal_entry_count: predecessor_journal_anchor.journal_entry_count(),
        predecessor_anchor_time_unix_ms: predecessor_journal_anchor.anchored_at_unix_ms(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KnownGoodExecutionCoordinatorError {
    #[error(transparent)]
    Execution(#[from] ExecutionCapabilityError),
    #[error(transparent)]
    Lineage(#[from] KnownGoodTransitionLineageError),
    #[error("preparation journal anchor belongs to another continuity subject")]
    PredecessorAnchorSubjectMismatch,
    #[error("preparation journal anchor is temporally after the A -> B commit")]
    PredecessorAnchorFromFuture,
    #[error("durable execution intent {attempt_id:?} is absent from reconstructed journal")]
    IntentMissingFromJournal { attempt_id: ExecutionAttemptId },
    #[error("journal spent-eligibility mapping does not match attempt {attempt_id:?}")]
    JournalEligibilityMismatch { attempt_id: ExecutionAttemptId },
    #[error("attempt {attempt_id:?} already has a result; reconcile instead of releasing execution")]
    AttemptAlreadyHasResult { attempt_id: ExecutionAttemptId },
    #[error("qualified journal anchor does not cover the exact reconstructed journal")]
    AnchorDoesNotCoverJournal,
    #[error("qualified journal anchor does not match the exact execution subject/trusted epoch")]
    AnchorExecutionContextMismatch,
    #[error("qualified journal anchor changed anchor profile/root lineage")]
    AnchorRootLineageMismatch,
    #[error("qualified journal anchor does not directly extend the preparation anchor")]
    AnchorDoesNotExtendPredecessor,
    #[error("journal entry count did not advance across the preparation anchor")]
    JournalDidNotAdvance,
    #[error("journal anchor sequence overflow")]
    AnchorSequenceOverflow,
    #[error("journal anchor time does not match the exact commit epoch or moved backwards")]
    AnchorTimeMismatch,
    #[error("private prepared attempt does not match the exact durable A -> B intent")]
    PreparedAttemptLineageMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinator_error_domains_are_distinct() {
        assert_ne!(
            "intent_missing_from_journal",
            "anchor_does_not_extend_predecessor"
        );
    }
}
