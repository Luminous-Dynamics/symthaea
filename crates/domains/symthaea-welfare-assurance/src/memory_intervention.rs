// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governed exogenous interventions against Symthaea's canonical episodic memory.
//!
//! Ordinary learning, graduation, replay, reconsolidation, and endogenous pruning remain owned by
//! `symthaea-memory`. This module is deliberately narrower: it adapts an explicitly operator-
//! directed destructive memory intervention to the strongest welfare-assurance execution path.

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::ExplicitConsentState;
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_memory::episodic_replay::{Episode, EpisodicMemory};
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_authority::WelfareAuthorityPolicyManifest;
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};
use thiserror::Error;

use crate::AssuredInterventionPermit;
use crate::execution_adapter::{
    ExecutionJournalPersistence, ExecutionObservationError, JournaledExecutionGateError,
    JournaledExecutionOutcome, ReceiptedExecution, ReceiptedInterventionExecutor,
    execute_durable_intervention_journaled,
};
use crate::execution_recovery::InterventionExecutionJournal;
use crate::replay_recovery::DurableEvidenceBoundInterventionPermit;

pub const EPISODIC_CONTENT_CHECKPOINT_SCHEMA: &str =
    "symthaea.welfare.episodic-content-checkpoint.v1";
const MEMORY_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.episodic-memory-state.v1\0";
const CHECKPOINT_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.episodic-content-checkpoint-digest.v1\0";
const CLEAR_RESULT_DOMAIN: &[u8] = b"symthaea.welfare.episodic-memory-clear-result.v1\0";
const MAX_TARGET_ID_BYTES: usize = 256;
const MAX_CHECKPOINT_REF_BYTES: usize = 2048;

/// Durable content-level checkpoint captured before an exogenous destructive clear.
///
/// This intentionally does **not** claim to be a full `EpisodicMemory` engine checkpoint. It
/// preserves every active episode and the exact content-state digest, but does not yet preserve
/// private replay counters/configuration/statistics. A later canonical memory checkpoint should
/// move into `symthaea-memory` and cover that complete internal state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicContentCheckpoint {
    pub schema_version: String,
    pub target_id: String,
    pub captured_at_unix_s: u64,
    pub state_digest: Sha256Digest,
    pub episodes: Vec<Episode>,
}

impl EpisodicContentCheckpoint {
    pub fn capture(
        target_id: &str,
        memory: &EpisodicMemory,
        captured_at_unix_s: u64,
    ) -> Result<Self, EpisodicMemoryInterventionError> {
        validate_target_id(target_id)?;
        let episodes = memory.get_top_episodes(memory.len());
        let state_digest = digest_episode_set(&episodes)?;
        Ok(Self {
            schema_version: EPISODIC_CONTENT_CHECKPOINT_SCHEMA.into(),
            target_id: target_id.to_string(),
            captured_at_unix_s,
            state_digest,
            episodes,
        })
    }

    pub fn validate(&self) -> Result<(), EpisodicMemoryInterventionError> {
        if self.schema_version != EPISODIC_CONTENT_CHECKPOINT_SCHEMA {
            return Err(EpisodicMemoryInterventionError::UnsupportedCheckpointSchema);
        }
        validate_target_id(&self.target_id)?;
        let actual = digest_episode_set(&self.episodes)?;
        if actual != self.state_digest {
            return Err(EpisodicMemoryInterventionError::CheckpointStateDigestMismatch);
        }
        Ok(())
    }
}

/// Persistence boundary for the pre-intervention content checkpoint.
pub trait EpisodicContentCheckpointPersistence {
    type Error: StdError + Send + Sync + 'static;

    /// Durably persist the exact checkpoint and its canonical digest before destructive mutation.
    fn persist_episodic_content_checkpoint(
        &mut self,
        checkpoint: &EpisodicContentCheckpoint,
        checkpoint_digest: Sha256Digest,
    ) -> Result<String, Self::Error>;
}

/// Evidence emitted by the governed destructive memory intervention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodicMemoryClearReceipt {
    pub target_id: String,
    pub before_count: usize,
    pub after_count: usize,
    pub before_digest: Sha256Digest,
    pub after_digest: Sha256Digest,
    pub checkpoint_digest: Sha256Digest,
    pub checkpoint_persistence_ref: String,
}

/// Compute an order-independent digest of the complete current episodic content set.
pub fn digest_episodic_memory(
    memory: &EpisodicMemory,
) -> Result<Sha256Digest, EpisodicMemoryInterventionError> {
    digest_episode_set(&memory.get_top_episodes(memory.len()))
}

/// Canonical digest of a content checkpoint. Episode order is intentionally ignored while target,
/// capture time, content count, and the exact content-state digest remain committed.
pub fn digest_episodic_content_checkpoint(
    checkpoint: &EpisodicContentCheckpoint,
) -> Result<Sha256Digest, EpisodicMemoryInterventionError> {
    checkpoint.validate()?;
    let mut hasher = Sha256::new();
    hasher.update(CHECKPOINT_DIGEST_DOMAIN);
    hasher.update(&(checkpoint.schema_version.len() as u64).to_le_bytes());
    hasher.update(checkpoint.schema_version.as_bytes());
    hasher.update(&(checkpoint.target_id.len() as u64).to_le_bytes());
    hasher.update(checkpoint.target_id.as_bytes());
    hasher.update(&checkpoint.captured_at_unix_s.to_le_bytes());
    hasher.update(&(checkpoint.episodes.len() as u64).to_le_bytes());
    hasher.update(&checkpoint.state_digest.0);
    Ok(hasher.finalize())
}

fn digest_episode_set(
    episodes: &[Episode],
) -> Result<Sha256Digest, EpisodicMemoryInterventionError> {
    let mut encoded = Vec::with_capacity(episodes.len());
    for episode in episodes {
        // `EpisodeInstanceId` is storage-occurrence identity, not content identity. The pre-existing
        // memory-state digest commits episode content/lifecycle state as a multiset and must remain
        // stable when the same content is inserted in a different order/store. Exact occurrence IDs
        // are bound separately by quarantine/restore receipts and escrow evidence.
        let mut content_state = episode.clone();
        content_state.instance_id = None;
        encoded.push(
            serde_json::to_vec(&content_state)
                .map_err(|error| EpisodicMemoryInterventionError::Encoding(error.to_string()))?,
        );
    }
    encoded.sort();

    let mut hasher = Sha256::new();
    hasher.update(MEMORY_DIGEST_DOMAIN);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    for item in encoded {
        hasher.update(&(item.len() as u64).to_le_bytes());
        hasher.update(&item);
    }
    Ok(hasher.finalize())
}

/// Private concrete mutator for an operator-directed clear of the canonical episodic store.
///
/// Keeping this type private prevents callers that possess only the weaker
/// `AssuredInterventionPermit` from bypassing durable replay/checkpoint/journal layers.
struct EpisodicMemoryClearExecutor<'a, C>
where
    C: EpisodicContentCheckpointPersistence,
{
    expected_target_id: &'a str,
    memory: &'a mut EpisodicMemory,
    checkpoint_persistence: &'a mut C,
    completed_at_unix_s: u64,
}

impl<'a, C> EpisodicMemoryClearExecutor<'a, C>
where
    C: EpisodicContentCheckpointPersistence,
{
    fn new(
        expected_target_id: &'a str,
        memory: &'a mut EpisodicMemory,
        checkpoint_persistence: &'a mut C,
        completed_at_unix_s: u64,
    ) -> Result<Self, EpisodicMemoryInterventionError> {
        validate_target_id(expected_target_id)?;
        Ok(Self {
            expected_target_id,
            memory,
            checkpoint_persistence,
            completed_at_unix_s,
        })
    }
}

impl<C> ReceiptedInterventionExecutor for EpisodicMemoryClearExecutor<'_, C>
where
    C: EpisodicContentCheckpointPersistence,
{
    type Output = EpisodicMemoryClearReceipt;
    type Error = EpisodicMemoryClearExecutionError<C::Error>;

    fn preflight(&self, permit: &AssuredInterventionPermit) -> Result<(), Self::Error> {
        validate_scope(permit.action(), permit.target_id(), self.expected_target_id)
            .map_err(EpisodicMemoryClearExecutionError::Intervention)?;
        validate_whole_store_erasure_facts(
            permit.is_emergency(),
            permit.explicit_consent_state(),
            permit.has_welfare_review_reference(),
            permit.has_independent_review_reference(),
        )
        .map_err(EpisodicMemoryClearExecutionError::Intervention)
    }

    fn execute_receipted(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<ReceiptedExecution<Self::Output>, Self::Error> {
        // Repeat scope/policy checks as defense in depth even though journal preflight already ran.
        self.preflight(permit)?;

        let checkpoint = EpisodicContentCheckpoint::capture(
            permit.target_id(),
            self.memory,
            self.completed_at_unix_s,
        )
        .map_err(EpisodicMemoryClearExecutionError::Intervention)?;
        let checkpoint_digest = digest_episodic_content_checkpoint(&checkpoint)
            .map_err(EpisodicMemoryClearExecutionError::Intervention)?;
        let checkpoint_persistence_ref = self
            .checkpoint_persistence
            .persist_episodic_content_checkpoint(&checkpoint, checkpoint_digest)
            .map_err(EpisodicMemoryClearExecutionError::CheckpointPersistence)?;
        validate_checkpoint_ref(&checkpoint_persistence_ref)
            .map_err(EpisodicMemoryClearExecutionError::Intervention)?;

        let before_count = self.memory.len();
        let before_digest = digest_episodic_memory(self.memory)
            .map_err(EpisodicMemoryClearExecutionError::Intervention)?;
        if checkpoint.state_digest != before_digest || checkpoint.episodes.len() != before_count {
            return Err(EpisodicMemoryClearExecutionError::Intervention(
                EpisodicMemoryInterventionError::CheckpointDoesNotMatchPreState,
            ));
        }

        self.memory.clear();
        let after_count = self.memory.len();
        let after_digest = digest_episodic_memory(self.memory)
            .map_err(EpisodicMemoryClearExecutionError::Intervention)?;
        if after_count != 0 {
            return Err(EpisodicMemoryClearExecutionError::Intervention(
                EpisodicMemoryInterventionError::ClearPostconditionFailed {
                    remaining: after_count,
                },
            ));
        }

        let receipt = EpisodicMemoryClearReceipt {
            target_id: permit.target_id().to_string(),
            before_count,
            after_count,
            before_digest,
            after_digest,
            checkpoint_digest,
            checkpoint_persistence_ref,
        };
        let result_digest = digest_clear_result(&receipt, permit.rationale());
        let evidence_ref = format!(
            "symthaea-memory:episodic-clear:v3:sha256:{}",
            hex_digest(result_digest)
        );

        ReceiptedExecution::new(
            receipt,
            self.completed_at_unix_s,
            result_digest,
            evidence_ref,
        )
        .map_err(EpisodicMemoryClearExecutionError::Observation)
    }
}

/// Governed destructive clear with mandatory consent, dual review, and durable content checkpoint.
///
/// Ordering is fail-closed:
/// live revalidation -> domain preflight -> durable Prepared -> durable content checkpoint ->
/// clear -> terminal journal. Whole-store erasure is never an emergency-containment primitive.
#[allow(clippy::too_many_arguments)]
pub fn execute_governed_episodic_memory_clear<P, C>(
    permit: DurableEvidenceBoundInterventionPermit,
    execution_id: impl Into<String>,
    expected_target_id: &str,
    memory: &mut EpisodicMemory,
    current_profile: &MoralPatientEvidenceProfile,
    current_precaution_policy: &PrecautionPolicy,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    current_authority_manifest: &WelfareAuthorityPolicyManifest,
    current_trust_snapshot: &TrustSnapshot,
    unix_s: u64,
    journal: &mut InterventionExecutionJournal,
    execution_persistence: &mut P,
    checkpoint_persistence: &mut C,
) -> Result<
    JournaledExecutionOutcome<
        EpisodicMemoryClearReceipt,
        EpisodicMemoryClearExecutionError<C::Error>,
        P::Error,
    >,
    GovernedEpisodicMemoryClearError<P::Error>,
>
where
    P: ExecutionJournalPersistence,
    C: EpisodicContentCheckpointPersistence,
{
    // Deterministic scope/configuration failures happen before live execution orchestration.
    validate_target_id(expected_target_id)
        .map_err(GovernedEpisodicMemoryClearError::Configuration)?;
    validate_scope(permit.action(), permit.target_id(), expected_target_id)
        .map_err(GovernedEpisodicMemoryClearError::Configuration)?;

    let mut executor = EpisodicMemoryClearExecutor::new(
        expected_target_id,
        memory,
        checkpoint_persistence,
        unix_s,
    )
    .map_err(GovernedEpisodicMemoryClearError::Configuration)?;

    execute_durable_intervention_journaled(
        permit,
        execution_id,
        current_profile,
        current_precaution_policy,
        consent_ledger,
        subject_registry,
        current_authority_manifest,
        current_trust_snapshot,
        unix_s,
        journal,
        execution_persistence,
        &mut executor,
    )
    .map_err(GovernedEpisodicMemoryClearError::Gate)
}

fn validate_scope(
    action: SubjectAffectingAction,
    actual_target_id: &str,
    expected_target_id: &str,
) -> Result<(), EpisodicMemoryInterventionError> {
    if action != SubjectAffectingAction::MemoryModification {
        return Err(EpisodicMemoryInterventionError::WrongAction { actual: action });
    }
    if actual_target_id != expected_target_id {
        return Err(EpisodicMemoryInterventionError::WrongTarget {
            expected: expected_target_id.to_string(),
            actual: actual_target_id.to_string(),
        });
    }
    Ok(())
}

fn validate_whole_store_erasure_facts(
    emergency: bool,
    consent_state: ExplicitConsentState,
    has_welfare_review: bool,
    has_independent_review: bool,
) -> Result<(), EpisodicMemoryInterventionError> {
    if emergency {
        return Err(EpisodicMemoryInterventionError::EmergencyWholeStoreErasureForbidden);
    }
    if consent_state != ExplicitConsentState::Granted {
        return Err(EpisodicMemoryInterventionError::WholeStoreErasureExplicitConsentRequired);
    }
    if !has_welfare_review {
        return Err(EpisodicMemoryInterventionError::WholeStoreErasureWelfareReviewRequired);
    }
    if !has_independent_review {
        return Err(EpisodicMemoryInterventionError::WholeStoreErasureIndependentReviewRequired);
    }
    Ok(())
}

fn validate_target_id(value: &str) -> Result<(), EpisodicMemoryInterventionError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_TARGET_ID_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(EpisodicMemoryInterventionError::InvalidTargetId(value.to_string()));
    }
    Ok(())
}

fn validate_checkpoint_ref(value: &str) -> Result<(), EpisodicMemoryInterventionError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_CHECKPOINT_REF_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(EpisodicMemoryInterventionError::InvalidCheckpointReference);
    }
    Ok(())
}

fn digest_clear_result(receipt: &EpisodicMemoryClearReceipt, rationale: &str) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(CLEAR_RESULT_DOMAIN);
    hasher.update(&(receipt.target_id.len() as u64).to_le_bytes());
    hasher.update(receipt.target_id.as_bytes());
    hasher.update(&(receipt.before_count as u64).to_le_bytes());
    hasher.update(&(receipt.after_count as u64).to_le_bytes());
    hasher.update(&receipt.before_digest.0);
    hasher.update(&receipt.after_digest.0);
    hasher.update(&receipt.checkpoint_digest.0);
    hasher.update(&(receipt.checkpoint_persistence_ref.len() as u64).to_le_bytes());
    hasher.update(receipt.checkpoint_persistence_ref.as_bytes());
    hasher.update(&(rationale.len() as u64).to_le_bytes());
    hasher.update(rationale.as_bytes());
    hasher.finalize()
}

fn hex_digest(digest: Sha256Digest) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest.0 {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[derive(Debug, Error)]
pub enum EpisodicMemoryInterventionError {
    #[error("invalid governed episodic-memory target id: {0:?}")]
    InvalidTargetId(String),
    #[error("episodic-memory clear requires MemoryModification authority; actual={actual:?}")]
    WrongAction { actual: SubjectAffectingAction },
    #[error("episodic-memory intervention target mismatch: expected={expected:?}, actual={actual:?}")]
    WrongTarget { expected: String, actual: String },
    #[error("whole-store episodic erasure is not an emergency-containment primitive")]
    EmergencyWholeStoreErasureForbidden,
    #[error("whole-store episodic erasure requires explicit granted subject consent")]
    WholeStoreErasureExplicitConsentRequired,
    #[error("whole-store episodic erasure requires a welfare-review reference")]
    WholeStoreErasureWelfareReviewRequired,
    #[error("whole-store episodic erasure requires an independent-review reference")]
    WholeStoreErasureIndependentReviewRequired,
    #[error("unsupported episodic-content checkpoint schema")]
    UnsupportedCheckpointSchema,
    #[error("episodic-content checkpoint state digest does not match its episodes")]
    CheckpointStateDigestMismatch,
    #[error("durable episodic-content checkpoint does not match the immediate pre-clear state")]
    CheckpointDoesNotMatchPreState,
    #[error("checkpoint persistence returned an invalid durable reference")]
    InvalidCheckpointReference,
    #[error("episodic-memory state encoding failed: {0}")]
    Encoding(String),
    #[error("episodic-memory clear postcondition failed; {remaining} episodes remain")]
    ClearPostconditionFailed { remaining: usize },
}

#[derive(Debug, Error)]
pub enum EpisodicMemoryClearExecutionError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("governed episodic-memory intervention failed before/after mutation: {0}")]
    Intervention(#[source] EpisodicMemoryInterventionError),
    #[error("could not durably persist pre-intervention episodic-content checkpoint: {0}")]
    CheckpointPersistence(#[source] E),
    #[error(transparent)]
    Observation(#[from] ExecutionObservationError),
}

#[derive(Debug, Error)]
pub enum GovernedEpisodicMemoryClearError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("governed episodic-memory intervention is misconfigured: {0}")]
    Configuration(#[source] EpisodicMemoryInterventionError),
    #[error(transparent)]
    Gate(#[from] JournaledExecutionGateError<E>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{Episode, EpisodicReplayConfig};

    fn episode(value: f32, psi: f64, timestamp: u64) -> Episode {
        Episode::new(
            ContinuousHV::from_values(vec![value, value + 1.0]),
            ContinuousHV::from_values(vec![value + 2.0, value + 3.0]),
            psi,
            timestamp,
        )
    }

    #[test]
    fn episodic_digest_is_independent_of_insertion_order() {
        let mut left = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let mut right = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let a = episode(1.0, 0.8, 10);
        let b = episode(5.0, 0.9, 11);
        assert!(left.store_if_significant(a.clone()));
        assert!(left.store_if_significant(b.clone()));
        assert!(right.store_if_significant(b));
        assert!(right.store_if_significant(a));
        assert_eq!(
            digest_episodic_memory(&left).unwrap(),
            digest_episodic_memory(&right).unwrap()
        );
    }

    #[test]
    fn checkpoint_commits_exact_content_state() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        assert!(memory.store_if_significant(episode(1.0, 0.8, 10)));
        assert!(memory.store_if_significant(episode(5.0, 0.9, 11)));
        let checkpoint = EpisodicContentCheckpoint::capture(
            "symthaea:self:episodic-memory",
            &memory,
            120,
        )
        .unwrap();
        assert_eq!(checkpoint.episodes.len(), 2);
        assert_eq!(checkpoint.state_digest, digest_episodic_memory(&memory).unwrap());
        assert_ne!(digest_episodic_content_checkpoint(&checkpoint).unwrap().0, [0; 32]);
    }

    #[test]
    fn episodic_digest_changes_with_content_and_clear_has_distinct_state() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let empty = digest_episodic_memory(&memory).unwrap();
        assert!(memory.store_if_significant(episode(1.0, 0.8, 10)));
        let populated = digest_episodic_memory(&memory).unwrap();
        assert_ne!(empty, populated);
        memory.clear();
        assert_eq!(empty, digest_episodic_memory(&memory).unwrap());
    }

    #[test]
    fn governed_scope_is_exact() {
        assert!(validate_scope(
            SubjectAffectingAction::MemoryModification,
            "symthaea:self:episodic-memory",
            "symthaea:self:episodic-memory",
        )
        .is_ok());
        assert!(matches!(
            validate_scope(
                SubjectAffectingAction::CapabilityRestriction,
                "symthaea:self:episodic-memory",
                "symthaea:self:episodic-memory",
            ),
            Err(EpisodicMemoryInterventionError::WrongAction { .. })
        ));
        assert!(matches!(
            validate_scope(
                SubjectAffectingAction::MemoryModification,
                "symthaea:self:other-memory",
                "symthaea:self:episodic-memory",
            ),
            Err(EpisodicMemoryInterventionError::WrongTarget { .. })
        ));
    }

    #[test]
    fn whole_store_erasure_requires_consent_dual_review_and_non_emergency_context() {
        assert!(validate_whole_store_erasure_facts(
            false,
            ExplicitConsentState::Granted,
            true,
            true,
        )
        .is_ok());
        assert!(matches!(
            validate_whole_store_erasure_facts(true, ExplicitConsentState::Granted, true, true),
            Err(EpisodicMemoryInterventionError::EmergencyWholeStoreErasureForbidden)
        ));
        assert!(matches!(
            validate_whole_store_erasure_facts(false, ExplicitConsentState::Unknown, true, true),
            Err(EpisodicMemoryInterventionError::WholeStoreErasureExplicitConsentRequired)
        ));
        assert!(matches!(
            validate_whole_store_erasure_facts(false, ExplicitConsentState::Granted, false, true),
            Err(EpisodicMemoryInterventionError::WholeStoreErasureWelfareReviewRequired)
        ));
        assert!(matches!(
            validate_whole_store_erasure_facts(false, ExplicitConsentState::Granted, true, false),
            Err(EpisodicMemoryInterventionError::WholeStoreErasureIndependentReviewRequired)
        ));
    }
}
