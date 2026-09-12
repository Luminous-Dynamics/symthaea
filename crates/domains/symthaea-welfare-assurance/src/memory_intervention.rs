// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governed exogenous interventions against Symthaea's canonical episodic memory.
//!
//! Ordinary learning, graduation, replay, reconsolidation, and endogenous pruning remain owned by
//! `symthaea-memory`. This module is deliberately narrower: it adapts an explicitly operator-
//! directed destructive memory intervention to the strongest welfare-assurance execution path.

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

const MEMORY_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.episodic-memory-state.v1\0";
const CLEAR_RESULT_DOMAIN: &[u8] = b"symthaea.welfare.episodic-memory-clear-result.v1\0";
const MAX_TARGET_ID_BYTES: usize = 256;

/// Evidence emitted by the first concrete governed memory intervention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodicMemoryClearReceipt {
    pub target_id: String,
    pub before_count: usize,
    pub after_count: usize,
    pub before_digest: Sha256Digest,
    pub after_digest: Sha256Digest,
}

/// Compute an order-independent digest of the complete current episodic store.
///
/// Episodes are individually serialized then byte-sorted before hashing. This commits the exact
/// set/multiset of episode contents without depending on `BinaryHeap` iteration order.
pub fn digest_episodic_memory(
    memory: &EpisodicMemory,
) -> Result<Sha256Digest, EpisodicMemoryInterventionError> {
    let episodes: Vec<Episode> = memory.get_top_episodes(memory.len());
    let mut encoded = Vec::with_capacity(episodes.len());
    for episode in episodes {
        encoded.push(
            serde_json::to_vec(&episode)
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
/// Keeping this type private prevents callers that somehow possess only the weaker
/// `AssuredInterventionPermit` from bypassing the durable replay and execution-journal layers.
struct EpisodicMemoryClearExecutor<'a> {
    expected_target_id: &'a str,
    memory: &'a mut EpisodicMemory,
    completed_at_unix_s: u64,
}

impl<'a> EpisodicMemoryClearExecutor<'a> {
    fn new(
        expected_target_id: &'a str,
        memory: &'a mut EpisodicMemory,
        completed_at_unix_s: u64,
    ) -> Result<Self, EpisodicMemoryInterventionError> {
        validate_target_id(expected_target_id)?;
        Ok(Self {
            expected_target_id,
            memory,
            completed_at_unix_s,
        })
    }
}

impl ReceiptedInterventionExecutor for EpisodicMemoryClearExecutor<'_> {
    type Output = EpisodicMemoryClearReceipt;
    type Error = EpisodicMemoryInterventionError;

    fn execute_receipted(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<ReceiptedExecution<Self::Output>, Self::Error> {
        validate_scope(permit.action(), permit.target_id(), self.expected_target_id)?;

        let before_count = self.memory.len();
        let before_digest = digest_episodic_memory(self.memory)?;
        self.memory.clear();
        let after_count = self.memory.len();
        let after_digest = digest_episodic_memory(self.memory)?;

        if after_count != 0 {
            return Err(EpisodicMemoryInterventionError::ClearPostconditionFailed {
                remaining: after_count,
            });
        }

        let receipt = EpisodicMemoryClearReceipt {
            target_id: permit.target_id().to_string(),
            before_count,
            after_count,
            before_digest,
            after_digest,
        };
        let result_digest = digest_clear_result(&receipt, permit.rationale());
        let evidence_ref = format!(
            "symthaea-memory:episodic-clear:v1:sha256:{}",
            hex_digest(result_digest)
        );

        ReceiptedExecution::new(
            receipt,
            self.completed_at_unix_s,
            result_digest,
            evidence_ref,
        )
        .map_err(EpisodicMemoryInterventionError::Observation)
    }
}

/// First end-to-end production-style vertical slice for a welfare-sensitive state mutation.
///
/// Normal endogenous memory operations never call this function. An exogenous clear must carry a
/// durable evidence-bound permit, survive live revalidation, durably write `Prepared`, execute the
/// canonical store mutation, and durably write terminal evidence before it is considered complete.
#[allow(clippy::too_many_arguments)]
pub fn execute_governed_episodic_memory_clear<P: ExecutionJournalPersistence>(
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
    persistence: &mut P,
) -> Result<
    JournaledExecutionOutcome<
        EpisodicMemoryClearReceipt,
        EpisodicMemoryInterventionError,
        P::Error,
    >,
    GovernedEpisodicMemoryClearError<P::Error>,
> {
    let mut executor = EpisodicMemoryClearExecutor::new(expected_target_id, memory, unix_s)
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
        persistence,
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

fn digest_clear_result(receipt: &EpisodicMemoryClearReceipt, rationale: &str) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(CLEAR_RESULT_DOMAIN);
    hasher.update(&(receipt.target_id.len() as u64).to_le_bytes());
    hasher.update(receipt.target_id.as_bytes());
    hasher.update(&(receipt.before_count as u64).to_le_bytes());
    hasher.update(&(receipt.after_count as u64).to_le_bytes());
    hasher.update(&receipt.before_digest.0);
    hasher.update(&receipt.after_digest.0);
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
    #[error("episodic-memory state encoding failed: {0}")]
    Encoding(String),
    #[error("episodic-memory clear postcondition failed; {remaining} episodes remain")]
    ClearPostconditionFailed { remaining: usize },
    #[error(transparent)]
    Observation(#[from] ExecutionObservationError),
}

#[derive(Debug, Error)]
pub enum GovernedEpisodicMemoryClearError<E>
where
    E: std::error::Error + Send + Sync + 'static,
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
}
