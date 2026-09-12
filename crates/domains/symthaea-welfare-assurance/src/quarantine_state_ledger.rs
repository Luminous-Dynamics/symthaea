// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only, rollback-detectable state ledger for exact episodic quarantine.
//!
//! This ledger deliberately stores *governance/lifecycle state*, not ordinary memory content.
//! `EpisodeContentId` answers "what experience?" while `EpisodeInstanceId` answers "which stored
//! occurrence?". Quarantine state is keyed by the latter and binds the former as an integrity
//! constraint. Keeping the ledger separate prevents governance metadata from contaminating the
//! semantic/content identity of the memory itself.
//!
//! Hash chaining provides integrity. Rollback resistance requires recovery against an externally
//! retained trusted head hash; a self-hashed local file alone is not a rollback-resistant anchor.
//!
//! Restore is explicitly two phase at this layer: `RestorePrepared` remains unresolved quarantine
//! and is safe to recover after a crash. Only a later matching `Restored` event releases the
//! occurrence from quarantine. A reconciler may instead append `RestoreAborted` and leave the
//! occurrence quarantined.

#![deny(unsafe_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_memory::episodic_replay::EpisodeInstanceId;
use thiserror::Error;

use crate::memory_identity::EpisodeContentId;

pub const EPISODIC_QUARANTINE_LEDGER_SCHEMA: &str =
    "symthaea.welfare.episodic-quarantine-ledger.v1";
const GENESIS_DOMAIN: &[u8] = b"symthaea.welfare.episodic-quarantine-ledger.genesis.v1\0";
const EVENT_DOMAIN: &[u8] = b"symthaea.welfare.episodic-quarantine-ledger.event.v1\0";
const MAX_TARGET_ID_BYTES: usize = 256;
const MAX_EXECUTION_ID_BYTES: usize = 256;
const MAX_REF_BYTES: usize = 2048;

/// One state transition committed by the quarantine ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuarantineLedgerEventKind {
    /// An exact occurrence became inactive and reversibly quarantined.
    Quarantined {
        target_id: String,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        quarantined_at_unix_s: u64,
        escrow_digest: Sha256Digest,
        escrow_persistence_ref: String,
    },
    /// Write-ahead intent to restore one exact quarantined occurrence.
    ///
    /// This event does not release quarantine. A restart that sees this as the latest transition
    /// must keep the occurrence inactive and require reconciliation.
    RestorePrepared {
        target_id: String,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        prepared_at_unix_s: u64,
        execution_id: String,
    },
    /// Successful restoration of the exact prepared occurrence.
    Restored {
        target_id: String,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        restored_at_unix_s: u64,
        execution_id: String,
        restore_result_digest: Sha256Digest,
    },
    /// Explicit reconciliation of a prepared restore that did not complete.
    ///
    /// The occurrence remains quarantined.
    RestoreAborted {
        target_id: String,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        aborted_at_unix_s: u64,
        execution_id: String,
        reason_digest: Sha256Digest,
    },
}

impl QuarantineLedgerEventKind {
    pub fn instance_id(&self) -> EpisodeInstanceId {
        match self {
            Self::Quarantined { instance_id, .. }
            | Self::RestorePrepared { instance_id, .. }
            | Self::Restored { instance_id, .. }
            | Self::RestoreAborted { instance_id, .. } => *instance_id,
        }
    }

    pub fn content_id(&self) -> EpisodeContentId {
        match self {
            Self::Quarantined { content_id, .. }
            | Self::RestorePrepared { content_id, .. }
            | Self::Restored { content_id, .. }
            | Self::RestoreAborted { content_id, .. } => *content_id,
        }
    }

    pub fn target_id(&self) -> &str {
        match self {
            Self::Quarantined { target_id, .. }
            | Self::RestorePrepared { target_id, .. }
            | Self::Restored { target_id, .. }
            | Self::RestoreAborted { target_id, .. } => target_id,
        }
    }
}

/// Hash-chained envelope for one state transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuarantineLedgerEnvelope {
    pub schema_version: String,
    pub generation: u64,
    pub previous_hash: Sha256Digest,
    pub event: QuarantineLedgerEventKind,
    pub event_hash: Sha256Digest,
}

/// Unresolved write-ahead restore intent. The occurrence is still quarantined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestorePendingState {
    pub execution_id: String,
    pub prepared_at_unix_s: u64,
    pub generation: u64,
}

/// Current unresolved quarantine state for one exact occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantineLedgerState {
    pub target_id: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub quarantined_at_unix_s: u64,
    pub escrow_digest: Sha256Digest,
    pub escrow_persistence_ref: String,
    pub quarantine_generation: u64,
    pub restore_pending: Option<RestorePendingState>,
}

/// Append-only state machine for exact episodic quarantine.
#[derive(Debug, Clone)]
pub struct EpisodicQuarantineStateLedger {
    events: Vec<QuarantineLedgerEnvelope>,
    unresolved: HashMap<EpisodeInstanceId, QuarantineLedgerState>,
    head_hash: Sha256Digest,
    next_generation: u64,
}

impl Default for EpisodicQuarantineStateLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl EpisodicQuarantineStateLedger {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            unresolved: HashMap::new(),
            head_hash: genesis_hash(),
            next_generation: 1,
        }
    }

    /// Current externally anchorable chain head.
    pub fn head_hash(&self) -> Sha256Digest {
        self.head_hash
    }

    /// Number of committed state transitions.
    pub fn generation(&self) -> u64 {
        self.next_generation.saturating_sub(1)
    }

    pub fn events(&self) -> &[QuarantineLedgerEnvelope] {
        &self.events
    }

    pub fn unresolved_count(&self) -> usize {
        self.unresolved.len()
    }

    pub fn unresolved_state(
        &self,
        instance_id: EpisodeInstanceId,
    ) -> Option<&QuarantineLedgerState> {
        self.unresolved.get(&instance_id)
    }

    /// Deterministic unresolved-state snapshot ordered by occurrence UUID.
    pub fn unresolved_states(&self) -> Vec<QuarantineLedgerState> {
        let mut states: Vec<_> = self.unresolved.values().cloned().collect();
        states.sort_by_key(|state| state.instance_id);
        states
    }

    /// Commit an exact occurrence as quarantined.
    #[allow(clippy::too_many_arguments)]
    pub fn append_quarantined(
        &mut self,
        target_id: impl Into<String>,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        quarantined_at_unix_s: u64,
        escrow_digest: Sha256Digest,
        escrow_persistence_ref: impl Into<String>,
    ) -> Result<Sha256Digest, QuarantineLedgerError> {
        if self.unresolved.contains_key(&instance_id) {
            return Err(QuarantineLedgerError::AlreadyQuarantined(instance_id));
        }
        let target_id = target_id.into();
        let escrow_persistence_ref = escrow_persistence_ref.into();
        validate_text("target_id", &target_id, MAX_TARGET_ID_BYTES)?;
        validate_text(
            "escrow_persistence_ref",
            &escrow_persistence_ref,
            MAX_REF_BYTES,
        )?;
        validate_nonzero_digest("escrow_digest", escrow_digest)?;

        self.append_event(QuarantineLedgerEventKind::Quarantined {
            target_id,
            instance_id,
            content_id,
            quarantined_at_unix_s,
            escrow_digest,
            escrow_persistence_ref,
        })
    }

    /// Persist write-ahead restore intent while keeping the occurrence quarantined.
    pub fn append_restore_prepared(
        &mut self,
        target_id: impl Into<String>,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        prepared_at_unix_s: u64,
        execution_id: impl Into<String>,
    ) -> Result<Sha256Digest, QuarantineLedgerError> {
        let target_id = target_id.into();
        let execution_id = execution_id.into();
        validate_text("target_id", &target_id, MAX_TARGET_ID_BYTES)?;
        validate_text("execution_id", &execution_id, MAX_EXECUTION_ID_BYTES)?;
        let current = self.validate_matching_unresolved(instance_id, content_id, &target_id)?;
        if let Some(pending) = &current.restore_pending {
            return Err(QuarantineLedgerError::RestoreAlreadyPrepared {
                instance_id,
                execution_id: pending.execution_id.clone(),
            });
        }
        self.append_event(QuarantineLedgerEventKind::RestorePrepared {
            target_id,
            instance_id,
            content_id,
            prepared_at_unix_s,
            execution_id,
        })
    }

    /// Commit successful restoration of the exact previously prepared occurrence.
    #[allow(clippy::too_many_arguments)]
    pub fn append_restored(
        &mut self,
        target_id: impl Into<String>,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        restored_at_unix_s: u64,
        execution_id: impl Into<String>,
        restore_result_digest: Sha256Digest,
    ) -> Result<Sha256Digest, QuarantineLedgerError> {
        let target_id = target_id.into();
        let execution_id = execution_id.into();
        validate_text("target_id", &target_id, MAX_TARGET_ID_BYTES)?;
        validate_text("execution_id", &execution_id, MAX_EXECUTION_ID_BYTES)?;
        validate_nonzero_digest("restore_result_digest", restore_result_digest)?;
        let current = self.validate_matching_unresolved(instance_id, content_id, &target_id)?;
        let pending = current
            .restore_pending
            .as_ref()
            .ok_or(QuarantineLedgerError::RestoreNotPrepared(instance_id))?;
        if pending.execution_id != execution_id {
            return Err(QuarantineLedgerError::RestoreExecutionMismatch {
                instance_id,
                expected: pending.execution_id.clone(),
                actual: execution_id,
            });
        }

        self.append_event(QuarantineLedgerEventKind::Restored {
            target_id,
            instance_id,
            content_id,
            restored_at_unix_s,
            execution_id: pending.execution_id.clone(),
            restore_result_digest,
        })
    }

    /// Reconcile a prepared restore as aborted; quarantine remains in force.
    #[allow(clippy::too_many_arguments)]
    pub fn append_restore_aborted(
        &mut self,
        target_id: impl Into<String>,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        aborted_at_unix_s: u64,
        execution_id: impl Into<String>,
        reason_digest: Sha256Digest,
    ) -> Result<Sha256Digest, QuarantineLedgerError> {
        let target_id = target_id.into();
        let execution_id = execution_id.into();
        validate_text("target_id", &target_id, MAX_TARGET_ID_BYTES)?;
        validate_text("execution_id", &execution_id, MAX_EXECUTION_ID_BYTES)?;
        validate_nonzero_digest("reason_digest", reason_digest)?;
        let current = self.validate_matching_unresolved(instance_id, content_id, &target_id)?;
        let pending = current
            .restore_pending
            .as_ref()
            .ok_or(QuarantineLedgerError::RestoreNotPrepared(instance_id))?;
        if pending.execution_id != execution_id {
            return Err(QuarantineLedgerError::RestoreExecutionMismatch {
                instance_id,
                expected: pending.execution_id.clone(),
                actual: execution_id,
            });
        }

        self.append_event(QuarantineLedgerEventKind::RestoreAborted {
            target_id,
            instance_id,
            content_id,
            aborted_at_unix_s,
            execution_id: pending.execution_id.clone(),
            reason_digest,
        })
    }

    /// Reconstruct and validate the complete ledger against an externally retained head hash.
    ///
    /// This verifies schema, generation continuity, previous-hash links, event hashes, and legal
    /// state transitions before accepting the state. Supplying the previously trusted head makes
    /// removal of a valid suffix detectable as rollback.
    pub fn recover_anchored(
        events: &[QuarantineLedgerEnvelope],
        expected_head_hash: Sha256Digest,
    ) -> Result<Self, QuarantineLedgerError> {
        let mut recovered = Self::new();
        for envelope in events {
            if envelope.schema_version != EPISODIC_QUARANTINE_LEDGER_SCHEMA {
                return Err(QuarantineLedgerError::UnsupportedSchema(
                    envelope.schema_version.clone(),
                ));
            }
            if envelope.generation != recovered.next_generation {
                return Err(QuarantineLedgerError::GenerationDiscontinuity {
                    expected: recovered.next_generation,
                    actual: envelope.generation,
                });
            }
            if envelope.previous_hash != recovered.head_hash {
                return Err(QuarantineLedgerError::PreviousHashMismatch {
                    generation: envelope.generation,
                });
            }
            validate_event(&envelope.event)?;
            let recomputed = digest_event(
                envelope.generation,
                envelope.previous_hash,
                &envelope.event,
            )?;
            if recomputed != envelope.event_hash {
                return Err(QuarantineLedgerError::EventHashMismatch {
                    generation: envelope.generation,
                });
            }
            recovered.apply_validated_transition(envelope.generation, &envelope.event)?;
            recovered.events.push(envelope.clone());
            recovered.head_hash = envelope.event_hash;
            recovered.next_generation = recovered.next_generation.saturating_add(1);
        }

        if recovered.head_hash != expected_head_hash {
            return Err(QuarantineLedgerError::HeadAnchorMismatch {
                expected: expected_head_hash,
                actual: recovered.head_hash,
            });
        }
        Ok(recovered)
    }

    fn validate_matching_unresolved(
        &self,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        target_id: &str,
    ) -> Result<&QuarantineLedgerState, QuarantineLedgerError> {
        let current = self
            .unresolved
            .get(&instance_id)
            .ok_or(QuarantineLedgerError::NotQuarantined(instance_id))?;
        if current.content_id != content_id {
            return Err(QuarantineLedgerError::ContentIdentityMismatch {
                instance_id,
                expected: current.content_id,
                actual: content_id,
            });
        }
        if current.target_id != target_id {
            return Err(QuarantineLedgerError::TargetMismatch {
                instance_id,
                expected: current.target_id.clone(),
                actual: target_id.to_string(),
            });
        }
        Ok(current)
    }

    fn append_event(
        &mut self,
        event: QuarantineLedgerEventKind,
    ) -> Result<Sha256Digest, QuarantineLedgerError> {
        validate_event(&event)?;
        let generation = self.next_generation;
        let previous_hash = self.head_hash;
        let event_hash = digest_event(generation, previous_hash, &event)?;

        // Apply before committing the envelope to guarantee that an illegal state transition never
        // appears in the event stream. No mutation below this point can fail.
        self.apply_validated_transition(generation, &event)?;
        self.events.push(QuarantineLedgerEnvelope {
            schema_version: EPISODIC_QUARANTINE_LEDGER_SCHEMA.to_string(),
            generation,
            previous_hash,
            event,
            event_hash,
        });
        self.head_hash = event_hash;
        self.next_generation = self.next_generation.saturating_add(1);
        Ok(event_hash)
    }

    fn apply_validated_transition(
        &mut self,
        generation: u64,
        event: &QuarantineLedgerEventKind,
    ) -> Result<(), QuarantineLedgerError> {
        match event {
            QuarantineLedgerEventKind::Quarantined {
                target_id,
                instance_id,
                content_id,
                quarantined_at_unix_s,
                escrow_digest,
                escrow_persistence_ref,
            } => {
                if self.unresolved.contains_key(instance_id) {
                    return Err(QuarantineLedgerError::AlreadyQuarantined(*instance_id));
                }
                self.unresolved.insert(
                    *instance_id,
                    QuarantineLedgerState {
                        target_id: target_id.clone(),
                        instance_id: *instance_id,
                        content_id: *content_id,
                        quarantined_at_unix_s: *quarantined_at_unix_s,
                        escrow_digest: *escrow_digest,
                        escrow_persistence_ref: escrow_persistence_ref.clone(),
                        quarantine_generation: generation,
                        restore_pending: None,
                    },
                );
            }
            QuarantineLedgerEventKind::RestorePrepared {
                target_id,
                instance_id,
                content_id,
                prepared_at_unix_s,
                execution_id,
            } => {
                let current = self.validate_matching_unresolved(*instance_id, *content_id, target_id)?;
                if let Some(pending) = &current.restore_pending {
                    return Err(QuarantineLedgerError::RestoreAlreadyPrepared {
                        instance_id: *instance_id,
                        execution_id: pending.execution_id.clone(),
                    });
                }
                let current = self.unresolved.get_mut(instance_id).expect("validated above");
                current.restore_pending = Some(RestorePendingState {
                    execution_id: execution_id.clone(),
                    prepared_at_unix_s: *prepared_at_unix_s,
                    generation,
                });
            }
            QuarantineLedgerEventKind::Restored {
                target_id,
                instance_id,
                content_id,
                execution_id,
                ..
            } => {
                let current = self.validate_matching_unresolved(*instance_id, *content_id, target_id)?;
                let pending = current
                    .restore_pending
                    .as_ref()
                    .ok_or(QuarantineLedgerError::RestoreNotPrepared(*instance_id))?;
                if pending.execution_id != *execution_id {
                    return Err(QuarantineLedgerError::RestoreExecutionMismatch {
                        instance_id: *instance_id,
                        expected: pending.execution_id.clone(),
                        actual: execution_id.clone(),
                    });
                }
                self.unresolved.remove(instance_id);
            }
            QuarantineLedgerEventKind::RestoreAborted {
                target_id,
                instance_id,
                content_id,
                execution_id,
                ..
            } => {
                let current = self.validate_matching_unresolved(*instance_id, *content_id, target_id)?;
                let pending = current
                    .restore_pending
                    .as_ref()
                    .ok_or(QuarantineLedgerError::RestoreNotPrepared(*instance_id))?;
                if pending.execution_id != *execution_id {
                    return Err(QuarantineLedgerError::RestoreExecutionMismatch {
                        instance_id: *instance_id,
                        expected: pending.execution_id.clone(),
                        actual: execution_id.clone(),
                    });
                }
                let current = self.unresolved.get_mut(instance_id).expect("validated above");
                current.restore_pending = None;
            }
        }
        Ok(())
    }
}

fn genesis_hash() -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(GENESIS_DOMAIN);
    hasher.finalize()
}

fn digest_event(
    generation: u64,
    previous_hash: Sha256Digest,
    event: &QuarantineLedgerEventKind,
) -> Result<Sha256Digest, QuarantineLedgerError> {
    let encoded = bincode::serialize(event)
        .map_err(|error| QuarantineLedgerError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(EVENT_DOMAIN);
    hasher.update(&generation.to_le_bytes());
    hasher.update(&previous_hash.0);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

fn validate_event(event: &QuarantineLedgerEventKind) -> Result<(), QuarantineLedgerError> {
    validate_text("target_id", event.target_id(), MAX_TARGET_ID_BYTES)?;
    match event {
        QuarantineLedgerEventKind::Quarantined {
            escrow_digest,
            escrow_persistence_ref,
            ..
        } => {
            validate_nonzero_digest("escrow_digest", *escrow_digest)?;
            validate_text(
                "escrow_persistence_ref",
                escrow_persistence_ref,
                MAX_REF_BYTES,
            )?;
        }
        QuarantineLedgerEventKind::RestorePrepared { execution_id, .. } => {
            validate_text("execution_id", execution_id, MAX_EXECUTION_ID_BYTES)?;
        }
        QuarantineLedgerEventKind::Restored {
            execution_id,
            restore_result_digest,
            ..
        } => {
            validate_text("execution_id", execution_id, MAX_EXECUTION_ID_BYTES)?;
            validate_nonzero_digest("restore_result_digest", *restore_result_digest)?;
        }
        QuarantineLedgerEventKind::RestoreAborted {
            execution_id,
            reason_digest,
            ..
        } => {
            validate_text("execution_id", execution_id, MAX_EXECUTION_ID_BYTES)?;
            validate_nonzero_digest("reason_digest", *reason_digest)?;
        }
    }
    Ok(())
}

fn validate_text(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), QuarantineLedgerError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > max
        || value.chars().any(char::is_control)
    {
        return Err(QuarantineLedgerError::InvalidText { field });
    }
    Ok(())
}

fn validate_nonzero_digest(
    field: &'static str,
    value: Sha256Digest,
) -> Result<(), QuarantineLedgerError> {
    if value.0 == [0; 32] {
        return Err(QuarantineLedgerError::ZeroDigest { field });
    }
    Ok(())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum QuarantineLedgerError {
    #[error("invalid quarantine-ledger text field `{field}`")]
    InvalidText { field: &'static str },
    #[error("quarantine-ledger digest `{field}` must not be zero")]
    ZeroDigest { field: &'static str },
    #[error("episodic occurrence is already quarantined: {0}")]
    AlreadyQuarantined(EpisodeInstanceId),
    #[error("episodic occurrence is not quarantined: {0}")]
    NotQuarantined(EpisodeInstanceId),
    #[error("restore already prepared for {instance_id} by execution {execution_id:?}")]
    RestoreAlreadyPrepared {
        instance_id: EpisodeInstanceId,
        execution_id: String,
    },
    #[error("restore is not prepared for episodic occurrence {0}")]
    RestoreNotPrepared(EpisodeInstanceId),
    #[error("restore execution mismatch for {instance_id}: expected={expected:?}, actual={actual:?}")]
    RestoreExecutionMismatch {
        instance_id: EpisodeInstanceId,
        expected: String,
        actual: String,
    },
    #[error("quarantine content identity mismatch for {instance_id}: expected={expected:?}, actual={actual:?}")]
    ContentIdentityMismatch {
        instance_id: EpisodeInstanceId,
        expected: EpisodeContentId,
        actual: EpisodeContentId,
    },
    #[error("quarantine target mismatch for {instance_id}: expected={expected:?}, actual={actual:?}")]
    TargetMismatch {
        instance_id: EpisodeInstanceId,
        expected: String,
        actual: String,
    },
    #[error("unsupported quarantine-ledger schema: {0:?}")]
    UnsupportedSchema(String),
    #[error("quarantine-ledger generation discontinuity: expected={expected}, actual={actual}")]
    GenerationDiscontinuity { expected: u64, actual: u64 },
    #[error("quarantine-ledger previous-hash mismatch at generation {generation}")]
    PreviousHashMismatch { generation: u64 },
    #[error("quarantine-ledger event-hash mismatch at generation {generation}")]
    EventHashMismatch { generation: u64 },
    #[error("quarantine-ledger head does not match trusted external anchor")]
    HeadAnchorMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("quarantine-ledger event encoding failed: {0}")]
    Encoding(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};

    use crate::memory_identity::episode_content_id;

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn identity(seed: u64) -> (EpisodeInstanceId, EpisodeContentId) {
        let episode = Episode::new(
            ContinuousHV::from_values(vec![seed as f32, seed as f32 + 1.0]),
            ContinuousHV::from_values(vec![seed as f32 + 2.0, seed as f32 + 3.0]),
            0.8,
            seed,
        );
        let content_id = episode_content_id(&episode).unwrap();
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let instance_id = memory.store_if_significant_with_id(episode).unwrap();
        (instance_id, content_id)
    }

    #[test]
    fn duplicate_content_instances_have_independent_quarantine_state() {
        let episode = Episode::new(
            ContinuousHV::from_values(vec![1.0, 2.0]),
            ContinuousHV::from_values(vec![3.0, 4.0]),
            0.8,
            10,
        );
        let same_content = episode_content_id(&episode).unwrap();
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let first = memory.store_if_significant_with_id(episode.clone()).unwrap();
        let second = memory.store_if_significant_with_id(episode).unwrap();
        assert_ne!(first, second);

        let mut ledger = EpisodicQuarantineStateLedger::new();
        ledger
            .append_quarantined(
                format!("symthaea:self:episodic-memory:instance:{first}"),
                first,
                same_content,
                100,
                digest(1),
                "escrow:first",
            )
            .unwrap();
        assert!(ledger.unresolved_state(first).is_some());
        assert!(ledger.unresolved_state(second).is_none());

        ledger
            .append_quarantined(
                format!("symthaea:self:episodic-memory:instance:{second}"),
                second,
                same_content,
                101,
                digest(2),
                "escrow:second",
            )
            .unwrap();
        assert_eq!(ledger.unresolved_count(), 2);
    }

    #[test]
    fn restore_is_write_ahead_and_execution_bound() {
        let (id, content_id) = identity(3);
        let target = format!("symthaea:self:episodic-memory:instance:{id}");
        let mut ledger = EpisodicQuarantineStateLedger::new();
        ledger
            .append_quarantined(&target, id, content_id, 100, digest(3), "escrow:3")
            .unwrap();

        assert!(matches!(
            ledger.append_restored(&target, id, content_id, 120, "restore:1", digest(4)),
            Err(QuarantineLedgerError::RestoreNotPrepared(_))
        ));

        ledger
            .append_restore_prepared(&target, id, content_id, 110, "restore:1")
            .unwrap();
        let pending = ledger
            .unresolved_state(id)
            .unwrap()
            .restore_pending
            .as_ref()
            .unwrap();
        assert_eq!(pending.execution_id, "restore:1");

        assert!(matches!(
            ledger.append_restored(&target, id, content_id, 120, "restore:2", digest(4)),
            Err(QuarantineLedgerError::RestoreExecutionMismatch { .. })
        ));
        ledger
            .append_restored(&target, id, content_id, 120, "restore:1", digest(4))
            .unwrap();
        assert!(ledger.unresolved_state(id).is_none());
        assert_eq!(ledger.generation(), 3);
    }

    #[test]
    fn aborted_restore_remains_quarantined_but_clears_pending_intent() {
        let (id, content_id) = identity(4);
        let target = format!("symthaea:self:episodic-memory:instance:{id}");
        let mut ledger = EpisodicQuarantineStateLedger::new();
        ledger
            .append_quarantined(&target, id, content_id, 100, digest(5), "escrow:4")
            .unwrap();
        ledger
            .append_restore_prepared(&target, id, content_id, 110, "restore:4")
            .unwrap();
        ledger
            .append_restore_aborted(&target, id, content_id, 115, "restore:4", digest(6))
            .unwrap();

        let state = ledger.unresolved_state(id).unwrap();
        assert!(state.restore_pending.is_none());
        assert_eq!(state.content_id, content_id);
        assert_eq!(ledger.generation(), 3);
    }

    #[test]
    fn anchored_recovery_detects_valid_suffix_rollback() {
        let (id, content_id) = identity(5);
        let target = format!("symthaea:self:episodic-memory:instance:{id}");
        let mut ledger = EpisodicQuarantineStateLedger::new();
        ledger
            .append_quarantined(&target, id, content_id, 100, digest(7), "escrow:5")
            .unwrap();
        let one_event_head = ledger.head_hash();
        ledger
            .append_restore_prepared(&target, id, content_id, 110, "restore:5")
            .unwrap();
        let trusted_two_event_head = ledger.head_hash();

        let rolled_back = vec![ledger.events()[0].clone()];
        assert!(EpisodicQuarantineStateLedger::recover_anchored(
            &rolled_back,
            one_event_head
        )
        .is_ok());
        assert!(matches!(
            EpisodicQuarantineStateLedger::recover_anchored(
                &rolled_back,
                trusted_two_event_head
            ),
            Err(QuarantineLedgerError::HeadAnchorMismatch { .. })
        ));
    }

    #[test]
    fn recovery_detects_event_tampering() {
        let (id, content_id) = identity(6);
        let target = format!("symthaea:self:episodic-memory:instance:{id}");
        let mut ledger = EpisodicQuarantineStateLedger::new();
        ledger
            .append_quarantined(&target, id, content_id, 100, digest(8), "escrow:6")
            .unwrap();
        let trusted_head = ledger.head_hash();
        let mut tampered = ledger.events().to_vec();
        if let QuarantineLedgerEventKind::Quarantined {
            escrow_persistence_ref,
            ..
        } = &mut tampered[0].event
        {
            *escrow_persistence_ref = "escrow:tampered".into();
        }
        assert!(matches!(
            EpisodicQuarantineStateLedger::recover_anchored(&tampered, trusted_head),
            Err(QuarantineLedgerError::EventHashMismatch { generation: 1 })
        ));
    }

    #[test]
    fn recovery_rejects_illegal_duplicate_quarantine_even_if_rehashed() {
        let (id, content_id) = identity(7);
        let target = format!("symthaea:self:episodic-memory:instance:{id}");
        let mut ledger = EpisodicQuarantineStateLedger::new();
        ledger
            .append_quarantined(&target, id, content_id, 100, digest(9), "escrow:7")
            .unwrap();

        let first = ledger.events()[0].clone();
        let duplicate_event = first.event.clone();
        let duplicate_hash = digest_event(2, first.event_hash, &duplicate_event).unwrap();
        let duplicate = QuarantineLedgerEnvelope {
            schema_version: EPISODIC_QUARANTINE_LEDGER_SCHEMA.into(),
            generation: 2,
            previous_hash: first.event_hash,
            event: duplicate_event,
            event_hash: duplicate_hash,
        };
        assert!(matches!(
            EpisodicQuarantineStateLedger::recover_anchored(
                &[first, duplicate],
                duplicate_hash
            ),
            Err(QuarantineLedgerError::AlreadyQuarantined(_))
        ));
    }
}
