// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Write-ahead, rollback-detectable intent ledger for exact episodic quarantine.
//!
//! This ledger closes the crash window before canonical memory mutation. `Prepared` means an
//! exact occurrence MUST NOT become active after restart until the intent is reconciled, even if
//! the process died before the in-memory quarantine mutation completed. `Committed` binds that
//! intent to the durable quarantine-state ledger head. `Aborted` explicitly releases a prepared
//! intent when the occurrence is known to remain active.
//!
//! Hash chaining provides integrity. Rollback resistance requires recovery against an externally
//! retained trusted head hash; a self-contained local file is not a monotonic anchor.

#![deny(unsafe_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_memory::episodic_replay::EpisodeInstanceId;
use thiserror::Error;

use crate::memory_identity::EpisodeContentId;

pub const EPISODIC_QUARANTINE_INTENT_LEDGER_SCHEMA: &str =
    "symthaea.welfare.episodic-quarantine-intent-ledger.v1";
const GENESIS_DOMAIN: &[u8] = b"symthaea.welfare.episodic-quarantine-intent.genesis.v1\0";
const EVENT_DOMAIN: &[u8] = b"symthaea.welfare.episodic-quarantine-intent.event.v1\0";
const MAX_TARGET_ID_BYTES: usize = 256;
const MAX_EXECUTION_ID_BYTES: usize = 256;
const MAX_REF_BYTES: usize = 2048;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuarantineIntentEventKind {
    Prepared {
        execution_id: String,
        target_id: String,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        prepared_at_unix_s: u64,
        pre_active_state_digest: Sha256Digest,
        escrow_digest: Sha256Digest,
        escrow_persistence_ref: String,
    },
    Committed {
        execution_id: String,
        target_id: String,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        committed_at_unix_s: u64,
        quarantine_state_ledger_head: Sha256Digest,
    },
    Aborted {
        execution_id: String,
        target_id: String,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        aborted_at_unix_s: u64,
        reason_digest: Sha256Digest,
    },
}

impl QuarantineIntentEventKind {
    fn execution_id(&self) -> &str {
        match self {
            Self::Prepared { execution_id, .. }
            | Self::Committed { execution_id, .. }
            | Self::Aborted { execution_id, .. } => execution_id,
        }
    }

    fn target_id(&self) -> &str {
        match self {
            Self::Prepared { target_id, .. }
            | Self::Committed { target_id, .. }
            | Self::Aborted { target_id, .. } => target_id,
        }
    }

    fn instance_id(&self) -> EpisodeInstanceId {
        match self {
            Self::Prepared { instance_id, .. }
            | Self::Committed { instance_id, .. }
            | Self::Aborted { instance_id, .. } => *instance_id,
        }
    }

    fn content_id(&self) -> EpisodeContentId {
        match self {
            Self::Prepared { content_id, .. }
            | Self::Committed { content_id, .. }
            | Self::Aborted { content_id, .. } => *content_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuarantineIntentEnvelope {
    pub schema_version: String,
    pub generation: u64,
    pub previous_hash: Sha256Digest,
    pub event: QuarantineIntentEventKind,
    pub event_hash: Sha256Digest,
}

/// A prepared but not yet durably committed quarantine intent.
///
/// Recovery policy is fail closed: while this state exists, the occurrence must remain inactive
/// until reconciliation proves either committed quarantine or an explicit abort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingQuarantineIntent {
    pub execution_id: String,
    pub target_id: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub prepared_at_unix_s: u64,
    pub pre_active_state_digest: Sha256Digest,
    pub escrow_digest: Sha256Digest,
    pub escrow_persistence_ref: String,
    pub generation: u64,
}

#[derive(Debug, Clone)]
pub struct EpisodicQuarantineIntentLedger {
    events: Vec<QuarantineIntentEnvelope>,
    pending: HashMap<EpisodeInstanceId, PendingQuarantineIntent>,
    head_hash: Sha256Digest,
    next_generation: u64,
}

impl Default for EpisodicQuarantineIntentLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl EpisodicQuarantineIntentLedger {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            pending: HashMap::new(),
            head_hash: genesis_hash(),
            next_generation: 1,
        }
    }

    pub fn events(&self) -> &[QuarantineIntentEnvelope] {
        &self.events
    }

    pub fn head_hash(&self) -> Sha256Digest {
        self.head_hash
    }

    pub fn generation(&self) -> u64 {
        self.next_generation.saturating_sub(1)
    }

    pub fn pending_state(
        &self,
        instance_id: EpisodeInstanceId,
    ) -> Option<&PendingQuarantineIntent> {
        self.pending.get(&instance_id)
    }

    pub fn pending_states(&self) -> Vec<PendingQuarantineIntent> {
        let mut states: Vec<_> = self.pending.values().cloned().collect();
        states.sort_by_key(|state| state.instance_id);
        states
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_prepared(
        &mut self,
        execution_id: impl Into<String>,
        target_id: impl Into<String>,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        prepared_at_unix_s: u64,
        pre_active_state_digest: Sha256Digest,
        escrow_digest: Sha256Digest,
        escrow_persistence_ref: impl Into<String>,
    ) -> Result<Sha256Digest, QuarantineIntentLedgerError> {
        if self.pending.contains_key(&instance_id) {
            return Err(QuarantineIntentLedgerError::AlreadyPrepared(instance_id));
        }
        let execution_id = execution_id.into();
        let target_id = target_id.into();
        let escrow_persistence_ref = escrow_persistence_ref.into();
        validate_text("execution_id", &execution_id, MAX_EXECUTION_ID_BYTES)?;
        validate_text("target_id", &target_id, MAX_TARGET_ID_BYTES)?;
        validate_text(
            "escrow_persistence_ref",
            &escrow_persistence_ref,
            MAX_REF_BYTES,
        )?;
        validate_nonzero_digest("pre_active_state_digest", pre_active_state_digest)?;
        validate_nonzero_digest("escrow_digest", escrow_digest)?;

        self.append_event(QuarantineIntentEventKind::Prepared {
            execution_id,
            target_id,
            instance_id,
            content_id,
            prepared_at_unix_s,
            pre_active_state_digest,
            escrow_digest,
            escrow_persistence_ref,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_committed(
        &mut self,
        execution_id: impl Into<String>,
        target_id: impl Into<String>,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        committed_at_unix_s: u64,
        quarantine_state_ledger_head: Sha256Digest,
    ) -> Result<Sha256Digest, QuarantineIntentLedgerError> {
        let execution_id = execution_id.into();
        let target_id = target_id.into();
        validate_text("execution_id", &execution_id, MAX_EXECUTION_ID_BYTES)?;
        validate_text("target_id", &target_id, MAX_TARGET_ID_BYTES)?;
        validate_nonzero_digest(
            "quarantine_state_ledger_head",
            quarantine_state_ledger_head,
        )?;
        self.validate_matching_pending(instance_id, content_id, &target_id, &execution_id)?;

        self.append_event(QuarantineIntentEventKind::Committed {
            execution_id,
            target_id,
            instance_id,
            content_id,
            committed_at_unix_s,
            quarantine_state_ledger_head,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_aborted(
        &mut self,
        execution_id: impl Into<String>,
        target_id: impl Into<String>,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        aborted_at_unix_s: u64,
        reason_digest: Sha256Digest,
    ) -> Result<Sha256Digest, QuarantineIntentLedgerError> {
        let execution_id = execution_id.into();
        let target_id = target_id.into();
        validate_text("execution_id", &execution_id, MAX_EXECUTION_ID_BYTES)?;
        validate_text("target_id", &target_id, MAX_TARGET_ID_BYTES)?;
        validate_nonzero_digest("reason_digest", reason_digest)?;
        self.validate_matching_pending(instance_id, content_id, &target_id, &execution_id)?;

        self.append_event(QuarantineIntentEventKind::Aborted {
            execution_id,
            target_id,
            instance_id,
            content_id,
            aborted_at_unix_s,
            reason_digest,
        })
    }

    pub fn recover_anchored(
        events: &[QuarantineIntentEnvelope],
        expected_head_hash: Sha256Digest,
    ) -> Result<Self, QuarantineIntentLedgerError> {
        let mut recovered = Self::new();
        for envelope in events {
            if envelope.schema_version != EPISODIC_QUARANTINE_INTENT_LEDGER_SCHEMA {
                return Err(QuarantineIntentLedgerError::UnsupportedSchema(
                    envelope.schema_version.clone(),
                ));
            }
            if envelope.generation != recovered.next_generation {
                return Err(QuarantineIntentLedgerError::GenerationDiscontinuity {
                    expected: recovered.next_generation,
                    actual: envelope.generation,
                });
            }
            if envelope.previous_hash != recovered.head_hash {
                return Err(QuarantineIntentLedgerError::PreviousHashMismatch {
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
                return Err(QuarantineIntentLedgerError::EventHashMismatch {
                    generation: envelope.generation,
                });
            }
            recovered.apply_validated_transition(envelope.generation, &envelope.event)?;
            recovered.events.push(envelope.clone());
            recovered.head_hash = envelope.event_hash;
            recovered.next_generation = recovered.next_generation.saturating_add(1);
        }

        if recovered.head_hash != expected_head_hash {
            return Err(QuarantineIntentLedgerError::HeadAnchorMismatch {
                expected: expected_head_hash,
                actual: recovered.head_hash,
            });
        }
        Ok(recovered)
    }

    fn validate_matching_pending(
        &self,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        target_id: &str,
        execution_id: &str,
    ) -> Result<&PendingQuarantineIntent, QuarantineIntentLedgerError> {
        let pending = self
            .pending
            .get(&instance_id)
            .ok_or(QuarantineIntentLedgerError::NotPrepared(instance_id))?;
        if pending.content_id != content_id {
            return Err(QuarantineIntentLedgerError::ContentIdentityMismatch {
                instance_id,
                expected: pending.content_id,
                actual: content_id,
            });
        }
        if pending.target_id != target_id {
            return Err(QuarantineIntentLedgerError::TargetMismatch {
                instance_id,
                expected: pending.target_id.clone(),
                actual: target_id.to_string(),
            });
        }
        if pending.execution_id != execution_id {
            return Err(QuarantineIntentLedgerError::ExecutionMismatch {
                instance_id,
                expected: pending.execution_id.clone(),
                actual: execution_id.to_string(),
            });
        }
        Ok(pending)
    }

    fn append_event(
        &mut self,
        event: QuarantineIntentEventKind,
    ) -> Result<Sha256Digest, QuarantineIntentLedgerError> {
        validate_event(&event)?;
        let generation = self.next_generation;
        let previous_hash = self.head_hash;
        let event_hash = digest_event(generation, previous_hash, &event)?;
        self.apply_validated_transition(generation, &event)?;
        self.events.push(QuarantineIntentEnvelope {
            schema_version: EPISODIC_QUARANTINE_INTENT_LEDGER_SCHEMA.into(),
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
        event: &QuarantineIntentEventKind,
    ) -> Result<(), QuarantineIntentLedgerError> {
        match event {
            QuarantineIntentEventKind::Prepared {
                execution_id,
                target_id,
                instance_id,
                content_id,
                prepared_at_unix_s,
                pre_active_state_digest,
                escrow_digest,
                escrow_persistence_ref,
            } => {
                if self.pending.contains_key(instance_id) {
                    return Err(QuarantineIntentLedgerError::AlreadyPrepared(*instance_id));
                }
                self.pending.insert(
                    *instance_id,
                    PendingQuarantineIntent {
                        execution_id: execution_id.clone(),
                        target_id: target_id.clone(),
                        instance_id: *instance_id,
                        content_id: *content_id,
                        prepared_at_unix_s: *prepared_at_unix_s,
                        pre_active_state_digest: *pre_active_state_digest,
                        escrow_digest: *escrow_digest,
                        escrow_persistence_ref: escrow_persistence_ref.clone(),
                        generation,
                    },
                );
            }
            QuarantineIntentEventKind::Committed {
                execution_id,
                target_id,
                instance_id,
                content_id,
                ..
            }
            | QuarantineIntentEventKind::Aborted {
                execution_id,
                target_id,
                instance_id,
                content_id,
                ..
            } => {
                self.validate_matching_pending(
                    *instance_id,
                    *content_id,
                    target_id,
                    execution_id,
                )?;
                self.pending.remove(instance_id);
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
    event: &QuarantineIntentEventKind,
) -> Result<Sha256Digest, QuarantineIntentLedgerError> {
    let encoded = bincode::serialize(event)
        .map_err(|error| QuarantineIntentLedgerError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(EVENT_DOMAIN);
    hasher.update(&generation.to_le_bytes());
    hasher.update(&previous_hash.0);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

fn validate_event(event: &QuarantineIntentEventKind) -> Result<(), QuarantineIntentLedgerError> {
    validate_text("execution_id", event.execution_id(), MAX_EXECUTION_ID_BYTES)?;
    validate_text("target_id", event.target_id(), MAX_TARGET_ID_BYTES)?;
    match event {
        QuarantineIntentEventKind::Prepared {
            pre_active_state_digest,
            escrow_digest,
            escrow_persistence_ref,
            ..
        } => {
            validate_nonzero_digest("pre_active_state_digest", *pre_active_state_digest)?;
            validate_nonzero_digest("escrow_digest", *escrow_digest)?;
            validate_text(
                "escrow_persistence_ref",
                escrow_persistence_ref,
                MAX_REF_BYTES,
            )?;
        }
        QuarantineIntentEventKind::Committed {
            quarantine_state_ledger_head,
            ..
        } => validate_nonzero_digest(
            "quarantine_state_ledger_head",
            *quarantine_state_ledger_head,
        )?,
        QuarantineIntentEventKind::Aborted { reason_digest, .. } => {
            validate_nonzero_digest("reason_digest", *reason_digest)?;
        }
    }
    Ok(())
}

fn validate_text(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), QuarantineIntentLedgerError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > max
        || value.chars().any(char::is_control)
    {
        Err(QuarantineIntentLedgerError::InvalidText { field })
    } else {
        Ok(())
    }
}

fn validate_nonzero_digest(
    field: &'static str,
    digest: Sha256Digest,
) -> Result<(), QuarantineIntentLedgerError> {
    if digest.0 == [0; 32] {
        Err(QuarantineIntentLedgerError::ZeroDigest { field })
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum QuarantineIntentLedgerError {
    #[error("invalid quarantine-intent text field `{field}`")]
    InvalidText { field: &'static str },
    #[error("quarantine-intent digest `{field}` must not be zero")]
    ZeroDigest { field: &'static str },
    #[error("quarantine intent is already prepared for occurrence {0}")]
    AlreadyPrepared(EpisodeInstanceId),
    #[error("quarantine intent is not prepared for occurrence {0}")]
    NotPrepared(EpisodeInstanceId),
    #[error("quarantine intent content mismatch for {instance_id}: expected={expected:?}, actual={actual:?}")]
    ContentIdentityMismatch {
        instance_id: EpisodeInstanceId,
        expected: EpisodeContentId,
        actual: EpisodeContentId,
    },
    #[error("quarantine intent target mismatch for {instance_id}: expected={expected:?}, actual={actual:?}")]
    TargetMismatch {
        instance_id: EpisodeInstanceId,
        expected: String,
        actual: String,
    },
    #[error("quarantine intent execution mismatch for {instance_id}: expected={expected:?}, actual={actual:?}")]
    ExecutionMismatch {
        instance_id: EpisodeInstanceId,
        expected: String,
        actual: String,
    },
    #[error("unsupported quarantine-intent ledger schema: {0:?}")]
    UnsupportedSchema(String),
    #[error("quarantine-intent generation discontinuity: expected={expected}, actual={actual}")]
    GenerationDiscontinuity { expected: u64, actual: u64 },
    #[error("quarantine-intent previous-hash mismatch at generation {generation}")]
    PreviousHashMismatch { generation: u64 },
    #[error("quarantine-intent event-hash mismatch at generation {generation}")]
    EventHashMismatch { generation: u64 },
    #[error("quarantine-intent head does not match trusted external anchor")]
    HeadAnchorMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("quarantine-intent event encoding failed: {0}")]
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
    fn prepared_is_fail_closed_until_commit_or_abort() {
        let (id, content_id) = identity(11);
        let target = format!("symthaea:self:episodic-memory:instance:{id}");
        let mut ledger = EpisodicQuarantineIntentLedger::new();
        ledger
            .append_prepared(
                "exec:q:1",
                &target,
                id,
                content_id,
                100,
                digest(1),
                digest(2),
                "escrow:q:1",
            )
            .unwrap();
        assert!(ledger.pending_state(id).is_some());

        let trusted_head = ledger.head_hash();
        let recovered =
            EpisodicQuarantineIntentLedger::recover_anchored(ledger.events(), trusted_head)
                .unwrap();
        assert!(recovered.pending_state(id).is_some());
    }

    #[test]
    fn commit_clears_pending_and_binds_state_ledger_head() {
        let (id, content_id) = identity(12);
        let target = format!("symthaea:self:episodic-memory:instance:{id}");
        let mut ledger = EpisodicQuarantineIntentLedger::new();
        ledger
            .append_prepared(
                "exec:q:2",
                &target,
                id,
                content_id,
                100,
                digest(3),
                digest(4),
                "escrow:q:2",
            )
            .unwrap();
        ledger
            .append_committed("exec:q:2", &target, id, content_id, 101, digest(5))
            .unwrap();
        assert!(ledger.pending_state(id).is_none());
    }

    #[test]
    fn abort_clears_pending_without_claiming_quarantine() {
        let (id, content_id) = identity(13);
        let target = format!("symthaea:self:episodic-memory:instance:{id}");
        let mut ledger = EpisodicQuarantineIntentLedger::new();
        ledger
            .append_prepared(
                "exec:q:3",
                &target,
                id,
                content_id,
                100,
                digest(6),
                digest(7),
                "escrow:q:3",
            )
            .unwrap();
        ledger
            .append_aborted("exec:q:3", &target, id, content_id, 101, digest(8))
            .unwrap();
        assert!(ledger.pending_state(id).is_none());
    }

    #[test]
    fn anchored_recovery_detects_suffix_rollback() {
        let (id, content_id) = identity(14);
        let target = format!("symthaea:self:episodic-memory:instance:{id}");
        let mut ledger = EpisodicQuarantineIntentLedger::new();
        ledger
            .append_prepared(
                "exec:q:4",
                &target,
                id,
                content_id,
                100,
                digest(9),
                digest(10),
                "escrow:q:4",
            )
            .unwrap();
        let prepared_head = ledger.head_hash();
        ledger
            .append_committed("exec:q:4", &target, id, content_id, 101, digest(11))
            .unwrap();
        let committed_head = ledger.head_hash();
        let truncated = vec![ledger.events()[0].clone()];

        assert!(EpisodicQuarantineIntentLedger::recover_anchored(
            &truncated,
            prepared_head
        )
        .is_ok());
        assert!(matches!(
            EpisodicQuarantineIntentLedger::recover_anchored(&truncated, committed_head),
            Err(QuarantineIntentLedgerError::HeadAnchorMismatch { .. })
        ));
    }
}
