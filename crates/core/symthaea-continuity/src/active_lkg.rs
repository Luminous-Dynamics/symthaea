// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit active Last Known Good selection over qualified checkpoints.
//!
//! A checkpoint proves that a realization earned known-good status. Selecting one
//! as the active recovery reference is a distinct policy event. Neither the record
//! nor the non-Serde selection grants recovery execution authority by itself.
//!
//! `QualifiedCheckpoint != ActiveLastKnownGood != RecoveryExecutionAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::known_good::{KnownGoodCheckpointId, QualifiedKnownGoodCheckpointV1};
use crate::scope::ContinuitySubjectId;
use crate::witness::TargetRealizationId;

pub const ACTIVE_KNOWN_GOOD_SELECTION_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-active-known-good-selection-record-v1";

const SELECTION_DOMAIN: &[u8] = b"symthaea.continuity.active-known-good-selection.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ActiveKnownGoodSelectionId([u8; 32]);

impl ActiveKnownGoodSelectionId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Persistent description of which qualified checkpoint was selected as the active
/// recovery reference. This is auditable state, not a capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveKnownGoodSelectionRecordV1 {
    schema_version: String,
    selection_generation: u64,
    predecessor_selection_id: Option<ActiveKnownGoodSelectionId>,
    checkpoint_id: KnownGoodCheckpointId,
    checkpoint_generation: u64,
    subject_id: ContinuitySubjectId,
    realization_id: TargetRealizationId,
    selected_at_unix_ms: u64,
    selection_basis_digest: [u8; 32],
    selection_id: ActiveKnownGoodSelectionId,
}

impl ActiveKnownGoodSelectionRecordV1 {
    pub fn validate(&self) -> Result<(), ActiveKnownGoodSelectionError> {
        if self.schema_version != ACTIVE_KNOWN_GOOD_SELECTION_RECORD_SCHEMA_V1 {
            return Err(ActiveKnownGoodSelectionError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.selection_generation == 0 {
            return Err(ActiveKnownGoodSelectionError::ZeroSelectionGeneration);
        }
        if self.checkpoint_generation == 0 {
            return Err(ActiveKnownGoodSelectionError::ZeroCheckpointGeneration);
        }
        if self.selected_at_unix_ms == 0 {
            return Err(ActiveKnownGoodSelectionError::ZeroSelectionTime);
        }
        if self.selection_basis_digest == [0; 32] {
            return Err(ActiveKnownGoodSelectionError::ZeroSelectionBasisDigest);
        }
        if self.selection_generation == 1 && self.predecessor_selection_id.is_some() {
            return Err(ActiveKnownGoodSelectionError::UnexpectedPredecessorForFirstSelection);
        }
        if self.selection_generation > 1 && self.predecessor_selection_id.is_none() {
            return Err(ActiveKnownGoodSelectionError::MissingPredecessorSelection);
        }
        let expected = ActiveKnownGoodSelectionId(hash_selection(
            self.selection_generation,
            self.predecessor_selection_id,
            self.checkpoint_id,
            self.checkpoint_generation,
            self.subject_id,
            self.realization_id,
            self.selected_at_unix_ms,
            self.selection_basis_digest,
        ));
        if expected != self.selection_id {
            return Err(ActiveKnownGoodSelectionError::SelectionIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ActiveKnownGoodSelectionId {
        self.selection_id
    }

    pub fn generation(&self) -> u64 {
        self.selection_generation
    }

    pub fn checkpoint_id(&self) -> KnownGoodCheckpointId {
        self.checkpoint_id
    }

    pub fn checkpoint_generation(&self) -> u64 {
        self.checkpoint_generation
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.subject_id
    }

    pub fn realization_id(&self) -> TargetRealizationId {
        self.realization_id
    }

    pub fn selected_at_unix_ms(&self) -> u64 {
        self.selected_at_unix_ms
    }
}

/// Non-Serde active recovery reference rebound to the exact qualified checkpoint.
/// Recovery still needs a separate authenticated authority/capability boundary.
#[derive(Debug, Clone)]
pub struct ActiveKnownGoodSelectionV1 {
    record: ActiveKnownGoodSelectionRecordV1,
}

impl ActiveKnownGoodSelectionV1 {
    /// Crate-owned selection constructor.
    ///
    /// Generation >1 selection is reached through promotion eligibility. Generation
    /// 1 is reserved for a dedicated baseline-admission path. Keeping this crate-owned
    /// prevents external callers from bypassing either theorem by directly selecting
    /// an arbitrary qualified checkpoint.
    pub(crate) fn select(
        checkpoint: &QualifiedKnownGoodCheckpointV1,
        predecessor: Option<&ActiveKnownGoodSelectionV1>,
        selected_at_unix_ms: u64,
        selection_basis_digest: [u8; 32],
    ) -> Result<Self, ActiveKnownGoodSelectionError> {
        checkpoint.record().validate()?;
        if selected_at_unix_ms == 0 {
            return Err(ActiveKnownGoodSelectionError::ZeroSelectionTime);
        }
        if selected_at_unix_ms < checkpoint.record().established_at_unix_ms() {
            return Err(ActiveKnownGoodSelectionError::SelectionPredatesCheckpoint);
        }
        if selection_basis_digest == [0; 32] {
            return Err(ActiveKnownGoodSelectionError::ZeroSelectionBasisDigest);
        }

        let (selection_generation, predecessor_selection_id) = match predecessor {
            Some(previous) => {
                previous.record.validate()?;
                if previous.subject_id() != checkpoint.subject_id() {
                    return Err(ActiveKnownGoodSelectionError::PredecessorSubjectMismatch);
                }
                if checkpoint.generation() <= previous.checkpoint_generation() {
                    return Err(ActiveKnownGoodSelectionError::CheckpointGenerationDidNotAdvance {
                        previous: previous.checkpoint_generation(),
                        next: checkpoint.generation(),
                    });
                }
                if selected_at_unix_ms <= previous.selected_at_unix_ms() {
                    return Err(ActiveKnownGoodSelectionError::SelectionTimeDidNotAdvance);
                }
                (
                    previous
                        .generation()
                        .checked_add(1)
                        .ok_or(ActiveKnownGoodSelectionError::GenerationOverflow)?,
                    Some(previous.id()),
                )
            }
            None => (1, None),
        };

        let checkpoint_id = checkpoint.id();
        let checkpoint_generation = checkpoint.generation();
        let subject_id = checkpoint.subject_id();
        let realization_id = checkpoint.realization_id();
        let selection_id = ActiveKnownGoodSelectionId(hash_selection(
            selection_generation,
            predecessor_selection_id,
            checkpoint_id,
            checkpoint_generation,
            subject_id,
            realization_id,
            selected_at_unix_ms,
            selection_basis_digest,
        ));
        let record = ActiveKnownGoodSelectionRecordV1 {
            schema_version: ACTIVE_KNOWN_GOOD_SELECTION_RECORD_SCHEMA_V1.to_owned(),
            selection_generation,
            predecessor_selection_id,
            checkpoint_id,
            checkpoint_generation,
            subject_id,
            realization_id,
            selected_at_unix_ms,
            selection_basis_digest,
            selection_id,
        };
        record.validate()?;
        Ok(Self { record })
    }

    /// Rebind persisted selection evidence to the exact qualified checkpoint and,
    /// for generations after one, the exact predecessor selection.
    pub fn rebind(
        record: ActiveKnownGoodSelectionRecordV1,
        checkpoint: &QualifiedKnownGoodCheckpointV1,
        predecessor: Option<&ActiveKnownGoodSelectionV1>,
    ) -> Result<Self, ActiveKnownGoodSelectionError> {
        record.validate()?;
        checkpoint.record().validate()?;
        if record.checkpoint_id != checkpoint.id()
            || record.checkpoint_generation != checkpoint.generation()
            || record.subject_id != checkpoint.subject_id()
            || record.realization_id != checkpoint.realization_id()
        {
            return Err(ActiveKnownGoodSelectionError::CheckpointBindingMismatch);
        }
        match (record.predecessor_selection_id, predecessor) {
            (None, None) if record.selection_generation == 1 => {}
            (Some(expected), Some(previous)) => {
                previous.record.validate()?;
                if expected != previous.id() {
                    return Err(ActiveKnownGoodSelectionError::PredecessorSelectionMismatch);
                }
                if previous.subject_id() != record.subject_id {
                    return Err(ActiveKnownGoodSelectionError::PredecessorSubjectMismatch);
                }
                if record.selection_generation != previous.generation() + 1 {
                    return Err(ActiveKnownGoodSelectionError::SelectionGenerationMismatch);
                }
                if record.checkpoint_generation <= previous.checkpoint_generation() {
                    return Err(ActiveKnownGoodSelectionError::CheckpointGenerationDidNotAdvance {
                        previous: previous.checkpoint_generation(),
                        next: record.checkpoint_generation,
                    });
                }
                if record.selected_at_unix_ms <= previous.selected_at_unix_ms() {
                    return Err(ActiveKnownGoodSelectionError::SelectionTimeDidNotAdvance);
                }
            }
            _ => return Err(ActiveKnownGoodSelectionError::PredecessorSelectionMismatch),
        }
        Ok(Self { record })
    }

    pub fn id(&self) -> ActiveKnownGoodSelectionId {
        self.record.id()
    }

    pub fn generation(&self) -> u64 {
        self.record.generation()
    }

    pub fn checkpoint_id(&self) -> KnownGoodCheckpointId {
        self.record.checkpoint_id()
    }

    pub fn checkpoint_generation(&self) -> u64 {
        self.record.checkpoint_generation()
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.record.subject_id()
    }

    pub fn realization_id(&self) -> TargetRealizationId {
        self.record.realization_id()
    }

    pub fn selected_at_unix_ms(&self) -> u64 {
        self.record.selected_at_unix_ms()
    }

    pub fn record(&self) -> &ActiveKnownGoodSelectionRecordV1 {
        &self.record
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ActiveKnownGoodSelectionError {
    #[error(transparent)]
    Checkpoint(#[from] crate::known_good::KnownGoodCheckpointError),
    #[error("unsupported active known-good selection schema: {0}")]
    UnsupportedSchema(String),
    #[error("active known-good selection generation must be non-zero")]
    ZeroSelectionGeneration,
    #[error("active known-good selection checkpoint generation must be non-zero")]
    ZeroCheckpointGeneration,
    #[error("active known-good selection time must be non-zero")]
    ZeroSelectionTime,
    #[error("active known-good selection basis digest must be non-zero")]
    ZeroSelectionBasisDigest,
    #[error("first active known-good selection may not name a predecessor selection")]
    UnexpectedPredecessorForFirstSelection,
    #[error("active known-good selections after generation one require a predecessor selection")]
    MissingPredecessorSelection,
    #[error("active known-good selection generation overflow")]
    GenerationOverflow,
    #[error("active known-good selection predecessor belongs to another continuity subject")]
    PredecessorSubjectMismatch,
    #[error("active known-good selection predates the checkpoint it selects")]
    SelectionPredatesCheckpoint,
    #[error("active known-good checkpoint generation did not advance: previous {previous}, next {next}")]
    CheckpointGenerationDidNotAdvance { previous: u64, next: u64 },
    #[error("active known-good selection time did not strictly advance")]
    SelectionTimeDidNotAdvance,
    #[error("persisted active known-good selection does not bind the exact checkpoint")]
    CheckpointBindingMismatch,
    #[error("persisted active known-good selection does not bind the exact predecessor selection")]
    PredecessorSelectionMismatch,
    #[error("active known-good selection generation is not the predecessor generation plus one")]
    SelectionGenerationMismatch,
    #[error("active known-good selection identity does not match canonical fields")]
    SelectionIdentityMismatch,
}

#[allow(clippy::too_many_arguments)]
fn hash_selection(
    generation: u64,
    predecessor: Option<ActiveKnownGoodSelectionId>,
    checkpoint_id: KnownGoodCheckpointId,
    checkpoint_generation: u64,
    subject_id: ContinuitySubjectId,
    realization_id: TargetRealizationId,
    selected_at_unix_ms: u64,
    selection_basis_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SELECTION_DOMAIN);
    hasher.update(&generation.to_le_bytes());
    match predecessor {
        Some(id) => {
            hasher.update(&[1]);
            hasher.update(id.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(checkpoint_id.as_bytes());
    hasher.update(&checkpoint_generation.to_le_bytes());
    hasher.update(subject_id.as_bytes());
    hasher.update(realization_id.as_bytes());
    hasher.update(&selected_at_unix_ms.to_le_bytes());
    hasher.update(&selection_basis_digest);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_basis_must_be_nonzero() {
        assert_eq!(
            ActiveKnownGoodSelectionError::ZeroSelectionBasisDigest,
            ActiveKnownGoodSelectionError::ZeroSelectionBasisDigest
        );
    }

    #[test]
    fn selection_generation_is_distinct_from_checkpoint_generation() {
        assert_ne!(SELECTION_DOMAIN, b"symthaea.continuity.known-good-checkpoint.v1\0");
    }
}