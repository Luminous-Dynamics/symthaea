// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned persisted representation and pre-import materialization for canonical episodic memory.
//!
//! This module deliberately stops one layer before private replay-heap mutation. It proves that
//! persisted bytes really encode the claimed exact occurrence/content state and then applies the
//! already-derived restart activation plan. Only occurrences explicitly planned `active` are
//! emitted as episode objects for the eventual canonical importer. Restricted occurrences are
//! represented only as withheld identities, reducing the chance that generic loader code can
//! accidentally reactivate them.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_memory::episodic_replay::{Episode, EpisodeInstanceId};
use thiserror::Error;

use crate::memory_identity::{EpisodeContentId, EpisodeContentIdError, episode_content_id};
use crate::restart_activation_gate::{
    EpisodicRestartActivationPlan, InactiveRestartReason, PersistedOccurrenceDescriptor,
};

pub const PERSISTED_EPISODIC_ENVELOPE_SCHEMA: &str =
    "symthaea.welfare.persisted-episodic-envelope.v1";
const MAX_STORE_TARGET_BYTES: usize = 256;
const MAX_RECORD_REF_BYTES: usize = 2048;
const MAX_ENVELOPE_BYTES: usize = 16 * 1024 * 1024;

/// Durable, versioned representation of one exact canonical episodic occurrence.
///
/// The content ID is repeated intentionally: it lets restart logic detect corruption or stale
/// indexing without treating the storage occurrence UUID as semantic identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedEpisodicEnvelope {
    pub schema_version: String,
    pub store_target_id: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub persisted_at_unix_s: u64,
    pub source_generation: u64,
    pub episode: Episode,
}

impl PersistedEpisodicEnvelope {
    pub fn new(
        store_target_id: impl Into<String>,
        episode: Episode,
        persisted_at_unix_s: u64,
        source_generation: u64,
    ) -> Result<Self, PersistedEpisodeEnvelopeError> {
        let store_target_id = store_target_id.into();
        validate_text(
            "store_target_id",
            &store_target_id,
            MAX_STORE_TARGET_BYTES,
        )?;
        let instance_id = episode
            .instance_id
            .ok_or(PersistedEpisodeEnvelopeError::MissingInstanceId)?;
        let content_id = episode_content_id(&episode)?;
        let envelope = Self {
            schema_version: PERSISTED_EPISODIC_ENVELOPE_SCHEMA.into(),
            store_target_id,
            instance_id,
            content_id,
            persisted_at_unix_s,
            source_generation,
            episode,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), PersistedEpisodeEnvelopeError> {
        if self.schema_version != PERSISTED_EPISODIC_ENVELOPE_SCHEMA {
            return Err(PersistedEpisodeEnvelopeError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        validate_text(
            "store_target_id",
            &self.store_target_id,
            MAX_STORE_TARGET_BYTES,
        )?;
        if self.episode.instance_id != Some(self.instance_id) {
            return Err(PersistedEpisodeEnvelopeError::InstanceIdentityMismatch);
        }
        validate_episode_numeric_state(&self.episode)?;
        let actual_content_id = episode_content_id(&self.episode)?;
        if actual_content_id != self.content_id {
            return Err(PersistedEpisodeEnvelopeError::ContentIdentityMismatch {
                expected: self.content_id,
                actual: actual_content_id,
            });
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, PersistedEpisodeEnvelopeError> {
        self.validate()?;
        let encoded = bincode::serialize(self)
            .map_err(|error| PersistedEpisodeEnvelopeError::Encoding(error.to_string()))?;
        if encoded.len() > MAX_ENVELOPE_BYTES {
            return Err(PersistedEpisodeEnvelopeError::EnvelopeTooLarge {
                actual: encoded.len(),
                max: MAX_ENVELOPE_BYTES,
            });
        }
        Ok(encoded)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, PersistedEpisodeEnvelopeError> {
        if bytes.len() > MAX_ENVELOPE_BYTES {
            return Err(PersistedEpisodeEnvelopeError::EnvelopeTooLarge {
                actual: bytes.len(),
                max: MAX_ENVELOPE_BYTES,
            });
        }
        let envelope: Self = bincode::deserialize(bytes)
            .map_err(|error| PersistedEpisodeEnvelopeError::Decoding(error.to_string()))?;
        envelope.validate()?;
        Ok(envelope)
    }

    /// Stable SQLite/record-store primary-key representation for this exact occurrence.
    ///
    /// This is storage identity, not semantic identity and not authority.
    pub fn stable_record_id(&self) -> String {
        format!("episodic:{}", self.instance_id)
    }

    pub fn validate_record_ref(
        &self,
        record_ref: impl Into<String>,
    ) -> Result<ValidatedPersistedOccurrence, PersistedEpisodeEnvelopeError> {
        let record_ref = record_ref.into();
        validate_text("record_ref", &record_ref, MAX_RECORD_REF_BYTES)?;
        self.validate()?;
        Ok(ValidatedPersistedOccurrence {
            envelope: self.clone(),
            record_ref,
        })
    }
}

/// Persisted occurrence that has passed exact UUID/content/state validation.
#[derive(Debug, Clone)]
pub struct ValidatedPersistedOccurrence {
    envelope: PersistedEpisodicEnvelope,
    record_ref: String,
}

impl ValidatedPersistedOccurrence {
    pub fn instance_id(&self) -> EpisodeInstanceId {
        self.envelope.instance_id
    }

    pub fn content_id(&self) -> EpisodeContentId {
        self.envelope.content_id
    }

    pub fn store_target_id(&self) -> &str {
        &self.envelope.store_target_id
    }

    pub fn record_ref(&self) -> &str {
        &self.record_ref
    }

    pub fn episode(&self) -> &Episode {
        &self.envelope.episode
    }

    pub fn descriptor(&self) -> PersistedOccurrenceDescriptor {
        PersistedOccurrenceDescriptor::new(
            self.instance_id(),
            self.content_id(),
            self.record_ref.clone(),
        )
        .expect("validated record reference must remain valid")
    }
}

/// Exact episode admitted by the restart activation theorem for private-heap import.
///
/// There is intentionally no constructor. The only public minting path is
/// [`materialize_canonical_import_batch`].
#[derive(Debug, Clone)]
pub struct CanonicalImportEpisode {
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    record_ref: String,
    episode: Episode,
}

impl CanonicalImportEpisode {
    pub fn instance_id(&self) -> EpisodeInstanceId {
        self.instance_id
    }

    pub fn content_id(&self) -> EpisodeContentId {
        self.content_id
    }

    pub fn record_ref(&self) -> &str {
        &self.record_ref
    }

    pub fn episode(&self) -> &Episode {
        &self.episode
    }

    pub fn into_episode(self) -> Episode {
        self.episode
    }
}

/// Restricted persisted identity deliberately withheld from active episode materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WithheldPersistedOccurrence {
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub reason: InactiveRestartReason,
    pub reconciliation_required: bool,
}

/// Output consumed by the eventual private replay-heap importer.
#[derive(Debug, Clone)]
pub struct CanonicalEpisodicImportBatch {
    active: Vec<CanonicalImportEpisode>,
    withheld: Vec<WithheldPersistedOccurrence>,
}

impl CanonicalEpisodicImportBatch {
    pub fn active(&self) -> &[CanonicalImportEpisode] {
        &self.active
    }

    pub fn withheld(&self) -> &[WithheldPersistedOccurrence] {
        &self.withheld
    }

    pub fn reconciliation_required(&self) -> bool {
        self.withheld
            .iter()
            .any(|entry| entry.reconciliation_required)
    }
}

/// Convert a fail-closed activation plan into exact importable episode objects.
///
/// This function independently rechecks plan/record agreement. Inactive occurrences are never
/// returned as `Episode` values in the active collection, even if their persisted active row still
/// exists because the process crashed between write-ahead intent and physical removal.
pub fn materialize_canonical_import_batch(
    store_target_id: &str,
    plan: &EpisodicRestartActivationPlan,
    persisted: &[ValidatedPersistedOccurrence],
) -> Result<CanonicalEpisodicImportBatch, CanonicalImportBatchError> {
    validate_text(
        "store_target_id",
        store_target_id,
        MAX_STORE_TARGET_BYTES,
    )
    .map_err(CanonicalImportBatchError::Envelope)?;

    let mut records = BTreeMap::new();
    for record in persisted {
        record
            .envelope
            .validate()
            .map_err(CanonicalImportBatchError::Envelope)?;
        if record.store_target_id() != store_target_id {
            return Err(CanonicalImportBatchError::StoreTargetMismatch {
                instance_id: record.instance_id(),
                expected: store_target_id.to_string(),
                actual: record.store_target_id().to_string(),
            });
        }
        if records
            .insert(record.instance_id(), record.clone())
            .is_some()
        {
            return Err(CanonicalImportBatchError::DuplicateValidatedRecord(
                record.instance_id(),
            ));
        }
    }

    let active_ids: BTreeSet<_> = plan.active.iter().map(|entry| entry.instance_id).collect();
    let inactive_ids: BTreeSet<_> = plan.inactive.iter().map(|entry| entry.instance_id).collect();
    if let Some(instance_id) = active_ids.intersection(&inactive_ids).next().copied() {
        return Err(CanonicalImportBatchError::PlanActiveInactiveCollision(
            instance_id,
        ));
    }

    let mut consumed = BTreeSet::new();
    let mut active = Vec::with_capacity(plan.active.len());
    for planned in &plan.active {
        let record = records
            .get(&planned.instance_id)
            .ok_or(CanonicalImportBatchError::MissingActiveRecord(
                planned.instance_id,
            ))?;
        if record.content_id() != planned.content_id {
            return Err(CanonicalImportBatchError::PlanContentMismatch {
                instance_id: planned.instance_id,
                expected: planned.content_id,
                actual: record.content_id(),
            });
        }
        if record.record_ref() != planned.record_ref {
            return Err(CanonicalImportBatchError::PlanRecordRefMismatch(
                planned.instance_id,
            ));
        }
        if record.episode().instance_id != Some(planned.instance_id) {
            return Err(CanonicalImportBatchError::EpisodeInstanceMismatch(
                planned.instance_id,
            ));
        }
        active.push(CanonicalImportEpisode {
            instance_id: planned.instance_id,
            content_id: planned.content_id,
            record_ref: planned.record_ref.clone(),
            episode: record.episode().clone(),
        });
        consumed.insert(planned.instance_id);
    }

    let mut withheld = Vec::with_capacity(plan.inactive.len());
    for planned in &plan.inactive {
        if let Some(record) = records.get(&planned.instance_id) {
            if record.content_id() != planned.content_id {
                return Err(CanonicalImportBatchError::PlanContentMismatch {
                    instance_id: planned.instance_id,
                    expected: planned.content_id,
                    actual: record.content_id(),
                });
            }
            consumed.insert(planned.instance_id);
        }
        withheld.push(WithheldPersistedOccurrence {
            instance_id: planned.instance_id,
            content_id: planned.content_id,
            reason: planned.reason,
            reconciliation_required: planned.reconciliation_required,
        });
    }

    if let Some(instance_id) = records
        .keys()
        .copied()
        .find(|instance_id| !consumed.contains(instance_id))
    {
        return Err(CanonicalImportBatchError::UnaccountedPersistedRecord(
            instance_id,
        ));
    }

    active.sort_by_key(|entry| entry.instance_id);
    withheld.sort_by_key(|entry| entry.instance_id);
    Ok(CanonicalEpisodicImportBatch { active, withheld })
}

fn validate_episode_numeric_state(
    episode: &Episode,
) -> Result<(), PersistedEpisodeEnvelopeError> {
    if !episode.psi.is_finite() {
        return Err(PersistedEpisodeEnvelopeError::NonFiniteNumericState("psi"));
    }
    for (field, value) in [
        ("prediction_error", episode.prediction_error),
        ("valence", episode.valence),
        ("coherence", episode.coherence),
        ("dopamine_at_encoding", episode.dopamine_at_encoding),
    ] {
        if value.is_some_and(|number| !number.is_finite()) {
            return Err(PersistedEpisodeEnvelopeError::NonFiniteNumericState(field));
        }
    }
    if !episode.consolidation_strength.is_finite() {
        return Err(PersistedEpisodeEnvelopeError::NonFiniteNumericState(
            "consolidation_strength",
        ));
    }
    if episode
        .input
        .values
        .iter()
        .chain(episode.output.values.iter())
        .any(|number| !number.is_finite())
    {
        return Err(PersistedEpisodeEnvelopeError::NonFiniteNumericState(
            "input_or_output",
        ));
    }
    if episode
        .semantic_embedding
        .as_ref()
        .is_some_and(|values| values.iter().any(|number| !number.is_finite()))
    {
        return Err(PersistedEpisodeEnvelopeError::NonFiniteNumericState(
            "semantic_embedding",
        ));
    }
    if episode
        .bath_state_at_encoding
        .is_some_and(|values| values.iter().any(|number| !number.is_finite()))
    {
        return Err(PersistedEpisodeEnvelopeError::NonFiniteNumericState(
            "bath_state_at_encoding",
        ));
    }
    Ok(())
}

fn validate_text(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), PersistedEpisodeEnvelopeError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > max
        || value.chars().any(char::is_control)
    {
        Err(PersistedEpisodeEnvelopeError::InvalidText { field })
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PersistedEpisodeEnvelopeError {
    #[error("unsupported persisted episodic envelope schema: {0:?}")]
    UnsupportedSchema(String),
    #[error("persisted episodic envelope is missing canonical occurrence identity")]
    MissingInstanceId,
    #[error("persisted episode occurrence identity disagrees with envelope key")]
    InstanceIdentityMismatch,
    #[error("persisted episode content identity mismatch: expected={expected:?}, actual={actual:?}")]
    ContentIdentityMismatch {
        expected: EpisodeContentId,
        actual: EpisodeContentId,
    },
    #[error("persisted episode contains non-finite numeric state in `{0}`")]
    NonFiniteNumericState(&'static str),
    #[error("invalid persisted episodic text field `{field}`")]
    InvalidText { field: &'static str },
    #[error("persisted episodic envelope is too large: actual={actual}, max={max}")]
    EnvelopeTooLarge { actual: usize, max: usize },
    #[error("could not encode persisted episodic envelope: {0}")]
    Encoding(String),
    #[error("could not decode persisted episodic envelope: {0}")]
    Decoding(String),
    #[error(transparent)]
    ContentIdentity(#[from] EpisodeContentIdError),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CanonicalImportBatchError {
    #[error(transparent)]
    Envelope(#[from] PersistedEpisodeEnvelopeError),
    #[error("duplicate validated persisted record for occurrence {0}")]
    DuplicateValidatedRecord(EpisodeInstanceId),
    #[error("restart plan marks occurrence both active and inactive: {0}")]
    PlanActiveInactiveCollision(EpisodeInstanceId),
    #[error("restart plan requires active occurrence {0} but its validated persisted record is missing")]
    MissingActiveRecord(EpisodeInstanceId),
    #[error("restart plan content does not match validated record for {instance_id}: expected={expected:?}, actual={actual:?}")]
    PlanContentMismatch {
        instance_id: EpisodeInstanceId,
        expected: EpisodeContentId,
        actual: EpisodeContentId,
    },
    #[error("restart plan record reference does not match validated persisted record for {0}")]
    PlanRecordRefMismatch(EpisodeInstanceId),
    #[error("persisted episode does not carry its planned occurrence identity: {0}")]
    EpisodeInstanceMismatch(EpisodeInstanceId),
    #[error("validated persisted record is not accounted for by active/inactive restart plan: {0}")]
    UnaccountedPersistedRecord(EpisodeInstanceId),
    #[error("persisted record belongs to a different episodic store for {instance_id}: expected={expected:?}, actual={actual:?}")]
    StoreTargetMismatch {
        instance_id: EpisodeInstanceId,
        expected: String,
        actual: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{EpisodicMemory, EpisodicReplayConfig};

    use crate::restart_activation_gate::{
        PlannedActiveOccurrence, PlannedInactiveOccurrence,
    };

    const STORE: &str = "symthaea:self:episodic-memory";

    fn stored_duplicate_pair() -> (Episode, Episode) {
        let source = Episode::new(
            ContinuousHV::from_values(vec![1.0, 2.0, 3.0]),
            ContinuousHV::from_values(vec![4.0, 5.0, 6.0]),
            0.82,
            42,
        );
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        memory.store_if_significant_with_id(source.clone()).unwrap();
        memory.store_if_significant_with_id(source).unwrap();
        let mut episodes: Vec<_> = memory
            .get_top_episode_instances(10)
            .into_iter()
            .map(|(_, episode)| episode)
            .collect();
        episodes.sort_by_key(|episode| episode.instance_id.unwrap());
        (episodes.remove(0), episodes.remove(0))
    }

    #[test]
    fn envelope_roundtrip_preserves_exact_occurrence_and_lifecycle_state() {
        let (mut episode, _) = stored_duplicate_pair();
        episode.replay_count = 9;
        episode.retrieval_count = 4;
        episode.consolidation_strength = 2.5;
        let envelope = PersistedEpisodicEnvelope::new(STORE, episode.clone(), 100, 7).unwrap();
        let encoded = envelope.encode().unwrap();
        let decoded = PersistedEpisodicEnvelope::decode(&encoded).unwrap();
        assert_eq!(decoded.instance_id, episode.instance_id.unwrap());
        assert_eq!(decoded.episode.instance_id, episode.instance_id);
        assert_eq!(decoded.episode.replay_count, 9);
        assert_eq!(decoded.episode.retrieval_count, 4);
        assert_eq!(decoded.episode.consolidation_strength, 2.5);
        assert_eq!(decoded.content_id, episode_content_id(&episode).unwrap());
        assert_eq!(
            decoded.stable_record_id(),
            format!("episodic:{}", decoded.instance_id)
        );
    }

    #[test]
    fn tampered_content_is_rejected_even_when_uuid_is_unchanged() {
        let (episode, _) = stored_duplicate_pair();
        let mut envelope = PersistedEpisodicEnvelope::new(STORE, episode, 100, 1).unwrap();
        envelope.episode.valence = Some(0.75);
        assert!(matches!(
            envelope.validate(),
            Err(PersistedEpisodeEnvelopeError::ContentIdentityMismatch { .. })
        ));
    }

    #[test]
    fn envelope_rejects_instance_id_swap() {
        let (first, second) = stored_duplicate_pair();
        let mut envelope = PersistedEpisodicEnvelope::new(STORE, first, 100, 1).unwrap();
        envelope.episode.instance_id = second.instance_id;
        assert!(matches!(
            envelope.validate(),
            Err(PersistedEpisodeEnvelopeError::InstanceIdentityMismatch)
        ));
    }

    #[test]
    fn non_finite_persisted_state_fails_closed() {
        let (mut episode, _) = stored_duplicate_pair();
        episode.consolidation_strength = f64::NAN;
        assert!(matches!(
            PersistedEpisodicEnvelope::new(STORE, episode, 100, 1),
            Err(PersistedEpisodeEnvelopeError::NonFiniteNumericState(
                "consolidation_strength"
            ))
        ));
    }

    #[test]
    fn materializer_emits_only_planned_active_duplicate() {
        let (first, second) = stored_duplicate_pair();
        let content_id = episode_content_id(&first).unwrap();
        assert_eq!(content_id, episode_content_id(&second).unwrap());
        let first_id = first.instance_id.unwrap();
        let second_id = second.instance_id.unwrap();
        let first = PersistedEpisodicEnvelope::new(STORE, first, 100, 1)
            .unwrap()
            .validate_record_ref("record:first")
            .unwrap();
        let second = PersistedEpisodicEnvelope::new(STORE, second, 100, 1)
            .unwrap()
            .validate_record_ref("record:second")
            .unwrap();
        let plan = EpisodicRestartActivationPlan {
            active: vec![PlannedActiveOccurrence {
                instance_id: second_id,
                content_id,
                record_ref: "record:second".into(),
            }],
            inactive: vec![PlannedInactiveOccurrence {
                instance_id: first_id,
                content_id,
                escrow_persistence_ref: "escrow:first".into(),
                reason: InactiveRestartReason::Quarantined,
                reconciliation_required: false,
            }],
            historical_escrow_only: Vec::new(),
        };
        let batch = materialize_canonical_import_batch(STORE, &plan, &[first, second]).unwrap();
        assert_eq!(batch.active().len(), 1);
        assert_eq!(batch.active()[0].instance_id(), second_id);
        assert_eq!(batch.active()[0].episode().instance_id, Some(second_id));
        assert_eq!(batch.withheld().len(), 1);
        assert_eq!(batch.withheld()[0].instance_id, first_id);
    }

    #[test]
    fn plan_record_reference_substitution_fails_closed() {
        let (first, _) = stored_duplicate_pair();
        let instance_id = first.instance_id.unwrap();
        let content_id = episode_content_id(&first).unwrap();
        let record = PersistedEpisodicEnvelope::new(STORE, first, 100, 1)
            .unwrap()
            .validate_record_ref("record:real")
            .unwrap();
        let plan = EpisodicRestartActivationPlan {
            active: vec![PlannedActiveOccurrence {
                instance_id,
                content_id,
                record_ref: "record:substituted".into(),
            }],
            inactive: Vec::new(),
            historical_escrow_only: Vec::new(),
        };
        assert!(matches!(
            materialize_canonical_import_batch(STORE, &plan, &[record]),
            Err(CanonicalImportBatchError::PlanRecordRefMismatch(id)) if id == instance_id
        ));
    }

    #[test]
    fn persisted_record_absent_from_plan_fails_closed() {
        let (first, _) = stored_duplicate_pair();
        let instance_id = first.instance_id.unwrap();
        let record = PersistedEpisodicEnvelope::new(STORE, first, 100, 1)
            .unwrap()
            .validate_record_ref("record:first")
            .unwrap();
        let empty_plan = EpisodicRestartActivationPlan {
            active: Vec::new(),
            inactive: Vec::new(),
            historical_escrow_only: Vec::new(),
        };
        assert!(matches!(
            materialize_canonical_import_batch(STORE, &empty_plan, &[record]),
            Err(CanonicalImportBatchError::UnaccountedPersistedRecord(id)) if id == instance_id
        ));
    }
}