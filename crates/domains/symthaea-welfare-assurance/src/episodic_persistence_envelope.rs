// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned persistence envelope for exact canonical episodic occurrences.
//!
//! The envelope deliberately carries two independent commitments:
//! - `EpisodeContentId`: replay-stable encoding-time identity ("what experience?");
//! - `episode_state_digest`: exact serialized occurrence/lifecycle state, including the canonical
//!   `EpisodeInstanceId` ("what exact stored state are we restoring?").
//!
//! Rank, heap position, and database query order are not identity and do not appear in the record
//! key. A valid envelope can be converted into the fail-closed restart activation descriptor, but
//! this module does not itself mutate the canonical replay heap.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_memory::episodic_replay::{Episode, EpisodeInstanceId};
use thiserror::Error;

use crate::memory_identity::{EpisodeContentId, EpisodeContentIdError, episode_content_id};
use crate::restart_activation_gate::{PersistedOccurrenceDescriptor, RestartActivationError};

pub const EPISODIC_PERSISTENCE_ENVELOPE_SCHEMA: &str =
    "symthaea.memory.episodic-persistence-envelope.v1";
const STATE_DIGEST_DOMAIN: &[u8] = b"symthaea.memory.episodic-state-digest.v1\0";
const ENVELOPE_DIGEST_DOMAIN: &[u8] = b"symthaea.memory.episodic-envelope-digest.v1\0";
const RECORD_KEY_PREFIX: &str = "episodic-v1:";
const MAX_RECORD_REF_BYTES: usize = 2048;

/// Durable, versioned representation of one exact canonical episodic occurrence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedEpisodeEnvelope {
    pub schema_version: String,
    pub record_key: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub episode_state_digest: Sha256Digest,
    pub persisted_at_unix_s: u64,
    pub episode: Episode,
}

impl PersistedEpisodeEnvelope {
    /// Capture one already-stored canonical episode. Unassigned/legacy episodes are rejected.
    pub fn capture(
        episode: &Episode,
        persisted_at_unix_s: u64,
    ) -> Result<Self, EpisodicPersistenceEnvelopeError> {
        validate_episode_numerics(episode)?;
        let instance_id = episode
            .instance_id
            .ok_or(EpisodicPersistenceEnvelopeError::MissingInstanceId)?;
        let content_id = episode_content_id(episode)?;
        let episode_state_digest = digest_episode_state(episode)?;
        Ok(Self {
            schema_version: EPISODIC_PERSISTENCE_ENVELOPE_SCHEMA.into(),
            record_key: canonical_episode_record_key(instance_id),
            instance_id,
            content_id,
            episode_state_digest,
            persisted_at_unix_s,
            episode: episode.clone(),
        })
    }

    /// Validate every redundant identity/state commitment before restart planning.
    pub fn validate(&self) -> Result<(), EpisodicPersistenceEnvelopeError> {
        if self.schema_version != EPISODIC_PERSISTENCE_ENVELOPE_SCHEMA {
            return Err(EpisodicPersistenceEnvelopeError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        validate_episode_numerics(&self.episode)?;
        if self.episode.instance_id != Some(self.instance_id) {
            return Err(EpisodicPersistenceEnvelopeError::InstanceIdentityMismatch);
        }
        let expected_key = canonical_episode_record_key(self.instance_id);
        if self.record_key != expected_key {
            return Err(EpisodicPersistenceEnvelopeError::RecordKeyMismatch {
                expected: expected_key,
                actual: self.record_key.clone(),
            });
        }
        let actual_content = episode_content_id(&self.episode)?;
        if actual_content != self.content_id {
            return Err(EpisodicPersistenceEnvelopeError::ContentIdentityMismatch {
                expected: self.content_id,
                actual: actual_content,
            });
        }
        let actual_state = digest_episode_state(&self.episode)?;
        if actual_state != self.episode_state_digest {
            return Err(EpisodicPersistenceEnvelopeError::StateDigestMismatch {
                expected: self.episode_state_digest,
                actual: actual_state,
            });
        }
        if self.episode_state_digest.0 == [0; 32] {
            return Err(EpisodicPersistenceEnvelopeError::ZeroStateDigest);
        }
        Ok(())
    }

    /// Convert a validated envelope into the restart gate's active-record descriptor.
    pub fn restart_descriptor(
        &self,
        durable_record_ref: impl Into<String>,
    ) -> Result<PersistedOccurrenceDescriptor, EpisodicPersistenceEnvelopeError> {
        self.validate()?;
        let durable_record_ref = durable_record_ref.into();
        validate_record_ref(&durable_record_ref)?;
        PersistedOccurrenceDescriptor::new(
            self.instance_id,
            self.content_id,
            durable_record_ref,
        )
        .map_err(EpisodicPersistenceEnvelopeError::RestartDescriptor)
    }
}

/// Stable durable key for one canonical occurrence. Rank/order never participates.
pub fn canonical_episode_record_key(instance_id: EpisodeInstanceId) -> String {
    format!("{RECORD_KEY_PREFIX}{instance_id}")
}

/// Exact lifecycle-state commitment, including occurrence UUID and mutable replay bookkeeping.
pub fn digest_episode_state(
    episode: &Episode,
) -> Result<Sha256Digest, EpisodicPersistenceEnvelopeError> {
    validate_episode_numerics(episode)?;
    if episode.instance_id.is_none() {
        return Err(EpisodicPersistenceEnvelopeError::MissingInstanceId);
    }
    let encoded = bincode::serialize(episode)
        .map_err(|error| EpisodicPersistenceEnvelopeError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(STATE_DIGEST_DOMAIN);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

/// Commitment to the full validated envelope, useful as persistence/audit evidence.
pub fn digest_persisted_episode_envelope(
    envelope: &PersistedEpisodeEnvelope,
) -> Result<Sha256Digest, EpisodicPersistenceEnvelopeError> {
    envelope.validate()?;
    let encoded = bincode::serialize(envelope)
        .map_err(|error| EpisodicPersistenceEnvelopeError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(ENVELOPE_DIGEST_DOMAIN);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

fn validate_episode_numerics(episode: &Episode) -> Result<(), EpisodicPersistenceEnvelopeError> {
    if !episode.psi.is_finite() {
        return Err(EpisodicPersistenceEnvelopeError::NonFiniteField("psi"));
    }
    if !episode.consolidation_strength.is_finite() {
        return Err(EpisodicPersistenceEnvelopeError::NonFiniteField(
            "consolidation_strength",
        ));
    }
    for (name, value) in [
        ("prediction_error", episode.prediction_error),
        ("valence", episode.valence),
        ("coherence", episode.coherence),
        ("dopamine_at_encoding", episode.dopamine_at_encoding),
    ] {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(EpisodicPersistenceEnvelopeError::NonFiniteField(name));
        }
    }
    if episode.input.values.iter().any(|value| !value.is_finite()) {
        return Err(EpisodicPersistenceEnvelopeError::NonFiniteField("input"));
    }
    if episode.output.values.iter().any(|value| !value.is_finite()) {
        return Err(EpisodicPersistenceEnvelopeError::NonFiniteField("output"));
    }
    if episode
        .bath_state_at_encoding
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(EpisodicPersistenceEnvelopeError::NonFiniteField(
            "bath_state_at_encoding",
        ));
    }
    if episode
        .semantic_embedding
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(EpisodicPersistenceEnvelopeError::NonFiniteField(
            "semantic_embedding",
        ));
    }
    Ok(())
}

fn validate_record_ref(value: &str) -> Result<(), EpisodicPersistenceEnvelopeError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_RECORD_REF_BYTES
        || value.chars().any(char::is_control)
    {
        Err(EpisodicPersistenceEnvelopeError::InvalidRecordReference)
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EpisodicPersistenceEnvelopeError {
    #[error("persisted episodic envelope has unsupported schema {0:?}")]
    UnsupportedSchema(String),
    #[error("persisted episodic episode has no canonical occurrence UUID")]
    MissingInstanceId,
    #[error("persisted episodic occurrence UUID disagrees with embedded episode")]
    InstanceIdentityMismatch,
    #[error("persisted episodic record key mismatch: expected={expected:?}, actual={actual:?}")]
    RecordKeyMismatch { expected: String, actual: String },
    #[error("persisted episodic content identity mismatch: expected={expected:?}, actual={actual:?}")]
    ContentIdentityMismatch {
        expected: EpisodeContentId,
        actual: EpisodeContentId,
    },
    #[error("persisted episodic state digest mismatch: expected={expected:?}, actual={actual:?}")]
    StateDigestMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("persisted episodic state digest may not be zero")]
    ZeroStateDigest,
    #[error("persisted episodic field is non-finite: {0}")]
    NonFiniteField(&'static str),
    #[error("persisted episodic durable record reference is invalid")]
    InvalidRecordReference,
    #[error("could not encode persisted episodic state: {0}")]
    Encoding(String),
    #[error(transparent)]
    ContentIdentity(#[from] EpisodeContentIdError),
    #[error("could not produce restart descriptor: {0}")]
    RestartDescriptor(#[source] RestartActivationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{EpisodicMemory, EpisodicReplayConfig};

    fn stored_episode(seed: f32) -> Episode {
        let episode = Episode::with_metadata(
            ContinuousHV::from_values(vec![seed, seed + 1.0]),
            ContinuousHV::from_values(vec![seed + 2.0, seed + 3.0]),
            0.84,
            42,
            0.2,
            -0.3,
            0.9,
        )
        .with_dopamine(0.6)
        .with_semantic_embedding(vec![0.25, -0.5]);
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        memory.store_if_significant_with_id(episode).unwrap();
        memory.get_top_episode_instances(1).remove(0).1
    }

    #[test]
    fn envelope_round_trip_validates_exact_occurrence_state() {
        let episode = stored_episode(1.0);
        let envelope = PersistedEpisodeEnvelope::capture(&episode, 100).unwrap();
        envelope.validate().unwrap();
        assert_eq!(envelope.instance_id, episode.instance_id.unwrap());
        assert_eq!(envelope.content_id, episode_content_id(&episode).unwrap());
        assert_ne!(envelope.episode_state_digest.0, [0; 32]);
        assert_ne!(digest_persisted_episode_envelope(&envelope).unwrap().0, [0; 32]);
        let descriptor = envelope.restart_descriptor("sqlite:episodic:1").unwrap();
        assert_eq!(descriptor.instance_id, envelope.instance_id);
        assert_eq!(descriptor.content_id, envelope.content_id);
    }

    #[test]
    fn duplicate_content_has_same_content_id_but_distinct_occurrence_state() {
        let source = Episode::new(
            ContinuousHV::from_values(vec![1.0, 2.0]),
            ContinuousHV::from_values(vec![3.0, 4.0]),
            0.9,
            10,
        );
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let first = memory.store_if_significant_with_id(source.clone()).unwrap();
        let second = memory.store_if_significant_with_id(source).unwrap();
        let instances = memory.get_top_episode_instances(10);
        let first_episode = instances.iter().find(|(id, _)| *id == first).unwrap().1.clone();
        let second_episode = instances.iter().find(|(id, _)| *id == second).unwrap().1.clone();
        let first_envelope = PersistedEpisodeEnvelope::capture(&first_episode, 100).unwrap();
        let second_envelope = PersistedEpisodeEnvelope::capture(&second_episode, 100).unwrap();
        assert_eq!(first_envelope.content_id, second_envelope.content_id);
        assert_ne!(first_envelope.instance_id, second_envelope.instance_id);
        assert_ne!(first_envelope.record_key, second_envelope.record_key);
        assert_ne!(
            first_envelope.episode_state_digest,
            second_envelope.episode_state_digest
        );
    }

    #[test]
    fn lifecycle_tampering_keeps_content_identity_but_breaks_state_digest() {
        let episode = stored_episode(2.0);
        let mut envelope = PersistedEpisodeEnvelope::capture(&episode, 100).unwrap();
        let content_before = envelope.content_id;
        envelope.episode.replay_count += 1;
        assert_eq!(episode_content_id(&envelope.episode).unwrap(), content_before);
        assert!(matches!(
            envelope.validate(),
            Err(EpisodicPersistenceEnvelopeError::StateDigestMismatch { .. })
        ));
    }

    #[test]
    fn legacy_unassigned_episode_cannot_be_promoted_to_canonical_persistence() {
        let episode = Episode::new(
            ContinuousHV::from_values(vec![1.0]),
            ContinuousHV::from_values(vec![2.0]),
            0.9,
            1,
        );
        assert!(matches!(
            PersistedEpisodeEnvelope::capture(&episode, 100),
            Err(EpisodicPersistenceEnvelopeError::MissingInstanceId)
        ));
    }

    #[test]
    fn non_finite_persisted_state_fails_closed() {
        let mut episode = stored_episode(3.0);
        episode.consolidation_strength = f64::NAN;
        assert!(matches!(
            PersistedEpisodeEnvelope::capture(&episode, 100),
            Err(EpisodicPersistenceEnvelopeError::NonFiniteField(
                "consolidation_strength"
            ))
        ));
    }

    #[test]
    fn record_key_is_uuid_stable_and_rank_independent() {
        let episode = stored_episode(4.0);
        let id = episode.instance_id.unwrap();
        let envelope = PersistedEpisodeEnvelope::capture(&episode, 100).unwrap();
        assert_eq!(envelope.record_key, canonical_episode_record_key(id));
        assert!(envelope.record_key.ends_with(&id.to_string()));
    }
}
