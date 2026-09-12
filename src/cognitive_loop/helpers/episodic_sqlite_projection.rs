// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact, versioned SQLite projection for canonical episodic occurrences.
//!
//! This module is intentionally storage-focused and does not depend on welfare/authority policy.
//! It preserves the occurrence UUID and full serialized `Episode` state so higher assurance layers
//! can later validate restart material without treating heap rank as identity.

use serde::{Deserialize, Serialize};

use crate::databases::{MemoryRecord, MemoryType};
use crate::memory::episodic_replay::{Episode, EpisodeInstanceId};

pub(super) const EPISODIC_SQLITE_PROJECTION_SCHEMA: &str =
    "symthaea.memory.episodic-sqlite-projection.v1";
const RECORD_KEY_PREFIX: &str = "episodic-v1:";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EpisodicSqliteProjectionV1 {
    pub schema_version: String,
    pub instance_id: EpisodeInstanceId,
    pub episode: Episode,
}

pub(super) fn canonical_episode_record_key(instance_id: EpisodeInstanceId) -> String {
    format!("{RECORD_KEY_PREFIX}{instance_id}")
}

/// Project one canonical stored episode into the existing generic memory-record schema.
///
/// Legacy/unassigned episodes are refused rather than being given a rank/timestamp-derived
/// identity. Full episode state is retained in versioned JSON metadata; the ordinary indexed
/// columns remain useful for similarity/search without becoming the source of restart truth.
pub(super) fn project_episode_to_memory_record(
    episode: &Episode,
) -> Result<MemoryRecord, EpisodicSqliteProjectionError> {
    validate_episode_state(episode)?;
    let instance_id = episode
        .instance_id
        .ok_or(EpisodicSqliteProjectionError::MissingInstanceId)?;

    let metadata = serde_json::to_string(&EpisodicSqliteProjectionV1 {
        schema_version: EPISODIC_SQLITE_PROJECTION_SCHEMA.to_string(),
        instance_id,
        episode: episode.clone(),
    })
    .map_err(|error| EpisodicSqliteProjectionError::Encoding(error.to_string()))?;

    // Threshold continuous HV to binary for the existing similarity index. This projection is an
    // index/search convenience only; exact restart state lives in the versioned metadata above.
    let mut bytes = [0u8; 2048];
    for (index, &value) in episode.input.values.iter().enumerate() {
        if index / 8 < bytes.len() && value > 0.0 {
            bytes[index / 8] |= 1 << (index % 8);
        }
    }

    Ok(MemoryRecord {
        id: canonical_episode_record_key(instance_id),
        memory_type: MemoryType::Episodic,
        encoding: symthaea_core::hdc::binary_hv::BinaryHV(bytes),
        content: String::new(),
        timestamp_ms: episode.timestamp.saturating_mul(20), // ~20ms/cycle at 50Hz legacy clock
        valence: episode.valence.unwrap_or(0.0),
        arousal: 0.5,
        psi: episode.psi,
        topics: Vec::new(),
        metadata,
        consolidation_strength: episode.consolidation_strength,
        retrieval_count: episode.retrieval_count,
    })
}

/// Decode and validate exact episode state from a versioned SQLite projection.
///
/// This function does not activate the episode. Restart activation remains a separate fail-closed
/// decision and canonical-import step.
pub(super) fn decode_projected_episode(
    metadata: &str,
) -> Result<Episode, EpisodicSqliteProjectionError> {
    let projection: EpisodicSqliteProjectionV1 = serde_json::from_str(metadata)
        .map_err(|error| EpisodicSqliteProjectionError::Decoding(error.to_string()))?;
    if projection.schema_version != EPISODIC_SQLITE_PROJECTION_SCHEMA {
        return Err(EpisodicSqliteProjectionError::UnsupportedSchema(
            projection.schema_version,
        ));
    }
    if projection.episode.instance_id != Some(projection.instance_id) {
        return Err(EpisodicSqliteProjectionError::InstanceIdentityMismatch);
    }
    validate_episode_state(&projection.episode)?;
    Ok(projection.episode)
}

fn validate_episode_state(episode: &Episode) -> Result<(), EpisodicSqliteProjectionError> {
    if !episode.psi.is_finite() {
        return Err(EpisodicSqliteProjectionError::NonFiniteField("psi"));
    }
    if !episode.consolidation_strength.is_finite() {
        return Err(EpisodicSqliteProjectionError::NonFiniteField(
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
            return Err(EpisodicSqliteProjectionError::NonFiniteField(name));
        }
    }
    if episode.input.values.iter().any(|value| !value.is_finite()) {
        return Err(EpisodicSqliteProjectionError::NonFiniteField("input"));
    }
    if episode.output.values.iter().any(|value| !value.is_finite()) {
        return Err(EpisodicSqliteProjectionError::NonFiniteField("output"));
    }
    if episode
        .bath_state_at_encoding
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(EpisodicSqliteProjectionError::NonFiniteField(
            "bath_state_at_encoding",
        ));
    }
    if episode
        .semantic_embedding
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(EpisodicSqliteProjectionError::NonFiniteField(
            "semantic_embedding",
        ));
    }
    Ok(())
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(super) enum EpisodicSqliteProjectionError {
    #[error("canonical episodic persistence requires an assigned occurrence UUID")]
    MissingInstanceId,
    #[error("episodic SQLite projection has unsupported schema {0:?}")]
    UnsupportedSchema(String),
    #[error("episodic SQLite projection UUID disagrees with embedded episode")]
    InstanceIdentityMismatch,
    #[error("episodic SQLite projection field is non-finite: {0}")]
    NonFiniteField(&'static str),
    #[error("could not encode episodic SQLite projection: {0}")]
    Encoding(String),
    #[error("could not decode episodic SQLite projection: {0}")]
    Decoding(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{EpisodicMemory, EpisodicReplayConfig};

    fn canonical_episode() -> Episode {
        let source = Episode::with_metadata(
            ContinuousHV::from_values(vec![1.0, 2.0, 3.0]),
            ContinuousHV::from_values(vec![4.0, 5.0, 6.0]),
            0.82,
            42,
            0.3,
            -0.4,
            0.9,
        )
        .with_dopamine(0.7)
        .with_bath_state([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        .with_semantic_embedding(vec![0.25, -0.5, 0.75]);
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        memory.store_if_significant_with_id(source).unwrap();
        let mut stored = memory.get_top_episode_instances(1);
        stored.remove(0).1
    }

    #[test]
    fn stable_key_and_full_lifecycle_state_round_trip() {
        let mut episode = canonical_episode();
        episode.replay_count = 7;
        episode.retrieval_count = 5;
        episode.consolidation_strength = 3.25;
        let id = episode.instance_id.unwrap();

        let record = project_episode_to_memory_record(&episode).unwrap();
        assert_eq!(record.id, canonical_episode_record_key(id));
        assert_eq!(record.consolidation_strength, 3.25);
        assert_eq!(record.retrieval_count, 5);

        let decoded = decode_projected_episode(&record.metadata).unwrap();
        assert_eq!(decoded.instance_id, Some(id));
        assert_eq!(decoded.replay_count, 7);
        assert_eq!(decoded.retrieval_count, 5);
        assert_eq!(decoded.consolidation_strength, 3.25);
        assert_eq!(decoded.input.values, episode.input.values);
        assert_eq!(decoded.output.values, episode.output.values);
    }

    #[test]
    fn identical_content_occurrences_receive_distinct_durable_keys() {
        let source = Episode::new(
            ContinuousHV::from_values(vec![1.0, 2.0]),
            ContinuousHV::from_values(vec![3.0, 4.0]),
            0.9,
            10,
        );
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let first = memory.store_if_significant_with_id(source.clone()).unwrap();
        let second = memory.store_if_significant_with_id(source).unwrap();
        assert_ne!(first, second);
        assert_ne!(
            canonical_episode_record_key(first),
            canonical_episode_record_key(second)
        );
    }

    #[test]
    fn unassigned_legacy_episode_is_not_promoted_to_exact_persistence() {
        let episode = Episode::new(
            ContinuousHV::from_values(vec![1.0]),
            ContinuousHV::from_values(vec![2.0]),
            0.9,
            1,
        );
        assert!(matches!(
            project_episode_to_memory_record(&episode),
            Err(EpisodicSqliteProjectionError::MissingInstanceId)
        ));
    }

    #[test]
    fn non_finite_state_is_rejected_before_json_projection() {
        let mut episode = canonical_episode();
        episode.consolidation_strength = f64::NAN;
        assert!(matches!(
            project_episode_to_memory_record(&episode),
            Err(EpisodicSqliteProjectionError::NonFiniteField(
                "consolidation_strength"
            ))
        ));
    }
}
