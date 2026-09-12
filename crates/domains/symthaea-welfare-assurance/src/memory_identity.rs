// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stable content identity for episodic-memory evidence and future quarantine workflows.
//!
//! `EpisodeContentId` is intentionally **not** a per-instance object identity. Two independently
//! stored episodes with identical encoding-time content receive the same ID. A future quarantine
//! implementation that must distinguish duplicate instances needs an insertion lineage/ordinal
//! owned by `symthaea-memory` itself.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_memory::episodic_replay::Episode;
use thiserror::Error;

const EPISODE_CONTENT_ID_SCHEMA: &str = "symthaea.welfare.episode-content-id.v1";
const EPISODE_CONTENT_ID_DOMAIN: &[u8] = b"symthaea.welfare.episode-content-id.v1\0";

/// Replay-stable digest of the encoded event represented by an `Episode`.
///
/// Lifecycle fields (`replay_count`, `consolidation_strength`, `retrieval_count`) are deliberately
/// excluded so ordinary replay/reconsolidation does not change content identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeContentId(Sha256Digest);

impl EpisodeContentId {
    pub fn digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.0.0 {
            use std::fmt::Write as _;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }
}

#[derive(Serialize)]
struct EpisodeEncodingIdentity<'a> {
    schema_version: &'static str,
    input: &'a ContinuousHV,
    output: &'a ContinuousHV,
    psi: f64,
    timestamp: u64,
    prediction_error: Option<f32>,
    valence: Option<f32>,
    coherence: Option<f32>,
    dopamine_at_encoding: Option<f32>,
    bath_state_at_encoding: Option<[f32; 9]>,
    semantic_embedding: &'a Option<Vec<f32>>,
}

/// Derive stable content identity from encoding-time fields only.
pub fn episode_content_id(episode: &Episode) -> Result<EpisodeContentId, EpisodeContentIdError> {
    let identity = EpisodeEncodingIdentity {
        schema_version: EPISODE_CONTENT_ID_SCHEMA,
        input: &episode.input,
        output: &episode.output,
        psi: episode.psi,
        timestamp: episode.timestamp,
        prediction_error: episode.prediction_error,
        valence: episode.valence,
        coherence: episode.coherence,
        dopamine_at_encoding: episode.dopamine_at_encoding,
        bath_state_at_encoding: episode.bath_state_at_encoding,
        semantic_embedding: &episode.semantic_embedding,
    };
    let encoded = bincode::serialize(&identity)
        .map_err(|error| EpisodeContentIdError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(EPISODE_CONTENT_ID_DOMAIN);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(EpisodeContentId(hasher.finalize()))
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EpisodeContentIdError {
    #[error("could not encode stable episode content identity: {0}")]
    Encoding(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_memory::episodic_replay::{EpisodicMemory, EpisodicReplayConfig};

    fn episode(seed: f32) -> Episode {
        Episode::with_metadata(
            ContinuousHV::from_values(vec![seed, seed + 1.0, seed + 2.0]),
            ContinuousHV::from_values(vec![seed + 3.0, seed + 4.0, seed + 5.0]),
            0.82,
            42,
            0.3,
            -0.4,
            0.9,
        )
        .with_dopamine(0.7)
        .with_bath_state([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        .with_semantic_embedding(vec![0.25, -0.5, 0.75])
    }

    #[test]
    fn replay_and_reconsolidation_bookkeeping_do_not_change_content_identity() {
        let mut episode = episode(1.0);
        let before = episode_content_id(&episode).unwrap();

        episode.replay_count = 17;
        episode.retrieval_count = 9;
        episode.consolidation_strength = 4.25;
        episode.reconsolidate(0.95);

        assert_eq!(before, episode_content_id(&episode).unwrap());
    }

    #[test]
    fn storage_occurrence_identity_does_not_change_content_identity() {
        let source = episode(1.5);
        let before = episode_content_id(&source).unwrap();
        assert!(source.instance_id.is_none());

        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let instance_id = memory
            .store_if_significant_with_id(source)
            .expect("episode should be stored");
        let (_, stored) = memory
            .get_top_episode_instances(1)
            .into_iter()
            .next()
            .expect("stored episode should be visible");

        assert_eq!(stored.instance_id, Some(instance_id));
        assert_eq!(before, episode_content_id(&stored).unwrap());
    }

    #[test]
    fn encoded_event_changes_do_change_content_identity() {
        let mut episode = episode(1.0);
        let before = episode_content_id(&episode).unwrap();
        episode.valence = Some(0.4);
        assert_ne!(before, episode_content_id(&episode).unwrap());
    }

    #[test]
    fn byte_identical_encoded_events_share_content_identity() {
        let left = episode(2.0);
        let right = left.clone();
        assert_eq!(
            episode_content_id(&left).unwrap(),
            episode_content_id(&right).unwrap()
        );
    }

    #[test]
    fn content_identity_is_not_zero() {
        assert_ne!(episode_content_id(&episode(3.0)).unwrap().digest().0, [0; 32]);
    }
}
