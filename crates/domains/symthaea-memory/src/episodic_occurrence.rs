// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Occurrence provenance for episodic records.
//!
//! This module answers a deliberately narrower question than semantic provenance:
//! "did this episodic record occur in the cognitive process?" It does **not** claim
//! that the episode's cognitive output is physically true. Occurrence identity is
//! domain-separated from the episode's semantic/content identity.

use crate::{episode_subject_sha256, Episode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;

const OCCURRENCE_DOMAIN: &[u8] = b"SYMTHAEA-EPISODIC-OCCURRENCE-v1\0";

pub fn episode_occurrence_subject_sha256(
    episode: &Episode,
    event_id: &str,
    recorder_id: &str,
) -> Result<String, EpisodicOccurrenceError> {
    if event_id.trim().is_empty() {
        return Err(EpisodicOccurrenceError::EmptyEventId);
    }
    if recorder_id.trim().is_empty() {
        return Err(EpisodicOccurrenceError::EmptyRecorderId);
    }
    let episode_sha = episode_subject_sha256(episode);
    let mut hasher = Sha256::new();
    hasher.update(OCCURRENCE_DOMAIN);
    hash_text(&mut hasher, &episode_sha);
    hash_text(&mut hasher, event_id);
    hash_text(&mut hasher, recorder_id);
    hasher.update(episode.timestamp.to_be_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_text(hasher: &mut Sha256, text: &str) {
    let bytes = text.as_bytes();
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

/// Immutable record that a particular episode was observed by a named recorder at a
/// particular cognitive event. This is occurrence evidence only; it conveys no truth
/// status for the episode's input/output semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodicOccurrenceRecord {
    pub occurrence_subject_sha256: String,
    pub episode_subject_sha256: String,
    pub event_id: String,
    pub recorder_id: String,
    pub event_time: u64,
}

impl EpisodicOccurrenceRecord {
    pub fn new(
        episode: &Episode,
        event_id: impl Into<String>,
        recorder_id: impl Into<String>,
    ) -> Result<Self, EpisodicOccurrenceError> {
        let event_id = event_id.into();
        let recorder_id = recorder_id.into();
        let occurrence_subject_sha256 =
            episode_occurrence_subject_sha256(episode, &event_id, &recorder_id)?;
        Ok(Self {
            occurrence_subject_sha256,
            episode_subject_sha256: episode_subject_sha256(episode),
            event_id,
            recorder_id,
            event_time: episode.timestamp,
        })
    }

    pub fn validate(&self, episode: &Episode) -> Result<(), EpisodicOccurrenceError> {
        let expected_episode = episode_subject_sha256(episode);
        if self.episode_subject_sha256 != expected_episode {
            return Err(EpisodicOccurrenceError::EpisodeSubjectMismatch {
                expected: expected_episode,
                got: self.episode_subject_sha256.clone(),
            });
        }
        if self.event_time != episode.timestamp {
            return Err(EpisodicOccurrenceError::EventTimeMismatch {
                expected: episode.timestamp,
                got: self.event_time,
            });
        }
        let expected_occurrence =
            episode_occurrence_subject_sha256(episode, &self.event_id, &self.recorder_id)?;
        if self.occurrence_subject_sha256 != expected_occurrence {
            return Err(EpisodicOccurrenceError::OccurrenceSubjectMismatch {
                expected: expected_occurrence,
                got: self.occurrence_subject_sha256.clone(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpisodicOccurrenceIndex {
    by_occurrence_sha256: HashMap<String, EpisodicOccurrenceRecord>,
}

impl EpisodicOccurrenceIndex {
    pub fn len(&self) -> usize {
        self.by_occurrence_sha256.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_occurrence_sha256.is_empty()
    }

    pub fn get(&self, occurrence_sha256: &str) -> Option<&EpisodicOccurrenceRecord> {
        self.by_occurrence_sha256.get(occurrence_sha256)
    }

    pub fn preflight(
        &self,
        episode: &Episode,
        record: &EpisodicOccurrenceRecord,
    ) -> Result<(), EpisodicOccurrenceError> {
        record.validate(episode)?;
        if let Some(existing) = self.by_occurrence_sha256.get(&record.occurrence_subject_sha256) {
            if existing != record {
                return Err(EpisodicOccurrenceError::ConflictingImmutableBinding {
                    occurrence_sha256: record.occurrence_subject_sha256.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn attach(
        &mut self,
        episode: &Episode,
        record: EpisodicOccurrenceRecord,
    ) -> Result<(), EpisodicOccurrenceError> {
        self.preflight(episode, &record)?;
        self.by_occurrence_sha256
            .entry(record.occurrence_subject_sha256.clone())
            .or_insert(record);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.by_occurrence_sha256.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodicOccurrenceError {
    EmptyEventId,
    EmptyRecorderId,
    EpisodeSubjectMismatch { expected: String, got: String },
    EventTimeMismatch { expected: u64, got: u64 },
    OccurrenceSubjectMismatch { expected: String, got: String },
    ConflictingImmutableBinding { occurrence_sha256: String },
}

impl fmt::Display for EpisodicOccurrenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEventId => write!(f, "episodic occurrence event_id must be non-empty"),
            Self::EmptyRecorderId => write!(f, "episodic occurrence recorder_id must be non-empty"),
            Self::EpisodeSubjectMismatch { expected, got } => write!(
                f,
                "occurrence episode subject mismatch: expected {expected}, got {got}"
            ),
            Self::EventTimeMismatch { expected, got } => write!(
                f,
                "occurrence event time mismatch: expected {expected}, got {got}"
            ),
            Self::OccurrenceSubjectMismatch { expected, got } => write!(
                f,
                "occurrence subject mismatch: expected {expected}, got {got}"
            ),
            Self::ConflictingImmutableBinding { occurrence_sha256 } => write!(
                f,
                "occurrence binding is immutable for {occurrence_sha256}"
            ),
        }
    }
}

impl std::error::Error for EpisodicOccurrenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    fn episode(seed: u64) -> Episode {
        Episode::new(
            ContinuousHV::random(64, seed),
            ContinuousHV::random(64, seed + 1),
            0.8,
            seed,
        )
    }

    #[test]
    fn occurrence_identity_is_distinct_from_semantic_episode_identity() {
        let ep = episode(1);
        let record = EpisodicOccurrenceRecord::new(&ep, "cycle-1", "cognitive-loop").unwrap();
        assert_ne!(record.occurrence_subject_sha256, episode_subject_sha256(&ep));
        record.validate(&ep).unwrap();
    }

    #[test]
    fn same_semantic_episode_different_event_identity_changes_occurrence_digest() {
        let ep = episode(2);
        let a = EpisodicOccurrenceRecord::new(&ep, "event-a", "cognitive-loop").unwrap();
        let b = EpisodicOccurrenceRecord::new(&ep, "event-b", "cognitive-loop").unwrap();
        assert_ne!(a.occurrence_subject_sha256, b.occurrence_subject_sha256);
    }

    #[test]
    fn occurrence_record_does_not_mutate_or_ground_episode_content() {
        let ep = episode(3);
        let before = episode_subject_sha256(&ep);
        let record = EpisodicOccurrenceRecord::new(&ep, "cycle-3", "cognitive-loop").unwrap();
        record.validate(&ep).unwrap();
        assert_eq!(before, episode_subject_sha256(&ep));
    }

    #[test]
    fn tampered_occurrence_rejects() {
        let ep = episode(4);
        let mut record = EpisodicOccurrenceRecord::new(&ep, "cycle-4", "cognitive-loop").unwrap();
        record.event_id = "different".into();
        assert!(matches!(
            record.validate(&ep),
            Err(EpisodicOccurrenceError::OccurrenceSubjectMismatch { .. })
        ));
    }

    #[test]
    fn conflicting_index_rewrite_rejects() {
        let ep = episode(5);
        let record = EpisodicOccurrenceRecord::new(&ep, "cycle-5", "cognitive-loop").unwrap();
        let mut index = EpisodicOccurrenceIndex::default();
        index.attach(&ep, record.clone()).unwrap();
        let mut changed = record.clone();
        changed.recorder_id = "other".into();
        assert!(index.attach(&ep, changed).is_err());
    }
}
