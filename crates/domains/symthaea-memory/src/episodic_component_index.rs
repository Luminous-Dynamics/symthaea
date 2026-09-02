// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable sidecar index for component-level episodic provenance.

use crate::{episode_subject_sha256, Episode, EpisodicComponentProvenance, EpisodicComponentProvenanceError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpisodicComponentProvenanceIndex {
    by_episode_sha256: HashMap<String, EpisodicComponentProvenance>,
}

impl EpisodicComponentProvenanceIndex {
    pub fn len(&self) -> usize {
        self.by_episode_sha256.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_episode_sha256.is_empty()
    }

    pub fn get(&self, episode_sha256: &str) -> Option<&EpisodicComponentProvenance> {
        self.by_episode_sha256.get(episode_sha256)
    }

    /// Validate a prospective immutable binding without mutating the index.
    pub fn preflight(
        &self,
        episode: &Episode,
        record: &EpisodicComponentProvenance,
    ) -> Result<(), EpisodicComponentIndexError> {
        record.validate(episode)?;
        let expected = episode_subject_sha256(episode);
        if record.episode_subject_sha256 != expected {
            return Err(EpisodicComponentIndexError::EpisodeDigestMismatch {
                expected,
                got: record.episode_subject_sha256.clone(),
            });
        }
        if let Some(existing) = self.by_episode_sha256.get(&expected) {
            if existing != record {
                return Err(EpisodicComponentIndexError::ConflictingImmutableBinding {
                    episode_sha256: expected,
                });
            }
        }
        Ok(())
    }

    /// Attach an already-validated record. Existing identical bindings are idempotent;
    /// conflicting rewrites fail closed.
    pub fn attach(
        &mut self,
        episode: &Episode,
        record: EpisodicComponentProvenance,
    ) -> Result<(), EpisodicComponentIndexError> {
        self.preflight(episode, &record)?;
        self.by_episode_sha256
            .entry(record.episode_subject_sha256.clone())
            .or_insert(record);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.by_episode_sha256.clear();
    }
}

#[derive(Debug)]
pub enum EpisodicComponentIndexError {
    EpisodeDigestMismatch { expected: String, got: String },
    ConflictingImmutableBinding { episode_sha256: String },
    Component(EpisodicComponentProvenanceError),
}

impl fmt::Display for EpisodicComponentIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EpisodeDigestMismatch { expected, got } => write!(
                f,
                "component index episode digest mismatch: expected {expected}, got {got}"
            ),
            Self::ConflictingImmutableBinding { episode_sha256 } => write!(
                f,
                "component provenance binding is immutable for episode {episode_sha256}"
            ),
            Self::Component(error) => write!(f, "invalid episodic component provenance: {error}"),
        }
    }
}

impl std::error::Error for EpisodicComponentIndexError {}

impl From<EpisodicComponentProvenanceError> for EpisodicComponentIndexError {
    fn from(value: EpisodicComponentProvenanceError) -> Self {
        Self::Component(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{episode_cognition_subject_sha256, episode_perception_subject_sha256};
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_epistemic_types::{GroundingEvidence, ProvenanceEnvelope, RealityDomain};

    fn episode(seed: u64) -> Episode {
        Episode::new(
            ContinuousHV::random(64, seed),
            ContinuousHV::random(64, seed + 1),
            0.8,
            seed,
        )
    }

    fn record(ep: &Episode) -> EpisodicComponentProvenance {
        let perception = ProvenanceEnvelope::from_grounding(
            GroundingEvidence::direct_observation(
                episode_perception_subject_sha256(ep),
                "obs",
                "sensor",
                None,
                0.95,
            )
            .unwrap(),
        );
        let cognition = ProvenanceEnvelope::new(
            episode_cognition_subject_sha256(ep),
            RealityDomain::Unknown,
            vec!["cognitive-loop".into()],
            None,
            0.8,
        )
        .unwrap();
        EpisodicComponentProvenance::compose(
            ep,
            perception,
            cognition,
            "symthaea.episode-compose",
            "v1",
        )
        .unwrap()
    }

    #[test]
    fn identical_reattach_is_idempotent() {
        let ep = episode(1);
        let rec = record(&ep);
        let mut index = EpisodicComponentProvenanceIndex::default();
        index.attach(&ep, rec.clone()).unwrap();
        index.attach(&ep, rec).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn conflicting_rewrite_fails_closed() {
        let ep = episode(2);
        let rec = record(&ep);
        let mut changed = rec.clone();
        changed.cognition.confidence = 0.4;
        let mut index = EpisodicComponentProvenanceIndex::default();
        index.attach(&ep, rec).unwrap();
        assert!(matches!(
            index.attach(&ep, changed),
            Err(EpisodicComponentIndexError::ConflictingImmutableBinding { .. })
        ));
    }

    #[test]
    fn tampered_component_record_is_rejected_before_mutation() {
        let ep = episode(3);
        let mut rec = record(&ep);
        rec.episode_derivation.parent_subject_sha256s.reverse();
        let mut index = EpisodicComponentProvenanceIndex::default();
        assert!(index.attach(&ep, rec).is_err());
        assert!(index.is_empty());
    }
}
