// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Component-level epistemic provenance for episodic memory.
//!
//! An [`Episode`](crate::episodic_replay::Episode) is a composite record: its input
//! represents encoded/perceived state while its output represents Symthaea's own
//! cognitive response. Treating that whole record as `PhysicalGrounded` would therefore
//! launder an internal inference into history.
//!
//! This module gives those components independent, domain-separated identities and binds
//! the composite episode to them with an immutable derivation receipt. It is additive and
//! does not change the historical `Episode` serialization format.

use crate::{episode_subject_sha256, Episode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use symthaea_epistemic_types::{
    derive_with_receipt, EpistemicDerivationError, EpistemicDerivationReceipt,
    EpistemicTransformKind, ProvenanceEnvelope, RealityDomain,
};

const PERCEPTION_DOMAIN: &[u8] = b"SYMTHAEA-EPISODE-PERCEPTION-v1\0";
const COGNITION_DOMAIN: &[u8] = b"SYMTHAEA-EPISODE-COGNITION-v1\0";

/// Stable SHA-256 identity for the encoded/perceived component of an episode.
///
/// The timestamp is included to distinguish identical encoded observations occurring at
/// different cognitive events. Internal output, replay counters, prediction error, and
/// other response-side metadata are intentionally excluded.
pub fn episode_perception_subject_sha256(episode: &Episode) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PERCEPTION_DOMAIN);
    hash_f32_slice(&mut hasher, &episode.input.values);
    hasher.update(episode.timestamp.to_be_bytes());
    format!("{:x}", hasher.finalize())
}

/// Stable SHA-256 identity for Symthaea's internal cognitive response component.
///
/// The event timestamp is included, while perception-side input and mutable replay state
/// are excluded.
pub fn episode_cognition_subject_sha256(episode: &Episode) -> String {
    let mut hasher = Sha256::new();
    hasher.update(COGNITION_DOMAIN);
    hash_f32_slice(&mut hasher, &episode.output.values);
    hasher.update(episode.timestamp.to_be_bytes());
    format!("{:x}", hasher.finalize())
}

fn hash_f32_slice(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_be_bytes());
    for value in values {
        hasher.update(value.to_bits().to_be_bytes());
    }
}

/// Provenance for the independently identified components of one episodic record.
///
/// The composite `episode` envelope is deliberately derived and ungrounded. Its parents
/// are the perception and cognition components, and `episode_derivation` proves that exact
/// parent relation. Grounding either component never implicitly grounds the other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodicComponentProvenance {
    pub episode_subject_sha256: String,
    pub perception_subject_sha256: String,
    pub cognition_subject_sha256: String,
    pub perception: ProvenanceEnvelope,
    pub cognition: ProvenanceEnvelope,
    pub episode: ProvenanceEnvelope,
    pub episode_derivation: EpistemicDerivationReceipt,
}

impl EpisodicComponentProvenance {
    /// Construct a composite provenance record from already-explicit component
    /// provenance. This function does not infer or manufacture grounding.
    pub fn compose(
        episode: &Episode,
        perception: ProvenanceEnvelope,
        cognition: ProvenanceEnvelope,
        transform_id: impl Into<String>,
        transform_version: impl Into<String>,
    ) -> Result<Self, EpisodicComponentProvenanceError> {
        let episode_subject_sha256 = episode_subject_sha256(episode);
        let perception_subject_sha256 = episode_perception_subject_sha256(episode);
        let cognition_subject_sha256 = episode_cognition_subject_sha256(episode);

        if perception.subject_sha256 != perception_subject_sha256 {
            return Err(EpisodicComponentProvenanceError::PerceptionSubjectMismatch {
                expected: perception_subject_sha256,
                got: perception.subject_sha256,
            });
        }
        if cognition.subject_sha256 != cognition_subject_sha256 {
            return Err(EpisodicComponentProvenanceError::CognitionSubjectMismatch {
                expected: cognition_subject_sha256,
                got: cognition.subject_sha256,
            });
        }

        let confidence = perception.confidence.min(cognition.confidence);
        let (composite, receipt) = derive_with_receipt(
            episode_subject_sha256.clone(),
            RealityDomain::Unknown,
            vec!["symthaea.episodic-recorder".into()],
            None,
            confidence,
            &[perception.clone(), cognition.clone()],
            EpistemicTransformKind::Other,
            transform_id,
            transform_version,
        )?;

        let record = Self {
            episode_subject_sha256,
            perception_subject_sha256,
            cognition_subject_sha256,
            perception,
            cognition,
            episode: composite,
            episode_derivation: receipt,
        };
        record.validate(episode)?;
        Ok(record)
    }

    /// Verify all component identities and the exact composite parent relation.
    pub fn validate(&self, episode: &Episode) -> Result<(), EpisodicComponentProvenanceError> {
        let expected_episode = episode_subject_sha256(episode);
        let expected_perception = episode_perception_subject_sha256(episode);
        let expected_cognition = episode_cognition_subject_sha256(episode);

        if self.episode_subject_sha256 != expected_episode
            || self.episode.subject_sha256 != expected_episode
            || self.episode_derivation.child_subject_sha256 != expected_episode
        {
            return Err(EpisodicComponentProvenanceError::EpisodeSubjectMismatch {
                expected: expected_episode,
            });
        }
        if self.perception_subject_sha256 != expected_perception
            || self.perception.subject_sha256 != expected_perception
        {
            return Err(EpisodicComponentProvenanceError::PerceptionSubjectMismatch {
                expected: expected_perception,
                got: self.perception.subject_sha256.clone(),
            });
        }
        if self.cognition_subject_sha256 != expected_cognition
            || self.cognition.subject_sha256 != expected_cognition
        {
            return Err(EpisodicComponentProvenanceError::CognitionSubjectMismatch {
                expected: expected_cognition,
                got: self.cognition.subject_sha256.clone(),
            });
        }

        let expected_parents = vec![
            self.perception_subject_sha256.clone(),
            self.cognition_subject_sha256.clone(),
        ];
        if self.episode_derivation.parent_subject_sha256s != expected_parents {
            return Err(EpisodicComponentProvenanceError::CompositeParentMismatch);
        }
        if self.episode.domain.is_grounded() {
            return Err(EpisodicComponentProvenanceError::CompositeMustRemainDerived);
        }
        Ok(())
    }

    /// True only when the perception component is actually grounded and untainted.
    /// This says nothing about the cognition component or the composite episode.
    pub const fn perception_is_grounded(&self) -> bool {
        self.perception.may_enter_grounded_history()
    }

    /// The composite may carry counterfactual taint from either component.
    pub const fn composite_is_counterfactually_tainted(&self) -> bool {
        self.episode.counterfactual_taint
    }
}

#[derive(Debug)]
pub enum EpisodicComponentProvenanceError {
    EpisodeSubjectMismatch { expected: String },
    PerceptionSubjectMismatch { expected: String, got: String },
    CognitionSubjectMismatch { expected: String, got: String },
    CompositeParentMismatch,
    CompositeMustRemainDerived,
    Derivation(EpistemicDerivationError),
}

impl fmt::Display for EpisodicComponentProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EpisodeSubjectMismatch { expected } => {
                write!(f, "episode component provenance does not bind episode {expected}")
            }
            Self::PerceptionSubjectMismatch { expected, got } => write!(
                f,
                "perception provenance subject mismatch: expected {expected}, got {got}"
            ),
            Self::CognitionSubjectMismatch { expected, got } => write!(
                f,
                "cognition provenance subject mismatch: expected {expected}, got {got}"
            ),
            Self::CompositeParentMismatch => {
                write!(f, "episode derivation must bind exact perception+cognition parents")
            }
            Self::CompositeMustRemainDerived => {
                write!(f, "composite episodic record cannot be declared grounded by composition")
            }
            Self::Derivation(error) => write!(f, "episode component derivation failed: {error}"),
        }
    }
}

impl std::error::Error for EpisodicComponentProvenanceError {}

impl From<EpistemicDerivationError> for EpisodicComponentProvenanceError {
    fn from(value: EpistemicDerivationError) -> Self {
        Self::Derivation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_epistemic_types::{GroundingEvidence, ProvenanceEnvelope};

    fn episode(seed: u64) -> Episode {
        Episode::new(
            ContinuousHV::random(64, seed),
            ContinuousHV::random(64, seed + 1),
            0.8,
            seed,
        )
    }

    fn grounded_perception(ep: &Episode) -> ProvenanceEnvelope {
        let subject = episode_perception_subject_sha256(ep);
        ProvenanceEnvelope::from_grounding(
            GroundingEvidence::direct_observation(subject, "obs", "sensor", None, 0.95).unwrap(),
        )
    }

    fn inferred_cognition(ep: &Episode) -> ProvenanceEnvelope {
        ProvenanceEnvelope::new(
            episode_cognition_subject_sha256(ep),
            RealityDomain::Unknown,
            vec!["symthaea.cognitive-loop".into()],
            None,
            0.8,
        )
        .unwrap()
    }

    #[test]
    fn component_digests_are_domain_separated_and_stable_across_replay_state() {
        let ep = episode(10);
        let p = episode_perception_subject_sha256(&ep);
        let c = episode_cognition_subject_sha256(&ep);
        assert_ne!(p, c);
        let mut replayed = ep.clone();
        replayed.replay_count = 99;
        replayed.retrieval_count = 42;
        assert_eq!(p, episode_perception_subject_sha256(&replayed));
        assert_eq!(c, episode_cognition_subject_sha256(&replayed));
    }

    #[test]
    fn grounded_perception_does_not_ground_internal_cognition_or_composite_episode() {
        let ep = episode(11);
        let record = EpisodicComponentProvenance::compose(
            &ep,
            grounded_perception(&ep),
            inferred_cognition(&ep),
            "symthaea.episode-compose",
            "v1",
        )
        .unwrap();

        assert!(record.perception_is_grounded());
        assert!(!record.cognition.may_enter_grounded_history());
        assert!(!record.episode.may_enter_grounded_history());
        assert_eq!(record.episode_derivation.parent_count(), 2);
        record.validate(&ep).unwrap();
    }

    #[test]
    fn counterfactual_cognition_taints_composite_without_tainting_observation() {
        let ep = episode(12);
        let counterfactual = ProvenanceEnvelope::new(
            episode_cognition_subject_sha256(&ep),
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            None,
            0.7,
        )
        .unwrap();
        let record = EpisodicComponentProvenance::compose(
            &ep,
            grounded_perception(&ep),
            counterfactual,
            "symthaea.episode-compose",
            "v1",
        )
        .unwrap();

        assert!(record.perception_is_grounded());
        assert!(record.composite_is_counterfactually_tainted());
        assert!(!record.perception.counterfactual_taint);
    }

    #[test]
    fn wrong_component_subjects_fail_closed() {
        let ep = episode(13);
        let wrong = ProvenanceEnvelope::new(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            RealityDomain::Unknown,
            vec!["sensor".into()],
            None,
            0.5,
        )
        .unwrap();
        assert!(matches!(
            EpisodicComponentProvenance::compose(
                &ep,
                wrong,
                inferred_cognition(&ep),
                "symthaea.episode-compose",
                "v1",
            ),
            Err(EpisodicComponentProvenanceError::PerceptionSubjectMismatch { .. })
        ));
    }

    #[test]
    fn tampered_parent_binding_rejects() {
        let ep = episode(14);
        let mut record = EpisodicComponentProvenance::compose(
            &ep,
            grounded_perception(&ep),
            inferred_cognition(&ep),
            "symthaea.episode-compose",
            "v1",
        )
        .unwrap();
        record.episode_derivation.parent_subject_sha256s.reverse();
        assert!(matches!(
            record.validate(&ep),
            Err(EpisodicComponentProvenanceError::CompositeParentMismatch)
        ));
    }
}
