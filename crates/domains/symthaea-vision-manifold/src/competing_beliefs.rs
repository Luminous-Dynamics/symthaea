// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Competing visual belief sets for VIS-004B.
//!
//! Semantic-class uncertainty and identity uncertainty are intentionally represented as separate
//! questions. Candidate masses are never silently renormalized; every set carries explicit
//! unassigned mass so weak evidence can remain weak.

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

use crate::entity_identity::{VisualEntityHypothesisRef, VisualTrackRef};
use crate::epistemic::{VisualEvidence, VisualOrigin};

const MASS_EPSILON: f32 = 1.0e-5;
const MAX_LABEL_LEN: usize = 256;
const MAX_VOCABULARY_LEN: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BeliefMass(f32);

impl BeliefMass {
    pub fn new(value: f32) -> Result<Self, CompetingBeliefError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(CompetingBeliefError::InvalidMass);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl<'de> Deserialize<'de> for BeliefMass {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SemanticClassCandidate {
    label: String,
    mass: BeliefMass,
    evidence: VisualEvidence,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct SemanticClassCandidateWire {
    label: String,
    mass: BeliefMass,
    evidence: VisualEvidence,
}

impl SemanticClassCandidate {
    pub fn new(
        label: impl Into<String>,
        mass: BeliefMass,
        evidence: VisualEvidence,
    ) -> Result<Self, CompetingBeliefError> {
        let label = label.into();
        validate_text(&label, MAX_LABEL_LEN, CompetingBeliefError::EmptyLabel, CompetingBeliefError::LabelTooLong)?;
        validate_historical_inference(&evidence)?;
        Ok(Self {
            label,
            mass,
            evidence,
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn mass(&self) -> BeliefMass {
        self.mass
    }

    pub fn evidence(&self) -> &VisualEvidence {
        &self.evidence
    }
}

impl<'de> Deserialize<'de> for SemanticClassCandidate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SemanticClassCandidateWire::deserialize(deserializer)?;
        Self::new(wire.label, wire.mass, wire.evidence).map_err(serde::de::Error::custom)
    }
}

/// A categorical class belief for one belief-layer entity hypothesis.
///
/// Candidate masses plus `unassigned_mass` must sum to 1 within a small floating-point tolerance.
/// The constructor never rescales them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SemanticClassBeliefSet {
    subject: VisualEntityHypothesisRef,
    vocabulary: String,
    candidates: Vec<SemanticClassCandidate>,
    unassigned_mass: BeliefMass,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct SemanticClassBeliefSetWire {
    subject: VisualEntityHypothesisRef,
    vocabulary: String,
    candidates: Vec<SemanticClassCandidate>,
    unassigned_mass: BeliefMass,
}

impl SemanticClassBeliefSet {
    pub fn new(
        subject: VisualEntityHypothesisRef,
        vocabulary: impl Into<String>,
        candidates: Vec<SemanticClassCandidate>,
        unassigned_mass: BeliefMass,
    ) -> Result<Self, CompetingBeliefError> {
        let vocabulary = vocabulary.into();
        validate_text(
            &vocabulary,
            MAX_VOCABULARY_LEN,
            CompetingBeliefError::EmptyVocabulary,
            CompetingBeliefError::VocabularyTooLong,
        )?;
        validate_unique_semantic_candidates(&candidates)?;
        validate_mass_partition(
            candidates.iter().map(|candidate| candidate.mass.get()),
            unassigned_mass.get(),
        )?;
        Ok(Self {
            subject,
            vocabulary,
            candidates,
            unassigned_mass,
        })
    }

    pub const fn subject(&self) -> VisualEntityHypothesisRef {
        self.subject
    }

    pub fn vocabulary(&self) -> &str {
        &self.vocabulary
    }

    pub fn candidates(&self) -> &[SemanticClassCandidate] {
        &self.candidates
    }

    pub const fn unassigned_mass(&self) -> BeliefMass {
        self.unassigned_mass
    }

    pub fn most_supported(&self) -> Option<&SemanticClassCandidate> {
        self.candidates.iter().max_by(|a, b| {
            a.mass
                .get()
                .partial_cmp(&b.mass.get())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl<'de> Deserialize<'de> for SemanticClassBeliefSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SemanticClassBeliefSetWire::deserialize(deserializer)?;
        Self::new(
            wire.subject,
            wire.vocabulary,
            wire.candidates,
            wire.unassigned_mass,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum IdentityCandidateValue {
    ExistingEntity(VisualEntityHypothesisRef),
    NewEntity,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IdentityCandidate {
    value: IdentityCandidateValue,
    mass: BeliefMass,
    evidence: VisualEvidence,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct IdentityCandidateWire {
    value: IdentityCandidateValue,
    mass: BeliefMass,
    evidence: VisualEvidence,
}

impl IdentityCandidate {
    pub fn new(
        value: IdentityCandidateValue,
        mass: BeliefMass,
        evidence: VisualEvidence,
    ) -> Result<Self, CompetingBeliefError> {
        validate_historical_inference(&evidence)?;
        Ok(Self {
            value,
            mass,
            evidence,
        })
    }

    pub const fn value(&self) -> IdentityCandidateValue {
        self.value
    }

    pub const fn mass(&self) -> BeliefMass {
        self.mass
    }

    pub fn evidence(&self) -> &VisualEvidence {
        &self.evidence
    }
}

impl<'de> Deserialize<'de> for IdentityCandidate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IdentityCandidateWire::deserialize(deserializer)?;
        Self::new(wire.value, wire.mass, wire.evidence).map_err(serde::de::Error::custom)
    }
}

/// Competing identity explanations for one tracker-local track.
///
/// A track may be associated with an existing entity hypothesis, treated as a new entity, or left
/// partly/entirely unresolved through `unassigned_mass`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrackIdentityBeliefSet {
    subject: VisualTrackRef,
    candidates: Vec<IdentityCandidate>,
    unassigned_mass: BeliefMass,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct TrackIdentityBeliefSetWire {
    subject: VisualTrackRef,
    candidates: Vec<IdentityCandidate>,
    unassigned_mass: BeliefMass,
}

impl TrackIdentityBeliefSet {
    pub fn new(
        subject: VisualTrackRef,
        candidates: Vec<IdentityCandidate>,
        unassigned_mass: BeliefMass,
    ) -> Result<Self, CompetingBeliefError> {
        validate_unique_identity_candidates(&candidates)?;
        validate_mass_partition(
            candidates.iter().map(|candidate| candidate.mass.get()),
            unassigned_mass.get(),
        )?;
        Ok(Self {
            subject,
            candidates,
            unassigned_mass,
        })
    }

    pub const fn subject(&self) -> VisualTrackRef {
        self.subject
    }

    pub fn candidates(&self) -> &[IdentityCandidate] {
        &self.candidates
    }

    pub const fn unassigned_mass(&self) -> BeliefMass {
        self.unassigned_mass
    }

    pub fn most_supported(&self) -> Option<&IdentityCandidate> {
        self.candidates.iter().max_by(|a, b| {
            a.mass
                .get()
                .partial_cmp(&b.mass.get())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl<'de> Deserialize<'de> for TrackIdentityBeliefSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TrackIdentityBeliefSetWire::deserialize(deserializer)?;
        Self::new(wire.subject, wire.candidates, wire.unassigned_mass)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompetingBeliefError {
    InvalidMass,
    MassPartitionDoesNotSumToOne,
    EmptyLabel,
    LabelTooLong,
    EmptyVocabulary,
    VocabularyTooLong,
    TextContainsControlCharacter,
    DuplicateSemanticCandidate,
    DuplicateIdentityCandidate,
    ObservedCandidateEvidence,
    GenerativeCandidateEvidence,
}

impl fmt::Display for CompetingBeliefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidMass => "belief mass must be finite and within [0, 1]",
            Self::MassPartitionDoesNotSumToOne => {
                "candidate masses plus unassigned mass must sum to 1 without implicit renormalization"
            }
            Self::EmptyLabel => "semantic candidate label must be non-empty",
            Self::LabelTooLong => "semantic candidate label exceeds length limit",
            Self::EmptyVocabulary => "semantic vocabulary identity must be non-empty",
            Self::VocabularyTooLong => "semantic vocabulary identity exceeds length limit",
            Self::TextContainsControlCharacter => "belief text fields cannot contain control characters",
            Self::DuplicateSemanticCandidate => "semantic belief set contains duplicate labels",
            Self::DuplicateIdentityCandidate => "identity belief set contains duplicate candidates",
            Self::ObservedCandidateEvidence => {
                "semantic/identity candidate support is inferred, not direct observation"
            }
            Self::GenerativeCandidateEvidence => {
                "predicted/simulated/counterfactual support cannot enter historical belief state"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for CompetingBeliefError {}

fn validate_historical_inference(evidence: &VisualEvidence) -> Result<(), CompetingBeliefError> {
    match evidence.origin() {
        VisualOrigin::Inferred | VisualOrigin::Remembered => Ok(()),
        VisualOrigin::Observed => Err(CompetingBeliefError::ObservedCandidateEvidence),
        VisualOrigin::Predicted | VisualOrigin::Simulated | VisualOrigin::Counterfactual => {
            Err(CompetingBeliefError::GenerativeCandidateEvidence)
        }
    }
}

fn validate_mass_partition(
    candidate_masses: impl Iterator<Item = f32>,
    unassigned_mass: f32,
) -> Result<(), CompetingBeliefError> {
    let sum = candidate_masses.fold(unassigned_mass as f64, |acc, mass| acc + mass as f64);
    if (sum - 1.0).abs() > MASS_EPSILON as f64 {
        return Err(CompetingBeliefError::MassPartitionDoesNotSumToOne);
    }
    Ok(())
}

fn validate_unique_semantic_candidates(
    candidates: &[SemanticClassCandidate],
) -> Result<(), CompetingBeliefError> {
    for (index, candidate) in candidates.iter().enumerate() {
        if candidates[..index]
            .iter()
            .any(|other| other.label == candidate.label)
        {
            return Err(CompetingBeliefError::DuplicateSemanticCandidate);
        }
    }
    Ok(())
}

fn validate_unique_identity_candidates(
    candidates: &[IdentityCandidate],
) -> Result<(), CompetingBeliefError> {
    for (index, candidate) in candidates.iter().enumerate() {
        if candidates[..index]
            .iter()
            .any(|other| other.value == candidate.value)
        {
            return Err(CompetingBeliefError::DuplicateIdentityCandidate);
        }
    }
    Ok(())
}

fn validate_text(
    value: &str,
    max_len: usize,
    empty_error: CompetingBeliefError,
    too_long_error: CompetingBeliefError,
) -> Result<(), CompetingBeliefError> {
    if value.trim().is_empty() {
        return Err(empty_error);
    }
    if value.len() > max_len {
        return Err(too_long_error);
    }
    if value.chars().any(char::is_control) {
        return Err(CompetingBeliefError::TextContainsControlCharacter);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity_identity::{VisualEntityHypothesisRef, VisualTrackRef};
    use crate::epistemic::{VisualCaptureClock, VisualObservationRef, VisualStreamRef};

    fn observation(frame: u64) -> VisualObservationRef {
        VisualObservationRef::new(
            VisualStreamRef::new(3, 9).unwrap(),
            frame,
            100 + frame,
            VisualCaptureClock::StreamMonotonic,
        )
    }

    fn inferred(frame: u64, confidence: f32) -> VisualEvidence {
        VisualEvidence::inferred(vec![observation(frame)], confidence).unwrap()
    }

    #[test]
    fn weak_semantic_evidence_can_remain_mostly_unassigned() {
        let subject = VisualEntityHypothesisRef::new(7, 1).unwrap();
        let set = SemanticClassBeliefSet::new(
            subject,
            "open-vocabulary-v1",
            vec![
                SemanticClassCandidate::new("cup", BeliefMass::new(0.18).unwrap(), inferred(1, 0.6))
                    .unwrap(),
                SemanticClassCandidate::new(
                    "container",
                    BeliefMass::new(0.12).unwrap(),
                    inferred(1, 0.5),
                )
                .unwrap(),
            ],
            BeliefMass::new(0.70).unwrap(),
        )
        .unwrap();
        assert_eq!(set.unassigned_mass().get(), 0.70);
        assert_eq!(set.most_supported().unwrap().label(), "cup");
    }

    #[test]
    fn semantic_mass_is_not_auto_normalized() {
        let subject = VisualEntityHypothesisRef::new(7, 1).unwrap();
        let error = SemanticClassBeliefSet::new(
            subject,
            "vocab",
            vec![SemanticClassCandidate::new(
                "cup",
                BeliefMass::new(0.2).unwrap(),
                inferred(1, 0.5),
            )
            .unwrap()],
            BeliefMass::new(0.2).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error, CompetingBeliefError::MassPartitionDoesNotSumToOne);
    }

    #[test]
    fn identity_question_is_separate_from_semantic_question() {
        let track = VisualTrackRef::new(5, 0).unwrap();
        let existing = VisualEntityHypothesisRef::new(11, 4).unwrap();
        let set = TrackIdentityBeliefSet::new(
            track,
            vec![
                IdentityCandidate::new(
                    IdentityCandidateValue::ExistingEntity(existing),
                    BeliefMass::new(0.45).unwrap(),
                    inferred(2, 0.8),
                )
                .unwrap(),
                IdentityCandidate::new(
                    IdentityCandidateValue::NewEntity,
                    BeliefMass::new(0.20).unwrap(),
                    inferred(2, 0.7),
                )
                .unwrap(),
            ],
            BeliefMass::new(0.35).unwrap(),
        )
        .unwrap();
        assert_eq!(set.unassigned_mass().get(), 0.35);
    }

    #[test]
    fn duplicate_candidates_are_rejected() {
        let subject = VisualEntityHypothesisRef::new(7, 1).unwrap();
        let a = SemanticClassCandidate::new(
            "cup",
            BeliefMass::new(0.25).unwrap(),
            inferred(1, 0.5),
        )
        .unwrap();
        let b = SemanticClassCandidate::new(
            "cup",
            BeliefMass::new(0.25).unwrap(),
            inferred(2, 0.5),
        )
        .unwrap();
        assert_eq!(
            SemanticClassBeliefSet::new(
                subject,
                "vocab",
                vec![a, b],
                BeliefMass::new(0.5).unwrap(),
            )
            .unwrap_err(),
            CompetingBeliefError::DuplicateSemanticCandidate
        );
    }

    #[test]
    fn direct_observation_cannot_be_semantic_candidate_support() {
        let observed = VisualEvidence::observed(observation(1), 1.0).unwrap();
        assert_eq!(
            SemanticClassCandidate::new("cup", BeliefMass::new(1.0).unwrap(), observed)
                .unwrap_err(),
            CompetingBeliefError::ObservedCandidateEvidence
        );
    }

    #[test]
    fn generative_support_cannot_rewrite_historical_identity_belief() {
        let predicted = VisualEvidence::predicted(vec![observation(1)], 0.9).unwrap();
        assert_eq!(
            IdentityCandidate::new(
                IdentityCandidateValue::NewEntity,
                BeliefMass::new(1.0).unwrap(),
                predicted,
            )
            .unwrap_err(),
            CompetingBeliefError::GenerativeCandidateEvidence
        );
    }

    #[test]
    fn complete_abstention_is_valid() {
        let subject = VisualEntityHypothesisRef::new(7, 1).unwrap();
        let set = SemanticClassBeliefSet::new(
            subject,
            "vocab",
            Vec::new(),
            BeliefMass::new(1.0).unwrap(),
        )
        .unwrap();
        assert!(set.candidates().is_empty());
    }

    #[test]
    fn wire_mass_partition_is_revalidated() {
        let subject = VisualEntityHypothesisRef::new(7, 1).unwrap();
        let malformed = serde_json::json!({
            "subject": subject,
            "vocabulary": "vocab",
            "candidates": [],
            "unassigned_mass": 0.5,
        });
        assert!(serde_json::from_value::<SemanticClassBeliefSet>(malformed).is_err());
    }
}
