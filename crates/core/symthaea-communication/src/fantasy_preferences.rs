// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Private, non-authoritative preference and boundary semantics for adult fantasy dialogue.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FANTASY_PREFERENCE_MODEL_SCHEMA_V1: &str =
    "symthaea.communication.fantasy-preference-model.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FantasyStyleDimensionV1 {
    Romance,
    Playfulness,
    Directness,
    VerbalIntensity,
    Tenderness,
    Initiative,
    Suspense,
    Humor,
    NarrativeDensity,
    CallbackDensity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FantasyPreferenceSourceV1 {
    PopulationPrior,
    BehavioralInference,
    RepeatedExplicitFeedback,
    ExplicitUserPreference,
}

impl FantasyPreferenceSourceV1 {
    pub const fn epistemic_rank(self) -> u8 {
        match self {
            Self::PopulationPrior => 1,
            Self::BehavioralInference => 2,
            Self::RepeatedExplicitFeedback => 3,
            Self::ExplicitUserPreference => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FantasyPreferenceRetentionV1 {
    #[default]
    EphemeralSession,
    DurablePreferenceOptIn,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FantasyPreferenceEvidenceV1 {
    evidence_id: String,
    dimension: FantasyStyleDimensionV1,
    value: f32,
    confidence: f32,
    source: FantasyPreferenceSourceV1,
    observed_at_ns: u64,
    source_ref: String,
    retention: FantasyPreferenceRetentionV1,
}

impl FantasyPreferenceEvidenceV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        evidence_id: impl Into<String>,
        dimension: FantasyStyleDimensionV1,
        value: f32,
        confidence: f32,
        source: FantasyPreferenceSourceV1,
        observed_at_ns: u64,
        source_ref: impl Into<String>,
        retention: FantasyPreferenceRetentionV1,
    ) -> Result<Self, FantasyPreferenceErrorV1> {
        let evidence_id = evidence_id.into().trim().to_owned();
        let source_ref = source_ref.into().trim().to_owned();
        if evidence_id.is_empty() || evidence_id.len() > 256 {
            return Err(FantasyPreferenceErrorV1::InvalidEvidenceId);
        }
        if source_ref.is_empty() || source_ref.len() > 512 {
            return Err(FantasyPreferenceErrorV1::InvalidSourceRef);
        }
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(FantasyPreferenceErrorV1::InvalidPreferenceValue);
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(FantasyPreferenceErrorV1::InvalidConfidence);
        }
        Ok(Self {
            evidence_id,
            dimension,
            value,
            confidence,
            source,
            observed_at_ns,
            source_ref,
            retention,
        })
    }

    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    pub const fn dimension(&self) -> FantasyStyleDimensionV1 {
        self.dimension
    }

    pub const fn value(&self) -> f32 {
        self.value
    }

    pub const fn confidence(&self) -> f32 {
        self.confidence
    }

    pub const fn source(&self) -> FantasyPreferenceSourceV1 {
        self.source
    }

    pub const fn observed_at_ns(&self) -> u64 {
        self.observed_at_ns
    }

    pub const fn retention(&self) -> FantasyPreferenceRetentionV1 {
        self.retention
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FantasyPreferenceEstimateV1 {
    pub dimension: FantasyStyleDimensionV1,
    pub value: f32,
    pub confidence: f32,
    pub source: FantasyPreferenceSourceV1,
    pub evidence_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FantasyTopicBoundaryV1 {
    blocked_topic_ids: BTreeSet<String>,
    boundary_epoch: u64,
}

impl FantasyTopicBoundaryV1 {
    pub fn new(boundary_epoch: u64) -> Result<Self, FantasyPreferenceErrorV1> {
        if boundary_epoch == 0 {
            return Err(FantasyPreferenceErrorV1::InvalidBoundaryEpoch);
        }
        Ok(Self {
            blocked_topic_ids: BTreeSet::new(),
            boundary_epoch,
        })
    }

    pub const fn boundary_epoch(&self) -> u64 {
        self.boundary_epoch
    }

    pub fn block_topic(
        &mut self,
        topic_id: impl Into<String>,
    ) -> Result<bool, FantasyPreferenceErrorV1> {
        let topic_id = canonical_topic_id(topic_id.into())?;
        Ok(self.blocked_topic_ids.insert(topic_id))
    }

    pub fn is_topic_allowed(&self, topic_id: &str) -> bool {
        canonical_topic_id(topic_id.to_owned())
            .map(|canonical| !self.blocked_topic_ids.contains(&canonical))
            .unwrap_or(false)
    }

    pub fn blocked_topics(&self) -> &BTreeSet<String> {
        &self.blocked_topic_ids
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FantasyPreferenceErrorV1 {
    InvalidEvidenceId,
    InvalidSourceRef,
    InvalidPreferenceValue,
    InvalidConfidence,
    InvalidBoundaryEpoch,
    InvalidTopicId,
    StaleBoundaryEpoch,
    DuplicateEvidenceId,
}

#[derive(Debug)]
pub struct FantasyPreferenceModelV1 {
    evidence_by_dimension: BTreeMap<FantasyStyleDimensionV1, Vec<FantasyPreferenceEvidenceV1>>,
    known_evidence_ids: BTreeSet<String>,
    boundaries: FantasyTopicBoundaryV1,
}

impl FantasyPreferenceModelV1 {
    pub fn new(boundary_epoch: u64) -> Result<Self, FantasyPreferenceErrorV1> {
        Ok(Self {
            evidence_by_dimension: BTreeMap::new(),
            known_evidence_ids: BTreeSet::new(),
            boundaries: FantasyTopicBoundaryV1::new(boundary_epoch)?,
        })
    }

    pub fn record(
        &mut self,
        evidence: FantasyPreferenceEvidenceV1,
    ) -> Result<(), FantasyPreferenceErrorV1> {
        if !self.known_evidence_ids.insert(evidence.evidence_id.clone()) {
            return Err(FantasyPreferenceErrorV1::DuplicateEvidenceId);
        }
        self.evidence_by_dimension
            .entry(evidence.dimension)
            .or_default()
            .push(evidence);
        Ok(())
    }

    pub fn effective_estimate(
        &self,
        dimension: FantasyStyleDimensionV1,
    ) -> Option<FantasyPreferenceEstimateV1> {
        let evidence = self.evidence_by_dimension.get(&dimension)?;
        let best = evidence.iter().max_by(|a, b| {
            a.source
                .epistemic_rank()
                .cmp(&b.source.epistemic_rank())
                .then_with(|| a.confidence.total_cmp(&b.confidence))
                .then_with(|| a.observed_at_ns.cmp(&b.observed_at_ns))
        })?;
        Some(FantasyPreferenceEstimateV1 {
            dimension,
            value: best.value,
            confidence: best.confidence,
            source: best.source,
            evidence_id: best.evidence_id.clone(),
        })
    }

    pub fn boundaries(&self) -> &FantasyTopicBoundaryV1 {
        &self.boundaries
    }

    pub fn replace_boundaries(
        &mut self,
        new_boundaries: FantasyTopicBoundaryV1,
    ) -> Result<(), FantasyPreferenceErrorV1> {
        if new_boundaries.boundary_epoch <= self.boundaries.boundary_epoch {
            return Err(FantasyPreferenceErrorV1::StaleBoundaryEpoch);
        }
        self.boundaries = new_boundaries;
        Ok(())
    }

    /// Preferences can suggest style only if the requested topic is not hard-blocked.
    pub fn may_apply_preference_to_topic(&self, topic_id: &str) -> bool {
        self.boundaries.is_topic_allowed(topic_id)
    }
}

fn canonical_topic_id(topic_id: String) -> Result<String, FantasyPreferenceErrorV1> {
    let topic_id = topic_id.trim().to_ascii_lowercase();
    if topic_id.is_empty() || topic_id.len() > 128 {
        return Err(FantasyPreferenceErrorV1::InvalidTopicId);
    }
    if !topic_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(FantasyPreferenceErrorV1::InvalidTopicId);
    }
    Ok(topic_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(
        id: &str,
        source: FantasyPreferenceSourceV1,
        value: f32,
        confidence: f32,
        observed_at_ns: u64,
    ) -> FantasyPreferenceEvidenceV1 {
        FantasyPreferenceEvidenceV1::new(
            id,
            FantasyStyleDimensionV1::Playfulness,
            value,
            confidence,
            source,
            observed_at_ns,
            format!("source:{id}"),
            FantasyPreferenceRetentionV1::EphemeralSession,
        )
        .unwrap()
    }

    #[test]
    fn explicit_preference_outranks_high_confidence_behavioral_inference() {
        let mut model = FantasyPreferenceModelV1::new(1).unwrap();
        model
            .record(evidence(
                "inferred",
                FantasyPreferenceSourceV1::BehavioralInference,
                0.9,
                0.99,
                200,
            ))
            .unwrap();
        model
            .record(evidence(
                "explicit",
                FantasyPreferenceSourceV1::ExplicitUserPreference,
                0.4,
                0.7,
                100,
            ))
            .unwrap();
        let effective = model
            .effective_estimate(FantasyStyleDimensionV1::Playfulness)
            .unwrap();
        assert_eq!(effective.evidence_id, "explicit");
        assert_eq!(effective.value, 0.4);
    }

    #[test]
    fn hard_topic_boundary_defeats_any_preference() {
        let mut model = FantasyPreferenceModelV1::new(1).unwrap();
        model.boundaries.block_topic("topic.alpha").unwrap();
        model
            .record(evidence(
                "explicit",
                FantasyPreferenceSourceV1::ExplicitUserPreference,
                1.0,
                1.0,
                100,
            ))
            .unwrap();
        assert!(!model.may_apply_preference_to_topic("topic.alpha"));
    }

    #[test]
    fn boundary_updates_require_fresh_epoch() {
        let mut model = FantasyPreferenceModelV1::new(3).unwrap();
        let stale = FantasyTopicBoundaryV1::new(3).unwrap();
        assert_eq!(
            model.replace_boundaries(stale),
            Err(FantasyPreferenceErrorV1::StaleBoundaryEpoch)
        );
        let fresh = FantasyTopicBoundaryV1::new(4).unwrap();
        model.replace_boundaries(fresh).unwrap();
        assert_eq!(model.boundaries().boundary_epoch(), 4);
    }

    #[test]
    fn topic_ids_are_canonical_and_fail_closed_when_malformed() {
        let mut boundaries = FantasyTopicBoundaryV1::new(1).unwrap();
        boundaries.block_topic("Topic.Alpha").unwrap();
        assert!(!boundaries.is_topic_allowed("topic.alpha"));
        assert!(!boundaries.is_topic_allowed("not a valid topic"));
    }

    #[test]
    fn duplicate_evidence_identity_is_rejected() {
        let mut model = FantasyPreferenceModelV1::new(1).unwrap();
        model
            .record(evidence(
                "same",
                FantasyPreferenceSourceV1::ExplicitUserPreference,
                0.5,
                0.8,
                100,
            ))
            .unwrap();
        assert_eq!(
            model.record(evidence(
                "same",
                FantasyPreferenceSourceV1::BehavioralInference,
                0.7,
                0.9,
                200,
            )),
            Err(FantasyPreferenceErrorV1::DuplicateEvidenceId)
        );
    }
}
