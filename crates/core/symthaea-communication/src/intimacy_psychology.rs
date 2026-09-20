// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dimensional, uncertainty-bearing intimacy psychology evidence.
//!
//! This module is descriptive only. It does not diagnose, infer consent, activate
//! fantasy mode, or grant physical authority.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const INTIMACY_PSYCHOLOGY_MODEL_SCHEMA_V1: &str =
    "symthaea.communication.intimacy-psychology-model.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntimacyPsychologyDimensionV1 {
    SexualExcitationPropensity,
    SexualInhibitionPropensity,
    DyadicDesire,
    SolitaryDesire,
    EmotionalClosenessNeed,
    ReassuranceNeed,
    IndependenceNeed,
    NoveltyPreference,
    FamiliarityPreference,
    CommunicationDirectness,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntimacyTemporalScopeV1 {
    TraitLike,
    CurrentState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntimacyRealityScopeV1 {
    RealWorld,
    FantasyOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntimacyPsychologySourceV1 {
    PopulationPrior,
    PhysiologicalInference,
    BehavioralInference,
    RepeatedExplicitFeedback,
    ValidatedSelfReportInstrument,
    ExplicitUserStatement,
}

impl IntimacyPsychologySourceV1 {
    pub const fn epistemic_rank(self) -> u8 {
        match self {
            Self::PopulationPrior => 1,
            Self::PhysiologicalInference => 2,
            Self::BehavioralInference => 3,
            Self::RepeatedExplicitFeedback => 4,
            Self::ValidatedSelfReportInstrument => 5,
            Self::ExplicitUserStatement => 6,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntimacyPsychologyRetentionV1 {
    #[default]
    EphemeralSession,
    DurableOptIn,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntimacyPsychologyEvidenceV1 {
    evidence_id: String,
    dimension: IntimacyPsychologyDimensionV1,
    value: f32,
    confidence: f32,
    source: IntimacyPsychologySourceV1,
    temporal_scope: IntimacyTemporalScopeV1,
    reality_scope: IntimacyRealityScopeV1,
    context_id: String,
    observed_at_ns: u64,
    valid_until_ns: Option<u64>,
    source_ref: String,
    retention: IntimacyPsychologyRetentionV1,
}

impl IntimacyPsychologyEvidenceV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        evidence_id: impl Into<String>,
        dimension: IntimacyPsychologyDimensionV1,
        value: f32,
        confidence: f32,
        source: IntimacyPsychologySourceV1,
        temporal_scope: IntimacyTemporalScopeV1,
        reality_scope: IntimacyRealityScopeV1,
        context_id: impl Into<String>,
        observed_at_ns: u64,
        valid_until_ns: Option<u64>,
        source_ref: impl Into<String>,
        retention: IntimacyPsychologyRetentionV1,
    ) -> Result<Self, IntimacyPsychologyErrorV1> {
        let evidence_id = bounded_id(evidence_id.into(), 256)
            .ok_or(IntimacyPsychologyErrorV1::InvalidEvidenceId)?;
        let context_id = bounded_id(context_id.into(), 192)
            .ok_or(IntimacyPsychologyErrorV1::InvalidContextId)?;
        let source_ref = bounded_id(source_ref.into(), 512)
            .ok_or(IntimacyPsychologyErrorV1::InvalidSourceRef)?;
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(IntimacyPsychologyErrorV1::InvalidValue);
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(IntimacyPsychologyErrorV1::InvalidConfidence);
        }
        if let Some(valid_until_ns) = valid_until_ns {
            if valid_until_ns < observed_at_ns {
                return Err(IntimacyPsychologyErrorV1::InvalidValidityWindow);
            }
        }
        if source == IntimacyPsychologySourceV1::PhysiologicalInference
            && temporal_scope != IntimacyTemporalScopeV1::CurrentState
        {
            return Err(IntimacyPsychologyErrorV1::PhysiologyCannotEstablishTrait);
        }
        Ok(Self {
            evidence_id,
            dimension,
            value,
            confidence,
            source,
            temporal_scope,
            reality_scope,
            context_id,
            observed_at_ns,
            valid_until_ns,
            source_ref,
            retention,
        })
    }

    pub fn is_current_at(&self, now_ns: u64) -> bool {
        now_ns >= self.observed_at_ns
            && self
                .valid_until_ns
                .map(|valid_until_ns| now_ns <= valid_until_ns)
                .unwrap_or(true)
    }

    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    pub const fn source(&self) -> IntimacyPsychologySourceV1 {
        self.source
    }

    pub const fn temporal_scope(&self) -> IntimacyTemporalScopeV1 {
        self.temporal_scope
    }

    pub const fn reality_scope(&self) -> IntimacyRealityScopeV1 {
        self.reality_scope
    }

    pub const fn observed_at_ns(&self) -> u64 {
        self.observed_at_ns
    }

    pub const fn valid_until_ns(&self) -> Option<u64> {
        self.valid_until_ns
    }

    pub const fn retention(&self) -> IntimacyPsychologyRetentionV1 {
        self.retention
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IntimacyPsychologyEstimateV1 {
    pub dimension: IntimacyPsychologyDimensionV1,
    pub value: f32,
    pub confidence: f32,
    pub source: IntimacyPsychologySourceV1,
    pub evidence_id: String,
    pub temporal_scope: IntimacyTemporalScopeV1,
    pub reality_scope: IntimacyRealityScopeV1,
    pub context_id: String,
    pub observed_at_ns: u64,
    pub valid_until_ns: Option<u64>,
    pub retention: IntimacyPsychologyRetentionV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntimacyPsychologyErrorV1 {
    InvalidEvidenceId,
    InvalidContextId,
    InvalidSourceRef,
    InvalidValue,
    InvalidConfidence,
    InvalidValidityWindow,
    PhysiologyCannotEstablishTrait,
    DuplicateEvidenceId,
}

#[derive(Debug, Default)]
pub struct IntimacyPsychologyModelV1 {
    evidence: BTreeMap<IntimacyPsychologyDimensionV1, Vec<IntimacyPsychologyEvidenceV1>>,
    known_evidence_ids: BTreeSet<String>,
}

impl IntimacyPsychologyModelV1 {
    pub fn record(
        &mut self,
        evidence: IntimacyPsychologyEvidenceV1,
    ) -> Result<(), IntimacyPsychologyErrorV1> {
        if !self.known_evidence_ids.insert(evidence.evidence_id.clone()) {
            return Err(IntimacyPsychologyErrorV1::DuplicateEvidenceId);
        }
        self.evidence.entry(evidence.dimension).or_default().push(evidence);
        Ok(())
    }

    /// Remove an evidence payload from active estimates while retaining its ID as a
    /// tombstone. This prevents a retracted sensitive item from being silently
    /// replayed under the same identity.
    ///
    /// This is model-local retraction only. It does not claim deletion of copies in
    /// external logs, summaries, exports, or other memory systems.
    pub fn retract_evidence(
        &mut self,
        evidence_id: &str,
    ) -> Result<bool, IntimacyPsychologyErrorV1> {
        let evidence_id = bounded_id(evidence_id.to_owned(), 256)
            .ok_or(IntimacyPsychologyErrorV1::InvalidEvidenceId)?;
        if !self.known_evidence_ids.contains(&evidence_id) {
            return Ok(false);
        }

        let mut removed = false;
        for evidence in self.evidence.values_mut() {
            let before = evidence.len();
            evidence.retain(|item| item.evidence_id != evidence_id);
            removed |= evidence.len() != before;
        }
        self.evidence.retain(|_, evidence| !evidence.is_empty());
        Ok(removed)
    }

    pub fn estimate_at(
        &self,
        dimension: IntimacyPsychologyDimensionV1,
        temporal_scope: IntimacyTemporalScopeV1,
        reality_scope: IntimacyRealityScopeV1,
        context_id: &str,
        now_ns: u64,
    ) -> Option<IntimacyPsychologyEstimateV1> {
        let context_id = bounded_id(context_id.to_owned(), 192)?;
        let best = self
            .evidence
            .get(&dimension)?
            .iter()
            .filter(|item| {
                item.temporal_scope == temporal_scope
                    && item.reality_scope == reality_scope
                    && item.context_id == context_id
                    && item.is_current_at(now_ns)
            })
            .max_by(|a, b| {
                a.source
                    .epistemic_rank()
                    .cmp(&b.source.epistemic_rank())
                    .then_with(|| {
                        if a.source == IntimacyPsychologySourceV1::ExplicitUserStatement
                            && b.source == IntimacyPsychologySourceV1::ExplicitUserStatement
                        {
                            a.observed_at_ns
                                .cmp(&b.observed_at_ns)
                                .then_with(|| a.confidence.total_cmp(&b.confidence))
                        } else {
                            a.confidence
                                .total_cmp(&b.confidence)
                                .then_with(|| a.observed_at_ns.cmp(&b.observed_at_ns))
                        }
                    })
            })?;
        Some(IntimacyPsychologyEstimateV1 {
            dimension,
            value: best.value,
            confidence: best.confidence,
            source: best.source,
            evidence_id: best.evidence_id.clone(),
            temporal_scope,
            reality_scope,
            context_id,
            observed_at_ns: best.observed_at_ns,
            valid_until_ns: best.valid_until_ns,
            retention: best.retention,
        })
    }
}

fn bounded_id(value: String, max_len: usize) -> Option<String> {
    let value = value.trim().to_owned();
    (!value.is_empty() && value.len() <= max_len).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn evidence_at(
        id: &str,
        source: IntimacyPsychologySourceV1,
        temporal: IntimacyTemporalScopeV1,
        reality: IntimacyRealityScopeV1,
        context: &str,
        value: f32,
        confidence: f32,
        observed_at_ns: u64,
        valid_until_ns: Option<u64>,
    ) -> IntimacyPsychologyEvidenceV1 {
        IntimacyPsychologyEvidenceV1::new(
            id,
            IntimacyPsychologyDimensionV1::NoveltyPreference,
            value,
            confidence,
            source,
            temporal,
            reality,
            context,
            observed_at_ns,
            valid_until_ns,
            format!("source:{id}"),
            IntimacyPsychologyRetentionV1::EphemeralSession,
        )
        .unwrap()
    }

    fn evidence(
        id: &str,
        source: IntimacyPsychologySourceV1,
        temporal: IntimacyTemporalScopeV1,
        reality: IntimacyRealityScopeV1,
        context: &str,
        value: f32,
        confidence: f32,
        valid_until_ns: Option<u64>,
    ) -> IntimacyPsychologyEvidenceV1 {
        evidence_at(
            id,
            source,
            temporal,
            reality,
            context,
            value,
            confidence,
            100,
            valid_until_ns,
        )
    }

    #[test]
    fn unknown_remains_unknown_without_evidence() {
        let model = IntimacyPsychologyModelV1::default();
        assert!(model
            .estimate_at(
                IntimacyPsychologyDimensionV1::NoveltyPreference,
                IntimacyTemporalScopeV1::TraitLike,
                IntimacyRealityScopeV1::RealWorld,
                "relationship:a",
                100,
            )
            .is_none());
    }

    #[test]
    fn explicit_statement_outranks_population_prior() {
        let mut model = IntimacyPsychologyModelV1::default();
        model
            .record(evidence(
                "prior",
                IntimacyPsychologySourceV1::PopulationPrior,
                IntimacyTemporalScopeV1::TraitLike,
                IntimacyRealityScopeV1::RealWorld,
                "relationship:a",
                0.9,
                0.99,
                None,
            ))
            .unwrap();
        model
            .record(evidence(
                "explicit",
                IntimacyPsychologySourceV1::ExplicitUserStatement,
                IntimacyTemporalScopeV1::TraitLike,
                IntimacyRealityScopeV1::RealWorld,
                "relationship:a",
                0.2,
                0.7,
                None,
            ))
            .unwrap();
        let estimate = model
            .estimate_at(
                IntimacyPsychologyDimensionV1::NoveltyPreference,
                IntimacyTemporalScopeV1::TraitLike,
                IntimacyRealityScopeV1::RealWorld,
                "relationship:a",
                100,
            )
            .unwrap();
        assert_eq!(estimate.evidence_id, "explicit");
        assert_eq!(estimate.value, 0.2);
    }

    #[test]
    fn newer_explicit_correction_beats_older_higher_confidence_statement() {
        let mut model = IntimacyPsychologyModelV1::default();
        model
            .record(evidence_at(
                "old-explicit",
                IntimacyPsychologySourceV1::ExplicitUserStatement,
                IntimacyTemporalScopeV1::CurrentState,
                IntimacyRealityScopeV1::RealWorld,
                "session:a",
                0.9,
                1.0,
                100,
                Some(300),
            ))
            .unwrap();
        model
            .record(evidence_at(
                "new-correction",
                IntimacyPsychologySourceV1::ExplicitUserStatement,
                IntimacyTemporalScopeV1::CurrentState,
                IntimacyRealityScopeV1::RealWorld,
                "session:a",
                0.1,
                0.6,
                200,
                Some(300),
            ))
            .unwrap();
        let estimate = model
            .estimate_at(
                IntimacyPsychologyDimensionV1::NoveltyPreference,
                IntimacyTemporalScopeV1::CurrentState,
                IntimacyRealityScopeV1::RealWorld,
                "session:a",
                250,
            )
            .unwrap();
        assert_eq!(estimate.evidence_id, "new-correction");
        assert_eq!(estimate.value, 0.1);
    }

    #[test]
    fn fantasy_and_real_world_contexts_do_not_leak() {
        let mut model = IntimacyPsychologyModelV1::default();
        model
            .record(evidence(
                "fantasy",
                IntimacyPsychologySourceV1::ExplicitUserStatement,
                IntimacyTemporalScopeV1::TraitLike,
                IntimacyRealityScopeV1::FantasyOnly,
                "story:a",
                1.0,
                1.0,
                None,
            ))
            .unwrap();
        assert!(model
            .estimate_at(
                IntimacyPsychologyDimensionV1::NoveltyPreference,
                IntimacyTemporalScopeV1::TraitLike,
                IntimacyRealityScopeV1::RealWorld,
                "relationship:a",
                100,
            )
            .is_none());
    }

    #[test]
    fn stale_current_state_evidence_is_ignored() {
        let mut model = IntimacyPsychologyModelV1::default();
        model
            .record(evidence(
                "state",
                IntimacyPsychologySourceV1::ExplicitUserStatement,
                IntimacyTemporalScopeV1::CurrentState,
                IntimacyRealityScopeV1::RealWorld,
                "session:a",
                0.8,
                1.0,
                Some(150),
            ))
            .unwrap();
        assert!(model
            .estimate_at(
                IntimacyPsychologyDimensionV1::NoveltyPreference,
                IntimacyTemporalScopeV1::CurrentState,
                IntimacyRealityScopeV1::RealWorld,
                "session:a",
                151,
            )
            .is_none());
    }

    #[test]
    fn physiology_cannot_establish_trait() {
        let result = IntimacyPsychologyEvidenceV1::new(
            "phys",
            IntimacyPsychologyDimensionV1::SexualExcitationPropensity,
            0.8,
            0.7,
            IntimacyPsychologySourceV1::PhysiologicalInference,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "session:a",
            100,
            Some(120),
            "sensor:evidence",
            IntimacyPsychologyRetentionV1::EphemeralSession,
        );
        assert_eq!(
            result,
            Err(IntimacyPsychologyErrorV1::PhysiologyCannotEstablishTrait)
        );
    }

    #[test]
    fn trait_evidence_does_not_answer_current_state() {
        let mut model = IntimacyPsychologyModelV1::default();
        model
            .record(evidence(
                "trait",
                IntimacyPsychologySourceV1::ExplicitUserStatement,
                IntimacyTemporalScopeV1::TraitLike,
                IntimacyRealityScopeV1::RealWorld,
                "session:a",
                0.8,
                1.0,
                None,
            ))
            .unwrap();
        assert!(model
            .estimate_at(
                IntimacyPsychologyDimensionV1::NoveltyPreference,
                IntimacyTemporalScopeV1::CurrentState,
                IntimacyRealityScopeV1::RealWorld,
                "session:a",
                100,
            )
            .is_none());
    }

    #[test]
    fn retraction_removes_payload_but_tombstones_identity() {
        let mut model = IntimacyPsychologyModelV1::default();
        let item = evidence(
            "sensitive",
            IntimacyPsychologySourceV1::ExplicitUserStatement,
            IntimacyTemporalScopeV1::TraitLike,
            IntimacyRealityScopeV1::RealWorld,
            "relationship:a",
            0.7,
            1.0,
            None,
        );
        model.record(item.clone()).unwrap();
        assert!(model.retract_evidence("sensitive").unwrap());
        assert!(model
            .estimate_at(
                IntimacyPsychologyDimensionV1::NoveltyPreference,
                IntimacyTemporalScopeV1::TraitLike,
                IntimacyRealityScopeV1::RealWorld,
                "relationship:a",
                100,
            )
            .is_none());
        assert_eq!(
            model.record(item),
            Err(IntimacyPsychologyErrorV1::DuplicateEvidenceId)
        );
    }

    #[test]
    fn estimate_preserves_freshness_and_retention_provenance() {
        let mut model = IntimacyPsychologyModelV1::default();
        model
            .record(IntimacyPsychologyEvidenceV1::new(
                "provenance",
                IntimacyPsychologyDimensionV1::DyadicDesire,
                0.6,
                0.8,
                IntimacyPsychologySourceV1::ValidatedSelfReportInstrument,
                IntimacyTemporalScopeV1::CurrentState,
                IntimacyRealityScopeV1::RealWorld,
                "session:a",
                110,
                Some(160),
                "instrument:versioned-ref",
                IntimacyPsychologyRetentionV1::DurableOptIn,
            ).unwrap())
            .unwrap();
        let estimate = model
            .estimate_at(
                IntimacyPsychologyDimensionV1::DyadicDesire,
                IntimacyTemporalScopeV1::CurrentState,
                IntimacyRealityScopeV1::RealWorld,
                "session:a",
                120,
            )
            .unwrap();
        assert_eq!(estimate.source, IntimacyPsychologySourceV1::ValidatedSelfReportInstrument);
        assert_eq!(estimate.observed_at_ns, 110);
        assert_eq!(estimate.valid_until_ns, Some(160));
        assert_eq!(estimate.retention, IntimacyPsychologyRetentionV1::DurableOptIn);
    }
}
