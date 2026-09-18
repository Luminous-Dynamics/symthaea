// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemically typed narrative ingestion.
//!
//! Autobiographical memory is deliberately narrower than empirical evidence:
//! only a newly [`WorldEvidenceKind::Recorded`] experience may enter the legacy
//! `NarrativeSelfModel` as a lived episode. Exact replay is grounded evidence, but
//! replaying an old transition is recollection rather than another life event.
//! Generated predictions/counterfactuals are imagination.
//!
//! Recollection and imagination may be retained in the runtime-only
//! [`NarrativeDerivedMemory`] sidecar without changing autobiographical episode
//! count, self-concept, or narrative coherence. Positive appraisal is narrative
//! valence only; it is never empirical validation or confidence authority.

use std::collections::VecDeque;

use super::epistemic_world::WorldEvidenceKind;
use crate::consciousness::narrative_self::NarrativeSelfModel;
use crate::hdc::binary_hv::BinaryHV;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NarrativeMemoryClass {
    /// A newly observed lived event. May enter autobiographical life story.
    Lived,
    /// Grounded replay/recollection of an already recorded event.
    Recollection,
    /// Prediction, counterfactual, interpolation, extrapolation, or adversarial generation.
    Imagination,
}

impl NarrativeMemoryClass {
    pub fn from_evidence_kind(kind: WorldEvidenceKind) -> Self {
        match kind {
            WorldEvidenceKind::Recorded => Self::Lived,
            WorldEvidenceKind::ReplayDerived => Self::Recollection,
            WorldEvidenceKind::Interpolated
            | WorldEvidenceKind::ModelPredicted
            | WorldEvidenceKind::Counterfactual
            | WorldEvidenceKind::Extrapolated
            | WorldEvidenceKind::AdversarialGenerated => Self::Imagination,
        }
    }
}

/// Typed narrative ingestion request.
///
/// Keeping epistemic kind and source identity adjacent to the representation makes
/// it harder for callers to accidentally submit generated content through a
/// lived-experience-only call path.
#[derive(Debug, Clone, Copy)]
pub struct NarrativeEvidenceInput<'a> {
    pub representation: &'a BinaryHV,
    pub description: &'a str,
    pub evidence_kind: WorldEvidenceKind,
    pub source_digest: &'a str,
    pub appraisal_positive: bool,
    pub effort: f64,
    pub significance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NarrativeEvidenceReceipt {
    pub evidence_kind: WorldEvidenceKind,
    pub source_digest: String,
    pub description: String,
    pub appraisal_positive: bool,
    pub significance: f64,
    /// Lived autobiographical episode count before ingestion.
    pub episodes_before: usize,
    /// Lived autobiographical episode count after ingestion.
    pub episodes_after: usize,
}

impl NarrativeEvidenceReceipt {
    /// True only for evidence classes that are empirical outcome evidence.
    pub fn is_direct_empirical(&self) -> bool {
        self.evidence_kind.is_empirical()
    }

    /// Narrative storage class is stricter than empirical status.
    pub fn memory_class(&self) -> NarrativeMemoryClass {
        NarrativeMemoryClass::from_evidence_kind(self.evidence_kind)
    }

    /// Whether ingestion created a new lived autobiographical episode.
    ///
    /// Recollection/imagination deliberately return false even if retained in the
    /// derived-memory sidecar.
    pub fn recorded_episode(&self) -> bool {
        self.memory_class() == NarrativeMemoryClass::Lived
            && self.episodes_after > self.episodes_before
    }
}

/// Runtime-only recollection/imagination record.
///
/// This type intentionally does not derive serde traits. Retaining a generated
/// narrative item across process restart must be an explicit persistence design,
/// not an accidental promotion into autobiographical history.
#[derive(Debug, Clone, PartialEq)]
pub struct NarrativeDerivedRecord {
    pub memory_class: NarrativeMemoryClass,
    pub evidence_kind: WorldEvidenceKind,
    pub source_digest: String,
    pub description: String,
    pub representation: BinaryHV,
    pub appraisal_positive: bool,
    pub effort: f64,
    pub significance: f64,
}

#[derive(Debug, Clone)]
pub struct NarrativeDerivedMemory {
    entries: VecDeque<NarrativeDerivedRecord>,
    max_entries: usize,
}

impl Default for NarrativeDerivedMemory {
    fn default() -> Self {
        Self::new(256)
    }
}

impl NarrativeDerivedMemory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries.min(256)),
            max_entries,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &NarrativeDerivedRecord> {
        self.entries.iter()
    }

    pub fn contains_source_digest(&self, source_digest: &str) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.source_digest == source_digest)
    }

    fn push(&mut self, record: NarrativeDerivedRecord) {
        if self.max_entries == 0 || self.contains_source_digest(&record.source_digest) {
            return;
        }
        while self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(record);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrativeEvidenceError {
    EmptySourceDigest,
    NonFiniteEffort,
    NonFiniteSignificance,
}

/// Ingest narrative material with the conservative default boundary.
///
/// Only direct `Recorded` observations may modify autobiographical narrative.
/// Replay-derived and generated material is not persisted by this overload; use
/// [`ingest_narrative_evidence_with_derived_memory`] when it should remain available
/// as recollection/imagination without becoming lived history.
pub fn ingest_narrative_evidence(
    model: &mut NarrativeSelfModel,
    input: NarrativeEvidenceInput<'_>,
) -> Result<NarrativeEvidenceReceipt, NarrativeEvidenceError> {
    ingest_narrative_evidence_impl(model, None, input)
}

/// Ingest narrative material while retaining non-lived material in a separate,
/// runtime-only sidecar.
///
/// `Recorded` observations follow the legacy autobiographical path. Replay-derived
/// material is stored as recollection. Generated material is stored as imagination.
/// Neither recollection nor imagination changes the life-story episode count.
pub fn ingest_narrative_evidence_with_derived_memory(
    model: &mut NarrativeSelfModel,
    derived_memory: &mut NarrativeDerivedMemory,
    input: NarrativeEvidenceInput<'_>,
) -> Result<NarrativeEvidenceReceipt, NarrativeEvidenceError> {
    ingest_narrative_evidence_impl(model, Some(derived_memory), input)
}

fn ingest_narrative_evidence_impl(
    model: &mut NarrativeSelfModel,
    derived_memory: Option<&mut NarrativeDerivedMemory>,
    input: NarrativeEvidenceInput<'_>,
) -> Result<NarrativeEvidenceReceipt, NarrativeEvidenceError> {
    if input.source_digest.trim().is_empty() {
        return Err(NarrativeEvidenceError::EmptySourceDigest);
    }
    if !input.effort.is_finite() {
        return Err(NarrativeEvidenceError::NonFiniteEffort);
    }
    if !input.significance.is_finite() {
        return Err(NarrativeEvidenceError::NonFiniteSignificance);
    }

    let effort = input.effort.clamp(0.0, 1.0);
    let significance = input.significance.clamp(0.0, 1.0);
    let episodes_before = model.autobio.life_story.len();
    let memory_class = NarrativeMemoryClass::from_evidence_kind(input.evidence_kind);

    match memory_class {
        NarrativeMemoryClass::Lived => model.process_experience(
            input.representation,
            input.description,
            input.appraisal_positive,
            effort,
            significance,
        ),
        NarrativeMemoryClass::Recollection | NarrativeMemoryClass::Imagination => {
            if let Some(memory) = derived_memory {
                memory.push(NarrativeDerivedRecord {
                    memory_class,
                    evidence_kind: input.evidence_kind,
                    source_digest: input.source_digest.to_string(),
                    description: input.description.to_string(),
                    representation: *input.representation,
                    appraisal_positive: input.appraisal_positive,
                    effort,
                    significance,
                });
            }
        }
    }

    Ok(NarrativeEvidenceReceipt {
        evidence_kind: input.evidence_kind,
        source_digest: input.source_digest.to_string(),
        description: input.description.to_string(),
        appraisal_positive: input.appraisal_positive,
        significance,
        episodes_before,
        episodes_after: model.autobio.life_story.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::narrative_self::NarrativeSelfConfig;

    fn model() -> NarrativeSelfModel {
        NarrativeSelfModel::new(NarrativeSelfConfig::default())
    }

    fn input<'a>(
        representation: &'a BinaryHV,
        description: &'a str,
        evidence_kind: WorldEvidenceKind,
        source_digest: &'a str,
    ) -> NarrativeEvidenceInput<'a> {
        NarrativeEvidenceInput {
            representation,
            description,
            evidence_kind,
            source_digest,
            appraisal_positive: true,
            effort: 0.2,
            significance: 0.9,
        }
    }

    #[test]
    fn counterfactual_default_ingestion_cannot_become_lived_history() {
        let mut model = model();
        let representation = BinaryHV::random(77);
        let receipt = ingest_narrative_evidence(
            &mut model,
            input(
                &representation,
                "dream found a promising alternative",
                WorldEvidenceKind::Counterfactual,
                "blake3:dream-content",
            ),
        )
        .unwrap();

        assert_eq!(receipt.memory_class(), NarrativeMemoryClass::Imagination);
        assert!(!receipt.recorded_episode());
        assert!(!receipt.is_direct_empirical());
        assert_eq!(model.autobio.life_story.len(), 0);
    }

    #[test]
    fn counterfactual_can_be_retained_in_imagination_sidecar_only() {
        let mut model = model();
        let mut derived = NarrativeDerivedMemory::default();
        let representation = BinaryHV::random(78);
        let receipt = ingest_narrative_evidence_with_derived_memory(
            &mut model,
            &mut derived,
            input(
                &representation,
                "dream alternative",
                WorldEvidenceKind::Counterfactual,
                "blake3:dream-sidecar",
            ),
        )
        .unwrap();

        assert!(!receipt.recorded_episode());
        assert_eq!(model.autobio.life_story.len(), 0);
        assert_eq!(derived.len(), 1);
        let stored = derived.iter().next().unwrap();
        assert_eq!(stored.memory_class, NarrativeMemoryClass::Imagination);
        assert_eq!(stored.evidence_kind, WorldEvidenceKind::Counterfactual);
    }

    #[test]
    fn recorded_observation_remains_lived_autobiographical_experience() {
        let mut model = model();
        let representation = BinaryHV::random(79);
        let receipt = ingest_narrative_evidence(
            &mut model,
            NarrativeEvidenceInput {
                representation: &representation,
                description: "observed outcome",
                evidence_kind: WorldEvidenceKind::Recorded,
                source_digest: "blake3:observation",
                appraisal_positive: true,
                effort: 0.4,
                significance: 0.8,
            },
        )
        .unwrap();

        assert_eq!(receipt.memory_class(), NarrativeMemoryClass::Lived);
        assert!(receipt.recorded_episode());
        assert!(receipt.is_direct_empirical());
        assert_eq!(model.autobio.life_story.len(), 1);
    }

    #[test]
    fn replay_is_grounded_but_not_a_second_lived_episode() {
        let mut model = model();
        let mut derived = NarrativeDerivedMemory::default();
        let representation = BinaryHV::random(80);
        let receipt = ingest_narrative_evidence_with_derived_memory(
            &mut model,
            &mut derived,
            input(
                &representation,
                "exact replay of prior observation",
                WorldEvidenceKind::ReplayDerived,
                "blake3:replay",
            ),
        )
        .unwrap();

        assert!(receipt.is_direct_empirical());
        assert_eq!(receipt.memory_class(), NarrativeMemoryClass::Recollection);
        assert!(!receipt.recorded_episode());
        assert_eq!(model.autobio.life_story.len(), 0);
        assert_eq!(derived.len(), 1);
        assert_eq!(
            derived.iter().next().unwrap().memory_class,
            NarrativeMemoryClass::Recollection
        );
    }

    #[test]
    fn model_prediction_is_imagination_not_autobiography() {
        let mut model = model();
        let mut derived = NarrativeDerivedMemory::default();
        let representation = BinaryHV::random(81);
        let receipt = ingest_narrative_evidence_with_derived_memory(
            &mut model,
            &mut derived,
            input(
                &representation,
                "predicted future state",
                WorldEvidenceKind::ModelPredicted,
                "blake3:model-output",
            ),
        )
        .unwrap();

        assert_eq!(receipt.memory_class(), NarrativeMemoryClass::Imagination);
        assert!(!receipt.recorded_episode());
        assert!(!receipt.is_direct_empirical());
        assert_eq!(model.autobio.life_story.len(), 0);
        assert_eq!(derived.len(), 1);
    }

    #[test]
    fn duplicate_derived_digest_does_not_create_pseudo_experience_count() {
        let mut model = model();
        let mut derived = NarrativeDerivedMemory::default();
        let representation = BinaryHV::random(82);

        for _ in 0..2 {
            ingest_narrative_evidence_with_derived_memory(
                &mut model,
                &mut derived,
                input(
                    &representation,
                    "same replay",
                    WorldEvidenceKind::ReplayDerived,
                    "blake3:same-replay",
                ),
            )
            .unwrap();
        }

        assert_eq!(derived.len(), 1);
        assert_eq!(model.autobio.life_story.len(), 0);
    }

    #[test]
    fn zero_capacity_derived_memory_discards_without_touching_life_story() {
        let mut model = model();
        let mut derived = NarrativeDerivedMemory::new(0);
        let representation = BinaryHV::random(83);
        ingest_narrative_evidence_with_derived_memory(
            &mut model,
            &mut derived,
            input(
                &representation,
                "generated",
                WorldEvidenceKind::Counterfactual,
                "blake3:discarded",
            ),
        )
        .unwrap();

        assert!(derived.is_empty());
        assert!(model.autobio.life_story.is_empty());
    }

    #[test]
    fn missing_source_identity_fails_closed_without_mutation() {
        let mut model = model();
        let mut derived = NarrativeDerivedMemory::default();
        let representation = BinaryHV::random(84);
        assert_eq!(
            ingest_narrative_evidence_with_derived_memory(
                &mut model,
                &mut derived,
                input(
                    &representation,
                    "dream",
                    WorldEvidenceKind::Counterfactual,
                    "   ",
                ),
            ),
            Err(NarrativeEvidenceError::EmptySourceDigest)
        );
        assert!(model.autobio.life_story.is_empty());
        assert!(derived.is_empty());
    }
}
