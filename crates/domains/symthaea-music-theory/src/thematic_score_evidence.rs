// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Completed-score evidence for FORM-002 thematic identities and derivations.
//!
//! FORM-002 declares thematic genealogy. This module does not trust those
//! declarations as evidence. It anchors each identity from the completed
//! score region where the work plan says that identity first appears, then
//! independently measures only transformation classes already supported by
//! the crate's transformation-aware motif-return analyzer.
//!
//! Unsupported transformation classes remain explicitly unmeasured. A
//! declaration therefore cannot become stronger merely by passing through
//! this adapter.

use crate::motif_return::{MotifReturnEvidence, compare_melodic_regions, melodic_notes_in_region};
use crate::obligation::ReturnTransformation;
use crate::rhythm::Duration;
use crate::score::Score;
use crate::thematic_identity::{
    ThematicGraphErrorV1, ThematicIdentityGraphV1, ThematicTransformationClassV1,
};
use crate::work_plan::{HierarchicalWorkPlanV1, WorkPlanErrorV1};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const THEMATIC_SCORE_EVIDENCE_VERSION: &str = "melothaea-thematic-score-evidence-v1";

/// Minimum amount of melodic material required to establish a useful thematic
/// observation. One isolated note can prove "a note existed" but cannot define
/// a melodic identity or support a transformation comparison.
pub const MIN_THEMATIC_MELODY_NOTES: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThematicIdentityObservationStatusV1 {
    Anchored,
    InsufficientMaterial,
}

/// Portable, explicit melodic identity for one observed score region.
///
/// This deliberately avoids an opaque implementation-dependent hash. Absolute
/// MIDI pitches are retained for exact identity, while semitone offsets from
/// the first pitch provide a transposition-invariant contour/pitch skeleton.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThematicMelodyFingerprintV1 {
    pub pitches_midi: Vec<u8>,
    pub pitch_offsets_from_first: Vec<i16>,
    pub relative_onsets: Vec<Duration>,
    pub durations: Vec<Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThematicIdentityObservationV1 {
    pub identity_id: String,
    pub work_node_id: String,
    pub start: Duration,
    pub end: Duration,
    pub status: ThematicIdentityObservationStatusV1,
    pub fingerprint: ThematicMelodyFingerprintV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThematicTransformationMeasurementV1 {
    pub declared_class: ThematicTransformationClassV1,
    pub expected_return_transformation: ReturnTransformation,
    pub threshold: f32,
    pub passes: bool,
    pub evidence: MotifReturnEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThematicDerivationEvidenceStatusV1 {
    Supported,
    Failed,
    PartiallyMeasured,
    NotMeasured,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThematicDerivationObservationV1 {
    pub derivation_id: String,
    pub source_identity_id: String,
    pub target_identity_id: String,
    pub source_work_node_id: String,
    pub target_work_node_id: String,
    pub status: ThematicDerivationEvidenceStatusV1,
    pub measured: Vec<ThematicTransformationMeasurementV1>,
    pub unmeasured_classes: Vec<ThematicTransformationClassV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThematicScoreEvidenceV1 {
    pub version: String,
    pub identity_observations: BTreeMap<String, ThematicIdentityObservationV1>,
    pub derivation_observations: BTreeMap<String, ThematicDerivationObservationV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThematicScoreEvidenceErrorV1 {
    InvalidWorkPlan(WorkPlanErrorV1),
    InvalidThematicGraph(ThematicGraphErrorV1),
    ScoreSpanMismatch {
        work_end: Duration,
        score_end: Duration,
    },
    MissingIdentityNode {
        identity_id: String,
        node_id: String,
    },
    MissingDerivationSource {
        derivation_id: String,
        identity_id: String,
    },
    MissingDerivationTarget {
        derivation_id: String,
        identity_id: String,
    },
}

pub fn measure_thematic_graph(
    score: &Score,
    work_plan: &HierarchicalWorkPlanV1,
    thematic_graph: &ThematicIdentityGraphV1,
) -> Result<ThematicScoreEvidenceV1, ThematicScoreEvidenceErrorV1> {
    work_plan
        .validate()
        .map_err(ThematicScoreEvidenceErrorV1::InvalidWorkPlan)?;
    thematic_graph
        .validate(work_plan)
        .map_err(ThematicScoreEvidenceErrorV1::InvalidThematicGraph)?;

    let root = work_plan
        .nodes
        .get(&work_plan.root_id)
        .expect("validated work plan has declared root");
    if root.end != score.total_beats {
        return Err(ThematicScoreEvidenceErrorV1::ScoreSpanMismatch {
            work_end: root.end,
            score_end: score.total_beats,
        });
    }

    let mut identity_observations = BTreeMap::new();
    for (identity_id, identity) in &thematic_graph.identities {
        let node = work_plan.nodes.get(&identity.introduced_in).ok_or_else(|| {
            ThematicScoreEvidenceErrorV1::MissingIdentityNode {
                identity_id: identity_id.clone(),
                node_id: identity.introduced_in.clone(),
            }
        })?;
        let notes = melodic_notes_in_region(score, node.start, node.end);
        let fingerprint = fingerprint_region(&notes, node.start);
        let status = if notes.len() >= MIN_THEMATIC_MELODY_NOTES {
            ThematicIdentityObservationStatusV1::Anchored
        } else {
            ThematicIdentityObservationStatusV1::InsufficientMaterial
        };
        identity_observations.insert(
            identity_id.clone(),
            ThematicIdentityObservationV1 {
                identity_id: identity_id.clone(),
                work_node_id: identity.introduced_in.clone(),
                start: node.start,
                end: node.end,
                status,
                fingerprint,
            },
        );
    }

    let mut derivation_observations = BTreeMap::new();
    for (derivation_id, derivation) in &thematic_graph.derivations {
        let source = thematic_graph
            .identities
            .get(&derivation.source_id)
            .ok_or_else(|| ThematicScoreEvidenceErrorV1::MissingDerivationSource {
                derivation_id: derivation_id.clone(),
                identity_id: derivation.source_id.clone(),
            })?;
        let target = thematic_graph
            .identities
            .get(&derivation.target_id)
            .ok_or_else(|| ThematicScoreEvidenceErrorV1::MissingDerivationTarget {
                derivation_id: derivation_id.clone(),
                identity_id: derivation.target_id.clone(),
            })?;
        let source_node = work_plan
            .nodes
            .get(&source.introduced_in)
            .expect("thematic graph validation checked source introduction node");
        let target_node = work_plan
            .nodes
            .get(&target.introduced_in)
            .expect("thematic graph validation checked target introduction node");

        let mut measured = Vec::new();
        let mut unmeasured_classes = Vec::new();
        for declared_class in &derivation.transformations {
            let Some((expected, threshold)) = measurable_transform(declared_class) else {
                unmeasured_classes.push(declared_class.clone());
                continue;
            };
            let evidence = compare_melodic_regions(
                score,
                source_node.start,
                source_node.end,
                target_node.start,
                target_node.end,
                expected,
            );
            let passes = evidence.meets_threshold(threshold);
            measured.push(ThematicTransformationMeasurementV1 {
                declared_class: declared_class.clone(),
                expected_return_transformation: expected,
                threshold,
                passes,
                evidence,
            });
        }

        let status = derivation_status(&measured, &unmeasured_classes);
        derivation_observations.insert(
            derivation_id.clone(),
            ThematicDerivationObservationV1 {
                derivation_id: derivation_id.clone(),
                source_identity_id: derivation.source_id.clone(),
                target_identity_id: derivation.target_id.clone(),
                source_work_node_id: source.introduced_in.clone(),
                target_work_node_id: target.introduced_in.clone(),
                status,
                measured,
                unmeasured_classes,
            },
        );
    }

    Ok(ThematicScoreEvidenceV1 {
        version: THEMATIC_SCORE_EVIDENCE_VERSION.into(),
        identity_observations,
        derivation_observations,
    })
}

fn fingerprint_region(
    notes: &[crate::score::ScoreNote],
    region_start: Duration,
) -> ThematicMelodyFingerprintV1 {
    let first_midi = notes.first().map(|note| note.pitch.midi());
    ThematicMelodyFingerprintV1 {
        pitches_midi: notes.iter().map(|note| note.pitch.midi()).collect(),
        pitch_offsets_from_first: notes
            .iter()
            .map(|note| {
                first_midi
                    .map(|first| i16::from(note.pitch.midi()) - i16::from(first))
                    .unwrap_or(0)
            })
            .collect(),
        relative_onsets: notes
            .iter()
            .map(|note| note.onset.saturating_sub(region_start))
            .collect(),
        durations: notes.iter().map(|note| note.duration).collect(),
    }
}

fn measurable_transform(
    declared: &ThematicTransformationClassV1,
) -> Option<(ReturnTransformation, f32)> {
    match declared {
        ThematicTransformationClassV1::LiteralReturn => Some((ReturnTransformation::Literal, 0.95)),
        ThematicTransformationClassV1::Transposition => {
            Some((ReturnTransformation::Transposed, 0.95))
        }
        ThematicTransformationClassV1::Inversion => Some((ReturnTransformation::Inverted, 0.85)),
        ThematicTransformationClassV1::Augmentation => {
            Some((ReturnTransformation::Augmented, 0.85))
        }
        ThematicTransformationClassV1::Diminution => {
            Some((ReturnTransformation::Diminished, 0.85))
        }
        ThematicTransformationClassV1::Fragmentation => {
            Some((ReturnTransformation::Fragmented, 0.75))
        }
        ThematicTransformationClassV1::Restoration => {
            Some((ReturnTransformation::Restored, 0.90))
        }
        ThematicTransformationClassV1::Retrograde
        | ThematicTransformationClassV1::Sequence
        | ThematicTransformationClassV1::RhythmicDisplacement
        | ThematicTransformationClassV1::MetricRecontextualization
        | ThematicTransformationClassV1::HarmonicReinterpretation
        | ThematicTransformationClassV1::RegistralMigration
        | ThematicTransformationClassV1::Reorchestration
        | ThematicTransformationClassV1::Erosion
        | ThematicTransformationClassV1::Other(_) => None,
    }
}

fn derivation_status(
    measured: &[ThematicTransformationMeasurementV1],
    unmeasured: &[ThematicTransformationClassV1],
) -> ThematicDerivationEvidenceStatusV1 {
    if measured.iter().any(|measurement| !measurement.passes) {
        ThematicDerivationEvidenceStatusV1::Failed
    } else if measured.is_empty() {
        ThematicDerivationEvidenceStatusV1::NotMeasured
    } else if unmeasured.is_empty() {
        ThematicDerivationEvidenceStatusV1::Supported
    } else {
        ThematicDerivationEvidenceStatusV1::PartiallyMeasured
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Emphasis, Key, PartId, Pitch, PitchClass, ScoreNote, ThematicDerivationV1,
        ThematicIdentityV1, ThematicOriginV1, VoiceRole, WorkNodeKindV1, WorkNodeV1,
    };

    fn note(midi: u8, onset: i64) -> ScoreNote {
        ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::from_midi(midi),
            onset: Duration::new(onset, 1),
            duration: Duration::quarter(),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn fixture() -> (Score, HierarchicalWorkPlanV1, ThematicIdentityGraphV1) {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for note in [note(60, 0), note(62, 1), note(65, 2), note(64, 3)] {
            score.push(note);
        }
        for note in [note(60, 4), note(62, 5), note(65, 6)] {
            score.push(note);
        }
        // Extend the observed work to the exact declared boundary without
        // changing the melody evidence under test.
        score.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::from_midi(48),
            onset: Duration::new(7, 1),
            duration: Duration::quarter(),
            velocity: 0.5,
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });

        let mut work = HierarchicalWorkPlanV1::new("work", Duration::new(8, 1)).unwrap();
        for (id, start, end) in [("a", 0, 4), ("a-frag", 4, 8)] {
            work.insert_node(
                id,
                WorkNodeV1 {
                    parent_id: Some("work".into()),
                    label: None,
                    kind: WorkNodeKindV1::Section,
                    start: Duration::new(start, 1),
                    end: Duration::new(end, 1),
                    functions: Vec::new(),
                },
            )
            .unwrap();
        }

        let mut graph = ThematicIdentityGraphV1::default();
        graph
            .insert_identity(
                "A",
                ThematicIdentityV1 {
                    label: None,
                    origin: ThematicOriginV1::Independent,
                    introduced_in: "a".into(),
                },
            )
            .unwrap();
        graph
            .insert_identity(
                "A-frag",
                ThematicIdentityV1 {
                    label: None,
                    origin: ThematicOriginV1::Derived,
                    introduced_in: "a-frag".into(),
                },
            )
            .unwrap();
        graph
            .insert_derivation(
                "a-fragmentation",
                ThematicDerivationV1 {
                    source_id: "A".into(),
                    target_id: "A-frag".into(),
                    transformations: vec![ThematicTransformationClassV1::Fragmentation],
                },
            )
            .unwrap();
        (score, work, graph)
    }

    #[test]
    fn observed_regions_anchor_identities_without_constructor_claims() {
        let (score, work, graph) = fixture();
        let evidence = measure_thematic_graph(&score, &work, &graph).unwrap();
        assert_eq!(
            evidence.identity_observations["A"].status,
            ThematicIdentityObservationStatusV1::Anchored
        );
        assert_eq!(
            evidence.identity_observations["A-frag"].status,
            ThematicIdentityObservationStatusV1::Anchored
        );
        assert_eq!(
            evidence.identity_observations["A"].fingerprint.pitch_offsets_from_first,
            vec![0, 2, 5, 4]
        );
    }

    #[test]
    fn fragmentation_uses_existing_transformation_aware_measurement() {
        let (score, work, graph) = fixture();
        let evidence = measure_thematic_graph(&score, &work, &graph).unwrap();
        let derivation = &evidence.derivation_observations["a-fragmentation"];
        assert_eq!(derivation.status, ThematicDerivationEvidenceStatusV1::Supported);
        assert_eq!(derivation.measured.len(), 1);
        assert_eq!(
            derivation.measured[0].expected_return_transformation,
            ReturnTransformation::Fragmented
        );
        assert!(derivation.measured[0].passes);
    }

    #[test]
    fn unsupported_relationships_remain_explicitly_unmeasured() {
        let (score, work, mut graph) = fixture();
        graph
            .derivations
            .get_mut("a-fragmentation")
            .unwrap()
            .transformations = vec![ThematicTransformationClassV1::Reorchestration];
        let evidence = measure_thematic_graph(&score, &work, &graph).unwrap();
        let derivation = &evidence.derivation_observations["a-fragmentation"];
        assert_eq!(derivation.status, ThematicDerivationEvidenceStatusV1::NotMeasured);
        assert!(derivation.measured.is_empty());
        assert_eq!(
            derivation.unmeasured_classes,
            vec![ThematicTransformationClassV1::Reorchestration]
        );
    }

    #[test]
    fn mixed_supported_and_unsupported_relationships_are_partial_not_full_support() {
        let (score, work, mut graph) = fixture();
        graph
            .derivations
            .get_mut("a-fragmentation")
            .unwrap()
            .transformations = vec![
            ThematicTransformationClassV1::Fragmentation,
            ThematicTransformationClassV1::Reorchestration,
        ];
        let evidence = measure_thematic_graph(&score, &work, &graph).unwrap();
        let derivation = &evidence.derivation_observations["a-fragmentation"];
        assert_eq!(
            derivation.status,
            ThematicDerivationEvidenceStatusV1::PartiallyMeasured
        );
    }

    #[test]
    fn damaged_fragment_fails_instead_of_becoming_unmeasured() {
        let (mut score, work, graph) = fixture();
        for note in &mut score.notes {
            if note.role == VoiceRole::Melody && note.onset.beats() >= 4.0 {
                note.pitch = Pitch::from_midi(84);
            }
        }
        let evidence = measure_thematic_graph(&score, &work, &graph).unwrap();
        assert_eq!(
            evidence.derivation_observations["a-fragmentation"].status,
            ThematicDerivationEvidenceStatusV1::Failed
        );
    }

    #[test]
    fn one_note_region_is_not_promoted_to_thematic_anchor() {
        let (mut score, work, graph) = fixture();
        score.notes.retain(|note| {
            note.role != VoiceRole::Melody
                || note.onset.beats() < 4.0
                || note.onset.beats() == 4.0
        });
        let evidence = measure_thematic_graph(&score, &work, &graph).unwrap();
        assert_eq!(
            evidence.identity_observations["A-frag"].status,
            ThematicIdentityObservationStatusV1::InsufficientMaterial
        );
    }

    #[test]
    fn work_and_score_must_describe_the_same_total_span() {
        let (mut score, work, graph) = fixture();
        score.total_beats = Duration::new(7, 1);
        assert!(matches!(
            measure_thematic_graph(&score, &work, &graph),
            Err(ThematicScoreEvidenceErrorV1::ScoreSpanMismatch { .. })
        ));
    }
}
