// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact completed-score Retrograde evidence layered above FORM-002B.
//!
//! `ThematicScoreEvidenceV1` intentionally remains frozen: this module reuses
//! its observed identity fingerprints and adds one independent evidence channel
//! for the FORM-002 `Retrograde` class. The foundational `ReturnTransformation`
//! enum is not widened, avoiding semantic churn in Sonata/foundry/fingerprint
//! consumers.
//!
//! Authority here is deliberately strict. A supported retrograde requires the
//! target's observed pitched-event order and note-duration order, when reversed,
//! to match the source exactly after removing one constant pitch translation.
//! No weighted similarity threshold is promoted into a pass.
//!
//! `ScoreNote` contains pitched events, not explicit rests, so rest/gap reversal
//! is outside this V1 measurement and is recorded as unmeasured. Palindromic
//! material that is equally exact forward and backward is indeterminate rather
//! than being credited as demonstrated order reversal.

use crate::score::Score;
use crate::thematic_identity::{ThematicIdentityGraphV1, ThematicTransformationClassV1};
use crate::thematic_score_evidence::{
    ThematicIdentityObservationStatusV1, ThematicMelodyFingerprintV1,
    ThematicScoreEvidenceErrorV1, ThematicScoreEvidenceV1, measure_thematic_graph,
};
use crate::work_plan::HierarchicalWorkPlanV1;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const THEMATIC_RETROGRADE_EVIDENCE_VERSION: &str =
    "melothaea-thematic-retrograde-evidence-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThematicRetrogradeStatusV1 {
    /// Exact pitched-event + duration reversal after one constant transposition,
    /// and the same material is not also exact in forward order.
    SupportedExact,
    /// Material is insufficient to distinguish an ordered transformation.
    InsufficientMaterial,
    /// Forward and reversed relations are both exact, so direction is not
    /// identifiable from the observed pitched-event material.
    IndeterminateSymmetry,
    /// The strict completed-score relation is not present.
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThematicRetrogradeEvidenceV1 {
    pub derivation_id: String,
    pub source_identity_id: String,
    pub target_identity_id: String,
    pub source_note_count: usize,
    pub target_note_count: usize,
    /// Aligned positions compared in the shorter observed sequence.
    pub aligned_note_count: usize,
    /// Pitch matches after reversing target event order and normalizing each
    /// sequence to its first pitch. This makes the channel invariant to one
    /// constant transposition caused by a different tonal context.
    pub reversed_pitch_offset_matches: usize,
    /// Exact note-duration matches after reversing target event order.
    pub reversed_duration_matches: usize,
    /// Forward controls use the same transposition-invariant pitch comparison.
    pub forward_pitch_offset_matches: usize,
    pub forward_duration_matches: usize,
    /// Constant semitone translation between source opening pitch and the
    /// reversed target opening pitch, when both exist.
    pub reversed_constant_transposition_semitones: Option<i16>,
    pub exact_retrograde_relation: bool,
    pub exact_forward_relation: bool,
    /// Pitched `ScoreNote` evidence cannot reconstruct omitted rests/gaps.
    pub rest_gap_order_measured: bool,
    pub status: ThematicRetrogradeStatusV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThematicRetrogradeEvidenceSetV1 {
    pub version: String,
    /// Retain the exact FORM-002B observation substrate consumed by this layer.
    pub base_evidence: ThematicScoreEvidenceV1,
    /// Only derivations that explicitly declare standalone `Retrograde` receive
    /// records. Ordered composites encoded as `Other(...)` remain outside V1.
    pub derivations: BTreeMap<String, ThematicRetrogradeEvidenceV1>,
}

pub fn measure_thematic_retrograde_evidence(
    score: &Score,
    work_plan: &HierarchicalWorkPlanV1,
    thematic_graph: &ThematicIdentityGraphV1,
) -> Result<ThematicRetrogradeEvidenceSetV1, ThematicScoreEvidenceErrorV1> {
    let base_evidence = measure_thematic_graph(score, work_plan, thematic_graph)?;
    let mut derivations = BTreeMap::new();

    for (derivation_id, derivation) in &thematic_graph.derivations {
        if !derivation
            .transformations
            .contains(&ThematicTransformationClassV1::Retrograde)
        {
            continue;
        }

        let source = base_evidence
            .identity_observations
            .get(&derivation.source_id)
            .expect("FORM-002B measured every validated source identity");
        let target = base_evidence
            .identity_observations
            .get(&derivation.target_id)
            .expect("FORM-002B measured every validated target identity");

        let evidence = measure_pair(
            derivation_id,
            &derivation.source_id,
            &derivation.target_id,
            source.status,
            &source.fingerprint,
            target.status,
            &target.fingerprint,
        );
        derivations.insert(derivation_id.clone(), evidence);
    }

    Ok(ThematicRetrogradeEvidenceSetV1 {
        version: THEMATIC_RETROGRADE_EVIDENCE_VERSION.into(),
        base_evidence,
        derivations,
    })
}

fn measure_pair(
    derivation_id: &str,
    source_identity_id: &str,
    target_identity_id: &str,
    source_status: ThematicIdentityObservationStatusV1,
    source: &ThematicMelodyFingerprintV1,
    target_status: ThematicIdentityObservationStatusV1,
    target: &ThematicMelodyFingerprintV1,
) -> ThematicRetrogradeEvidenceV1 {
    let source_count = source.pitches_midi.len();
    let target_count = target.pitches_midi.len();
    let aligned = source_count.min(target_count);

    let reversed_target_pitches: Vec<u8> = target.pitches_midi.iter().rev().copied().collect();
    let reversed_target_durations: Vec<_> = target.durations.iter().rev().copied().collect();

    let source_offsets = normalized_pitch_offsets(&source.pitches_midi);
    let reversed_offsets = normalized_pitch_offsets(&reversed_target_pitches);
    let forward_offsets = normalized_pitch_offsets(&target.pitches_midi);

    let reversed_pitch_offset_matches = source_offsets
        .iter()
        .zip(&reversed_offsets)
        .filter(|(left, right)| left == right)
        .count();
    let reversed_duration_matches = source
        .durations
        .iter()
        .zip(&reversed_target_durations)
        .filter(|(left, right)| left == right)
        .count();
    let forward_pitch_offset_matches = source_offsets
        .iter()
        .zip(&forward_offsets)
        .filter(|(left, right)| left == right)
        .count();
    let forward_duration_matches = source
        .durations
        .iter()
        .zip(&target.durations)
        .filter(|(left, right)| left == right)
        .count();

    let enough_material = source_status == ThematicIdentityObservationStatusV1::Anchored
        && target_status == ThematicIdentityObservationStatusV1::Anchored;
    let same_length = source_count == target_count;
    let exact_retrograde_relation = enough_material
        && same_length
        && reversed_pitch_offset_matches == source_count
        && reversed_duration_matches == source_count;
    let exact_forward_relation = enough_material
        && same_length
        && forward_pitch_offset_matches == source_count
        && forward_duration_matches == source_count;

    let status = if !enough_material {
        ThematicRetrogradeStatusV1::InsufficientMaterial
    } else if exact_retrograde_relation && exact_forward_relation {
        ThematicRetrogradeStatusV1::IndeterminateSymmetry
    } else if exact_retrograde_relation {
        ThematicRetrogradeStatusV1::SupportedExact
    } else {
        ThematicRetrogradeStatusV1::Failed
    };

    let reversed_constant_transposition_semitones = source
        .pitches_midi
        .first()
        .zip(reversed_target_pitches.first())
        .map(|(source_pitch, target_pitch)| i16::from(*target_pitch) - i16::from(*source_pitch));

    ThematicRetrogradeEvidenceV1 {
        derivation_id: derivation_id.into(),
        source_identity_id: source_identity_id.into(),
        target_identity_id: target_identity_id.into(),
        source_note_count: source_count,
        target_note_count: target_count,
        aligned_note_count: aligned,
        reversed_pitch_offset_matches,
        reversed_duration_matches,
        forward_pitch_offset_matches,
        forward_duration_matches,
        reversed_constant_transposition_semitones,
        exact_retrograde_relation,
        exact_forward_relation,
        rest_gap_order_measured: false,
        status,
    }
}

fn normalized_pitch_offsets(pitches: &[u8]) -> Vec<i16> {
    let Some(first) = pitches.first().copied() else {
        return Vec::new();
    };
    pitches
        .iter()
        .map(|pitch| i16::from(*pitch) - i16::from(first))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Emphasis, FormalFunctionV1, Key, PartId, Pitch, PitchClass, Score,
        ScoreNote, ThematicDerivationV1, ThematicIdentityV1, ThematicOriginV1, VoiceRole,
        WorkNodeKindV1, WorkNodeV1,
    };

    fn note(midi: u8, onset: i64, duration_num: i64, duration_den: i64) -> ScoreNote {
        ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::from_midi(midi),
            onset: Duration::new(onset, 1),
            duration: Duration::new(duration_num, duration_den),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn two_section_work() -> HierarchicalWorkPlanV1 {
        let mut work = HierarchicalWorkPlanV1::new("work", Duration::new(8, 1)).unwrap();
        work.insert_node(
            "source",
            WorkNodeV1 {
                parent_id: Some("work".into()),
                label: None,
                kind: WorkNodeKindV1::Section,
                start: Duration::zero(),
                end: Duration::new(4, 1),
                functions: vec![FormalFunctionV1::Establish],
            },
        )
        .unwrap();
        work.insert_node(
            "target",
            WorkNodeV1 {
                parent_id: Some("work".into()),
                label: None,
                kind: WorkNodeKindV1::Section,
                start: Duration::new(4, 1),
                end: Duration::new(8, 1),
                functions: vec![FormalFunctionV1::Return],
            },
        )
        .unwrap();
        work
    }

    fn graph(class: ThematicTransformationClassV1) -> ThematicIdentityGraphV1 {
        let mut graph = ThematicIdentityGraphV1::default();
        graph
            .insert_identity(
                "p",
                ThematicIdentityV1 {
                    label: None,
                    origin: ThematicOriginV1::Independent,
                    introduced_in: "source".into(),
                },
            )
            .unwrap();
        graph
            .insert_identity(
                "r",
                ThematicIdentityV1 {
                    label: None,
                    origin: ThematicOriginV1::Derived,
                    introduced_in: "target".into(),
                },
            )
            .unwrap();
        graph
            .insert_derivation(
                "p-to-r",
                ThematicDerivationV1 {
                    source_id: "p".into(),
                    target_id: "r".into(),
                    transformations: vec![class],
                },
            )
            .unwrap();
        graph
    }

    fn score(source: &[(u8, i64, i64)], target: &[(u8, i64, i64)]) -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        let mut source_cursor = 0_i64;
        for &(pitch, num, den) in source {
            score.push(note(pitch, source_cursor, num, den));
            source_cursor += 1;
        }
        let mut target_cursor = 4_i64;
        for &(pitch, num, den) in target {
            score.push(note(pitch, target_cursor, num, den));
            target_cursor += 1;
        }
        // Work binding is about the declared 8-beat work span. Add a non-melody
        // event at the final boundary so Score::total_beats equals the work end
        // without contaminating thematic observations.
        let mut tail = note(48, 7, 1, 1);
        tail.role = VoiceRole::Bass;
        score.push(tail);
        score
    }

    #[test]
    fn exact_transposed_retrograde_is_supported() {
        let source = [(60, 1, 1), (62, 1, 2), (65, 3, 2), (69, 1, 1)];
        // Reverse pitch AND duration order, then transpose every pitch +5.
        let target = [(74, 1, 1), (70, 3, 2), (67, 1, 2), (65, 1, 1)];
        let evidence = measure_thematic_retrograde_evidence(
            &score(&source, &target),
            &two_section_work(),
            &graph(ThematicTransformationClassV1::Retrograde),
        )
        .unwrap();
        let record = &evidence.derivations["p-to-r"];
        assert_eq!(record.status, ThematicRetrogradeStatusV1::SupportedExact);
        assert!(record.exact_retrograde_relation);
        assert!(!record.exact_forward_relation);
        assert_eq!(record.reversed_constant_transposition_semitones, Some(5));
        assert_eq!(record.reversed_pitch_offset_matches, 4);
        assert_eq!(record.reversed_duration_matches, 4);
        assert!(!record.rest_gap_order_measured);
    }

    #[test]
    fn one_damaged_event_fails_strict_retrograde_relation() {
        let source = [(60, 1, 1), (62, 1, 2), (65, 3, 2), (69, 1, 1)];
        let target = [(74, 1, 1), (71, 3, 2), (67, 1, 2), (65, 1, 1)];
        let evidence = measure_thematic_retrograde_evidence(
            &score(&source, &target),
            &two_section_work(),
            &graph(ThematicTransformationClassV1::Retrograde),
        )
        .unwrap();
        let record = &evidence.derivations["p-to-r"];
        assert_eq!(record.status, ThematicRetrogradeStatusV1::Failed);
        assert!(!record.exact_retrograde_relation);
    }

    #[test]
    fn palindromic_material_is_indeterminate_not_false_positive() {
        let source = [(60, 1, 1), (64, 1, 2), (60, 1, 1)];
        let target = [(65, 1, 1), (69, 1, 2), (65, 1, 1)];
        let evidence = measure_thematic_retrograde_evidence(
            &score(&source, &target),
            &two_section_work(),
            &graph(ThematicTransformationClassV1::Retrograde),
        )
        .unwrap();
        let record = &evidence.derivations["p-to-r"];
        assert!(record.exact_retrograde_relation);
        assert!(record.exact_forward_relation);
        assert_eq!(
            record.status,
            ThematicRetrogradeStatusV1::IndeterminateSymmetry
        );
    }

    #[test]
    fn insufficient_material_never_receives_retrograde_authority() {
        let source = [(60, 1, 1)];
        let target = [(65, 1, 1)];
        let evidence = measure_thematic_retrograde_evidence(
            &score(&source, &target),
            &two_section_work(),
            &graph(ThematicTransformationClassV1::Retrograde),
        )
        .unwrap();
        assert_eq!(
            evidence.derivations["p-to-r"].status,
            ThematicRetrogradeStatusV1::InsufficientMaterial
        );
    }

    #[test]
    fn unrelated_transformation_class_does_not_receive_retrograde_record() {
        let source = [(60, 1, 1), (62, 1, 1)];
        let target = [(67, 1, 1), (65, 1, 1)];
        let evidence = measure_thematic_retrograde_evidence(
            &score(&source, &target),
            &two_section_work(),
            &graph(ThematicTransformationClassV1::Inversion),
        )
        .unwrap();
        assert!(evidence.derivations.is_empty());
    }

    #[test]
    fn custom_ordered_composite_remains_outside_retrograde_v1() {
        let source = [(60, 1, 1), (62, 1, 1)];
        let target = [(67, 1, 1), (65, 1, 1)];
        let evidence = measure_thematic_retrograde_evidence(
            &score(&source, &target),
            &two_section_work(),
            &graph(ThematicTransformationClassV1::Other(
                "prog-suite:retrograde-inversion-v1".into(),
            )),
        )
        .unwrap();
        assert!(evidence.derivations.is_empty());
    }
}
