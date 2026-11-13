// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic evaluation of source-native MusicXML notation-event graphs.
//!
//! This is the score-semantic complement to MIDI note-event and measure/barline
//! evaluation. It reports independent dimensions and never collapses them into
//! one quality scalar. All time matching is performed on exact rational
//! quarter-note beats; no floating-point modulo or time-signature inference is
//! used.

use crate::barline_eval::RationalBeat;
use crate::evidence_digest::canonical_json_sha256;
use crate::musicxml_barline_adapter::MusicXmlPartNotationGraphV1;
use crate::musicxml_notation_events::{
    MusicXmlNotationEventKindV1, MusicXmlNotationEventV1, MusicXmlNotationEventsStatusV1,
    MusicXmlPartNotationEventsV1, MusicXmlTieV1,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const MUSICXML_SCORE_EVALUATOR_VERSION_V1: &str =
    "muse-mir-019-musicxml-notation-event-evaluator-v1";
pub const MUSICXML_SCORE_MATCHING_ALGORITHM_V1: &str =
    "bounded-deterministic-kuhn-maximum-cardinality-v1";
pub const MUSICXML_SCORE_DEFAULT_ONSET_TOLERANCE_V1: (i64, i64) = (0, 1);
pub const MUSICXML_SCORE_DEFAULT_DURATION_TOLERANCE_V1: (i64, i64) = (0, 1);

const MAX_PARTS_V1: usize = 256;
const MAX_EVENTS_PER_PART_V1: usize = 20_000;
const MAX_CANDIDATE_PAIRS_HARD_V1: usize = 2_000_000;
const MAX_EDGE_SCANS_HARD_V1: usize = 10_000_000;
const MAX_AUGMENT_DEPTH_HARD_V1: usize = 512;
const MAX_ATTRIBUTE_EDIT_CELLS_HARD_V1: usize = 2_000_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlScoreEvaluationConfigV1 {
    /// Tolerance in exact quarter-note beats. Zero means exact position.
    pub onset_tolerance: RationalBeat,
    /// Tolerance in exact quarter-note beats. Zero means exact duration.
    pub duration_tolerance: RationalBeat,
    /// Maximum compatible edges in any one metric's bipartite matching graph.
    pub maximum_candidate_pairs: usize,
    /// Maximum candidate comparisons plus augmenting-path edge scans.
    pub maximum_edge_scans: usize,
    /// Maximum recursive augmenting-path depth. Exceeding this fails closed.
    pub maximum_augment_depth: usize,
    /// Maximum dynamic-programming cells used by the attribute sequence metric.
    pub maximum_attribute_edit_cells: usize,
}

impl Default for MusicXmlScoreEvaluationConfigV1 {
    fn default() -> Self {
        Self {
            onset_tolerance: RationalBeat::new(0, 1).expect("zero rational is valid"),
            duration_tolerance: RationalBeat::new(0, 1).expect("zero rational is valid"),
            maximum_candidate_pairs: 250_000,
            maximum_edge_scans: 2_000_000,
            maximum_augment_depth: 128,
            maximum_attribute_edit_cells: 250_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicXmlScoreEvaluationErrorV1 {
    InvalidConfig { field: String },
    NoGoldParts,
    NoPredictionParts,
    TooManyParts { count: usize, maximum: usize },
    DuplicatePartId { side: String, part_id: String },
    PartIdentityMismatch { graph_part_id: String, event_part_id: String },
    ScoreClockMismatch,
    MeasureCanonicalizationMismatch,
    NotationCanonicalizationMismatch,
    TooManyEvents { part_id: String, count: usize, maximum: usize },
    InvalidRational { part_id: String, field: String },
    DuplicateSourceOrdinal { part_id: String, source_order: usize },
    NonMonotonicSourceOrder { part_id: String },
    InvalidPitchAlter { part_id: String, source_order: usize },
    CandidatePairBudgetExceeded { count: usize, maximum: usize },
    MatchingWorkBudgetExceeded { scans: usize, maximum: usize },
    MatchingDepthBudgetExceeded { depth: usize, maximum: usize },
    AttributeEditBudgetExceeded { cells: usize, maximum: usize },
    ArithmeticOverflow,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlScoreMetricV1 {
    pub gold_count: usize,
    pub prediction_count: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    /// None when there are no prediction positives.
    pub precision: Option<f64>,
    /// None when there are no reference positives.
    pub recall: Option<f64>,
    /// None when either side has no positives.
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlScoreAgreementV1 {
    pub compared: usize,
    pub agreements: usize,
    pub disagreements: usize,
    pub gold_unlabelled: usize,
    pub prediction_unlabelled: usize,
    /// Accuracy over pairs where both sides supplied a label.
    pub accuracy: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlAttributeEditSummaryV1 {
    pub gold_event_count: usize,
    pub prediction_event_count: usize,
    pub edit_distance: usize,
    /// None when both attribute sequences are empty.
    pub normalized_edit_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlScoreEvaluationReportV1 {
    pub evaluator_version: String,
    pub matching_algorithm: String,
    pub config: MusicXmlScoreEvaluationConfigV1,
    pub gold_input_sha256: String,
    pub prediction_input_sha256: String,
    pub report_sha256: String,
    pub score_clock_id: String,
    pub measure_canonicalization_version: String,
    pub notation_canonicalization_version: String,
    pub gold_part_count: usize,
    pub prediction_part_count: usize,
    pub matched_supported_part_count: usize,
    pub part_presence: MusicXmlScoreMetricV1,
    /// Same event kind and onset; deliberately ignores pitch/duration.
    pub event_presence: MusicXmlScoreMetricV1,
    /// Same event kind, onset and performed duration.
    pub duration_aware_events: MusicXmlScoreMetricV1,
    /// Exact spelled pitch (step, normalized decimal alter, octave) + onset.
    pub written_pitch: MusicXmlScoreMetricV1,
    pub rest_events: MusicXmlScoreMetricV1,
    /// Same unpitched display position + onset/duration.
    pub unpitched_events: MusicXmlScoreMetricV1,
    pub sound_tie_annotations: MusicXmlScoreMetricV1,
    pub visual_tie_annotations: MusicXmlScoreMetricV1,
    /// Exact source-tree tuplet notation attached to matched events.
    pub tuplet_annotations: MusicXmlScoreMetricV1,
    pub grace_flag_agreement: MusicXmlScoreAgreementV1,
    pub cue_flag_agreement: MusicXmlScoreAgreementV1,
    pub chord_member_agreement: MusicXmlScoreAgreementV1,
    pub rest_measure_flag_agreement: MusicXmlScoreAgreementV1,
    pub written_type_agreement: MusicXmlScoreAgreementV1,
    pub dot_count_agreement: MusicXmlScoreAgreementV1,
    pub time_modification_agreement: MusicXmlScoreAgreementV1,
    pub voice_assignment: MusicXmlScoreAgreementV1,
    pub staff_assignment: MusicXmlScoreAgreementV1,
    pub attribute_timeline: MusicXmlAttributeEditSummaryV1,
    pub missing_prediction_part_ids: Vec<String>,
    pub extra_prediction_part_ids: Vec<String>,
    pub unknown_gold_part_ids: Vec<String>,
    pub unknown_prediction_part_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicXmlScoreEvaluationUnavailableReasonV1 {
    NoComparableNotationEventParts,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MusicXmlScoreEvaluationOutcomeV1 {
    Evaluated(MusicXmlScoreEvaluationReportV1),
    Unavailable {
        reason: MusicXmlScoreEvaluationUnavailableReasonV1,
        unknown_gold_part_ids: Vec<String>,
        unknown_prediction_part_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventMatchMode {
    Presence,
    Duration,
    WrittenPitch,
    Rest,
    Unpitched,
}

#[derive(Debug, Clone, Default)]
struct Counts {
    gold: usize,
    prediction: usize,
    matched: usize,
}

impl Counts {
    fn add(
        &mut self,
        gold: usize,
        prediction: usize,
        matched: usize,
    ) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
        self.gold = self.gold.checked_add(gold).ok_or(
            MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow,
        )?;
        self.prediction = self.prediction.checked_add(prediction).ok_or(
            MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow,
        )?;
        self.matched = self.matched.checked_add(matched).ok_or(
            MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow,
        )?;
        Ok(())
    }

    fn metric(&self) -> MusicXmlScoreMetricV1 {
        let false_positives = self.prediction.saturating_sub(self.matched);
        let false_negatives = self.gold.saturating_sub(self.matched);
        let precision = (self.prediction > 0)
            .then(|| self.matched as f64 / self.prediction as f64);
        let recall = (self.gold > 0).then(|| self.matched as f64 / self.gold as f64);
        let f1 = match (precision, recall) {
            (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
            (Some(_), Some(_)) => Some(0.0),
            _ => None,
        };
        MusicXmlScoreMetricV1 {
            gold_count: self.gold,
            prediction_count: self.prediction,
            true_positives: self.matched,
            false_positives,
            false_negatives,
            precision,
            recall,
            f1,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Agreement {
    compared: usize,
    agreements: usize,
    disagreements: usize,
    gold_unlabelled: usize,
    prediction_unlabelled: usize,
}

impl Agreement {
    fn add_bool(&mut self, gold: bool, prediction: bool) {
        self.compared += 1;
        if gold == prediction {
            self.agreements += 1;
        } else {
            self.disagreements += 1;
        }
    }

    fn add_option<T: PartialEq>(&mut self, gold: Option<&T>, prediction: Option<&T>) {
        match (gold, prediction) {
            (Some(gold), Some(prediction)) => {
                self.compared += 1;
                if gold == prediction {
                    self.agreements += 1;
                } else {
                    self.disagreements += 1;
                }
            }
            (None, Some(_)) => self.gold_unlabelled += 1,
            (Some(_), None) => self.prediction_unlabelled += 1,
            (None, None) => {
                self.gold_unlabelled += 1;
                self.prediction_unlabelled += 1;
            }
        }
    }

    fn value(&self) -> MusicXmlScoreAgreementV1 {
        MusicXmlScoreAgreementV1 {
            compared: self.compared,
            agreements: self.agreements,
            disagreements: self.disagreements,
            gold_unlabelled: self.gold_unlabelled,
            prediction_unlabelled: self.prediction_unlabelled,
            accuracy: (self.compared > 0)
                .then(|| self.agreements as f64 / self.compared as f64),
        }
    }
}

/// Evaluate source-native MusicXML graph semantics with bounded one-to-one
/// maximum-cardinality matching. Parts with Unknown notation events are
/// reported separately and excluded from the metric denominators.
pub fn evaluate_musicxml_score_graphs(
    gold: &[MusicXmlPartNotationGraphV1],
    prediction: &[MusicXmlPartNotationGraphV1],
    config: &MusicXmlScoreEvaluationConfigV1,
) -> Result<MusicXmlScoreEvaluationOutcomeV1, MusicXmlScoreEvaluationErrorV1> {
    validate_config(config)?;
    if gold.is_empty() {
        return Err(MusicXmlScoreEvaluationErrorV1::NoGoldParts);
    }
    if prediction.is_empty() {
        return Err(MusicXmlScoreEvaluationErrorV1::NoPredictionParts);
    }
    if gold.len() > MAX_PARTS_V1 {
        return Err(MusicXmlScoreEvaluationErrorV1::TooManyParts {
            count: gold.len(),
            maximum: MAX_PARTS_V1,
        });
    }
    if prediction.len() > MAX_PARTS_V1 {
        return Err(MusicXmlScoreEvaluationErrorV1::TooManyParts {
            count: prediction.len(),
            maximum: MAX_PARTS_V1,
        });
    }

    let gold_by_id = index_parts(gold, "gold")?;
    let prediction_by_id = index_parts(prediction, "prediction")?;
    let (score_clock_id, measure_canonicalization_version, notation_canonicalization_version) =
        validate_graph_authorities(gold, prediction)?;

    let all_part_ids: BTreeSet<&str> = gold_by_id
        .keys()
        .chain(prediction_by_id.keys())
        .copied()
        .collect();

    let mut missing_prediction_part_ids = Vec::new();
    let mut extra_prediction_part_ids = Vec::new();
    let mut unknown_gold_part_ids = Vec::new();
    let mut unknown_prediction_part_ids = Vec::new();
    let mut matched_supported_part_count = 0usize;
    let mut part_counts = Counts::default();
    let mut event_presence = Counts::default();
    let mut duration_aware_events = Counts::default();
    let mut written_pitch = Counts::default();
    let mut rest_events = Counts::default();
    let mut unpitched_events = Counts::default();
    let mut sound_ties = Counts::default();
    let mut visual_ties = Counts::default();
    let mut tuplets = Counts::default();
    let mut grace_agreement = Agreement::default();
    let mut cue_agreement = Agreement::default();
    let mut chord_agreement = Agreement::default();
    let mut rest_measure_agreement = Agreement::default();
    let mut written_type_agreement = Agreement::default();
    let mut dot_count_agreement = Agreement::default();
    let mut time_modification_agreement = Agreement::default();
    let mut voice_agreement = Agreement::default();
    let mut staff_agreement = Agreement::default();
    let mut attribute_edit_distance = 0usize;
    let mut gold_attribute_count = 0usize;
    let mut prediction_attribute_count = 0usize;
    let mut any_supported_part = false;

    for part_id in all_part_ids {
        let gold_graph = gold_by_id.get(part_id).copied();
        let prediction_graph = prediction_by_id.get(part_id).copied();
        match (gold_graph, prediction_graph) {
            (Some(gold_graph), Some(prediction_graph)) => {
                let gold_events = &gold_graph.notation_events;
                let prediction_events = &prediction_graph.notation_events;
                match (&gold_events.status, &prediction_events.status) {
                    (
                        MusicXmlNotationEventsStatusV1::Supported,
                        MusicXmlNotationEventsStatusV1::Supported,
                    ) => {
                        validate_part_events(part_id, gold_events)?;
                        validate_part_events(part_id, prediction_events)?;
                        any_supported_part = true;
                        matched_supported_part_count += 1;
                        part_counts.add(1, 1, 1)?;

                        let presence_pairs = match_events(
                            &gold_events.note_events,
                            &prediction_events.note_events,
                            config,
                            EventMatchMode::Presence,
                        )?;
                        let duration_pairs = match_events(
                            &gold_events.note_events,
                            &prediction_events.note_events,
                            config,
                            EventMatchMode::Duration,
                        )?;
                        let pitch_pairs = match_events(
                            &gold_events.note_events,
                            &prediction_events.note_events,
                            config,
                            EventMatchMode::WrittenPitch,
                        )?;
                        let rest_pairs = match_events(
                            &gold_events.note_events,
                            &prediction_events.note_events,
                            config,
                            EventMatchMode::Rest,
                        )?;
                        let unpitched_pairs = match_events(
                            &gold_events.note_events,
                            &prediction_events.note_events,
                            config,
                            EventMatchMode::Unpitched,
                        )?;

                        event_presence.add(
                            gold_events.note_events.len(),
                            prediction_events.note_events.len(),
                            presence_pairs.len(),
                        )?;
                        duration_aware_events.add(
                            gold_events.note_events.len(),
                            prediction_events.note_events.len(),
                            duration_pairs.len(),
                        )?;
                        let gold_pitch_count = gold_events
                            .note_events
                            .iter()
                            .filter(|event| is_pitched(&event.kind))
                            .count();
                        let prediction_pitch_count = prediction_events
                            .note_events
                            .iter()
                            .filter(|event| is_pitched(&event.kind))
                            .count();
                        written_pitch.add(gold_pitch_count, prediction_pitch_count, pitch_pairs.len())?;

                        let gold_rest_count = gold_events
                            .note_events
                            .iter()
                            .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Rest { .. }))
                            .count();
                        let prediction_rest_count = prediction_events
                            .note_events
                            .iter()
                            .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Rest { .. }))
                            .count();
                        rest_events.add(gold_rest_count, prediction_rest_count, rest_pairs.len())?;

                        let gold_unpitched_count = gold_events
                            .note_events
                            .iter()
                            .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Unpitched { .. }))
                            .count();
                        let prediction_unpitched_count = prediction_events
                            .note_events
                            .iter()
                            .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Unpitched { .. }))
                            .count();
                        unpitched_events.add(
                            gold_unpitched_count,
                            prediction_unpitched_count,
                            unpitched_pairs.len(),
                        )?;

                        accumulate_event_agreements(
                            &gold_events.note_events,
                            &prediction_events.note_events,
                            &presence_pairs,
                            &mut grace_agreement,
                            &mut cue_agreement,
                            &mut chord_agreement,
                            &mut rest_measure_agreement,
                            &mut written_type_agreement,
                            &mut dot_count_agreement,
                            &mut time_modification_agreement,
                            &mut voice_agreement,
                            &mut staff_agreement,
                        );

                        let gold_sound_ties = count_sound_ties(&gold_events.note_events);
                        let prediction_sound_ties = count_sound_ties(&prediction_events.note_events);
                        let gold_visual_ties = count_visual_ties(&gold_events.note_events);
                        let prediction_visual_ties = count_visual_ties(&prediction_events.note_events);
                        let gold_tuplets = count_tuplets(&gold_events.note_events);
                        let prediction_tuplets = count_tuplets(&prediction_events.note_events);
                        let sound_tie_tp = matched_annotation_count(
                            &gold_events.note_events,
                            &prediction_events.note_events,
                            &presence_pairs,
                            |event| &event.sound_ties,
                        );
                        let visual_tie_tp = matched_annotation_count(
                            &gold_events.note_events,
                            &prediction_events.note_events,
                            &presence_pairs,
                            |event| &event.visual_ties,
                        );
                        let tuplet_tp = matched_annotation_count(
                            &gold_events.note_events,
                            &prediction_events.note_events,
                            &presence_pairs,
                            |event| &event.tuplet_notations,
                        );
                        sound_ties.add(gold_sound_ties, prediction_sound_ties, sound_tie_tp)?;
                        visual_ties.add(gold_visual_ties, prediction_visual_ties, visual_tie_tp)?;
                        tuplets.add(gold_tuplets, prediction_tuplets, tuplet_tp)?;

                        let (gold_attr, prediction_attr, distance) = attribute_distance(
                            gold_events,
                            prediction_events,
                            config.maximum_attribute_edit_cells,
                        )?;
                        gold_attribute_count = gold_attribute_count
                            .checked_add(gold_attr)
                            .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
                        prediction_attribute_count = prediction_attribute_count
                            .checked_add(prediction_attr)
                            .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
                        attribute_edit_distance = attribute_edit_distance
                            .checked_add(distance)
                            .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
                    }
                    (MusicXmlNotationEventsStatusV1::Unknown { .. }, _) => {
                        unknown_gold_part_ids.push(part_id.to_string());
                        if matches!(prediction_events.status, MusicXmlNotationEventsStatusV1::Unknown { .. }) {
                            unknown_prediction_part_ids.push(part_id.to_string());
                        }
                    }
                    (_, MusicXmlNotationEventsStatusV1::Unknown { .. }) => {
                        unknown_prediction_part_ids.push(part_id.to_string());
                    }
                }
            }
            (Some(gold_graph), None) => {
                match gold_graph.notation_events.status {
                    MusicXmlNotationEventsStatusV1::Supported => {
                        validate_part_events(part_id, &gold_graph.notation_events)?;
                        any_supported_part = true;
                        missing_prediction_part_ids.push(part_id.to_string());
                        part_counts.add(1, 0, 0)?;
                        add_unmatched_part(
                            &gold_graph.notation_events,
                            true,
                            &mut event_presence,
                            &mut duration_aware_events,
                            &mut written_pitch,
                            &mut rest_events,
                            &mut unpitched_events,
                            &mut sound_ties,
                            &mut visual_ties,
                            &mut tuplets,
                            &mut gold_attribute_count,
                            &mut attribute_edit_distance,
                        )?;
                    }
                    MusicXmlNotationEventsStatusV1::Unknown { .. } => {
                        unknown_gold_part_ids.push(part_id.to_string());
                    }
                }
            }
            (None, Some(prediction_graph)) => {
                match prediction_graph.notation_events.status {
                    MusicXmlNotationEventsStatusV1::Supported => {
                        validate_part_events(part_id, &prediction_graph.notation_events)?;
                        any_supported_part = true;
                        extra_prediction_part_ids.push(part_id.to_string());
                        part_counts.add(0, 1, 0)?;
                        add_unmatched_part(
                            &prediction_graph.notation_events,
                            false,
                            &mut event_presence,
                            &mut duration_aware_events,
                            &mut written_pitch,
                            &mut rest_events,
                            &mut unpitched_events,
                            &mut sound_ties,
                            &mut visual_ties,
                            &mut tuplets,
                            &mut prediction_attribute_count,
                            &mut attribute_edit_distance,
                        )?;
                    }
                    MusicXmlNotationEventsStatusV1::Unknown { .. } => {
                        unknown_prediction_part_ids.push(part_id.to_string());
                    }
                }
            }
            (None, None) => unreachable!("part ID union always resolves on at least one side"),
        }
    }

    if !any_supported_part {
        return Ok(MusicXmlScoreEvaluationOutcomeV1::Unavailable {
            reason: MusicXmlScoreEvaluationUnavailableReasonV1::NoComparableNotationEventParts,
            unknown_gold_part_ids,
            unknown_prediction_part_ids,
        });
    }

    let mut report = MusicXmlScoreEvaluationReportV1 {
        evaluator_version: MUSICXML_SCORE_EVALUATOR_VERSION_V1.into(),
        matching_algorithm: MUSICXML_SCORE_MATCHING_ALGORITHM_V1.into(),
        config: config.clone(),
        gold_input_sha256: canonical_json_sha256(gold)
            .map_err(|_| MusicXmlScoreEvaluationErrorV1::SerializationFailed)?,
        prediction_input_sha256: canonical_json_sha256(prediction)
            .map_err(|_| MusicXML_SCORE_EVALUATIONErrorV1::SerializationFailed)?,
        report_sha256: String::new(),
        score_clock_id,
        measure_canonicalization_version,
        notation_canonicalization_version,
        gold_part_count: gold.len(),
        prediction_part_count: prediction.len(),
        matched_supported_part_count,
        part_presence: part_counts.metric(),
        event_presence: event_presence.metric(),
        duration_aware_events: duration_aware_events.metric(),
        written_pitch: written_pitch.metric(),
        rest_events: rest_events.metric(),
        unpitched_events: unpitched_events.metric(),
        sound_tie_annotations: sound_ties.metric(),
        visual_tie_annotations: visual_ties.metric(),
        tuplet_annotations: tuplets.metric(),
        grace_flag_agreement: grace_agreement.value(),
        cue_flag_agreement: cue_agreement.value(),
        chord_member_agreement: chord_agreement.value(),
        rest_measure_flag_agreement: rest_measure_agreement.value(),
        written_type_agreement: written_type_agreement.value(),
        dot_count_agreement: dot_count_agreement.value(),
        time_modification_agreement: time_modification_agreement.value(),
        voice_assignment: voice_agreement.value(),
        staff_assignment: staff_agreement.value(),
        attribute_timeline: MusicXmlAttributeEditSummaryV1 {
            gold_event_count: gold_attribute_count,
            prediction_event_count: prediction_attribute_count,
            edit_distance: attribute_edit_distance,
            normalized_edit_rate: {
                let denominator = gold_attribute_count.max(prediction_attribute_count);
                (denominator > 0).then(|| attribute_edit_distance as f64 / denominator as f64)
            },
        },
        missing_prediction_part_ids,
        extra_prediction_part_ids,
        unknown_gold_part_ids,
        unknown_prediction_part_ids,
    };
    report.report_sha256 = score_report_commitment(&report)
        .map_err(|_| MusicXmlScoreEvaluationErrorV1::SerializationFailed)?;
    Ok(MusicXmlScoreEvaluationOutcomeV1::Evaluated(report))
}

/// Recompute a report from its exact graph inputs and config. Any input or
/// metric mutation makes validation return false.
pub fn validate_musicxml_score_evaluation_report(
    gold: &[MusicXmlPartNotationGraphV1],
    prediction: &[MusicXmlPartNotationGraphV1],
    report: &MusicXmlScoreEvaluationReportV1,
) -> Result<bool, MusicXmlScoreEvaluationErrorV1> {
    match evaluate_musicxml_score_graphs(gold, prediction, &report.config)? {
        MusicXmlScoreEvaluationOutcomeV1::Evaluated(recomputed) => Ok(recomputed == *report),
        MusicXmlScoreEvaluationOutcomeV1::Unavailable { .. } => Ok(false),
    }
}

fn score_report_commitment(
    report: &MusicXmlScoreEvaluationReportV1,
) -> Result<String, serde_json::Error> {
    let mut payload = report.clone();
    payload.report_sha256.clear();
    canonical_json_sha256(&payload)
}

fn validate_config(
    config: &MusicXmlScoreEvaluationConfigV1,
) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
    for (field, value) in [
        ("onset_tolerance", config.onset_tolerance),
        ("duration_tolerance", config.duration_tolerance),
    ] {
        validate_rational(value, "<config>", field)?;
    }
    if config.maximum_candidate_pairs == 0
        || config.maximum_candidate_pairs > MAX_CANDIDATE_PAIRS_HARD_V1
    {
        return Err(MusicXmlScoreEvaluationErrorV1::InvalidConfig {
            field: "maximum_candidate_pairs".into(),
        });
    }
    if config.maximum_edge_scans == 0 || config.maximum_edge_scans > MAX_EDGE_SCANS_HARD_V1 {
        return Err(MusicXmlScoreEvaluationErrorV1::InvalidConfig {
            field: "maximum_edge_scans".into(),
        });
    }
    if config.maximum_augment_depth == 0
        || config.maximum_augment_depth > MAX_AUGMENT_DEPTH_HARD_V1
    {
        return Err(MusicXmlScoreEvaluationErrorV1::InvalidConfig {
            field: "maximum_augment_depth".into(),
        });
    }
    if config.maximum_attribute_edit_cells == 0
        || config.maximum_attribute_edit_cells > MAX_ATTRIBUTE_EDIT_CELLS_HARD_V1
    {
        return Err(MusicXmlScoreEvaluationErrorV1::InvalidConfig {
            field: "maximum_attribute_edit_cells".into(),
        });
    }
    Ok(())
}

fn index_parts<'a>(
    graphs: &'a [MusicXmlPartNotationGraphV1],
    side: &str,
) -> Result<BTreeMap<&'a str, &'a MusicXmlPartNotationGraphV1>, MusicXmlScoreEvaluationErrorV1>
{
    let mut indexed = BTreeMap::new();
    for graph in graphs {
        if graph.part_id.trim().is_empty() {
            return Err(MusicXmlScoreEvaluationErrorV1::DuplicatePartId {
                side: side.into(),
                part_id: String::new(),
            });
        }
        if indexed.insert(graph.part_id.as_str(), graph).is_some() {
            return Err(MusicXmlScoreEvaluationErrorV1::DuplicatePartId {
                side: side.into(),
                part_id: graph.part_id.clone(),
            });
        }
    }
    Ok(indexed)
}

fn validate_graph_authorities(
    gold: &[MusicXmlPartNotationGraphV1],
    prediction: &[MusicXmlPartNotationGraphV1],
) -> Result<(String, String, String), MusicXmlScoreEvaluationErrorV1> {
    let reference = &gold[0];
    let score_clock_id = reference.map.score_clock_id.clone();
    let measure_canonicalization_version = reference.map.canonicalization_version.clone();
    let notation_canonicalization_version =
        reference.notation_events.canonicalization_version.clone();
    for graph in gold.iter().chain(prediction) {
        if graph.notation_events.part_id != graph.part_id {
            return Err(MusicXmlScoreEvaluationErrorV1::PartIdentityMismatch {
                graph_part_id: graph.part_id.clone(),
                event_part_id: graph.notation_events.part_id.clone(),
            });
        }
        if graph.map.score_clock_id != score_clock_id {
            return Err(MusicXmlScoreEvaluationErrorV1::ScoreClockMismatch);
        }
        if graph.map.canonicalization_version != measure_canonicalization_version {
            return Err(MusicXmlScoreEvaluationErrorV1::MeasureCanonicalizationMismatch);
        }
        if graph.notation_events.canonicalization_version != notation_canonicalization_version {
            return Err(MusicXmlScoreEvaluationErrorV1::NotationCanonicalizationMismatch);
        }
    }
    Ok((
        score_clock_id,
        measure_canonicalization_version,
        notation_canonicalization_version,
    ))
}

fn validate_part_events(
    part_id: &str,
    events: &MusicXmlPartNotationEventsV1,
) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
    if events.note_events.len() > MAX_EVENTS_PER_PART_V1 {
        return Err(MusicXmlScoreEvaluationErrorV1::TooManyEvents {
            part_id: part_id.into(),
            count: events.note_events.len(),
            maximum: MAX_EVENTS_PER_PART_V1,
        });
    }
    let mut source_orders = BTreeSet::new();
    let mut previous_note_order = None;
    for event in &events.note_events {
        if event.part_id != part_id {
            return Err(MusicXmlScoreEvaluationErrorV1::PartIdentityMismatch {
                graph_part_id: part_id.into(),
                event_part_id: event.part_id.clone(),
            });
        }
        if previous_note_order.is_some_and(|previous| event.source_order <= previous) {
            return Err(MusicXmlScoreEvaluationErrorV1::NonMonotonicSourceOrder {
                part_id: part_id.into(),
            });
        }
        previous_note_order = Some(event.source_order);
        if !source_orders.insert(event.source_order) {
            return Err(MusicXmlScoreEvaluationErrorV1::DuplicateSourceOrdinal {
                part_id: part_id.into(),
                source_order: event.source_order,
            });
        }
        validate_rational(event.score_offset, part_id, "note.score_offset")?;
        if let Some(duration) = event.performed_duration {
            validate_rational(duration, part_id, "note.performed_duration")?;
        }
        if let MusicXmlNotationEventKindV1::Pitched(pitch) = &event.kind {
            if normalize_decimal(pitch.alter.as_deref().unwrap_or("0")).is_none() {
                return Err(MusicXmlScoreEvaluationErrorV1::InvalidPitchAlter {
                    part_id: part_id.into(),
                    source_order: event.source_order,
                });
            }
        }
    }
    for event in &events.attribute_events {
        if event.part_id != part_id {
            return Err(MusicXmlScoreEvaluationErrorV1::PartIdentityMismatch {
                graph_part_id: part_id.into(),
                event_part_id: event.part_id.clone(),
            });
        }
        if !source_orders.insert(event.source_order) {
            return Err(MusicXmlScoreEvaluationErrorV1::DuplicateSourceOrdinal {
                part_id: part_id.into(),
                source_order: event.source_order,
            });
        }
        validate_rational(event.score_offset, part_id, "attribute.score_offset")?;
    }
    for event in &events.cursor_moves {
        if event.part_id != part_id {
            return Err(MusicXmlScoreEvaluationErrorV1::PartIdentityMismatch {
                graph_part_id: part_id.into(),
                event_part_id: event.part_id.clone(),
            });
        }
        if !source_orders.insert(event.source_order) {
            return Err(MusicXmlScoreEvaluationErrorV1::DuplicateSourceOrdinal {
                part_id: part_id.into(),
                source_order: event.source_order,
            });
        }
        validate_rational(event.score_offset_before, part_id, "cursor.score_offset_before")?;
        validate_rational(event.duration, part_id, "cursor.duration")?;
        validate_rational(event.score_offset_after, part_id, "cursor.score_offset_after")?;
    }
    Ok(())
}

fn validate_rational(
    value: RationalBeat,
    part_id: &str,
    field: &str,
) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
    if value.denominator() <= 0
        || value.numerator() < 0
        || RationalBeat::new(value.numerator(), value.denominator()).ok() != Some(value)
    {
        return Err(MusicXmlScoreEvaluationErrorV1::InvalidRational {
            part_id: part_id.into(),
            field: field.into(),
        });
    }
    Ok(())
}

fn within(
    left: RationalBeat,
    right: RationalBeat,
    tolerance: RationalBeat,
) -> Result<bool, MusicXmlScoreEvaluationErrorV1> {
    let difference = if left.compare_exact(right) == Ordering::Greater {
        left.checked_sub(right)
    } else {
        right.checked_sub(left)
    }
    .map_err(|_| MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    Ok(difference.compare_exact(tolerance) != Ordering::Greater)
}

fn event_kind_matches(
    gold: &MusicXmlNotationEventKindV1,
    prediction: &MusicXmlNotationEventKindV1,
) -> bool {
    matches!(
        (gold, prediction),
        (MusicXmlNotationEventKindV1::Pitched(_), MusicXmlNotationEventKindV1::Pitched(_))
            | (MusicXmlNotationEventKindV1::Rest { .. }, MusicXmlNotationEventKindV1::Rest { .. })
            | (MusicXmlNotationEventKindV1::Unpitched { .. }, MusicXmlNotationEventKindV1::Unpitched { .. })
    )
}

fn is_pitched(kind: &MusicXmlNotationEventKindV1) -> bool {
    matches!(kind, MusicXmlNotationEventKindV1::Pitched(_))
}

fn is_rest(kind: &MusicXmlNotationEventKindV1) -> bool {
    matches!(kind, MusicXmlNotationEventKindV1::Rest { .. })
}

fn is_unpitched(kind: &MusicXmlNotationEventKindV1) -> bool {
    matches!(kind, MusicXmlNotationEventKindV1::Unpitched { .. })
}

fn normalize_decimal(value: &str) -> Option<String> {
    let (negative, unsigned) = match value.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, value.strip_prefix('+').unwrap_or(value)),
    };
    let mut pieces = unsigned.split('.');
    let whole = pieces.next()?;
    let fraction = pieces.next().unwrap_or("");
    if pieces.next().is_some()
        || whole.is_empty() && fraction.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let whole = whole.trim_start_matches('0');
    let whole = if whole.is_empty() { "0" } else { whole };
    let fraction = fraction.trim_end_matches('0');
    let is_zero = whole == "0" && fraction.is_empty();
    let mut normalized = String::new();
    if negative && !is_zero {
        normalized.push('-');
    }
    normalized.push_str(whole);
    if !fraction.is_empty() {
        normalized.push('.');
        normalized.push_str(fraction);
    }
    Some(normalized)
}

fn pitch_equal(
    gold: &MusicXmlNotationEventV1,
    prediction: &MusicXmlNotationEventV1,
) -> bool {
    match (&gold.kind, &prediction.kind) {
        (
            MusicXmlNotationEventKindV1::Pitched(gold_pitch),
            MusicXmlNotationEventKindV1::Pitched(prediction_pitch),
        ) => {
            gold_pitch.step == prediction_pitch.step
                && gold_pitch.octave == prediction_pitch.octave
                && normalize_decimal(gold_pitch.alter.as_deref().unwrap_or("0"))
                    == normalize_decimal(prediction_pitch.alter.as_deref().unwrap_or("0"))
        }
        _ => false,
    }
}

fn duration_equal(
    gold: &MusicXmlNotationEventV1,
    prediction: &MusicXmlNotationEventV1,
    tolerance: RationalBeat,
) -> Result<bool, MusicXmlScoreEvaluationErrorV1> {
    match (gold.performed_duration, prediction.performed_duration) {
        (None, None) => Ok(true),
        (Some(gold), Some(prediction)) => within(gold, prediction, tolerance),
        _ => Ok(false),
    }
}

fn candidate_matches(
    gold: &MusicXmlNotationEventV1,
    prediction: &MusicXmlNotationEventV1,
    config: &MusicXmlScoreEvaluationConfigV1,
    mode: EventMatchMode,
) -> Result<bool, MusicXmlScoreEvaluationErrorV1> {
    if !event_kind_matches(&gold.kind, &prediction.kind)
        || !within(gold.score_offset, prediction.score_offset, config.onset_tolerance)?
    {
        return Ok(false);
    }
    match mode {
        EventMatchMode::Presence => Ok(true),
        EventMatchMode::Duration => duration_equal(gold, prediction, config.duration_tolerance),
        EventMatchMode::WrittenPitch => Ok(pitch_equal(gold, prediction)),
        EventMatchMode::Rest => {
            is_rest(&gold.kind)
                .then(|| duration_equal(gold, prediction, config.duration_tolerance))
                .unwrap_or(Ok(false))
        }
        EventMatchMode::Unpitched => {
            if !is_unpitched(&gold.kind) {
                return Ok(false);
            }
            let same_display = match (&gold.kind, &prediction.kind) {
                (
                    MusicXmlNotationEventKindV1::Unpitched { display: gold_display },
                    MusicXmlNotationEventKindV1::Unpitched { display: prediction_display },
                ) => gold_display == prediction_display,
                _ => false,
            };
            Ok(same_display
                && duration_equal(gold, prediction, config.duration_tolerance)?)
        }
    }
}

fn match_events(
    gold: &[MusicXmlNotationEventV1],
    prediction: &[MusicXmlNotationEventV1],
    config: &MusicXmlScoreEvaluationConfigV1,
    mode: EventMatchMode,
) -> Result<Vec<(usize, usize)>, MusicXmlScoreEvaluationErrorV1> {
    let mut adjacency = vec![Vec::<usize>::new(); gold.len()];
    let mut candidate_pairs = 0usize;
    let mut scans = 0usize;
    for (gold_index, gold_event) in gold.iter().enumerate() {
        for (prediction_index, prediction_event) in prediction.iter().enumerate() {
            scans = scans.checked_add(1).ok_or(
                MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow,
            )?;
            if scans > config.maximum_edge_scans {
                return Err(MusicXmlScoreEvaluationErrorV1::MatchingWorkBudgetExceeded {
                    scans,
                    maximum: config.maximum_edge_scans,
                });
            }
            if candidate_matches(gold_event, prediction_event, config, mode)? {
                candidate_pairs = candidate_pairs.checked_add(1).ok_or(
                    MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow,
                )?;
                if candidate_pairs > config.maximum_candidate_pairs {
                    return Err(MusicXmlScoreEvaluationErrorV1::CandidatePairBudgetExceeded {
                        count: candidate_pairs,
                        maximum: config.maximum_candidate_pairs,
                    });
                }
                adjacency[gold_index].push(prediction_index);
            }
        }
    }

    let mut matched_prediction = vec![None; prediction.len()];
    for gold_index in 0..gold.len() {
        let mut seen_prediction = vec![false; prediction.len()];
        let _ = augment(
            gold_index,
            0,
            &adjacency,
            &mut matched_prediction,
            &mut seen_prediction,
            &mut scans,
            config,
        )?;
    }
    let mut pairs: Vec<_> = matched_prediction
        .into_iter()
        .enumerate()
        .filter_map(|(prediction_index, gold_index)| {
            gold_index.map(|gold_index| (gold_index, prediction_index))
        })
        .collect();
    pairs.sort_unstable();
    Ok(pairs)
}

fn augment(
    gold_index: usize,
    depth: usize,
    adjacency: &[Vec<usize>],
    matched_prediction: &mut [Option<usize>],
    seen_prediction: &mut [bool],
    scans: &mut usize,
    config: &MusicXmlScoreEvaluationConfigV1,
) -> Result<bool, MusicXmlScoreEvaluationErrorV1> {
    if depth > config.maximum_augment_depth {
        return Err(MusicXmlScoreEvaluationErrorV1::MatchingDepthBudgetExceeded {
            depth,
            maximum: config.maximum_augment_depth,
        });
    }
    for prediction_index in &adjacency[gold_index] {
        *scans = scans.checked_add(1).ok_or(
            MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow,
        )?;
        if *scans > config.maximum_edge_scans {
            return Err(MusicXmlScoreEvaluationErrorV1::MatchingWorkBudgetExceeded {
                scans: *scans,
                maximum: config.maximum_edge_scans,
            });
        }
        if seen_prediction[*prediction_index] {
            continue;
        }
        seen_prediction[*prediction_index] = true;
        if let Some(previous_gold) = matched_prediction[*prediction_index] {
            if augment(
                previous_gold,
                depth + 1,
                adjacency,
                matched_prediction,
                seen_prediction,
                scans,
                config,
            )? {
                matched_prediction[*prediction_index] = Some(gold_index);
                return Ok(true);
            }
        } else {
            matched_prediction[*prediction_index] = Some(gold_index);
            return Ok(true);
        }
    }
    Ok(false)
}

fn accumulate_event_agreements(
    gold: &[MusicXmlNotationEventV1],
    prediction: &[MusicXmlNotationEventV1],
    pairs: &[(usize, usize)],
    grace: &mut Agreement,
    cue: &mut Agreement,
    chord: &mut Agreement,
    rest_measure: &mut Agreement,
    written_type: &mut Agreement,
    dot_count: &mut Agreement,
    time_modification: &mut Agreement,
    voice: &mut Agreement,
    staff: &mut Agreement,
) {
    for (gold_index, prediction_index) in pairs {
        let gold_event = &gold[*gold_index];
        let prediction_event = &prediction[*prediction_index];
        grace.add_bool(gold_event.is_grace, prediction_event.is_grace);
        cue.add_bool(gold_event.is_cue, prediction_event.is_cue);
        chord.add_bool(gold_event.is_chord_member, prediction_event.is_chord_member);
        if let (
            MusicXmlNotationEventKindV1::Rest { measure: gold_measure, .. },
            MusicXmlNotationEventKindV1::Rest { measure: prediction_measure, .. },
        ) = (&gold_event.kind, &prediction_event.kind)
        {
            rest_measure.add_bool(*gold_measure, *prediction_measure);
        }
        written_type.add_option(gold_event.written_type.as_ref(), prediction_event.written_type.as_ref());
        dot_count.add_bool(gold_event.dot_count == prediction_event.dot_count, true);
        time_modification.add_option(
            gold_event.time_modification.as_ref(),
            prediction_event.time_modification.as_ref(),
        );
        voice.add_option(gold_event.voice.as_ref(), prediction_event.voice.as_ref());
        staff.add_option(gold_event.staff.as_ref(), prediction_event.staff.as_ref());
    }
}

fn matched_annotation_count<T: PartialEq>(
    gold_events: &[MusicXmlNotationEventV1],
    prediction_events: &[MusicXmlNotationEventV1],
    pairs: &[(usize, usize)],
    select: impl Fn(&MusicXmlNotationEventV1) -> &[T],
) -> usize {
    let mut matched = 0usize;
    for (gold_index, prediction_index) in pairs {
        let gold = select(&gold_events[*gold_index]);
        let prediction = select(&prediction_events[*prediction_index]);
        let mut used = vec![false; prediction.len()];
        for gold_annotation in gold {
            if let Some((index, _)) = prediction
                .iter()
                .enumerate()
                .find(|(index, candidate)| !used[*index] && *candidate == gold_annotation)
            {
                used[index] = true;
                matched += 1;
            }
        }
    }
    matched
}

fn count_sound_ties(events: &[MusicXmlNotationEventV1]) -> usize {
    events.iter().map(|event| event.sound_ties.len()).sum()
}

fn count_visual_ties(events: &[MusicXmlNotationEventV1]) -> usize {
    events.iter().map(|event| event.visual_ties.len()).sum()
}

fn count_tuplets(events: &[MusicXmlNotationEventV1]) -> usize {
    events.iter().map(|event| event.tuplet_notations.len()).sum()
}

fn attribute_distance(
    gold: &MusicXmlPartNotationEventsV1,
    prediction: &MusicXmlPartNotationEventsV1,
    maximum_cells: usize,
) -> Result<(usize, usize, usize), MusicXmlScoreEvaluationErrorV1> {
    let rows = gold.attribute_events.len().checked_add(1).ok_or(
        MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow,
    )?;
    let columns = prediction.attribute_events.len().checked_add(1).ok_or(
        MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow,
    )?;
    let cells = rows.checked_mul(columns).ok_or(
        MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow,
    )?;
    if cells > maximum_cells {
        return Err(MusicXmlScoreEvaluationErrorV1::AttributeEditBudgetExceeded {
            cells,
            maximum: maximum_cells,
        });
    }
    let mut previous: Vec<usize> = (0..columns).collect();
    let mut current = vec![0usize; columns];
    for (gold_index, gold_event) in gold.attribute_events.iter().enumerate() {
        current[0] = gold_index + 1;
        for (prediction_index, prediction_event) in
            prediction.attribute_events.iter().enumerate()
        {
            let equal = gold_event.score_offset == prediction_event.score_offset
                && gold_event.source.element == prediction_event.source.element
                && gold_event.source == prediction_event.source;
            let substitution_cost = usize::from(!equal);
            current[prediction_index + 1] = (previous[prediction_index + 1] + 1)
                .min(current[prediction_index] + 1)
                .min(previous[prediction_index] + substitution_cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    Ok((
        gold.attribute_events.len(),
        prediction.attribute_events.len(),
        previous[columns - 1],
    ))
}

fn add_unmatched_part(
    events: &MusicXmlPartNotationEventsV1,
    is_gold: bool,
    event_presence: &mut Counts,
    duration_aware_events: &mut Counts,
    written_pitch: &mut Counts,
    rest_events: &mut Counts,
    unpitched_events: &mut Counts,
    sound_ties: &mut Counts,
    visual_ties: &mut Counts,
    tuplets: &mut Counts,
    attribute_count: &mut usize,
    attribute_edit_distance: &mut usize,
) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
    let total = events.note_events.len();
    let pitches = events.note_events.iter().filter(|event| is_pitched(&event.kind)).count();
    let rests = events.note_events.iter().filter(|event| is_rest(&event.kind)).count();
    let unpitched = events.note_events.iter().filter(|event| is_unpitched(&event.kind)).count();
    let ties = count_sound_ties(&events.note_events);
    let visual = count_visual_ties(&events.note_events);
    let tuplets_count = count_tuplets(&events.note_events);
    let (gold, prediction) = if is_gold { (total, 0) } else { (0, total) };
    let (gold_pitch, prediction_pitch) = if is_gold { (pitches, 0) } else { (0, pitches) };
    let (gold_rest, prediction_rest) = if is_gold { (rests, 0) } else { (0, rests) };
    let (gold_unpitched, prediction_unpitched) =
        if is_gold { (unpitched, 0) } else { (0, unpitched) };
    event_presence.add(gold, prediction, 0)?;
    duration_aware_events.add(gold, prediction, 0)?;
    written_pitch.add(gold_pitch, prediction_pitch, 0)?;
    rest_events.add(gold_rest, prediction_rest, 0)?;
    unpitched_events.add(gold_unpitched, prediction_unpitched, 0)?;
    sound_ties.add(if is_gold { ties } else { 0 }, if is_gold { 0 } else { ties }, 0)?;
    visual_ties.add(if is_gold { visual } else { 0 }, if is_gold { 0 } else { visual }, 0)?;
    tuplets.add(
        if is_gold { tuplets_count } else { 0 },
        if is_gold { 0 } else { tuplets_count },
        0,
    )?;
    let count = events.attribute_events.len();
    *attribute_count = attribute_count
        .checked_add(count)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    *attribute_edit_distance = attribute_edit_distance
        .checked_add(count)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    Ok(())
}
