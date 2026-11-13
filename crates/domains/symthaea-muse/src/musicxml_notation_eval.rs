// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, multi-dimensional evaluation of source-native MusicXML graphs.
//!
//! This evaluator compares notated score events in exact rational quarter-beat
//! coordinates. It is not an audio-to-MIDI metric, a MIREX MV2H implementation,
//! MUSTER/OMR-NED parity, or a playback-similarity score. Those objectives have
//! distinct clocks and protocols and remain separate.
//!
//! Event presence matching is one-to-one and ordered. For a fixed event family,
//! admissible pairs form a convex one-dimensional interval graph on score onset.
//! The earliest-compatible sorted matcher is maximum-cardinality for that
//! graph; source ordinals provide deterministic tie-breaking for simultaneous
//! events. Annotation fidelity is scored only after event correspondence.

use crate::barline_eval::{MeasureMapStatus, RationalBeat};
use crate::evidence_digest::canonical_json_sha256;
use crate::musicxml_barline_adapter::MusicXmlPartNotationGraphV1;
use crate::musicxml_notation_events::{
    MusicXmlNotationEventKindV1, MusicXmlNotationEventsStatusV1,
    MusicXmlPartNotationEventsV1,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const MUSICXML_NOTATION_EVALUATOR_VERSION_V1: &str =
    "muse-mir-019-musicxml-notation-event-evaluator-v1";
pub const MUSICXML_NOTATION_EVALUATION_SCHEMA_V1: &str =
    "muse-musicxml-notation-evaluation-v1";
pub const MUSICXML_NOTATION_EVALUATION_MAX_PARTS_V1: usize = 256;
pub const MUSICXML_NOTATION_EVALUATION_MAX_EVENTS_PER_PART_V1: usize = 200_000;
pub const MUSICXML_NOTATION_EVALUATION_MAX_TOTAL_EVENTS_V1: usize = 2_000_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlNotationEvaluationConfigV1 {
    /// Symmetric tolerance in exact quarter-note beats for event correspondence.
    pub onset_tolerance: RationalBeat,
    /// Symmetric tolerance in exact quarter-note beats for performed duration.
    pub duration_tolerance: RationalBeat,
}

impl Default for MusicXmlNotationEvaluationConfigV1 {
    fn default() -> Self {
        Self {
            onset_tolerance: RationalBeat::new(1, 32)
                .expect("positive rational fixture"),
            duration_tolerance: RationalBeat::new(1, 32)
                .expect("positive rational fixture"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationMetricV1 {
    pub true_positives: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationErrorDistributionV1 {
    pub sample_count: u64,
    pub mean_absolute_beats: Option<f64>,
    pub median_absolute_beats: Option<f64>,
    pub p95_absolute_beats: Option<f64>,
    pub maximum_absolute_beats: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationPartScorecardV1 {
    /// None denotes the micro-aggregate across all matched parts.
    pub part_id: Option<String>,
    pub gold_note_event_count: u64,
    pub prediction_note_event_count: u64,
    pub gold_attribute_event_count: u64,
    pub prediction_attribute_event_count: u64,
    pub gold_auxiliary_event_count: u64,
    pub prediction_auxiliary_event_count: u64,
    pub gold_cursor_move_count: u64,
    pub prediction_cursor_move_count: u64,
    pub event_presence: MusicXmlNotationMetricV1,
    pub pitched_event_presence: MusicXmlNotationMetricV1,
    pub rest_event_presence: MusicXmlNotationMetricV1,
    pub unpitched_event_presence: MusicXmlNotationMetricV1,
    pub pitch_spelling: MusicXmlNotationMetricV1,
    pub performed_duration: MusicXmlNotationMetricV1,
    pub written_value: MusicXmlNotationMetricV1,
    pub grace_cue_chord_semantics: MusicXmlNotationMetricV1,
    pub voice_identity: MusicXmlNotationMetricV1,
    pub staff_identity: MusicXmlNotationMetricV1,
    pub instrument_identity: MusicXmlNotationMetricV1,
    pub sound_tie_semantics: MusicXmlNotationMetricV1,
    pub visual_tie_semantics: MusicXmlNotationMetricV1,
    pub tuplet_notation: MusicXmlNotationMetricV1,
    pub rest_semantics: MusicXmlNotationMetricV1,
    pub unpitched_display_semantics: MusicXmlNotationMetricV1,
    pub attribute_event_presence: MusicXmlNotationMetricV1,
    pub attribute_event_exactness: MusicXmlNotationMetricV1,
    pub auxiliary_event_presence: MusicXmlNotationMetricV1,
    pub auxiliary_event_exactness: MusicXmlNotationMetricV1,
    pub cursor_move_presence: MusicXmlNotationMetricV1,
    pub cursor_move_exactness: MusicXmlNotationMetricV1,
    pub onset_error: MusicXmlNotationErrorDistributionV1,
    pub duration_error: MusicXmlNotationErrorDistributionV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationEvaluationReportV1 {
    pub schema: String,
    pub evaluator_version: String,
    pub config: MusicXmlNotationEvaluationConfigV1,
    pub score_clock_id: String,
    pub gold_graph_sha256: String,
    pub prediction_graph_sha256: String,
    pub gold_part_count: u64,
    pub prediction_part_count: u64,
    pub part_scorecards: Vec<MusicXmlNotationPartScorecardV1>,
    /// Micro-count aggregation. Per-part scorecards remain authoritative for
    /// exposing a weak or unresolved part.
    pub aggregate: MusicXmlNotationPartScorecardV1,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicXmlNotationEvaluationErrorV1 {
    EmptyGoldGraph,
    EmptyPredictionGraph,
    TooManyParts { count: usize, maximum: usize },
    TooManyEvents { count: usize, maximum: usize },
    DuplicatePartId { part_id: String },
    PartSetMismatch,
    ScoreClockMismatch,
    CanonicalizationMismatch,
    UnsupportedMeasureMap { part_id: String },
    UnknownNotationEvents { part_id: String },
    GraphPartIdentityMismatch { part_id: String },
    DuplicateSourceOrdinal { part_id: String, ordinal: usize },
    SourceMeasureIndexOutOfRange { part_id: String, index: usize },
    EventOutsideMeasureSpan { part_id: String, source_order: usize },
    ArithmeticOverflow,
    SerializationFailed,
    ReportReplayMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventFamily {
    Pitched,
    Rest,
    Unpitched,
}

impl EventFamily {
    fn key(self) -> &'static str {
        match self {
            Self::Pitched => "pitched",
            Self::Rest => "rest",
            Self::Unpitched => "unpitched",
        }
    }
}

#[derive(Debug, Clone)]
struct PositionedSource {
    index: usize,
    offset: RationalBeat,
    source_order: usize,
    key: String,
}

#[derive(Debug, Clone, Default)]
struct MetricCounts {
    true_positives: u64,
    false_positives: u64,
    false_negatives: u64,
}

#[derive(Debug, Clone, Default)]
struct PartCounts {
    part_id: Option<String>,
    gold_note_event_count: u64,
    prediction_note_event_count: u64,
    gold_attribute_event_count: u64,
    prediction_attribute_event_count: u64,
    gold_auxiliary_event_count: u64,
    prediction_auxiliary_event_count: u64,
    gold_cursor_move_count: u64,
    prediction_cursor_move_count: u64,
    event_presence: MetricCounts,
    pitched_event_presence: MetricCounts,
    rest_event_presence: MetricCounts,
    unpitched_event_presence: MetricCounts,
    pitch_spelling: MetricCounts,
    performed_duration: MetricCounts,
    written_value: MetricCounts,
    grace_cue_chord_semantics: MetricCounts,
    voice_identity: MetricCounts,
    staff_identity: MetricCounts,
    instrument_identity: MetricCounts,
    sound_tie_semantics: MetricCounts,
    visual_tie_semantics: MetricCounts,
    tuplet_notation: MetricCounts,
    rest_semantics: MetricCounts,
    unpitched_display_semantics: MetricCounts,
    attribute_event_presence: MetricCounts,
    attribute_event_exactness: MetricCounts,
    auxiliary_event_presence: MetricCounts,
    auxiliary_event_exactness: MetricCounts,
    cursor_move_presence: MetricCounts,
    cursor_move_exactness: MetricCounts,
    onset_errors: Vec<f64>,
    duration_errors: Vec<f64>,
}

fn metric_from_counts(counts: &MetricCounts) -> MusicXmlNotationMetricV1 {
    let prediction_count = counts.true_positives + counts.false_positives;
    let gold_count = counts.true_positives + counts.false_negatives;
    let precision = (prediction_count > 0)
        .then_some(counts.true_positives as f64 / prediction_count as f64);
    let recall = (gold_count > 0)
        .then_some(counts.true_positives as f64 / gold_count as f64);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };
    MusicXmlNotationMetricV1 {
        true_positives: counts.true_positives,
        false_positives: counts.false_positives,
        false_negatives: counts.false_negatives,
        precision,
        recall,
        f1,
    }
}

fn metric_counts(gold: usize, prediction: usize, true_positives: usize) -> MetricCounts {
    let true_positives = true_positives.min(gold).min(prediction) as u64;
    MetricCounts {
        true_positives,
        false_positives: prediction as u64 - true_positives,
        false_negatives: gold as u64 - true_positives,
    }
}

fn combine_counts(target: &mut MetricCounts, source: &MetricCounts) {
    target.true_positives += source.true_positives;
    target.false_positives += source.false_positives;
    target.false_negatives += source.false_negatives;
}

fn combine_part_counts(target: &mut PartCounts, source: &PartCounts) {
    target.gold_note_event_count += source.gold_note_event_count;
    target.prediction_note_event_count += source.prediction_note_event_count;
    target.gold_attribute_event_count += source.gold_attribute_event_count;
    target.prediction_attribute_event_count += source.prediction_attribute_event_count;
    target.gold_auxiliary_event_count += source.gold_auxiliary_event_count;
    target.prediction_auxiliary_event_count += source.prediction_auxiliary_event_count;
    target.gold_cursor_move_count += source.gold_cursor_move_count;
    target.prediction_cursor_move_count += source.prediction_cursor_move_count;
    macro_rules! combine {
        ($($field:ident),+ $(,)?) => {
            $(combine_counts(&mut target.$field, &source.$field);)+
        };
    }
    combine!(
        event_presence,
        pitched_event_presence,
        rest_event_presence,
        unpitched_event_presence,
        pitch_spelling,
        performed_duration,
        written_value,
        grace_cue_chord_semantics,
        voice_identity,
        staff_identity,
        instrument_identity,
        sound_tie_semantics,
        visual_tie_semantics,
        tuplet_notation,
        rest_semantics,
        unpitched_display_semantics,
        attribute_event_presence,
        attribute_event_exactness,
        auxiliary_event_presence,
        auxiliary_event_exactness,
        cursor_move_presence,
        cursor_move_exactness,
    );
    target.onset_errors.extend_from_slice(&source.onset_errors);
    target.duration_errors.extend_from_slice(&source.duration_errors);
}

fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = (((sorted.len() - 1) as f64) * q).round() as usize;
    sorted.get(index).copied()
}

fn error_distribution(values: &[f64]) -> MusicXmlNotationErrorDistributionV1 {
    if values.is_empty() {
        return MusicXmlNotationErrorDistributionV1 {
            sample_count: 0,
            mean_absolute_beats: None,
            median_absolute_beats: None,
            p95_absolute_beats: None,
            maximum_absolute_beats: None,
        };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    MusicXmlNotationErrorDistributionV1 {
        sample_count: sorted.len() as u64,
        mean_absolute_beats: Some(sorted.iter().sum::<f64>() / sorted.len() as f64),
        median_absolute_beats: quantile(&sorted, 0.5),
        p95_absolute_beats: quantile(&sorted, 0.95),
        maximum_absolute_beats: sorted.last().copied(),
    }
}

fn scorecard_from_counts(counts: &PartCounts) -> MusicXmlNotationPartScorecardV1 {
    macro_rules! metric {
        ($field:ident) => {
            metric_from_counts(&counts.$field)
        };
    }
    MusicXmlNotationPartScorecardV1 {
        part_id: counts.part_id.clone(),
        gold_note_event_count: counts.gold_note_event_count,
        prediction_note_event_count: counts.prediction_note_event_count,
        gold_attribute_event_count: counts.gold_attribute_event_count,
        prediction_attribute_event_count: counts.prediction_attribute_event_count,
        gold_auxiliary_event_count: counts.gold_auxiliary_event_count,
        prediction_auxiliary_event_count: counts.prediction_auxiliary_event_count,
        gold_cursor_move_count: counts.gold_cursor_move_count,
        prediction_cursor_move_count: counts.prediction_cursor_move_count,
        event_presence: metric!(event_presence),
        pitched_event_presence: metric!(pitched_event_presence),
        rest_event_presence: metric!(rest_event_presence),
        unpitched_event_presence: metric!(unpitched_event_presence),
        pitch_spelling: metric!(pitch_spelling),
        performed_duration: metric!(performed_duration),
        written_value: metric!(written_value),
        grace_cue_chord_semantics: metric!(grace_cue_chord_semantics),
        voice_identity: metric!(voice_identity),
        staff_identity: metric!(staff_identity),
        instrument_identity: metric!(instrument_identity),
        sound_tie_semantics: metric!(sound_tie_semantics),
        visual_tie_semantics: metric!(visual_tie_semantics),
        tuplet_notation: metric!(tuplet_notation),
        rest_semantics: metric!(rest_semantics),
        unpitched_display_semantics: metric!(unpitched_display_semantics),
        attribute_event_presence: metric!(attribute_event_presence),
        attribute_event_exactness: metric!(attribute_event_exactness),
        auxiliary_event_presence: metric!(auxiliary_event_presence),
        auxiliary_event_exactness: metric!(auxiliary_event_exactness),
        cursor_move_presence: metric!(cursor_move_presence),
        cursor_move_exactness: metric!(cursor_move_exactness),
        onset_error: error_distribution(&counts.onset_errors),
        duration_error: error_distribution(&counts.duration_errors),
    }
}

fn event_family(event: &crate::musicxml_notation_events::MusicXmlNotationEventV1) -> EventFamily {
    match event.kind {
        MusicXmlNotationEventKindV1::Pitched(_) => EventFamily::Pitched,
        MusicXmlNotationEventKindV1::Rest { .. } => EventFamily::Rest,
        MusicXmlNotationEventKindV1::Unpitched { .. } => EventFamily::Unpitched,
    }
}

fn absolute_difference(
    left: RationalBeat,
    right: RationalBeat,
) -> Result<RationalBeat, MusicXmlNotationEvaluationErrorV1> {
    let ordering = left.compare_exact(right);
    if ordering == Ordering::Less {
        right.checked_sub(left)
    } else {
        left.checked_sub(right)
    }
    .map_err(|_| MusicXmlNotationEvaluationErrorV1::ArithmeticOverflow)
}

fn within_tolerance(
    left: RationalBeat,
    right: RationalBeat,
    tolerance: RationalBeat,
) -> Result<bool, MusicXmlNotationEvaluationErrorV1> {
    Ok(absolute_difference(left, right)?.compare_exact(tolerance) != Ordering::Greater)
}

fn validate_part_graph(
    graph: &MusicXmlPartNotationGraphV1,
) -> Result<(), MusicXmlNotationEvaluationErrorV1> {
    let part_id = &graph.part_id;
    if !matches!(graph.map.status, MeasureMapStatus::Supported) {
        return Err(MusicXmlNotationEvaluationErrorV1::UnsupportedMeasureMap {
            part_id: part_id.clone(),
        });
    }
    if !matches!(
        graph.notation_events.status,
        MusicXmlNotationEventsStatusV1::Supported
    ) {
        return Err(MusicXmlNotationEvaluationErrorV1::UnknownNotationEvents {
            part_id: part_id.clone(),
        });
    }
    if graph.notation_events.part_id != graph.part_id {
        return Err(MusicXmlNotationEvaluationErrorV1::GraphPartIdentityMismatch {
            part_id: part_id.clone(),
        });
    }
    if graph.map.measures.is_empty() {
        return Err(MusicXmlNotationEvaluationErrorV1::UnsupportedMeasureMap {
            part_id: part_id.clone(),
        });
    }

    let mut ordinals = BTreeSet::new();
    let mut total_events = 0usize;
    let mut add_ordinal = |ordinal: usize| {
        total_events += 1;
        if ordinals.insert(ordinal) {
            Ok(())
        } else {
            Err(MusicXmlNotationEvaluationErrorV1::DuplicateSourceOrdinal {
                part_id: part_id.clone(),
                ordinal,
            })
        }
    };

    for event in &graph.notation_events.note_events {
        add_ordinal(event.source_order)?;
        let Some(span) = graph.map.measures.get(event.source_measure_index) else {
            return Err(
                MusicXmlNotationEvaluationErrorV1::SourceMeasureIndexOutOfRange {
                    part_id: part_id.clone(),
                    index: event.source_measure_index,
                },
            );
        };
        let event_end = match event.performed_duration {
            Some(duration) => event
                .score_offset
                .checked_add(duration)
                .map_err(|_| MusicXmlNotationEvaluationErrorV1::ArithmeticOverflow)?,
            None => event.score_offset,
        };
        if event.score_offset.compare_exact(span.start) == Ordering::Less
            || event_end.compare_exact(span.end) == Ordering::Greater
        {
            return Err(
                MusicXmlNotationEvaluationErrorV1::EventOutsideMeasureSpan {
                    part_id: part_id.clone(),
                    source_order: event.source_order,
                },
            );
        }
    }
    for event in &graph.notation_events.attribute_events {
        add_ordinal(event.source_order)?;
        let Some(span) = graph.map.measures.get(event.measure_index) else {
            return Err(
                MusicXmlNotationEvaluationErrorV1::SourceMeasureIndexOutOfRange {
                    part_id: part_id.clone(),
                    index: event.measure_index,
                },
            );
        };
        if event.score_offset.compare_exact(span.start) == Ordering::Less
            || event.score_offset.compare_exact(span.end) == Ordering::Greater
        {
            return Err(
                MusicXmlNotationEvaluationErrorV1::EventOutsideMeasureSpan {
                    part_id: part_id.clone(),
                    source_order: event.source_order,
                },
            );
        }
    }
    for event in &graph.notation_events.auxiliary_events {
        add_ordinal(event.source_order)?;
        let Some(span) = graph.map.measures.get(event.measure_index) else {
            return Err(
                MusicXmlNotationEvaluationErrorV1::SourceMeasureIndexOutOfRange {
                    part_id: part_id.clone(),
                    index: event.measure_index,
                },
            );
        };
        if event.score_offset.compare_exact(span.start) == Ordering::Less
            || event.score_offset.compare_exact(span.end) == Ordering::Greater
        {
            return Err(
                MusicXmlNotationEvaluationErrorV1::EventOutsideMeasureSpan {
                    part_id: part_id.clone(),
                    source_order: event.source_order,
                },
            );
        }
    }
    for event in &graph.notation_events.cursor_moves {
        add_ordinal(event.source_order)?;
        let Some(span) = graph.map.measures.get(event.measure_index) else {
            return Err(
                MusicXmlNotationEvaluationErrorV1::SourceMeasureIndexOutOfRange {
                    part_id: part_id.clone(),
                    index: event.measure_index,
                },
            );
        };
        if event.score_offset_before.compare_exact(span.start) == Ordering::Less
            || event.score_offset_before.compare_exact(span.end) == Ordering::Greater
            || event.score_offset_after.compare_exact(span.start) == Ordering::Less
            || event.score_offset_after.compare_exact(span.end) == Ordering::Greater
        {
            return Err(
                MusicXmlNotationEvaluationErrorV1::EventOutsideMeasureSpan {
                    part_id: part_id.clone(),
                    source_order: event.source_order,
                },
            );
        }
    }
    if total_events > MUSICXML_NOTATION_EVALUATION_MAX_EVENTS_PER_PART_V1 {
        return Err(MusicXmlNotationEvaluationErrorV1::TooManyEvents {
            count: total_events,
            maximum: MUSICXML_NOTATION_EVALUATION_MAX_EVENTS_PER_PART_V1,
        });
    }
    Ok(())
}

/// Build ordered event references grouped by semantic family. Indexes refer to
/// the original array so metric calculations never depend on a copied value.
fn positioned_sources(
    events: &[crate::musicxml_notation_events::MusicXmlNotationEventV1],
) -> Vec<PositionedSource> {
    events
        .iter()
        .enumerate()
        .map(|(index, event)| PositionedSource {
            index,
            offset: event.score_offset,
            source_order: event.source_order,
            key: event_family(event).key().into(),
        })
        .collect()
}

fn ordered_matches(
    gold: Vec<PositionedSource>,
    prediction: Vec<PositionedSource>,
    tolerance: RationalBeat,
) -> Result<Vec<(usize, usize)>, MusicXmlNotationEvaluationErrorV1> {
    let mut gold_groups: BTreeMap<String, Vec<PositionedSource>> = BTreeMap::new();
    let mut prediction_groups: BTreeMap<String, Vec<PositionedSource>> = BTreeMap::new();
    for event in gold {
        gold_groups.entry(event.key.clone()).or_default().push(event);
    }
    for event in prediction {
        prediction_groups
            .entry(event.key.clone())
            .or_default()
            .push(event);
    }

    let keys: BTreeSet<_> = gold_groups
        .keys()
        .chain(prediction_groups.keys())
        .cloned()
        .collect();
    let mut matches = Vec::new();
    for key in keys {
        let mut gold_group = gold_groups.remove(&key).unwrap_or_default();
        let mut prediction_group = prediction_groups.remove(&key).unwrap_or_default();
        let sort = |left: &PositionedSource, right: &PositionedSource| {
            left.offset
                .compare_exact(right.offset)
                .then_with(|| left.source_order.cmp(&right.source_order))
                .then_with(|| left.index.cmp(&right.index))
        };
        gold_group.sort_by(sort);
        prediction_group.sort_by(sort);

        let mut gold_index = 0usize;
        let mut prediction_index = 0usize;
        while gold_index < gold_group.len() && prediction_index < prediction_group.len() {
            let gold_event = &gold_group[gold_index];
            let prediction_event = &prediction_group[prediction_index];
            let difference =
                absolute_difference(gold_event.offset, prediction_event.offset)?;
            if difference.compare_exact(tolerance) != Ordering::Greater {
                matches.push((gold_event.index, prediction_event.index));
                gold_index += 1;
                prediction_index += 1;
            } else if gold_event.offset.compare_exact(prediction_event.offset)
                == Ordering::Less
            {
                // The earlier reference event cannot match this or any later
                // prediction event; discard it to maximize remaining matches.
                gold_index += 1;
            } else {
                prediction_index += 1;
            }
        }
    }
    Ok(matches)
}

fn attribute_positions(
    events: &[crate::musicxml_notation_events::MusicXmlAttributeEventV1],
) -> Vec<PositionedSource> {
    events
        .iter()
        .enumerate()
        .map(|(index, event)| PositionedSource {
            index,
            offset: event.score_offset,
            source_order: event.source_order,
            key: event.source.element.clone(),
        })
        .collect()
}

fn auxiliary_positions(
    events: &[crate::musicxml_notation_events::MusicXmlAuxiliaryEventV1],
) -> Vec<PositionedSource> {
    events
        .iter()
        .enumerate()
        .map(|(index, event)| PositionedSource {
            index,
            offset: event.score_offset,
            source_order: event.source_order,
            key: event.source.element.clone(),
        })
        .collect()
}

fn cursor_positions(
    events: &[crate::musicxml_notation_events::MusicXmlCursorMoveV1],
) -> Vec<PositionedSource> {
    events
        .iter()
        .enumerate()
        .map(|(index, event)| PositionedSource {
            index,
            offset: event.score_offset_before,
            source_order: event.source_order,
            key: event.direction.clone(),
        })
        .collect()
}

fn exact_option_duration(
    gold: Option<RationalBeat>,
    prediction: Option<RationalBeat>,
    tolerance: RationalBeat,
) -> Result<bool, MusicXmlNotationEvaluationErrorV1> {
    match (gold, prediction) {
        (Some(gold), Some(prediction)) => within_tolerance(gold, prediction, tolerance),
        (None, None) => Ok(true),
        _ => Ok(false),
    }
}

fn evaluate_part(
    gold: &MusicXmlPartNotationEventsV1,
    prediction: &MusicXmlPartNotationEventsV1,
    part_id: &str,
    config: &MusicXmlNotationEvaluationConfigV1,
) -> Result<PartCounts, MusicXmlNotationEvaluationErrorV1> {
    let gold_notes = &gold.note_events;
    let prediction_notes = &prediction.note_events;
    let note_matches = ordered_matches(
        positioned_sources(gold_notes),
        positioned_sources(prediction_notes),
        config.onset_tolerance,
    )?;

    let gold_pitched: Vec<_> = gold_notes
        .iter()
        .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Pitched(_)))
        .collect();
    let prediction_pitched: Vec<_> = prediction_notes
        .iter()
        .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Pitched(_)))
        .collect();
    let gold_rests: Vec<_> = gold_notes
        .iter()
        .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Rest { .. }))
        .collect();
    let prediction_rests: Vec<_> = prediction_notes
        .iter()
        .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Rest { .. }))
        .collect();
    let gold_unpitched: Vec<_> = gold_notes
        .iter()
        .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Unpitched { .. }))
        .collect();
    let prediction_unpitched: Vec<_> = prediction_notes
        .iter()
        .filter(|event| matches!(event.kind, MusicXmlNotationEventKindV1::Unpitched { .. }))
        .collect();

    let mut counts = PartCounts {
        part_id: Some(part_id.to_string()),
        gold_note_event_count: gold_notes.len() as u64,
        prediction_note_event_count: prediction_notes.len() as u64,
        gold_attribute_event_count: gold.attribute_events.len() as u64,
        prediction_attribute_event_count: prediction.attribute_events.len() as u64,
        gold_auxiliary_event_count: gold.auxiliary_events.len() as u64,
        prediction_auxiliary_event_count: prediction.auxiliary_events.len() as u64,
        gold_cursor_move_count: gold.cursor_moves.len() as u64,
        prediction_cursor_move_count: prediction.cursor_moves.len() as u64,
        event_presence: metric_counts(gold_notes.len(), prediction_notes.len(), note_matches.len()),
        pitched_event_presence: MetricCounts::default(),
        rest_event_presence: MetricCounts::default(),
        unpitched_event_presence: MetricCounts::default(),
        ..PartCounts::default()
    };

    let count_family_matches = |family: EventFamily| {
        note_matches
            .iter()
            .filter(|(gold_index, prediction_index)| {
                event_family(&gold_notes[*gold_index]) == family
                    && event_family(&prediction_notes[*prediction_index]) == family
            })
            .count()
    };
    counts.pitched_event_presence =
        metric_counts(gold_pitched.len(), prediction_pitched.len(), count_family_matches(EventFamily::Pitched));
    counts.rest_event_presence =
        metric_counts(gold_rests.len(), prediction_rests.len(), count_family_matches(EventFamily::Rest));
    counts.unpitched_event_presence = metric_counts(
        gold_unpitched.len(),
        prediction_unpitched.len(),
        count_family_matches(EventFamily::Unpitched),
    );

    let matched_pitched = note_matches
        .iter()
        .filter(|(g, p)| {
            matches!(gold_notes[*g].kind, MusicXmlNotationEventKindV1::Pitched(_))
                && matches!(prediction_notes[*p].kind, MusicXmlNotationEventKindV1::Pitched(_))
        })
        .copied()
        .collect::<Vec<_>>();
    let pitch_correct = matched_pitched
        .iter()
        .filter(|(g, p)| gold_notes[*g].kind == prediction_notes[*p].kind)
        .count();
    counts.pitch_spelling = metric_counts(gold_pitched.len(), prediction_pitched.len(), pitch_correct);

    let performed_gold = gold_notes.iter().filter(|event| event.performed_duration.is_some()).count();
    let performed_prediction = prediction_notes.iter().filter(|event| event.performed_duration.is_some()).count();
    let mut duration_correct = 0usize;
    for (g, p) in &note_matches {
        if let (Some(gold_duration), Some(prediction_duration)) =
            (gold_notes[*g].performed_duration, prediction_notes[*p].performed_duration)
        {
            let difference = absolute_difference(gold_duration, prediction_duration)?;
            counts.duration_errors.push(difference.as_beats());
            if difference.compare_exact(config.duration_tolerance) != Ordering::Greater {
                duration_correct += 1;
            }
        }
    }
    counts.performed_duration =
        metric_counts(performed_gold, performed_prediction, duration_correct);

    let has_written_value = |event: &crate::musicxml_notation_events::MusicXmlNotationEventV1| {
        event.written_type.is_some() || event.dot_count > 0 || event.time_modification.is_some()
    };
    let gold_written = gold_notes.iter().filter(|event| has_written_value(event)).count();
    let prediction_written = prediction_notes.iter().filter(|event| has_written_value(event)).count();
    let written_correct = note_matches
        .iter()
        .filter(|(g, p)| {
            has_written_value(&gold_notes[*g])
                && has_written_value(&prediction_notes[*p])
                && gold_notes[*g].written_type == prediction_notes[*p].written_type
                && gold_notes[*g].dot_count == prediction_notes[*p].dot_count
                && gold_notes[*g].time_modification == prediction_notes[*p].time_modification
        })
        .count();
    counts.written_value = metric_counts(gold_written, prediction_written, written_correct);

    let has_grace_cue_chord = |event: &crate::musicxml_notation_events::MusicXmlNotationEventV1| {
        event.is_grace || event.is_cue || event.is_chord_member
    };
    let gold_grace_cue_chord = gold_notes.iter().filter(|event| has_grace_cue_chord(event)).count();
    let prediction_grace_cue_chord = prediction_notes
        .iter()
        .filter(|event| has_grace_cue_chord(event))
        .count();
    let grace_cue_chord_correct = note_matches
        .iter()
        .filter(|(g, p)| {
            has_grace_cue_chord(&gold_notes[*g])
                && has_grace_cue_chord(&prediction_notes[*p])
                && gold_notes[*g].is_grace == prediction_notes[*p].is_grace
                && gold_notes[*g].is_cue == prediction_notes[*p].is_cue
                && gold_notes[*g].is_chord_member == prediction_notes[*p].is_chord_member
                && gold_notes[*g].chord_anchor_source_order.is_some()
                    == prediction_notes[*p].chord_anchor_source_order.is_some()
        })
        .count();
    counts.grace_cue_chord_semantics = metric_counts(
        gold_grace_cue_chord,
        prediction_grace_cue_chord,
        grace_cue_chord_correct,
    );

    let assigned_count = |events: &[&crate::musicxml_notation_events::MusicXmlNotationEventV1],
                          field: fn(&crate::musicxml_notation_events::MusicXmlNotationEventV1) -> bool| {
        events.iter().filter(|event| field(event)).count()
    };
    let gold_voice_count = assigned_count(&gold_notes.iter().collect::<Vec<_>>(), |event| event.voice.is_some());
    let prediction_voice_count = assigned_count(
        &prediction_notes.iter().collect::<Vec<_>>(),
        |event| event.voice.is_some(),
    );
    let voice_correct = note_matches
        .iter()
        .filter(|(g, p)| {
            gold_notes[*g].voice.is_some()
                && prediction_notes[*p].voice.is_some()
                && gold_notes[*g].voice == prediction_notes[*p].voice
        })
        .count();
    counts.voice_identity = metric_counts(gold_voice_count, prediction_voice_count, voice_correct);

    let gold_staff_count = gold_notes.iter().filter(|event| event.staff.is_some()).count();
    let prediction_staff_count = prediction_notes.iter().filter(|event| event.staff.is_some()).count();
    let staff_correct = note_matches
        .iter()
        .filter(|(g, p)| {
            gold_notes[*g].staff.is_some()
                && prediction_notes[*p].staff.is_some()
                && gold_notes[*g].staff == prediction_notes[*p].staff
        })
        .count();
    counts.staff_identity = metric_counts(gold_staff_count, prediction_staff_count, staff_correct);

    let gold_instrument_count = gold_notes.iter().filter(|event| !event.instrument_ids.is_empty()).count();
    let prediction_instrument_count = prediction_notes
        .iter()
        .filter(|event| !event.instrument_ids.is_empty())
        .count();
    let instrument_correct = note_matches
        .iter()
        .filter(|(g, p)| {
            !gold_notes[*g].instrument_ids.is_empty()
                && !prediction_notes[*p].instrument_ids.is_empty()
                && gold_notes[*g].instrument_ids == prediction_notes[*p].instrument_ids
        })
        .count();
    counts.instrument_identity =
        metric_counts(gold_instrument_count, prediction_instrument_count, instrument_correct);

    let tie_metric = |gold_tied: usize,
                      prediction_tied: usize,
                      exact: usize| metric_counts(gold_tied, prediction_tied, exact);
    let gold_sound_tie_events = gold_notes.iter().filter(|event| !event.sound_ties.is_empty()).count();
    let prediction_sound_tie_events = prediction_notes
        .iter()
        .filter(|event| !event.sound_ties.is_empty())
        .count();
    let sound_tie_correct = note_matches
        .iter()
        .filter(|(g, p)| {
            !gold_notes[*g].sound_ties.is_empty()
                && !prediction_notes[*p].sound_ties.is_empty()
                && gold_notes[*g].sound_ties == prediction_notes[*p].sound_ties
        })
        .count();
    counts.sound_tie_semantics =
        tie_metric(gold_sound_tie_events, prediction_sound_tie_events, sound_tie_correct);

    let gold_visual_tie_events = gold_notes.iter().filter(|event| !event.visual_ties.is_empty()).count();
    let prediction_visual_tie_events = prediction_notes
        .iter()
        .filter(|event| !event.visual_ties.is_empty())
        .count();
    let visual_tie_correct = note_matches
        .iter()
        .filter(|(g, p)| {
            !gold_notes[*g].visual_ties.is_empty()
                && !prediction_notes[*p].visual_ties.is_empty()
                && gold_notes[*g].visual_ties == prediction_notes[*p].visual_ties
        })
        .count();
    counts.visual_tie_semantics =
        tie_metric(gold_visual_tie_events, prediction_visual_tie_events, visual_tie_correct);

    let gold_tuplet_events = gold_notes.iter().filter(|event| !event.tuplet_notations.is_empty()).count();
    let prediction_tuplet_events = prediction_notes
        .iter()
        .filter(|event| !event.tuplet_notations.is_empty())
        .count();
    let tuplet_correct = note_matches
        .iter()
        .filter(|(g, p)| {
            !gold_notes[*g].tuplet_notations.is_empty()
                && !prediction_notes[*p].tuplet_notations.is_empty()
                && gold_notes[*g].tuplet_notations == prediction_notes[*p].tuplet_notations
        })
        .count();
    counts.tuplet_notation = metric_counts(gold_tuplet_events, prediction_tuplet_events, tuplet_correct);

    let rest_exact = note_matches
        .iter()
        .filter(|(g, p)| match (&gold_notes[*g].kind, &prediction_notes[*p].kind) {
            (
                MusicXmlNotationEventKindV1::Rest {
                    measure: gold_measure,
                    display: gold_display,
                },
                MusicXmlNotationEventKindV1::Rest {
                    measure: prediction_measure,
                    display: prediction_display,
                },
            ) => gold_measure == prediction_measure && gold_display == prediction_display,
            _ => false,
        })
        .count();
    counts.rest_semantics = metric_counts(gold_rests.len(), prediction_rests.len(), rest_exact);

    let unpitched_exact = note_matches
        .iter()
        .filter(|(g, p)| match (&gold_notes[*g].kind, &prediction_notes[*p].kind) {
            (
                MusicXmlNotationEventKindV1::Unpitched { display: gold_display },
                MusicXmlNotationEventKindV1::Unpitched {
                    display: prediction_display,
                },
            ) => gold_display == prediction_display,
            _ => false,
        })
        .count();
    counts.unpitched_display_semantics =
        metric_counts(gold_unpitched.len(), prediction_unpitched.len(), unpitched_exact);

    for (g, p) in &note_matches {
        counts.onset_errors.push(
            absolute_difference(gold_notes[*g].score_offset, prediction_notes[*p].score_offset)?
                .as_beats(),
        );
    }

    let gold_attribute_events = &gold.attribute_events;
    let prediction_attribute_events = &prediction.attribute_events;
    let attribute_matches = ordered_matches(
        attribute_positions(gold_attribute_events),
        attribute_positions(prediction_attribute_events),
        config.onset_tolerance,
    )?;
    counts.attribute_event_presence = metric_counts(
        gold_attribute_events.len(),
        prediction_attribute_events.len(),
        attribute_matches.len(),
    );
    let attribute_exact = attribute_matches
        .iter()
        .filter(|(g, p)| gold_attribute_events[*g].source == prediction_attribute_events[*p].source)
        .count();
    counts.attribute_event_exactness = metric_counts(
        gold_attribute_events.len(),
        prediction_attribute_events.len(),
        attribute_exact,
    );

    let gold_auxiliary_events = &gold.auxiliary_events;
    let prediction_auxiliary_events = &prediction.auxiliary_events;
    let auxiliary_matches = ordered_matches(
        auxiliary_positions(gold_auxiliary_events),
        auxiliary_positions(prediction_auxiliary_events),
        config.onset_tolerance,
    )?;
    counts.auxiliary_event_presence = metric_counts(
        gold_auxiliary_events.len(),
        prediction_auxiliary_events.len(),
        auxiliary_matches.len(),
    );
    let auxiliary_exact = auxiliary_matches
        .iter()
        .filter(|(g, p)| gold_auxiliary_events[*g].source == prediction_auxiliary_events[*p].source)
        .count();
    counts.auxiliary_event_exactness = metric_counts(
        gold_auxiliary_events.len(),
        prediction_auxiliary_events.len(),
        auxiliary_exact,
    );

    let gold_cursor_moves = &gold.cursor_moves;
    let prediction_cursor_moves = &prediction.cursor_moves;
    let cursor_matches = ordered_matches(
        cursor_positions(gold_cursor_moves),
        cursor_positions(prediction_cursor_moves),
        config.onset_tolerance,
    )?;
    counts.cursor_move_presence = metric_counts(
        gold_cursor_moves.len(),
        prediction_cursor_moves.len(),
        cursor_matches.len(),
    );
    let cursor_exact = cursor_matches
        .iter()
        .filter(|(g, p)| {
            gold_cursor_moves[*g].direction == prediction_cursor_moves[*p].direction
                && gold_cursor_moves[*g].duration == prediction_cursor_moves[*p].duration
                && gold_cursor_moves[*g].voice == prediction_cursor_moves[*p].voice
                && gold_cursor_moves[*g].staff == prediction_cursor_moves[*p].staff
                && gold_cursor_moves[*g].source == prediction_cursor_moves[*p].source
        })
        .count();
    counts.cursor_move_exactness = metric_counts(
        gold_cursor_moves.len(),
        prediction_cursor_moves.len(),
        cursor_exact,
    );
    Ok(counts)
}

fn graph_map<'a>(
    graphs: &'a [MusicXmlPartNotationGraphV1],
) -> Result<BTreeMap<&'a str, &'a MusicXmlPartNotationGraphV1>, MusicXmlNotationEvaluationErrorV1> {
    let mut map = BTreeMap::new();
    for graph in graphs {
        if map.insert(graph.part_id.as_str(), graph).is_some() {
            return Err(MusicXmlNotationEvaluationErrorV1::DuplicatePartId {
                part_id: graph.part_id.clone(),
            });
        }
    }
    Ok(map)
}

/// Evaluate two supported MusicXML part-graph sets in one exact notated clock.
/// The evaluator will not infer missing part correspondence or repair Unknown
/// source structure.
pub fn evaluate_musicxml_notation_graphs(
    gold_graphs: &[MusicXmlPartNotationGraphV1],
    prediction_graphs: &[MusicXmlPartNotationGraphV1],
    config: MusicXmlNotationEvaluationConfigV1,
) -> Result<MusicXmlNotationEvaluationReportV1, MusicXmlNotationEvaluationErrorV1> {
    if gold_graphs.is_empty() {
        return Err(MusicXmlNotationEvaluationErrorV1::EmptyGoldGraph);
    }
    if prediction_graphs.is_empty() {
        return Err(MusicXmlNotationEvaluationErrorV1::EmptyPredictionGraph);
    }
    let part_count = gold_graphs.len().max(prediction_graphs.len());
    if part_count > MUSICXML_NOTATION_EVALUATION_MAX_PARTS_V1 {
        return Err(MusicXmlNotationEvaluationErrorV1::TooManyParts {
            count: part_count,
            maximum: MUSICXML_NOTATION_EVALUATION_MAX_PARTS_V1,
        });
    }
    let gold_map = graph_map(gold_graphs)?;
    let prediction_map = graph_map(prediction_graphs)?;
    if gold_map.keys().ne(prediction_map.keys()) {
        return Err(MusicXmlNotationEvaluationErrorV1::PartSetMismatch);
    }

    let score_clock_id = gold_graphs[0].map.score_clock_id.clone();
    let mut part_scorecards = Vec::with_capacity(part_count);
    let mut aggregate_counts = PartCounts::default();
    for (part_id, gold) in &gold_map {
        let prediction = prediction_map[part_id];
        validate_part_graph(gold)?;
        validate_part_graph(prediction)?;
        if gold.map.score_clock_id != prediction.map.score_clock_id
            || gold.map.score_clock_id != score_clock_id
        {
            return Err(MusicXmlNotationEvaluationErrorV1::ScoreClockMismatch);
        }
        if gold.map.canonicalization_version != prediction.map.canonicalization_version
            || gold.notation_events.canonicalization_version
                != prediction.notation_events.canonicalization_version
            || gold.notation_events.schema != prediction.notation_events.schema
        {
            return Err(MusicXmlNotationEvaluationErrorV1::CanonicalizationMismatch);
        }
        let part_counts = evaluate_part(
            &gold.notation_events,
            &prediction.notation_events,
            part_id,
            &config,
        )?;
        combine_part_counts(&mut aggregate_counts, &part_counts);
        part_scorecards.push(scorecard_from_counts(&part_counts));
    }
    aggregate_counts.part_id = None;

    let gold_graph_sha256 = canonical_json_sha256(gold_graphs)
        .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?;
    let prediction_graph_sha256 = canonical_json_sha256(prediction_graphs)
        .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?;
    let mut report = MusicXmlNotationEvaluationReportV1 {
        schema: MUSICXML_NOTATION_EVALUATION_SCHEMA_V1.into(),
        evaluator_version: MUSICXML_NOTATION_EVALUATOR_VERSION_V1.into(),
        config,
        score_clock_id,
        gold_graph_sha256,
        prediction_graph_sha256,
        gold_part_count: gold_graphs.len() as u64,
        prediction_part_count: prediction_graphs.len() as u64,
        part_scorecards,
        aggregate: scorecard_from_counts(&aggregate_counts),
        report_sha256: String::new(),
    };
    report.report_sha256 = musicxml_notation_evaluation_commitment(&report)
        .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?;
    Ok(report)
}

#[derive(Serialize)]
struct ReportCommitment<'a> {
    schema: &'a str,
    evaluator_version: &'a str,
    config: &'a MusicXmlNotationEvaluationConfigV1,
    score_clock_id: &'a str,
    gold_graph_sha256: &'a str,
    prediction_graph_sha256: &'a str,
    gold_part_count: u64,
    prediction_part_count: u64,
    part_scorecards: &'a [MusicXmlNotationPartScorecardV1],
    aggregate: &'a MusicXmlNotationPartScorecardV1,
}

pub fn musicxml_notation_evaluation_commitment(
    report: &MusicXmlNotationEvaluationReportV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ReportCommitment {
        schema: &report.schema,
        evaluator_version: &report.evaluator_version,
        config: &report.config,
        score_clock_id: &report.score_clock_id,
        gold_graph_sha256: &report.gold_graph_sha256,
        prediction_graph_sha256: &report.prediction_graph_sha256,
        gold_part_count: report.gold_part_count,
        prediction_part_count: report.prediction_part_count,
        part_scorecards: &report.part_scorecards,
        aggregate: &report.aggregate,
    })
}

pub fn validate_musicxml_notation_evaluation_report(
    gold_graphs: &[MusicXmlPartNotationGraphV1],
    prediction_graphs: &[MusicXmlPartNotationGraphV1],
    report: &MusicXmlNotationEvaluationReportV1,
) -> Result<bool, MusicXmlNotationEvaluationErrorV1> {
    let replayed = evaluate_musicxml_notation_graphs(
        gold_graphs,
        prediction_graphs,
        report.config.clone(),
    )?;
    Ok(replayed == *report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::musicxml_barline_adapter::parse_musicxml_part_notation_graph;

    fn graph(xml: &str, source_id: &str, score_clock_id: &str) -> Vec<MusicXmlPartNotationGraphV1> {
        parse_musicxml_part_notation_graph(xml.as_bytes(), source_id, score_clock_id).unwrap()
    }

    fn config() -> MusicXmlNotationEvaluationConfigV1 {
        MusicXmlNotationEvaluationConfigV1 {
            onset_tolerance: RationalBeat::new(1, 32).unwrap(),
            duration_tolerance: RationalBeat::new(1, 32).unwrap(),
        }
    }

    fn note_scorecard(xml: &str) -> Vec<MusicXmlPartNotationGraphV1> {
        graph(
            xml,
            "sha256:musicxml-eval-fixture",
            "musicxml-eval-clock",
        )
    }

    #[test]
    fn identical_graphs_score_perfect_independent_dimensions_and_replay() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice><staff>1</staff><type>eighth</type><instrument id="P1-I1"/></note>
            <note><rest/><duration>8</duration><voice>1</voice><staff>1</staff></note>
            <note><unpitched><display-step>F</display-step><display-octave>4</display-octave></unpitched><duration>8</duration><voice>2</voice><staff>2</staff></note>
        </measure></part></score-partwise>"#;
        let gold = note_scorecard(xml);
        let prediction = note_scorecard(xml);
        let report = evaluate_musicxml_notation_graphs(&gold, &prediction, config()).unwrap();
        let aggregate = &report.aggregate;
        assert_eq!(aggregate.event_presence.f1, Some(1.0));
        assert_eq!(aggregate.pitch_spelling.f1, Some(1.0));
        assert_eq!(aggregate.performed_duration.f1, Some(1.0));
        assert_eq!(aggregate.rest_semantics.f1, Some(1.0));
        assert_eq!(aggregate.unpitched_display_semantics.f1, Some(1.0));
        assert_eq!(aggregate.voice_identity.f1, Some(1.0));
        assert_eq!(aggregate.staff_identity.f1, Some(1.0));
        assert_eq!(aggregate.instrument_identity.f1, Some(1.0));
        assert_eq!(
            report.report_sha256.len(),
            64
        );
        assert!(validate_musicxml_notation_evaluation_report(
            &gold,
            &prediction,
            &report
        ).unwrap());
    }

    #[test]
    fn wrong_written_pitch_is_not_hidden_by_event_presence() {
        let gold_xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let prediction_xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>G</step><alter>0</alter><octave>4</octave></pitch><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let gold = graph(gold_xml, "sha256:gold", "same-clock");
        let prediction = graph(prediction_xml, "sha256:prediction", "same-clock");
        let report = evaluate_musicxml_notation_graphs(&gold, &prediction, config()).unwrap();
        assert_eq!(report.aggregate.event_presence.f1, Some(1.0));
        assert_eq!(report.aggregate.pitch_spelling.f1, Some(0.0));
    }

    #[test]
    fn duration_error_is_separate_from_pitch_and_event_presence() {
        let gold_xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>16</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let prediction_xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>16</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>15</duration></note>
            <note><rest/><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let gold = graph(gold_xml, "sha256:gold-duration", "same-clock");
        let prediction = graph(prediction_xml, "sha256:pred-duration", "same-clock");
        let report = evaluate_musicxml_notation_graphs(&gold, &prediction, config()).unwrap();
        assert_eq!(report.aggregate.event_presence.true_positives, 1);
        assert_eq!(report.aggregate.event_presence.false_positives, 1);
        assert_eq!(report.aggregate.performed_duration.f1, Some(0.0));
    }

    #[test]
    fn one_to_one_onset_window_matching_is_maximum_cardinality() {
        let gold = vec![
            PositionedSource {
                index: 0,
                offset: RationalBeat::new(0, 1).unwrap(),
                source_order: 0,
                key: "pitched".into(),
            },
            PositionedSource {
                index: 1,
                offset: RationalBeat::new(1, 4).unwrap(),
                source_order: 1,
                key: "pitched".into(),
            },
        ];
        let prediction = vec![
            PositionedSource {
                index: 0,
                offset: RationalBeat::new(1, 8).unwrap(),
                source_order: 0,
                key: "pitched".into(),
            },
            PositionedSource {
                index: 1,
                offset: RationalBeat::new(3, 8).unwrap(),
                source_order: 1,
                key: "pitched".into(),
            },
        ];
        let matches = ordered_matches(
            gold,
            prediction,
            RationalBeat::new(1, 8).unwrap(),
        ).unwrap();
        assert_eq!(matches, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn tolerance_edge_is_inclusive_and_one_rational_unit_outside_is_not() {
        let left = RationalBeat::new(0, 1).unwrap();
        let edge = RationalBeat::new(1, 32).unwrap();
        let outside = RationalBeat::new(2, 32).unwrap();
        let tolerance = RationalBeat::new(1, 32).unwrap();
        assert!(within_tolerance(left, edge, tolerance).unwrap());
        assert!(!within_tolerance(left, outside, tolerance).unwrap());
    }

    #[test]
    fn changed_tuplet_and_visual_tie_semantics_only_reduce_their_dimensions() {
        let gold_xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>12</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><tie type="start"/><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tied type="start"/><tuplet type="start" number="1"/></notations></note>
        </measure></part></score-partwise>"#;
        let prediction_xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>12</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><tie type="start"/><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><notations><tied type="stop"/><tuplet type="stop" number="1"/></notations></note>
        </measure></part></score-partwise>"#;
        let gold = graph(gold_xml, "sha256:tie-gold", "same-clock");
        let prediction = graph(prediction_xml, "sha256:tie-pred", "same-clock");
        let report = evaluate_musicxml_notation_graphs(&gold, &prediction, config()).unwrap();
        assert_eq!(report.aggregate.event_presence.f1, Some(1.0));
        assert_eq!(report.aggregate.sound_tie_semantics.f1, Some(1.0));
        assert_eq!(report.aggregate.visual_tie_semantics.f1, Some(0.0));
        assert_eq!(report.aggregate.tuplet_notation.f1, Some(0.0));
    }

    #[test]
    fn score_clock_part_and_unknown_status_mismatches_fail_closed() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><rest/><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let gold = graph(xml, "sha256:gold", "clock-a");
        let wrong_clock = graph(xml, "sha256:pred", "clock-b");
        assert_eq!(
            evaluate_musicxml_notation_graphs(&gold, &wrong_clock, config()),
            Err(MusicXmlNotationEvaluationErrorV1::ScoreClockMismatch)
        );
        let wrong_part_xml = xml.replace("P1", "P2");
        let wrong_part = graph(&wrong_part_xml, "sha256:pred", "clock-a");
        assert_eq!(
            evaluate_musicxml_notation_graphs(&gold, &wrong_part, config()),
            Err(MusicXmlNotationEvaluationErrorV1::PartSetMismatch)
        );
        let mut unknown = gold.clone();
        unknown[0].notation_events.status =
            MusicXmlNotationEventsStatusV1::Unknown {
                reason: "test unknown".into(),
            };
        assert!(matches!(
            evaluate_musicxml_notation_graphs(&unknown, &gold, config()),
            Err(MusicXmlNotationEvaluationErrorV1::UnknownNotationEvents { .. })
        ));
    }

    #[test]
    fn report_replay_rejects_tampered_counts_or_inputs() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let gold = graph(xml, "sha256:gold-replay", "same-clock");
        let prediction = graph(xml, "sha256:pred-replay", "same-clock");
        let mut report = evaluate_musicxml_notation_graphs(&gold, &prediction, config()).unwrap();
        report.aggregate.event_presence.false_positives += 1;
        assert!(!validate_musicxml_notation_evaluation_report(
            &gold,
            &prediction,
            &report
        ).unwrap());
        let original = evaluate_musicxml_notation_graphs(&gold, &prediction, config()).unwrap();
        let changed = graph(
            &xml.replace("<step>C</step>", "<step>D</step>"),
            "sha256:pred-replay",
            "same-clock",
        );
        assert!(!validate_musicxml_notation_evaluation_report(
            &gold,
            &changed,
            &original
        ).unwrap());
    }

    #[test]
    fn attribute_auxiliary_and_cursor_dimensions_are_reported_independently() {
        let gold_xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
            <forward><duration>4</duration><voice>2</voice><staff>2</staff></forward>
            <direction><direction-type><dynamics><f/></dynamics></direction-type><sound tempo="80"/></direction>
            <note><rest/><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let prediction_xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>F</sign><line>4</line></clef></attributes>
            <forward><duration>4</duration><voice>2</voice><staff>2</staff></forward>
            <direction><direction-type><dynamics><p/></dynamics></direction-type><sound tempo="80"/></direction>
            <note><rest/><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let gold = graph(gold_xml, "sha256:aux-gold", "same-clock");
        let prediction = graph(prediction_xml, "sha256:aux-pred", "same-clock");
        let report = evaluate_musicxml_notation_graphs(&gold, &prediction, config()).unwrap();
        assert_eq!(report.aggregate.attribute_event_presence.f1, Some(1.0));
        assert_eq!(report.aggregate.attribute_event_exactness.f1, Some(0.0));
        assert_eq!(report.aggregate.auxiliary_event_presence.f1, Some(1.0));
        assert_eq!(report.aggregate.auxiliary_event_exactness.f1, Some(0.0));
        assert_eq!(report.aggregate.cursor_move_presence.f1, Some(1.0));
        assert_eq!(report.aggregate.cursor_move_exactness.f1, Some(1.0));
    }
}
