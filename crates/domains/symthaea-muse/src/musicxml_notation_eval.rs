// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded, multidimensional evaluation of source-native MusicXML score events.
//!
//! This is deliberately not a note-only MIDI evaluator and does not emit one
//! weighted "score quality" scalar. It compares explicit MusicXML notation
//! graphs on a shared score clock and reports note/rest, spelling, duration,
//! written-value, voice/staff, tie, tuplet, attribute, auxiliary and cursor
//! dimensions independently.

use crate::barline_eval::{MeasureMapStatus, RationalBeat};
use crate::musicxml_barline_adapter::MUSICXML_BARLINE_CANONICALIZATION_V1;
use crate::evidence_digest::canonical_json_sha256;
use crate::musicxml_barline_adapter::MusicXmlPartNotationGraphV1;
use crate::musicxml_notation_events::{
    MUSICXML_NOTATION_EVENTS_CANONICALIZATION_V1,
    MUSICXML_NOTATION_EVENTS_SCHEMA_V1,
    MusicXmlNotationEventKindV1,
    MusicXmlNotationEventsStatusV1,
    MusicXmlPartNotationEventsV1,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;

pub const MUSICXML_NOTATION_EVALUATION_SCHEMA_V1: &str =
    "muse-musicxml-notation-evaluation-v1";
pub const MUSICXML_NOTATION_EVALUATION_ALGORITHM_V1: &str =
    "exact-quarter-beat-kind-onset-one-to-one-augmenting-path-v1";
pub const MUSICXML_NOTATION_EVALUATION_MAX_EVENTS_V1: usize = 10_000;
pub const MUSICXML_NOTATION_EVALUATION_MAX_CANDIDATE_PAIRS_V1: usize = 2_000_000;
pub const MUSICXML_NOTATION_EVALUATION_MAX_MATCHING_SCANS_V1: usize = 10_000_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlNotationEvaluationConfigV1 {
    /// Exact quarter-note-beat tolerance for note/auxiliary/attribute positions.
    pub position_tolerance: RationalBeat,
    /// Exact quarter-note-beat tolerance used for duration correctness.
    pub duration_tolerance: RationalBeat,
    pub maximum_events_per_part: usize,
    pub maximum_candidate_pairs: usize,
    pub maximum_matching_scans: usize,
}

impl Default for MusicXmlNotationEvaluationConfigV1 {
    fn default() -> Self {
        Self {
            position_tolerance: RationalBeat::new(1, 16).expect("positive rational"),
            duration_tolerance: RationalBeat::new(1, 16).expect("positive rational"),
            maximum_events_per_part: MUSICXML_NOTATION_EVALUATION_MAX_EVENTS_V1,
            maximum_candidate_pairs: MUSICXML_NOTATION_EVALUATION_MAX_CANDIDATE_PAIRS_V1,
            maximum_matching_scans: MUSICXML_NOTATION_EVALUATION_MAX_MATCHING_SCANS_V1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationMetricV1 {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationAccuracyV1 {
    pub sample_count: usize,
    pub correct_count: usize,
    pub accuracy: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationErrorDistributionV1 {
    pub sample_count: usize,
    pub mean_absolute_quarter_beats: Option<f64>,
    pub median_absolute_quarter_beats: Option<f64>,
    pub p95_absolute_quarter_beats: Option<f64>,
    pub p99_absolute_quarter_beats: Option<f64>,
    pub within_tolerance_count: usize,
    pub within_tolerance_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationEvaluationReportV1 {
    pub schema: String,
    pub algorithm: String,
    pub config: MusicXmlNotationEvaluationConfigV1,
    pub part_id: String,
    pub score_clock_id: String,
    pub canonicalization_version: String,
    pub reference_graph_sha256: String,
    pub prediction_graph_sha256: String,
    pub reference_note_event_count: usize,
    pub prediction_note_event_count: usize,
    pub note_event_presence: MusicXmlNotationMetricV1,
    pub note_onset_error: MusicXmlNotationErrorDistributionV1,
    pub written_pitch_spelling: MusicXmlNotationAccuracyV1,
    pub rest_semantics: MusicXmlNotationAccuracyV1,
    pub unpitched_display: MusicXmlNotationAccuracyV1,
    pub duration_presence: MusicXmlNotationAccuracyV1,
    pub duration_error: MusicXmlNotationErrorDistributionV1,
    pub written_type_and_dots: MusicXmlNotationAccuracyV1,
    pub voice_assignment: MusicXmlNotationAccuracyV1,
    pub voice_reference_unassigned_count: usize,
    pub voice_unexpected_assignment_count: usize,
    pub staff_assignment: MusicXmlNotationAccuracyV1,
    pub staff_reference_unassigned_count: usize,
    pub staff_unexpected_assignment_count: usize,
    pub instrument_assignment: MusicXmlNotationAccuracyV1,
    pub instrument_reference_unassigned_count: usize,
    pub instrument_unexpected_assignment_count: usize,
    pub sound_tie_semantics: MusicXmlNotationAccuracyV1,
    pub visual_tie_semantics: MusicXmlNotationAccuracyV1,
    pub time_modification: MusicXmlNotationAccuracyV1,
    pub tuplet_notation: MusicXmlNotationAccuracyV1,
    pub full_note_source_subtree: MusicXmlNotationAccuracyV1,
    pub reference_auxiliary_event_count: usize,
    pub prediction_auxiliary_event_count: usize,
    pub auxiliary_event_presence: MusicXmlNotationMetricV1,
    pub auxiliary_source_subtree: MusicXmlNotationAccuracyV1,
    pub reference_attribute_event_count: usize,
    pub prediction_attribute_event_count: usize,
    pub attribute_event_presence: MusicXmlNotationMetricV1,
    pub attribute_source_subtree: MusicXmlNotationAccuracyV1,
    pub reference_cursor_move_count: usize,
    pub prediction_cursor_move_count: usize,
    pub cursor_move_presence: MusicXmlNotationMetricV1,
    pub cursor_move_semantics: MusicXmlNotationAccuracyV1,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicXmlNotationEvaluationErrorV1 {
    ReferenceMeasureMapUnavailable,
    PredictionMeasureMapUnavailable,
    ReferenceNotationEventsUnavailable,
    PredictionNotationEventsUnavailable,
    PartIdentityMismatch,
    ScoreClockMismatch,
    MeasureCanonicalizationMismatch,
    NotationCanonicalizationMismatch,
    InvalidGraphStructure { side: String, reason: String },
    EventBudgetExceeded { side: String, count: usize, maximum: usize },
    CandidatePairBudgetExceeded { count: usize, maximum: usize },
    MatchingWorkBudgetExceeded { scans: usize, maximum: usize },
    InvalidConfiguration { field: String },
    SerializationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventFamily {
    Pitched,
    Rest,
    Unpitched,
}

fn event_family(kind: &MusicXmlNotationEventKindV1) -> EventFamily {
    match kind {
        MusicXmlNotationEventKindV1::Pitched(_) => EventFamily::Pitched,
        MusicXmlNotationEventKindV1::Rest { .. } => EventFamily::Rest,
        MusicXmlNotationEventKindV1::Unpitched { .. } => EventFamily::Unpitched,
    }
}

fn exact_within(
    left: RationalBeat,
    right: RationalBeat,
    tolerance: RationalBeat,
) -> Result<bool, MusicXmlNotationEvaluationErrorV1> {
    let difference = if left.compare_exact(right) == Ordering::Greater {
        left.checked_sub(right)
    } else {
        right.checked_sub(left)
    }
    .map_err(|_| MusicXmlNotationEvaluationErrorV1::InvalidConfiguration {
        field: "rational comparison overflow".into(),
    })?;
    Ok(difference.compare_exact(tolerance) != Ordering::Greater)
}

fn gcd_i128(mut left: i128, mut right: i128) -> i128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.abs().max(1)
}

/// Parse an XML decimal to a reduced exact rational; never use f64 for
/// written-pitch alteration equality. Equivalent lexical forms (0 and 0.0)
/// denote the same score alteration.
fn canonical_decimal(value: &str) -> Option<(i128, i128)> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let (negative, unsigned) = if let Some(rest) = value.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = value.strip_prefix('+') {
        (false, rest)
    } else {
        (false, value)
    };
    let mut pieces = unsigned.split('.');
    let whole = pieces.next()?;
    let fraction = pieces.next().unwrap_or("");
    if pieces.next().is_some()
        || (whole.is_empty() && fraction.is_empty())
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let scale = 10_i128.checked_pow(u32::try_from(fraction.len()).ok()?)?;
    let whole_value = if whole.is_empty() { 0 } else { whole.parse::<i128>().ok()? };
    let fraction_value = if fraction.is_empty() { 0 } else { fraction.parse::<i128>().ok()? };
    let magnitude = whole_value.checked_mul(scale)?.checked_add(fraction_value)?;
    let numerator = if negative { -magnitude } else { magnitude };
    let divisor = gcd_i128(numerator, scale);
    Some((numerator / divisor, scale / divisor))
}

fn written_pitch_equal(
    left: &crate::musicxml_notation_events::MusicXmlWrittenPitchV1,
    right: &crate::musicxml_notation_events::MusicXmlWrittenPitchV1,
) -> bool {
    if left.step != right.step || left.octave != right.octave {
        return false;
    }
    match (&left.alter, &right.alter) {
        (None, None) => true,
        (Some(left), Some(right)) => match (
            canonical_decimal(left),
            canonical_decimal(right),
        ) {
            (Some(left), Some(right)) => left == right,
            _ => left == right,
        },
        _ => false,
    }
}

fn accuracy(samples: usize, correct: usize) -> MusicXmlNotationAccuracyV1 {
    MusicXmlNotationAccuracyV1 {
        sample_count: samples,
        correct_count: correct,
        accuracy: (samples > 0).then_some(correct as f64 / samples as f64),
    }
}

fn metric(
    reference_count: usize,
    prediction_count: usize,
    true_positives: usize,
) -> MusicXmlNotationMetricV1 {
    let precision = (prediction_count > 0)
        .then_some(true_positives as f64 / prediction_count as f64);
    let recall = (reference_count > 0)
        .then_some(true_positives as f64 / reference_count as f64);
    let f1 = if reference_count == 0 && prediction_count == 0 {
        None
    } else if true_positives == 0 {
        Some(0.0)
    } else {
        match (precision, recall) {
            (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
            _ => Some(0.0),
        }
    };
    MusicXmlNotationMetricV1 {
        true_positives,
        false_positives: prediction_count.saturating_sub(true_positives),
        false_negatives: reference_count.saturating_sub(true_positives),
        precision,
        recall,
        f1,
    }
}

fn rational_is_canonical(value: RationalBeat) -> bool {
    RationalBeat::new(value.numerator(), value.denominator())
        .is_ok_and(|canonical| canonical == value)
}

fn span_contains(span: &crate::barline_eval::MeasureSpan, value: RationalBeat) -> bool {
    value.compare_exact(span.start) != Ordering::Less
        && value.compare_exact(span.end) != Ordering::Greater
}

fn validate_graph(
    graph: &MusicXmlPartNotationGraphV1,
    side: &str,
    config: &MusicXmlNotationEvaluationConfigV1,
) -> Result<(), MusicXmlNotationEvaluationErrorV1> {
    if graph.part_id.trim().is_empty()
        || graph.notation_events.part_id != graph.part_id
        || graph.map.score_clock_id.trim().is_empty()
    {
        return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
            side: side.into(),
            reason: "empty or inconsistent part/score-clock identity".into(),
        });
    }
    if !matches!(&graph.map.status, MeasureMapStatus::Supported) {
        return Err(if side == "reference" {
            MusicXmlNotationEvaluationErrorV1::ReferenceMeasureMapUnavailable
        } else {
            MusicXmlNotationEvaluationErrorV1::PredictionMeasureMapUnavailable
        });
    }
    if !matches!(
        &graph.notation_events.status,
        MusicXmlNotationEventsStatusV1::Supported
    ) {
        return Err(if side == "reference" {
            MusicXmlNotationEvaluationErrorV1::ReferenceNotationEventsUnavailable
        } else {
            MusicXmlNotationEvaluationErrorV1::PredictionNotationEventsUnavailable
        });
    }
    if graph.notation_events.schema != MUSICXML_NOTATION_EVENTS_SCHEMA_V1
        || graph.notation_events.canonicalization_version
            != MUSICXML_NOTATION_EVENTS_CANONICALIZATION_V1
    {
        return Err(MusicXmlNotationEvaluationErrorV1::NotationCanonicalizationMismatch);
    }
    if graph.map.canonicalization_version != MUSICXML_BARLINE_CANONICALIZATION_V1 {
        return Err(MusicXmlNotationEvaluationErrorV1::MeasureCanonicalizationMismatch);
    }
    if graph.map.measures.is_empty() {
        return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
            side: side.into(),
            reason: "supported score clock has no measure spans".into(),
        });
    }
    let mut previous_end = None;
    for (index, span) in graph.map.measures.iter().enumerate() {
        if !rational_is_canonical(span.start)
            || !rational_is_canonical(span.end)
            || (index == 0 && span.start.numerator() != 0)
            || span.start.compare_exact(span.end) != Ordering::Less
            || previous_end.is_some_and(|end| end != span.start)
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
                side: side.into(),
                reason: "measure spans are noncanonical, empty, overlapping or discontinuous".into(),
            });
        }
        previous_end = Some(span.end);
    }
    let note_count = graph.notation_events.note_events.len();
    let other_count = graph
        .notation_events
        .attribute_events
        .len()
        .saturating_add(graph.notation_events.auxiliary_events.len())
        .saturating_add(graph.notation_events.cursor_moves.len());
    let total = note_count.saturating_add(other_count);
    if total > config.maximum_events_per_part {
        return Err(MusicXmlNotationEvaluationErrorV1::EventBudgetExceeded {
            side: side.into(),
            count: total,
            maximum: config.maximum_events_per_part,
        });
    }

    let mut orders = BTreeSet::new();
    for event in &graph.notation_events.note_events {
        if event.part_id != graph.part_id
            || event.source_measure_index >= graph.map.measures.len()
            || !orders.insert(event.source_order)
            || !rational_is_canonical(event.score_offset)
            || !span_contains(
                &graph.map.measures[event.source_measure_index],
                event.score_offset,
            )
            || event.performed_duration.is_some_and(|duration| {
                !rational_is_canonical(duration)
                    || duration.numerator() == 0
                    || match event.score_offset.checked_add(duration) {
                        Ok(end) => {
                            end.compare_exact(graph.map.measures[event.source_measure_index].end)
                                == Ordering::Greater
                        }
                        Err(_) => true,
                    }
            })
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
                side: side.into(),
                reason: "invalid note-event part/measure/source-order/coordinate binding".into(),
            });
        }
    }
    for event in &graph.notation_events.attribute_events {
        if event.part_id != graph.part_id
            || event.measure_index >= graph.map.measures.len()
            || !orders.insert(event.source_order)
            || !rational_is_canonical(event.score_offset)
            || !span_contains(&graph.map.measures[event.measure_index], event.score_offset)
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
                side: side.into(),
                reason: "invalid attribute-event part/measure/source-order binding".into(),
            });
        }
    }
    for event in &graph.notation_events.auxiliary_events {
        if event.part_id != graph.part_id
            || event.measure_index >= graph.map.measures.len()
            || !orders.insert(event.source_order)
            || !rational_is_canonical(event.score_offset)
            || !span_contains(&graph.map.measures[event.measure_index], event.score_offset)
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
                side: side.into(),
                reason: "invalid auxiliary-event part/measure/source-order binding".into(),
            });
        }
    }
    for event in &graph.notation_events.cursor_moves {
        let in_span = event.measure_index < graph.map.measures.len()
            && rational_is_canonical(event.score_offset_before)
            && rational_is_canonical(event.score_offset_after)
            && rational_is_canonical(event.duration)
            && span_contains(
                &graph.map.measures[event.measure_index],
                event.score_offset_before,
            )
            && span_contains(
                &graph.map.measures[event.measure_index],
                event.score_offset_after,
            );
        let duration_matches_direction = match event.direction.as_str() {
            "backup" => event.score_offset_before
                .checked_sub(event.score_offset_after)
                .is_ok_and(|distance| distance == event.duration),
            "forward" => event.score_offset_after
                .checked_sub(event.score_offset_before)
                .is_ok_and(|distance| distance == event.duration),
            _ => false,
        };
        if event.part_id != graph.part_id
            || event.measure_index >= graph.map.measures.len()
            || !orders.insert(event.source_order)
            || !in_span
            || !duration_matches_direction
            || event.duration.numerator() == 0
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
                side: side.into(),
                reason: "invalid cursor-move part/measure/source-order binding".into(),
            });
        }
    }
    let family_order_is_strict = [
        graph.notation_events.note_events.iter().map(|event| event.source_order).collect::<Vec<_>>(),
        graph.notation_events.attribute_events.iter().map(|event| event.source_order).collect::<Vec<_>>(),
        graph.notation_events.auxiliary_events.iter().map(|event| event.source_order).collect::<Vec<_>>(),
        graph.notation_events.cursor_moves.iter().map(|event| event.source_order).collect::<Vec<_>>(),
    ]
    .iter()
    .all(|family| family.windows(2).all(|pair| pair[0] < pair[1]));
    if !family_order_is_strict
        || orders.len() != total
        || orders.iter().copied().enumerate().any(|(expected, observed)| expected != observed)
    {
        return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
            side: side.into(),
            reason: "source-event order must be unique and contiguous across event families".into(),
        });
    }
    Ok(())
}

/// Deterministic one-to-one maximum-cardinality bipartite matching.
///
/// An iterative alternating-path search avoids recursive stack growth. Candidate
/// adjacency is emitted in input order, so equal-cardinality solutions have a
/// stable, source-order tie-break. Both candidate and scan work are bounded.
fn maximum_cardinality_matching<A, B, F>(
    left: &[A],
    right: &[B],
    mut candidate: F,
    config: &MusicXmlNotationEvaluationConfigV1,
) -> Result<Vec<(usize, usize)>, MusicXmlNotationEvaluationErrorV1>
where
    F: FnMut(&A, &B) -> Result<bool, MusicXmlNotationEvaluationErrorV1>,
{
    let mut edges = vec![Vec::<usize>::new(); left.len()];
    let mut candidate_scans = 0usize;
    for (left_index, left_item) in left.iter().enumerate() {
        for (right_index, right_item) in right.iter().enumerate() {
            candidate_scans = candidate_scans.saturating_add(1);
            if candidate_scans > config.maximum_candidate_pairs {
                return Err(MusicXmlNotationEvaluationErrorV1::CandidatePairBudgetExceeded {
                    count: candidate_scans,
                    maximum: config.maximum_candidate_pairs,
                });
            }
            if candidate(left_item, right_item)? {
                edges[left_index].push(right_index);
            }
        }
    }

    let mut left_match = vec![None; left.len()];
    let mut right_match = vec![None; right.len()];
    let mut seen_left_generation = vec![0usize; left.len()];
    let mut seen_right_generation = vec![0usize; right.len()];
    let mut parent_right = vec![None; right.len()];
    let mut queue = std::collections::VecDeque::new();
    let mut generation = 0usize;
    let mut scans = 0usize;

    for root in 0..left.len() {
        if left_match[root].is_some() || edges[root].is_empty() {
            continue;
        }
        generation = generation.checked_add(1).ok_or_else(|| {
            MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
                side: "matching".into(),
                reason: "matching generation counter overflow".into(),
            }
        })?;
        queue.clear();
        queue.push_back(root);
        seen_left_generation[root] = generation;
        let mut free_right = None;

        'search: while let Some(left_index) = queue.pop_front() {
            for &right_index in &edges[left_index] {
                scans = scans.saturating_add(1);
                if scans > config.maximum_matching_scans {
                    return Err(
                        MusicXmlNotationEvaluationErrorV1::MatchingWorkBudgetExceeded {
                            scans,
                            maximum: config.maximum_matching_scans,
                        },
                    );
                }
                if seen_right_generation[right_index] == generation {
                    continue;
                }
                seen_right_generation[right_index] = generation;
                parent_right[right_index] = Some(left_index);
                if let Some(next_left) = right_match[right_index] {
                    if seen_left_generation[next_left] != generation {
                        seen_left_generation[next_left] = generation;
                        queue.push_back(next_left);
                    }
                } else {
                    free_right = Some(right_index);
                    break 'search;
                }
            }
        }

        if let Some(mut right_index) = free_right {
            loop {
                let left_index = parent_right[right_index].ok_or_else(|| {
                    MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure {
                        side: "matching".into(),
                        reason: "augmenting path predecessor is missing".into(),
                    }
                })?;
                let previous_right = left_match[left_index];
                left_match[left_index] = Some(right_index);
                right_match[right_index] = Some(left_index);
                match previous_right {
                    Some(previous) => right_index = previous,
                    None => break,
                }
            }
        }
    }

    Ok(left_match
        .into_iter()
        .enumerate()
        .filter_map(|(left_index, right_index)| {
            right_index.map(|right_index| (left_index, right_index))
        })
        .collect())
}

fn error_distribution(
    errors: &mut Vec<f64>,
    within_tolerance_count: usize,
) -> MusicXmlNotationErrorDistributionV1 {
    if errors.is_empty() {
        return MusicXmlNotationErrorDistributionV1 {
            sample_count: 0,
            mean_absolute_quarter_beats: None,
            median_absolute_quarter_beats: None,
            p95_absolute_quarter_beats: None,
            p99_absolute_quarter_beats: None,
            within_tolerance_count: 0,
            within_tolerance_rate: None,
        };
    }
    errors.sort_by(f64::total_cmp);
    let n = errors.len();
    let percentile = |p: f64| {
        let index = ((p * n as f64).ceil() as usize)
            .saturating_sub(1)
            .min(n.saturating_sub(1));
        errors[index]
    };
    MusicXmlNotationErrorDistributionV1 {
        sample_count: n,
        mean_absolute_quarter_beats: Some(errors.iter().sum::<f64>() / n as f64),
        median_absolute_quarter_beats: Some(percentile(0.50)),
        p95_absolute_quarter_beats: Some(percentile(0.95)),
        p99_absolute_quarter_beats: Some(percentile(0.99)),
        within_tolerance_count,
        within_tolerance_rate: Some(within_tolerance_count as f64 / n as f64),
    }
}

fn absolute_error_beats(
    left: RationalBeat,
    right: RationalBeat,
) -> Result<f64, MusicXmlNotationEvaluationErrorV1> {
    let difference = if left.compare_exact(right) == Ordering::Greater {
        left.checked_sub(right)
    } else {
        right.checked_sub(left)
    }
    .map_err(|_| MusicXmlNotationEvaluationErrorV1::InvalidConfiguration {
        field: "duration difference overflow".into(),
    })?;
    Ok(difference.as_beats())
}

#[derive(Debug, Clone)]
struct NotePairDimensionsV1 {
    written_pitch_spelling: MusicXmlNotationAccuracyV1,
    rest_semantics: MusicXmlNotationAccuracyV1,
    unpitched_display: MusicXmlNotationAccuracyV1,
    duration_presence: MusicXmlNotationAccuracyV1,
    duration_error: MusicXmlNotationErrorDistributionV1,
    written_type_and_dots: MusicXmlNotationAccuracyV1,
    voice_assignment: MusicXmlNotationAccuracyV1,
    voice_reference_unassigned_count: usize,
    voice_unexpected_assignment_count: usize,
    staff_assignment: MusicXmlNotationAccuracyV1,
    staff_reference_unassigned_count: usize,
    staff_unexpected_assignment_count: usize,
    instrument_assignment: MusicXmlNotationAccuracyV1,
    instrument_reference_unassigned_count: usize,
    instrument_unexpected_assignment_count: usize,
    sound_tie_semantics: MusicXmlNotationAccuracyV1,
    visual_tie_semantics: MusicXmlNotationAccuracyV1,
    time_modification: MusicXmlNotationAccuracyV1,
    tuplet_notation: MusicXmlNotationAccuracyV1,
    full_note_source_subtree: MusicXmlNotationAccuracyV1,
}

fn note_pair_dimensions(
    reference: &MusicXmlPartNotationEventsV1,
    prediction: &MusicXmlPartNotationEventsV1,
    pairs: &[(usize, usize)],
    config: &MusicXmlNotationEvaluationConfigV1,
) -> Result<NotePairDimensionsV1, MusicXmlNotationEvaluationErrorV1> {
    let mut pitch_samples = 0;
    let mut pitch_correct = 0;
    let mut rest_samples = 0;
    let mut rest_correct = 0;
    let mut unpitched_samples = 0;
    let mut unpitched_correct = 0;
    let mut duration_presence_samples = 0;
    let mut duration_presence_correct = 0;
    let mut duration_errors = Vec::new();
    let mut duration_within = 0;
    let mut written_samples = 0;
    let mut written_correct = 0;
    let mut voice_samples = 0;
    let mut voice_correct = 0;
    let mut voice_reference_unassigned_count = 0;
    let mut voice_unexpected_assignment_count = 0;
    let mut staff_samples = 0;
    let mut staff_correct = 0;
    let mut staff_reference_unassigned_count = 0;
    let mut staff_unexpected_assignment_count = 0;
    let mut instrument_samples = 0;
    let mut instrument_correct = 0;
    let mut instrument_reference_unassigned_count = 0;
    let mut instrument_unexpected_assignment_count = 0;
    let mut sound_tie_samples = 0;
    let mut sound_tie_correct = 0;
    let mut visual_tie_samples = 0;
    let mut visual_tie_correct = 0;
    let mut time_mod_samples = 0;
    let mut time_mod_correct = 0;
    let mut tuplet_samples = 0;
    let mut tuplet_correct = 0;
    let mut source_samples = 0;
    let mut source_correct = 0;

    for &(reference_index, prediction_index) in pairs {
        let gold = &reference.note_events[reference_index];
        let predicted = &prediction.note_events[prediction_index];
        match (&gold.kind, &predicted.kind) {
            (
                MusicXmlNotationEventKindV1::Pitched(gold_pitch),
                MusicXmlNotationEventKindV1::Pitched(predicted_pitch),
            ) => {
                pitch_samples += 1;
                if written_pitch_equal(gold_pitch, predicted_pitch) {
                    pitch_correct += 1;
                }
            }
            (
                MusicXmlNotationEventKindV1::Rest {
                    measure: gold_measure,
                    display: gold_display,
                },
                MusicXmlNotationEventKindV1::Rest {
                    measure: predicted_measure,
                    display: predicted_display,
                },
            ) => {
                rest_samples += 1;
                if gold_measure == predicted_measure && gold_display == predicted_display {
                    rest_correct += 1;
                }
            }
            (
                MusicXmlNotationEventKindV1::Unpitched { display: gold_display },
                MusicXmlNotationEventKindV1::Unpitched { display: predicted_display },
            ) => {
                unpitched_samples += 1;
                if gold_display == predicted_display {
                    unpitched_correct += 1;
                }
            }
            _ => {}
        }

        duration_presence_samples += 1;
        if gold.performed_duration.is_some() == predicted.performed_duration.is_some() {
            duration_presence_correct += 1;
        }
        if let (Some(gold_duration), Some(predicted_duration)) =
            (gold.performed_duration, predicted.performed_duration)
        {
            let error = absolute_error_beats(gold_duration, predicted_duration)?;
            if exact_within(
                gold_duration,
                predicted_duration,
                config.duration_tolerance,
            )? {
                duration_within += 1;
            }
            duration_errors.push(error);
        }

        written_samples += 1;
        if gold.written_type == predicted.written_type && gold.dot_count == predicted.dot_count {
            written_correct += 1;
        }

        match &gold.voice {
            Some(_) => {
                voice_samples += 1;
                if gold.voice == predicted.voice {
                    voice_correct += 1;
                }
            }
            None => {
                voice_reference_unassigned_count += 1;
                if predicted.voice.is_some() {
                    voice_unexpected_assignment_count += 1;
                }
            }
        }
        match &gold.staff {
            Some(_) => {
                staff_samples += 1;
                if gold.staff == predicted.staff {
                    staff_correct += 1;
                }
            }
            None => {
                staff_reference_unassigned_count += 1;
                if predicted.staff.is_some() {
                    staff_unexpected_assignment_count += 1;
                }
            }
        }
        if gold.instrument_ids.is_empty() {
            instrument_reference_unassigned_count += 1;
            if !predicted.instrument_ids.is_empty() {
                instrument_unexpected_assignment_count += 1;
            }
        } else {
            instrument_samples += 1;
            if gold.instrument_ids == predicted.instrument_ids {
                instrument_correct += 1;
            }
        }

        sound_tie_samples += 1;
        if gold.sound_ties == predicted.sound_ties {
            sound_tie_correct += 1;
        }
        visual_tie_samples += 1;
        if gold.visual_ties == predicted.visual_ties {
            visual_tie_correct += 1;
        }
        time_mod_samples += 1;
        if gold.time_modification == predicted.time_modification {
            time_mod_correct += 1;
        }
        tuplet_samples += 1;
        if gold.tuplet_notations == predicted.tuplet_notations {
            tuplet_correct += 1;
        }
        source_samples += 1;
        if gold.source_note == predicted.source_note {
            source_correct += 1;
        }
    }

    Ok(NotePairDimensionsV1 {
        written_pitch_spelling: accuracy(pitch_samples, pitch_correct),
        rest_semantics: accuracy(rest_samples, rest_correct),
        unpitched_display: accuracy(unpitched_samples, unpitched_correct),
        duration_presence: accuracy(duration_presence_samples, duration_presence_correct),
        duration_error: error_distribution(&mut duration_errors, duration_within),
        written_type_and_dots: accuracy(written_samples, written_correct),
        voice_assignment: accuracy(voice_samples, voice_correct),
        voice_reference_unassigned_count,
        voice_unexpected_assignment_count,
        staff_assignment: accuracy(staff_samples, staff_correct),
        staff_reference_unassigned_count,
        staff_unexpected_assignment_count,
        instrument_assignment: accuracy(instrument_samples, instrument_correct),
        instrument_reference_unassigned_count,
        instrument_unexpected_assignment_count,
        sound_tie_semantics: accuracy(sound_tie_samples, sound_tie_correct),
        visual_tie_semantics: accuracy(visual_tie_samples, visual_tie_correct),
        time_modification: accuracy(time_mod_samples, time_mod_correct),
        tuplet_notation: accuracy(tuplet_samples, tuplet_correct),
        full_note_source_subtree: accuracy(source_samples, source_correct),
    })
}

fn note_onset_error(
    reference: &MusicXmlPartNotationEventsV1,
    prediction: &MusicXmlPartNotationEventsV1,
    pairs: &[(usize, usize)],
    tolerance: RationalBeat,
) -> Result<MusicXmlNotationErrorDistributionV1, MusicXmlNotationEvaluationErrorV1> {
    let mut errors = Vec::with_capacity(pairs.len());
    let mut within = 0usize;
    for &(reference_index, prediction_index) in pairs {
        let gold = reference.note_events[reference_index].score_offset;
        let predicted = prediction.note_events[prediction_index].score_offset;
        if exact_within(gold, predicted, tolerance)? {
            within += 1;
        }
        errors.push(absolute_error_beats(gold, predicted)?);
    }
    Ok(error_distribution(&mut errors, within))
}

fn source_tree_accuracy<A, B, F>(
    reference: &[A],
    prediction: &[B],
    pairs: &[(usize, usize)],
    equal: F,
) -> MusicXmlNotationAccuracyV1
where
    F: Fn(&A, &B) -> bool,
{
    accuracy(
        pairs.len(),
        pairs
            .iter()
            .filter(|&&(left, right)| equal(&reference[left], &prediction[right]))
            .count(),
    )
}

#[derive(Serialize)]
struct ReportBodyV1<'a> {
    schema: &'a str,
    algorithm: &'a str,
    config: &'a MusicXmlNotationEvaluationConfigV1,
    part_id: &'a str,
    score_clock_id: &'a str,
    canonicalization_version: &'a str,
    reference_graph_sha256: &'a str,
    prediction_graph_sha256: &'a str,
    reference_note_event_count: usize,
    prediction_note_event_count: usize,
    note_event_presence: &'a MusicXmlNotationMetricV1,
    note_onset_error: &'a MusicXmlNotationErrorDistributionV1,
    written_pitch_spelling: &'a MusicXmlNotationAccuracyV1,
    rest_semantics: &'a MusicXmlNotationAccuracyV1,
    unpitched_display: &'a MusicXmlNotationAccuracyV1,
    duration_presence: &'a MusicXmlNotationAccuracyV1,
    duration_error: &'a MusicXmlNotationErrorDistributionV1,
    written_type_and_dots: &'a MusicXmlNotationAccuracyV1,
    voice_assignment: &'a MusicXmlNotationAccuracyV1,
    voice_reference_unassigned_count: usize,
    voice_unexpected_assignment_count: usize,
    staff_assignment: &'a MusicXmlNotationAccuracyV1,
    staff_reference_unassigned_count: usize,
    staff_unexpected_assignment_count: usize,
    instrument_assignment: &'a MusicXmlNotationAccuracyV1,
    instrument_reference_unassigned_count: usize,
    instrument_unexpected_assignment_count: usize,
    sound_tie_semantics: &'a MusicXmlNotationAccuracyV1,
    visual_tie_semantics: &'a MusicXmlNotationAccuracyV1,
    time_modification: &'a MusicXmlNotationAccuracyV1,
    tuplet_notation: &'a MusicXmlNotationAccuracyV1,
    full_note_source_subtree: &'a MusicXmlNotationAccuracyV1,
    reference_auxiliary_event_count: usize,
    prediction_auxiliary_event_count: usize,
    auxiliary_event_presence: &'a MusicXmlNotationMetricV1,
    auxiliary_source_subtree: &'a MusicXmlNotationAccuracyV1,
    reference_attribute_event_count: usize,
    prediction_attribute_event_count: usize,
    attribute_event_presence: &'a MusicXmlNotationMetricV1,
    attribute_source_subtree: &'a MusicXmlNotationAccuracyV1,
    reference_cursor_move_count: usize,
    prediction_cursor_move_count: usize,
    cursor_move_presence: &'a MusicXmlNotationMetricV1,
    cursor_move_semantics: &'a MusicXmlNotationAccuracyV1,
}

fn report_commitment(
    report: &MusicXmlNotationEvaluationReportV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ReportBodyV1 {
        schema: &report.schema,
        algorithm: &report.algorithm,
        config: &report.config,
        part_id: &report.part_id,
        score_clock_id: &report.score_clock_id,
        canonicalization_version: &report.canonicalization_version,
        reference_graph_sha256: &report.reference_graph_sha256,
        prediction_graph_sha256: &report.prediction_graph_sha256,
        reference_note_event_count: report.reference_note_event_count,
        prediction_note_event_count: report.prediction_note_event_count,
        note_event_presence: &report.note_event_presence,
        note_onset_error: &report.note_onset_error,
        written_pitch_spelling: &report.written_pitch_spelling,
        rest_semantics: &report.rest_semantics,
        unpitched_display: &report.unpitched_display,
        duration_presence: &report.duration_presence,
        duration_error: &report.duration_error,
        written_type_and_dots: &report.written_type_and_dots,
        voice_assignment: &report.voice_assignment,
        voice_reference_unassigned_count: report.voice_reference_unassigned_count,
        voice_unexpected_assignment_count: report.voice_unexpected_assignment_count,
        staff_assignment: &report.staff_assignment,
        staff_reference_unassigned_count: report.staff_reference_unassigned_count,
        staff_unexpected_assignment_count: report.staff_unexpected_assignment_count,
        instrument_assignment: &report.instrument_assignment,
        instrument_reference_unassigned_count: report.instrument_reference_unassigned_count,
        instrument_unexpected_assignment_count: report.instrument_unexpected_assignment_count,
        sound_tie_semantics: &report.sound_tie_semantics,
        visual_tie_semantics: &report.visual_tie_semantics,
        time_modification: &report.time_modification,
        tuplet_notation: &report.tuplet_notation,
        full_note_source_subtree: &report.full_note_source_subtree,
        reference_auxiliary_event_count: report.reference_auxiliary_event_count,
        prediction_auxiliary_event_count: report.prediction_auxiliary_event_count,
        auxiliary_event_presence: &report.auxiliary_event_presence,
        auxiliary_source_subtree: &report.auxiliary_source_subtree,
        reference_attribute_event_count: report.reference_attribute_event_count,
        prediction_attribute_event_count: report.prediction_attribute_event_count,
        attribute_event_presence: &report.attribute_event_presence,
        attribute_source_subtree: &report.attribute_source_subtree,
        reference_cursor_move_count: report.reference_cursor_move_count,
        prediction_cursor_move_count: report.prediction_cursor_move_count,
        cursor_move_presence: &report.cursor_move_presence,
        cursor_move_semantics: &report.cursor_move_semantics,
    })
}

/// Evaluate one reference and one prediction per-part MusicXML graph.
///
/// Note event presence is matched by event family and score onset only. Exact
/// pitch spelling, duration, written value, tie, tuplet, staff and source-tree
/// correctness are then reported independently among the matched pairs.
pub fn evaluate_musicxml_notation_graphs(
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    config: MusicXmlNotationEvaluationConfigV1,
) -> Result<MusicXmlNotationEvaluationReportV1, MusicXmlNotationEvaluationErrorV1> {
    if !rational_is_canonical(config.position_tolerance)
        || !rational_is_canonical(config.duration_tolerance)
    {
        return Err(MusicXmlNotationEvaluationErrorV1::InvalidConfiguration {
            field: "tolerances must be canonical nonnegative rational beats".into(),
        });
    }
    if config.maximum_events_per_part == 0
        || config.maximum_events_per_part > MUSICXML_NOTATION_EVALUATION_MAX_EVENTS_V1
        || config.maximum_candidate_pairs == 0
        || config.maximum_candidate_pairs > MUSICXML_NOTATION_EVALUATION_MAX_CANDIDATE_PAIRS_V1
        || config.maximum_matching_scans == 0
        || config.maximum_matching_scans > MUSICXML_NOTATION_EVALUATION_MAX_MATCHING_SCANS_V1
    {
        return Err(MusicXmlNotationEvaluationErrorV1::InvalidConfiguration {
            field: "event and matching work budgets must be positive and no greater than the V1 caps".into(),
        });
    }
    validate_graph(reference, "reference", &config)?;
    validate_graph(prediction, "prediction", &config)?;

    if reference.part_id != prediction.part_id {
        return Err(MusicXmlNotationEvaluationErrorV1::PartIdentityMismatch);
    }
    if reference.map.score_clock_id != prediction.map.score_clock_id {
        return Err(MusicXmlNotationEvaluationErrorV1::ScoreClockMismatch);
    }
    if reference.map.canonicalization_version != prediction.map.canonicalization_version {
        return Err(MusicXmlNotationEvaluationErrorV1::MeasureCanonicalizationMismatch);
    }

    let gold = &reference.notation_events;
    let predicted = &prediction.notation_events;
    let note_pairs = maximum_cardinality_matching(
        &gold.note_events,
        &predicted.note_events,
        |left, right| {
            Ok(
            event_family(&left.kind) == event_family(&right.kind)
                && exact_within(
                    left.score_offset,
                    right.score_offset,
                    config.position_tolerance,
                )?,
        )
        },
        &config,
    )?;
    let note_dimensions =
        note_pair_dimensions(gold, predicted, &note_pairs, &config)?;

    let auxiliary_pairs = maximum_cardinality_matching(
        &gold.auxiliary_events,
        &predicted.auxiliary_events,
        |left, right| {
            Ok(left.source.element == right.source.element
                && exact_within(
                    left.score_offset,
                    right.score_offset,
                    config.position_tolerance,
                )?)
        },
        &config,
    )?;
    let auxiliary_source_subtree = source_tree_accuracy(
        &gold.auxiliary_events,
        &predicted.auxiliary_events,
        &auxiliary_pairs,
        |left, right| left.source == right.source,
    );

    let attribute_pairs = maximum_cardinality_matching(
        &gold.attribute_events,
        &predicted.attribute_events,
        |left, right| {
            Ok(left.source.element == right.source.element
                && exact_within(
                    left.score_offset,
                    right.score_offset,
                    config.position_tolerance,
                )?)
        },
        &config,
    )?;
    let attribute_source_subtree = source_tree_accuracy(
        &gold.attribute_events,
        &predicted.attribute_events,
        &attribute_pairs,
        |left, right| left.source == right.source,
    );

    let cursor_pairs = maximum_cardinality_matching(
        &gold.cursor_moves,
        &predicted.cursor_moves,
        |left, right| {
            Ok(left.direction == right.direction
                && exact_within(
                    left.score_offset_before,
                    right.score_offset_before,
                    config.position_tolerance,
                )?
                && exact_within(
                    left.score_offset_after,
                    right.score_offset_after,
                    config.position_tolerance,
                )?)
        },
        &config,
    )?;
    let cursor_move_semantics = source_tree_accuracy(
        &gold.cursor_moves,
        &predicted.cursor_moves,
        &cursor_pairs,
        |left, right| {
            left.duration == right.duration
                && left.voice == right.voice
                && left.staff == right.staff
                && left.source == right.source
        },
    );

    let reference_graph_sha256 =
        canonical_json_sha256(reference).map_err(|_| {
            MusicXmlNotationEvaluationErrorV1::SerializationFailed
        })?;
    let prediction_graph_sha256 =
        canonical_json_sha256(prediction).map_err(|_| {
            MusicXmlNotationEvaluationErrorV1::SerializationFailed
        })?;
    let note_onset_error = note_onset_error(
        gold,
        predicted,
        &note_pairs,
        config.position_tolerance,
    )?;

    let mut report = MusicXmlNotationEvaluationReportV1 {
        schema: MUSICXML_NOTATION_EVALUATION_SCHEMA_V1.into(),
        algorithm: MUSICXML_NOTATION_EVALUATION_ALGORITHM_V1.into(),
        config,
        part_id: reference.part_id.clone(),
        score_clock_id: reference.map.score_clock_id.clone(),
        canonicalization_version: reference.map.canonicalization_version.clone(),
        reference_graph_sha256,
        prediction_graph_sha256,
        reference_note_event_count: gold.note_events.len(),
        prediction_note_event_count: predicted.note_events.len(),
        note_event_presence: metric(
            gold.note_events.len(),
            predicted.note_events.len(),
            note_pairs.len(),
        ),
        note_onset_error,
        written_pitch_spelling: note_dimensions.written_pitch_spelling,
        rest_semantics: note_dimensions.rest_semantics,
        unpitched_display: note_dimensions.unpitched_display,
        duration_presence: note_dimensions.duration_presence,
        duration_error: note_dimensions.duration_error,
        written_type_and_dots: note_dimensions.written_type_and_dots,
        voice_assignment: note_dimensions.voice_assignment,
        voice_reference_unassigned_count: note_dimensions.voice_reference_unassigned_count,
        voice_unexpected_assignment_count: note_dimensions.voice_unexpected_assignment_count,
        staff_assignment: note_dimensions.staff_assignment,
        staff_reference_unassigned_count: note_dimensions.staff_reference_unassigned_count,
        staff_unexpected_assignment_count: note_dimensions.staff_unexpected_assignment_count,
        instrument_assignment: note_dimensions.instrument_assignment,
        instrument_reference_unassigned_count: note_dimensions.instrument_reference_unassigned_count,
        instrument_unexpected_assignment_count: note_dimensions.instrument_unexpected_assignment_count,
        sound_tie_semantics: note_dimensions.sound_tie_semantics,
        visual_tie_semantics: note_dimensions.visual_tie_semantics,
        time_modification: note_dimensions.time_modification,
        tuplet_notation: note_dimensions.tuplet_notation,
        full_note_source_subtree: note_dimensions.full_note_source_subtree,
        reference_auxiliary_event_count: gold.auxiliary_events.len(),
        prediction_auxiliary_event_count: predicted.auxiliary_events.len(),
        auxiliary_event_presence: metric(
            gold.auxiliary_events.len(),
            predicted.auxiliary_events.len(),
            auxiliary_pairs.len(),
        ),
        auxiliary_source_subtree,
        reference_attribute_event_count: gold.attribute_events.len(),
        prediction_attribute_event_count: predicted.attribute_events.len(),
        attribute_event_presence: metric(
            gold.attribute_events.len(),
            predicted.attribute_events.len(),
            attribute_pairs.len(),
        ),
        attribute_source_subtree,
        reference_cursor_move_count: gold.cursor_moves.len(),
        prediction_cursor_move_count: predicted.cursor_moves.len(),
        cursor_move_presence: metric(
            gold.cursor_moves.len(),
            predicted.cursor_moves.len(),
            cursor_pairs.len(),
        ),
        cursor_move_semantics,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_commitment(&report)
        .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?;
    Ok(report)
}

/// Recompute all dimensions and the domain-separated commitment. A changed
/// input graph, configuration, metric or count invalidates the supplied report.
pub fn validate_musicxml_notation_evaluation_report(
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    config: MusicXmlNotationEvaluationConfigV1,
    supplied: &MusicXmlNotationEvaluationReportV1,
) -> Result<bool, MusicXmlNotationEvaluationErrorV1> {
    let replay = evaluate_musicxml_notation_graphs(reference, prediction, config)?;
    Ok(replay == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::musicxml_barline_adapter::parse_musicxml_part_notation_graph;

    fn graph(xml: &str) -> MusicXmlPartNotationGraphV1 {
        parse_musicxml_part_notation_graph(
            xml.as_bytes(),
            "sha256:fixture",
            "score-clock-fixture",
        )
        .unwrap()
        .remove(0)
    }

    fn fixture_xml() -> &'static str {
        r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>16</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
            <note><pitch><step>C</step><alter>0</alter><octave>4</octave></pitch><duration>16</duration><voice>1</voice><staff>1</staff><type>quarter</type><tie type="start"/><notations><tied type="start"/></notations></note>
            <note><pitch><step>D</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice><staff>1</staff><type>quarter</type></note>
            <note><rest/><duration>16</duration><voice>1</voice><staff>1</staff><type>quarter</type></note>
            <note><unpitched><display-step>F</display-step><display-octave>4</display-octave></unpitched><duration>16</duration><voice>2</voice><staff>2</staff><type>quarter</type></note>
            <backup><duration>16</duration></backup>
            <forward><duration>16</duration><voice>1</voice><staff>1</staff></forward>
            <direction><direction-type><dynamics><f/></dynamics></direction-type><sound tempo="96"/></direction>
            <harmony><root><root-step>G</root-step></root><kind>major</kind></harmony>
            <barline location="right"><repeat direction="backward" times="2"/></barline>
        </measure></part></score-partwise>"#
    }

    fn default_config() -> MusicXmlNotationEvaluationConfigV1 {
        MusicXmlNotationEvaluationConfigV1::default()
    }

    #[test]
    fn identical_graphs_produce_perfect_independent_dimensions() {
        let reference = graph(fixture_xml());
        let report = evaluate_musicxml_notation_graphs(
            &reference,
            &reference,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.note_event_presence.f1, Some(1.0));
        assert_eq!(report.written_pitch_spelling.accuracy, Some(1.0));
        assert_eq!(report.rest_semantics.accuracy, Some(1.0));
        assert_eq!(report.unpitched_display.accuracy, Some(1.0));
        assert_eq!(report.duration_presence.accuracy, Some(1.0));
        assert_eq!(report.duration_error.mean_absolute_quarter_beats, Some(0.0));
        assert_eq!(report.written_type_and_dots.accuracy, Some(1.0));
        assert_eq!(report.voice_assignment.accuracy, Some(1.0));
        assert_eq!(report.staff_assignment.accuracy, Some(1.0));
        assert_eq!(report.instrument_assignment.accuracy, None);
        assert_eq!(report.instrument_reference_unassigned_count, 4);
        assert_eq!(report.instrument_unexpected_assignment_count, 0);
        assert_eq!(report.sound_tie_semantics.accuracy, Some(1.0));
        assert_eq!(report.visual_tie_semantics.accuracy, Some(1.0));
        assert_eq!(report.time_modification.accuracy, Some(1.0));
        assert_eq!(report.tuplet_notation.accuracy, Some(1.0));
        assert_eq!(report.full_note_source_subtree.accuracy, Some(1.0));
        assert_eq!(report.note_onset_error.mean_absolute_quarter_beats, Some(0.0));
        assert_eq!(report.auxiliary_event_presence.f1, Some(1.0));
        assert_eq!(report.auxiliary_source_subtree.accuracy, Some(1.0));
        assert_eq!(report.attribute_event_presence.f1, Some(1.0));
        assert_eq!(report.attribute_source_subtree.accuracy, Some(1.0));
        assert_eq!(report.cursor_move_presence.f1, Some(1.0));
        assert_eq!(report.cursor_move_semantics.accuracy, Some(1.0));
        assert_eq!(report.report_sha256.len(), 64);
    }

    #[test]
    fn missing_reference_assignments_are_reported_not_scored_as_correct() {
        let mut reference = graph(fixture_xml());
        let mut prediction = reference.clone();
        reference.notation_events.note_events[0].voice = None;
        reference.notation_events.note_events[0].staff = None;
        prediction.notation_events.note_events[0].instrument_ids = vec!["P1-I1".into()];
        let report = evaluate_musicxml_notation_graphs(
            &reference,
            &prediction,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.voice_assignment.accuracy, Some(1.0));
        assert_eq!(report.voice_reference_unassigned_count, 1);
        assert_eq!(report.voice_unexpected_assignment_count, 0);
        assert_eq!(report.staff_assignment.accuracy, Some(1.0));
        assert_eq!(report.staff_reference_unassigned_count, 1);
        assert_eq!(report.staff_unexpected_assignment_count, 0);
        assert_eq!(report.instrument_assignment.accuracy, None);
        assert_eq!(report.instrument_reference_unassigned_count, 4);
        assert_eq!(report.instrument_unexpected_assignment_count, 1);
    }

    #[test]
    fn wrong_written_pitch_is_not_hidden_by_note_presence_matching() {
        let reference = graph(fixture_xml());
        let prediction_xml = fixture_xml().replace(
            "<step>C</step><alter>0</alter><octave>4</octave>",
            "<step>C</step><alter>1</alter><octave>4</octave>",
        );
        let prediction = graph(&prediction_xml);
        let report = evaluate_musicxml_notation_graphs(
            &reference,
            &prediction,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.note_event_presence.f1, Some(1.0));
        assert_eq!(report.written_pitch_spelling.accuracy, Some(0.5));
    }

    #[test]
    fn extra_note_is_false_positive_without_erasing_matched_note_accuracy() {
        let reference = graph(fixture_xml());
        let prediction_xml = fixture_xml().replace(
            r#"</notations></note>
            <note><pitch><step>D</step>"#,
            r#"</notations></note>
            <note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice><staff>1</staff><type>eighth</type></note>
            <note><pitch><step>D</step>"#,
        );
        let prediction = graph(&prediction_xml);
        let report = evaluate_musicxml_notation_graphs(
            &reference,
            &prediction,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.note_event_presence.true_positives, 4);
        assert_eq!(report.note_event_presence.false_positives, 1);
        assert_eq!(report.note_event_presence.false_negatives, 0);
    }

    #[test]
    fn decimal_pitch_alteration_lexemes_compare_by_exact_numeric_value() {
        let reference = graph(fixture_xml());
        let prediction = graph(&fixture_xml().replace(
            "<step>C</step><alter>0</alter><octave>4</octave>",
            "<step>C</step><alter>0.000</alter><octave>4</octave>",
        ));
        let report = evaluate_musicxml_notation_graphs(
            &reference,
            &prediction,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.written_pitch_spelling.accuracy, Some(1.0));
    }

    #[test]
    fn sound_tie_visual_tie_and_tuplet_semantics_are_independent_dimensions() {
        let reference = graph(fixture_xml());
        let prediction_xml = fixture_xml().replace(
            r#"<tie type="start"/><notations><tied type="start"/></notations>"#,
            r#"<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><tie type="stop"/><notations><tied type="continue"/><tuplet type="start" number="1"/></notations>"#,
        );
        let prediction = graph(&prediction_xml);
        let report = evaluate_musicxml_notation_graphs(
            &reference,
            &prediction,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.note_event_presence.f1, Some(1.0));
        assert_eq!(report.sound_tie_semantics.accuracy, Some(0.75));
        assert_eq!(report.visual_tie_semantics.accuracy, Some(0.75));
        assert_eq!(report.time_modification.accuracy, Some(0.75));
        assert_eq!(report.tuplet_notation.accuracy, Some(0.75));
    }

    #[test]
    fn score_clock_and_part_mismatches_refuse_evaluation() {
        let reference = graph(fixture_xml());
        let mut prediction = reference.clone();
        prediction.map.score_clock_id = "other-clock".into();
        assert_eq!(
            evaluate_musicxml_notation_graphs(&reference, &prediction, default_config()),
            Err(MusicXmlNotationEvaluationErrorV1::ScoreClockMismatch)
        );
        prediction = reference.clone();
        prediction.part_id = "P2".into();
        prediction.notation_events.part_id = "P2".into();
        assert_eq!(
            evaluate_musicxml_notation_graphs(&reference, &prediction, default_config()),
            Err(MusicXmlNotationEvaluationErrorV1::PartIdentityMismatch)
        );
    }

    #[test]
    fn unknown_notation_layer_refuses_a_perfect_score() {
        let reference = graph(fixture_xml());
        let mut prediction = reference.clone();
        prediction.notation_events.status = MusicXmlNotationEventsStatusV1::Unknown {
            reason: "fixture unresolved".into(),
        };
        assert_eq!(
            evaluate_musicxml_notation_graphs(&reference, &prediction, default_config()),
            Err(MusicXmlNotationEvaluationErrorV1::PredictionNotationEventsUnavailable)
        );
    }

    #[test]
    fn duration_error_uses_exact_rational_tolerance() {
        let reference = graph(fixture_xml());
        let prediction_xml = fixture_xml()
            .replacen(
                "<pitch><step>C</step><alter>0</alter><octave>4</octave></pitch><duration>16</duration>",
                "<pitch><step>C</step><alter>0</alter><octave>4</octave></pitch><duration>17</duration>",
                1,
            )
            .replacen(
                "<pitch><step>D</step><octave>4</octave></pitch><duration>16</duration>",
                "<pitch><step>D</step><octave>4</octave></pitch><duration>15</duration>",
                1,
            );
        let prediction = graph(&prediction_xml);
        let mut config = default_config();
        config.duration_tolerance = RationalBeat::new(1, 32).unwrap();
        let report = evaluate_musicxml_notation_graphs(&reference, &prediction, config).unwrap();
        assert!(report.duration_error.mean_absolute_quarter_beats.unwrap() > 0.0);
        assert!(report.duration_error.within_tolerance_rate.unwrap() < 1.0);
        assert_eq!(
            report.note_onset_error.mean_absolute_quarter_beats,
            Some(1.0 / 64.0)
        );
        assert_eq!(
            report.note_onset_error.p95_absolute_quarter_beats,
            Some(1.0 / 16.0)
        );
    }

    #[test]
    fn report_replay_rejects_tampered_metrics_and_changed_graphs() {
        let reference = graph(fixture_xml());
        let prediction = graph(fixture_xml());
        let config = default_config();
        let mut report = evaluate_musicxml_notation_graphs(
            &reference, &prediction, config.clone()
        ).unwrap();
        assert!(validate_musicxml_notation_evaluation_report(
            &reference,&prediction,config.clone(),&report
        ).unwrap());
        report.note_event_presence.false_positives = 1;
        assert!(!validate_musicxml_notation_evaluation_report(
            &reference,&prediction,config.clone(),&report
        ).unwrap());
        let changed = graph(&fixture_xml().replace(
            "<step>D</step><octave>4</octave>",
            "<step>E</step><octave>4</octave>",
        ));
        let original = evaluate_musicxml_notation_graphs(
            &reference, &prediction, config.clone()
        ).unwrap();
        assert!(!validate_musicxml_notation_evaluation_report(
            &reference,&changed,config, &original
        ).unwrap());
    }

    #[test]
    fn noncanonical_and_discontinuous_measure_maps_refuse_evaluation() {
        let reference = graph(fixture_xml());
        let mut prediction = reference.clone();
        prediction.map.measures[0].start = RationalBeat::new(1, 1).unwrap();
        assert!(matches!(
            evaluate_musicxml_notation_graphs(&reference, &prediction, default_config()),
            Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure { .. })
        ));
    }

    #[test]
    fn duplicated_or_reordered_source_ordinals_refuse_evaluation() {
        let reference = graph(fixture_xml());
        let mut prediction = reference.clone();
        prediction.notation_events.note_events[1].source_order =
            prediction.notation_events.note_events[0].source_order;
        assert!(matches!(
            evaluate_musicxml_notation_graphs(&reference, &prediction, default_config()),
            Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure { .. })
        ));
        prediction = reference.clone();
        prediction.notation_events.note_events.swap(0, 1);
        assert!(matches!(
            evaluate_musicxml_notation_graphs(&reference, &prediction, default_config()),
            Err(MusicXmlNotationEvaluationErrorV1::InvalidGraphStructure { .. })
        ));
    }

    #[test]
    fn event_and_candidate_budgets_fail_closed() {
        let reference = graph(fixture_xml());
        let mut config = default_config();
        config.maximum_events_per_part = 1;
        assert!(matches!(
            evaluate_musicxml_notation_graphs(&reference,&reference,config),
            Err(MusicXmlNotationEvaluationErrorV1::EventBudgetExceeded { .. })
        ));
        config = default_config();
        config.maximum_candidate_pairs = 1;
        assert!(matches!(
            evaluate_musicxml_notation_graphs(&reference,&reference,config),
            Err(MusicXmlNotationEvaluationErrorV1::CandidatePairBudgetExceeded { .. })
        ));
    }

    #[test]
    fn deleted_event_is_a_false_negative_and_nonempty_empty_side_has_zero_f1() {
        let reference = graph(fixture_xml());
        let prediction_xml = fixture_xml().replace(
            r#"<note><unpitched><display-step>F</display-step><display-octave>4</display-octave></unpitched><duration>16</duration><voice>2</voice><staff>2</staff><type>quarter</type></note>"#,
            r#"<forward><duration>16</duration><voice>2</voice><staff>2</staff></forward>"#,
        );
        let prediction = graph(&prediction_xml);
        let report = evaluate_musicxml_notation_graphs(
            &reference,
            &prediction,
            default_config(),
        )
        .unwrap();
        assert_eq!(report.note_event_presence.false_negatives, 1);
        assert_eq!(report.note_event_presence.false_positives, 0);
        assert_eq!(report.note_event_presence.true_positives, 3);
        assert_eq!(report.note_event_presence.f1, Some(0.8571428571428571));
        assert_eq!(metric(1, 0, 0).f1, Some(0.0));
        assert_eq!(metric(0, 0, 0).f1, None);
    }

    #[test]
    fn augmenting_path_matching_finds_maximum_cardinality_not_greedy_pairs() {
        // left[0] can match either right; left[1] can only match right[0].
        // The maximum-cardinality assignment is left[0]->right[1], left[1]->right[0].
        let left = [0u8, 1u8];
        let right = [0u8, 1u8];
        let config = default_config();
        let pairs = maximum_cardinality_matching(
            &left,
            &right,
            |l, r| Ok(match l {
                0 => true,
                1 => *r == 0,
                _ => false,
            }),
            &config,
        )
        .unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs, vec![(0, 1), (1, 0)]);
    }
}
