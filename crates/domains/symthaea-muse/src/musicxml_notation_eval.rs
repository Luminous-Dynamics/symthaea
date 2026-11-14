// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound comparison of source-native MusicXML notation events.
//!
//! This evaluator compares already-parsed symbolic graphs. It does not
//! transcribe audio and does not claim MIREX MV2H, MUSTER or OMR-NED parity.
//! Primary event alignment uses event family and exact score-clock onset;
//! duration and notation identity are scored after alignment so duration
//! errors cannot be hidden by the matching rule.

use crate::barline_eval::{MeasureMapStatus, RationalBeat};
use crate::evidence_digest::canonical_json_sha256;
use crate::musicxml_barline_adapter::MusicXmlPartNotationGraphV1;
use crate::musicxml_notation_events::{
    MusicXmlNotationEventKindV1, MusicXmlNotationEventsStatusV1, MusicXmlSourceNodeV1,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const MUSICXML_NOTATION_EVALUATOR_VERSION_V1: &str =
    "muse-musicxml-notation-event-evaluator-v1";
pub const MUSICXML_NOTATION_EVALUATION_SCHEMA_V1: &str =
    "muse-musicxml-notation-evaluation-v1";
pub const MUSICXML_NOTATION_MAX_PART_EVENTS_V1: usize = 20_000;
pub const MUSICXML_NOTATION_MAX_CANDIDATE_EDGES_V1: usize = 2_000_000;
pub const MUSICXML_NOTATION_MAX_PART_BINDINGS_V1: usize = 512;
pub const MUSICXML_NOTATION_MAX_DOCUMENT_EVENTS_V1: usize = 200_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlNotationPartBindingV1 {
    pub reference_part_id: String,
    pub prediction_part_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlNotationEvaluationConfigV1 {
    /// Inclusive onset tolerance in rational quarter-note beats.
    pub onset_tolerance: RationalBeat,
}

impl Default for MusicXmlNotationEvaluationConfigV1 {
    fn default() -> Self {
        Self {
            onset_tolerance: RationalBeat::new(0, 1)
                .expect("zero is a valid rational score coordinate"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlNotationCountsV1 {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

impl MusicXmlNotationCountsV1 {
    fn add_assign(&mut self, other: &Self) {
        self.true_positives = self.true_positives.saturating_add(other.true_positives);
        self.false_positives = self.false_positives.saturating_add(other.false_positives);
        self.false_negatives = self.false_negatives.saturating_add(other.false_negatives);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationMetricV1 {
    pub counts: MusicXmlNotationCountsV1,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationErrorSummaryV1 {
    pub sample_count: usize,
    pub mean_absolute_error_beats: Option<f64>,
    pub maximum_absolute_error_beats: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlNotationEvaluationReportV1 {
    pub schema: String,
    pub evaluator_version: String,
    pub reference_graphs_sha256: String,
    pub prediction_graphs_sha256: String,
    pub config_sha256: String,
    pub config: MusicXmlNotationEvaluationConfigV1,
    pub part_bindings: Vec<MusicXmlNotationPartBindingV1>,
    /// One-to-one event alignment by family and onset only.
    pub event_alignment: MusicXmlNotationMetricV1,
    pub pitched_event_alignment: MusicXmlNotationMetricV1,
    pub rest_event_alignment: MusicXmlNotationMetricV1,
    pub unpitched_event_alignment: MusicXmlNotationMetricV1,
    pub written_pitch_spelling: MusicXmlNotationMetricV1,
    pub rest_semantics: MusicXmlNotationMetricV1,
    pub unpitched_display_pitch: MusicXmlNotationMetricV1,
    /// Exact optional divisions-derived performed duration.
    pub performed_duration: MusicXmlNotationMetricV1,
    pub written_value: MusicXmlNotationMetricV1,
    pub grace_cue_chord_flags: MusicXmlNotationMetricV1,
    pub chord_anchor_relation: MusicXmlNotationMetricV1,
    pub voice_staff_identity: MusicXmlNotationMetricV1,
    pub instrument_references: MusicXmlNotationMetricV1,
    pub instrument_reference_resolution: MusicXmlNotationMetricV1,
    pub performance_directives: MusicXmlNotationMetricV1,
    pub sound_tie_annotations: MusicXmlNotationMetricV1,
    pub visual_tie_annotations: MusicXmlNotationMetricV1,
    pub tuplet_notations: MusicXmlNotationMetricV1,
    pub attribute_events: MusicXmlNotationMetricV1,
    pub attribute_event_values: MusicXmlNotationMetricV1,
    pub auxiliary_events: MusicXmlNotationMetricV1,
    pub auxiliary_event_values: MusicXmlNotationMetricV1,
    pub cursor_moves: MusicXmlNotationMetricV1,
    pub cursor_move_values: MusicXmlNotationMetricV1,
    pub onset_error_beats: MusicXmlNotationErrorSummaryV1,
    pub duration_error_beats: MusicXmlNotationErrorSummaryV1,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicXmlNotationUnavailableV1 {
    ReferenceMeasureStructureUnknown { part_id: String, reason: String },
    ReferenceMeasureStructureUnsupported { part_id: String, reason: String },
    PredictionMeasureStructureUnknown { part_id: String, reason: String },
    PredictionMeasureStructureUnsupported { part_id: String, reason: String },
    ReferenceNotationUnknown { part_id: String, reason: String },
    PredictionNotationUnknown { part_id: String, reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MusicXmlNotationEvaluationOutcomeV1 {
    Evaluated(MusicXmlNotationEvaluationReportV1),
    Unavailable(Vec<MusicXmlNotationUnavailableV1>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicXmlNotationEvaluationErrorV1 {
    InvalidTolerance,
    EmptyPartBindings,
    EmptyPartId,
    DuplicateReferencePartBinding { part_id: String },
    DuplicatePredictionPartBinding { part_id: String },
    MissingReferencePart { part_id: String },
    MissingPredictionPart { part_id: String },
    UnmappedReferenceParts { part_ids: Vec<String> },
    UnmappedPredictionParts { part_ids: Vec<String> },
    ScoreClockMismatch,
    CanonicalizationMismatch,
    NotationCanonicalizationMismatch,
    InvalidGraph { part_id: String, reason: String },
    EventBudgetExceeded { part_id: String, count: usize, maximum: usize },
    DocumentEventBudgetExceeded { count: usize, maximum: usize },
    PartBindingBudgetExceeded { count: usize, maximum: usize },
    CandidateEdgeBudgetExceeded { count: usize, maximum: usize },
    ArithmeticOverflow,
    SerializationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    let difference = match left.compare_exact(right) {
        Ordering::Less => right.checked_sub(left),
        Ordering::Equal | Ordering::Greater => left.checked_sub(right),
    }
    .map_err(|_| MusicXmlNotationEvaluationErrorV1::ArithmeticOverflow)?;
    let lhs = i128::from(difference.numerator())
        .checked_mul(i128::from(tolerance.denominator()))
        .ok_or(MusicXmlNotationEvaluationErrorV1::ArithmeticOverflow)?;
    let rhs = i128::from(tolerance.numerator())
        .checked_mul(i128::from(difference.denominator()))
        .ok_or(MusicXmlNotationEvaluationErrorV1::ArithmeticOverflow)?;
    Ok(lhs <= rhs)
}

fn absolute_beats(left: RationalBeat, right: RationalBeat) -> f64 {
    (left.as_beats() - right.as_beats()).abs()
}

fn validate_tolerance(
    tolerance: RationalBeat,
) -> Result<(), MusicXmlNotationEvaluationErrorV1> {
    if tolerance.numerator() < 0
        || tolerance.denominator() <= 0
        || RationalBeat::new(tolerance.numerator(), tolerance.denominator()).ok()
            != Some(tolerance)
    {
        return Err(MusicXmlNotationEvaluationErrorV1::InvalidTolerance);
    }
    Ok(())
}

fn metric_from_counts(counts: MusicXmlNotationCountsV1) -> MusicXmlNotationMetricV1 {
    let precision_denominator = counts.true_positives + counts.false_positives;
    let recall_denominator = counts.true_positives + counts.false_negatives;
    let precision = (precision_denominator > 0)
        .then_some(counts.true_positives as f64 / precision_denominator as f64);
    let recall = (recall_denominator > 0)
        .then_some(counts.true_positives as f64 / recall_denominator as f64);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        _ => None,
    };
    MusicXmlNotationMetricV1 {
        counts,
        precision,
        recall,
        f1,
    }
}

fn metric_from_correct(
    correct: usize,
    reference_total: usize,
    prediction_total: usize,
) -> MusicXmlNotationMetricV1 {
    metric_from_counts(MusicXmlNotationCountsV1 {
        true_positives: correct,
        false_positives: prediction_total.saturating_sub(correct),
        false_negatives: reference_total.saturating_sub(correct),
    })
}

fn error_summary(values: &[f64]) -> MusicXmlNotationErrorSummaryV1 {
    if values.is_empty() {
        return MusicXmlNotationErrorSummaryV1 {
            sample_count: 0,
            mean_absolute_error_beats: None,
            maximum_absolute_error_beats: None,
        };
    }
    MusicXmlNotationErrorSummaryV1 {
        sample_count: values.len(),
        mean_absolute_error_beats: Some(
            values.iter().copied().sum::<f64>() / values.len() as f64,
        ),
        maximum_absolute_error_beats: Some(
            values.iter().copied().fold(0.0_f64, f64::max),
        ),
    }
}

fn position_within_measure(
    position: RationalBeat,
    measure: &crate::barline_eval::MeasureSpan,
    allow_end: bool,
) -> bool {
    let after_start = position.compare_exact(measure.start) != Ordering::Less;
    let before_end = if allow_end {
        position.compare_exact(measure.end) != Ordering::Greater
    } else {
        position.compare_exact(measure.end) == Ordering::Less
    };
    after_start && before_end
}

fn validate_part_graph(
    graph: &MusicXmlPartNotationGraphV1,
) -> Result<(), MusicXmlNotationEvaluationErrorV1> {
    let part_id = graph.part_id.as_str();
    if graph.part_id.trim().is_empty()
        || graph.map.source_id.trim().is_empty()
        || graph.map.score_clock_id.trim().is_empty()
        || graph.map.canonicalization_version.trim().is_empty()
        || graph.notation_events.part_id != graph.part_id
        || graph.notation_events.schema
            != crate::musicxml_notation_events::MUSICXML_NOTATION_EVENTS_SCHEMA_V1
        || graph.notation_events.canonicalization_version
            != crate::musicxml_notation_events::MUSICXML_NOTATION_EVENTS_CANONICALIZATION_V1
    {
        return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
            part_id: part_id.into(),
            reason: "part identity, source identity, schema or canonicalization is invalid".into(),
        });
    }
    let total_event_count = graph.notation_events.note_events.len()
        .checked_add(graph.notation_events.attribute_events.len())
        .and_then(|count| count.checked_add(graph.notation_events.auxiliary_events.len()))
        .and_then(|count| count.checked_add(graph.notation_events.cursor_moves.len()))
        .ok_or(MusicXmlNotationEvaluationErrorV1::ArithmeticOverflow)?;
    if total_event_count > MUSICXML_NOTATION_MAX_PART_EVENTS_V1 {
        return Err(MusicXmlNotationEvaluationErrorV1::EventBudgetExceeded {
            part_id: part_id.into(),
            count: total_event_count,
            maximum: MUSICXML_NOTATION_MAX_PART_EVENTS_V1,
        });
    }
    if graph.map.measures.is_empty() {
        return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
            part_id: part_id.into(),
            reason: "supported graph contains no explicit measure spans".into(),
        });
    }
    for (index, measure) in graph.map.measures.iter().enumerate() {
        if measure.start.compare_exact(measure.end) != Ordering::Less {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                part_id: part_id.into(),
                reason: format!("measure {index} is empty or reversed"),
            });
        }
        if index > 0
            && graph.map.measures[index - 1].end.compare_exact(measure.start)
                != Ordering::Equal
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                part_id: part_id.into(),
                reason: format!("measure {index} is not contiguous with its predecessor"),
            });
        }
    }

    let mut source_orders = BTreeSet::new();
    for event in &graph.notation_events.note_events {
        if event.part_id != graph.part_id
            || event.source_measure_index >= graph.map.measures.len()
            || !source_orders.insert(event.source_order)
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                part_id: part_id.into(),
                reason: "note event has wrong part, invalid measure locator or duplicate source order".into(),
            });
        }
        let measure = &graph.map.measures[event.source_measure_index];
        if event.score_offset.compare_exact(measure.start) == Ordering::Less
            || event.score_offset.compare_exact(measure.end) != Ordering::Less
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                part_id: part_id.into(),
                reason: "note event lies outside its start-inclusive/end-exclusive measure span".into(),
            });
        }
        if let Some(duration) = event.performed_duration {
            if duration.numerator() <= 0
                || event
                    .score_offset
                    .checked_add(duration)
                    .map(|end| end.compare_exact(measure.end) == Ordering::Greater)
                    .unwrap_or(true)
            {
                return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                    part_id: part_id.into(),
                    reason: "note event duration is invalid or exceeds its measure".into(),
                });
            }
        } else if !event.is_grace {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                part_id: part_id.into(),
                reason: "non-grace note event has no performed duration".into(),
            });
        }
        if event.is_chord_member != event.chord_anchor_source_order.is_some() {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                part_id: part_id.into(),
                reason: "chord-member flag and anchor presence disagree".into(),
            });
        }
    }
    for event in &graph.notation_events.attribute_events {
        if event.part_id != graph.part_id
            || event.measure_index >= graph.map.measures.len()
            || !source_orders.insert(event.source_order)
            || !position_within_measure(
                event.score_offset,
                &graph.map.measures[event.measure_index],
                true,
            )
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                part_id: part_id.into(),
                reason: "attribute event has wrong part, invalid measure position or duplicate source order".into(),
            });
        }
    }
    for event in &graph.notation_events.auxiliary_events {
        if event.part_id != graph.part_id
            || event.measure_index >= graph.map.measures.len()
            || !source_orders.insert(event.source_order)
            || !position_within_measure(
                event.score_offset,
                &graph.map.measures[event.measure_index],
                true,
            )
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                part_id: part_id.into(),
                reason: "auxiliary event has wrong part, invalid measure position or duplicate source order".into(),
            });
        }
    }
    for event in &graph.notation_events.cursor_moves {
        if event.part_id != graph.part_id
            || event.measure_index >= graph.map.measures.len()
            || !source_orders.insert(event.source_order)
            || event.duration.numerator() <= 0
            || !position_within_measure(
                event.score_offset_before,
                &graph.map.measures[event.measure_index],
                true,
            )
            || !position_within_measure(
                event.score_offset_after,
                &graph.map.measures[event.measure_index],
                true,
            )
        {
            return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                part_id: part_id.into(),
                reason: "cursor move has wrong part, invalid measure position, duplicate source order or nonpositive duration".into(),
            });
        }
    }

    let note_orders: BTreeMap<usize, &crate::musicxml_notation_events::MusicXmlNotationEventV1> =
        graph.notation_events.note_events.iter()
            .map(|event| (event.source_order, event))
            .collect();
    for event in &graph.notation_events.note_events {
        if let Some(anchor_order) = event.chord_anchor_source_order {
            let Some(anchor) = note_orders.get(&anchor_order) else {
                return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                    part_id: part_id.into(),
                    reason: "chord anchor does not refer to an event in the same part".into(),
                });
            };
            if anchor.is_chord_member
                || event.source_order <= anchor.source_order
                || event.score_offset != anchor.score_offset
            {
                return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
                    part_id: part_id.into(),
                    reason: "chord anchor is not a preceding base event at the same onset".into(),
                });
            }
        }
    }
    Ok(())
}

/// Builds bounded candidate edges. Both sequences are sorted by exact score
/// position and stable source ordinal, allowing a sliding onset window.
fn note_adjacency(
    reference: &[crate::musicxml_notation_events::MusicXmlNotationEventV1],
    prediction: &[crate::musicxml_notation_events::MusicXmlNotationEventV1],
    tolerance: RationalBeat,
    candidate_budget: &mut usize,
) -> Result<Vec<Vec<usize>>, MusicXmlNotationEvaluationErrorV1> {
    let mut gold_order: Vec<usize> = (0..reference.len()).collect();
    let mut pred_order: Vec<usize> = (0..prediction.len()).collect();
    gold_order.sort_by(|left, right| {
        reference[*left].score_offset.compare_exact(reference[*right].score_offset)
            .then_with(|| reference[*left].source_order.cmp(&reference[*right].source_order))
    });
    pred_order.sort_by(|left, right| {
        prediction[*left].score_offset.compare_exact(prediction[*right].score_offset)
            .then_with(|| prediction[*left].source_order.cmp(&prediction[*right].source_order))
    });

    let mut adjacency = vec![Vec::new(); reference.len()];
    let mut lower = 0usize;
    for gold_index in gold_order {
        let gold = &reference[gold_index];
        while lower < pred_order.len() {
            let pred = &prediction[pred_order[lower]];
            if pred.score_offset.compare_exact(gold.score_offset) == Ordering::Less
                && !exact_within(pred.score_offset, gold.score_offset, tolerance)?
            {
                lower += 1;
            } else {
                break;
            }
        }
        let mut scan = lower;
        while scan < pred_order.len() {
            let pred_index = pred_order[scan];
            let pred = &prediction[pred_index];
            let within = exact_within(pred.score_offset, gold.score_offset, tolerance)?;
            if pred.score_offset.compare_exact(gold.score_offset) == Ordering::Greater && !within {
                break;
            }
            if within && event_family(&pred.kind) == event_family(&gold.kind) {
                *candidate_budget = candidate_budget
                    .checked_add(1)
                    .ok_or(MusicXmlNotationEvaluationErrorV1::ArithmeticOverflow)?;
                if *candidate_budget > MUSICXML_NOTATION_MAX_CANDIDATE_EDGES_V1 {
                    return Err(MusicXmlNotationEvaluationErrorV1::CandidateEdgeBudgetExceeded {
                        count: *candidate_budget,
                        maximum: MUSICXML_NOTATION_MAX_CANDIDATE_EDGES_V1,
                    });
                }
                adjacency[gold_index].push(pred_index);
            }
            scan += 1;
        }
    }
    Ok(adjacency)
}

fn keyed_adjacency(
    reference: &[(RationalBeat, String)],
    prediction: &[(RationalBeat, String)],
    tolerance: RationalBeat,
    candidate_budget: &mut usize,
) -> Result<Vec<Vec<usize>>, MusicXmlNotationEvaluationErrorV1> {
    let mut gold_order: Vec<usize> = (0..reference.len()).collect();
    let mut pred_order: Vec<usize> = (0..prediction.len()).collect();
    gold_order.sort_by(|left, right| {
        reference[*left].0.compare_exact(reference[*right].0).then_with(|| left.cmp(right))
    });
    pred_order.sort_by(|left, right| {
        prediction[*left].0.compare_exact(prediction[*right].0).then_with(|| left.cmp(right))
    });

    let mut adjacency = vec![Vec::new(); reference.len()];
    let mut lower = 0usize;
    for gold_index in gold_order {
        let (gold_offset, gold_key) = &reference[gold_index];
        while lower < pred_order.len() {
            let (pred_offset, _) = &prediction[pred_order[lower]];
            if pred_offset.compare_exact(*gold_offset) == Ordering::Less
                && !exact_within(*pred_offset, *gold_offset, tolerance)?
            {
                lower += 1;
            } else {
                break;
            }
        }
        let mut scan = lower;
        while scan < pred_order.len() {
            let pred_index = pred_order[scan];
            let (pred_offset, pred_key) = &prediction[pred_index];
            let within = exact_within(*pred_offset, *gold_offset, tolerance)?;
            if pred_offset.compare_exact(*gold_offset) == Ordering::Greater && !within {
                break;
            }
            if within && pred_key == gold_key {
                *candidate_budget = candidate_budget
                    .checked_add(1)
                    .ok_or(MusicXmlNotationEvaluationErrorV1::ArithmeticOverflow)?;
                if *candidate_budget > MUSICXML_NOTATION_MAX_CANDIDATE_EDGES_V1 {
                    return Err(MusicXmlNotationEvaluationErrorV1::CandidateEdgeBudgetExceeded {
                        count: *candidate_budget,
                        maximum: MUSICXML_NOTATION_MAX_CANDIDATE_EDGES_V1,
                    });
                }
                adjacency[gold_index].push(pred_index);
            }
            scan += 1;
        }
    }
    Ok(adjacency)
}

/// Deterministic Hopcroft–Karp. The layered DFS is iterative so deep
/// augmenting paths cannot overflow the process stack.
fn maximum_matching(
    adjacency: &[Vec<usize>],
    prediction_count: usize,
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut pair_gold = vec![None; adjacency.len()];
    let mut pair_pred = vec![None; prediction_count];
    let mut distance = vec![usize::MAX; adjacency.len()];

    while let Some(shortest) = matching_bfs(adjacency, &pair_gold, &pair_pred, &mut distance) {
        let mut next_edge = vec![0usize; adjacency.len()];
        for start in 0..adjacency.len() {
            if pair_gold[start].is_none() {
                let _ = matching_dfs_iterative(
                    start,
                    adjacency,
                    &mut pair_gold,
                    &mut pair_pred,
                    &mut distance,
                    &mut next_edge,
                    shortest,
                );
            }
        }
    }
    (pair_gold, pair_pred)
}

fn matching_bfs(
    adjacency: &[Vec<usize>],
    pair_gold: &[Option<usize>],
    pair_pred: &[Option<usize>],
    distance: &mut [usize],
) -> Option<usize> {
    let mut queue = VecDeque::new();
    for index in 0..pair_gold.len() {
        if pair_gold[index].is_none() {
            distance[index] = 0;
            queue.push_back(index);
        } else {
            distance[index] = usize::MAX;
        }
    }
    let mut shortest = usize::MAX;
    while let Some(gold_index) = queue.pop_front() {
        let next_depth = distance[gold_index].saturating_add(1);
        if next_depth > shortest {
            continue;
        }
        for pred_index in adjacency[gold_index].iter().copied() {
            if let Some(next_gold) = pair_pred[pred_index] {
                if distance[next_gold] == usize::MAX && next_depth < shortest {
                    distance[next_gold] = next_depth;
                    queue.push_back(next_gold);
                }
            } else {
                shortest = shortest.min(next_depth);
            }
        }
    }
    (shortest != usize::MAX).then_some(shortest)
}

fn matching_dfs_iterative(
    start: usize,
    adjacency: &[Vec<usize>],
    pair_gold: &mut [Option<usize>],
    pair_pred: &mut [Option<usize>],
    distance: &mut [usize],
    next_edge: &mut [usize],
    shortest_depth: usize,
) -> bool {
    let mut stack = vec![start];
    // path_predictions[i] links stack[i] to stack[i + 1].
    let mut path_predictions = Vec::new();

    while let Some(&gold_index) = stack.last() {
        let mut descended = false;
        while next_edge[gold_index] < adjacency[gold_index].len() {
            let pred_index = adjacency[gold_index][next_edge[gold_index]];
            next_edge[gold_index] += 1;
            match pair_pred[pred_index] {
                None if distance[gold_index].saturating_add(1) == shortest_depth => {
                    for (index, matched_pred) in path_predictions.iter().copied().enumerate() {
                        let path_gold = stack[index];
                        pair_gold[path_gold] = Some(matched_pred);
                        pair_pred[matched_pred] = Some(path_gold);
                    }
                    pair_gold[gold_index] = Some(pred_index);
                    pair_pred[pred_index] = Some(gold_index);
                    return true;
                }
                Some(next_gold)
                    if distance[next_gold] == distance[gold_index].saturating_add(1) =>
                {
                    path_predictions.push(pred_index);
                    stack.push(next_gold);
                    descended = true;
                    break;
                }
                _ => {}
            }
        }
        if !descended {
            distance[gold_index] = usize::MAX;
            stack.pop();
            if !path_predictions.is_empty() {
                path_predictions.pop();
            }
        }
    }
    false
}

fn source_node_identity(node: &MusicXmlSourceNodeV1) -> String {
    format!("{}\u{1f}{}", node.namespace_uri.as_deref().unwrap_or(""), node.element)
}

fn event_counts(
    reference_count: usize,
    prediction_count: usize,
    matched_count: usize,
) -> MusicXmlNotationMetricV1 {
    metric_from_counts(MusicXmlNotationCountsV1 {
        true_positives: matched_count,
        false_positives: prediction_count.saturating_sub(matched_count),
        false_negatives: reference_count.saturating_sub(matched_count),
    })
}

fn family_count(
    events: &[crate::musicxml_notation_events::MusicXmlNotationEventV1],
    family: EventFamily,
) -> usize {
    events.iter().filter(|event| event_family(&event.kind) == family).count()
}

fn duration_count(
    events: &[crate::musicxml_notation_events::MusicXmlNotationEventV1],
) -> usize {
    events.iter().filter(|event| event.performed_duration.is_some()).count()
}

#[derive(Default)]
struct Accumulator {
    event_alignment: MusicXmlNotationCountsV1,
    pitched_event_alignment: MusicXmlNotationCountsV1,
    rest_event_alignment: MusicXmlNotationCountsV1,
    unpitched_event_alignment: MusicXmlNotationCountsV1,
    written_pitch_spelling: MusicXmlNotationCountsV1,
    rest_semantics: MusicXmlNotationCountsV1,
    unpitched_display_pitch: MusicXmlNotationCountsV1,
    performed_duration: MusicXmlNotationCountsV1,
    written_value: MusicXmlNotationCountsV1,
    grace_cue_chord_flags: MusicXmlNotationCountsV1,
    chord_anchor_relation: MusicXmlNotationCountsV1,
    voice_staff_identity: MusicXmlNotationCountsV1,
    instrument_references: MusicXmlNotationCountsV1,
    instrument_reference_resolution: MusicXmlNotationCountsV1,
    performance_directives: MusicXmlNotationCountsV1,
    sound_tie_annotations: MusicXmlNotationCountsV1,
    visual_tie_annotations: MusicXmlNotationCountsV1,
    tuplet_notations: MusicXmlNotationCountsV1,
    attribute_events: MusicXmlNotationCountsV1,
    attribute_event_values: MusicXmlNotationCountsV1,
    auxiliary_events: MusicXmlNotationCountsV1,
    auxiliary_event_values: MusicXmlNotationCountsV1,
    cursor_moves: MusicXmlNotationCountsV1,
    cursor_move_values: MusicXmlNotationCountsV1,
    onset_errors: Vec<f64>,
    duration_errors: Vec<f64>,
}

impl Accumulator {
    fn into_report(
        self,
        reference: &[MusicXmlPartNotationGraphV1],
        prediction: &[MusicXmlPartNotationGraphV1],
        config: MusicXmlNotationEvaluationConfigV1,
        bindings: Vec<MusicXmlNotationPartBindingV1>,
    ) -> Result<MusicXmlNotationEvaluationReportV1, MusicXmlNotationEvaluationErrorV1> {
        let mut report = MusicXmlNotationEvaluationReportV1 {
            schema: MUSICXML_NOTATION_EVALUATION_SCHEMA_V1.into(),
            evaluator_version: MUSICXML_NOTATION_EVALUATOR_VERSION_V1.into(),
            reference_graphs_sha256: canonical_json_sha256(reference)
                .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?,
            prediction_graphs_sha256: canonical_json_sha256(prediction)
                .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?,
            config_sha256: canonical_json_sha256(&(&config, &bindings))
                .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?,
            config,
            part_bindings: bindings,
            event_alignment: metric_from_counts(self.event_alignment),
            pitched_event_alignment: metric_from_counts(self.pitched_event_alignment),
            rest_event_alignment: metric_from_counts(self.rest_event_alignment),
            unpitched_event_alignment: metric_from_counts(self.unpitched_event_alignment),
            written_pitch_spelling: metric_from_counts(self.written_pitch_spelling),
            rest_semantics: metric_from_counts(self.rest_semantics),
            unpitched_display_pitch: metric_from_counts(self.unpitched_display_pitch),
            performed_duration: metric_from_counts(self.performed_duration),
            written_value: metric_from_counts(self.written_value),
            grace_cue_chord_flags: metric_from_counts(self.grace_cue_chord_flags),
            chord_anchor_relation: metric_from_counts(self.chord_anchor_relation),
            voice_staff_identity: metric_from_counts(self.voice_staff_identity),
            instrument_references: metric_from_counts(self.instrument_references),
            sound_tie_annotations: metric_from_counts(self.sound_tie_annotations),
            visual_tie_annotations: metric_from_counts(self.visual_tie_annotations),
            tuplet_notations: metric_from_counts(self.tuplet_notations),
            attribute_events: metric_from_counts(self.attribute_events),
            attribute_event_values: metric_from_counts(self.attribute_event_values),
            auxiliary_events: metric_from_counts(self.auxiliary_events),
            auxiliary_event_values: metric_from_counts(self.auxiliary_event_values),
            cursor_moves: metric_from_counts(self.cursor_moves),
            cursor_move_values: metric_from_counts(self.cursor_move_values),
            onset_error_beats: error_summary(&self.onset_errors),
            duration_error_beats: error_summary(&self.duration_errors),
            report_sha256: String::new(),
        };
        report.report_sha256 = musicxml_notation_report_commitment(&report)
            .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?;
        Ok(report)
    }
}

fn instrument_ids_resolve(
    graph: &MusicXmlPartNotationGraphV1,
    instrument_ids: &[String],
) -> bool {
    let Some(metadata) = graph.part_metadata.as_ref() else {
        return false;
    };
    instrument_ids
        .iter()
        .all(|instrument_id| metadata.score_instrument_ids.contains(instrument_id))
}

fn compare_graph_pair(
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    tolerance: RationalBeat,
    candidate_budget: &mut usize,
    accum: &mut Accumulator,
) -> Result<(), MusicXmlNotationEvaluationErrorV1> {
    let gold = &reference.notation_events;
    let predicted = &prediction.notation_events;
    let adjacency = note_adjacency(
        &gold.note_events,
        &predicted.note_events,
        tolerance,
        candidate_budget,
    )?;
    let (pair_gold, _) = maximum_matching(&adjacency, predicted.note_events.len());
    let matched_pairs: Vec<(usize, usize)> = pair_gold.iter().enumerate()
        .filter_map(|(gold_index, pred_index)| pred_index.map(|pred_index| (gold_index, pred_index)))
        .collect();

    let mut anchor_mapping = BTreeMap::new();
    let mut onset_errors = Vec::new();
    let mut duration_errors = Vec::new();
    let mut matched_by_family = BTreeMap::from([
        (EventFamily::Pitched, 0usize),
        (EventFamily::Rest, 0usize),
        (EventFamily::Unpitched, 0usize),
    ]);
    let mut correct_spelling = 0;
    let mut correct_rest = 0;
    let mut correct_unpitched = 0;
    let mut correct_duration = 0;
    let mut correct_written = 0;
    let mut correct_flags = 0;
    let mut correct_voice_staff = 0;
    let mut correct_instruments = 0;
    let mut correct_instrument_resolution = 0;
    let mut correct_performance_directives = 0;
    let mut correct_sound_ties = 0;
    let mut correct_visual_ties = 0;
    let mut correct_tuplets = 0;

    for (gold_index, pred_index) in &matched_pairs {
        let gold_event = &gold.note_events[*gold_index];
        let pred_event = &predicted.note_events[*pred_index];
        *matched_by_family.entry(event_family(&gold_event.kind)).or_default() += 1;
        if !gold_event.is_chord_member && !pred_event.is_chord_member {
            anchor_mapping.insert(gold_event.source_order, pred_event.source_order);
        }
        onset_errors.push(absolute_beats(gold_event.score_offset, pred_event.score_offset));
        if let (Some(gold_duration), Some(pred_duration)) =
            (gold_event.performed_duration, pred_event.performed_duration)
        {
            duration_errors.push(absolute_beats(gold_duration, pred_duration));
            if gold_duration == pred_duration {
                correct_duration += 1;
            }
        }

        match (&gold_event.kind, &pred_event.kind) {
            (MusicXmlNotationEventKindV1::Pitched(gold_pitch),
                MusicXmlNotationEventKindV1::Pitched(pred_pitch)) => {
                if gold_pitch == pred_pitch { correct_spelling += 1; }
            }
            (MusicXmlNotationEventKindV1::Rest { measure: gm, display: gd },
                MusicXmlNotationEventKindV1::Rest { measure: pm, display: pd }) => {
                if gm == pm && gd == pd { correct_rest += 1; }
            }
            (MusicXmlNotationEventKindV1::Unpitched { display: gd },
                MusicXmlNotationEventKindV1::Unpitched { display: pd }) => {
                if gd == pd { correct_unpitched += 1; }
            }
            _ => {}
        }
        if gold_event.written_type == pred_event.written_type
            && gold_event.dot_count == pred_event.dot_count
            && gold_event.time_modification == pred_event.time_modification
        { correct_written += 1; }
        if gold_event.is_grace == pred_event.is_grace
            && gold_event.is_cue == pred_event.is_cue
            && gold_event.is_chord_member == pred_event.is_chord_member
        { correct_flags += 1; }
        if gold_event.voice == pred_event.voice && gold_event.staff == pred_event.staff
        { correct_voice_staff += 1; }
        if gold_event.instrument_ids == pred_event.instrument_ids {
            correct_instruments += 1;
            if !gold_event.instrument_ids.is_empty()
                && instrument_ids_resolve(reference, &gold_event.instrument_ids)
                && instrument_ids_resolve(prediction, &pred_event.instrument_ids)
            {
                correct_instrument_resolution += 1;
            }
        }
        if gold_event.time_only == pred_event.time_only
            && gold_event.attack_divisions == pred_event.attack_divisions
            && gold_event.release_divisions == pred_event.release_divisions
        {
            correct_performance_directives += 1;
        }
        if gold_event.sound_ties == pred_event.sound_ties
        { correct_sound_ties += 1; }
        if gold_event.visual_ties == pred_event.visual_ties
        { correct_visual_ties += 1; }
        if gold_event.tuplet_notations == pred_event.tuplet_notations
        { correct_tuplets += 1; }
    }

    let mut correct_chord_relations = 0;
    for (gold_index, pred_index) in &matched_pairs {
        let gold_event = &gold.note_events[*gold_index];
        let pred_event = &predicted.note_events[*pred_index];
        let anchors_match = match (gold_event.chord_anchor_source_order, pred_event.chord_anchor_source_order) {
            (None, None) => true,
            (Some(gold_anchor), Some(pred_anchor)) =>
                anchor_mapping.get(&gold_anchor) == Some(&pred_anchor),
            _ => false,
        };
        if gold_event.is_chord_member == pred_event.is_chord_member && anchors_match {
            correct_chord_relations += 1;
        }
    }

    let gold_total = gold.note_events.len();
    let pred_total = predicted.note_events.len();
    accum.event_alignment.add_assign(&event_counts(gold_total, pred_total, matched_pairs.len()));
    let family_metrics = [
        (
            EventFamily::Pitched,
            family_count(&gold.note_events, EventFamily::Pitched),
            family_count(&predicted.note_events, EventFamily::Pitched),
            matched_by_family[&EventFamily::Pitched],
        ),
        (
            EventFamily::Rest,
            family_count(&gold.note_events, EventFamily::Rest),
            family_count(&predicted.note_events, EventFamily::Rest),
            matched_by_family[&EventFamily::Rest],
        ),
        (
            EventFamily::Unpitched,
            family_count(&gold.note_events, EventFamily::Unpitched),
            family_count(&predicted.note_events, EventFamily::Unpitched),
            matched_by_family[&EventFamily::Unpitched],
        ),
    ];
    for (family, gold_count, pred_count, matched) in family_metrics {
        let metric = event_counts(gold_count, pred_count, matched);
        match family {
            EventFamily::Pitched => accum.pitched_event_alignment.add_assign(&metric.counts),
            EventFamily::Rest => accum.rest_event_alignment.add_assign(&metric.counts),
            EventFamily::Unpitched => accum.unpitched_event_alignment.add_assign(&metric.counts),
        }
    }

    accum.written_pitch_spelling.add_assign(&metric_from_correct(
        correct_spelling,
        family_count(&gold.note_events, EventFamily::Pitched),
        family_count(&predicted.note_events, EventFamily::Pitched),
    ).counts);
    accum.rest_semantics.add_assign(&metric_from_correct(
        correct_rest,
        family_count(&gold.note_events, EventFamily::Rest),
        family_count(&predicted.note_events, EventFamily::Rest),
    ).counts);
    accum.unpitched_display_pitch.add_assign(&metric_from_correct(
        correct_unpitched,
        family_count(&gold.note_events, EventFamily::Unpitched),
        family_count(&predicted.note_events, EventFamily::Unpitched),
    ).counts);
    accum.performed_duration.add_assign(&metric_from_correct(
        correct_duration,
        duration_count(&gold.note_events),
        duration_count(&predicted.note_events),
    ).counts);
    accum.written_value.add_assign(&metric_from_correct(correct_written, gold_total, pred_total).counts);
    accum.grace_cue_chord_flags.add_assign(&metric_from_correct(correct_flags, gold_total, pred_total).counts);
    accum.chord_anchor_relation.add_assign(&metric_from_correct(correct_chord_relations, gold_total, pred_total).counts);
    accum.voice_staff_identity.add_assign(&metric_from_correct(correct_voice_staff, gold_total, pred_total).counts);
    accum.instrument_references.add_assign(&metric_from_correct(correct_instruments, gold_total, pred_total).counts);
    let gold_instrumented = gold.note_events.iter().filter(|event| !event.instrument_ids.is_empty()).count();
    let pred_instrumented = predicted.note_events.iter().filter(|event| !event.instrument_ids.is_empty()).count();
    accum.instrument_reference_resolution.add_assign(&metric_from_correct(
        correct_instrument_resolution, gold_instrumented, pred_instrumented
    ).counts);
    accum.performance_directives.add_assign(&metric_from_correct(
        correct_performance_directives, gold_total, pred_total
    ).counts);
    accum.sound_tie_annotations.add_assign(&metric_from_correct(correct_sound_ties, gold_total, pred_total).counts);
    accum.visual_tie_annotations.add_assign(&metric_from_correct(correct_visual_ties, gold_total, pred_total).counts);
    accum.tuplet_notations.add_assign(&metric_from_correct(correct_tuplets, gold_total, pred_total).counts);
    accum.onset_errors.extend(onset_errors);
    accum.duration_errors.extend(duration_errors);

    let gold_attributes: Vec<_> = gold.attribute_events.iter()
        .map(|event| (event.score_offset, source_node_identity(&event.source))).collect();
    let pred_attributes: Vec<_> = predicted.attribute_events.iter()
        .map(|event| (event.score_offset, source_node_identity(&event.source))).collect();
    let adjacency = keyed_adjacency(&gold_attributes, &pred_attributes, tolerance, candidate_budget)?;
    let (pairs, _) = maximum_matching(&adjacency, pred_attributes.len());
    let mut exact = 0;
    for (gold_index, pred_index) in pairs.iter().enumerate() {
        if let Some(pred_index) = pred_index {
            if gold.attribute_events[gold_index].source == predicted.attribute_events[*pred_index].source {
                exact += 1;
            }
        }
    }
    accum.attribute_events.add_assign(&event_counts(
        gold_attributes.len(), pred_attributes.len(), pairs.iter().filter(|p| p.is_some()).count()
    ).counts);
    accum.attribute_event_values.add_assign(&metric_from_correct(
        exact, gold_attributes.len(), pred_attributes.len()
    ).counts);

    let gold_aux: Vec<_> = gold.auxiliary_events.iter()
        .map(|event| (event.score_offset, source_node_identity(&event.source))).collect();
    let pred_aux: Vec<_> = predicted.auxiliary_events.iter()
        .map(|event| (event.score_offset, source_node_identity(&event.source))).collect();
    let adjacency = keyed_adjacency(&gold_aux, &pred_aux, tolerance, candidate_budget)?;
    let (pairs, _) = maximum_matching(&adjacency, pred_aux.len());
    let mut exact = 0;
    for (gold_index, pred_index) in pairs.iter().enumerate() {
        if let Some(pred_index) = pred_index {
            if gold.auxiliary_events[gold_index].source == predicted.auxiliary_events[*pred_index].source {
                exact += 1;
            }
        }
    }
    accum.auxiliary_events.add_assign(&event_counts(
        gold_aux.len(), pred_aux.len(), pairs.iter().filter(|p| p.is_some()).count()
    ).counts);
    accum.auxiliary_event_values.add_assign(&metric_from_correct(
        exact, gold_aux.len(), pred_aux.len()
    ).counts);

    let gold_cursor: Vec<_> = gold.cursor_moves.iter()
        .map(|event| (event.score_offset_before, event.direction.clone())).collect();
    let pred_cursor: Vec<_> = predicted.cursor_moves.iter()
        .map(|event| (event.score_offset_before, event.direction.clone())).collect();
    let adjacency = keyed_adjacency(&gold_cursor, &pred_cursor, tolerance, candidate_budget)?;
    let (pairs, _) = maximum_matching(&adjacency, pred_cursor.len());
    let mut exact = 0;
    for (gold_index, pred_index) in pairs.iter().enumerate() {
        if let Some(pred_index) = pred_index {
            let g = &gold.cursor_moves[gold_index];
            let p = &predicted.cursor_moves[*pred_index];
            if g.duration == p.duration && g.voice == p.voice && g.staff == p.staff && g.source == p.source {
                exact += 1;
            }
        }
    }
    accum.cursor_moves.add_assign(&event_counts(
        gold_cursor.len(), pred_cursor.len(), pairs.iter().filter(|p| p.is_some()).count()
    ).counts);
    accum.cursor_move_values.add_assign(&metric_from_correct(
        exact, gold_cursor.len(), pred_cursor.len()
    ).counts);
    Ok(())
}

pub fn musicxml_notation_report_commitment(
    report: &MusicXmlNotationEvaluationReportV1,
) -> Result<String, serde_json::Error> {
    let mut payload = report.clone();
    payload.report_sha256.clear();
    canonical_json_sha256(&payload)
}

pub fn evaluate_musicxml_notation_graphs(
    reference: &[MusicXmlPartNotationGraphV1],
    prediction: &[MusicXmlPartNotationGraphV1],
    bindings: &[MusicXmlNotationPartBindingV1],
    config: MusicXmlNotationEvaluationConfigV1,
) -> Result<MusicXmlNotationEvaluationOutcomeV1, MusicXmlNotationEvaluationErrorV1> {
    validate_tolerance(config.onset_tolerance)?;
    if bindings.is_empty() {
        return Err(MusicXmlNotationEvaluationErrorV1::EmptyPartBindings);
    }
    if bindings.len() > MUSICXML_NOTATION_MAX_PART_BINDINGS_V1 {
        return Err(MusicXmlNotationEvaluationErrorV1::PartBindingBudgetExceeded {
            count: bindings.len(),
            maximum: MUSICXML_NOTATION_MAX_PART_BINDINGS_V1,
        });
    }

    let reference_by_id: BTreeMap<_, _> =
        reference.iter().map(|graph| (graph.part_id.as_str(), graph)).collect();
    let prediction_by_id: BTreeMap<_, _> =
        prediction.iter().map(|graph| (graph.part_id.as_str(), graph)).collect();
    if reference_by_id.len() != reference.len() || prediction_by_id.len() != prediction.len() {
        return Err(MusicXmlNotationEvaluationErrorV1::InvalidGraph {
            part_id: "<document>".into(),
            reason: "duplicate part IDs in graph set".into(),
        });
    }
    let total_document_events = reference.iter().chain(prediction.iter()).try_fold(
        0usize,
        |total, graph| {
            let events = graph.notation_events.note_events.len()
                .checked_add(graph.notation_events.attribute_events.len())?
                .checked_add(graph.notation_events.auxiliary_events.len())?
                .checked_add(graph.notation_events.cursor_moves.len())?;
            total.checked_add(events)
        },
    ).ok_or(MusicXmlNotationEvaluationErrorV1::ArithmeticOverflow)?;
    if total_document_events > MUSICXML_NOTATION_MAX_DOCUMENT_EVENTS_V1 {
        return Err(MusicXmlNotationEvaluationErrorV1::DocumentEventBudgetExceeded {
            count: total_document_events,
            maximum: MUSICXML_NOTATION_MAX_DOCUMENT_EVENTS_V1,
        });
    }

    let mut seen_gold = BTreeSet::new();
    let mut seen_pred = BTreeSet::new();
    let mut unavailable = Vec::new();
    let mut accumulator = Accumulator::default();
    let mut candidate_budget = 0usize;
    let mut shared_clock: Option<String> = None;
    let mut shared_map_version: Option<String> = None;
    let mut shared_event_version: Option<String> = None;

    for binding in bindings {
        if binding.reference_part_id.trim().is_empty() || binding.prediction_part_id.trim().is_empty() {
            return Err(MusicXmlNotationEvaluationErrorV1::EmptyPartId);
        }
        if !seen_gold.insert(binding.reference_part_id.as_str()) {
            return Err(MusicXmlNotationEvaluationErrorV1::DuplicateReferencePartBinding {
                part_id: binding.reference_part_id.clone(),
            });
        }
        if !seen_pred.insert(binding.prediction_part_id.as_str()) {
            return Err(MusicXmlNotationEvaluationErrorV1::DuplicatePredictionPartBinding {
                part_id: binding.prediction_part_id.clone(),
            });
        }
        let Some(gold) = reference_by_id.get(binding.reference_part_id.as_str()).copied() else {
            return Err(MusicXmlNotationEvaluationErrorV1::MissingReferencePart {
                part_id: binding.reference_part_id.clone(),
            });
        };
        let Some(pred) = prediction_by_id.get(binding.prediction_part_id.as_str()).copied() else {
            return Err(MusicXmlNotationEvaluationErrorV1::MissingPredictionPart {
                part_id: binding.prediction_part_id.clone(),
            });
        };

        match &gold.map.status {
            MeasureMapStatus::Unknown { reason } => unavailable.push(
                MusicXmlNotationUnavailableV1::ReferenceMeasureStructureUnknown {
                    part_id: gold.part_id.clone(), reason: reason.clone()
                }),
            MeasureMapStatus::Unsupported { reason } => unavailable.push(
                MusicXmlNotationUnavailableV1::ReferenceMeasureStructureUnsupported {
                    part_id: gold.part_id.clone(), reason: reason.clone()
                }),
            MeasureMapStatus::Supported => {}
        }
        match &pred.map.status {
            MeasureMapStatus::Unknown { reason } => unavailable.push(
                MusicXmlNotationUnavailableV1::PredictionMeasureStructureUnknown {
                    part_id: pred.part_id.clone(), reason: reason.clone()
                }),
            MeasureMapStatus::Unsupported { reason } => unavailable.push(
                MusicXmlNotationUnavailableV1::PredictionMeasureStructureUnsupported {
                    part_id: pred.part_id.clone(), reason: reason.clone()
                }),
            MeasureMapStatus::Supported => {}
        }
        match &gold.notation_events.status {
            MusicXmlNotationEventsStatusV1::Unknown { reason } => unavailable.push(
                MusicXmlNotationUnavailableV1::ReferenceNotationUnknown {
                    part_id: gold.part_id.clone(), reason: reason.clone()
                }),
            MusicXmlNotationEventsStatusV1::Supported => {}
        }
        match &pred.notation_events.status {
            MusicXmlNotationEventsStatusV1::Unknown { reason } => unavailable.push(
                MusicXmlNotationUnavailableV1::PredictionNotationUnknown {
                    part_id: pred.part_id.clone(), reason: reason.clone()
                }),
            MusicXmlNotationEventsStatusV1::Supported => {}
        }

        if let Some(clock) = &shared_clock {
            if clock != &gold.map.score_clock_id || clock != &pred.map.score_clock_id {
                return Err(MusicXmlNotationEvaluationErrorV1::ScoreClockMismatch);
            }
        } else {
            if gold.map.score_clock_id != pred.map.score_clock_id {
                return Err(MusicXmlNotationEvaluationErrorV1::ScoreClockMismatch);
            }
            shared_clock = Some(gold.map.score_clock_id.clone());
        }
        if gold.map.canonicalization_version != pred.map.canonicalization_version {
            return Err(MusicXmlNotationEvaluationErrorV1::CanonicalizationMismatch);
        }
        if gold.notation_events.canonicalization_version != pred.notation_events.canonicalization_version {
            return Err(MusicXmlNotationEvaluationErrorV1::NotationCanonicalizationMismatch);
        }
        if let Some(version) = &shared_map_version {
            if version != &gold.map.canonicalization_version {
                return Err(MusicXmlNotationEvaluationErrorV1::CanonicalizationMismatch);
            }
        } else {
            shared_map_version = Some(gold.map.canonicalization_version.clone());
        }
        if let Some(version) = &shared_event_version {
            if version != &gold.notation_events.canonicalization_version {
                return Err(MusicXmlNotationEvaluationErrorV1::NotationCanonicalizationMismatch);
            }
        } else {
            shared_event_version = Some(gold.notation_events.canonicalization_version.clone());
        }
    }

    if !unavailable.is_empty() {
        return Ok(MusicXmlNotationEvaluationOutcomeV1::Unavailable(unavailable));
    }
    if seen_gold.len() != reference.len() {
        return Err(MusicXmlNotationEvaluationErrorV1::UnmappedReferenceParts {
            part_ids: reference.iter()
                .filter(|graph| !seen_gold.contains(graph.part_id.as_str()))
                .map(|graph| graph.part_id.clone()).collect(),
        });
    }
    if seen_pred.len() != prediction.len() {
        return Err(MusicXmlNotationEvaluationErrorV1::UnmappedPredictionParts {
            part_ids: prediction.iter()
                .filter(|graph| !seen_pred.contains(graph.part_id.as_str()))
                .map(|graph| graph.part_id.clone()).collect(),
        });
    }

    for binding in bindings {
        let gold = reference_by_id[binding.reference_part_id.as_str()];
        let pred = prediction_by_id[binding.prediction_part_id.as_str()];
        validate_part_graph(gold)?;
        validate_part_graph(pred)?;
        compare_graph_pair(gold, pred, config.onset_tolerance, &mut candidate_budget, &mut accumulator)?;
    }

    let mut report = MusicXmlNotationEvaluationReportV1 {
        schema: MUSICXML_NOTATION_EVALUATION_SCHEMA_V1.into(),
        evaluator_version: MUSICXML_NOTATION_EVALUATOR_VERSION_V1.into(),
        reference_graphs_sha256: canonical_json_sha256(reference)
            .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?,
        prediction_graphs_sha256: canonical_json_sha256(prediction)
            .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?,
        config_sha256: canonical_json_sha256(&(&config, bindings))
            .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?,
        config,
        part_bindings: bindings.to_vec(),
        event_alignment: metric_from_counts(accumulator.event_alignment),
        pitched_event_alignment: metric_from_counts(accumulator.pitched_event_alignment),
        rest_event_alignment: metric_from_counts(accumulator.rest_event_alignment),
        unpitched_event_alignment: metric_from_counts(accumulator.unpitched_event_alignment),
        written_pitch_spelling: metric_from_counts(accumulator.written_pitch_spelling),
        rest_semantics: metric_from_counts(accumulator.rest_semantics),
        unpitched_display_pitch: metric_from_counts(accumulator.unpitched_display_pitch),
        performed_duration: metric_from_counts(accumulator.performed_duration),
        written_value: metric_from_counts(accumulator.written_value),
        grace_cue_chord_flags: metric_from_counts(accumulator.grace_cue_chord_flags),
        chord_anchor_relation: metric_from_counts(accumulator.chord_anchor_relation),
        voice_staff_identity: metric_from_counts(accumulator.voice_staff_identity),
        instrument_references: metric_from_counts(accumulator.instrument_references),
        sound_tie_annotations: metric_from_counts(accumulator.sound_tie_annotations),
        visual_tie_annotations: metric_from_counts(accumulator.visual_tie_annotations),
        tuplet_notations: metric_from_counts(accumulator.tuplet_notations),
        attribute_events: metric_from_counts(accumulator.attribute_events),
        attribute_event_values: metric_from_counts(accumulator.attribute_event_values),
        auxiliary_events: metric_from_counts(accumulator.auxiliary_events),
        auxiliary_event_values: metric_from_counts(accumulator.auxiliary_event_values),
        cursor_moves: metric_from_counts(accumulator.cursor_moves),
        cursor_move_values: metric_from_counts(accumulator.cursor_move_values),
        onset_error_beats: error_summary(&accumulator.onset_errors),
        duration_error_beats: error_summary(&accumulator.duration_errors),
        report_sha256: String::new(),
    };
    report.report_sha256 = musicxml_notation_report_commitment(&report)
        .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?;
    Ok(MusicXmlNotationEvaluationOutcomeV1::Evaluated(report))
}

pub fn validate_musicxml_notation_evaluation_report(
    reference: &[MusicXmlPartNotationGraphV1],
    prediction: &[MusicXmlPartNotationGraphV1],
    bindings: &[MusicXmlNotationPartBindingV1],
    config: MusicXmlNotationEvaluationConfigV1,
    supplied: &MusicXmlNotationEvaluationReportV1,
) -> Result<bool, MusicXmlNotationEvaluationErrorV1> {
    match evaluate_musicxml_notation_graphs(reference, prediction, bindings, config)? {
        MusicXmlNotationEvaluationOutcomeV1::Unavailable(_) => Ok(false),
        MusicXmlNotationEvaluationOutcomeV1::Evaluated(recomputed) => Ok(recomputed == *supplied
            && musicxml_notation_report_commitment(supplied)
                .map_err(|_| MusicXmlNotationEvaluationErrorV1::SerializationFailed)?
                == supplied.report_sha256),
    }
}

#[cfg(test)]
mod tests {
    fn brute_matching_size(
        adjacency: &[Vec<usize>],
        gold_index: usize,
        used_predictions: &mut [bool],
    ) -> usize {
        if gold_index == adjacency.len() {
            return 0;
        }
        let mut best = brute_matching_size(adjacency, gold_index + 1, used_predictions);
        for prediction_index in adjacency[gold_index].iter().copied() {
            if !used_predictions[prediction_index] {
                used_predictions[prediction_index] = true;
                best = best.max(
                    1 + brute_matching_size(adjacency, gold_index + 1, used_predictions)
                );
                used_predictions[prediction_index] = false;
            }
        }
        best
    }

    #[test]
    fn hopcroft_karp_matches_exhaustive_oracle_for_every_three_by_three_graph() {
        for mask in 0usize..(1usize << 9) {
            let mut adjacency = vec![Vec::new(); 3];
            for gold_index in 0..3 {
                for prediction_index in 0..3 {
                    let bit = gold_index * 3 + prediction_index;
                    if mask & (1 << bit) != 0 {
                        adjacency[gold_index].push(prediction_index);
                    }
                }
            }
            let (pair_gold, pair_prediction) = maximum_matching(&adjacency, 3);
            let observed = pair_gold.iter().filter(|pair| pair.is_some()).count();
            let mut used = [false; 3];
            let expected = brute_matching_size(&adjacency, 0, &mut used);
            assert_eq!(observed, expected, "adjacency={adjacency:?}");
            for (gold_index, prediction_index) in pair_gold.iter().enumerate() {
                if let Some(prediction_index) = prediction_index {
                    assert_eq!(pair_prediction[*prediction_index], Some(gold_index));
                }
            }
        }
    }


    use super::*;
    use crate::musicxml_barline_adapter::parse_musicxml_part_notation_graph;

    fn graph(xml: &str, source: &str, clock: &str) -> Vec<MusicXmlPartNotationGraphV1> {
        parse_musicxml_part_notation_graph(xml.as_bytes(), source, clock).unwrap()
    }

    fn binding(reference: &str, prediction: &str) -> MusicXmlNotationPartBindingV1 {
        MusicXmlNotationPartBindingV1 {
            reference_part_id: reference.into(),
            prediction_part_id: prediction.into(),
        }
    }

    fn xml(part: &str, notes: &str) -> String {
        format!(
            r#"<score-partwise><part id="{part}"><measure number="1">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                {notes}
            </measure></part></score-partwise>"#
        )
    }

    fn report(outcome: MusicXmlNotationEvaluationOutcomeV1) -> MusicXmlNotationEvaluationReportV1 {
        match outcome {
            MusicXmlNotationEvaluationOutcomeV1::Evaluated(report) => report,
            other => panic!("expected evaluation, got {other:?}"),
        }
    }

    #[test]
    fn identical_graphs_have_perfect_independent_scores_and_replay() {
        let source = xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><type>quarter</type></note>
            <note><rest/><duration>4</duration><voice>1</voice><staff>1</staff><type>quarter</type></note>
        "#);
        let gold = graph(&source, "gold-source", "shared-clock");
        let pred = graph(&source, "prediction-source", "shared-clock");
        let config = MusicXmlNotationEvaluationConfigV1::default();
        let bindings = [binding("P1", "P1")];
        let evaluated = report(evaluate_musicxml_notation_graphs(&gold, &pred, &bindings, config.clone()).unwrap());
        assert_eq!(evaluated.event_alignment.f1, Some(1.0));
        assert_eq!(evaluated.written_pitch_spelling.f1, Some(1.0));
        assert_eq!(evaluated.rest_semantics.f1, Some(1.0));
        assert_eq!(evaluated.performed_duration.f1, Some(1.0));
        assert_eq!(evaluated.voice_staff_identity.f1, Some(1.0));
        assert!(validate_musicxml_notation_evaluation_report(&gold, &pred, &bindings, config, &evaluated).unwrap());
    }

    #[test]
    fn pitch_and_duration_errors_are_not_hidden_by_onset_matching() {
        let gold = graph(&xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
        "#), "gold", "clock");
        let pred = graph(&xml("P1", r#"
            <note><pitch><step>D</step><octave>4</octave></pitch><duration>8</duration><type>half</type></note>
        "#), "pred", "clock");
        let evaluated = report(evaluate_musicxml_notation_graphs(
            &gold, &pred, &[binding("P1", "P1")], MusicXmlNotationEvaluationConfigV1::default()
        ).unwrap());
        assert_eq!(evaluated.event_alignment.counts.true_positives, 1);
        assert_eq!(evaluated.written_pitch_spelling.counts.true_positives, 0);
        assert_eq!(evaluated.performed_duration.counts.true_positives, 0);
        assert_eq!(evaluated.onset_error_beats.mean_absolute_error_beats, Some(0.0));
        assert_eq!(evaluated.duration_error_beats.mean_absolute_error_beats, Some(1.0));
    }

    #[test]
    fn one_prediction_cannot_match_two_reference_events() {
        let mut gold = graph(&xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
        "#), "gold", "clock");
        let pred = graph(&xml("P1", r#"
            <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration></note>
        "#), "pred", "clock");
        gold[0].notation_events.note_events[1].score_offset =
            gold[0].notation_events.note_events[0].score_offset;
        let evaluated = report(evaluate_musicxml_notation_graphs(
            &gold, &pred, &[binding("P1", "P1")], MusicXmlNotationEvaluationConfigV1::default()
        ).unwrap());
        assert_eq!(evaluated.event_alignment.counts.true_positives, 1);
        assert_eq!(evaluated.event_alignment.counts.false_negatives, 1);
    }

    #[test]
    fn wrong_event_family_does_not_align() {
        let gold = graph(&xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
        "#), "gold", "clock");
        let pred = graph(&xml("P1", r#"<note><rest/><duration>4</duration></note>"#), "pred", "clock");
        let evaluated = report(evaluate_musicxml_notation_graphs(
            &gold, &pred, &[binding("P1", "P1")], MusicXmlNotationEvaluationConfigV1::default()
        ).unwrap());
        assert_eq!(evaluated.event_alignment.counts.true_positives, 0);
        assert_eq!(evaluated.event_alignment.counts.false_positives, 1);
        assert_eq!(evaluated.event_alignment.counts.false_negatives, 1);
    }

    #[test]
    fn unknown_notation_is_unavailable_not_numeric_success() {
        let source = xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
        "#);
        let gold = graph(&source, "gold", "clock");
        let mut pred = graph(&source, "pred", "clock");
        pred[0].notation_events.status = MusicXmlNotationEventsStatusV1::Unknown {
            reason: "unsupported extension".into(),
        };
        let outcome = evaluate_musicxml_notation_graphs(
            &gold, &pred, &[binding("P1", "P1")], MusicXmlNotationEvaluationConfigV1::default()
        ).unwrap();
        assert!(matches!(outcome, MusicXmlNotationEvaluationOutcomeV1::Unavailable(_)));
    }

    #[test]
    fn score_clock_mismatch_is_rejected() {
        let source = xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
        "#);
        let gold = graph(&source, "gold", "clock-a");
        let pred = graph(&source, "pred", "clock-b");
        assert_eq!(
            evaluate_musicxml_notation_graphs(
                &gold, &pred, &[binding("P1", "P1")], MusicXmlNotationEvaluationConfigV1::default()
            ),
            Err(MusicXmlNotationEvaluationErrorV1::ScoreClockMismatch)
        );
    }

    #[test]
    fn explicit_part_mapping_supports_renamed_parts_and_rejects_duplicates() {
        let gold = graph(&xml("Violin", r#"
            <note><pitch><step>A</step><octave>4</octave></pitch><duration>4</duration></note>
        "#), "gold", "clock");
        let pred = graph(&xml("Part-1", r#"
            <note><pitch><step>A</step><octave>4</octave></pitch><duration>4</duration></note>
        "#), "pred", "clock");
        let binding = binding("Violin", "Part-1");
        let evaluated = report(evaluate_musicxml_notation_graphs(
            &gold, &pred, &[binding.clone()], MusicXmlNotationEvaluationConfigV1::default()
        ).unwrap());
        assert_eq!(evaluated.event_alignment.f1, Some(1.0));
        assert_eq!(
            evaluate_musicxml_notation_graphs(
                &gold, &pred, &[binding.clone(), binding], MusicXmlNotationEvaluationConfigV1::default()
            ),
            Err(MusicXmlNotationEvaluationErrorV1::DuplicateReferencePartBinding { part_id: "Violin".into() })
        );
    }

    #[test]
    fn attributes_and_auxiliary_events_are_scored_separately() {
        let source = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <direction><direction-type><words>dolce</words></direction-type><offset>0</offset></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let gold = graph(source, "gold", "clock");
        let pred = graph(source, "pred", "clock");
        let evaluated = report(evaluate_musicxml_notation_graphs(
            &gold, &pred, &[binding("P1", "P1")], MusicXmlNotationEvaluationConfigV1::default()
        ).unwrap());
        assert_eq!(evaluated.attribute_events.f1, Some(1.0));
        assert_eq!(evaluated.attribute_event_values.f1, Some(1.0));
        assert_eq!(evaluated.auxiliary_events.f1, Some(1.0));
        assert_eq!(evaluated.auxiliary_event_values.f1, Some(1.0));
    }

    #[test]
    fn tampered_report_is_rejected() {
        let source = xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
        "#);
        let gold = graph(&source, "gold", "clock");
        let pred = graph(&source, "pred", "clock");
        let config = MusicXmlNotationEvaluationConfigV1::default();
        let bindings = [binding("P1", "P1")];
        let mut evaluated = report(evaluate_musicxml_notation_graphs(
            &gold, &pred, &bindings, config.clone()
        ).unwrap());
        evaluated.written_pitch_spelling.counts.false_positives += 1;
        assert!(!validate_musicxml_notation_evaluation_report(
            &gold, &pred, &bindings, config, &evaluated
        ).unwrap());
    }

    #[test]
    fn report_commits_reference_prediction_and_config() {
        let source = xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
        "#);
        let gold = graph(&source, "gold", "clock");
        let pred = graph(&source, "pred", "clock");
        let evaluated = report(evaluate_musicxml_notation_graphs(
            &gold, &pred, &[binding("P1", "P1")], MusicXmlNotationEvaluationConfigV1::default()
        ).unwrap());
        assert_eq!(evaluated.reference_graphs_sha256.len(), 64);
        assert_eq!(evaluated.prediction_graphs_sha256.len(), 64);
        assert_eq!(evaluated.config_sha256.len(), 64);
        assert_eq!(evaluated.report_sha256.len(), 64);
    }

    #[test]
    fn maximum_matching_uses_an_augmenting_path_instead_of_greedy_first_fit() {
        // Gold 0 can match either prediction; gold 1 can only match prediction 0.
        // A first-fit greedy matcher gets one match. Maximum cardinality is two.
        let adjacency = vec![vec![0, 1], vec![0]];
        let (pair_gold, pair_pred) = maximum_matching(&adjacency, 2);
        assert_eq!(pair_gold, vec![Some(1), Some(0)]);
        assert_eq!(pair_pred, vec![Some(1), Some(0)]);
    }

    #[test]
    fn matching_with_exact_rational_tolerance_is_inclusive() {
        let gold = xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
        "#);
        let pred = xml("P1", r#"
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
        "#);
        let mut reference = graph(&gold, "gold", "clock");
        let mut prediction = graph(&pred, "pred", "clock");
        reference[0].notation_events.note_events[0].score_offset =
            RationalBeat::new(1, 4).unwrap();
        prediction[0].notation_events.note_events[0].score_offset =
            RationalBeat::new(3, 8).unwrap();
        let exact = MusicXmlNotationEvaluationConfigV1 {
            onset_tolerance: RationalBeat::new(1, 8).unwrap(),
        };
        let outcome = evaluate_musicxml_notation_graphs(
            &reference, &prediction, &[binding("P1", "P1")], exact
        ).unwrap();
        assert_eq!(
            report(outcome).event_alignment.counts.true_positives,
            1
        );
    }

}
