// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, measure-localized correction actions for MusicXML score
//! comparison. This is an explanation layer over MUSE-MIR-019's exact
//! one-to-one match set, not a second aligner and not a human-time estimator.
//!
//! Every action points to the reference/prediction source event ordinals and
//! carries digests of the before/after field values. The original score remains
//! immutable. No default effort/severity weights are invented.

use crate::barline_eval::RationalBeat;
use crate::evidence_digest::canonical_json_sha256;
use crate::musicxml_barline_adapter::MusicXmlPartNotationGraphV1;
use crate::musicxml_notation_eval::{
    evaluate_musicxml_notation_graphs_with_matches, MusicXmlNotationEvaluationConfigV1,
    MusicXmlNotationEvaluationErrorV1, MusicXmlNotationEvaluationReportV1,
    MusicXmlNotationMatchSetV1,
};
use crate::musicxml_notation_events::{
    MusicXmlNotationEventV1, MusicXmlSourceNodeV1,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const MUSICXML_CORRECTION_LEDGER_SCHEMA_V1: &str =
    "muse-musicxml-correction-ledger-v1";
pub const MUSICXML_CORRECTION_LEDGER_ALGORITHM_V1: &str =
    "mir019-shared-matches-field-diff-source-order-v1";
pub const MUSICXML_CORRECTION_LEDGER_MAX_ACTIONS_V1: usize = 50_000;
pub const MUSICXML_CORRECTION_LEDGER_MAX_BYTES_V1: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicXmlCorrectionFamilyV1 {
    Measure,
    NoteEvent,
    AuxiliaryEvent,
    AttributeEvent,
    CursorMove,
    PartMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicXmlCorrectionOperationV1 {
    InsertInPrediction,
    DeleteFromPrediction,
    ReplaceField,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlCorrectionActionV1 {
    /// Domain-separated SHA-256 of this action with action_id omitted.
    pub action_id: String,
    pub part_id: String,
    pub family: MusicXmlCorrectionFamilyV1,
    pub operation: MusicXmlCorrectionOperationV1,
    pub field_path: String,
    pub reference_measure_index: Option<usize>,
    pub prediction_measure_index: Option<usize>,
    pub reference_measure_number: Option<String>,
    pub prediction_measure_number: Option<String>,
    pub reference_source_order: Option<usize>,
    pub prediction_source_order: Option<usize>,
    pub reference_score_offset: Option<RationalBeat>,
    pub prediction_score_offset: Option<RationalBeat>,
    /// Hashes, not executable or unbounded XML fragments.
    pub before_sha256: Option<String>,
    pub after_sha256: Option<String>,
    /// A short stable explanation; contains no copied source text.
    pub explanation_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlCorrectionLedgerConfigV1 {
    pub maximum_actions: usize,
    pub maximum_serialized_bytes: usize,
}

impl Default for MusicXmlCorrectionLedgerConfigV1 {
    fn default() -> Self {
        Self {
            maximum_actions: 50_000,
            maximum_serialized_bytes: MUSICXML_CORRECTION_LEDGER_MAX_BYTES_V1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlCorrectionLedgerReportV1 {
    pub schema: String,
    pub algorithm: String,
    pub part_id: String,
    pub score_clock_id: String,
    pub reference_graph_sha256: String,
    pub prediction_graph_sha256: String,
    pub mir019_report_sha256: String,
    pub config: MusicXmlCorrectionLedgerConfigV1,
    pub part_metadata_actions_covered: bool,
    pub actions: Vec<MusicXmlCorrectionActionV1>,
    pub action_count: usize,
    pub by_family: BTreeMap<String, usize>,
    pub by_operation: BTreeMap<String, usize>,
    pub by_field: BTreeMap<String, usize>,
    pub affected_measure_count: usize,
    pub affected_reference_source_event_count: usize,
    pub affected_prediction_source_event_count: usize,
    pub unweighted_action_count: usize,
    pub ledger_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicXmlCorrectionLedgerErrorV1 {
    Evaluation(MusicXmlNotationEvaluationErrorV1),
    InvalidConfiguration { field: String },
    ActionBudgetExceeded { count: usize, maximum: usize },
    SerializedReportTooLarge { bytes: usize, maximum: usize },
    SerializationFailed,
}

impl From<MusicXmlNotationEvaluationErrorV1> for MusicXmlCorrectionLedgerErrorV1 {
    fn from(value: MusicXmlNotationEvaluationErrorV1) -> Self {
        Self::Evaluation(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ActionBodyV1<'a> {
    part_id: &'a str,
    family: MusicXmlCorrectionFamilyV1,
    operation: MusicXmlCorrectionOperationV1,
    field_path: &'a str,
    reference_measure_index: Option<usize>,
    prediction_measure_index: Option<usize>,
    reference_measure_number: &'a Option<String>,
    prediction_measure_number: &'a Option<String>,
    reference_source_order: Option<usize>,
    prediction_source_order: Option<usize>,
    reference_score_offset: Option<RationalBeat>,
    prediction_score_offset: Option<RationalBeat>,
    before_sha256: &'a Option<String>,
    after_sha256: &'a Option<String>,
    explanation_code: &'a str,
}

#[derive(Serialize)]
struct LedgerBodyV1<'a> {
    schema: &'a str,
    algorithm: &'a str,
    part_id: &'a str,
    score_clock_id: &'a str,
    reference_graph_sha256: &'a str,
    prediction_graph_sha256: &'a str,
    mir019_report_sha256: &'a str,
    config: &'a MusicXmlCorrectionLedgerConfigV1,
    part_metadata_actions_covered: bool,
    actions: &'a [MusicXmlCorrectionActionV1],
    action_count: usize,
    by_family: &'a BTreeMap<String, usize>,
    by_operation: &'a BTreeMap<String, usize>,
    by_field: &'a BTreeMap<String, usize>,
    affected_measure_count: usize,
    affected_reference_source_event_count: usize,
    affected_prediction_source_event_count: usize,
    unweighted_action_count: usize,
}

fn digest<T: Serialize>(value: &T) -> Result<String, MusicXmlCorrectionLedgerErrorV1> {
    canonical_json_sha256(value).map_err(|_| MusicXmlCorrectionLedgerErrorV1::SerializationFailed)
}

fn maybe_digest<T: Serialize>(value: Option<&T>) -> Result<Option<String>, MusicXmlCorrectionLedgerErrorV1> {
    value.map(digest).transpose()
}

fn locator_number(
    graph: &MusicXmlPartNotationGraphV1,
    measure_index: Option<usize>,
) -> Option<String> {
    measure_index
        .and_then(|index| graph.measure_locators.get(index))
        .and_then(|locator| locator.number.clone())
}

fn make_action(
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    family: MusicXmlCorrectionFamilyV1,
    operation: MusicXmlCorrectionOperationV1,
    field_path: impl Into<String>,
    reference_measure_index: Option<usize>,
    prediction_measure_index: Option<usize>,
    reference_source_order: Option<usize>,
    prediction_source_order: Option<usize>,
    reference_score_offset: Option<RationalBeat>,
    prediction_score_offset: Option<RationalBeat>,
    before_sha256: Option<String>,
    after_sha256: Option<String>,
    explanation_code: impl Into<String>,
) -> Result<MusicXmlCorrectionActionV1, MusicXmlCorrectionLedgerErrorV1> {
    let field_path = field_path.into();
    let explanation_code = explanation_code.into();
    let mut action = MusicXmlCorrectionActionV1 {
        action_id: String::new(),
        part_id: reference.part_id.clone(),
        family,
        operation,
        field_path,
        reference_measure_index,
        prediction_measure_index,
        reference_measure_number: locator_number(reference, reference_measure_index),
        prediction_measure_number: locator_number(prediction, prediction_measure_index),
        reference_source_order,
        prediction_source_order,
        reference_score_offset,
        prediction_score_offset,
        before_sha256,
        after_sha256,
        explanation_code,
    };
    let body = ActionBodyV1 {
        part_id: &action.part_id,
        family: action.family,
        operation: action.operation,
        field_path: &action.field_path,
        reference_measure_index: action.reference_measure_index,
        prediction_measure_index: action.prediction_measure_index,
        reference_measure_number: &action.reference_measure_number,
        prediction_measure_number: &action.prediction_measure_number,
        reference_source_order: action.reference_source_order,
        prediction_source_order: action.prediction_source_order,
        reference_score_offset: action.reference_score_offset,
        prediction_score_offset: action.prediction_score_offset,
        before_sha256: &action.before_sha256,
        after_sha256: &action.after_sha256,
        explanation_code: &action.explanation_code,
    };
    action.action_id = digest(&body)?;
    Ok(action)
}

fn add_action(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    action: MusicXmlCorrectionActionV1,
    maximum: usize,
) -> Result<(), MusicXmlCorrectionLedgerErrorV1> {
    if actions.len() >= maximum {
        return Err(MusicXmlCorrectionLedgerErrorV1::ActionBudgetExceeded {
            count: actions.len().saturating_add(1),
            maximum,
        });
    }
    actions.push(action);
    Ok(())
}

fn add_insert_delete<T: Serialize>(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    family: MusicXmlCorrectionFamilyV1,
    value: &T,
    operation: MusicXmlCorrectionOperationV1,
    measure_index: Option<usize>,
    source_order: Option<usize>,
    score_offset: Option<RationalBeat>,
    maximum: usize,
    explanation_code: &str,
) -> Result<(), MusicXmlCorrectionLedgerErrorV1> {
    let value_digest = digest(value)?;
    let (before, after, ref_measure, pred_measure, ref_order, pred_order, ref_offset, pred_offset) =
        match operation {
            MusicXmlCorrectionOperationV1::DeleteFromPrediction => (
                Some(value_digest),
                None,
                measure_index,
                None,
                source_order,
                None,
                score_offset,
                None,
            ),
            MusicXmlCorrectionOperationV1::InsertInPrediction => (
                None,
                Some(value_digest),
                None,
                measure_index,
                None,
                source_order,
                None,
                score_offset,
            ),
            MusicXmlCorrectionOperationV1::ReplaceField => {
                return Err(MusicXmlCorrectionLedgerErrorV1::InvalidConfiguration {
                    field: "insert/delete helper received ReplaceField".into(),
                });
            }
        };
    let action = make_action(
        reference,
        prediction,
        family,
        operation,
        "$event",
        ref_measure,
        pred_measure,
        ref_order,
        pred_order,
        ref_offset,
        pred_offset,
        before,
        after,
        explanation_code,
    )?;
    add_action(actions, action, maximum)
}

fn add_replace<T: Serialize, U: Serialize>(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    family: MusicXmlCorrectionFamilyV1,
    field_path: &str,
    before: &T,
    after: &U,
    reference_measure_index: Option<usize>,
    prediction_measure_index: Option<usize>,
    reference_source_order: Option<usize>,
    prediction_source_order: Option<usize>,
    reference_score_offset: Option<RationalBeat>,
    prediction_score_offset: Option<RationalBeat>,
    maximum: usize,
    explanation_code: &str,
) -> Result<(), MusicXmlCorrectionLedgerErrorV1> {
    let action = make_action(
        reference,
        prediction,
        family,
        MusicXmlCorrectionOperationV1::ReplaceField,
        field_path,
        reference_measure_index,
        prediction_measure_index,
        reference_source_order,
        prediction_source_order,
        reference_score_offset,
        prediction_score_offset,
        Some(digest(before)?),
        Some(digest(after)?),
        explanation_code,
    )?;
    add_action(actions, action, maximum)
}

fn note_field_actions(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    gold: &MusicXmlNotationEventV1,
    predicted: &MusicXmlNotationEventV1,
    mapped_anchor: Option<usize>,
    maximum: usize,
) -> Result<(), MusicXmlCorrectionLedgerErrorV1> {
    use MusicXmlCorrectionFamilyV1::NoteEvent;
    let ref_measure = Some(gold.source_measure_index);
    let pred_measure = Some(predicted.source_measure_index);
    let ref_order = Some(gold.source_order);
    let pred_order = Some(predicted.source_order);
    let ref_offset = Some(gold.score_offset);
    let pred_offset = Some(predicted.score_offset);

    match (&gold.kind, &predicted.kind) {
        (
            crate::musicxml_notation_events::MusicXmlNotationEventKindV1::Pitched(left),
            crate::musicxml_notation_events::MusicXmlNotationEventKindV1::Pitched(right),
        ) => {
            if !crate::musicxml_notation_eval::written_pitch_equal(left, right) {
                add_replace(
                    actions, reference, prediction, NoteEvent, "written_pitch_spelling",
                    left, right, ref_measure, pred_measure, ref_order, pred_order,
                    ref_offset, pred_offset, maximum, "written_pitch_spelling_differs"
                )?;
            }
        }
        (
            crate::musicxml_notation_events::MusicXmlNotationEventKindV1::Rest { .. },
            crate::musicxml_notation_events::MusicXmlNotationEventKindV1::Rest { .. },
        ) => {
            if gold.kind != predicted.kind {
                add_replace(
                    actions, reference, prediction, NoteEvent, "rest_semantics",
                    &gold.kind, &predicted.kind, ref_measure, pred_measure, ref_order, pred_order,
                    ref_offset, pred_offset, maximum, "rest_display_or_measure_flag_differs"
                )?;
            }
        }
        (
            crate::musicxml_notation_events::MusicXmlNotationEventKindV1::Unpitched { .. },
            crate::musicxml_notation_events::MusicXmlNotationEventKindV1::Unpitched { .. },
        ) => {
            if gold.kind != predicted.kind {
                add_replace(
                    actions, reference, prediction, NoteEvent, "unpitched_display",
                    &gold.kind, &predicted.kind, ref_measure, pred_measure, ref_order, pred_order,
                    ref_offset, pred_offset, maximum, "unpitched_display_differs"
                )?;
            }
        }
        _ => {
            // The MIR-019 matcher only pairs within an event family. This
            // branch is defensive and should be unreachable after validation.
            add_replace(
                actions, reference, prediction, NoteEvent, "event_kind",
                &gold.kind, &predicted.kind, ref_measure, pred_measure, ref_order, pred_order,
                ref_offset, pred_offset, maximum, "event_family_changed"
            )?;
        }
    }

    macro_rules! compare_field {
        ($field:ident, $path:literal, $reason:literal) => {
            if gold.$field != predicted.$field {
                add_replace(
                    actions, reference, prediction, NoteEvent, $path,
                    &gold.$field, &predicted.$field,
                    ref_measure, pred_measure, ref_order, pred_order,
                    ref_offset, pred_offset, maximum, $reason
                )?;
            }
        };
    }

    compare_field!(performed_duration, "performed_duration", "performed_duration_differs");
    compare_field!(is_grace, "grace.flag", "grace_flag_differs");
    compare_field!(grace, "grace.attributes", "grace_attributes_differ");
    compare_field!(is_cue, "cue.flag", "cue_flag_differs");
    compare_field!(is_chord_member, "chord_member.flag", "chord_member_flag_differs");
    if mapped_anchor != predicted.chord_anchor_source_order {
        add_replace(
            actions, reference, prediction, NoteEvent, "chord_member.anchor",
            &mapped_anchor, &predicted.chord_anchor_source_order,
            ref_measure, pred_measure, ref_order, pred_order,
            ref_offset, pred_offset, maximum, "chord_anchor_target_differs"
        )?;
    }
    compare_field!(voice, "voice", "voice_assignment_differs");
    compare_field!(staff, "staff", "staff_assignment_differs");
    compare_field!(instrument_ids, "instrument_ids", "note_instrument_reference_differs");
    compare_field!(written_type, "written_type", "written_note_type_differs");
    compare_field!(dot_count, "dot_count", "written_dot_count_differs");
    compare_field!(time_modification, "time_modification", "time_modification_differs");
    compare_field!(sound_ties, "sound_ties", "sound_tie_semantics_differ");
    compare_field!(visual_ties, "visual_ties", "visual_tie_semantics_differ");
    compare_field!(tuplet_notations, "tuplet_notations", "tuplet_notation_differs");
    compare_field!(stem, "stem", "stem_notation_differs");
    compare_field!(notehead, "notehead", "notehead_notation_differs");
    compare_field!(accidental, "accidental", "accidental_notation_differs");
    compare_field!(beams, "beams", "beam_notation_differs");
    compare_field!(source_note, "source_note.subtree", "source_note_subtree_differs");
    Ok(())
}

fn add_event_family_actions<T: Serialize>(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    ref_items: &[T],
    pred_items: &[T],
    pairs: &[(usize, usize)],
    family: MusicXmlCorrectionFamilyV1,
    config_max: usize,
    mut meta: impl FnMut(&T) -> (usize, usize, RationalBeat),
) -> Result<(), MusicXmlCorrectionLedgerErrorV1> {
    let matched_ref: BTreeSet<usize> = pairs.iter().map(|(left, _)| *left).collect();
    let matched_pred: BTreeSet<usize> = pairs.iter().map(|(_, right)| *right).collect();
    for (index, item) in ref_items.iter().enumerate() {
        if !matched_ref.contains(&index) {
            let (measure, order, offset) = meta(item);
            add_insert_delete(
                actions, reference, prediction, family, item,
                MusicXmlCorrectionOperationV1::DeleteFromPrediction,
                Some(measure), Some(order), Some(offset), config_max, "reference_event_missing_from_prediction"
            )?;
        }
    }
    for (index, item) in pred_items.iter().enumerate() {
        if !matched_pred.contains(&index) {
            let (measure, order, offset) = meta(item);
            add_insert_delete(
                actions, reference, prediction, family, item,
                MusicXmlCorrectionOperationV1::InsertInPrediction,
                Some(measure), Some(order), Some(offset), config_max, "unexpected_prediction_event"
            )?;
        }
    }
    Ok(())
}

fn note_actions(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    pairs: &[(usize, usize)],
    maximum: usize,
) -> Result<(), MusicXmlCorrectionLedgerErrorV1> {
    let gold = &reference.notation_events.note_events;
    let predicted = &prediction.notation_events.note_events;
    let matched_ref: BTreeSet<usize> = pairs.iter().map(|(left, _)| *left).collect();
    let matched_pred: BTreeSet<usize> = pairs.iter().map(|(_, right)| *right).collect();
    let reference_by_order: BTreeMap<usize, usize> = pairs.iter().map(|(left, right)| {
        (gold[*left].source_order, predicted[*right].source_order)
    }).collect();

    for (index, event) in gold.iter().enumerate() {
        if !matched_ref.contains(&index) {
            add_insert_delete(
                actions, reference, prediction, MusicXmlCorrectionFamilyV1::NoteEvent,
                event, MusicXmlCorrectionOperationV1::DeleteFromPrediction,
                Some(event.source_measure_index), Some(event.source_order), Some(event.score_offset),
                maximum, "reference_note_event_missing_from_prediction"
            )?;
        }
    }
    for (index, event) in predicted.iter().enumerate() {
        if !matched_pred.contains(&index) {
            add_insert_delete(
                actions, reference, prediction, MusicXmlCorrectionFamilyV1::NoteEvent,
                event, MusicXmlCorrectionOperationV1::InsertInPrediction,
                Some(event.source_measure_index), Some(event.source_order), Some(event.score_offset),
                maximum, "unexpected_prediction_note_event"
            )?;
        }
    }
    for &(left, right) in pairs {
        let gold_event = &gold[left];
        let predicted_event = &predicted[right];
        let mapped_anchor = gold_event
            .chord_anchor_source_order
            .and_then(|anchor| reference_by_order.get(&anchor).copied());
        note_field_actions(
            actions, reference, prediction, gold_event, predicted_event,
            mapped_anchor, maximum
        )?;
    }
    Ok(())
}

fn auxiliary_and_attribute_actions(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    matches: &MusicXmlNotationMatchSetV1,
    maximum: usize,
) -> Result<(), MusicXmlCorrectionLedgerErrorV1> {
    let gold = &reference.notation_events;
    let predicted = &prediction.notation_events;

    add_event_family_actions(
        actions, reference, prediction,
        &gold.auxiliary_events, &predicted.auxiliary_events,
        &matches.auxiliary_pairs, MusicXmlCorrectionFamilyV1::AuxiliaryEvent, maximum,
        |event| (event.measure_index, event.source_order, event.score_offset),
    )?;
    for &(left, right) in &matches.auxiliary_pairs {
        let a = &gold.auxiliary_events[left];
        let b = &predicted.auxiliary_events[right];
        if a.score_offset != b.score_offset {
            add_replace(
                actions, reference, prediction, MusicXmlCorrectionFamilyV1::AuxiliaryEvent,
                "score_offset", &a.score_offset, &b.score_offset,
                Some(a.measure_index), Some(b.measure_index), Some(a.source_order), Some(b.source_order),
                Some(a.score_offset), Some(b.score_offset), maximum, "auxiliary_event_position_differs"
            )?;
        }
        if a.source != b.source {
            add_replace(
                actions, reference, prediction, MusicXmlCorrectionFamilyV1::AuxiliaryEvent,
                "source.subtree", &a.source, &b.source,
                Some(a.measure_index), Some(b.measure_index), Some(a.source_order), Some(b.source_order),
                Some(a.score_offset), Some(b.score_offset), maximum, "auxiliary_event_source_differs"
            )?;
        }
    }

    add_event_family_actions(
        actions, reference, prediction,
        &gold.attribute_events, &predicted.attribute_events,
        &matches.attribute_pairs, MusicXmlCorrectionFamilyV1::AttributeEvent, maximum,
        |event| (event.measure_index, event.source_order, event.score_offset),
    )?;
    for &(left, right) in &matches.attribute_pairs {
        let a = &gold.attribute_events[left];
        let b = &predicted.attribute_events[right];
        if a.score_offset != b.score_offset {
            add_replace(
                actions, reference, prediction, MusicXmlCorrectionFamilyV1::AttributeEvent,
                "score_offset", &a.score_offset, &b.score_offset,
                Some(a.measure_index), Some(b.measure_index), Some(a.source_order), Some(b.source_order),
                Some(a.score_offset), Some(b.score_offset), maximum, "attribute_event_position_differs"
            )?;
        }
        if a.source != b.source {
            add_replace(
                actions, reference, prediction, MusicXmlCorrectionFamilyV1::AttributeEvent,
                "source.subtree", &a.source, &b.source,
                Some(a.measure_index), Some(b.measure_index), Some(a.source_order), Some(b.source_order),
                Some(a.score_offset), Some(b.score_offset), maximum, "attribute_event_source_differs"
            )?;
        }
    }
    Ok(())
}

fn cursor_actions(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    matches: &MusicXmlNotationMatchSetV1,
    maximum: usize,
) -> Result<(), MusicXmlCorrectionLedgerErrorV1> {
    let gold = &reference.notation_events.cursor_moves;
    let predicted = &prediction.notation_events.cursor_moves;
    let matched_ref: BTreeSet<usize> = matches.cursor_pairs.iter().map(|(left, _)| *left).collect();
    let matched_pred: BTreeSet<usize> = matches.cursor_pairs.iter().map(|(_, right)| *right).collect();
    for (index, event) in gold.iter().enumerate() {
        if !matched_ref.contains(&index) {
            add_insert_delete(
                actions, reference, prediction, MusicXmlCorrectionFamilyV1::CursorMove, event,
                MusicXmlCorrectionOperationV1::DeleteFromPrediction,
                Some(event.measure_index), Some(event.source_order),
                Some(event.score_offset_before), maximum, "reference_cursor_move_missing_from_prediction"
            )?;
        }
    }
    for (index, event) in predicted.iter().enumerate() {
        if !matched_pred.contains(&index) {
            add_insert_delete(
                actions, reference, prediction, MusicXmlCorrectionFamilyV1::CursorMove, event,
                MusicXmlCorrectionOperationV1::InsertInPrediction,
                Some(event.measure_index), Some(event.source_order),
                Some(event.score_offset_before), maximum, "unexpected_prediction_cursor_move"
            )?;
        }
    }
    for &(left, right) in &matches.cursor_pairs {
        let a = &gold[left];
        let b = &predicted[right];
        macro_rules! cmp {
            ($field:ident, $path:literal, $reason:literal) => {
                if a.$field != b.$field {
                    add_replace(
                        actions, reference, prediction, MusicXmlCorrectionFamilyV1::CursorMove,
                        $path, &a.$field, &b.$field,
                        Some(a.measure_index), Some(b.measure_index), Some(a.source_order), Some(b.source_order),
                        Some(a.score_offset_before), Some(b.score_offset_before), maximum, $reason
                    )?;
                }
            };
        }
        cmp!(direction, "direction", "cursor_move_direction_differs");
        cmp!(voice, "voice", "cursor_move_voice_differs");
        cmp!(staff, "staff", "cursor_move_staff_differs");
        cmp!(score_offset_before, "score_offset_before", "cursor_move_start_differs");
        cmp!(duration, "duration", "cursor_move_duration_differs");
        cmp!(score_offset_after, "score_offset_after", "cursor_move_end_differs");
        cmp!(source, "source.subtree", "cursor_move_source_differs");
    }
    Ok(())
}

fn measure_actions(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    maximum: usize,
) -> Result<(), MusicXmlCorrectionLedgerErrorV1> {
    let max_len = reference.map.measures.len().max(prediction.map.measures.len());
    for index in 0..max_len {
        match (reference.map.measures.get(index), prediction.map.measures.get(index)) {
            (Some(gold), Some(predicted)) => {
                if gold.start != predicted.start {
                    add_replace(actions, reference, prediction, MusicXmlCorrectionFamilyV1::Measure,
                        "measure.start", &gold.start, &predicted.start,
                        Some(index), Some(index), None, None, Some(gold.start), Some(predicted.start),
                        maximum, "measure_start_differs")?;
                }
                if gold.end != predicted.end {
                    add_replace(actions, reference, prediction, MusicXmlCorrectionFamilyV1::Measure,
                        "measure.end", &gold.end, &predicted.end,
                        Some(index), Some(index), None, None, Some(gold.end), Some(predicted.end),
                        maximum, "measure_end_differs")?;
                }
                if gold.kind != predicted.kind {
                    add_replace(actions, reference, prediction, MusicXmlCorrectionFamilyV1::Measure,
                        "measure.kind", &gold.kind, &predicted.kind,
                        Some(index), Some(index), None, None, Some(gold.start), Some(predicted.start),
                        maximum, "measure_kind_differs")?;
                }
                let gold_locator = reference.measure_locators.get(index);
                let pred_locator = prediction.measure_locators.get(index);
                if let (Some(a), Some(b)) = (gold_locator, pred_locator) {
                    macro_rules! locator_cmp {
                        ($field:ident, $path:literal, $reason:literal) => {
                            if a.$field != b.$field {
                                add_replace(actions, reference, prediction, MusicXmlCorrectionFamilyV1::Measure,
                                    $path, &a.$field, &b.$field, Some(index), Some(index), None, None,
                                    Some(gold.start), Some(predicted.start), maximum, $reason)?;
                            }
                        };
                    }
                    locator_cmp!(number, "measure.number", "measure_number_differs");
                    locator_cmp!(source_id, "measure.source_id", "measure_source_id_differs");
                    locator_cmp!(display_text, "measure.display_text", "measure_display_text_differs");
                    locator_cmp!(implicit, "measure.implicit", "measure_implicit_flag_differs");
                    locator_cmp!(non_controlling, "measure.non_controlling", "measure_control_flag_differs");
                }
            }
            (Some(gold), None) => {
                add_insert_delete(actions, reference, prediction, MusicXmlCorrectionFamilyV1::Measure,
                    gold, MusicXmlCorrectionOperationV1::DeleteFromPrediction,
                    Some(index), None, Some(gold.start), maximum, "reference_measure_missing_from_prediction")?;
            }
            (None, Some(predicted)) => {
                add_insert_delete(actions, reference, prediction, MusicXmlCorrectionFamilyV1::Measure,
                    predicted, MusicXmlCorrectionOperationV1::InsertInPrediction,
                    None, Some(index), None, maximum, "unexpected_prediction_measure")?;
            }
            (None, None) => {}
        }
    }
    Ok(())
}

fn metadata_actions(
    actions: &mut Vec<MusicXmlCorrectionActionV1>,
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    maximum: usize,
) -> Result<bool, MusicXmlCorrectionLedgerErrorV1> {
    if !reference.part_metadata_layer_present || !prediction.part_metadata_layer_present {
        return Ok(false);
    }
    match (&reference.part_metadata, &prediction.part_metadata) {
        (Some(gold), None) => {
            add_insert_delete(actions, reference, prediction, MusicXmlCorrectionFamilyV1::PartMetadata,
                gold, MusicXmlCorrectionOperationV1::DeleteFromPrediction,
                None, None, None, maximum, "reference_part_metadata_missing_from_prediction")?;
        }
        (None, Some(predicted)) => {
            add_insert_delete(actions, reference, prediction, MusicXmlCorrectionFamilyV1::PartMetadata,
                predicted, MusicXmlCorrectionOperationV1::InsertInPrediction,
                None, None, None, maximum, "unexpected_prediction_part_metadata")?;
        }
        (Some(gold), Some(predicted)) => {
            macro_rules! cmp {
                ($left:expr, $right:expr, $path:literal, $reason:literal) => {
                    if $left != $right {
                        add_replace(actions, reference, prediction, MusicXmlCorrectionFamilyV1::PartMetadata,
                            $path, &$left, &$right, None, None, None, None, None, None,
                            maximum, $reason)?;
                    }
                };
            }
            cmp!(gold.part_name, predicted.part_name, "part.name", "part_name_differs");
            cmp!(gold.part_abbreviation, predicted.part_abbreviation, "part.abbreviation", "part_abbreviation_differs");
            cmp!(gold.score_instrument_ids, predicted.score_instrument_ids, "part.score_instrument_ids", "score_instrument_ids_differ");
            let gold_midi: Vec<_> = collect_midi_instruments(&gold.source_score_part);
            let pred_midi: Vec<_> = collect_midi_instruments(&predicted.source_score_part);
            cmp!(gold_midi, pred_midi, "part.midi_instrument_setup", "midi_instrument_setup_differs");
            cmp!(gold.source_score_part, predicted.source_score_part, "part.source_subtree", "score_part_source_subtree_differs");
        }
        (None, None) => {}
    }
    Ok(true)
}

fn collect_midi_instruments<'a>(node: &'a MusicXmlSourceNodeV1) -> Vec<&'a MusicXmlSourceNodeV1> {
    let mut output = Vec::new();
    collect_midi_instruments_inner(node, &mut output);
    output
}

fn collect_midi_instruments_inner<'a>(
    node: &'a MusicXmlSourceNodeV1,
    output: &mut Vec<&'a MusicXmlSourceNodeV1>,
) {
    if node.element == "midi-instrument" {
        output.push(node);
    }
    for child in &node.children {
        collect_midi_instruments_inner(child, output);
    }
}

fn action_sort(left: &MusicXmlCorrectionActionV1, right: &MusicXmlCorrectionActionV1) -> Ordering {
    let position_order = match (left.reference_score_offset.or(left.prediction_score_offset),
                                right.reference_score_offset.or(right.prediction_score_offset)) {
        (Some(a), Some(b)) => a.compare_exact(b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    left.reference_measure_index
        .or(left.prediction_measure_index)
        .cmp(&right.reference_measure_index.or(right.prediction_measure_index))
        .then(position_order)
        .then(left.family.cmp(&right.family))
        .then(left.reference_source_order.cmp(&right.reference_source_order))
        .then(left.prediction_source_order.cmp(&right.prediction_source_order))
        .then(left.operation.cmp(&right.operation))
        .then(left.field_path.cmp(&right.field_path))
        .then(left.action_id.cmp(&right.action_id))
}

fn ledger_commitment(report: &MusicXmlCorrectionLedgerReportV1) -> Result<String, MusicXmlCorrectionLedgerErrorV1> {
    digest(&LedgerBodyV1 {
        schema: &report.schema,
        algorithm: &report.algorithm,
        part_id: &report.part_id,
        score_clock_id: &report.score_clock_id,
        reference_graph_sha256: &report.reference_graph_sha256,
        prediction_graph_sha256: &report.prediction_graph_sha256,
        mir019_report_sha256: &report.mir019_report_sha256,
        config: &report.config,
        part_metadata_actions_covered: report.part_metadata_actions_covered,
        actions: &report.actions,
        action_count: report.action_count,
        by_family: &report.by_family,
        by_operation: &report.by_operation,
        by_field: &report.by_field,
        affected_measure_count: report.affected_measure_count,
        affected_reference_source_event_count: report.affected_reference_source_event_count,
        affected_prediction_source_event_count: report.affected_prediction_source_event_count,
        unweighted_action_count: report.unweighted_action_count,
    })
}

pub fn build_musicxml_correction_ledger(
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    evaluation_config: MusicXmlNotationEvaluationConfigV1,
    config: MusicXmlCorrectionLedgerConfigV1,
) -> Result<MusicXmlCorrectionLedgerReportV1, MusicXmlCorrectionLedgerErrorV1> {
    if config.maximum_actions == 0
        || config.maximum_actions > MUSICXML_CORRECTION_LEDGER_MAX_ACTIONS_V1
        || config.maximum_serialized_bytes == 0
        || config.maximum_serialized_bytes > MUSICXML_CORRECTION_LEDGER_MAX_BYTES_V1
    {
        return Err(MusicXmlCorrectionLedgerErrorV1::InvalidConfiguration {
            field: "action and serialized-byte budgets must be positive and within V1 caps".into(),
        });
    }

    // The evaluator performs all graph validation and alignment once. The
    // returned pairs are consumed directly; no second matching pass occurs.
    let (mir019, matches): (
        MusicXmlNotationEvaluationReportV1,
        MusicXmlNotationMatchSetV1,
    ) = evaluate_musicxml_notation_graphs_with_matches(
        reference,
        prediction,
        evaluation_config,
    )?;

    let mut actions = Vec::new();
    note_actions(
        &mut actions, reference, prediction, &matches.note_pairs, config.maximum_actions
    )?;
    auxiliary_and_attribute_actions(
        &mut actions, reference, prediction, &matches, config.maximum_actions
    )?;
    cursor_actions(
        &mut actions, reference, prediction, &matches, config.maximum_actions
    )?;
    measure_actions(&mut actions, reference, prediction, config.maximum_actions)?;
    let part_metadata_actions_covered = metadata_actions(
        &mut actions, reference, prediction, config.maximum_actions
    )?;

    actions.sort_by(action_sort);
    let mut by_family = BTreeMap::new();
    let mut by_operation = BTreeMap::new();
    let mut by_field = BTreeMap::new();
    let mut affected_measures = BTreeSet::new();
    let mut affected_reference_sources = BTreeSet::new();
    let mut affected_prediction_sources = BTreeSet::new();
    for action in &actions {
        *by_family.entry(format!("{:?}", action.family)).or_insert(0usize) += 1;
        *by_operation.entry(format!("{:?}", action.operation)).or_insert(0usize) += 1;
        *by_field.entry(action.field_path.clone()).or_insert(0usize) += 1;
        if let Some(index) = action.reference_measure_index {
            affected_measures.insert(index);
        }
        if let Some(index) = action.prediction_measure_index {
            affected_measures.insert(index);
        }
        if let Some(order) = action.reference_source_order {
            affected_reference_sources.insert((action.family, order));
        }
        if let Some(order) = action.prediction_source_order {
            affected_prediction_sources.insert((action.family, order));
        }
    }

    let reference_graph_sha256 = digest(reference)?;
    let prediction_graph_sha256 = digest(prediction)?;
    let mut report = MusicXmlCorrectionLedgerReportV1 {
        schema: MUSICXML_CORRECTION_LEDGER_SCHEMA_V1.into(),
        algorithm: MUSICXML_CORRECTION_LEDGER_ALGORITHM_V1.into(),
        part_id: reference.part_id.clone(),
        score_clock_id: reference.map.score_clock_id.clone(),
        reference_graph_sha256,
        prediction_graph_sha256,
        mir019_report_sha256: mir019.report_sha256,
        config,
        part_metadata_actions_covered,
        action_count: actions.len(),
        by_family,
        by_operation,
        by_field,
        affected_measure_count: affected_measures.len(),
        affected_reference_source_event_count: affected_reference_sources.len(),
        affected_prediction_source_event_count: affected_prediction_sources.len(),
        unweighted_action_count: actions.len(),
        actions,
        ledger_sha256: String::new(),
    };
    report.ledger_sha256 = ledger_commitment(&report)?;

    let serialized_bytes = serde_json::to_vec(&report)
        .map_err(|_| MusicXmlCorrectionLedgerErrorV1::SerializationFailed)?
        .len();
    if serialized_bytes > report.config.maximum_serialized_bytes {
        return Err(MusicXmlCorrectionLedgerErrorV1::SerializedReportTooLarge {
            bytes: serialized_bytes,
            maximum: report.config.maximum_serialized_bytes,
        });
    }
    Ok(report)
}

pub fn validate_musicxml_correction_ledger_replay(
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
    evaluation_config: MusicXmlNotationEvaluationConfigV1,
    config: MusicXmlCorrectionLedgerConfigV1,
    supplied: &MusicXmlCorrectionLedgerReportV1,
) -> Result<bool, MusicXmlCorrectionLedgerErrorV1> {
    let replay = build_musicxml_correction_ledger(
        reference, prediction, evaluation_config, config
    )?;
    Ok(replay == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::musicxml_barline_adapter::parse_musicxml_part_notation_graph;

    fn graph(xml: &str) -> MusicXmlPartNotationGraphV1 {
        parse_musicxml_part_notation_graph(
            xml.as_bytes(), "sha256:correction-fixture", "correction-score-clock"
        ).unwrap().remove(0)
    }

    fn config() -> MusicXmlNotationEvaluationConfigV1 {
        MusicXmlNotationEvaluationConfigV1::default()
    }

    fn fixture() -> &'static str {
        r#"<score-partwise>
            <part-list><score-part id="P1">
                <part-name>Violin</part-name>
                <score-instrument id="P1-I1"><instrument-name>Violin</instrument-name>
                    <midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>41</midi-program></midi-instrument>
                </score-instrument>
            </score-part></part-list>
            <part id="P1"><measure number="1">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><instrument id="P1-I1"/></note>
                <note><rest/><duration>4</duration><voice>1</voice><staff>1</staff></note>
                <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
                <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
                <direction><direction-type><dynamics><f/></dynamics></direction-type><sound tempo="90"/></direction>
            </measure>
            <measure number="2">
                <note><pitch><step>A</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice><staff>1</staff></note>
            </measure></part></score-partwise>"#
    }

    fn actions_by(report: &MusicXmlCorrectionLedgerReportV1, operation: MusicXmlCorrectionOperationV1) -> Vec<&MusicXmlCorrectionActionV1> {
        report.actions.iter().filter(|action| action.operation == operation).collect()
    }

    #[test]
    fn identical_graphs_produce_empty_deterministic_ledger() {
        let reference = graph(fixture());
        let ledger = build_musicxml_correction_ledger(
            &reference, &reference, config(), MusicXmlCorrectionLedgerConfigV1::default()
        ).unwrap();
        assert_eq!(ledger.action_count, 0);
        assert!(ledger.actions.is_empty());
        assert_eq!(ledger.affected_measure_count, 0);
        assert_eq!(ledger.ledger_sha256.len(), 64);
        assert_eq!(
            ledger,
            build_musicxml_correction_ledger(
                &reference, &reference, config(), MusicXmlCorrectionLedgerConfigV1::default()
            ).unwrap()
        );
    }

    #[test]
    fn missing_and_extra_events_produce_distinct_measure_localized_actions() {
        let reference = graph(fixture());
        let predicted_xml = fixture()
            .replace(
                r#"<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><instrument id="P1-I1"/></note>"#,
                ""
            )
            .replace(
                r#"<note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>"#,
                r#"<note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
                <note><pitch><step>B</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>"#
            );
        let prediction = graph(&predicted_xml);
        let ledger = build_musicxml_correction_ledger(
            &reference, &prediction, config(), MusicXmlCorrectionLedgerConfigV1::default()
        ).unwrap();
        assert!(actions_by(&ledger, MusicXmlCorrectionOperationV1::DeleteFromPrediction)
            .iter().any(|action| action.family == MusicXmlCorrectionFamilyV1::NoteEvent));
        assert!(actions_by(&ledger, MusicXmlCorrectionOperationV1::InsertInPrediction)
            .iter().any(|action| action.family == MusicXmlCorrectionFamilyV1::NoteEvent));
        assert!(ledger.affected_measure_count >= 1);
    }

    #[test]
    fn wrong_written_pitch_is_a_replace_action_not_a_presence_deletion() {
        let reference = graph(fixture());
        let prediction = graph(&fixture().replace(
            "<step>C</step><octave>4</octave>",
            "<step>D</step><octave>4</octave>"
        ));
        let ledger = build_musicxml_correction_ledger(
            &reference, &prediction, config(), MusicXmlCorrectionLedgerConfigV1::default()
        ).unwrap();
        assert_eq!(ledger.action_count, 1);
        let action = &ledger.actions[0];
        assert_eq!(action.operation, MusicXmlCorrectionOperationV1::ReplaceField);
        assert_eq!(action.field_path, "written_pitch_spelling");
        assert_eq!(action.family, MusicXmlCorrectionFamilyV1::NoteEvent);
        assert_eq!(action.reference_measure_number.as_deref(), Some("1"));
        assert_eq!(action.reference_source_order, Some(1));
        assert!(action.before_sha256.is_some());
        assert!(action.after_sha256.is_some());
    }

    #[test]
    fn one_event_can_have_multiple_field_edits_but_counts_once_as_source_event() {
        let reference = graph(fixture());
        let prediction = graph(&fixture().replace(
            r#"<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><instrument id="P1-I1"/></note>"#,
            r#"<note><pitch><step>D</step><octave>4</octave></pitch><duration>8</duration><voice>2</voice><staff>2</staff><instrument id="P1-I2"/></note>"#
        ));
        let ledger = build_musicxml_correction_ledger(
            &reference, &prediction, config(), MusicXmlCorrectionLedgerConfigV1::default()
        ).unwrap();
        assert!(ledger.action_count >= 4);
        assert_eq!(ledger.affected_reference_source_event_count, 1);
        assert_eq!(ledger.affected_prediction_source_event_count, 1);
        assert!(ledger.by_field.contains_key("written_pitch_spelling"));
        assert!(ledger.by_field.contains_key("performed_duration"));
        assert!(ledger.by_field.contains_key("voice"));
        assert!(ledger.by_field.contains_key("staff"));
    }

    #[test]
    fn part_metadata_changes_are_independent_and_legacy_metadata_abstains() {
        let reference = graph(fixture());
        let prediction = graph(&fixture().replace("<part-name>Violin</part-name>","<part-name>Viola</part-name>")
            .replace("<midi-channel>1</midi-channel>","<midi-channel>2</midi-channel>"));
        let ledger = build_musicxml_correction_ledger(
            &reference, &prediction, config(), MusicXmlCorrectionLedgerConfigV1::default()
        ).unwrap();
        assert!(ledger.part_metadata_actions_covered);
        assert!(ledger.by_field.contains_key("part.name"));
        assert!(ledger.by_field.contains_key("part.midi_instrument_setup"));

        let mut legacy = reference.clone();
        legacy.part_metadata = None;
        legacy.part_metadata_layer_present = false;
        let legacy_ledger = build_musicxml_correction_ledger(
            &legacy, &prediction, config(), MusicXmlCorrectionLedgerConfigV1::default()
        ).unwrap();
        assert!(!legacy_ledger.part_metadata_actions_covered);
        assert!(!legacy_ledger.actions.iter().any(|action| action.family == MusicXmlCorrectionFamilyV1::PartMetadata));
    }

    #[test]
    fn cursor_auxiliary_attribute_and_measure_differences_are_localized() {
        let reference = graph(fixture());
        let prediction = graph(&fixture()
            .replace("tempo=\"90\"","tempo=\"110\"")
            .replace("<step>G</step>","<step>F</step>"));
        let ledger = build_musicxml_correction_ledger(
            &reference, &prediction, config(), MusicXmlCorrectionLedgerConfigV1::default()
        ).unwrap();
        assert!(ledger.actions.iter().any(|action| action.family == MusicXmlCorrectionFamilyV1::AuxiliaryEvent));
    }

    #[test]
    fn action_budget_overflow_fails_closed() {
        let reference = graph(fixture());
        let prediction = graph(&fixture().replace(
            "<step>C</step><octave>4</octave>",
            "<step>D</step><octave>4</octave>"
        ));
        let mut budget = MusicXmlCorrectionLedgerConfigV1::default();
        budget.maximum_actions = 1;
        assert!(matches!(
            build_musicxml_correction_ledger(&reference,&prediction,config(),budget),
            Err(MusicXmlCorrectionLedgerErrorV1::ActionBudgetExceeded { .. })
        ));
    }

    #[test]
    fn replay_rejects_tampered_action_digest_and_changed_inputs() {
        let reference = graph(fixture());
        let prediction = graph(&fixture().replace(
            "<step>C</step><octave>4</octave>",
            "<step>D</step><octave>4</octave>"
        ));
        let config = MusicXmlCorrectionLedgerConfigV1::default();
        let mut ledger = build_musicxml_correction_ledger(
            &reference, &prediction, config(), config.clone()
        ).unwrap();
        assert!(validate_musicxml_correction_ledger_replay(
            &reference,&prediction,config(),config.clone(),&ledger
        ).unwrap());
        ledger.actions[0].field_path = "tampered".into();
        assert!(!validate_musicxml_correction_ledger_replay(
            &reference,&prediction,config(),config.clone(),&ledger
        ).unwrap());
        let changed = graph(&fixture().replace("<part-name>Violin</part-name>","<part-name>Cello</part-name>"));
        assert!(!validate_musicxml_correction_ledger_replay(
            &changed,&prediction,config(),config,&build_musicxml_correction_ledger(
                &reference,&prediction,config(),MusicXmlCorrectionLedgerConfigV1::default()
            ).unwrap()
        ).unwrap());
    }
}
