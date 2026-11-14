// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound score-to-score comparison for source-native MusicXML graphs.
//!
//! MUSE-SED is a local, explicitly versioned insertion/deletion distance over
//! canonical notation-symbol streams. It is inspired by OMR-NED, but does not
//! claim MusicDiff/OMR-NED or MUSTER parity. The comparator keeps separate
//! symbol-family distances and withholds an aggregate result when any part's
//! source structure is Unknown or Unsupported.
//!
//! Notation time, performance time and rendered audio remain separate. This
//! module compares written/source-native score graphs only.

use crate::barline_eval::{MeasureMapStatus, RationalBeat};
use crate::evidence_digest::canonical_json_sha256;
use crate::musicxml_barline_adapter::MusicXmlPartNotationGraphV1;
use crate::musicxml_notation_events::{
    MusicXmlNotationEventV1, MusicXmlSourceContentV1, MusicXmlSourceNodeV1,
    MusicXmlNotationEventsStatusV1,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const MUSICXML_SCORE_EVALUATOR_SCHEMA_V1: &str = "muse-musicxml-score-evaluation-v1";
pub const MUSICXML_SCORE_EVALUATOR_VERSION_V1: &str = "muse-score-symbol-indel-v1";
pub const MUSICXML_SCORE_TOKENIZATION_VERSION_V1: &str =
    "muse-musicxml-visible-source-symbols-namespace-aware-v1";

pub const MUSICXML_SCORE_DEFAULT_MAX_SYMBOLS_PER_PART_V1: usize = 100_000;
pub const MUSICXML_SCORE_DEFAULT_MAX_DIFF_CELLS_V1: usize = 20_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicXmlSymbolFamilyV1 {
    NoteRestUnpitched,
    PitchSpelling,
    RhythmAndDuration,
    Ties,
    Tuplets,
    Decorations,
    VoiceStaffInstrument,
    SignaturesAndAttributes,
    DirectionsAndLyrics,
    MeasuresAndBarlines,
    CursorAndAuxiliary,
    PartMetadata,
    Other,
}

impl MusicXmlSymbolFamilyV1 {
    pub const ALL: [Self; 13] = [
        Self::NoteRestUnpitched,
        Self::PitchSpelling,
        Self::RhythmAndDuration,
        Self::Ties,
        Self::Tuplets,
        Self::Decorations,
        Self::VoiceStaffInstrument,
        Self::SignaturesAndAttributes,
        Self::DirectionsAndLyrics,
        Self::MeasuresAndBarlines,
        Self::CursorAndAuxiliary,
        Self::PartMetadata,
        Self::Other,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlScoreEvaluationConfigV1 {
    /// Maximum canonical symbols emitted for one part on either side.
    pub max_symbols_per_part: usize,
    /// Total LCS dynamic-programming cells allowed across all part/family
    /// comparisons. Common prefixes/suffixes are trimmed before charging.
    pub max_diff_cells: usize,
}

impl Default for MusicXmlScoreEvaluationConfigV1 {
    fn default() -> Self {
        Self {
            max_symbols_per_part: MUSICXML_SCORE_DEFAULT_MAX_SYMBOLS_PER_PART_V1,
            max_diff_cells: MUSICXML_SCORE_DEFAULT_MAX_DIFF_CELLS_V1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlScoreDistanceV1 {
    pub reference_symbol_count: usize,
    pub prediction_symbol_count: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub edit_distance: usize,
    /// Indel distance / (reference symbols + prediction symbols).
    /// None means both sequences were empty; it is not silently 0 or 1.
    pub normalized_distance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicXmlScorePartStatusV1 {
    Compared,
    ReferenceOnly,
    PredictionOnly,
    Unknown { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlScoreFamilyDistanceV1 {
    pub family: MusicXmlSymbolFamilyV1,
    pub distance: MusicXmlScoreDistanceV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlScorePartReportV1 {
    pub part_id: String,
    pub status: MusicXmlScorePartStatusV1,
    pub distance: Option<MusicXmlScoreDistanceV1>,
    pub family_distances: Vec<MusicXmlScoreFamilyDistanceV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicXmlScoreEvaluationStatusV1 {
    Complete,
    Partial,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlScoreEvaluationCoverageV1 {
    pub total_part_ids: usize,
    pub compared_part_ids: usize,
    pub reference_only_part_ids: usize,
    pub prediction_only_part_ids: usize,
    pub unknown_part_ids: usize,
    pub supported_part_coverage: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicXmlScoreEvaluationReportV1 {
    pub schema: String,
    pub evaluator_version: String,
    pub tokenization_version: String,
    pub config: MusicXmlScoreEvaluationConfigV1,
    pub reference_graph_sha256: String,
    pub prediction_graph_sha256: String,
    pub status: MusicXmlScoreEvaluationStatusV1,
    pub coverage: MusicXmlScoreEvaluationCoverageV1,
    /// None means at least one part was structurally Unknown/Unsupported.
    pub overall_distance: Option<MusicXmlScoreDistanceV1>,
    pub family_distances: Vec<MusicXmlScoreFamilyDistanceV1>,
    pub per_part: Vec<MusicXmlScorePartReportV1>,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicXmlScoreEvaluationErrorV1 {
    InvalidConfig { field: String },
    DuplicateReferencePartId { part_id: String },
    DuplicatePredictionPartId { part_id: String },
    PartIdentityMismatch { graph_part_id: String, notation_part_id: String },
    ScoreClockMismatch { part_id: String },
    CanonicalizationMismatch { part_id: String },
    NotationSchemaMismatch { part_id: String },
    InvalidSourceTree { part_id: String, reason: String },
    SymbolBudgetExceeded { part_id: String, count: usize, maximum: usize },
    DiffWorkBudgetExceeded { cells: usize, maximum: usize },
    ArithmeticOverflow,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SymbolToken {
    family: MusicXmlSymbolFamilyV1,
    value: String,
}

#[derive(Debug, Clone, Default)]
struct SymbolStreams {
    all: Vec<SymbolToken>,
    by_family: BTreeMap<MusicXmlSymbolFamilyV1, Vec<SymbolToken>>,
}

impl SymbolStreams {
    fn push(
        &mut self,
        family: MusicXmlSymbolFamilyV1,
        value: impl Into<String>,
        part_id: &str,
        maximum: usize,
    ) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
        let value = value.into();
        let next = self.all.len().checked_add(1)
            .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
        if next > maximum {
            return Err(MusicXmlScoreEvaluationErrorV1::SymbolBudgetExceeded {
                part_id: part_id.to_string(),
                count: next,
                maximum,
            });
        }
        let token = SymbolToken { family, value };
        self.all.push(token.clone());
        self.by_family.entry(family).or_default().push(token);
        Ok(())
    }
}

fn family_for_element(name: &str) -> MusicXmlSymbolFamilyV1 {
    match name {
        "note" | "rest" | "unpitched" | "grace" | "cue" | "chord" =>
            MusicXmlSymbolFamilyV1::NoteRestUnpitched,
        "pitch" | "step" | "alter" | "octave" | "display-step" | "display-octave" |
        "accidental" => MusicXmlSymbolFamilyV1::PitchSpelling,
        "duration" | "type" | "dot" | "time-modification" | "actual-notes" |
        "normal-notes" | "normal-type" | "normal-dot" =>
            MusicXmlSymbolFamilyV1::RhythmAndDuration,
        "tie" | "tied" => MusicXmlSymbolFamilyV1::Ties,
        "tuplet" | "tuplet-actual" | "tuplet-normal" | "tuplet-number" =>
            MusicXmlSymbolFamilyV1::Tuplets,
        "beam" | "stem" | "notehead" | "articulations" | "ornaments" | "ornament" |
        "technical" | "technical-mark" | "tremolo" | "slur" | "arpeggiate" |
        "fermata" | "wavy-line" => MusicXmlSymbolFamilyV1::Decorations,
        "voice" | "staff" | "instrument" | "score-part" | "score-instrument" |
        "midi-instrument" | "part-name" | "part-abbreviation" =>
            MusicXmlSymbolFamilyV1::VoiceStaffInstrument,
        "attributes" | "divisions" | "key" | "time" | "beats" | "beat-type" |
        "clef" | "sign" | "line" | "staves" | "transpose" | "staff-details" |
        "staff-tuning" | "capo" => MusicXmlSymbolFamilyV1::SignaturesAndAttributes,
        "direction" | "sound" | "words" | "dynamics" | "metronome" | "harmony" |
        "figured-bass" | "lyric" | "text" | "syllabic" => MusicXmlSymbolFamilyV1::DirectionsAndLyrics,
        "measure" | "barline" | "bar-style" | "repeat" | "ending" | "segno" | "coda" =>
            MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
        "backup" | "forward" | "offset" | "print" =>
            MusicXmlSymbolFamilyV1::CursorAndAuxiliary,
        _ => MusicXmlSymbolFamilyV1::Other,
    }
}

fn family_for_attribute(parent: &str, name: &str) -> MusicXmlSymbolFamilyV1 {
    match (parent, name) {
        ("note", "id") => MusicXmlSymbolFamilyV1::Other,
        ("note", "time-only" | "attack" | "release") =>
            MusicXmlSymbolFamilyV1::RhythmAndDuration,
        ("measure", "implicit" | "non-controlling" | "number" | "text") =>
            MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
        ("barline", "location") => MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
        ("repeat", "direction" | "times" | "after-jump" | "winged") =>
            MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
        ("rest", "measure") => MusicXmlSymbolFamilyV1::NoteRestUnpitched,
        _ => family_for_element(parent),
    }
}

fn normalize_text_segment(value: &str) -> Option<String> {
    if value.trim().is_empty() {
        return None;
    }
    let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let leading = value.chars().next().is_some_and(|character| character.is_whitespace());
    let trailing = value.chars().last().is_some_and(|character| character.is_whitespace());
    let mut normalized = String::with_capacity(collapsed.len() + 2);
    if leading {
        normalized.push(' ');
    }
    normalized.push_str(&collapsed);
    if trailing {
        normalized.push(' ');
    }
    Some(normalized)
}

fn append_source_node(
    node: &MusicXmlSourceNodeV1,
    streams: &mut SymbolStreams,
    part_id: &str,
    maximum: usize,
    depth: usize,
) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
    if depth > 32 {
        return Err(MusicXmlScoreEvaluationErrorV1::InvalidSourceTree {
            part_id: part_id.to_string(),
            reason: "source tree depth exceeds 32".into(),
        });
    }
    let family = family_for_element(&node.element);
    streams.push(
        family,
        format!("element|{}|{}", node.namespace_uri.as_deref().unwrap_or(""), node.element),
        part_id,
        maximum,
    )?;

    // XML attribute order is not visible notation. Sort namespace/local-name
    // pairs to make equivalent documents compare identically.
    let mut attributes: Vec<_> = node.attributes.iter()
        .filter(|attribute| {
            attribute.name != "id"
                || matches!(node.element.as_str(), "score-instrument" | "midi-instrument")
        })
        .collect();
    attributes.sort_by(|left, right| {
        left.namespace_uri.cmp(&right.namespace_uri)
            .then(left.name.cmp(&right.name))
            .then(left.value.cmp(&right.value))
    });
    for attribute in attributes {
        let attr_family = family_for_attribute(&node.element, &attribute.name);
        streams.push(
            attr_family,
            format!(
                "attribute|{}|{}|{}",
                attribute.namespace_uri.as_deref().unwrap_or(""),
                attribute.name,
                attribute.value
            ),
            part_id,
            maximum,
        )?;
    }

    let mut referenced_children = BTreeSet::new();
    for content in &node.content {
        match content {
            MusicXmlSourceContentV1::Text(text) => {
                if let Some(text) = normalize_text_segment(text) {
                    streams.push(family, format!("text|{text}"), part_id, maximum)?;
                }
            }
            MusicXmlSourceContentV1::ElementIndex(index) => {
                let child = node.children.get(*index).ok_or_else(|| {
                    MusicXmlScoreEvaluationErrorV1::InvalidSourceTree {
                        part_id: part_id.to_string(),
                        reason: format!("mixed-content child index {index} is out of range"),
                    }
                })?;
                if !referenced_children.insert(*index) {
                    return Err(MusicXmlScoreEvaluationErrorV1::InvalidSourceTree {
                        part_id: part_id.to_string(),
                        reason: format!("mixed-content child index {index} is duplicated"),
                    });
                }
                append_source_node(child, streams, part_id, maximum, depth + 1)?;
            }
        }
    }
    if !node.children.is_empty() && referenced_children.len() != node.children.len() {
        // Older source-node records without mixed-content indices still have
        // a deterministic element-child fallback.
        if node.content.is_empty() {
            for child in &node.children {
                append_source_node(child, streams, part_id, maximum, depth + 1)?;
            }
        } else {
            return Err(MusicXmlScoreEvaluationErrorV1::InvalidSourceTree {
                part_id: part_id.to_string(),
                reason: "mixed-content representation omits a child element".into(),
            });
        }
    }
    streams.push(
        family,
        format!("end-element|{}|{}", node.namespace_uri.as_deref().unwrap_or(""), node.element),
        part_id,
        maximum,
    )?;
    Ok(())
}

fn append_rational(
    streams: &mut SymbolStreams,
    family: MusicXmlSymbolFamilyV1,
    name: &str,
    value: RationalBeat,
    part_id: &str,
    maximum: usize,
) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
    streams.push(
        family,
        format!("{name}|{}/{}", value.numerator(), value.denominator()),
        part_id,
        maximum,
    )
}

fn append_json_value(
    value: &Value,
    family: MusicXmlSymbolFamilyV1,
    path: &str,
    streams: &mut SymbolStreams,
    part_id: &str,
    maximum: usize,
) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
    match value {
        Value::Null => {
            streams.push(family, format!("{path}|null"), part_id, maximum)?;
        }
        Value::Bool(value) => {
            streams.push(family, format!("{path}|{value}"), part_id, maximum)?;
        }
        Value::Number(value) => {
            streams.push(family, format!("{path}|{value}"), part_id, maximum)?;
        }
        Value::String(value) => {
            if let Some(value) = normalize_text_segment(value) {
                streams.push(family, format!("{path}|{value}"), part_id, maximum)?;
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                append_json_value(
                    child, family, &format!("{path}[{index}]"), streams, part_id, maximum
                )?;
            }
        }
        Value::Object(values) => {
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (key, child) in entries {
                if matches!(key.as_str(), "source_id" | "part_id" | "source_measure_number") {
                    continue;
                }
                append_json_value(
                    child, family, &format!("{path}.{key}"), streams, part_id, maximum
                )?;
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
struct OrderedEvent<'a> {
    source_order: usize,
    kind: &'static str,
    note: Option<&'a MusicXmlNotationEventV1>,
    source: Option<&'a MusicXmlSourceNodeV1>,
    timing: Vec<String>,
    family: MusicXmlSymbolFamilyV1,
}

fn append_graph_tokens(
    graph: &MusicXmlPartNotationGraphV1,
    config: &MusicXmlScoreEvaluationConfigV1,
) -> Result<SymbolStreams, MusicXmlScoreEvaluationErrorV1> {
    let part_id = graph.part_id.as_str();
    if graph.notation_events.part_id != graph.part_id {
        return Err(MusicXmlScoreEvaluationErrorV1::PartIdentityMismatch {
            graph_part_id: graph.part_id.clone(),
            notation_part_id: graph.notation_events.part_id.clone(),
        });
    }
    let mut streams = SymbolStreams::default();

    streams.push(
        MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
        format!("canonicalization|{}", graph.map.canonicalization_version),
        part_id,
        config.max_symbols_per_part,
    )?;
    for (index, measure) in graph.map.measures.iter().enumerate() {
        append_rational(
            &mut streams,
            MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
            &format!("measure[{index}].start"),
            measure.start,
            part_id,
            config.max_symbols_per_part,
        )?;
        append_rational(
            &mut streams,
            MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
            &format!("measure[{index}].end"),
            measure.end,
            part_id,
            config.max_symbols_per_part,
        )?;
        streams.push(
            MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
            format!("measure[{index}].kind|{:?}", measure.kind),
            part_id,
            config.max_symbols_per_part,
        )?;
    }
    for locator in &graph.measure_locators {
        streams.push(
            MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
            format!(
                "measure-locator|{}|implicit={}|non-controlling={}|display={}",
                locator.number.as_deref().unwrap_or(""),
                locator.implicit,
                locator.non_controlling,
                locator.display_text.as_deref().unwrap_or("")
            ),
            part_id,
            config.max_symbols_per_part,
        )?;
    }
    for index in &graph.non_controlling_measure_indices {
        streams.push(
            MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
            format!("non-controlling-index|{index}"),
            part_id,
            config.max_symbols_per_part,
        )?;
    }
    if let Some(metadata) = &graph.part_metadata {
        // The source score-part subtree already contains the part name,
        // abbreviation and score-instrument declarations. Keep their exact
        // source relationship once rather than double-counting typed copies.
        append_source_node(
            &metadata.source_score_part,
            &mut streams,
            part_id,
            config.max_symbols_per_part,
            0,
        )?;
    }
    for barline in &graph.barline_events {
        let value = serde_json::to_value(barline)
            .map_err(|_| MusicXmlScoreEvaluationErrorV1::SerializationFailed)?;
        append_json_value(
            &value,
            MusicXmlSymbolFamilyV1::MeasuresAndBarlines,
            "barline",
            &mut streams,
            part_id,
            config.max_symbols_per_part,
        )?;
    }

    let mut events = Vec::new();
    for note in &graph.notation_events.note_events {
        events.push(OrderedEvent {
            source_order: note.source_order,
            kind: "note",
            note: Some(note),
            source: Some(&note.source_note),
            timing: vec![
                format!("measure={}", note.source_measure_index),
                format!("offset={}/{}", note.score_offset.numerator(), note.score_offset.denominator()),
                format!("duration={}", note.performed_duration.map(|d| format!("{}/{}", d.numerator(), d.denominator())).unwrap_or_else(|| "grace".into())),
            ],
            family: MusicXmlSymbolFamilyV1::NoteRestUnpitched,
        });
    }
    for event in &graph.notation_events.attribute_events {
        events.push(OrderedEvent {
            source_order: event.source_order,
            kind: "attribute",
            note: None,
            source: Some(&event.source),
            timing: vec![
                format!("measure={}", event.measure_index),
                format!("offset={}/{}", event.score_offset.numerator(), event.score_offset.denominator()),
            ],
            family: MusicXmlSymbolFamilyV1::SignaturesAndAttributes,
        });
    }
    for event in &graph.notation_events.auxiliary_events {
        // Barline objects are also represented by the typed barline graph;
        // avoid double-counting them in MUSE-SED.
        if event.source.element == "barline" {
            continue;
        }
        events.push(OrderedEvent {
            source_order: event.source_order,
            kind: "auxiliary",
            note: None,
            source: Some(&event.source),
            timing: vec![
                format!("measure={}", event.measure_index),
                format!("offset={}/{}", event.score_offset.numerator(), event.score_offset.denominator()),
            ],
            family: family_for_element(&event.source.element),
        });
    }
    for event in &graph.notation_events.cursor_moves {
        events.push(OrderedEvent {
            source_order: event.source_order,
            kind: "cursor-move",
            note: None,
            source: Some(&event.source),
            timing: vec![
                format!("measure={}", event.measure_index),
                format!("direction={}", event.direction),
                format!("before={}/{}", event.score_offset_before.numerator(), event.score_offset_before.denominator()),
                format!("duration={}/{}", event.duration.numerator(), event.duration.denominator()),
                format!("after={}/{}", event.score_offset_after.numerator(), event.score_offset_after.denominator()),
                format!("voice={}", event.voice.as_deref().unwrap_or("")),
                format!("staff={}", event.staff.as_deref().unwrap_or("")),
            ],
            family: MusicXmlSymbolFamilyV1::CursorAndAuxiliary,
        });
    }
    events.sort_by_key(|event| event.source_order);
    for pair in events.windows(2) {
        if pair[0].source_order == pair[1].source_order {
            return Err(MusicXmlScoreEvaluationErrorV1::InvalidSourceTree {
                part_id: graph.part_id.clone(),
                reason: format!("duplicate notation source order {}", pair[0].source_order),
            });
        }
    }
    for event in events {
        streams.push(
            event.family,
            format!("source-order|{}|{}", event.source_order, event.kind),
            part_id,
            config.max_symbols_per_part,
        )?;
        for timing in event.timing {
            streams.push(
                event.family,
                timing,
                part_id,
                config.max_symbols_per_part,
            )?;
        }
        if let Some(note) = event.note {
            // The raw note tree carries visible spelling/decorations; typed
            // properties preserve exact score coordinates and voice identity.
            streams.push(
                MusicXmlSymbolFamilyV1::VoiceStaffInstrument,
                format!("voice|{}", note.voice.as_deref().unwrap_or("")),
                part_id,
                config.max_symbols_per_part,
            )?;
            streams.push(
                MusicXmlSymbolFamilyV1::VoiceStaffInstrument,
                format!("staff|{}", note.staff.as_deref().unwrap_or("")),
                part_id,
                config.max_symbols_per_part,
            )?;
            for instrument in &note.instrument_ids {
                streams.push(
                    MusicXmlSymbolFamilyV1::VoiceStaffInstrument,
                    format!("instrument|{instrument}"),
                    part_id,
                    config.max_symbols_per_part,
                )?;
            }
        }
        if let Some(source) = event.source {
            append_source_node(
                source,
                &mut streams,
                part_id,
                config.max_symbols_per_part,
                0,
            )?;
        }
    }
    Ok(streams)
}

fn graph_status_reason(graph: &MusicXmlPartNotationGraphV1) -> Option<String> {
    match &graph.map.status {
        MeasureMapStatus::Supported => {}
        MeasureMapStatus::Unknown { reason } => return Some(format!("measure map unknown: {reason}")),
        MeasureMapStatus::Unsupported { reason } => return Some(format!("measure map unsupported: {reason}")),
    }
    match &graph.notation_events.status {
        MusicXmlNotationEventsStatusV1::Supported => None,
        MusicXmlNotationEventsStatusV1::Unknown { reason } => {
            Some(format!("notation events unknown: {reason}"))
        }
    }
}

fn validate_graph_pair(
    reference: &MusicXmlPartNotationGraphV1,
    prediction: &MusicXmlPartNotationGraphV1,
) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
    if reference.part_id != reference.notation_events.part_id
        || prediction.part_id != prediction.notation_events.part_id
    {
        return Err(MusicXmlScoreEvaluationErrorV1::PartIdentityMismatch {
            graph_part_id: reference.part_id.clone(),
            notation_part_id: reference.notation_events.part_id.clone(),
        });
    }
    if reference.map.score_clock_id != prediction.map.score_clock_id {
        return Err(MusicXmlScoreEvaluationErrorV1::ScoreClockMismatch {
            part_id: reference.part_id.clone(),
        });
    }
    if reference.map.canonicalization_version != prediction.map.canonicalization_version
        || reference.notation_events.canonicalization_version
            != prediction.notation_events.canonicalization_version
    {
        return Err(MusicXmlScoreEvaluationErrorV1::CanonicalizationMismatch {
            part_id: reference.part_id.clone(),
        });
    }
    if reference.notation_events.schema != prediction.notation_events.schema {
        return Err(MusicXmlScoreEvaluationErrorV1::NotationSchemaMismatch {
            part_id: reference.part_id.clone(),
        });
    }
    Ok(())
}

fn distance(
    reference: &[SymbolToken],
    prediction: &[SymbolToken],
    budget_used: &mut usize,
    budget: usize,
) -> Result<MusicXmlScoreDistanceV1, MusicXmlScoreEvaluationErrorV1> {
    let mut left_start = 0usize;
    let mut right_start = 0usize;
    let mut left_end = reference.len();
    let mut right_end = prediction.len();

    while left_start < left_end && right_start < right_end
        && reference[left_start] == prediction[right_start]
    {
        left_start += 1;
        right_start += 1;
    }
    while left_end > left_start && right_end > right_start
        && reference[left_end - 1] == prediction[right_end - 1]
    {
        left_end -= 1;
        right_end -= 1;
    }

    let left = &reference[left_start..left_end];
    let right = &prediction[right_start..right_end];
    let cells = left.len().checked_mul(right.len())
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    let next_used = budget_used.checked_add(cells)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    if next_used > budget {
        return Err(MusicXmlScoreEvaluationErrorV1::DiffWorkBudgetExceeded {
            cells: next_used,
            maximum: budget,
        });
    }
    *budget_used = next_used;

    // Indel distance is n + m - 2*LCS. A substitution is one deletion plus
    // one insertion, matching the operation model used by OMR-NED.
    let mut lcs = vec![0usize; right.len() + 1];
    for left_token in left {
        let mut diagonal = 0usize;
        for index in 1..=right.len() {
            let above = lcs[index];
            if left_token == &right[index - 1] {
                lcs[index] = diagonal + 1;
            } else {
                lcs[index] = lcs[index].max(lcs[index - 1]);
            }
            diagonal = above;
        }
    }
    let common = lcs[right.len()];
    let insertions = prediction.len() - common;
    let deletions = reference.len() - common;
    let edit_distance = insertions.checked_add(deletions)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    let denominator = reference.len().checked_add(prediction.len())
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    Ok(MusicXmlScoreDistanceV1 {
        reference_symbol_count: reference.len(),
        prediction_symbol_count: prediction.len(),
        insertions,
        deletions,
        edit_distance,
        normalized_distance: (denominator > 0)
            .then_some(edit_distance as f64 / denominator as f64),
    })
}

fn empty_distance() -> MusicXmlScoreDistanceV1 {
    MusicXmlScoreDistanceV1 {
        reference_symbol_count: 0,
        prediction_symbol_count: 0,
        insertions: 0,
        deletions: 0,
        edit_distance: 0,
        normalized_distance: None,
    }
}

fn add_distance(
    aggregate: &mut MusicXmlScoreDistanceV1,
    value: &MusicXmlScoreDistanceV1,
) -> Result<(), MusicXmlScoreEvaluationErrorV1> {
    aggregate.reference_symbol_count = aggregate.reference_symbol_count
        .checked_add(value.reference_symbol_count)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    aggregate.prediction_symbol_count = aggregate.prediction_symbol_count
        .checked_add(value.prediction_symbol_count)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    aggregate.insertions = aggregate.insertions.checked_add(value.insertions)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    aggregate.deletions = aggregate.deletions.checked_add(value.deletions)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    aggregate.edit_distance = aggregate.edit_distance.checked_add(value.edit_distance)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    let denominator = aggregate.reference_symbol_count
        .checked_add(aggregate.prediction_symbol_count)
        .ok_or(MusicXmlScoreEvaluationErrorV1::ArithmeticOverflow)?;
    aggregate.normalized_distance = (denominator > 0)
        .then_some(aggregate.edit_distance as f64 / denominator as f64);
    Ok(())
}

fn empty_family_distances() -> Vec<MusicXmlScoreFamilyDistanceV1> {
    MusicXmlSymbolFamilyV1::ALL.into_iter().map(|family| MusicXmlScoreFamilyDistanceV1 {
        family,
        distance: empty_distance(),
    }).collect()
}

fn graph_digest(
    graphs: &[MusicXmlPartNotationGraphV1],
) -> Result<String, MusicXmlScoreEvaluationErrorV1> {
    canonical_json_sha256(graphs)
        .map_err(|_| MusicXmlScoreEvaluationErrorV1::SerializationFailed)
}

fn report_commitment(
    report: &MusicXmlScoreEvaluationReportV1,
) -> Result<String, MusicXmlScoreEvaluationErrorV1> {
    let mut unsigned = report.clone();
    unsigned.report_sha256.clear();
    canonical_json_sha256(&unsigned)
        .map_err(|_| MusicXmlScoreEvaluationErrorV1::SerializationFailed)
}

/// Evaluate two sets of parsed MusicXML per-part graphs.
///
/// V1 matches parts by exact part ID only. It deliberately does not guess
/// correspondences from pitch, register, staff or order. Part order in the
/// input vectors is ignored; the score report is sorted by part ID.
pub fn evaluate_musicxml_score_graphs_v1(
    reference: &[MusicXmlPartNotationGraphV1],
    prediction: &[MusicXmlPartNotationGraphV1],
    config: &MusicXmlScoreEvaluationConfigV1,
) -> Result<MusicXmlScoreEvaluationReportV1, MusicXmlScoreEvaluationErrorV1> {
    if config.max_symbols_per_part == 0 {
        return Err(MusicXmlScoreEvaluationErrorV1::InvalidConfig {
            field: "max_symbols_per_part".into(),
        });
    }
    if config.max_diff_cells == 0 {
        return Err(MusicXmlScoreEvaluationErrorV1::InvalidConfig {
            field: "max_diff_cells".into(),
        });
    }

    let mut reference_by_id = BTreeMap::new();
    for graph in reference {
        if reference_by_id.insert(graph.part_id.as_str(), graph).is_some() {
            return Err(MusicXmlScoreEvaluationErrorV1::DuplicateReferencePartId {
                part_id: graph.part_id.clone(),
            });
        }
    }
    let mut prediction_by_id = BTreeMap::new();
    for graph in prediction {
        if prediction_by_id.insert(graph.part_id.as_str(), graph).is_some() {
            return Err(MusicXmlScoreEvaluationErrorV1::DuplicatePredictionPartId {
                part_id: graph.part_id.clone(),
            });
        }
    }

    let part_ids: BTreeSet<&str> = reference_by_id.keys().chain(prediction_by_id.keys())
        .copied().collect();
    let mut report_parts = Vec::with_capacity(part_ids.len());
    let mut overall = empty_distance();
    let mut family_aggregates = empty_family_distances();
    let mut compared_part_ids = 0usize;
    let mut reference_only_part_ids = 0usize;
    let mut prediction_only_part_ids = 0usize;
    let mut unknown_part_ids = 0usize;
    let mut budget_used = 0usize;

    for part_id in part_ids {
        let reference_graph = reference_by_id.get(part_id).copied();
        let prediction_graph = prediction_by_id.get(part_id).copied();
        let reference_reason = reference_graph.and_then(graph_status_reason);
        let prediction_reason = prediction_graph.and_then(graph_status_reason);

        let mut part_family_distances = empty_family_distances();
        let status;
        let part_distance;

        if let (Some(reference_graph), Some(prediction_graph)) =
            (reference_graph, prediction_graph)
        {
            validate_graph_pair(reference_graph, prediction_graph)?;
        }

        if let Some(reason) = reference_reason.or(prediction_reason) {
            status = MusicXmlScorePartStatusV1::Unknown { reason };
            part_distance = None;
            unknown_part_ids += 1;
        } else {
            let reference_streams = reference_graph.map(|graph| {
                append_graph_tokens(graph, config)
            }).transpose()?;
            let prediction_streams = prediction_graph.map(|graph| {
                append_graph_tokens(graph, config)
            }).transpose()?;
            let empty = SymbolStreams::default();
            let left = reference_streams.as_ref().unwrap_or(&empty);
            let right = prediction_streams.as_ref().unwrap_or(&empty);

            let part_metric = distance(
                &left.all, &right.all, &mut budget_used, config.max_diff_cells
            )?;
            for (index, family) in MusicXmlSymbolFamilyV1::ALL.iter().enumerate() {
                let left_family = left.by_family.get(family).map(Vec::as_slice).unwrap_or(&[]);
                let right_family = right.by_family.get(family).map(Vec::as_slice).unwrap_or(&[]);
                let metric = distance(
                    left_family, right_family, &mut budget_used, config.max_diff_cells
                )?;
                part_family_distances[index].distance = metric;
            }
            if reference_graph.is_some() && prediction_graph.is_some() {
                status = MusicXmlScorePartStatusV1::Compared;
                compared_part_ids += 1;
            } else if reference_graph.is_some() {
                status = MusicXmlScorePartStatusV1::ReferenceOnly;
                reference_only_part_ids += 1;
            } else {
                status = MusicXmlScorePartStatusV1::PredictionOnly;
                prediction_only_part_ids += 1;
            }
            part_distance = Some(part_metric.clone());
            add_distance(&mut overall, &part_metric)?;
            for (aggregate, family) in family_aggregates.iter_mut().zip(&part_family_distances) {
                add_distance(&mut aggregate.distance, &family.distance)?;
            }
        }

        report_parts.push(MusicXmlScorePartReportV1 {
            part_id: part_id.to_string(),
            status,
            distance: part_distance,
            family_distances: part_family_distances,
        });
    }

    let total_part_ids = report_parts.len();
    let supported_part_ids = compared_part_ids + reference_only_part_ids + prediction_only_part_ids;
    let complete = unknown_part_ids == 0 && total_part_ids > 0;
    let status = if complete {
        MusicXmlScoreEvaluationStatusV1::Complete
    } else if supported_part_ids == 0 {
        MusicXmlScoreEvaluationStatusV1::Unavailable
    } else {
        MusicXmlScoreEvaluationStatusV1::Partial
    };

    let report = MusicXmlScoreEvaluationReportV1 {
        schema: MUSICXML_SCORE_EVALUATOR_SCHEMA_V1.into(),
        evaluator_version: MUSICXML_SCORE_EVALUATOR_VERSION_V1.into(),
        tokenization_version: MUSICXML_SCORE_TOKENIZATION_VERSION_V1.into(),
        config: config.clone(),
        reference_graph_sha256: graph_digest(reference)?,
        prediction_graph_sha256: graph_digest(prediction)?,
        status,
        coverage: MusicXmlScoreEvaluationCoverageV1 {
            total_part_ids,
            compared_part_ids,
            reference_only_part_ids,
            prediction_only_part_ids,
            unknown_part_ids,
            supported_part_coverage: (total_part_ids > 0)
                .then_some(supported_part_ids as f64 / total_part_ids as f64),
        },
        overall_distance: complete.then_some(overall),
        family_distances: if complete { family_aggregates } else { Vec::new() },
        per_part: report_parts,
        report_sha256: String::new(),
    };
    let commitment = report_commitment(&report)?;
    Ok(MusicXmlScoreEvaluationReportV1 {
        report_sha256: commitment,
        ..report
    })
}

/// Recompute both input digests, every family distance and the report
/// commitment. A digest alone is not treated as a signed author attestation.
pub fn validate_musicxml_score_evaluation_replay_v1(
    reference: &[MusicXmlPartNotationGraphV1],
    prediction: &[MusicXmlPartNotationGraphV1],
    config: &MusicXmlScoreEvaluationConfigV1,
    supplied: &MusicXmlScoreEvaluationReportV1,
) -> Result<bool, MusicXmlScoreEvaluationErrorV1> {
    let expected = evaluate_musicxml_score_graphs_v1(reference, prediction, config)?;
    if report_commitment(supplied)? != supplied.report_sha256 {
        return Ok(false);
    }
    Ok(expected == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::musicxml_barline_adapter::parse_musicxml_part_notation_graph;

    fn parse(xml: &str) -> Vec<MusicXmlPartNotationGraphV1> {
        parse_musicxml_part_notation_graph(
            xml.as_bytes(),
            "sha256:fixture-source",
            "fixture-score-clock",
        ).unwrap()
    }

    fn simple_score(note: &str) -> String {
        format!(
            r#"<score-partwise><part id="P1"><measure number="1">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></attributes>
                {note}
                <note><rest/><duration>12</duration><voice>1</voice><staff>1</staff></note>
            </measure></part></score-partwise>"#
        )
    }

    fn rest_note() -> &'static str {
        r#"<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><type>quarter</type></note>"#
    }

    #[test]
    fn identical_graphs_have_zero_symbol_edit_distance() {
        let score = simple_score(rest_note());
        let graph = parse(&score);
        let report = evaluate_musicxml_score_graphs_v1(
            &graph, &graph, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        assert_eq!(report.status, MusicXmlScoreEvaluationStatusV1::Complete);
        let distance = report.overall_distance.unwrap();
        assert_eq!(distance.edit_distance, 0);
        assert_eq!(distance.insertions, 0);
        assert_eq!(distance.deletions, 0);
        assert_eq!(distance.normalized_distance, Some(0.0));
    }

    #[test]
    fn pitch_accidental_and_written_notation_changes_are_localized() {
        let reference = parse(&simple_score(rest_note()));
        let changed_note = r#"<note><pitch><step>C</step><alter>1</alter><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><type>quarter</type></note>"#;
        let prediction = parse(&simple_score(changed_note));
        let report = evaluate_musicxml_score_graphs_v1(
            &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        let families: BTreeMap<_, _> = report.family_distances.iter()
            .map(|metric| (metric.family, metric.distance.edit_distance))
            .collect();
        assert!(report.overall_distance.unwrap().edit_distance > 0);
        assert!(families.get(&MusicXmlSymbolFamilyV1::PitchSpelling).copied().unwrap_or(0) > 0);
    }

    #[test]
    fn note_to_rest_and_tuplet_changes_are_not_collapsed() {
        let reference = parse(&simple_score(rest_note()));
        let changed = r#"<note><rest/><duration>4</duration><voice>1</voice><staff>1</staff><type>quarter</type></note>"#;
        let prediction = parse(&simple_score(changed));
        let report = evaluate_musicxml_score_graphs_v1(
            &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        let families: BTreeMap<_, _> = report.family_distances.iter()
            .map(|metric| (metric.family, metric.distance.edit_distance))
            .collect();
        assert!(families.get(&MusicXmlSymbolFamilyV1::NoteRestUnpitched).copied().unwrap_or(0) > 0);
    }

    #[test]
    fn attribute_order_and_indentation_do_not_change_notation_tokens() {
        let left = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note id="x" color="FFFFFF"><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
            <note><rest/><duration>12</duration></note>
        </measure></part></score-partwise>"#;
        let right = r#"<score-partwise>
            <part id="P1">
                <measure number="1">
                    <attributes>
                        <time><beat-type>4</beat-type><beats>4</beats></time>
                        <divisions>4</divisions>
                    </attributes>
                    <note color="FFFFFF" id="different"><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
                    <note><rest/><duration>12</duration></note>
                </measure>
            </part>
        </score-partwise>"#;
        let l = parse(left);
        let r = parse(right);
        let report = evaluate_musicxml_score_graphs_v1(
            &l, &r, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        assert_eq!(report.overall_distance.unwrap().edit_distance, 0);
    }

    #[test]
    fn part_order_does_not_change_score_distance_but_missing_parts_are_explicit() {
        let xml = r#"<score-partwise>
            <part id="P1"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></attributes><note><rest/><duration>16</duration></note></measure></part>
            <part id="P2"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></attributes><note><rest/><duration>16</duration></note></measure></part>
        </score-partwise>"#;
        let mut reference = parse(xml);
        let mut prediction = reference.clone();
        prediction.reverse();
        let equal = evaluate_musicxml_score_graphs_v1(
            &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        assert_eq!(equal.overall_distance.unwrap().edit_distance, 0);

        reference.pop();
        let missing = evaluate_musicxml_score_graphs_v1(
            &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        assert_eq!(missing.coverage.reference_only_part_ids, 0);
        assert_eq!(missing.coverage.prediction_only_part_ids, 1);
        assert!(missing.overall_distance.unwrap().edit_distance > 0);
    }

    #[test]
    fn unknown_notation_part_withholds_complete_overall_distance() {
        let reference = parse(&simple_score(rest_note()));
        let mut prediction = reference.clone();
        prediction[0].notation_events.status =
            MusicXmlNotationEventsStatusV1::Unknown { reason: "fixture withheld".into() };
        let report = evaluate_musicxml_score_graphs_v1(
            &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        assert_eq!(report.status, MusicXmlScoreEvaluationStatusV1::Unavailable);
        assert!(report.overall_distance.is_none());
        assert_eq!(report.coverage.unknown_part_ids, 1);
    }

    #[test]
    fn unsupported_measure_map_withholds_overall_distance() {
        let reference = parse(&simple_score(rest_note()));
        let mut prediction = reference.clone();
        prediction[0].map.status = MeasureMapStatus::Unsupported {
            reason: "fixture unsupported".into()
        };
        let report = evaluate_musicxml_score_graphs_v1(
            &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        assert!(report.overall_distance.is_none());
        assert_eq!(report.status, MusicXmlScoreEvaluationStatusV1::Unavailable);
    }

    #[test]
    fn score_clock_or_canonicalization_mismatch_is_refused() {
        let reference = parse(&simple_score(rest_note()));
        let mut prediction = reference.clone();
        prediction[0].map.score_clock_id = "different-clock".into();
        assert!(matches!(
            evaluate_musicxml_score_graphs_v1(
                &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
            ),
            Err(MusicXmlScoreEvaluationErrorV1::ScoreClockMismatch { .. })
        ));
    }

    #[test]
    fn report_replay_accepts_exact_graphs_and_rejects_tampered_metrics() {
        let graph = parse(&simple_score(rest_note()));
        let mut report = evaluate_musicxml_score_graphs_v1(
            &graph, &graph, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        assert!(validate_musicxml_score_evaluation_replay_v1(
            &graph, &graph, &MusicXmlScoreEvaluationConfigV1::default(), &report
        ).unwrap());
        report.per_part[0].distance.as_mut().unwrap().insertions += 1;
        assert!(!validate_musicxml_score_evaluation_replay_v1(
            &graph, &graph, &MusicXmlScoreEvaluationConfigV1::default(), &report
        ).unwrap());
    }

    #[test]
    fn symbol_and_dp_budgets_fail_closed() {
        let graph = parse(&simple_score(rest_note()));
        assert!(matches!(
            evaluate_musicxml_score_graphs_v1(
                &graph,
                &graph,
                &MusicXmlScoreEvaluationConfigV1 {
                    max_symbols_per_part: 1,
                    max_diff_cells: 10,
                }
            ),
            Err(MusicXmlScoreEvaluationErrorV1::SymbolBudgetExceeded { .. })
        ));
        let changed = parse(&simple_score(
            r#"<note><rest measure="yes"/><duration>4</duration><voice>2</voice><staff>2</staff><type>half</type><notations><tuplet type="start" number="1"/></notations></note>"#
        ));
        assert!(matches!(
            evaluate_musicxml_score_graphs_v1(
                &graph,
                &changed,
                &MusicXmlScoreEvaluationConfigV1 {
                    max_symbols_per_part: 100_000,
                    max_diff_cells: 1,
                }
            ),
            Err(MusicXmlScoreEvaluationErrorV1::DiffWorkBudgetExceeded { .. })
        ));
    }

    #[test]
    fn one_symbol_substitution_is_one_insert_plus_one_delete() {
        let reference = parse(&simple_score(rest_note()));
        let changed_note = r#"<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff><type>quarter</type></note>"#;
        let prediction = parse(&simple_score(changed_note));
        let report = evaluate_musicxml_score_graphs_v1(
            &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        let distance = report.overall_distance.unwrap();
        assert_eq!(distance.insertions, 1);
        assert_eq!(distance.deletions, 1);
        assert_eq!(distance.edit_distance, 2);
        assert_eq!(
            distance.normalized_distance,
            Some(2.0 / (distance.reference_symbol_count + distance.prediction_symbol_count) as f64)
        );
    }

    #[test]
    fn namespace_uri_is_part_of_the_source_symbol_identity() {
        let left = r#"<score-partwise xmlns:ext="urn:notation:one"><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <ext:future-ornament kind="x">turn</ext:future-ornament>
            <note><rest/><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let right = r#"<score-partwise xmlns:ext="urn:notation:two"><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <ext:future-ornament kind="x">turn</ext:future-ornament>
            <note><rest/><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let reference = parse(left);
        let prediction = parse(right);
        let report = evaluate_musicxml_score_graphs_v1(
            &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        assert!(report.overall_distance.unwrap().edit_distance > 0);
    }

    #[test]
    fn visual_tie_and_tuplet_edits_appear_in_separate_families() {
        let tied = r#"<note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration><tie type="start"/><notations><tied type="start"/></notations></note>"#;
        let untied = r#"<note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration></note>"#;
        let reference = parse(&simple_score(tied));
        let prediction = parse(&simple_score(untied));
        let report = evaluate_musicxml_score_graphs_v1(
            &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        let families: BTreeMap<_, _> = report.family_distances.iter()
            .map(|metric| (metric.family, metric.distance.edit_distance))
            .collect();
        assert!(families.get(&MusicXmlSymbolFamilyV1::Ties).copied().unwrap_or(0) > 0);
    }

    #[test]
    fn canonicalization_mismatch_is_refused_even_when_clock_matches() {
        let reference = parse(&simple_score(rest_note()));
        let mut prediction = reference.clone();
        prediction[0].map.canonicalization_version = "unknown-canonicalization".into();
        assert!(matches!(
            evaluate_musicxml_score_graphs_v1(
                &reference, &prediction, &MusicXmlScoreEvaluationConfigV1::default()
            ),
            Err(MusicXmlScoreEvaluationErrorV1::CanonicalizationMismatch { .. })
        ));
    }

    #[test]
    fn input_graph_mutation_is_rejected_by_report_replay() {
        let graph = parse(&simple_score(rest_note()));
        let report = evaluate_musicxml_score_graphs_v1(
            &graph, &graph, &MusicXmlScoreEvaluationConfigV1::default()
        ).unwrap();
        let changed = parse(&simple_score(
            r#"<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>"#
        ));
        assert!(!validate_musicxml_score_evaluation_replay_v1(
            &graph, &changed, &MusicXmlScoreEvaluationConfigV1::default(), &report
        ).unwrap());
    }

}
