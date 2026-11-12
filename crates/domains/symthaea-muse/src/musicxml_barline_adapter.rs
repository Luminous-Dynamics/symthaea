// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative MusicXML adapter for the explicit barline evaluation contract.
//!
//! Only score-partwise measure containers are handled in V1. A measure span
//! is normalized from its explicit content cursor and, where present, its
//! own declared time signature. This is not a MIDI meter-to-barline inference
//! path. Multi-part measure sequences must reconcile exactly after rational
//! normalization; disagreement yields Unknown.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use crate::barline_eval::{
    MeasureKind, MeasureMap, MeasureMapStatus, MeasureSpan, RationalBeat,
};

pub const MUSICXML_BARLINE_CANONICALIZATION_V1: &str =
    "musicxml-partwise-explicit-measures-quarter-beats-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicXmlMeasureAdapterError {
    InvalidUtf8,
    MalformedXml(String),
    EmptySourceId,
    EmptyScoreClockId,
    InvalidDivisions { value: i64 },
    InvalidDuration { value: i64 },
    ArithmeticOverflow,
    MissingPartId,
    UnsupportedRoot { root: String },
    NoParts,
    UnsupportedPartMap { part_id: String },
    PartClockMismatch,
    PartCanonicalizationMismatch,
    SamePartCompared,
    DuplicatePartId { id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedMeasure {
    duration: RationalBeat,
    kind: MeasureKind,
    non_controlling: bool,
    has_unmodelled_barline_events: bool,
}

/// Parse explicit MusicXML partwise measures into a common score-clock map.
///
/// The caller must supply an immutable source identifier (preferably the
/// source artifact SHA-256) and a score-clock identifier shared with the
/// reference map. This adapter does not authenticate either identifier.
pub fn parse_musicxml_measure_map(
    bytes: &[u8],
    source_id: &str,
    score_clock_id: &str,
) -> Result<MeasureMap, MusicXmlMeasureAdapterError> {
    if source_id.trim().is_empty() {
        return Err(MusicXmlMeasureAdapterError::EmptySourceId);
    }
    if score_clock_id.trim().is_empty() {
        return Err(MusicXmlMeasureAdapterError::EmptyScoreClockId);
    }
    let text =
        std::str::from_utf8(bytes).map_err(|_| MusicXmlMeasureAdapterError::InvalidUtf8)?;
    let document = roxmltree::Document::parse(text)
        .map_err(|error| MusicXmlMeasureAdapterError::MalformedXml(error.to_string()))?;
    let root = document.root_element();

    if root.has_tag_name("score-timewise") {
        return Ok(unavailable_map(
            source_id,
            score_clock_id,
            MeasureMapStatus::Unsupported {
                reason: "MusicXML score-timewise is not supported by this adapter version".into(),
            },
        ));
    }
    if !root.has_tag_name("score-partwise") {
        return Ok(unavailable_map(
            source_id,
            score_clock_id,
            MeasureMapStatus::Unsupported {
                reason: "root element is not MusicXML score-partwise".into(),
            },
        ));
    }

    let parts: Vec<_> = root
        .children()
        .filter(|node| node.has_tag_name("part"))
        .collect();
    if parts.is_empty() {
        return Ok(unknown_map(
            source_id,
            score_clock_id,
            "score-partwise document contains no parts",
        ));
    }

    let mut normalized_parts = Vec::with_capacity(parts.len());
    for part in parts {
        match parse_part(part)? {
            Some(measures) if !measures.is_empty() => normalized_parts.push(measures),
            Some(_) => {
                return Ok(unknown_map(
                    source_id,
                    score_clock_id,
                    "a part contains no explicit measure containers",
                ));
            }
            None => {
                return Ok(unknown_map(
                    source_id,
                    score_clock_id,
                    "a part contains a measure whose score-beat extent is unresolved",
                ));
            }
        }
    }

    let reference = &normalized_parts[0];
    if normalized_parts.iter().flatten().any(|measure| measure.non_controlling) {
        return Ok(unknown_map(
            source_id,
            score_clock_id,
            "a part declares non-controlling measures; use per-part measure maps",
        ));
    }
    if normalized_parts
        .iter()
        .flatten()
        .any(|measure| measure.has_unmodelled_barline_events)
    {
        return Ok(unknown_map(
            source_id,
            score_clock_id,
            "barline events require the MusicXML notation-graph API",
        ));
    }
    for (part_index, measures) in normalized_parts.iter().enumerate().skip(1) {
        if measures.len() != reference.len() {
            return Ok(unknown_map(
                source_id,
                score_clock_id,
                &format!(
                    "part {part_index} has {} measures; first part has {}",
                    measures.len(),
                    reference.len()
                ),
            ));
        }
        for (measure_index, (left, right)) in
            reference.iter().zip(measures.iter()).enumerate()
        {
            if left.duration != right.duration {
                return Ok(unknown_map(
                    source_id,
                    score_clock_id,
                    &format!(
                        "part {part_index} measure {measure_index} has a different exact span"
                    ),
                ));
            }
        }
    }

    let mut start = RationalBeat::new(0, 1)
        .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
    let mut spans = Vec::with_capacity(reference.len());
    for (measure_index, first_part_measure) in reference.iter().enumerate() {
        let kind = if normalized_parts
            .iter()
            .all(|part| part[measure_index].kind == first_part_measure.kind)
        {
            first_part_measure.kind
        } else {
            MeasureKind::Unclassified
        };
        let end = start
            .checked_add(first_part_measure.duration)
            .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
        spans.push(MeasureSpan { start, end, kind });
        start = end;
    }

    Ok(MeasureMap {
        source_id: source_id.to_string(),
        score_clock_id: score_clock_id.to_string(),
        canonicalization_version: MUSICXML_BARLINE_CANONICALIZATION_V1.to_string(),
        status: MeasureMapStatus::Supported,
        measures: spans,
    })
}


/// One part's independent notated measure lattice.
///
/// `non_controlling_measure_indices` records the zero-based measure indices
/// whose left barline MusicXML explicitly declares non-coincident with other
/// parts. All per-part maps share the same score-clock identity; the separate
/// part ID prevents accidental mixing of note/measure ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MusicXmlPartMeasureMapV1 {
    pub part_id: String,
    pub non_controlling_measure_indices: Vec<usize>,
    pub map: MeasureMap,
}

/// Parse every MusicXML score-partwise part into its own exact measure map.
///
/// Unlike parse_musicxml_measure_map, this API preserves polymetric structure
/// instead of trying to collapse all part barlines into one global lattice.
/// A part with unresolved measure extents is returned with Unknown status;
/// valid sibling maps are retained for independent evaluation.
pub fn parse_musicxml_part_measure_maps(
    bytes: &[u8],
    source_id: &str,
    score_clock_id: &str,
) -> Result<Vec<MusicXmlPartMeasureMapV1>, MusicXmlMeasureAdapterError> {
    if source_id.trim().is_empty() {
        return Err(MusicXmlMeasureAdapterError::EmptySourceId);
    }
    if score_clock_id.trim().is_empty() {
        return Err(MusicXmlMeasureAdapterError::EmptyScoreClockId);
    }
    let text =
        std::str::from_utf8(bytes).map_err(|_| MusicXmlMeasureAdapterError::InvalidUtf8)?;
    let document = roxmltree::Document::parse(text)
        .map_err(|error| MusicXmlMeasureAdapterError::MalformedXml(error.to_string()))?;
    let root = document.root_element();
    if !root.has_tag_name("score-partwise") {
        return Err(MusicXmlMeasureAdapterError::UnsupportedRoot {
            root: root.tag_name().name().to_string(),
        });
    }

    let parts: Vec<_> = root
        .children()
        .filter(|node| node.has_tag_name("part"))
        .collect();
    if parts.is_empty() {
        return Err(MusicXmlMeasureAdapterError::NoParts);
    }

    let mut output = Vec::with_capacity(parts.len());
    let mut seen_part_ids = std::collections::BTreeSet::new();
    for part in parts {
        let part_id = part
            .attribute("id")
            .filter(|value| !value.trim().is_empty())
            .ok_or(MusicXmlMeasureAdapterError::MissingPartId)?;
        if !seen_part_ids.insert(part_id.to_string()) {
            return Err(MusicXmlMeasureAdapterError::DuplicatePartId {
                id: part_id.to_string(),
            });
        }
        let parsed = parse_part(part)?;
        let (status, measures, non_controlling_indices) = match parsed {
            Some(parsed)
                if !parsed.is_empty()
                    && parsed.iter().all(|measure| !measure.has_unmodelled_barline_events) =>
            {
                let mut start = RationalBeat::new(0, 1)
                    .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                let mut spans = Vec::with_capacity(parsed.len());
                let mut indices = Vec::new();
                for (index, measure) in parsed.iter().enumerate() {
                    let end = start
                        .checked_add(measure.duration)
                        .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                    spans.push(MeasureSpan {
                        start,
                        end,
                        kind: measure.kind,
                    });
                    if measure.non_controlling {
                        indices.push(index);
                    }
                    start = end;
                }
                (MeasureMapStatus::Supported, spans, indices)
            }
            Some(parsed) if parsed.is_empty() => (
                MeasureMapStatus::Unknown {
                    reason: "part contains no explicit measure containers".into(),
                },
                Vec::new(),
                Vec::new(),
            ),
            Some(_) => (
                MeasureMapStatus::Unknown {
                    reason: "barline events require the MusicXML notation-graph API".into(),
                },
                Vec::new(),
                Vec::new(),
            ),
            None => (
                MeasureMapStatus::Unknown {
                    reason: "part contains a measure whose score-beat extent is unresolved".into(),
                },
                Vec::new(),
                Vec::new(),
            ),
        };
        output.push(MusicXmlPartMeasureMapV1 {
            part_id: part_id.to_string(),
            non_controlling_measure_indices: non_controlling_indices,
            map: MeasureMap {
                source_id: format!("{source_id}#part={part_id}"),
                score_clock_id: score_clock_id.to_string(),
                canonicalization_version: MUSICXML_BARLINE_CANONICALIZATION_V1.into(),
                status,
                measures,
            },
        });
    }
    Ok(output)
}



/// Typed MusicXML barline location in a single partwise measure.
# [derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicXmlBarlineLocationV1 {
    Left,
    Middle,
    Right,
}

/// Explicit repeat semantics retained from a MusicXML barline.
# [derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlRepeatV1 {
    pub direction: Option<String>,
    pub times: Option<String>,
    pub winged: Option<String>,
}

/// Explicit numbered-ending semantics retained from a MusicXML barline.
# [derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlEndingV1 {
    pub number: Option<String>,
    pub ending_type: Option<String>,
    pub text: Option<String>,
}

/// One source barline event at an exact position in the notated score clock.
///
/// Repeats and endings remain graph annotations; this type does not expand
/// playback traversal or duplicate notated measures.
# [derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlPartBarlineEventV1 {
    pub source_order: usize,
    pub source_id: Option<String>,
    pub source_measure_number: String,
    pub part_id: String,
    pub measure_index: usize,
    pub location: MusicXmlBarlineLocationV1,
    pub measure_offset: RationalBeat,
    pub score_offset: RationalBeat,
    pub bar_style: Option<String>,
    pub divisions_override: Option<String>,
    pub repeat: Option<MusicXmlRepeatV1>,
    pub ending: Option<MusicXmlEndingV1>,
    pub segno: Option<String>,
    pub coda: Option<String>,
    pub fermatas: Vec<MusicXmlFermataV1>,
    pub wavy_lines: Vec<MusicXmlWavyLineV1>,
    pub annotations: Vec<MusicXmlBarlineAnnotationV1>,
    pub unrecognized_attributes: Vec<(String, String)>,
    pub staff: Option<String>,
}

/// Complete measure map plus every explicit MusicXML barline event for a part.
# [derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlFermataV1 {
    pub fermata_type: Option<String>,
    pub text: Option<String>,
    pub content: Option<String>,
}

/// Typed wavy-line marker attached to a barline (for example, a trill line).
# [derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlWavyLineV1 {
    pub line_type: Option<String>,
    pub number: Option<String>,
    pub placement: Option<String>,
}

/// MusicXML footnote/level annotation retained without interpreting its prose.
# [derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlBarlineAnnotationV1 {
    pub element: String,
    pub attributes: Vec<(String, String)>,
    pub text: Option<String>,
}

# [derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlPartNotationGraphV1 {
    pub part_id: String,
    pub map: MeasureMap,
    pub non_controlling_measure_indices: Vec<usize>,
    pub barline_events: Vec<MusicXmlPartBarlineEventV1>,
}

/// Parse score-partwise into exact per-part measure maps plus typed barline
/// events. Middle bars, repeats and endings are preserved here instead of
/// forcing the measure-only APIs to silently discard them.
pub fn parse_musicxml_part_notation_graph(
    bytes: &[u8],
    source_id: &str,
    score_clock_id: &str,
) -> Result<Vec<MusicXmlPartNotationGraphV1>, MusicXmlMeasureAdapterError> {
    if source_id.trim().is_empty() {
        return Err(MusicXmlMeasureAdapterError::EmptySourceId);
    }
    if score_clock_id.trim().is_empty() {
        return Err(MusicXmlMeasureAdapterError::EmptyScoreClockId);
    }
    let text =
        std::str::from_utf8(bytes).map_err(|_| MusicXmlMeasureAdapterError::InvalidUtf8)?;
    let document = roxmltree::Document::parse(text)
        .map_err(|error| MusicXmlMeasureAdapterError::MalformedXml(error.to_string()))?;
    let root = document.root_element();
    if !root.has_tag_name("score-partwise") {
        return Err(MusicXmlMeasureAdapterError::UnsupportedRoot {
            root: root.tag_name().name().to_string(),
        });
    }
    let parts: Vec<_> = root
        .children()
        .filter(|node| node.has_tag_name("part"))
        .collect();
    if parts.is_empty() {
        return Err(MusicXmlMeasureAdapterError::NoParts);
    }

    let mut seen_part_ids = std::collections::BTreeSet::new();
    let mut output = Vec::with_capacity(parts.len());
    for part in parts {
        let part_id = part
            .attribute("id")
            .filter(|value| !value.trim().is_empty())
            .ok_or(MusicXmlMeasureAdapterError::MissingPartId)?;
        if !seen_part_ids.insert(part_id.to_string()) {
            return Err(MusicXmlMeasureAdapterError::DuplicatePartId {
                id: part_id.to_string(),
            });
        }
        let parsed = parse_part(part)?;
        match parsed {
            Some(measures) if !measures.is_empty() => {
                let mut start = RationalBeat::new(0, 1)
                    .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                let mut spans = Vec::with_capacity(measures.len());
                for measure in &measures {
                    let end = start
                        .checked_add(measure.duration)
                        .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                    spans.push(MeasureSpan {
                        start,
                        end,
                        kind: measure.kind,
                    });
                    start = end;
                }
                let barline_events = parse_barline_events_for_part(
                    part,
                    part_id,
                    &measures,
                    &spans,
                )?;
                let map = MeasureMap {
                    source_id: format!("{source_id}#part={part_id}"),
                    score_clock_id: score_clock_id.to_string(),
                    canonicalization_version: MUSICXML_BARLINE_CANONICALIZATION_V1.into(),
                    status: MeasureMapStatus::Supported,
                    measures: spans,
                };
                let non_controlling_measure_indices = measures
                    .iter()
                    .enumerate()
                    .filter_map(|(index, measure)| measure.non_controlling.then_some(index))
                    .collect();
                output.push(MusicXmlPartNotationGraphV1 {
                    part_id: part_id.to_string(),
                    map,
                    non_controlling_measure_indices,
                    barline_events,
                });
            }
            Some(_) => output.push(MusicXmlPartNotationGraphV1 {
                part_id: part_id.to_string(),
                map: MeasureMap {
                    source_id: format!("{source_id}#part={part_id}"),
                    score_clock_id: score_clock_id.to_string(),
                    canonicalization_version: MUSICXML_BARLINE_CANONICALIZATION_V1.into(),
                    status: MeasureMapStatus::Unknown {
                        reason: "part contains no explicit measure containers".into(),
                    },
                    measures: Vec::new(),
                },
                non_controlling_measure_indices: Vec::new(),
                barline_events: Vec::new(),
            }),
            None => output.push(MusicXmlPartNotationGraphV1 {
                part_id: part_id.to_string(),
                map: MeasureMap {
                    source_id: format!("{source_id}#part={part_id}"),
                    score_clock_id: score_clock_id.to_string(),
                    canonicalization_version: MUSICXML_BARLINE_CANONICALIZATION_V1.into(),
                    status: MeasureMapStatus::Unknown {
                        reason: "part contains an unresolved measure or score-beat extent".into(),
                    },
                    measures: Vec::new(),
                },
                non_controlling_measure_indices: Vec::new(),
                barline_events: Vec::new(),
            }),
        }
    }
    Ok(output)
}


/// Canonical SHA-256 commitment to one parsed per-part notation graph.
///
/// The commitment includes the measure map, non-controlling flags and every
/// barline event. It is a content digest, not a signature or source-authenticity
/// proof.
pub fn musicxml_notation_graph_commitment(
    graph: &MusicXmlPartNotationGraphV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(graph)
}

/// Reparse the same source and compare the complete per-part notation graph.
/// Tampered maps, event offsets, repeat semantics or source locators return
/// false. The source itself remains identified by the caller-supplied ID.
pub fn validate_musicxml_notation_graph_replay(
    bytes: &[u8],
    source_id: &str,
    score_clock_id: &str,
    supplied: &[MusicXmlPartNotationGraphV1],
) -> Result<bool, MusicXmlMeasureAdapterError> {
    let replayed = parse_musicxml_part_notation_graph(bytes, source_id, score_clock_id)?;
    Ok(replayed == supplied)
}

fn parse_barline_events_for_part(
    part: roxmltree::Node<'_, '_>,
    part_id: &str,
    parsed_measures: &[ParsedMeasure],
    spans: &[MeasureSpan],
) -> Result<Vec<MusicXmlPartBarlineEventV1>, MusicXmlMeasureAdapterError> {
    let measure_nodes: Vec<_> = part
        .children()
        .filter(|node| node.has_tag_name("measure"))
        .collect();
    if measure_nodes.len() != parsed_measures.len() || spans.len() != parsed_measures.len() {
        return Err(MusicXmlMeasureAdapterError::ArithmeticOverflow);
    }

    let mut divisions: Option<i64> = None;
    let mut source_order = 0usize;
    let mut events = Vec::new();
    for (measure_index, measure) in measure_nodes.iter().enumerate() {
        let mut cursor = RationalBeat::new(0, 1)
            .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
        for element in measure.children().filter(|node| node.is_element()) {
            if element.has_tag_name("attributes") {
                for attribute in element.children().filter(|node| node.is_element()) {
                    if attribute.has_tag_name("divisions") {
                        let Some(value) = parse_i64(attribute.text()) else {
                            return Err(MusicXmlMeasureAdapterError::InvalidDivisions { value: 0 });
                        };
                        if value <= 0 {
                            return Err(MusicXmlMeasureAdapterError::InvalidDivisions { value });
                        }
                        divisions = Some(value);
                    }
                }
            } else if element.has_tag_name("note")
                || element.has_tag_name("forward")
                || element.has_tag_name("backup")
            {
                let is_grace = element.has_tag_name("note")
                    && element.children().any(|node| node.has_tag_name("grace"));
                let duration_node = element
                    .children()
                    .find(|node| node.has_tag_name("duration"));
                let Some(duration_node) = duration_node else {
                    if is_grace {
                        continue;
                    }
                    return Err(MusicXmlMeasureAdapterError::InvalidDuration { value: 0 });
                };
                let Some(duration_value) = parse_i64(duration_node.text()) else {
                    return Err(MusicXmlMeasureAdapterError::InvalidDuration { value: 0 });
                };
                if duration_value <= 0 {
                    return Err(MusicXmlMeasureAdapterError::InvalidDuration {
                        value: duration_value,
                    });
                }
                let Some(divisions_value) = divisions else {
                    return Err(MusicXmlMeasureAdapterError::InvalidDivisions { value: 0 });
                };
                let duration = RationalBeat::new(duration_value, divisions_value)
                    .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                if element.has_tag_name("backup") {
                    cursor = cursor
                        .checked_sub(duration)
                        .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                } else if element.has_tag_name("forward") {
                    cursor = cursor
                        .checked_add(duration)
                        .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                } else if !element.children().any(|node| node.has_tag_name("chord")) && !is_grace {
                    cursor = cursor
                        .checked_add(duration)
                        .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                }
            } else if element.has_tag_name("barline") {
                let location = match element.attribute("location").unwrap_or("right") {
                    "left" => MusicXmlBarlineLocationV1::Left,
                    "middle" => MusicXmlBarlineLocationV1::Middle,
                    "right" => MusicXmlBarlineLocationV1::Right,
                    _ => return Err(MusicXmlMeasureAdapterError::ArithmeticOverflow),
                };
                let measure_offset = match location {
                    MusicXmlBarlineLocationV1::Left => RationalBeat::new(0, 1)
                        .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?,
                    MusicXmlBarlineLocationV1::Middle => cursor,
                    MusicXmlBarlineLocationV1::Right => parsed_measures[measure_index].duration,
                };
                if location == MusicXmlBarlineLocationV1::Middle
                    && (measure_offset.compare_exact(RationalBeat::new(0, 1).unwrap())
                        != std::cmp::Ordering::Greater
                        || measure_offset.compare_exact(parsed_measures[measure_index].duration)
                            != std::cmp::Ordering::Less)
                {
                    return Err(MusicXmlMeasureAdapterError::ArithmeticOverflow);
                }
                let score_offset = spans[measure_index]
                    .start
                    .checked_add(measure_offset)
                    .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                let repeat_node = element.children().find(|node| node.has_tag_name("repeat"));
                let repeat = repeat_node.map(|node| MusicXmlRepeatV1 {
                    direction: node.attribute("direction").map(str::to_string),
                    times: node.attribute("times").map(str::to_string),
                    winged: node.attribute("winged").map(str::to_string),
                });
                let ending_node = element.children().find(|node| node.has_tag_name("ending"));
                let ending = ending_node.map(|node| MusicXmlEndingV1 {
                    number: node.attribute("number").map(str::to_string),
                    ending_type: node.attribute("type").map(str::to_string),
                    text: node.text().map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty()),
                });
                let child_text = |name: &str| {
                    element.children()
                        .find(|node| node.has_tag_name(name))
                        .map(|node| node.attribute("id").map(str::to_string)
                            .or_else(|| node.text().map(|value| value.trim().to_string()))
                            .filter(|value| !value.is_empty()))
                        .flatten()
                };
                let segno = element.attribute("segno").map(str::to_string).or_else(|| child_text("segno"));
                let coda = element.attribute("coda").map(str::to_string).or_else(|| child_text("coda"));
                let bar_style = element.children()
                    .find(|node| node.has_tag_name("bar-style"))
                    .and_then(|node| node.text())
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty());
                let fermatas = element.children()
                    .filter(|node| node.has_tag_name("fermata"))
                    .map(|node| MusicXmlFermataV1 {
                        fermata_type: node.attribute("type").map(str::to_string),
                        text: node.attribute("text").map(str::to_string),
                        content: node.text().map(|value| value.trim().to_string())
                            .filter(|value| !value.is_empty()),
                    })
                    .collect();
                let wavy_lines = element.children()
                    .filter(|node| node.has_tag_name("wavy-line"))
                    .map(|node| MusicXmlWavyLineV1 {
                        line_type: node.attribute("type").map(str::to_string),
                        number: node.attribute("number").map(str::to_string),
                        placement: node.attribute("placement").map(str::to_string),
                    })
                    .collect();
                let annotations = element.children()
                    .filter(|node| node.has_tag_name("footnote") || node.has_tag_name("level"))
                    .map(|node| MusicXmlBarlineAnnotationV1 {
                        element: node.tag_name().name().to_string(),
                        attributes: node.attributes()
                            .map(|attribute| (
                                attribute.name().to_string(),
                                attribute.value().to_string(),
                            ))
                            .collect(),
                        text: node.text().map(|value| value.trim().to_string())
                            .filter(|value| !value.is_empty()),
                    })
                    .collect();
                let known_attributes = ["id", "location", "segno", "coda", "divisions", "staff"];
                let unrecognized_attributes = element.attributes()
                    .filter(|attribute| !known_attributes.contains(&attribute.name()))
                    .map(|attribute| (
                        attribute.name().to_string(),
                        attribute.value().to_string(),
                    ))
                    .collect();
                let measure_number = measure.attribute("number").unwrap_or("");
                events.push(MusicXmlPartBarlineEventV1 {
                    source_order,
                    source_id: element.attribute("id").map(str::to_string),
                    source_measure_number: measure_number.to_string(),
                    part_id: part_id.to_string(),
                    measure_index,
                    location,
                    measure_offset,
                    score_offset,
                    bar_style,
                    divisions_override: element.attribute("divisions").map(str::to_string),
                    repeat,
                    ending,
                    segno,
                    coda,
                    fermatas,
                    wavy_lines,
                    annotations,
                    unrecognized_attributes,
                    staff: element.attribute("staff").map(str::to_string),
                });
                source_order = source_order.saturating_add(1);
            }
        }
    }
    Ok(events)
}

/// Exact cross-part boundary alignment without erasing non-controlling bars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MusicXmlPartBoundaryAlignmentV1 {
    pub left_part_id: String,
    pub right_part_id: String,
    pub coincident_boundaries: Vec<RationalBeat>,
    pub unmatched_required_left: Vec<RationalBeat>,
    pub unmatched_required_right: Vec<RationalBeat>,
    pub unmatched_non_controlling_left: Vec<RationalBeat>,
    pub unmatched_non_controlling_right: Vec<RationalBeat>,
}

/// Compare two per-part maps in their shared exact score-clock coordinate
/// system. Coincidence is exact; no tolerance or audio-clock conversion applies.
pub fn compare_musicxml_part_boundaries(
    left: &MusicXmlPartMeasureMapV1,
    right: &MusicXmlPartMeasureMapV1,
) -> Result<MusicXmlPartBoundaryAlignmentV1, MusicXmlMeasureAdapterError> {
    if left.part_id == right.part_id {
        return Err(MusicXmlMeasureAdapterError::SamePartCompared);
    }
    if left.map.score_clock_id != right.map.score_clock_id {
        return Err(MusicXmlMeasureAdapterError::PartClockMismatch);
    }
    if left.map.canonicalization_version != right.map.canonicalization_version {
        return Err(MusicXmlMeasureAdapterError::PartCanonicalizationMismatch);
    }
    if left.map.status != MeasureMapStatus::Supported {
        return Err(MusicXmlMeasureAdapterError::UnsupportedPartMap { part_id: left.part_id.clone() });
    }
    if right.map.status != MeasureMapStatus::Supported {
        return Err(MusicXmlMeasureAdapterError::UnsupportedPartMap { part_id: right.part_id.clone() });
    }

    let left_boundaries = part_internal_boundaries(left);
    let right_boundaries = part_internal_boundaries(right);
    let mut result = MusicXmlPartBoundaryAlignmentV1 {
        left_part_id: left.part_id.clone(),
        right_part_id: right.part_id.clone(),
        coincident_boundaries: Vec::new(),
        unmatched_required_left: Vec::new(),
        unmatched_required_right: Vec::new(),
        unmatched_non_controlling_left: Vec::new(),
        unmatched_non_controlling_right: Vec::new(),
    };
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left_boundaries.len() && right_index < right_boundaries.len() {
        let (left_beat, left_nc) = left_boundaries[left_index];
        let (right_beat, right_nc) = right_boundaries[right_index];
        match left_beat.compare_exact(right_beat) {
            std::cmp::Ordering::Equal => {
                result.coincident_boundaries.push(left_beat);
                left_index += 1;
                right_index += 1;
            }
            std::cmp::Ordering::Less => {
                push_unmatched_boundary(
                    &mut result.unmatched_required_left,
                    &mut result.unmatched_non_controlling_left,
                    left_beat,
                    left_nc,
                );
                left_index += 1;
            }
            std::cmp::Ordering::Greater => {
                push_unmatched_boundary(
                    &mut result.unmatched_required_right,
                    &mut result.unmatched_non_controlling_right,
                    right_beat,
                    right_nc,
                );
                right_index += 1;
            }
        }
    }
    while left_index < left_boundaries.len() {
        let (beat, nc) = left_boundaries[left_index];
        push_unmatched_boundary(
            &mut result.unmatched_required_left,
            &mut result.unmatched_non_controlling_left,
            beat,
            nc,
        );
        left_index += 1;
    }
    while right_index < right_boundaries.len() {
        let (beat, nc) = right_boundaries[right_index];
        push_unmatched_boundary(
            &mut result.unmatched_required_right,
            &mut result.unmatched_non_controlling_right,
            beat,
            nc,
        );
        right_index += 1;
    }
    Ok(result)
}

fn part_internal_boundaries(
    part: &MusicXmlPartMeasureMapV1,
) -> Vec<(RationalBeat, bool)> {
    part.map.measures.iter().enumerate().skip(1).map(|(index, measure)| {
        (measure.start, part.non_controlling_measure_indices.contains(&index))
    }).collect()
}

fn push_unmatched_boundary(
    required: &mut Vec<RationalBeat>,
    non_controlling: &mut Vec<RationalBeat>,
    beat: RationalBeat,
    is_non_controlling: bool,
) {
    if is_non_controlling {
        non_controlling.push(beat);
    } else {
        required.push(beat);
    }
}

fn unavailable_map(
    source_id: &str,
    score_clock_id: &str,
    status: MeasureMapStatus,
) -> MeasureMap {
    MeasureMap {
        source_id: source_id.to_string(),
        score_clock_id: score_clock_id.to_string(),
        canonicalization_version: MUSICXML_BARLINE_CANONICALIZATION_V1.to_string(),
        status,
        measures: Vec::new(),
    }
}

fn unknown_map(source_id: &str, score_clock_id: &str, reason: &str) -> MeasureMap {
    unavailable_map(
        source_id,
        score_clock_id,
        MeasureMapStatus::Unknown {
            reason: reason.to_string(),
        },
    )
}

fn parse_part(
    part: roxmltree::Node<'_, '_>,
) -> Result<Option<Vec<ParsedMeasure>>, MusicXmlMeasureAdapterError> {
    let mut divisions: Option<i64> = None;
    let mut expected_measure_length: Option<RationalBeat> = None;
    let measure_nodes: Vec<_> = part
        .children()
        .filter(|node| node.has_tag_name("measure"))
        .collect();
    if measure_nodes.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let mut measures = Vec::with_capacity(measure_nodes.len());
    for (measure_index, measure) in measure_nodes.iter().enumerate() {
        let implicit = measure
            .attribute("implicit")
            .is_some_and(|value| matches!(value, "yes" | "true" | "1"));
        let non_controlling = measure
            .attribute("non-controlling")
            .is_some_and(|value| matches!(value, "yes" | "true" | "1"));
        let mut cursor = RationalBeat::new(0, 1)
            .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
        let mut maximum_cursor = cursor;
        let mut has_unmodelled_barline_events = false;

        for element in measure.children().filter(|node| node.is_element()) {
            if element.has_tag_name("attributes") {
                for attribute in element.children().filter(|node| node.is_element()) {
                    if attribute.has_tag_name("divisions") {
                        let value = parse_i64(attribute.text());
                        let Some(value) = value else {
                            return Ok(None);
                        };
                        if value <= 0 {
                            return Err(MusicXmlMeasureAdapterError::InvalidDivisions {
                                value,
                            });
                        }
                        divisions = Some(value);
                    } else if attribute.has_tag_name("time") {
                        // V1 cannot collapse a mid-measure meter change into
                        // one measure-span label without a richer event graph.
                        if cursor.compare_exact(RationalBeat::new(0, 1).unwrap())
                            != std::cmp::Ordering::Equal
                        {
                            return Ok(None);
                        }
                        let Some(length) = parse_time_length(attribute) else {
                            return Ok(None);
                        };
                        expected_measure_length = Some(length);
                    }
                }
            } else if element.has_tag_name("note") {
                let grace = element.children().any(|node| node.has_tag_name("grace"));
                let duration_node = element
                    .children()
                    .find(|node| node.has_tag_name("duration"));
                let Some(duration_node) = duration_node else {
                    if grace {
                        continue;
                    }
                    return Ok(None);
                };
                let Some(duration_value) = parse_i64(duration_node.text()) else {
                    return Ok(None);
                };
                if duration_value <= 0 {
                    return Err(MusicXmlMeasureAdapterError::InvalidDuration {
                        value: duration_value,
                    });
                }
                let Some(divisions_value) = divisions else {
                    return Ok(None);
                };
                let duration = RationalBeat::new(duration_value, divisions_value)
                    .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                let chord_tone = element.children().any(|node| node.has_tag_name("chord"));
                if !chord_tone {
                    cursor = cursor
                        .checked_add(duration)
                        .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                    maximum_cursor = max_exact(maximum_cursor, cursor);
                }
            } else if element.has_tag_name("barline") {
                // MeasureMap V1 models outer measure spans, not a complete
                // notational-boundary graph. Never discard internal bars or
                // repeat/ending/playback semantics while claiming Supported.
                let location = element.attribute("location").unwrap_or("right");
                let has_unmodelled_semantics = element.children().any(|child| {
                    child.is_element()
                        && matches!(
                            child.tag_name().name(),
                            "wavy-line" | "segno" | "coda" | "fermata" | "ending" | "repeat"
                        )
                }) || element.attribute("segno").is_some()
                    || element.attribute("coda").is_some();
                if location == "middle"
                    || !matches!(location, "left" | "right")
                    || has_unmodelled_semantics
                {
                    has_unmodelled_barline_events = true;
                }
            } else if element.has_tag_name("forward") || element.has_tag_name("backup") {
                let duration_node = element
                    .children()
                    .find(|node| node.has_tag_name("duration"));
                let Some(duration_node) = duration_node else {
                    return Ok(None);
                };
                let Some(duration_value) = parse_i64(duration_node.text()) else {
                    return Ok(None);
                };
                if duration_value <= 0 {
                    return Err(MusicXmlMeasureAdapterError::InvalidDuration {
                        value: duration_value,
                    });
                }
                let Some(divisions_value) = divisions else {
                    return Ok(None);
                };
                let duration = RationalBeat::new(duration_value, divisions_value)
                    .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                if element.has_tag_name("forward") {
                    cursor = cursor
                        .checked_add(duration)
                        .map_err(|_| MusicXmlMeasureAdapterError::ArithmeticOverflow)?;
                    maximum_cursor = max_exact(maximum_cursor, cursor);
                } else {
                    cursor = match cursor.checked_sub(duration) {
                        Ok(value) => value,
                        Err(_) => return Ok(None),
                    };
                }
            }
        }

        if maximum_cursor == RationalBeat::new(0, 1).unwrap() {
            return Ok(None);
        }

        let (measure_duration, kind) = if implicit {
            (
                maximum_cursor,
                if measure_index == 0 {
                    MeasureKind::Pickup
                } else {
                    MeasureKind::Incomplete
                },
            )
        } else if let Some(expected) = expected_measure_length {
            match maximum_cursor.checked_sub(expected) {
                Ok(_) if maximum_cursor == expected => (expected, MeasureKind::Regular),
                Ok(_) => (maximum_cursor, MeasureKind::Irregular),
                Err(_) => {
                    // The explicit measure container plus an active time
                    // signature establishes the nominal bar length. An
                    // under-filled content cursor is not enough to decide
                    // whether trailing silence or a genuinely incomplete bar
                    // was omitted, so preserve the span but withhold class.
                    (expected, MeasureKind::Unclassified)
                }
            }
        } else {
            (maximum_cursor, MeasureKind::Unclassified)
        };

        measures.push(ParsedMeasure {
            duration: measure_duration,
            kind,
            non_controlling,
            has_unmodelled_barline_events,
        });
    }
    Ok(Some(measures))
}

fn parse_time_length(time: roxmltree::Node<'_, '_>) -> Option<RationalBeat> {
    let beats_nodes: Vec<_> = time
        .children()
        .filter(|node| node.has_tag_name("beats"))
        .collect();
    let denominator_nodes: Vec<_> = time
        .children()
        .filter(|node| node.has_tag_name("beat-type"))
        .collect();
    if beats_nodes.len() != 1 || denominator_nodes.len() != 1 {
        return None;
    }
    let denominator = parse_i64(denominator_nodes[0].text())?;
    if denominator <= 0 {
        return None;
    }
    let numerator = beats_nodes[0]
        .text()?
        .split('+')
        .map(|part| part.trim().parse::<i64>().ok())
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .try_fold(0_i64, |sum, value| sum.checked_add(value))?;
    if numerator <= 0 {
        return None;
    }
    let quarter_beats = numerator.checked_mul(4)?;
    RationalBeat::new(quarter_beats, denominator).ok()
}

fn parse_i64(value: Option<&str>) -> Option<i64> {
    value?.trim().parse().ok()
}

fn max_exact(left: RationalBeat, right: RationalBeat) -> RationalBeat {
    if left.compare_exact(right) != std::cmp::Ordering::Less {
        left
    } else {
        right
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(xml: &str) -> MeasureMap {
        parse_musicxml_measure_map(
            xml.as_bytes(),
            "sha256:fixture",
            "fixture-score-clock",
        )
        .unwrap()
    }

    fn supported(map: &MeasureMap) -> &[MeasureSpan] {
        assert_eq!(map.status, MeasureMapStatus::Supported);
        &map.measures
    }

    #[test]
    fn preserves_explicit_pickup_and_complete_musicxml_measures() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="0" implicit="yes">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                <note><rest/><duration>4</duration></note>
            </measure>
            <measure number="1"><note><rest/><duration>16</duration></note></measure>
            <measure number="2"><note><rest/><duration>16</duration></note></measure>
        </part></score-partwise>"#;
        let measures = supported(&parse(xml));
        assert_eq!(measures.len(), 3);
        assert_eq!(measures[0].start, RationalBeat::new(0, 1).unwrap());
        assert_eq!(measures[0].end, RationalBeat::new(1, 1).unwrap());
        assert_eq!(measures[0].kind, MeasureKind::Pickup);
        assert_eq!(measures[1].end, RationalBeat::new(5, 1).unwrap());
        assert_eq!(measures[1].kind, MeasureKind::Regular);
        assert_eq!(measures[2].end, RationalBeat::new(9, 1).unwrap());
    }

    #[test]
    fn normalizes_mixed_meter_lengths_in_quarter_note_beats() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>2</divisions><time><beats>7</beats><beat-type>8</beat-type></time></attributes>
                <note><rest/><duration>7</duration></note>
            </measure>
            <measure number="2">
                <attributes><time><beats>5</beats><beat-type>8</beat-type></time></attributes>
                <note><rest/><duration>5</duration></note>
            </measure>
            <measure number="3">
                <attributes><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                <note><rest/><duration>8</duration></note>
            </measure>
        </part></score-partwise>"#;
        let measures = supported(&parse(xml));
        assert_eq!(measures[0].end, RationalBeat::new(7, 2).unwrap());
        assert_eq!(measures[1].end, RationalBeat::new(6, 1).unwrap());
        assert_eq!(measures[2].end, RationalBeat::new(10, 1).unwrap());
    }

    #[test]
    fn backup_and_forward_do_not_double_count_simultaneous_voices() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                <note><rest/><duration>4</duration></note><note><rest/><duration>4</duration></note>
                <note><rest/><duration>4</duration></note><note><rest/><duration>4</duration></note>
                <backup><duration>16</duration></backup>
                <note><rest/><duration>4</duration></note><note><rest/><duration>4</duration></note>
                <note><rest/><duration>4</duration></note><note><rest/><duration>4</duration></note>
            </measure>
        </part></score-partwise>"#;
        let measures = supported(&parse(xml));
        assert_eq!(measures.len(), 1);
        assert_eq!(measures[0].end, RationalBeat::new(4, 1).unwrap());
        assert_eq!(measures[0].kind, MeasureKind::Regular);
    }

    #[test]
    fn divisions_changes_preserve_exact_quarter_note_duration() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                <note><rest/><duration>4</duration></note>
                <attributes><divisions>4</divisions></attributes>
                <note><rest/><duration>8</duration></note>
            </measure>
        </part></score-partwise>"#;
        let measures = supported(&parse(xml));
        assert_eq!(measures[0].end, RationalBeat::new(4, 1).unwrap());
        assert_eq!(measures[0].kind, MeasureKind::Regular);
    }

    #[test]
    fn rest_only_measure_is_retained() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></attributes>
                <note><rest measure="yes"/><duration>16</duration></note>
            </measure>
        </part></score-partwise>"#;
        let measures = supported(&parse(xml));
        assert_eq!(measures.len(), 1);
        assert_eq!(measures[0].end, RationalBeat::new(4, 1).unwrap());
    }

    #[test]
    fn unsupported_timewise_document_returns_unsupported() {
        let map = parse(r#"<score-timewise><measure number="1"/></score-timewise>"#);
        assert!(matches!(map.status, MeasureMapStatus::Unsupported { .. }));
    }

    #[test]
    fn empty_measure_without_explicit_duration_remains_unknown() {
        let map = parse(r#"<score-partwise><part id="P1"><measure number="1"/></part></score-partwise>"#);
        assert!(matches!(map.status, MeasureMapStatus::Unknown { .. }));
        assert!(map.measures.is_empty());
    }

    #[test]
    fn cross_part_measure_mismatch_remains_unknown() {
        let xml = r#"<score-partwise>
            <part id="P1"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>16</duration></note></measure></part>
            <part id="P2"><measure number="1"><attributes><divisions>4</divisions><time><beats>3</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>12</duration></note></measure></part>
        </score-partwise>"#;
        let map = parse(xml);
        assert!(matches!(map.status, MeasureMapStatus::Unknown { .. }));
    }

    #[test]
    fn unsupported_or_unresolved_measures_do_not_emit_partial_maps() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="1"><attributes><divisions>4</divisions></attributes>
            <note><rest/><duration>16</duration></note></measure>
            <measure number="2"><note><rest/></note></measure>
        </part></score-partwise>"#;
        let map = parse(xml);
        assert!(matches!(map.status, MeasureMapStatus::Unknown { .. }));
        assert!(map.measures.is_empty());
    }

    #[test]
    fn polymetric_musicxml_preserves_independent_part_measure_lattices() {
        let xml = r#"<score-partwise>
            <part id="P1">
                <measure number="1"><attributes><divisions>4</divisions><time><beats>3</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>12</duration></note></measure>
                <measure number="2"><note><rest/><duration>12</duration></note></measure>
            </part>
            <part id="P2">
                <measure number="1"><attributes><divisions>4</divisions><time><beats>2</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>8</duration></note></measure>
                <measure number="2" non-controlling="yes"><note><rest/><duration>8</duration></note></measure>
                <measure number="3" non-controlling="yes"><note><rest/><duration>8</duration></note></measure>
            </part>
        </score-partwise>"#;

        let maps = parse_musicxml_part_measure_maps(
            xml.as_bytes(),
            "sha256:polymeter",
            "score-clock-1",
        )
        .unwrap();
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[0].part_id, "P1");
        assert_eq!(maps[1].part_id, "P2");
        assert_eq!(maps[0].map.measures.len(), 2);
        assert_eq!(maps[1].map.measures.len(), 3);
        assert_eq!(maps[0].map.measures[1].start, RationalBeat::new(3, 1).unwrap());
        assert_eq!(maps[1].map.measures[1].start, RationalBeat::new(2, 1).unwrap());
        assert_eq!(maps[1].non_controlling_measure_indices, vec![1, 2]);
        assert_eq!(
            maps[0].map.score_clock_id,
            maps[1].map.score_clock_id,
            "all parts share one exact notated score-clock coordinate frame"
        );
    }

    #[test]
    fn global_map_withholds_non_controlling_musicxml_bar_lines() {
        let xml = r#"<score-partwise>
            <part id="P1">
                <measure number="1"><attributes><divisions>4</divisions><time><beats>3</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>12</duration></note></measure>
                <measure number="2"><note><rest/><duration>12</duration></note></measure>
            </part>
            <part id="P2">
                <measure number="1"><attributes><divisions>4</divisions><time><beats>2</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>8</duration></note></measure>
                <measure number="2" non-controlling="yes"><note><rest/><duration>8</duration></note></measure>
                <measure number="3" non-controlling="yes"><note><rest/><duration>8</duration></note></measure>
            </part>
        </score-partwise>"#;
        let global = parse(xml);
        assert!(matches!(global.status, MeasureMapStatus::Unknown { .. }));
        assert!(global.measures.is_empty());
    }

    #[test]
    fn ordinary_aligned_parts_remain_globally_reconcilable() {
        let xml = r#"<score-partwise>
            <part id="P1">
                <measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>16</duration></note></measure>
            </part>
            <part id="P2">
                <measure number="1"><attributes><divisions>8</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>32</duration></note></measure>
            </part>
        </score-partwise>"#;
        let global = parse(xml);
        assert_eq!(global.status, MeasureMapStatus::Supported);
        assert_eq!(global.measures[0].end, RationalBeat::new(4, 1).unwrap());
    }

    #[test]
    fn pairwise_alignment_separates_non_controlling_polymetric_bars() {
        let xml = r#"<score-partwise>
            <part id="P1">
                <measure number="1"><attributes><divisions>4</divisions><time><beats>3</beats><beat-type>4</beat-type></attributes><note><rest/><duration>12</duration></note></measure>
                <measure number="2" non-controlling="yes"><note><rest/><duration>12</duration></note></measure>
            </part>
            <part id="P2">
                <measure number="1"><attributes><divisions>4</divisions><time><beats>2</beats><beat-type>4</beat-type></attributes><note><rest/><duration>8</duration></note></measure>
                <measure number="2" non-controlling="yes"><note><rest/><duration>8</duration></note></measure>
                <measure number="3" non-controlling="yes"><note><rest/><duration>8</duration></note></measure>
            </part>
        </score-partwise>"#;
        let maps = parse_musicxml_part_measure_maps(xml.as_bytes(), "sha256:poly", "score-clock").unwrap();
        let alignment = compare_musicxml_part_boundaries(&maps[0], &maps[1]).unwrap();
        assert!(alignment.coincident_boundaries.is_empty());
        assert!(alignment.unmatched_required_left.is_empty());
        assert!(alignment.unmatched_required_right.is_empty());
        assert_eq!(alignment.unmatched_non_controlling_left, vec![RationalBeat::new(3, 1).unwrap()]);
        assert_eq!(alignment.unmatched_non_controlling_right, vec![
            RationalBeat::new(2, 1).unwrap(), RationalBeat::new(4, 1).unwrap()
        ]);
    }

    #[test]
    fn aligned_part_boundaries_are_exact_coincidences() {
        let xml = r#"<score-partwise>
            <part id="P1">
                <measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>16</duration></note></measure>
                <measure number="2"><note><rest/><duration>16</duration></note></measure>
            </part>
            <part id="P2">
                <measure number="1"><attributes><divisions>8</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>32</duration></note></measure>
                <measure number="2"><note><rest/><duration>32</duration></note></measure>
            </part>
        </score-partwise>"#;
        let maps = parse_musicxml_part_measure_maps(xml.as_bytes(), "sha256:aligned", "score-clock").unwrap();
        let alignment = compare_musicxml_part_boundaries(&maps[0], &maps[1]).unwrap();
        assert_eq!(alignment.coincident_boundaries, vec![RationalBeat::new(4, 1).unwrap()]);
        assert!(alignment.unmatched_required_left.is_empty());
        assert!(alignment.unmatched_required_right.is_empty());
    }

    #[test]
    fn incompatible_part_score_clock_is_rejected() {
        let xml = r#"<score-partwise>
            <part id="P1"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></attributes><note><rest/><duration>16</duration></note></measure></part>
            <part id="P2"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></attributes><note><rest/><duration>16</duration></note></measure></part>
        </score-partwise>"#;
        let mut maps = parse_musicxml_part_measure_maps(xml.as_bytes(), "sha256:clock", "score-clock").unwrap();
        maps[1].map.score_clock_id = "different-clock".into();
        assert_eq!(compare_musicxml_part_boundaries(&maps[0], &maps[1]), Err(MusicXmlMeasureAdapterError::PartClockMismatch));
    }

    #[test]
    fn duplicate_musicxml_part_ids_are_rejected() {
        let xml = r#"<score-partwise>
            <part id="P1"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>16</duration></note></measure></part>
            <part id="P1"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><rest/><duration>16</duration></note></measure></part>
        </score-partwise>"#;
        assert_eq!(
            parse_musicxml_part_measure_maps(xml.as_bytes(), "sha256:duplicate-id", "score-clock"),
            Err(MusicXmlMeasureAdapterError::DuplicatePartId { id: "P1".into() })
        );
    }

    #[test]
    fn mid_measure_time_signature_change_is_not_flattened() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                <note><rest/><duration>8</duration></note>
                <attributes><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
                <note><rest/><duration>8</duration></note>
            </measure>
        </part></score-partwise>"#;
        let global = parse(xml);
        assert!(matches!(global.status, MeasureMapStatus::Unknown { .. }));
        assert!(global.measures.is_empty());
    }

    #[test]
    fn malformed_explicit_time_signature_is_not_replaced_by_previous_meter() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                <note><rest/><duration>16</duration></note>
            </measure>
            <measure number="2">
                <attributes><time><beats>0</beats><beat-type>8</beat-type></time></attributes>
                <note><rest/><duration>16</duration></note>
            </measure>
        </part></score-partwise>"#;
        let global = parse(xml);
        assert!(matches!(global.status, MeasureMapStatus::Unknown { .. }));
        assert!(global.measures.is_empty());
    }

    #[test]
    fn middle_barline_is_not_silently_collapsed_into_measure_span() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                <note><rest/><duration>8</duration></note>
                <barline location="middle"><bar-style>dashed</bar-style></barline>
                <note><rest/><duration>8</duration></note>
            </measure>
        </part></score-partwise>"#;
        let global = parse(xml);
        assert!(matches!(global.status, MeasureMapStatus::Unknown { .. }));
        assert!(global.measures.is_empty());
        let parts = parse_musicxml_part_measure_maps(
            xml.as_bytes(),
            "sha256:internal-barline",
            "score-clock-1",
        ).unwrap();
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0].map.status, MeasureMapStatus::Unknown { .. }));
        assert!(parts[0].map.measures.is_empty());
    }

    #[test]
    fn repeat_ending_and_playback_signposts_are_not_silently_dropped() {
        let semantic_barline_fragments = [
            r#"<barline location="left"><bar-style>heavy-light</bar-style><repeat direction="forward"/></barline>"#,
            r#"<barline location="right"><bar-style>light-heavy</bar-style><repeat direction="backward" times="2"/></barline>"#,
            r#"<barline location="right"><ending number="1" type="start">1.</ending></barline>"#,
            r#"<barline location="right"><ending number="1" type="stop"/></barline>"#,
            r#"<barline location="left"><segno/></barline>"#,
            r#"<barline location="right"><coda/></barline>"#,
            r#"<barline location="right"><fermata type="upright">normal</fermata></barline>"#,
            r#"<barline location="right" segno="S1"><bar-style>light-heavy</bar-style></barline>"#,
            r#"<barline location="left" coda="C1"><bar-style>heavy-light</bar-style></barline>"#,
        ];
        for (index, barline) in semantic_barline_fragments.iter().enumerate() {
            let xml = format!(
                r#"<score-partwise><part id="P1"><measure number="1">
                    <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></attributes>
                    <note><rest/><duration>16</duration></note>
                    {barline}
                </measure></part></score-partwise>"#
            );
            let map = parse(&xml);
            assert!(
                matches!(map.status, MeasureMapStatus::Unknown { .. }),
                "barline semantic fixture {index} must not be flattened"
            );
            assert!(map.measures.is_empty());
        }
    }

    #[test]
    fn notation_graph_preserves_exact_middle_barline_repeat_and_ending_events() {
        let xml = r#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></attributes>
                <note><rest/><duration>8</duration></note>
                <barline location="middle" id="mid-1"><bar-style>dashed</bar-style></barline>
                <note><rest/><duration>8</duration></note>
                <barline location="right" id="end-1" divisions="4" custom="x"><bar-style>light-heavy</bar-style><footnote>closing note</footnote><level>1</level><wavy-line type="start" number="2" placement="above"/><segno id="S1"/><coda id="C1"/><fermata type="upright">normal</fermata><ending number="1" type="start">1.</ending><repeat direction="backward" times="2"/></barline>
            </measure>
            <measure number="2" implicit="yes">
                <note><rest/><duration>4</duration></note>
                <barline location="right"><repeat direction="forward"/></barline>
            </measure>
        </part></score-partwise>"#;
        let graphs = parse_musicxml_part_notation_graph(
            xml.as_bytes(),
            "sha256:notation-graph",
            "score-clock",
        ).unwrap();
        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].map.status, MeasureMapStatus::Supported);
        assert_eq!(graphs[0].map.measures[0].end, RationalBeat::new(4, 1).unwrap());
        assert_eq!(graphs[0].barline_events.len(), 3);
        let middle = &graphs[0].barline_events[0];
        assert_eq!(middle.source_id.as_deref(), Some("mid-1"));
        assert_eq!(middle.location, MusicXmlBarlineLocationV1::Middle);
        assert_eq!(middle.measure_offset, RationalBeat::new(2, 1).unwrap());
        assert_eq!(middle.score_offset, RationalBeat::new(2, 1).unwrap());
        assert_eq!(middle.bar_style.as_deref(), Some("dashed"));
        assert_eq!(graphs[0].non_controlling_measure_indices, Vec::<usize>::new());

        let right = &graphs[0].barline_events[1];
        assert_eq!(right.location, MusicXmlBarlineLocationV1::Right);
        assert_eq!(right.score_offset, RationalBeat::new(4, 1).unwrap());
        assert_eq!(right.ending.as_ref().and_then(|value| value.number.as_deref()), Some("1"));
        assert_eq!(right.repeat.as_ref().and_then(|value| value.direction.as_deref()), Some("backward"));
        assert_eq!(right.repeat.as_ref().and_then(|value| value.times.as_deref()), Some("2"));
        assert_eq!(right.divisions_override.as_deref(), Some("4"));
        assert_eq!(right.segno.as_deref(), Some("S1"));
        assert_eq!(right.coda.as_deref(), Some("C1"));
        assert_eq!(right.fermatas.len(), 1);
        assert_eq!(right.fermatas[0].fermata_type.as_deref(), Some("upright"));
        assert_eq!(right.fermatas[0].content.as_deref(), Some("normal"));
        assert_eq!(right.wavy_lines.len(), 1);
        assert_eq!(right.wavy_lines[0].line_type.as_deref(), Some("start"));
        assert_eq!(right.annotations.len(), 2);
        assert_eq!(right.annotations[0].element, "footnote");
        assert_eq!(right.annotations[0].text.as_deref(), Some("closing note"));
        assert!(right.unrecognized_attributes.contains(&("custom".into(), "x".into())));
        let second_right = &graphs[0].barline_events[2];
        assert_eq!(second_right.score_offset, RationalBeat::new(5, 1).unwrap());
        assert_eq!(second_right.repeat.as_ref().and_then(|value| value.direction.as_deref()), Some("forward"));
    }

    #[test]
    fn multiple_barline_events_at_one_score_offset_are_not_deduplicated() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><rest/><duration>16</duration></note>
            <barline location="right" id="b1"><bar-style>light-light</bar-style></barline>
            <barline location="right" id="b2"><repeat direction="backward"/></barline>
        </measure></part></score-partwise>"#;
        let graph = parse_musicxml_part_notation_graph(
            xml.as_bytes(), "sha256:duplicate-boundaries", "score-clock"
        ).unwrap();
        assert_eq!(graph[0].barline_events.len(), 2);
        assert_eq!(
            graph[0].barline_events[0].score_offset,
            graph[0].barline_events[1].score_offset
        );
        assert_ne!(
            graph[0].barline_events[0].source_id,
            graph[0].barline_events[1].source_id
        );
    }

    #[test]
    fn simple_measure_map_stays_unknown_when_graph_only_events_are_present() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><rest/><duration>8</duration></note>
            <barline location="middle"><bar-style>dashed</bar-style></barline>
            <note><rest/><duration>8</duration></note>
        </measure></part></score-partwise>"#;
        let map = parse(xml);
        assert!(matches!(map.status, MeasureMapStatus::Unknown { .. }));
        let graph = parse_musicxml_part_notation_graph(
            xml.as_bytes(), "sha256:middle-bar", "score-clock"
        ).unwrap();
        assert_eq!(graph[0].map.status, MeasureMapStatus::Supported);
        assert_eq!(graph[0].barline_events.len(), 1);
    }

    #[test]
    fn notation_graph_commitment_and_replay_reject_tampering() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><rest/><duration>8</duration></note>
            <barline location="middle" id="mid"><bar-style>dashed</bar-style></barline>
            <note><rest/><duration>8</duration></note>
            <barline location="right"><repeat direction="backward" times="2"/></barline>
        </measure></part></score-partwise>"#;
        let graphs = parse_musicxml_part_notation_graph(
            xml.as_bytes(), "sha256:sealed-graph", "score-clock"
        ).unwrap();
        let digest = musicxml_notation_graph_commitment(&graphs[0]).unwrap();
        assert_eq!(digest.len(), 64);
        assert!(validate_musicxml_notation_graph_replay(
            xml.as_bytes(), "sha256:sealed-graph", "score-clock", &graphs
        ).unwrap());

        let mut tampered = graphs.clone();
        tampered[0].barline_events[0].score_offset = RationalBeat::new(3, 1).unwrap();
        assert!(!validate_musicxml_notation_graph_replay(
            xml.as_bytes(), "sha256:sealed-graph", "score-clock", &tampered
        ).unwrap());

        let mut tampered_repeat = graphs.clone();
        tampered_repeat[0].barline_events[1].repeat.as_mut().unwrap().times = Some("3".into());
        assert!(!validate_musicxml_notation_graph_replay(
            xml.as_bytes(), "sha256:sealed-graph", "score-clock", &tampered_repeat
        ).unwrap());
    }

}
