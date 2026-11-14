// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-ordered MusicXML notation events.
//!
//! The legacy Score intentionally remains note-centric. This layer preserves
//! written and performed-symbolic semantics beside the measure/barline graph:
//! rests, unpitched notes, grace/cue/chord relationships, exact durations,
//! written values, tuplets, sound ties, visual ties, voice/staff identity and
//! score-order attribute changes. Unknown XML children are retained as source
//! trees rather than silently discarded.

use crate::barline_eval::{MeasureSpan, RationalBeat};
use serde::{Deserialize, Serialize};

pub const MUSICXML_NOTATION_EVENTS_SCHEMA_V1: &str = "muse-musicxml-notation-events-v1";
pub const MUSICXML_NOTATION_EVENTS_CANONICALIZATION_V1: &str =
    "musicxml-partwise-source-ordered-exact-quarter-beats-v1";

const MAX_NOTATION_EVENTS_PER_PART: usize = 200_000;
const MAX_SOURCE_TREE_DEPTH: usize = 32;
const MAX_SOURCE_TREE_NODES: usize = 500_000;

/// A recursively retained XML element. Attributes and child order follow the
/// source document. The convenience text field is trimmed; ordered mixed
/// content retains exact text-node values and element positions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlSourceNodeV1 {
    pub element: String,
    /// Namespace URI of the element; local names alone are not globally unique.
    pub namespace_uri: Option<String>,
    pub attributes: Vec<MusicXmlSourceAttributeV1>,
    pub text: Option<String>,
    pub children: Vec<MusicXmlSourceNodeV1>,
    /// Ordered mixed content (text and child elements) so extensions such as
    /// lyrics, words and unknown annotation markup do not lose text position.
    pub content: Vec<MusicXmlSourceContentV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlSourceAttributeV1 {
    pub name: String,
    pub namespace_uri: Option<String>,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicXmlSourceContentV1 {
    Text(String),
    /// Index into the owning node's `children` vector; avoids duplicating
    /// nested source trees in the bounded representation.
    ElementIndex(usize),
}

/// Part-list identity and instrument declarations that sit outside the
/// score-partwise <part> body. These identifiers link written parts to
/// score-instruments and MIDI program/channel metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlPartMetadataV1 {
    pub source_part_id: String,
    pub part_name: Option<String>,
    pub part_abbreviation: Option<String>,
    pub score_instrument_ids: Vec<String>,
    /// Full score-part subtree retains names, MIDI instruments, channels,
    /// banks, programs, device declarations and future extensions.
    pub source_score_part: MusicXmlSourceNodeV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlWrittenPitchV1 {
    pub step: String,
    /// Exact source spelling from MusicXML; may be fractional for microtones.
    pub alter: Option<String>,
    pub octave: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlDisplayPitchV1 {
    pub step: Option<String>,
    pub octave: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicXmlNotationEventKindV1 {
    Pitched(MusicXmlWrittenPitchV1),
    Rest {
        measure: bool,
        display: MusicXmlDisplayPitchV1,
    },
    Unpitched {
        display: MusicXmlDisplayPitchV1,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlGraceV1 {
    pub slash: Option<String>,
    pub steal_time_previous: Option<String>,
    pub steal_time_following: Option<String>,
    pub make_time: Option<String>,
    pub source: MusicXmlSourceNodeV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlTimeModificationV1 {
    pub actual_notes: u32,
    pub normal_notes: u32,
    pub normal_type: Option<String>,
    pub normal_dot_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlTieV1 {
    pub tie_type: Option<String>,
    pub time_only: Option<String>,
    pub attributes: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlBeamV1 {
    pub number: Option<String>,
    pub value: Option<String>,
    pub attributes: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlNotationEventV1 {
    pub source_order: usize,
    pub source_measure_index: usize,
    pub source_measure_number: Option<String>,
    pub part_id: String,
    pub score_offset: RationalBeat,
    /// Original MusicXML note-level performance attributes. They remain
    /// source strings in divisions, separate from the notated score cursor.
    pub time_only: Option<String>,
    pub attack_divisions: Option<String>,
    pub release_divisions: Option<String>,
    /// None for grace notes without a performed duration.
    pub performed_duration: Option<RationalBeat>,
    pub kind: MusicXmlNotationEventKindV1,
    pub is_grace: bool,
    pub grace: Option<MusicXmlGraceV1>,
    pub is_cue: bool,
    pub is_chord_member: bool,
    pub chord_anchor_source_order: Option<usize>,
    pub voice: Option<String>,
    pub staff: Option<String>,
    pub instrument_ids: Vec<String>,
    pub written_type: Option<String>,
    pub dot_count: u8,
    pub time_modification: Option<MusicXmlTimeModificationV1>,
    /// MusicXML sound ties are not the same thing as visual tied notation.
    pub sound_ties: Vec<MusicXmlTieV1>,
    pub visual_ties: Vec<MusicXmlTieV1>,
    /// Full source tuplet nodes preserve nested actual/normal display details.
    pub tuplet_notations: Vec<MusicXmlSourceNodeV1>,
    pub stem: Option<String>,
    pub notehead: Option<MusicXmlSourceNodeV1>,
    pub accidental: Option<MusicXmlSourceNodeV1>,
    pub beams: Vec<MusicXmlBeamV1>,
    /// Entire original note subtree, including articulations, ornaments,
    /// technical marks, lyrics and attributes not yet interpreted by V1.
    pub source_note: MusicXmlSourceNodeV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlAttributeEventV1 {
    pub source_order: usize,
    pub measure_index: usize,
    pub measure_number: Option<String>,
    pub part_id: String,
    pub score_offset: RationalBeat,
    /// Raw child of an <attributes> element: divisions/key/time/clef/staves/
    /// transpose/staff-details/etc. Source order is retained.
    pub source: MusicXmlSourceNodeV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlAuxiliaryEventV1 {
    pub source_order: usize,
    pub measure_index: usize,
    pub measure_number: Option<String>,
    pub part_id: String,
    pub score_offset: RationalBeat,
    /// Raw direction/harmony/figured-bass/barline/sound/print event, etc.
    pub source: MusicXmlSourceNodeV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlCursorMoveV1 {
    pub source_order: usize,
    pub measure_index: usize,
    pub measure_number: Option<String>,
    pub part_id: String,
    pub direction: String,
    pub voice: Option<String>,
    pub staff: Option<String>,
    pub score_offset_before: RationalBeat,
    pub duration: RationalBeat,
    pub score_offset_after: RationalBeat,
    pub source: MusicXmlSourceNodeV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicXmlNotationEventsStatusV1 {
    Supported,
    Unknown { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicXmlPartNotationEventsV1 {
    pub schema: String,
    pub canonicalization_version: String,
    pub part_id: String,
    pub status: MusicXmlNotationEventsStatusV1,
    pub note_events: Vec<MusicXmlNotationEventV1>,
    pub attribute_events: Vec<MusicXmlAttributeEventV1>,
    pub auxiliary_events: Vec<MusicXmlAuxiliaryEventV1>,
    pub cursor_moves: Vec<MusicXmlCursorMoveV1>,
}

/// Default used only when deserializing pre-MIR-018 notation graphs. It is
/// omitted again during serialization so older graph commitments remain
/// byte-for-byte stable under the prior canonical JSON representation.
pub fn legacy_missing_notation_events() -> MusicXmlPartNotationEventsV1 {
    MusicXmlPartNotationEventsV1 {
        schema: MUSICXML_NOTATION_EVENTS_SCHEMA_V1.into(),
        canonicalization_version: MUSICXML_NOTATION_EVENTS_CANONICALIZATION_V1.into(),
        part_id: String::new(),
        status: MusicXmlNotationEventsStatusV1::Unknown {
            reason: "legacy graph predates the notation-event layer".into(),
        },
        note_events: Vec::new(),
        attribute_events: Vec::new(),
        auxiliary_events: Vec::new(),
        cursor_moves: Vec::new(),
    }
}

pub fn is_legacy_missing_notation_events(
    value: &MusicXmlPartNotationEventsV1,
) -> bool {
    value.part_id.is_empty()
        && value.note_events.is_empty()
        && value.attribute_events.is_empty()
        && value.auxiliary_events.is_empty()
        && value.cursor_moves.is_empty()
        && matches!(
            &value.status,
            MusicXmlNotationEventsStatusV1::Unknown { reason }
                if reason == "legacy graph predates the notation-event layer"
        )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChordAnchor {
    source_order: usize,
    onset: RationalBeat,
    duration: Option<RationalBeat>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParseEventError {
    Unknown(String),
}

impl ParseEventError {
    fn reason(&self) -> String {
        match self {
            Self::Unknown(reason) => reason.clone(),
        }
    }
}

fn unknown(part_id: &str, reason: impl Into<String>) -> MusicXmlPartNotationEventsV1 {
    MusicXmlPartNotationEventsV1 {
        schema: MUSICXML_NOTATION_EVENTS_SCHEMA_V1.into(),
        canonicalization_version: MUSICXML_NOTATION_EVENTS_CANONICALIZATION_V1.into(),
        part_id: part_id.to_string(),
        status: MusicXmlNotationEventsStatusV1::Unknown {
            reason: reason.into(),
        },
        note_events: Vec::new(),
        attribute_events: Vec::new(),
        auxiliary_events: Vec::new(),
        cursor_moves: Vec::new(),
    }
}

/// Resolve the matching <score-part> record from the document's
/// <part-list>. It remains optional because valid MusicXML documents may
/// omit a part-list; absence is None, never guessed from the note body.
/// Duplicate declarations fail closed.
pub fn parse_musicxml_part_metadata(
    root: roxmltree::Node<'_, '_>,
    part_id: &str,
) -> Result<Option<MusicXmlPartMetadataV1>, String> {
    if !root.has_tag_name("score-partwise") || part_id.trim().is_empty() {
        return Err("invalid MusicXML root or empty part identity".into());
    }
    let Some(part_list) = root.children().find(|node| node.has_tag_name("part-list")) else {
        return Ok(None);
    };
    let mut matching = part_list.children().filter(|node| {
        node.has_tag_name("score-part") && node.attribute("id") == Some(part_id)
    });
    let Some(score_part) = matching.next() else {
        return Ok(None);
    };
    if matching.next().is_some() {
        return Err(format!("duplicate score-part metadata for {part_id}"));
    }
    let mut total_nodes = 0usize;
    let source_score_part = source_node(score_part, 0, &mut total_nodes)
        .map_err(|error| error.reason())?;
    let score_instrument_ids = score_part
        .children()
        .filter(|node| node.has_tag_name("score-instrument"))
        .filter_map(|node| node.attribute("id").map(str::to_string))
        .collect();
    Ok(Some(MusicXmlPartMetadataV1 {
        source_part_id: part_id.to_string(),
        part_name: child_text(score_part, "part-name"),
        part_abbreviation: child_text(score_part, "part-abbreviation"),
        score_instrument_ids,
        source_score_part,
    }))
}

/// Parse note/rest and score-order attribute semantics for one part.
///
/// Measure spans come from the explicit MusicXML measure adapter. This parser
/// never infers a part or barline from note density or the legacy Score meter.
/// If any source position cannot be established, it returns Unknown with no
/// partial event list.
pub fn parse_musicxml_part_notation_events(
    part: roxmltree::Node<'_, '_>,
    part_id: &str,
    measure_spans: &[MeasureSpan],
) -> MusicXmlPartNotationEventsV1 {
    if part_id.trim().is_empty() {
        return unknown(part_id, "empty MusicXML part identity");
    }
    if !part.has_tag_name("part") || part.attribute("id") != Some(part_id) {
        return unknown(
            part_id,
            "supplied part identity does not match the source MusicXML part locator",
        );
    }
    match parse_part_events(part, part_id, measure_spans) {
        Ok((note_events, attribute_events, auxiliary_events, cursor_moves)) => MusicXmlPartNotationEventsV1 {
            schema: MUSICXML_NOTATION_EVENTS_SCHEMA_V1.into(),
            canonicalization_version: MUSICXML_NOTATION_EVENTS_CANONICALIZATION_V1.into(),
            part_id: part_id.to_string(),
            status: MusicXmlNotationEventsStatusV1::Supported,
            note_events,
            attribute_events,
            auxiliary_events,
            cursor_moves,
        },
        Err(error) => unknown(part_id, error.reason()),
    }
}

fn parse_part_events(
    part: roxmltree::Node<'_, '_>,
    part_id: &str,
    measure_spans: &[MeasureSpan],
) -> Result<
    (
        Vec<MusicXmlNotationEventV1>,
        Vec<MusicXmlAttributeEventV1>,
        Vec<MusicXmlAuxiliaryEventV1>,
        Vec<MusicXmlCursorMoveV1>,
    ),
    ParseEventError,
> {
    let measure_nodes: Vec<_> = part
        .children()
        .filter(|node| node.has_tag_name("measure"))
        .collect();
    if measure_nodes.len() != measure_spans.len() || measure_nodes.is_empty() {
        return Err(ParseEventError::Unknown(
            "explicit measure nodes and supported measure spans do not align".into(),
        ));
    }

    let zero = beat(0, 1)?;
    let mut divisions: Option<i64> = None;
    let mut source_order = 0usize;
    let mut source_tree_nodes = 0usize;
    let mut note_events = Vec::new();
    let mut attribute_events = Vec::new();
    let mut auxiliary_events = Vec::new();
    let mut cursor_moves = Vec::new();

    for (measure_index, measure) in measure_nodes.iter().enumerate() {
        let measure_start = measure_spans[measure_index].start;
        let measure_duration = measure_spans[measure_index]
            .end
            .checked_sub(measure_start)
            .map_err(|_| ParseEventError::Unknown("invalid measure span".into()))?;
        let measure_number = measure.attribute("number").map(str::to_string);
        let mut cursor = zero;
        let mut chord_anchor: Option<ChordAnchor> = None;

        for element in measure.children().filter(|node| node.is_element()) {
            if element.has_tag_name("attributes") {
                chord_anchor = None;
                for attribute in element.children().filter(|node| node.is_element()) {
                    let source = source_node(attribute, 0, &mut source_tree_nodes)?;
                    attribute_events.push(MusicXmlAttributeEventV1 {
                        source_order,
                        measure_index,
                        measure_number: measure_number.clone(),
                        part_id: part_id.to_string(),
                        score_offset: measure_start
                            .checked_add(cursor)
                            .map_err(|_| ParseEventError::Unknown("attribute score offset overflow".into()))?,
                        source,
                    });
                    source_order = next_order(source_order)?;
                    if attribute.has_tag_name("divisions") {
                        let value = parse_i64(attribute.text()).ok_or_else(|| {
                            ParseEventError::Unknown("invalid MusicXML divisions value".into())
                        })?;
                        if value <= 0 {
                            return Err(ParseEventError::Unknown(
                                "MusicXML divisions must be positive".into(),
                            ));
                        }
                        divisions = Some(value);
                    }
                }
            } else if element.has_tag_name("note") {
                if note_events.len() >= MAX_NOTATION_EVENTS_PER_PART {
                    return Err(ParseEventError::Unknown(
                        "MusicXML note-event budget exceeded".into(),
                    ));
                }
                let is_grace = element.children().any(|node| node.has_tag_name("grace"));
                let is_cue = element.children().any(|node| node.has_tag_name("cue"));
                let is_chord_member = element.children().any(|node| node.has_tag_name("chord"));
                let duration_node = element
                    .children()
                    .find(|node| node.has_tag_name("duration"));
                let performed_duration = match duration_node {
                    Some(node) => {
                        let raw = parse_i64(node.text()).ok_or_else(|| {
                            ParseEventError::Unknown("invalid MusicXML note duration".into())
                        })?;
                        if raw <= 0 {
                            return Err(ParseEventError::Unknown(
                                "MusicXML note duration must be positive".into(),
                            ));
                        }
                        let division_value = divisions.ok_or_else(|| {
                            ParseEventError::Unknown(
                                "note duration appears before an explicit divisions value".into(),
                            )
                        })?;
                        Some(
                            RationalBeat::new(raw, division_value)
                                .map_err(|_| ParseEventError::Unknown("note duration overflow".into()))?,
                        )
                    }
                    None if is_grace => None,
                    None => {
                        return Err(ParseEventError::Unknown(
                            "non-grace note has no duration".into(),
                        ));
                    }
                };
                // A grace note may retain a source duration annotation, but
                // it never advances the score cursor (the branch below enforces it).

                let (note_onset, chord_anchor_source_order) = if is_chord_member {
                    let anchor = chord_anchor.as_ref().ok_or_else(|| {
                        ParseEventError::Unknown(
                            "chord member has no immediately preceding base note".into(),
                        )
                    })?;
                    if let (Some(duration), Some(base_duration)) =
                        (performed_duration, anchor.duration)
                    {
                        if duration.compare_exact(base_duration) == std::cmp::Ordering::Greater {
                            return Err(ParseEventError::Unknown(
                                "chord-member duration exceeds its base note".into(),
                            ));
                        }
                    }
                    (anchor.onset, Some(anchor.source_order))
                } else {
                    (cursor, None)
                };

                if let Some(duration) = performed_duration {
                    let event_end = note_onset
                        .checked_add(duration)
                        .map_err(|_| ParseEventError::Unknown("note end overflow".into()))?;
                    if event_end.compare_exact(measure_duration) == std::cmp::Ordering::Greater {
                        return Err(ParseEventError::Unknown(
                            "note event extends beyond its explicit measure span".into(),
                        ));
                    }
                }
                let score_offset = measure_start
                    .checked_add(note_onset)
                    .map_err(|_| ParseEventError::Unknown("note score offset overflow".into()))?;
                let kind = parse_event_kind(element)?;
                let grace = element
                    .children()
                    .find(|node| node.has_tag_name("grace"))
                    .map(|node| {
                        Ok(MusicXmlGraceV1 {
                            slash: node.attribute("slash").map(str::to_string),
                            steal_time_previous: node.attribute("steal-time-previous").map(str::to_string),
                            steal_time_following: node.attribute("steal-time-following").map(str::to_string),
                            make_time: node.attribute("make-time").map(str::to_string),
                            source: source_node(node, 0, &mut source_tree_nodes)?,
                        })
                    })
                    .transpose()?;
                let time_modification = element
                    .children()
                    .find(|node| node.has_tag_name("time-modification"))
                    .map(parse_time_modification)
                    .transpose()?;
                let source_note = source_node(element, 0, &mut source_tree_nodes)?;
                let sound_ties = element
                    .children()
                    .filter(|node| node.has_tag_name("tie"))
                    .map(parse_tie)
                    .collect();
                let visual_ties = element
                    .children()
                    .filter(|node| node.has_tag_name("notations"))
                    .flat_map(|node| node.children())
                    .filter(|node| node.has_tag_name("tied"))
                    .map(parse_tie)
                    .collect();
                let tuplet_notations = element
                    .children()
                    .filter(|node| node.has_tag_name("notations"))
                    .flat_map(|node| node.children())
                    .filter(|node| node.has_tag_name("tuplet"))
                    .map(|node| source_node(node, 0, &mut source_tree_nodes))
                    .collect::<Result<Vec<_>, _>>()?;
                let instrument_ids = element
                    .children()
                    .filter(|node| node.has_tag_name("instrument"))
                    .filter_map(|node| node.attribute("id").map(str::to_string))
                    .collect();
                let beams = element
                    .children()
                    .filter(|node| node.has_tag_name("beam"))
                    .map(|node| MusicXmlBeamV1 {
                        number: node.attribute("number").map(str::to_string),
                        value: node.text().map(|value| value.trim().to_string()),
                        attributes: attributes(node),
                    })
                    .collect();
                let note_type = child_text(element, "type");
                let dot_count = element
                    .children()
                    .filter(|node| node.has_tag_name("dot"))
                    .count()
                    .min(255) as u8;
                let stem = child_text(element, "stem");
                let notehead = element
                    .children()
                    .find(|node| node.has_tag_name("notehead"))
                    .map(|node| source_node(node, 0, &mut source_tree_nodes))
                    .transpose()?;
                let accidental = element
                    .children()
                    .find(|node| node.has_tag_name("accidental"))
                    .map(|node| source_node(node, 0, &mut source_tree_nodes))
                    .transpose()?;
                let voice = child_text(element, "voice");
                let staff = child_text(element, "staff");

                note_events.push(MusicXmlNotationEventV1 {
                    source_order,
                    source_measure_index: measure_index,
                    source_measure_number: measure_number.clone(),
                    part_id: part_id.to_string(),
                    score_offset,
                    time_only: element.attribute("time-only").map(str::to_string),
                    attack_divisions: element.attribute("attack").map(str::to_string),
                    release_divisions: element.attribute("release").map(str::to_string),
                    performed_duration,
                    kind,
                    is_grace,
                    grace,
                    is_cue,
                    is_chord_member,
                    chord_anchor_source_order,
                    voice,
                    staff,
                    instrument_ids,
                    written_type: note_type,
                    dot_count,
                    time_modification,
                    sound_ties,
                    visual_ties,
                    tuplet_notations,
                    stem,
                    notehead,
                    accidental,
                    beams,
                    source_note,
                });

                let this_order = source_order;
                source_order = next_order(source_order)?;
                if !is_chord_member {
                    chord_anchor = Some(ChordAnchor {
                        source_order: this_order,
                        onset: note_onset,
                        duration: performed_duration,
                    });
                    if !is_grace {
                        if let Some(duration) = performed_duration {
                            cursor = cursor
                                .checked_add(duration)
                                .map_err(|_| ParseEventError::Unknown("note cursor overflow".into()))?;
                        }
                    }
                }
            } else if element.has_tag_name("backup") || element.has_tag_name("forward") {
                chord_anchor = None;
                let duration_node = element
                    .children()
                    .find(|node| node.has_tag_name("duration"))
                    .ok_or_else(|| ParseEventError::Unknown("backup/forward has no duration".into()))?;
                let raw = parse_i64(duration_node.text()).ok_or_else(|| {
                    ParseEventError::Unknown("invalid backup/forward duration".into())
                })?;
                if raw <= 0 {
                    return Err(ParseEventError::Unknown(
                        "backup/forward duration must be positive".into(),
                    ));
                }
                let division_value = divisions.ok_or_else(|| {
                    ParseEventError::Unknown(
                        "backup/forward appears before explicit divisions".into(),
                    )
                })?;
                let duration = RationalBeat::new(raw, division_value)
                    .map_err(|_| ParseEventError::Unknown("backup/forward duration overflow".into()))?;
                let before = cursor;
                cursor = if element.has_tag_name("backup") {
                    cursor
                        .checked_sub(duration)
                        .map_err(|_| ParseEventError::Unknown("MusicXML backup underflow".into()))?
                } else {
                    cursor
                        .checked_add(duration)
                        .map_err(|_| ParseEventError::Unknown("MusicXML forward overflow".into()))?
                };
                if cursor.compare_exact(measure_duration) == std::cmp::Ordering::Greater {
                    return Err(ParseEventError::Unknown(
                        "backup/forward cursor extends beyond its explicit measure span".into(),
                    ));
                }
                cursor_moves.push(MusicXmlCursorMoveV1 {
                    source_order,
                    measure_index,
                    measure_number: measure_number.clone(),
                    part_id: part_id.to_string(),
                    direction: if element.has_tag_name("backup") { "backup" } else { "forward" }.into(),
                    voice: child_text(element, "voice"),
                    staff: child_text(element, "staff"),
                    score_offset_before: measure_start
                        .checked_add(before)
                        .map_err(|_| ParseEventError::Unknown("cursor move offset overflow".into()))?,
                    duration,
                    score_offset_after: measure_start
                        .checked_add(cursor)
                        .map_err(|_| ParseEventError::Unknown("cursor move offset overflow".into()))?,
                    source: source_node(element, 0, &mut source_tree_nodes)?,
                });
                source_order = next_order(source_order)?;
            } else {
                chord_anchor = None;
                // Retain every other measure-level element at its actual
                // cursor position. Direction/harmony offsets are signed in
                // divisions and must not be silently rounded or discarded.
                let offset = element
                    .children()
                    .find(|node| node.has_tag_name("offset"))
                    .map(|node| {
                        let raw = parse_i64(node.text()).ok_or_else(|| {
                            ParseEventError::Unknown("invalid MusicXML auxiliary offset".into())
                        })?;
                        let magnitude = raw.checked_abs().ok_or_else(|| {
                            ParseEventError::Unknown("MusicXML auxiliary offset overflow".into())
                        })?;
                        let division_value = divisions.ok_or_else(|| {
                            ParseEventError::Unknown(
                                "MusicXML auxiliary offset appears before explicit divisions".into(),
                            )
                        })?;
                        let duration = RationalBeat::new(magnitude, division_value)
                            .map_err(|_| ParseEventError::Unknown("auxiliary offset overflow".into()))?;
                        Ok::<_, ParseEventError>((raw < 0, duration))
                    })
                    .transpose()?;
                let mut score_offset = measure_start
                    .checked_add(cursor)
                    .map_err(|_| ParseEventError::Unknown("auxiliary score offset overflow".into()))?;
                if let Some((negative, amount)) = offset {
                    score_offset = if negative {
                        score_offset.checked_sub(amount)
                    } else {
                        score_offset.checked_add(amount)
                    }
                    .map_err(|_| ParseEventError::Unknown(
                        "MusicXML auxiliary offset leaves the supported score coordinate range".into()
                    ))?;
                }
                let measure_end = measure_start
                    .checked_add(measure_duration)
                    .map_err(|_| ParseEventError::Unknown("measure end overflow".into()))?;
                if score_offset.compare_exact(measure_start) == std::cmp::Ordering::Less
                    || score_offset.compare_exact(measure_end) == std::cmp::Ordering::Greater
                {
                    return Err(ParseEventError::Unknown(
                        "MusicXML auxiliary offset leaves its explicit measure span".into(),
                    ));
                }
                let source = source_node(element, 0, &mut source_tree_nodes)?;
                auxiliary_events.push(MusicXmlAuxiliaryEventV1 {
                    source_order,
                    measure_index,
                    measure_number: measure_number.clone(),
                    part_id: part_id.to_string(),
                    score_offset,
                    source,
                });
                source_order = next_order(source_order)?;
            }
        }
    }

    Ok((note_events, attribute_events, auxiliary_events, cursor_moves))
}

fn parse_event_kind(
    note: roxmltree::Node<'_, '_>,
) -> Result<MusicXmlNotationEventKindV1, ParseEventError> {
    let pitched = note.children().find(|node| node.has_tag_name("pitch"));
    let rest = note.children().find(|node| node.has_tag_name("rest"));
    let unpitched = note.children().find(|node| node.has_tag_name("unpitched"));
    let present = (pitched.is_some() as usize)
        + (rest.is_some() as usize)
        + (unpitched.is_some() as usize);
    if present != 1 {
        return Err(ParseEventError::Unknown(
            "note must contain exactly one of pitch, rest or unpitched".into(),
        ));
    }
    if let Some(pitch) = pitched {
        let step = child_text(pitch, "step")
            .ok_or_else(|| ParseEventError::Unknown("pitched note has no step".into()))?;
        if !matches!(step.as_str(), "A" | "B" | "C" | "D" | "E" | "F" | "G") {
            return Err(ParseEventError::Unknown("invalid MusicXML pitch step".into()));
        }
        let octave = child_text(pitch, "octave")
            .and_then(|value| value.parse::<i32>().ok())
            .ok_or_else(|| ParseEventError::Unknown("pitched note has invalid octave".into()))?;
        Ok(MusicXmlNotationEventKindV1::Pitched(MusicXmlWrittenPitchV1 {
            step,
            alter: child_text(pitch, "alter"),
            octave,
        }))
    } else if let Some(rest) = rest {
        Ok(MusicXmlNotationEventKindV1::Rest {
            measure: rest.attribute("measure").is_some_and(|value| value == "yes"),
            display: display_pitch(rest),
        })
    } else {
        Ok(MusicXmlNotationEventKindV1::Unpitched {
            display: display_pitch(unpitched.unwrap()),
        })
    }
}

fn display_pitch(node: roxmltree::Node<'_, '_>) -> MusicXmlDisplayPitchV1 {
    MusicXmlDisplayPitchV1 {
        step: child_text(node, "display-step"),
        octave: child_text(node, "display-octave").and_then(|value| value.parse::<i32>().ok()),
    }
}

fn parse_time_modification(
    node: roxmltree::Node<'_, '_>,
) -> Result<MusicXmlTimeModificationV1, ParseEventError> {
    let actual_notes = child_text(node, "actual-notes")
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| ParseEventError::Unknown("invalid tuplet actual-notes".into()))?;
    let normal_notes = child_text(node, "normal-notes")
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| ParseEventError::Unknown("invalid tuplet normal-notes".into()))?;
    Ok(MusicXmlTimeModificationV1 {
        actual_notes,
        normal_notes,
        normal_type: child_text(node, "normal-type"),
        normal_dot_count: node
            .children()
            .filter(|child| child.has_tag_name("normal-dot"))
            .count()
            .min(255) as u8,
    })
}

fn parse_tie(node: roxmltree::Node<'_, '_>) -> MusicXmlTieV1 {
    MusicXmlTieV1 {
        tie_type: node.attribute("type").map(str::to_string),
        time_only: node.attribute("time-only").map(str::to_string),
        attributes: attributes(node),
    }
}

fn source_node(
    node: roxmltree::Node<'_, '_>,
    depth: usize,
    total_nodes: &mut usize,
) -> Result<MusicXmlSourceNodeV1, ParseEventError> {
    if depth > MAX_SOURCE_TREE_DEPTH {
        return Err(ParseEventError::Unknown(
            "MusicXML source subtree exceeded maximum depth".into(),
        ));
    }
    *total_nodes = total_nodes
        .checked_add(1)
        .ok_or_else(|| ParseEventError::Unknown("MusicXML source tree node counter overflow".into()))?;
    if *total_nodes > MAX_SOURCE_TREE_NODES {
        return Err(ParseEventError::Unknown(
            "MusicXML source tree node budget exceeded".into(),
        ));
    }
    let mut children = Vec::new();
    let mut content = Vec::new();
    for child in node.children() {
        if child.is_element() {
            let child_index = children.len();
            children.push(source_node(child, depth + 1, total_nodes)?);
            content.push(MusicXmlSourceContentV1::ElementIndex(child_index));
        } else if let Some(value) = child.text() {
            // Preserve exact text-node whitespace and its position relative
            // to nested XML elements. The convenience `text` field below
            // remains trimmed for simple leaf lookups.
            content.push(MusicXmlSourceContentV1::Text(value.to_string()));
        }
    }
    Ok(MusicXmlSourceNodeV1 {
        element: node.tag_name().name().to_string(),
        namespace_uri: node.tag_name().namespace().map(str::to_string),
        attributes: source_attributes(node),
        text: node
            .text()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        children,
        content,
    })
}

fn source_attributes(node: roxmltree::Node<'_, '_>) -> Vec<MusicXmlSourceAttributeV1> {
    node.attributes()
        .map(|attribute| MusicXmlSourceAttributeV1 {
            name: attribute.name().to_string(),
            namespace_uri: attribute.namespace().map(str::to_string),
            value: attribute.value().to_string(),
        })
        .collect()
}

fn attributes(node: roxmltree::Node<'_, '_>) -> Vec<(String, String)> {
    node.attributes()
        .map(|attribute| (attribute.name().to_string(), attribute.value().to_string()))
        .collect()
}

fn child_text(node: roxmltree::Node<'_, '_>, name: &str) -> Option<String> {
    node.children()
        .find(|child| child.has_tag_name(name))
        .and_then(|child| child.text())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_i64(value: Option<&str>) -> Option<i64> {
    value?.trim().parse().ok()
}

fn beat(num: i64, den: i64) -> Result<RationalBeat, ParseEventError> {
    RationalBeat::new(num, den)
        .map_err(|_| ParseEventError::Unknown("invalid rational score coordinate".into()))
}

fn next_order(order: usize) -> Result<usize, ParseEventError> {
    order
        .checked_add(1)
        .ok_or_else(|| ParseEventError::Unknown("MusicXML source-order overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::barline_eval::MeasureKind;

    fn span(start: (i64, i64), end: (i64, i64)) -> MeasureSpan {
        MeasureSpan {
            start: RationalBeat::new(start.0, start.1).unwrap(),
            end: RationalBeat::new(end.0, end.1).unwrap(),
            kind: MeasureKind::Regular,
        }
    }

    fn parse_with_span(
        xml: &str,
        end: (i64, i64),
    ) -> MusicXmlPartNotationEventsV1 {
        let document = roxmltree::Document::parse(xml).unwrap();
        let part = document
            .root_element()
            .children()
            .find(|node| node.has_tag_name("part"))
            .unwrap();
        let spans = [span((0, 1), end)];
        parse_musicxml_part_notation_events(part, "P1", &spans)
    }

    fn parse(xml: &str) -> MusicXmlPartNotationEventsV1 {
        parse_with_span(xml, (4, 1))
    }

    fn supported(events: &MusicXmlPartNotationEventsV1) {
        assert_eq!(
            events.status,
            MusicXmlNotationEventsStatusV1::Supported
        );
    }

    #[test]
    fn preserves_rests_grace_cue_unpitched_voice_staff_and_instruments() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves></attributes>
            <note><rest/><duration>4</duration><voice>1</voice><staff>1</staff></note>
            <note><grace slash="yes" steal-time-previous="20"/><pitch><step>D</step><alter>0.5</alter><octave>5</octave></pitch><voice>1</voice><staff>1</staff></note>
            <note><cue/><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><instrument id="P1-I2"/><voice>2</voice><staff>2</staff></note>
            <note><unpitched><display-step>F</display-step><display-octave>4</display-octave></unpitched><duration>8</duration><voice>2</voice><staff>2</staff></note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        assert_eq!(events.note_events.len(), 4);
        assert!(matches!(
            &events.note_events[0].kind,
            MusicXmlNotationEventKindV1::Rest { measure: false, .. }
        ));
        assert_eq!(events.note_events[0].score_offset, RationalBeat::new(0, 1).unwrap());
        assert_eq!(events.note_events[0].performed_duration, Some(RationalBeat::new(1, 1).unwrap()));
        assert!(events.note_events[1].is_grace);
        assert_eq!(events.note_events[1].performed_duration, None);
        assert_eq!(
            events.note_events[1].grace.as_ref().and_then(|grace| grace.steal_time_previous.as_deref()),
            Some("20")
        );
        match &events.note_events[1].kind {
            MusicXmlNotationEventKindV1::Pitched(pitch) => {
                assert_eq!(pitch.step, "D");
                assert_eq!(pitch.alter.as_deref(), Some("0.5"));
                assert_eq!(pitch.octave, 5);
            }
            other => panic!("expected written pitch, got {other:?}"),
        }
        assert!(events.note_events[2].is_cue);
        assert_eq!(events.note_events[2].voice.as_deref(), Some("2"));
        assert_eq!(events.note_events[2].staff.as_deref(), Some("2"));
        assert_eq!(events.note_events[2].instrument_ids, vec!["P1-I2"]);
        assert!(matches!(
            &events.note_events[3].kind,
            MusicXmlNotationEventKindV1::Unpitched { .. }
        ));
        assert_eq!(
            events.note_events[3].performed_duration,
            Some(RationalBeat::new(2, 1).unwrap())
        );

        let measure_rest_xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><rest measure="yes"/><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let measure_rest = parse(measure_rest_xml);
        supported(&measure_rest);
        assert!(matches!(
            &measure_rest.note_events[0].kind,
            MusicXmlNotationEventKindV1::Rest { measure: true, .. }
        ));
    }

    #[test]
    fn chord_members_share_base_onset_and_do_not_advance_measure_cursor() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>8</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration><voice>1</voice></note>
            <note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice></note>
            <note><pitch><step>G</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice></note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        assert_eq!(events.note_events.len(), 3);
        assert_eq!(events.note_events[0].score_offset, RationalBeat::new(0, 1).unwrap());
        assert_eq!(events.note_events[1].score_offset, RationalBeat::new(0, 1).unwrap());
        assert!(events.note_events[1].is_chord_member);
        assert_eq!(events.note_events[1].chord_anchor_source_order, Some(2));
        assert_eq!(events.note_events[2].score_offset, RationalBeat::new(2, 1).unwrap());
    }

    #[test]
    fn triplet_sounding_duration_and_written_type_are_both_preserved() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>12</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes><normal-type>eighth</normal-type><normal-dot/></time-modification><notations><tuplet type="start" number="1" bracket="no"><tuplet-actual><tuplet-number>3</tuplet-number></tuplet-actual><tuplet-normal><tuplet-number>2</tuplet-number></tuplet-normal></tuplet></notations></note>
            <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification></note>
            <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification><dot/></note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        assert_eq!(events.note_events.len(), 3);
        assert_eq!(events.note_events[0].performed_duration, Some(RationalBeat::new(1, 3).unwrap()));
        assert_eq!(events.note_events[1].score_offset, RationalBeat::new(1, 3).unwrap());
        assert_eq!(events.note_events[2].score_offset, RationalBeat::new(2, 3).unwrap());
        assert_eq!(events.note_events[0].written_type.as_deref(), Some("eighth"));
        let ratio = events.note_events[0].time_modification.as_ref().unwrap();
        assert_eq!((ratio.actual_notes, ratio.normal_notes), (3, 2));
        assert_eq!(ratio.normal_dot_count, 1);
        assert_eq!(events.note_events[0].tuplet_notations.len(), 1);
        assert_eq!(
            events.note_events[0].tuplet_notations[0]
                .attributes
                .iter()
                .map(|attribute| (attribute.name.as_str(), attribute.value.as_str()))
                .collect::<Vec<_>>(),
            vec![("type", "start"), ("number", "1"), ("bracket", "no")]
        );
        assert_eq!(events.note_events[2].dot_count, 1);
    }

    #[test]
    fn sound_tie_and_visual_tie_are_kept_as_distinct_semantics() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration><tie type="start"/><notations><tied type="start" number="2"/><slur type="start" number="1"/><articulations><staccato/></articulations></notations></note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        let note = &events.note_events[0];
        assert_eq!(note.sound_ties.len(), 1);
        assert_eq!(note.sound_ties[0].tie_type.as_deref(), Some("start"));
        assert_eq!(note.visual_ties.len(), 1);
        assert_eq!(note.visual_ties[0].tie_type.as_deref(), Some("start"));
        assert_eq!(note.tuplet_notations.len(), 0);
        let notations = note
            .source_note
            .children
            .iter()
            .find(|node| node.element == "notations")
            .unwrap();
        assert!(notations.children.iter().any(|node| node.element == "slur"));
        assert!(notations.children.iter().any(|node| node.element == "articulations"));
    }

    #[test]
    fn attributes_are_recorded_at_score_order_positions_with_exact_offsets() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration></note>
            <attributes><clef number="2"><sign>F</sign><line>4</line></clef><key><fifths>-2</fifths><mode>minor</mode></key></attributes>
            <note><pitch><step>D</step><octave>3</octave></pitch><duration>8</duration><staff>2</staff></note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        assert_eq!(events.attribute_events.len(), 5);
        assert_eq!(events.attribute_events[0].source.element, "divisions");
        assert_eq!(events.attribute_events[3].score_offset, RationalBeat::new(2, 1).unwrap());
        assert_eq!(events.attribute_events[3].source.element, "clef");
        assert_eq!(events.attribute_events[4].score_offset, RationalBeat::new(2, 1).unwrap());
        assert_eq!(events.attribute_events[4].source.element, "key");
        assert_eq!(events.note_events[1].score_offset, RationalBeat::new(2, 1).unwrap());
    }

    #[test]
    fn backup_and_forward_cursor_moves_preserve_secondary_voice_positions() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>6</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>5</octave></pitch><duration>16</duration><voice>1</voice><staff>1</staff></note>
            <backup><duration>16</duration></backup>
            <note><pitch><step>C</step><octave>3</octave></pitch><duration>8</duration><voice>2</voice><staff>2</staff></note>
            <forward><duration>8</duration></forward>
            <note><rest/><duration>8</duration><voice>2</voice><staff>2</staff></note>
        </measure></part></score-partwise>"#;
        let events = parse_with_span(xml, (6, 1));
        supported(&events);
        assert_eq!(events.note_events.len(), 3);
        assert_eq!(events.note_events[0].score_offset, RationalBeat::new(0, 1).unwrap());
        assert_eq!(events.note_events[1].score_offset, RationalBeat::new(0, 1).unwrap());
        assert_eq!(events.note_events[2].score_offset, RationalBeat::new(2, 1).unwrap());
        assert_eq!(events.cursor_moves.len(), 2);
        assert_eq!(events.cursor_moves[0].direction, "backup");
        assert_eq!(events.cursor_moves[1].direction, "forward");
    }

    #[test]
    fn source_order_and_unknown_note_extensions_are_preserved() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note id="n-1" time-only="2" attack="1"><pitch><step>F</step><octave>4</octave></pitch><duration>16</duration><future-symbol code="x">opaque</future-symbol></note>
            <note id="n-2"><rest/><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        assert_eq!(events.note_events[0].source_order, 2);
        assert_eq!(events.note_events[1].source_order, 3);
        let first = &events.note_events[0].source_note;
        assert_eq!(
            first.attributes.iter()
                .map(|attribute| (attribute.name.as_str(), attribute.value.as_str()))
                .collect::<Vec<_>>(),
            vec![("id", "n-1"), ("time-only", "2"), ("attack", "1")]
        );
        assert_eq!(events.note_events[0].time_only.as_deref(), Some("2"));
        assert_eq!(events.note_events[0].attack_divisions.as_deref(), Some("1"));
        assert!(first.children.iter().any(|node| node.element == "future-symbol"));
        assert_eq!(
            first.children.iter().find(|node| node.element == "future-symbol")
                .and_then(|node| node.text.as_deref()),
            Some("opaque")
        );
    }

    #[test]
    fn ambiguous_chord_and_invalid_duration_withhold_the_event_layer() {
        let no_anchor = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><chord/><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
        </measure></part></score-partwise>"#;
        assert!(matches!(
            parse(no_anchor).status,
            MusicXmlNotationEventsStatusV1::Unknown { .. }
        ));

        let invalid_duration = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>0</duration></note>
        </measure></part></score-partwise>"#;
        assert!(matches!(
            parse(invalid_duration).status,
            MusicXmlNotationEventsStatusV1::Unknown { .. }
        ));
    }


    #[test]
    fn event_extent_beyond_explicit_measure_span_withholds_all_events() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>20</duration></note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        assert!(matches!(
            events.status,
            MusicXmlNotationEventsStatusV1::Unknown { .. }
        ));
        assert!(events.note_events.is_empty());
        assert!(events.attribute_events.is_empty());
    }

    #[test]
    fn grace_chord_members_share_onset_without_inventing_duration() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><grace slash="yes"/><pitch><step>C</step><octave>5</octave></pitch><voice>1</voice></note>
            <note><grace slash="yes"/><chord/><pitch><step>E</step><octave>5</octave></pitch><voice>1</voice></note>
            <note><pitch><step>G</step><octave>5</octave></pitch><duration>16</duration><voice>1</voice></note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        assert_eq!(events.note_events.len(), 3);
        assert!(events.note_events[0].is_grace);
        assert!(events.note_events[1].is_grace);
        assert!(events.note_events[1].is_chord_member);
        assert_eq!(events.note_events[1].score_offset, events.note_events[0].score_offset);
        assert_eq!(events.note_events[1].chord_anchor_source_order, Some(2));
        assert_eq!(events.note_events[2].score_offset, RationalBeat::new(0, 1).unwrap());
    }

    #[test]
    fn opaque_mixed_content_retains_text_element_order() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration>
                <other-notation>before<future-code value="x"/>after</other-notation>
            </note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        let other = events.note_events[0].source_note.children.iter()
            .find(|node| node.element == "other-notation").unwrap();
        assert!(matches!(&other.content[0], MusicXmlSourceContentV1::Text(value) if value == "before"));
        assert!(matches!(&other.content[1], MusicXmlSourceContentV1::ElementIndex(0)));
        assert_eq!(other.children[0].element, "future-code");
        assert!(matches!(&other.content[2], MusicXmlSourceContentV1::Text(value) if value == "after"));
    }

    #[test]
    fn forward_cursor_move_keeps_voice_and_staff_identity() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <forward><duration>8</duration><voice>2</voice><staff>2</staff></forward>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        assert_eq!(events.cursor_moves.len(), 1);
        assert_eq!(events.cursor_moves[0].voice.as_deref(), Some("2"));
        assert_eq!(events.cursor_moves[0].staff.as_deref(), Some("2"));
        assert_eq!(events.cursor_moves[0].source.element, "forward");
    }

    #[test]
    fn measure_level_directions_harmony_and_sound_keep_exact_offsets_and_source_trees() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice></note>
            <direction><direction-type><dynamics><f/></dynamics></direction-type><offset>2</offset><sound tempo="88"/></direction>
            <harmony><root><root-step>G</root-step></root><kind>major</kind><offset>-2</offset></harmony>
            <note><rest/><duration>8</duration></note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        assert_eq!(events.auxiliary_events.len(), 2);
        assert_eq!(events.auxiliary_events[0].score_offset, RationalBeat::new(5, 2).unwrap());
        assert_eq!(events.auxiliary_events[1].score_offset, RationalBeat::new(3, 2).unwrap());
        assert_eq!(events.auxiliary_events[0].source.element, "direction");
        assert_eq!(events.auxiliary_events[1].source.element, "harmony");
        assert!(events.auxiliary_events[0].source.children.iter().any(|node| node.element == "sound"));
        assert_eq!(events.note_events.len(), 2);
        assert_eq!(events.note_events[1].score_offset, RationalBeat::new(2, 1).unwrap());
    }


    #[test]
    fn auxiliary_offsets_outside_explicit_measure_span_withhold_all_events() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <direction><direction-type><words>Too early</words></direction-type><offset>-1</offset></direction>
            <note><rest/><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        assert!(matches!(
            parse(xml).status,
            MusicXmlNotationEventsStatusV1::Unknown { .. }
        ));
    }


    #[test]
    fn part_metadata_preserves_score_instrument_and_midi_setup() {
        let xml = r#"<score-partwise>
            <part-list>
                <score-part id="P1">
                    <part-name>Violin I</part-name>
                    <part-abbreviation>Vln. I</part-abbreviation>
                    <score-instrument id="P1-I1"><instrument-name>Violin</instrument-name></score-instrument>
                    <midi-device>Port A</midi-device>
                    <midi-instrument id="P1-I1">
                        <midi-channel>1</midi-channel>
                        <midi-program>41</midi-program>
                        <volume>86</volume>
                    </midi-instrument>
                </score-part>
            </part-list>
            <part id="P1"><measure number="1"/></part>
        </score-partwise>"#;
        let document = roxmltree::Document::parse(xml).unwrap();
        let root = document.root_element();
        let metadata = parse_musicxml_part_metadata(root, "P1").unwrap().unwrap();
        assert_eq!(metadata.source_part_id, "P1");
        assert_eq!(metadata.part_name.as_deref(), Some("Violin I"));
        assert_eq!(metadata.part_abbreviation.as_deref(), Some("Vln. I"));
        assert_eq!(metadata.score_instrument_ids, vec!["P1-I1"]);
        let midi = metadata.source_score_part.children.iter()
            .find(|node| node.element == "midi-instrument").unwrap();
        assert_eq!(
            midi.children.iter().find(|node| node.element == "midi-program")
                .and_then(|node| node.text.as_deref()),
            Some("41")
        );
        assert!(metadata.source_score_part.children.iter()
            .any(|node| node.element == "midi-device"));
    }

    #[test]
    fn missing_or_duplicate_part_metadata_is_not_guessed() {
        let without_part_list = roxmltree::Document::parse(
            r#"<score-partwise><part id="P1"><measure/></part></score-partwise>"#
        ).unwrap();
        assert_eq!(
            parse_musicxml_part_metadata(without_part_list.root_element(), "P1").unwrap(),
            None
        );

        let duplicate = roxmltree::Document::parse(
            r#"<score-partwise><part-list>
                <score-part id="P1"><part-name>One</part-name></score-part>
                <score-part id="P1"><part-name>Two</part-name></score-part>
            </part-list><part id="P1"/></score-partwise>"#
        ).unwrap();
        assert!(parse_musicxml_part_metadata(duplicate.root_element(), "P1").is_err());
    }

    #[test]
    fn notation_event_parser_rejects_a_mismatched_source_part_id() {
        let xml = r#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><rest/><duration>16</duration></note>
        </measure></part></score-partwise>"#;
        let document = roxmltree::Document::parse(xml).unwrap();
        let part = document.root_element().children()
            .find(|node| node.has_tag_name("part")).unwrap();
        let events = parse_musicxml_part_notation_events(
            part, "forged-part-id", &[span((0, 1), (4, 1))]
        );
        assert!(matches!(events.status, MusicXmlNotationEventsStatusV1::Unknown { .. }));
        assert!(events.note_events.is_empty());
    }


    #[test]
    fn opaque_source_tree_preserves_element_and_attribute_namespace_uris() {
        let xml = r#"<score-partwise xmlns:ext="urn:example:extension-a" xmlns:other="urn:example:extension-b"><part id="P1"><measure number="1">
            <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note id="n1" ext:state="held" other:state="released" attack="-2" release="3" time-only="1">
                <pitch><step>C</step><octave>4</octave></pitch><duration>16</duration>
                <ext:articulation ext:kind="ornamental"><ext:label>first</ext:label></ext:articulation>
                <other:articulation other:kind="editorial"><other:label>second</other:label></other:articulation>
            </note>
        </measure></part></score-partwise>"#;
        let events = parse(xml);
        supported(&events);
        let note = &events.note_events[0];
        assert_eq!(note.time_only.as_deref(), Some("1"));
        assert_eq!(note.attack_divisions.as_deref(), Some("-2"));
        assert_eq!(note.release_divisions.as_deref(), Some("3"));
        let node = &note.source_note;
        let mut attribute_namespaces: Vec<_> = node.attributes.iter()
            .filter(|attribute| attribute.name == "state")
            .map(|attribute| attribute.namespace_uri.as_deref().unwrap())
            .collect();
        attribute_namespaces.sort_unstable();
        assert_eq!(
            attribute_namespaces,
            vec!["urn:example:extension-a", "urn:example:extension-b"]
        );
        let elements: Vec<_> = node.children.iter()
            .filter(|child| child.element == "articulation")
            .collect();
        assert_eq!(elements.len(), 2);
        assert_eq!(
            elements.iter().map(|child| child.namespace_uri.as_deref().unwrap()).collect::<Vec<_>>(),
            vec!["urn:example:extension-a", "urn:example:extension-b"]
        );
    }

}
