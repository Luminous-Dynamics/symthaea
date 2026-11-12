// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative MusicXML adapter for the explicit barline evaluation contract.
//!
//! Only score-partwise measure containers are handled in V1. A measure span
//! is normalized from its explicit content cursor and, where present, its
//! own declared time signature. This is not a MIDI meter-to-barline inference
//! path. Multi-part measure sequences must reconcile exactly after rational
//! normalization; disagreement yields Unknown.

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedMeasure {
    duration: RationalBeat,
    kind: MeasureKind,
    non_controlling: bool,
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
    if normalized_parts
        .iter()
        .flatten()
        .any(|measure| measure.non_controlling)
    {
        return Ok(unknown_map(
            source_id,
            score_clock_id,
            "a part declares non-controlling measures; use per-part measure maps",
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
/// parts. The contained score-clock ID is part-qualified so a consumer cannot
/// accidentally evaluate one part against another part's barline sequence.
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
    for part in parts {
        let part_id = part
            .attribute("id")
            .filter(|value| !value.trim().is_empty())
            .ok_or(MusicXmlMeasureAdapterError::MissingPartId)?;
        let parsed = parse_part(part)?;
        let (status, measures, non_controlling_indices) = match parsed {
            Some(parsed) if !parsed.is_empty() => {
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
            Some(_) => (
                MeasureMapStatus::Unknown {
                    reason: "part contains no explicit measure containers".into(),
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
                        expected_measure_length = parse_time_length(attribute);
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
}
