// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded playback traversal derived from a MusicXML notation graph.
//!
//! The notation graph is immutable. Repeat expansion emits separate playback
//! visits that point back to the original part/measure indices. V1 supports
//! non-overlapping sequential forward/backward repeat regions only. Numbered
//! endings, D.C./D.S., segno/coda navigation, nested repeats and unrecognized
//! barline extensions are explicitly Unsupported rather than guessed.

use crate::evidence_digest::canonical_json_sha256;
use crate::musicxml_barline_adapter::{
    MusicXmlBarlineLocationV1, MusicXmlPartNotationGraphV1,
};
use serde::{Deserialize, Serialize};

pub const REPEAT_TRAVERSAL_SCHEMA_V1: &str = "muse-repeat-playback-traversal-v1";
pub const REPEAT_TRAVERSAL_ALGORITHM_V1: &str =
    "sequential-nonnested-forward-backward-repeat-expansion-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatTraversalConfigV1 {
    pub maximum_source_measures: usize,
    pub maximum_repeat_regions: usize,
    pub maximum_passes_per_region: u16,
    pub maximum_playback_visits: usize,
}

impl Default for RepeatTraversalConfigV1 {
    fn default() -> Self {
        Self {
            maximum_source_measures: 100_000,
            maximum_repeat_regions: 10_000,
            maximum_passes_per_region: 16,
            maximum_playback_visits: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatRegionV1 {
    pub region_id: u32,
    pub start_measure_index: usize,
    pub end_measure_index: usize,
    /// Total traversals of this region, including the initial pass.
    pub total_passes: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaybackVisitV1 {
    /// Zero-based position in the expanded playback sequence.
    pub playback_ordinal: u32,
    pub part_id: String,
    /// Stable reference to the unchanged notated measure.
    pub source_measure_index: usize,
    /// MusicXML measure number, if a barline event supplied one.
    pub source_measure_number: Option<String>,
    pub repeat_region_id: Option<u32>,
    /// One-based pass number; absent for measures outside repeat regions.
    pub repeat_pass: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaybackTraversalReportV1 {
    pub schema: String,
    pub algorithm: String,
    pub source_graph_sha256: String,
    pub part_id: String,
    pub config: RepeatTraversalConfigV1,
    pub repeat_regions: Vec<RepeatRegionV1>,
    pub visits: Vec<PlaybackVisitV1>,
    /// Canonical report digest excluding this field.
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackTraversalOutcomeV1 {
    Supported(PlaybackTraversalReportV1),
    Unsupported { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepeatTraversalError {
    InvalidConfig { field: String },
    SourceMeasureBudgetExceeded { count: usize, maximum: usize },
    RepeatRegionBudgetExceeded { count: usize, maximum: usize },
    PlaybackVisitBudgetExceeded { count: usize, maximum: usize },
    InvalidRepeatCount { value: String },
    RepeatPassBudgetExceeded { count: u16, maximum: u16 },
    InvalidRepeatBoundary { boundary: usize },
    UnmatchedForwardRepeat,
    ArithmeticOverflow,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RepeatDiscovery {
    Supported(Vec<RepeatRegionV1>),
    Unsupported(String),
}

/// Derive a performed measure sequence without modifying the notated graph.
///
/// V1 treats each part independently. A part with non-controlling measures
/// can still be traversed because its own measure lattice is authoritative.
/// This function does not synchronize playback between polymetric parts.
pub fn derive_playback_traversal(
    graph: &MusicXmlPartNotationGraphV1,
    config: &RepeatTraversalConfigV1,
) -> Result<PlaybackTraversalOutcomeV1, RepeatTraversalError> {
    validate_config(config)?;
    if graph.map.measures.len() > config.maximum_source_measures {
        return Err(RepeatTraversalError::SourceMeasureBudgetExceeded {
            count: graph.map.measures.len(),
            maximum: config.maximum_source_measures,
        });
    }
    if graph.map.measures.is_empty() {
        return Ok(PlaybackTraversalOutcomeV1::Unsupported {
            reason: "the part has no explicit notated measures".into(),
        });
    }
    match &graph.map.status {
        crate::barline_eval::MeasureMapStatus::Supported => {}
        crate::barline_eval::MeasureMapStatus::Unknown { reason } => {
            return Ok(PlaybackTraversalOutcomeV1::Unsupported {
                reason: format!("measure map is Unknown: {reason}"),
            });
        }
        crate::barline_eval::MeasureMapStatus::Unsupported { reason } => {
            return Ok(PlaybackTraversalOutcomeV1::Unsupported {
                reason: format!("measure map is Unsupported: {reason}"),
            });
        }
    }
    validate_graph(graph)?;

    let regions = match discover_repeat_regions(graph, config)? {
        RepeatDiscovery::Supported(regions) => regions,
        RepeatDiscovery::Unsupported(reason) => {
            return Ok(PlaybackTraversalOutcomeV1::Unsupported { reason });
        }
    };
    let expected_visits = expanded_visit_count(graph.map.measures.len(), &regions)?;
    if expected_visits > config.maximum_playback_visits {
        return Err(RepeatTraversalError::PlaybackVisitBudgetExceeded {
            count: expected_visits,
            maximum: config.maximum_playback_visits,
        });
    }

    let visits = expand_visits(graph, &regions, expected_visits)?;
    let mut report = PlaybackTraversalReportV1 {
        schema: REPEAT_TRAVERSAL_SCHEMA_V1.into(),
        algorithm: REPEAT_TRAVERSAL_ALGORITHM_V1.into(),
        source_graph_sha256: canonical_json_sha256(graph)
            .map_err(|_| RepeatTraversalError::SerializationFailed)?,
        part_id: graph.part_id.clone(),
        config: config.clone(),
        repeat_regions: regions,
        visits,
        report_sha256: String::new(),
    };
    report.report_sha256 =
        playback_traversal_commitment(&report).map_err(|_| RepeatTraversalError::SerializationFailed)?;
    Ok(PlaybackTraversalOutcomeV1::Supported(report))
}

fn validate_config(config: &RepeatTraversalConfigV1) -> Result<(), RepeatTraversalError> {
    if config.maximum_source_measures == 0 {
        return Err(RepeatTraversalError::InvalidConfig {
            field: "maximum_source_measures".into(),
        });
    }
    if config.maximum_repeat_regions == 0 {
        return Err(RepeatTraversalError::InvalidConfig {
            field: "maximum_repeat_regions".into(),
        });
    }
    if config.maximum_passes_per_region == 0 {
        return Err(RepeatTraversalError::InvalidConfig {
            field: "maximum_passes_per_region".into(),
        });
    }
    if config.maximum_playback_visits == 0 {
        return Err(RepeatTraversalError::InvalidConfig {
            field: "maximum_playback_visits".into(),
        });
    }
    Ok(())
}

fn validate_graph(graph: &MusicXmlPartNotationGraphV1) -> Result<(), RepeatTraversalError> {
    let zero = crate::barline_eval::RationalBeat::new(0, 1)
        .map_err(|_| RepeatTraversalError::ArithmeticOverflow)?;
    for (index, measure) in graph.map.measures.iter().enumerate() {
        if measure.start.compare_exact(measure.end) != std::cmp::Ordering::Less {
            return Err(RepeatTraversalError::InvalidRepeatBoundary { boundary: index });
        }
        if index == 0 && measure.start != zero {
            return Err(RepeatTraversalError::InvalidRepeatBoundary { boundary: index });
        }
        if index > 0 && graph.map.measures[index - 1].end != measure.start {
            return Err(RepeatTraversalError::InvalidRepeatBoundary { boundary: index });
        }
    }
    for event in &graph.barline_events {
        if event.part_id != graph.part_id || event.measure_index >= graph.map.measures.len() {
            return Err(RepeatTraversalError::InvalidRepeatBoundary {
                boundary: event.measure_index,
            });
        }
    }
    Ok(())
}

fn discover_repeat_regions(
    graph: &MusicXmlPartNotationGraphV1,
    config: &RepeatTraversalConfigV1,
) -> Result<RepeatDiscovery, RepeatTraversalError> {
    let mut events: Vec<_> = graph.barline_events.iter().collect();
    events.sort_by(|left, right| {
        left.score_offset
            .compare_exact(right.score_offset)
            .then_with(|| left.source_order.cmp(&right.source_order))
    });

    let mut forward_start: Option<usize> = None;
    let mut regions = Vec::new();
    for event in events {
        if event.ending.is_some() || event.segno.is_some() || event.coda.is_some() {
            return Ok(RepeatDiscovery::Unsupported(
                "numbered endings and segno/coda navigation are not implemented in playback traversal V1".into(),
            ));
        }
        if !event.unrecognized_attributes.is_empty() || !event.unrecognized_children.is_empty() {
            return Ok(RepeatDiscovery::Unsupported(
                "an unrecognized barline extension may affect playback order".into(),
            ));
        }
        let Some(repeat) = &event.repeat else {
            continue;
        };
        let Some(direction) = repeat.direction.as_deref() else {
            return Ok(RepeatDiscovery::Unsupported(
                "repeat event has no direction".into(),
            ));
        };
        let boundary = match (direction, event.location) {
            ("forward", MusicXmlBarlineLocationV1::Left) => event.measure_index,
            ("forward", MusicXmlBarlineLocationV1::Right) => event.measure_index + 1,
            ("backward", MusicXmlBarlineLocationV1::Right) => event.measure_index + 1,
            ("backward", MusicXmlBarlineLocationV1::Left) => event.measure_index,
            (_, MusicXmlBarlineLocationV1::Middle) => {
                return Ok(RepeatDiscovery::Unsupported(
                    "repeat markers on middle barlines are not supported".into(),
                ));
            }
            (other, _) => {
                return Ok(RepeatDiscovery::Unsupported(format!(
                    "unknown repeat direction: {other}"
                )));
            }
        };
        if boundary > graph.map.measures.len() {
            return Err(RepeatTraversalError::InvalidRepeatBoundary { boundary });
        }
        match direction {
            "forward" => {
                if repeat.times.is_some() {
                    return Ok(RepeatDiscovery::Unsupported(
                        "a repeat count on a forward marker is not interpreted by V1".into(),
                    ));
                }
                if forward_start.is_some() {
                    return Ok(RepeatDiscovery::Unsupported(
                        "nested or overlapping repeat regions are not supported by V1".into(),
                    ));
                }
                if boundary >= graph.map.measures.len() {
                    return Err(RepeatTraversalError::InvalidRepeatBoundary { boundary });
                }
                forward_start = Some(boundary);
            }
            "backward" => {
                let total_passes = match repeat.times.as_deref() {
                    None => 2,
                    Some(value) => value
                        .parse::<u16>()
                        .map_err(|_| RepeatTraversalError::InvalidRepeatCount {
                            value: value.to_string(),
                        })?,
                };
                if total_passes == 0 {
                    return Err(RepeatTraversalError::InvalidRepeatCount {
                        value: "0".into(),
                    });
                }
                if total_passes > config.maximum_passes_per_region {
                    return Err(RepeatTraversalError::RepeatPassBudgetExceeded {
                        count: total_passes,
                        maximum: config.maximum_passes_per_region,
                    });
                }
                let Some(end_measure_index) = boundary.checked_sub(1) else {
                    return Err(RepeatTraversalError::InvalidRepeatBoundary { boundary });
                };
                let start_measure_index = forward_start.take().unwrap_or(0);
                if end_measure_index < start_measure_index {
                    return Err(RepeatTraversalError::InvalidRepeatBoundary { boundary });
                }
                if regions.len() >= config.maximum_repeat_regions {
                    return Err(RepeatTraversalError::RepeatRegionBudgetExceeded {
                        count: regions.len() + 1,
                        maximum: config.maximum_repeat_regions,
                    });
                }
                let region_id = u32::try_from(regions.len())
                    .map_err(|_| RepeatTraversalError::ArithmeticOverflow)?;
                regions.push(RepeatRegionV1 {
                    region_id,
                    start_measure_index,
                    end_measure_index,
                    total_passes,
                });
            }
            _ => unreachable!("repeat direction was exhaustively handled above"),
        }
    }
    if forward_start.is_some() {
        return Err(RepeatTraversalError::UnmatchedForwardRepeat);
    }
    for adjacent in regions.windows(2) {
        if adjacent[0].end_measure_index >= adjacent[1].start_measure_index {
            return Ok(RepeatDiscovery::Unsupported(
                "overlapping repeat regions are not supported by V1".into(),
            ));
        }
    }
    Ok(RepeatDiscovery::Supported(regions))
}

fn expanded_visit_count(
    source_measure_count: usize,
    regions: &[RepeatRegionV1],
) -> Result<usize, RepeatTraversalError> {
    let mut count = source_measure_count;
    for region in regions {
        let length = region
            .end_measure_index
            .checked_sub(region.start_measure_index)
            .and_then(|value| value.checked_add(1))
            .ok_or(RepeatTraversalError::ArithmeticOverflow)?;
        let additional_passes = usize::from(region.total_passes - 1);
        let additional = length
            .checked_mul(additional_passes)
            .ok_or(RepeatTraversalError::ArithmeticOverflow)?;
        count = count
            .checked_add(additional)
            .ok_or(RepeatTraversalError::ArithmeticOverflow)?;
    }
    Ok(count)
}

fn source_measure_number(
    graph: &MusicXmlPartNotationGraphV1,
    measure_index: usize,
) -> Option<String> {
    graph
        .barline_events
        .iter()
        .find(|event| event.measure_index == measure_index)
        .map(|event| event.source_measure_number.clone())
        .filter(|number| !number.is_empty())
}

fn expand_visits(
    graph: &MusicXmlPartNotationGraphV1,
    regions: &[RepeatRegionV1],
    expected_visit_count: usize,
) -> Result<Vec<PlaybackVisitV1>, RepeatTraversalError> {
    let mut visits = Vec::with_capacity(expected_visit_count);
    let mut cursor = 0usize;
    for region in regions {
        while cursor < region.start_measure_index {
            push_visit(graph, &mut visits, cursor, None, None)?;
            cursor += 1;
        }
        for pass in 1..=region.total_passes {
            for measure_index in region.start_measure_index..=region.end_measure_index {
                push_visit(
                    graph,
                    &mut visits,
                    measure_index,
                    Some(region.region_id),
                    Some(pass),
                )?;
            }
        }
        cursor = region
            .end_measure_index
            .checked_add(1)
            .ok_or(RepeatTraversalError::ArithmeticOverflow)?;
    }
    while cursor < graph.map.measures.len() {
        push_visit(graph, &mut visits, cursor, None, None)?;
        cursor += 1;
    }
    if visits.len() != expected_visit_count {
        return Err(RepeatTraversalError::ArithmeticOverflow);
    }
    Ok(visits)
}

fn push_visit(
    graph: &MusicXmlPartNotationGraphV1,
    visits: &mut Vec<PlaybackVisitV1>,
    measure_index: usize,
    repeat_region_id: Option<u32>,
    repeat_pass: Option<u16>,
) -> Result<(), RepeatTraversalError> {
    let playback_ordinal = u32::try_from(visits.len())
        .map_err(|_| RepeatTraversalError::ArithmeticOverflow)?;
    visits.push(PlaybackVisitV1 {
        playback_ordinal,
        part_id: graph.part_id.clone(),
        source_measure_index: measure_index,
        source_measure_number: source_measure_number(graph, measure_index),
        repeat_region_id,
        repeat_pass,
    });
    Ok(())
}

/// Canonical commitment to a traversal report, excluding its own digest.
pub fn playback_traversal_commitment(
    report: &PlaybackTraversalReportV1,
) -> Result<String, serde_json::Error> {
    let mut payload = report.clone();
    payload.report_sha256.clear();
    canonical_json_sha256(&payload)
}

/// Replay the graph and configuration and compare the complete traversal.
/// Any changed graph, repeat command, config, visit ordering or commitment
/// returns false.
pub fn validate_playback_traversal_replay(
    graph: &MusicXmlPartNotationGraphV1,
    config: &RepeatTraversalConfigV1,
    supplied: &PlaybackTraversalReportV1,
) -> Result<bool, RepeatTraversalError> {
    match derive_playback_traversal(graph, config)? {
        PlaybackTraversalOutcomeV1::Supported(recomputed) => Ok(recomputed == *supplied),
        PlaybackTraversalOutcomeV1::Unsupported { .. } => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::musicxml_barline_adapter::parse_musicxml_part_notation_graph;

    fn graph(measures: &str) -> MusicXmlPartNotationGraphV1 {
        let xml = format!(
            r#"<score-partwise><part id="P1">
                {measures}
            </part></score-partwise>"#
        );
        parse_musicxml_part_notation_graph(
            xml.as_bytes(),
            "sha256:playback-fixture",
            "score-clock",
        )
        .unwrap()
        .remove(0)
    }

    fn measure(index: usize, extra: &str) -> String {
        let number = index + 1;
        if index == 0 {
            format!(
                r#"<measure number="{number}">
                    <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                    <note><rest/><duration>4</duration></note>
                    {extra}
                </measure>"#
            )
        } else {
            format!(
                r#"<measure number="{number}">
                    <note><rest/><duration>4</duration></note>
                    {extra}
                </measure>"#
            )
        }
    }

    fn graph_with_repeats(repeat_events: &[(&str, &str, &str)]) -> MusicXmlPartNotationGraphV1 {
        let mut xml_measures = Vec::new();
        for index in 0..4 {
            let mut barlines = String::new();
            for (measure_index, location, repeat) in repeat_events {
                if measure_index.parse::<usize>().ok() == Some(index) {
                    barlines.push_str(&format!(
                        r#"<barline location="{location}">{repeat}</barline>"#
                    ));
                }
            }
            xml_measures.push(measure(index, &barlines));
        }
        graph(&xml_measures.join("
"))
    }

    fn supported(report: PlaybackTraversalOutcomeV1) -> PlaybackTraversalReportV1 {
        match report {
            PlaybackTraversalOutcomeV1::Supported(report) => report,
            other => panic!("expected supported traversal, got {other:?}"),
        }
    }

    #[test]
    fn score_without_repeats_visits_each_notated_measure_once() {
        let graph = graph(&[
            measure(0, ""),
            measure(1, ""),
            measure(2, ""),
        ].join("
"));
        let original_measure_count = graph.map.measures.len();
        let report = supported(derive_playback_traversal(&graph, &RepeatTraversalConfigV1::default()).unwrap());
        assert_eq!(
            report.visits.iter().map(|visit| visit.source_measure_index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(graph.map.measures.len(), original_measure_count);
        assert!(report.repeat_regions.is_empty());
    }

    #[test]
    fn default_forward_backward_repeat_expands_two_total_passes() {
        let graph = graph_with_repeats(&[
            ("0", "left", r#"<repeat direction="forward"/>"#),
            ("1", "right", r#"<repeat direction="backward"/>"#),
        ]);
        let report = supported(derive_playback_traversal(&graph, &RepeatTraversalConfigV1::default()).unwrap());
        assert_eq!(
            report.visits.iter().map(|visit| visit.source_measure_index).collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 2, 3]
        );
        assert_eq!(report.repeat_regions.len(), 1);
        assert_eq!(report.repeat_regions[0].total_passes, 2);
        assert_eq!(report.visits[0].repeat_pass, Some(1));
        assert_eq!(report.visits[2].repeat_pass, Some(2));
    }

    #[test]
    fn explicit_repeat_count_three_produces_three_total_passes() {
        let graph = graph_with_repeats(&[
            ("0", "left", r#"<repeat direction="forward"/>"#),
            ("1", "right", r#"<repeat direction="backward" times="3"/>"#),
        ]);
        let report = supported(derive_playback_traversal(&graph, &RepeatTraversalConfigV1::default()).unwrap());
        assert_eq!(
            report.visits.iter().map(|visit| visit.source_measure_index).collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 0, 1, 2, 3]
        );
        assert_eq!(report.visits[4].repeat_pass, Some(3));
    }

    #[test]
    fn backward_repeat_without_forward_uses_score_start() {
        let graph = graph_with_repeats(&[
            ("1", "right", r#"<repeat direction="backward"/>"#),
        ]);
        let report = supported(derive_playback_traversal(&graph, &RepeatTraversalConfigV1::default()).unwrap());
        assert_eq!(
            report.visits.iter().map(|visit| visit.source_measure_index).collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 2, 3]
        );
    }

    #[test]
    fn sequential_repeat_regions_expand_independently() {
        let graph = graph_with_repeats(&[
            ("0", "left", r#"<repeat direction="forward"/>"#),
            ("1", "right", r#"<repeat direction="backward"/>"#),
            ("2", "left", r#"<repeat direction="forward"/>"#),
            ("3", "right", r#"<repeat direction="backward"/>"#),
        ]);
        let report = supported(derive_playback_traversal(&graph, &RepeatTraversalConfigV1::default()).unwrap());
        assert_eq!(
            report.visits.iter().map(|visit| visit.source_measure_index).collect::<Vec<_>>(),
            vec![0, 1, 0, 1, 2, 3, 2, 3]
        );
        assert_eq!(report.repeat_regions.len(), 2);
    }

    #[test]
    fn nested_repeats_are_explicitly_unsupported() {
        let graph = graph_with_repeats(&[
            ("0", "left", r#"<repeat direction="forward"/>"#),
            ("1", "left", r#"<repeat direction="forward"/>"#),
            ("2", "right", r#"<repeat direction="backward"/>"#),
        ]);
        assert!(matches!(
            derive_playback_traversal(&graph, &RepeatTraversalConfigV1::default()).unwrap(),
            PlaybackTraversalOutcomeV1::Unsupported { .. }
        ));
    }

    #[test]
    fn endings_and_navigation_signposts_are_not_guessed() {
        let with_ending = graph_with_repeats(&[
            ("0", "left", r#"<repeat direction="forward"/>"#),
            ("1", "right", r#"<repeat direction="backward"/><ending number="1" type="start"/>"#),
        ]);
        assert!(matches!(
            derive_playback_traversal(&with_ending, &RepeatTraversalConfigV1::default()).unwrap(),
            PlaybackTraversalOutcomeV1::Unsupported { .. }
        ));

        let with_segno = graph_with_repeats(&[
            ("0", "left", r#"<segno id="S1"/>"#),
        ]);
        assert!(matches!(
            derive_playback_traversal(&with_segno, &RepeatTraversalConfigV1::default()).unwrap(),
            PlaybackTraversalOutcomeV1::Unsupported { .. }
        ));
    }

    #[test]
    fn unknown_extensions_make_playback_traversal_unsupported() {
        let graph = graph_with_repeats(&[
            ("0", "left", r#"<future-navigation code="x"/>"#),
        ]);
        assert!(matches!(
            derive_playback_traversal(&graph, &RepeatTraversalConfigV1::default()).unwrap(),
            PlaybackTraversalOutcomeV1::Unsupported { .. }
        ));
    }

    #[test]
    fn zero_and_malformed_repeat_counts_are_rejected() {
        for value in ["0", "invalid"] {
            let repeat_markup = format!(
                r#"<repeat direction="backward" times="{value}"/>"#
            );
            let graph = graph_with_repeats(&[
                ("0", "left", r#"<repeat direction="forward"/>"#),
                ("1", "right", &repeat_markup),
            ]);
            assert!(matches!(
                derive_playback_traversal(&graph, &RepeatTraversalConfigV1::default()),
                Err(RepeatTraversalError::InvalidRepeatCount { .. })
            ));
        }
    }

    #[test]
    fn repeat_pass_and_playback_visit_budgets_fail_before_expansion() {
        let graph = graph_with_repeats(&[
            ("0", "left", r#"<repeat direction="forward"/>"#),
            ("1", "right", r#"<repeat direction="backward" times="4"/>"#),
        ]);
        let mut config = RepeatTraversalConfigV1::default();
        config.maximum_passes_per_region = 3;
        assert!(matches!(
            derive_playback_traversal(&graph, &config),
            Err(RepeatTraversalError::RepeatPassBudgetExceeded { .. })
        ));

        let mut config = RepeatTraversalConfigV1::default();
        config.maximum_playback_visits = 3;
        assert!(matches!(
            derive_playback_traversal(&graph, &config),
            Err(RepeatTraversalError::PlaybackVisitBudgetExceeded { .. })
        ));
    }

    #[test]
    fn graph_and_report_commitment_replay_reject_tampering() {
        let graph = graph_with_repeats(&[
            ("0", "left", r#"<repeat direction="forward"/>"#),
            ("1", "right", r#"<repeat direction="backward" times="2"/>"#),
        ]);
        let config = RepeatTraversalConfigV1::default();
        let report = supported(derive_playback_traversal(&graph, &config).unwrap());
        assert_eq!(report.source_graph_sha256.len(), 64);
        assert_eq!(report.report_sha256.len(), 64);
        assert!(validate_playback_traversal_replay(&graph, &config, &report).unwrap());

        let mut tampered = report.clone();
        tampered.visits[2].source_measure_index = 3;
        assert!(!validate_playback_traversal_replay(&graph, &config, &tampered).unwrap());

        let mut changed_graph = graph.clone();
        changed_graph.barline_events[1].repeat.as_mut().unwrap().times = Some("3".into());
        assert!(!validate_playback_traversal_replay(&changed_graph, &config, &report).unwrap());
    }
}
