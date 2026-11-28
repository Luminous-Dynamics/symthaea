// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-valid-segment MIREX beat/downbeat scoring (MEL-DJ-011).
//!
//! Each valid region is scored independently. Event arrays are filtered to
//! the region and rebased to its local sample clock; no synthetic interval is
//! ever formed across an invalid gap. Segment score vectors are aggregated
//! by declared valid-frame duration.
//!
//! The MIREX 2026 JSIR notes specify independent segment scoring and
//! duration weighting. The precise five-second beat-trim origin is represented
//! as an explicit policy because the public note does not fully specify
//! whether the trim is global to the source timeline or local to each region.

use serde::{Deserialize, Serialize};

use crate::beat_benchmark::BeatValidSegment;
use crate::beat_metrics::{
    aggregate_mirex_beat_scores, score_mirex_beat_track_with_trim,
    score_mirex_downbeat_events, validate_mirex_beat_score_evidence,
    MirexBeatAggregationMode, MirexBeatMetricScores, MirexBeatScoreEvidenceV1,
    MirexBeatTrackReportV1,
    MirexDownbeatEventScoresV1, MirexValidSegmentPolicy, WeightedMirexBeatScores,
};
use crate::transition::MAX_SAFE_JSON_INTEGER;

/// Current schema version for segment-level metric results.
pub const MIREX_VALID_SEGMENT_SCORE_SCHEMA_VERSION: u16 = 1;

/// Initial-trim interpretation for beat events in valid segments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirexSegmentBeatTrimPolicy {
    /// Remove beat events before 5 seconds on the original source timeline,
    /// then score each remaining valid region on its local timeline.
    SourceGlobalFiveSeconds,
    /// Rebase each region to time zero and apply mir_eval's default 5-second
    /// trim independently within each valid segment.
    SegmentLocalFiveSeconds,
    /// Do not trim beat events before scoring each region.
    NoTrim,
}

/// Segment aggregation policy. Duration weighting is the MIREX JSIR mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirexSegmentAggregationMode {
    DurationWeighted,
    ArithmeticMeanPerSegment,
}

/// Explicit policy applied to a valid-segment score request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexValidSegmentScoringConfigV1 {
    pub beat_trim_policy: MirexSegmentBeatTrimPolicy,
    pub aggregation_mode: MirexSegmentAggregationMode,
}

/// One independently scored valid region.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexValidSegmentResultV1 {
    pub segment_index: u32,
    pub source_start_frame: u64,
    pub source_end_frame: u64,
    pub duration_frames: u64,
    pub reference_beats_in_segment: u64,
    pub estimated_beats_in_segment: u64,
    pub beat_report: MirexBeatTrackReportV1,
    pub downbeat_report: MirexDownbeatEventScoresV1,
}

/// Beat metric aggregate over valid segments.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexSegmentBeatAggregateV1 {
    pub aggregation_mode: MirexSegmentAggregationMode,
    pub segment_count: u64,
    pub total_weight_frames: u64,
    pub scores: MirexBeatMetricScores,
}

/// Downbeat metrics aggregated over valid segments.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexDownbeatSegmentAggregateV1 {
    pub aggregation_mode: MirexSegmentAggregationMode,
    pub segment_count: u64,
    pub total_weight_frames: u64,
    pub precision: f64,
    pub recall: f64,
    pub f_measure: f64,
}

/// Overall segment report, retaining every region-level score vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexValidSegmentScoreReportV1 {
    pub schema_version: u16,
    pub config: MirexValidSegmentScoringConfigV1,
    pub evidence: MirexBeatScoreEvidenceV1,
    pub segments: Vec<MirexValidSegmentResultV1>,
    pub reference_beats_excluded_by_valid_mask: u64,
    pub estimated_beats_excluded_by_valid_mask: u64,
    pub reference_downbeats_excluded_by_valid_mask: u64,
    pub estimated_downbeats_excluded_by_valid_mask: u64,
    pub beat_aggregate: MirexSegmentBeatAggregateV1,
    pub downbeat_aggregate: MirexDownbeatSegmentAggregateV1,
}

/// Deterministic segment-adapter input failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirexValidSegmentError {
    InvalidScoringEvidence,
    EvidenceMustDeclareValidSegments,
    ZeroSampleRate,
    EmptySourceClock,
    SourceClockExceedsJsonSafeInteger,
    SourceDurationExceedsMirEvalLimit,
    InvalidEventOrder,
    EventOutOfSourceBounds,
    EmptySegmentList,
    TooManySegments,
    TooManyEvents,
    EmptySegment,
    SegmentOutOfSourceBounds,
    SegmentsNotOrderedOrOverlapping,
    AggregateDurationExceedsJsonSafeInteger,
    MetricScorerRejectedInput,
}

/// Score every valid region independently and aggregate by declared duration.
///
/// All event arrays are non-decreasing source-local frame coordinates. A
/// beat/downbeat located outside every valid region is excluded. A segment's
/// event coordinates are rebased to zero before local scoring, preventing
/// artificial cross-gap intervals.
///
/// The supplied evidence must declare `DeclaredValidSegments` and carry the
/// exact valid-segment manifest digest.
pub fn score_mirex_valid_segments(
    reference_beat_frames: &[u64],
    estimated_beat_frames: &[u64],
    reference_downbeat_frames: &[u64],
    estimated_downbeat_frames: &[u64],
    sample_rate_hz: u32,
    total_source_frames: u64,
    valid_segments: &[BeatValidSegment],
    config: MirexValidSegmentScoringConfigV1,
    evidence: MirexBeatScoreEvidenceV1,
) -> Result<MirexValidSegmentScoreReportV1, MirexValidSegmentError> {
    validate_mirex_beat_score_evidence(&evidence)
        .map_err(|_| MirexValidSegmentError::InvalidScoringEvidence)?;
    if evidence.valid_segment_policy != MirexValidSegmentPolicy::DeclaredValidSegments
        || evidence.valid_segment_manifest_sha256.is_none()
    {
        return Err(MirexValidSegmentError::EvidenceMustDeclareValidSegments);
    }
    if sample_rate_hz == 0 {
        return Err(MirexValidSegmentError::ZeroSampleRate);
    }
    if total_source_frames == 0 {
        return Err(MirexValidSegmentError::EmptySourceClock);
    }
    if total_source_frames > MAX_SAFE_JSON_INTEGER {
        return Err(MirexValidSegmentError::SourceClockExceedsJsonSafeInteger);
    }
    if u128::from(total_source_frames.saturating_sub(1))
        > u128::from(sample_rate_hz) * 30_000u128
    {
        return Err(MirexValidSegmentError::SourceDurationExceedsMirEvalLimit);
    }
    let event_limit = crate::beat_benchmark::MAX_REFERENCE_EVENTS_PER_TRACK;
    if reference_beat_frames.len() > event_limit
        || estimated_beat_frames.len() > event_limit
        || reference_downbeat_frames.len() > event_limit
        || estimated_downbeat_frames.len() > event_limit
    {
        return Err(MirexValidSegmentError::TooManyEvents);
    }
    validate_source_events(
        reference_beat_frames,
        estimated_beat_frames,
        total_source_frames,
    )?;
    validate_source_events(
        reference_downbeat_frames,
        estimated_downbeat_frames,
        total_source_frames,
    )?;

    if valid_segments.is_empty() {
        return Err(MirexValidSegmentError::EmptySegmentList);
    }
    if valid_segments.len() > crate::beat_benchmark::MAX_VALID_SEGMENTS_PER_TRACK {
        return Err(MirexValidSegmentError::TooManySegments);
    }

    let mut previous_end = None;
    let mut total_weight = 0u128;
    let mut segment_results = Vec::with_capacity(valid_segments.len());
    let mut weighted_beats = Vec::with_capacity(valid_segments.len());

    let mut downbeat_precision_sum = 0.0;
    let mut downbeat_recall_sum = 0.0;
    let mut downbeat_f_measure_sum = 0.0;

    for (segment_index, segment) in valid_segments.iter().enumerate() {
        if segment.start_frame >= segment.end_frame {
            return Err(MirexValidSegmentError::EmptySegment);
        }
        if segment.end_frame > total_source_frames {
            return Err(MirexValidSegmentError::SegmentOutOfSourceBounds);
        }
        if previous_end.is_some_and(|end| segment.start_frame < end) {
            return Err(MirexValidSegmentError::SegmentsNotOrderedOrOverlapping);
        }
        previous_end = Some(segment.end_frame);

        let duration_frames = segment.end_frame - segment.start_frame;
        total_weight = total_weight.saturating_add(u128::from(duration_frames));
        if total_weight > u128::from(MAX_SAFE_JSON_INTEGER) {
            return Err(MirexValidSegmentError::AggregateDurationExceedsJsonSafeInteger);
        }

        let reference_beats_source = events_in_segment(
            reference_beat_frames,
            segment.start_frame,
            segment.end_frame,
        );
        let estimated_beats_source = events_in_segment(
            estimated_beat_frames,
            segment.start_frame,
            segment.end_frame,
        );
        let reference_downbeats_source = events_in_segment(
            reference_downbeat_frames,
            segment.start_frame,
            segment.end_frame,
        );
        let estimated_downbeats_source = events_in_segment(
            estimated_downbeat_frames,
            segment.start_frame,
            segment.end_frame,
        );

        let source_global_trim = config.beat_trim_policy
            == MirexSegmentBeatTrimPolicy::SourceGlobalFiveSeconds;
        let reference_beats = rebase_and_apply_global_trim(
            &reference_beats_source,
            segment.start_frame,
            sample_rate_hz,
            source_global_trim,
        );
        let estimated_beats = rebase_and_apply_global_trim(
            &estimated_beats_source,
            segment.start_frame,
            sample_rate_hz,
            source_global_trim,
        );
        let initial_trim = match config.beat_trim_policy {
            MirexSegmentBeatTrimPolicy::SourceGlobalFiveSeconds => 0,
            MirexSegmentBeatTrimPolicy::SegmentLocalFiveSeconds => 5,
            MirexSegmentBeatTrimPolicy::NoTrim => 0,
        };
        let beat_report = score_mirex_beat_track_with_trim(
            &reference_beats,
            &estimated_beats,
            sample_rate_hz,
            duration_frames,
            initial_trim,
        )
        .map_err(|_| MirexValidSegmentError::MetricScorerRejectedInput)?;

        let mut beat_report = beat_report;
        if source_global_trim {
            // The events were already filtered using the source-global clock.
            // Record the effective five-second trim in the scoring receipt;
            // input counts below are the segment's pre-trim counts.
            beat_report.scoring_receipt.initial_trim_seconds = 5;
            beat_report.initial_trim_seconds = 5;
            beat_report.reference_beats_input = reference_beats_source.len() as u64;
            beat_report.estimated_beats_input = estimated_beats_source.len() as u64;
        }

        let reference_downbeats = rebase_events(
            &reference_downbeats_source,
            segment.start_frame,
        );
        let estimated_downbeats = rebase_events(
            &estimated_downbeats_source,
            segment.start_frame,
        );
        let downbeat_report = score_mirex_downbeat_events(
            &reference_downbeats,
            &estimated_downbeats,
            sample_rate_hz,
            duration_frames,
        )
        .map_err(|_| MirexValidSegmentError::MetricScorerRejectedInput)?;

        weighted_beats.push(WeightedMirexBeatScores {
            scores: beat_report.scores,
            weight_frames: duration_frames,
        });
        let aggregation_weight = match config.aggregation_mode {
            MirexSegmentAggregationMode::DurationWeighted => duration_frames as f64,
            MirexSegmentAggregationMode::ArithmeticMeanPerSegment => 1.0,
        };
        downbeat_precision_sum += downbeat_report.precision * aggregation_weight;
        downbeat_recall_sum += downbeat_report.recall * aggregation_weight;
        downbeat_f_measure_sum += downbeat_report.f_measure * aggregation_weight;

        segment_results.push(MirexValidSegmentResultV1 {
            segment_index: segment_index as u32,
            source_start_frame: segment.start_frame,
            source_end_frame: segment.end_frame,
            duration_frames,
            reference_beats_in_segment: reference_beats_source.len() as u64,
            estimated_beats_in_segment: estimated_beats_source.len() as u64,
            beat_report,
            downbeat_report,
        });
    }

    let track_aggregate = aggregate_mirex_beat_scores(
        &weighted_beats,
        match config.aggregation_mode {
            MirexSegmentAggregationMode::DurationWeighted => {
                MirexBeatAggregationMode::DurationWeighted
            }
            MirexSegmentAggregationMode::ArithmeticMeanPerSegment => {
                MirexBeatAggregationMode::ArithmeticMeanPerTrack
            }
        },
    )
    .ok_or(MirexValidSegmentError::MetricScorerRejectedInput)?;
    let beat_aggregate = MirexSegmentBeatAggregateV1 {
        aggregation_mode: config.aggregation_mode,
        segment_count: track_aggregate.track_count,
        total_weight_frames: track_aggregate.total_weight_frames,
        scores: track_aggregate.scores,
    };
    let denominator = match config.aggregation_mode {
        MirexSegmentAggregationMode::DurationWeighted => total_weight as f64,
        MirexSegmentAggregationMode::ArithmeticMeanPerSegment => {
            segment_results.len() as f64
        }
    };
    let downbeat_aggregate = MirexDownbeatSegmentAggregateV1 {
        aggregation_mode: config.aggregation_mode,
        segment_count: segment_results.len() as u64,
        total_weight_frames: total_weight as u64,
        precision: downbeat_precision_sum / denominator,
        recall: downbeat_recall_sum / denominator,
        f_measure: downbeat_f_measure_sum / denominator,
    };

    Ok(MirexValidSegmentScoreReportV1 {
        schema_version: MIREX_VALID_SEGMENT_SCORE_SCHEMA_VERSION,
        config,
        evidence,
        segments: segment_results,
        reference_beats_excluded_by_valid_mask: count_events_outside_segments(
            reference_beat_frames,
            valid_segments,
        ) as u64,
        estimated_beats_excluded_by_valid_mask: count_events_outside_segments(
            estimated_beat_frames,
            valid_segments,
        ) as u64,
        reference_downbeats_excluded_by_valid_mask: count_events_outside_segments(
            reference_downbeat_frames,
            valid_segments,
        ) as u64,
        estimated_downbeats_excluded_by_valid_mask: count_events_outside_segments(
            estimated_downbeat_frames,
            valid_segments,
        ) as u64,
        beat_aggregate,
        downbeat_aggregate,
    })
}

fn validate_source_events(
    reference: &[u64],
    estimated: &[u64],
    total_frames: u64,
) -> Result<(), MirexValidSegmentError> {
    if reference.windows(2).any(|pair| pair[0] > pair[1])
        || estimated.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(MirexValidSegmentError::InvalidEventOrder);
    }
    if reference
        .iter()
        .chain(estimated.iter())
        .any(|frame| *frame >= total_frames)
    {
        return Err(MirexValidSegmentError::EventOutOfSourceBounds);
    }
    Ok(())
}

fn count_events_outside_segments(
    events: &[u64],
    segments: &[BeatValidSegment],
) -> usize {
    events
        .iter()
        .filter(|frame| {
            let insertion = segments.partition_point(|segment| segment.start_frame <= **frame);
            insertion == 0 || **frame >= segments[insertion - 1].end_frame
        })
        .count()
}

fn events_in_segment(events: &[u64], start_frame: u64, end_frame: u64) -> Vec<u64> {
    let start = events.partition_point(|frame| *frame < start_frame);
    let end = events.partition_point(|frame| *frame < end_frame);
    events[start..end].to_vec()
}

fn rebase_events(events: &[u64], segment_start: u64) -> Vec<u64> {
    events.iter().map(|frame| *frame - segment_start).collect()
}

fn rebase_and_apply_global_trim(
    events: &[u64],
    segment_start: u64,
    sample_rate_hz: u32,
    source_global_trim: bool,
) -> Vec<u64> {
    if source_global_trim {
        let trim_frame = u64::from(sample_rate_hz) * 5;
        events
            .iter()
            .filter(|frame| **frame >= trim_frame)
            .map(|frame| *frame - segment_start)
            .collect()
    } else {
        rebase_events(events, segment_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> MirexBeatScoreEvidenceV1 {
        MirexBeatScoreEvidenceV1 {
            track_id: "jsir-synthetic-track".to_string(),
            benchmark_manifest_sha256: "1".repeat(64),
            source_audio_sha256: "2".repeat(64),
            reference_annotation_sha256: "3".repeat(64),
            estimated_event_output_sha256: "4".repeat(64),
            source_master_family_id: "master-jsir-synthetic".to_string(),
            valid_segment_policy: MirexValidSegmentPolicy::DeclaredValidSegments,
            valid_segment_manifest_sha256: Some("5".repeat(64)),
        }
    }

    fn config(trim: MirexSegmentBeatTrimPolicy) -> MirexValidSegmentScoringConfigV1 {
        MirexValidSegmentScoringConfigV1 {
            beat_trim_policy: trim,
            aggregation_mode: MirexSegmentAggregationMode::DurationWeighted,
        }
    }

    #[test]
    fn checked_in_segment_score_schema_parses_and_pins_bounded_region_modes() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/mirex-valid-segment-score-v1.schema.json"
        ))
        .unwrap();
        assert_eq!(schema["properties"]["schema_version"]["const"], 1);
        assert_eq!(schema["$defs"]["score_evidence"]["properties"]["valid_segment_policy"]["const"],
            "declared_valid_segments");
        assert_eq!(
            schema["$defs"]["config"]["properties"]["beat_trim_policy"]["enum"].as_array().unwrap().len(),
            3
        );
        assert_eq!(
            schema["$defs"]["segment_result"]["properties"]["segment_index"]["maximum"]
                .as_u64(),
            Some(u64::from(u32::MAX))
        );
    }

    #[test]
    fn scores_each_region_independently_without_creating_cross_gap_intervals() {
        let result = score_mirex_valid_segments(
            &[5_000, 6_000, 7_000, 20_000, 21_000, 22_000],
            &[5_000, 6_000, 7_000, 20_000, 21_000, 22_000],
            &[5_000, 7_000, 20_000, 22_000],
            &[5_000, 7_000, 20_000, 22_000],
            1_000,
            30_000,
            &[
                BeatValidSegment {
                    start_frame: 5_000,
                    end_frame: 8_000,
                },
                BeatValidSegment {
                    start_frame: 20_000,
                    end_frame: 23_000,
                },
            ],
            config(MirexSegmentBeatTrimPolicy::NoTrim),
            evidence(),
        )
        .unwrap();
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].beat_report.scores.f_measure, 1.0);
        assert_eq!(result.segments[1].beat_report.scores.f_measure, 1.0);
        assert_eq!(result.beat_aggregate.scores.f_measure, 1.0);
        assert_eq!(result.beat_aggregate.total_weight_frames, 6_000);
    }

    #[test]
    fn source_global_trim_and_segment_local_trim_are_distinct_and_explicit() {
        let reference = [2_000, 4_000, 10_000, 11_000, 12_000, 13_000, 14_000, 15_000];
        let estimated = reference;
        let segments = [BeatValidSegment {
            start_frame: 10_000,
            end_frame: 16_000,
        }];
        let source_global = score_mirex_valid_segments(
            &reference,
            &estimated,
            &[],
            &[],
            1_000,
            20_000,
            &segments,
            config(MirexSegmentBeatTrimPolicy::SourceGlobalFiveSeconds),
            evidence(),
        )
        .unwrap();
        let segment_local = score_mirex_valid_segments(
            &reference,
            &estimated,
            &[],
            &[],
            1_000,
            20_000,
            &segments,
            config(MirexSegmentBeatTrimPolicy::SegmentLocalFiveSeconds),
            evidence(),
        )
        .unwrap();
        assert_eq!(source_global.segments[0].beat_report.reference_beats_scored, 6);
        assert_eq!(segment_local.segments[0].beat_report.reference_beats_scored, 1);
    }

    #[test]
    fn downbeats_never_receive_the_beat_task_initial_trim() {
        let result = score_mirex_valid_segments(
            &[1_000, 2_000, 3_000],
            &[1_000, 2_000, 3_000],
            &[1_000, 2_000],
            &[1_000, 2_000],
            1_000,
            10_000,
            &[BeatValidSegment {
                start_frame: 0,
                end_frame: 5_000,
            }],
            config(MirexSegmentBeatTrimPolicy::SegmentLocalFiveSeconds),
            evidence(),
        )
        .unwrap();
        assert_eq!(result.segments[0].downbeat_report.f_measure, 1.0);
        assert_eq!(result.segments[0].downbeat_report.scoring_receipt.initial_trim_seconds, 0);
    }

    #[test]
    fn duration_weighting_and_arithmetic_segment_mean_are_distinct() {
        let reference = [
            0, 1_000, 2_000, 3_000,
            10_000, 11_000, 12_000, 13_000, 14_000, 15_000, 16_000, 17_000,
        ];
        let estimated = [0, 1_000, 2_000, 3_000, 10_000];
        let segments = [
            BeatValidSegment {
                start_frame: 0,
                end_frame: 4_000,
            },
            BeatValidSegment {
                start_frame: 10_000,
                end_frame: 18_000,
            },
        ];
        let mut duration_config = config(MirexSegmentBeatTrimPolicy::NoTrim);
        duration_config.aggregation_mode = MirexSegmentAggregationMode::DurationWeighted;
        let mut arithmetic_config = duration_config;
        arithmetic_config.aggregation_mode =
            MirexSegmentAggregationMode::ArithmeticMeanPerSegment;

        let duration = score_mirex_valid_segments(
            &reference,
            &estimated,
            &[],
            &[],
            1_000,
            20_000,
            &segments,
            duration_config,
            evidence(),
        )
        .unwrap();
        let arithmetic = score_mirex_valid_segments(
            &reference,
            &estimated,
            &[],
            &[],
            1_000,
            20_000,
            &segments,
            arithmetic_config,
            evidence(),
        )
        .unwrap();
        assert!(
            (duration.beat_aggregate.scores.f_measure
                - arithmetic.beat_aggregate.scores.f_measure)
                .abs()
                > 0.1
        );
        assert_eq!(
            duration.beat_aggregate.aggregation_mode,
            MirexSegmentAggregationMode::DurationWeighted
        );
        assert_eq!(
            arithmetic.beat_aggregate.aggregation_mode,
            MirexSegmentAggregationMode::ArithmeticMeanPerSegment
        );
    }

    #[test]
    fn counts_events_outside_the_union_of_valid_regions_once() {
        let events = [0, 1_000, 5_000, 6_000, 30_000];
        let segments = [
            BeatValidSegment {
                start_frame: 0,
                end_frame: 2_000,
            },
            BeatValidSegment {
                start_frame: 5_000,
                end_frame: 7_000,
            },
        ];
        let result = score_mirex_valid_segments(
            &events,
            &events,
            &events,
            &events,
            1_000,
            40_000,
            &segments,
            config(MirexSegmentBeatTrimPolicy::NoTrim),
            evidence(),
        )
        .unwrap();
        assert_eq!(result.reference_beats_excluded_by_valid_mask, 1);
        assert_eq!(result.estimated_beats_excluded_by_valid_mask, 1);
        assert_eq!(result.reference_downbeats_excluded_by_valid_mask, 1);
        assert_eq!(result.estimated_downbeats_excluded_by_valid_mask, 1);
    }

    #[test]
    fn rejects_event_arrays_over_manifest_resource_limit() {
        let frames: Vec<u64> =
            (0..=crate::beat_benchmark::MAX_REFERENCE_EVENTS_PER_TRACK as u64)
                .collect();
        let error = score_mirex_valid_segments(
            &frames,
            &[],
            &[],
            &[],
            1_000,
            crate::beat_benchmark::MAX_REFERENCE_EVENTS_PER_TRACK as u64 + 1,
            &[BeatValidSegment {
                start_frame: 0,
                end_frame: crate::beat_benchmark::MAX_REFERENCE_EVENTS_PER_TRACK as u64 + 1,
            }],
            config(MirexSegmentBeatTrimPolicy::NoTrim),
            evidence(),
        )
        .unwrap_err();
        assert_eq!(error, MirexValidSegmentError::TooManyEvents);
    }

    #[test]
    fn rejects_overlapping_segments_before_scoring() {
        let error = score_mirex_valid_segments(
            &[1_000],
            &[1_000],
            &[],
            &[],
            1_000,
            10_000,
            &[
                BeatValidSegment { start_frame: 0, end_frame: 6_000 },
                BeatValidSegment { start_frame: 5_000, end_frame: 9_000 },
            ],
            config(MirexSegmentBeatTrimPolicy::NoTrim),
            evidence(),
        )
        .unwrap_err();
        assert_eq!(error, MirexValidSegmentError::SegmentsNotOrderedOrOverlapping);
    }
}
