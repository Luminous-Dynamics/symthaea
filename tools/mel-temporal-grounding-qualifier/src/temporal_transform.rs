//! Transformation-equivariance audit for artifact-bound musical event grounding.
//!
//! This module verifies a declared monotonic piecewise-rational frame map and
//! projects predictions from a transformed audio view back onto the canonical
//! source timeline. It does not transform audio bytes or establish perceptual
//! equivalence; those remain owned by the task-scoped qualification process.

use serde::{Deserialize, Serialize};

use crate::temporal_grounding::{
    score_temporal_grounding, validate_temporal_ground_truth,
    validate_temporal_grounding_prediction, AudioArtifactClockV1, FrameIntervalV1,
    GroundingPredictionStateV1, TemporalGroundTruthV1, TemporalGroundingPlanV1,
    TemporalGroundingPredictionV1, TemporalGroundingScoreV1,
};

const TRANSFORM_MAP_SCHEMA_V1: &str = "mel-temporal-view-map-v1";
const TRANSFORM_CASE_SCHEMA_V1: &str = "mel-temporal-equivariance-case-v1";
const TRANSFORM_EVALUATOR_V1: &str = "mel-temporal-equivariance-evaluator-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformRelationV1 {
    NuisanceInvariant,
    TimeWarpEquivariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalTransformSegmentV1 {
    /// Half-open interval in the transformed view's frame coordinate.
    pub view_start_frame: u64,
    pub view_end_frame: u64,
    /// Corresponding half-open interval in the canonical source coordinate.
    pub canonical_start_frame: u64,
    pub canonical_end_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalViewMapV1 {
    pub schema: String,
    pub transform_id: String,
    pub transform_family: String,
    pub task_scope: String,
    pub relation: TransformRelationV1,
    pub source_artifact: AudioArtifactClockV1,
    pub view_artifact: AudioArtifactClockV1,
    /// Opaque reference to the upstream task-specific transform qualification.
    pub upstream_qualification_receipt_sha256: String,
    /// Piecewise linear view-frame → canonical-frame mapping.
    pub segments: Vec<TemporalTransformSegmentV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalTransformCaseV1 {
    pub schema: String,
    pub truth: TemporalGroundTruthV1,
    pub plan: TemporalGroundingPlanV1,
    pub view_map: TemporalViewMapV1,
    pub baseline_prediction: TemporalGroundingPredictionV1,
    pub transformed_prediction: TemporalGroundingPredictionV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TemporalTransformScoreSummaryV1 {
    pub predicted_occurrences: usize,
    pub ground_truth_occurrences: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub f1: Option<f64>,
    pub exact_occurrence_count: Option<bool>,
    pub count_gated_f1: Option<f64>,
    pub matched_interval_iou_mean: Option<f64>,
    pub start_boundary_mae_frames: Option<f64>,
    pub end_boundary_mae_frames: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TemporalTransformReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub transform_id: String,
    pub transform_family: String,
    pub task_scope: String,
    pub relation: TransformRelationV1,
    pub source_audio_sha256: String,
    pub view_audio_sha256: String,
    pub query_id: String,
    pub baseline: TemporalTransformScoreSummaryV1,
    pub transformed_mapped_to_canonical: TemporalTransformScoreSummaryV1,
    pub mapped_prediction: TemporalGroundingPredictionV1,
    pub predicted_count_delta: Option<i64>,
    pub exact_count_preserved: Option<bool>,
    pub f1_delta_after_canonical_mapping: Option<f64>,
    pub count_gated_f1_delta_after_canonical_mapping: Option<f64>,
    /// This module only validates the opaque digest's syntax and scope binding.
    pub upstream_qualification_receipt_authenticated: bool,
    /// No task-equivalence verdict may be promoted without the #13260 verifier.
    pub qualified_equivariance_claim_permitted: bool,
    /// The transform bytes are not available to this coordinate-only evaluator.
    pub transform_bytes_recomputed: bool,
}

pub fn audit_temporal_transform_case_json(input: &str) -> Result<TemporalTransformReportV1, String> {
    let case: TemporalTransformCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid temporal transform case JSON: {error}"))?;
    audit_temporal_transform_case(&case)
}

pub fn audit_temporal_transform_case(
    case: &TemporalTransformCaseV1,
) -> Result<TemporalTransformReportV1, String> {
    if case.schema != TRANSFORM_CASE_SCHEMA_V1 {
        return Err(format!("unsupported temporal transform case schema: {}", case.schema));
    }
    validate_map(&case.view_map)?;

    let truth_issues = validate_temporal_ground_truth(&case.truth);
    if !truth_issues.is_empty() {
        return Err(format!("invalid canonical ground truth: {truth_issues:?}"));
    }
    if case.truth.artifact != case.view_map.source_artifact {
        return Err("canonical ground-truth artifact differs from view-map source artifact".into());
    }

    for (label, prediction, expected_artifact) in [
        (
            "baseline_prediction",
            &case.baseline_prediction,
            &case.view_map.source_artifact,
        ),
        (
            "transformed_prediction",
            &case.transformed_prediction,
            &case.view_map.view_artifact,
        ),
    ] {
        let issues = validate_temporal_grounding_prediction(prediction);
        if !issues.is_empty() {
            return Err(format!("invalid {label}: {issues:?}"));
        }
        if prediction.artifact != *expected_artifact {
            return Err(format!("{label} is not bound to its exact source/view artifact"));
        }
        if prediction.query_id != case.truth.query_id {
            return Err(format!("{label} query ID differs from canonical ground truth"));
        }
    }

    let baseline_score = score_temporal_grounding(
        &case.truth,
        &case.baseline_prediction,
        &case.plan,
    )
    .map_err(|issues| format!("baseline prediction rejected by scorer: {issues:?}"))?;

    let mapped_prediction = map_prediction_to_canonical(
        &case.transformed_prediction,
        &case.view_map,
    )?;
    let transformed_score = score_temporal_grounding(
        &case.truth,
        &mapped_prediction,
        &case.plan,
    )
    .map_err(|issues| format!("mapped transformed prediction rejected by scorer: {issues:?}"))?;

    let baseline = summarize_score(&baseline_score);
    let transformed_mapped = summarize_score(&transformed_score);
    let predicted_count_delta = signed_count_delta(
        transformed_score.predicted_occurrences,
        baseline_score.predicted_occurrences,
    );
    let exact_count_preserved = Some(
        transformed_score.predicted_occurrences == baseline_score.predicted_occurrences,
    );
    let f1_delta = paired_delta(transformed_score.f1, baseline_score.f1);
    let count_gated_f1_delta = paired_delta(
        count_gated_f1(&transformed_score),
        count_gated_f1(&baseline_score),
    );

    Ok(TemporalTransformReportV1 {
        schema: "mel-temporal-equivariance-report-v1",
        evaluator_version: TRANSFORM_EVALUATOR_V1,
        transform_id: case.view_map.transform_id.clone(),
        transform_family: case.view_map.transform_family.clone(),
        task_scope: case.view_map.task_scope.clone(),
        relation: case.view_map.relation,
        source_audio_sha256: case.view_map.source_artifact.audio_sha256.clone(),
        view_audio_sha256: case.view_map.view_artifact.audio_sha256.clone(),
        query_id: case.truth.query_id.clone(),
        baseline,
        transformed_mapped_to_canonical: transformed_mapped,
        mapped_prediction,
        predicted_count_delta,
        exact_count_preserved,
        f1_delta_after_canonical_mapping: f1_delta,
        count_gated_f1_delta_after_canonical_mapping: count_gated_f1_delta,
        upstream_qualification_receipt_authenticated: false,
        qualified_equivariance_claim_permitted: false,
        transform_bytes_recomputed: false,
    })
}

/// Map all predicted intervals from the view's frame domain back to canonical
/// source frames. The original query and non-occurrence dispositions are kept.
pub fn map_prediction_to_canonical(
    prediction: &TemporalGroundingPredictionV1,
    view_map: &TemporalViewMapV1,
) -> Result<TemporalGroundingPredictionV1, String> {
    validate_map(view_map)?;
    let prediction_issues = validate_temporal_grounding_prediction(prediction);
    if !prediction_issues.is_empty() {
        return Err(format!("invalid transformed-view prediction: {prediction_issues:?}"));
    }
    if prediction.artifact != view_map.view_artifact {
        return Err("prediction artifact does not match transformed view".into());
    }
    let mut mapped = prediction.clone();
    mapped.artifact = view_map.source_artifact.clone();
    mapped.state = match &prediction.state {
        GroundingPredictionStateV1::Occurrences { intervals } => {
            GroundingPredictionStateV1::Occurrences {
                intervals: map_interval_list(intervals, view_map)?,
            }
        }
        GroundingPredictionStateV1::Ambiguous {
            candidate_intervals,
            reason,
        } => GroundingPredictionStateV1::Ambiguous {
            candidate_intervals: map_interval_list(candidate_intervals, view_map)?,
            reason: reason.clone(),
        },
        other => other.clone(),
    };
    Ok(mapped)
}

pub fn validate_map(view_map: &TemporalViewMapV1) -> Result<(), String> {
    if view_map.schema != TRANSFORM_MAP_SCHEMA_V1 {
        return Err(format!("unsupported temporal view map schema: {}", view_map.schema));
    }
    require_text(&view_map.transform_id, "transform_id")?;
    require_text(&view_map.transform_family, "transform_family")?;
    require_text(&view_map.task_scope, "task_scope")?;
    validate_digest(&view_map.source_artifact.audio_sha256, "source_artifact.audio_sha256")?;
    validate_digest(&view_map.view_artifact.audio_sha256, "view_artifact.audio_sha256")?;
    validate_digest(
        &view_map.upstream_qualification_receipt_sha256,
        "upstream_qualification_receipt_sha256",
    )?;
    validate_artifact_clock(&view_map.source_artifact, "source_artifact")?;
    validate_artifact_clock(&view_map.view_artifact, "view_artifact")?;
    if view_map.source_artifact.audio_sha256 == view_map.view_artifact.audio_sha256
        && view_map.source_artifact != view_map.view_artifact
    {
        return Err("identical source/view digests cannot declare different audio clocks".into());
    }
    if view_map.segments.is_empty() {
        return Err("view map must contain at least one segment".into());
    }

    let mut expected_view_start = 0_u64;
    let mut previous_canonical_end = None;
    for (index, segment) in view_map.segments.iter().enumerate() {
        if segment.view_start_frame != expected_view_start {
            return Err(format!(
                "view map segment {index} does not start at the previous view boundary"
            ));
        }
        if segment.view_end_frame <= segment.view_start_frame {
            return Err(format!("view map segment {index} has an empty or reversed view range"));
        }
        if segment.canonical_end_frame <= segment.canonical_start_frame {
            return Err(format!("view map segment {index} has an empty or reversed canonical range"));
        }
        if segment.canonical_end_frame > view_map.source_artifact.total_frames {
            return Err(format!("view map segment {index} exceeds canonical source duration"));
        }
        if let Some(previous_end) = previous_canonical_end
            && segment.canonical_start_frame != previous_end
        {
            return Err(format!(
                "view map segment {index} is not contiguous and monotonic in canonical time"
            ));
        }
        expected_view_start = segment.view_end_frame;
        previous_canonical_end = Some(segment.canonical_end_frame);
    }
    if expected_view_start != view_map.view_artifact.total_frames {
        return Err("view map does not cover the entire transformed view".into());
    }
    Ok(())
}

fn map_interval_list(
    intervals: &[FrameIntervalV1],
    view_map: &TemporalViewMapV1,
) -> Result<Vec<FrameIntervalV1>, String> {
    let mut mapped = Vec::with_capacity(intervals.len());
    for (interval_index, interval) in intervals.iter().enumerate() {
        if interval.start_frame >= interval.end_frame
            || interval.end_frame > view_map.view_artifact.total_frames
        {
            return Err(format!("prediction interval {interval_index} is outside the view clock"));
        }
        let mut pieces: Vec<FrameIntervalV1> = Vec::new();
        for segment in &view_map.segments {
            let view_start = interval.start_frame.max(segment.view_start_frame);
            let view_end = interval.end_frame.min(segment.view_end_frame);
            if view_start >= view_end {
                continue;
            }
            let canonical_start = map_offset_floor(
                view_start - segment.view_start_frame,
                segment.view_end_frame - segment.view_start_frame,
                segment.canonical_end_frame - segment.canonical_start_frame,
            )?
            .checked_add(segment.canonical_start_frame)
            .ok_or_else(|| "canonical mapped start overflow".to_string())?;
            let canonical_end = map_offset_ceil(
                view_end - segment.view_start_frame,
                segment.view_end_frame - segment.view_start_frame,
                segment.canonical_end_frame - segment.canonical_start_frame,
            )?
            .checked_add(segment.canonical_start_frame)
            .ok_or_else(|| "canonical mapped end overflow".to_string())?;
            if canonical_start >= canonical_end {
                return Err(format!("mapped prediction interval {interval_index} collapsed to zero frames"));
            }
            let piece = FrameIntervalV1 {
                start_frame: canonical_start,
                end_frame: canonical_end,
            };
            if let Some(previous) = pieces.last_mut()
                && previous.end_frame >= piece.start_frame
            {
                previous.end_frame = previous.end_frame.max(piece.end_frame);
                continue;
            }
            pieces.push(piece);
        }
        if pieces.is_empty() {
            return Err(format!("prediction interval {interval_index} has no mapped view support"));
        }
        if pieces.first().is_some_and(|piece| piece.start_frame < view_map.segments[0].canonical_start_frame)
            || pieces.last().is_some_and(|piece| piece.end_frame > view_map.source_artifact.total_frames)
        {
            return Err(format!("mapped prediction interval {interval_index} exceeds canonical source"));
        }
        mapped.extend(pieces);
    }
    Ok(mapped)
}

fn map_offset_floor(offset: u64, view_len: u64, canonical_len: u64) -> Result<u64, String> {
    let numerator = (offset as u128)
        .checked_mul(canonical_len as u128)
        .ok_or_else(|| "frame-map multiplication overflow".to_string())?;
    let mapped = numerator / view_len as u128;
    u64::try_from(mapped).map_err(|_| "mapped frame does not fit in u64".into())
}

fn map_offset_ceil(offset: u64, view_len: u64, canonical_len: u64) -> Result<u64, String> {
    let denominator = view_len as u128;
    let numerator = (offset as u128)
        .checked_mul(canonical_len as u128)
        .ok_or_else(|| "frame-map multiplication overflow".to_string())?;
    let adjusted = numerator
        .checked_add(denominator - 1)
        .ok_or_else(|| "frame-map ceiling overflow".to_string())?;
    let mapped = adjusted / denominator;
    u64::try_from(mapped).map_err(|_| "mapped frame does not fit in u64".into())
}

fn validate_artifact_clock(artifact: &AudioArtifactClockV1, label: &str) -> Result<(), String> {
    if artifact.sample_rate_hz == 0 {
        return Err(format!("{label}.sample_rate_hz must be positive"));
    }
    if artifact.channels == 0 {
        return Err(format!("{label}.channels must be positive"));
    }
    if artifact.total_frames == 0 {
        return Err(format!("{label}.total_frames must be positive"));
    }
    Ok(())
}

fn summarize_score(score: &TemporalGroundingScoreV1) -> TemporalTransformScoreSummaryV1 {
    let start_errors: Vec<_> = score
        .matches
        .iter()
        .map(|matched| matched.start_error_frames.unsigned_abs() as f64)
        .collect();
    let end_errors: Vec<_> = score
        .matches
        .iter()
        .map(|matched| matched.end_error_frames.unsigned_abs() as f64)
        .collect();
    TemporalTransformScoreSummaryV1 {
        predicted_occurrences: score.predicted_occurrences,
        ground_truth_occurrences: score.ground_truth_occurrences,
        true_positives: score.true_positives,
        false_positives: score.false_positives,
        false_negatives: score.false_negatives,
        f1: score.f1,
        exact_occurrence_count: score.exact_occurrence_count,
        count_gated_f1: count_gated_f1(score),
        matched_interval_iou_mean: score.matched_interval_iou_mean,
        start_boundary_mae_frames: mean(&start_errors),
        end_boundary_mae_frames: mean(&end_errors),
    }
}

fn count_gated_f1(score: &TemporalGroundingScoreV1) -> Option<f64> {
    if score.ground_truth_occurrences <= 1
        || score.f1.is_none()
    {
        None
    } else if score.exact_occurrence_count == Some(true) {
        score.f1
    } else {
        Some(0.0)
    }
}

fn signed_count_delta(after: usize, before: usize) -> Option<i64> {
    if after >= before {
        i64::try_from(after - before).ok()
    } else {
        i64::try_from(before - after).ok().map(|delta| -delta)
    }
}

fn paired_delta(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    left.zip(right).map(|(left, right)| left - right)
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_grounding::{
        GroundTruthStateV1, GroundingPredictionStateV1, TemporalGroundingPredictionV1,
        TemporalGroundTruthV1, TEMPORAL_GROUNDING_EVALUATOR_V1,
        TEMPORAL_GROUNDING_SCHEMA_V1,
    };

    const SOURCE_DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const VIEW_DIGEST: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const RECEIPT_DIGEST: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const ANNOTATION_DIGEST: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn artifact(digest: &str, total_frames: u64) -> AudioArtifactClockV1 {
        AudioArtifactClockV1 {
            audio_sha256: digest.into(),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames,
        }
    }

    fn map(
        source_frames: u64,
        view_frames: u64,
        segments: Vec<TemporalTransformSegmentV1>,
    ) -> TemporalViewMapV1 {
        TemporalViewMapV1 {
            schema: TRANSFORM_MAP_SCHEMA_V1.into(),
            transform_id: "synthetic-transform-001".into(),
            transform_family: "synthetic-piecewise-time-warp".into(),
            task_scope: "instrument-entrance-occurrence-and-interval".into(),
            relation: TransformRelationV1::TimeWarpEquivariant,
            source_artifact: artifact(SOURCE_DIGEST, source_frames),
            view_artifact: artifact(VIEW_DIGEST, view_frames),
            upstream_qualification_receipt_sha256: RECEIPT_DIGEST.into(),
            segments,
        }
    }

    fn truth(intervals: Vec<FrameIntervalV1>) -> TemporalGroundTruthV1 {
        TemporalGroundTruthV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: "query-piano-entry".into(),
            query_text: "Every piano entrance".into(),
            artifact: artifact(SOURCE_DIGEST, 1_000),
            annotation_policy_version: "synthetic-annotation-v1".into(),
            annotation_set_sha256: ANNOTATION_DIGEST.into(),
            annotator_count: 2,
            state: GroundTruthStateV1::Present,
            intervals,
            status_reason: None,
        }
    }

    fn prediction(
        artifact: AudioArtifactClockV1,
        intervals: Vec<FrameIntervalV1>,
    ) -> TemporalGroundingPredictionV1 {
        TemporalGroundingPredictionV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: "query-piano-entry".into(),
            artifact,
            state: GroundingPredictionStateV1::Occurrences { intervals },
        }
    }

    fn plan() -> TemporalGroundingPlanV1 {
        TemporalGroundingPlanV1 {
            evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
            minimum_interval_iou: 0.5,
        }
    }

    fn case(
        truth_intervals: Vec<FrameIntervalV1>,
        baseline: Vec<FrameIntervalV1>,
        view_prediction: Vec<FrameIntervalV1>,
        view_map: TemporalViewMapV1,
    ) -> TemporalTransformCaseV1 {
        TemporalTransformCaseV1 {
            schema: TRANSFORM_CASE_SCHEMA_V1.into(),
            truth: truth(truth_intervals),
            plan: plan(),
            baseline_prediction: prediction(view_map.source_artifact.clone(), baseline),
            transformed_prediction: prediction(view_map.view_artifact.clone(), view_prediction),
            view_map,
        }
    }

    #[test]
    fn identity_map_preserves_intervals_and_score_exactly() {
        let identity = map(
            1_000,
            1_000,
            vec![TemporalTransformSegmentV1 {
                view_start_frame: 0,
                view_end_frame: 1_000,
                canonical_start_frame: 0,
                canonical_end_frame: 1_000,
            }],
        );
        let input = case(
            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            identity,
        );
        let report = audit_temporal_transform_case(&input).unwrap();
        assert_eq!(report.mapped_prediction.state, input.baseline_prediction.state);
        assert_eq!(report.f1_delta_after_canonical_mapping, Some(0.0));
        assert_eq!(report.exact_count_preserved, Some(true));
    }

    #[test]
    fn constant_two_x_view_stretch_maps_predictions_back_to_canonical_time() {
        let stretched = map(
            1_000,
            2_000,
            vec![TemporalTransformSegmentV1 {
                view_start_frame: 0,
                view_end_frame: 2_000,
                canonical_start_frame: 0,
                canonical_end_frame: 1_000,
            }],
        );
        let input = case(
            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            vec![FrameIntervalV1 { start_frame: 200, end_frame: 400 }],
            stretched,
        );
        let report = audit_temporal_transform_case(&input).unwrap();
        assert_eq!(
            report.mapped_prediction.state,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }]
            }
        );
        assert_eq!(report.transformed_mapped_to_canonical.f1, Some(1.0));
        assert_eq!(report.exact_count_preserved, Some(true));
    }

    #[test]
    fn piecewise_time_warp_maps_an_interval_crossing_a_segment_boundary() {
        let warped = map(
            1_000,
            2_000,
            vec![
                TemporalTransformSegmentV1 {
                    view_start_frame: 0,
                    view_end_frame: 1_000,
                    canonical_start_frame: 0,
                    canonical_end_frame: 500,
                },
                TemporalTransformSegmentV1 {
                    view_start_frame: 1_000,
                    view_end_frame: 2_000,
                    canonical_start_frame: 500,
                    canonical_end_frame: 1_000,
                },
            ],
        );
        let mapped = map_prediction_to_canonical(
            &prediction(
                warped.view_artifact.clone(),
                vec![FrameIntervalV1 { start_frame: 900, end_frame: 1_100 }],
            ),
            &warped,
        )
        .unwrap();
        assert_eq!(
            mapped.state,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![FrameIntervalV1 { start_frame: 450, end_frame: 550 }]
            }
        );
    }

    #[test]
    fn crop_map_preserves_nonzero_canonical_source_offset() {
        let crop = map(
            1_000,
            200,
            vec![TemporalTransformSegmentV1 {
                view_start_frame: 0,
                view_end_frame: 200,
                canonical_start_frame: 400,
                canonical_end_frame: 600,
            }],
        );
        let mapped = map_prediction_to_canonical(
            &prediction(
                crop.view_artifact.clone(),
                vec![FrameIntervalV1 { start_frame: 10, end_frame: 30 }],
            ),
            &crop,
        )
        .unwrap();
        assert_eq!(
            mapped.state,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![FrameIntervalV1 { start_frame: 410, end_frame: 430 }]
            }
        );
    }

    #[test]
    fn fractional_mapping_rounds_outward_not_inward() {
        let fractional = map(
            2,
            3,
            vec![TemporalTransformSegmentV1 {
                view_start_frame: 0,
                view_end_frame: 3,
                canonical_start_frame: 0,
                canonical_end_frame: 2,
            }],
        );
        let mapped = map_prediction_to_canonical(
            &prediction(
                fractional.view_artifact.clone(),
                vec![FrameIntervalV1 { start_frame: 1, end_frame: 2 }],
            ),
            &fractional,
        )
        .unwrap();
        assert_eq!(
            mapped.state,
            GroundingPredictionStateV1::Occurrences {
                intervals: vec![FrameIntervalV1 { start_frame: 0, end_frame: 2 }]
            }
        );
    }

    #[test]
    fn same_audio_digest_cannot_claim_a_different_frame_clock() {
        let mut invalid = map(
            1_000,
            1_000,
            vec![TemporalTransformSegmentV1 {
                view_start_frame: 0,
                view_end_frame: 1_000,
                canonical_start_frame: 0,
                canonical_end_frame: 1_000,
            }],
        );
        invalid.view_artifact.audio_sha256 = SOURCE_DIGEST.into();
        invalid.view_artifact.total_frames = 999;
        assert!(validate_map(&invalid).is_err());
    }

    #[test]
    fn empty_or_out_of_view_prediction_intervals_are_rejected() {
        let identity = map(
            1_000,
            1_000,
            vec![TemporalTransformSegmentV1 {
                view_start_frame: 0,
                view_end_frame: 1_000,
                canonical_start_frame: 0,
                canonical_end_frame: 1_000,
            }],
        );
        let empty = prediction(
            identity.view_artifact.clone(),
            vec![FrameIntervalV1 {
                start_frame: 100,
                end_frame: 100,
            }],
        );
        assert!(map_prediction_to_canonical(&empty, &identity).is_err());
        let outside = prediction(
            identity.view_artifact.clone(),
            vec![FrameIntervalV1 {
                start_frame: 999,
                end_frame: 1_001,
            }],
        );
        assert!(map_prediction_to_canonical(&outside, &identity).is_err());
    }

    #[test]
    fn noncontiguous_view_segments_are_rejected() {
        let invalid = map(
            1_000,
            2_000,
            vec![
                TemporalTransformSegmentV1 {
                    view_start_frame: 0,
                    view_end_frame: 999,
                    canonical_start_frame: 0,
                    canonical_end_frame: 500,
                },
                TemporalTransformSegmentV1 {
                    view_start_frame: 1_000,
                    view_end_frame: 2_000,
                    canonical_start_frame: 500,
                    canonical_end_frame: 1_000,
                },
            ],
        );
        assert!(validate_map(&invalid).is_err());
    }

    #[test]
    fn nonmonotonic_canonical_segments_are_rejected() {
        let invalid = map(
            1_000,
            2_000,
            vec![
                TemporalTransformSegmentV1 {
                    view_start_frame: 0,
                    view_end_frame: 1_000,
                    canonical_start_frame: 0,
                    canonical_end_frame: 700,
                },
                TemporalTransformSegmentV1 {
                    view_start_frame: 1_000,
                    view_end_frame: 2_000,
                    canonical_start_frame: 600,
                    canonical_end_frame: 1_000,
                },
            ],
        );
        assert!(validate_map(&invalid).is_err());
    }

    #[test]
    fn wrong_view_digest_or_query_is_rejected() {
        let identity = map(
            1_000,
            1_000,
            vec![TemporalTransformSegmentV1 {
                view_start_frame: 0,
                view_end_frame: 1_000,
                canonical_start_frame: 0,
                canonical_end_frame: 1_000,
            }],
        );
        let mut input = case(
            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            identity,
        );
        input.transformed_prediction.artifact.audio_sha256 = SOURCE_DIGEST.into();
        assert!(audit_temporal_transform_case(&input).is_err());
    }

    #[test]
    fn transformed_prediction_that_loses_an_occurrence_fails_count_gate() {
        let identity = map(
            1_000,
            1_000,
            vec![TemporalTransformSegmentV1 {
                view_start_frame: 0,
                view_end_frame: 1_000,
                canonical_start_frame: 0,
                canonical_end_frame: 1_000,
            }],
        );
        let input = case(
            vec![
                FrameIntervalV1 { start_frame: 100, end_frame: 200 },
                FrameIntervalV1 { start_frame: 500, end_frame: 600 },
            ],
            vec![
                FrameIntervalV1 { start_frame: 100, end_frame: 200 },
                FrameIntervalV1 { start_frame: 500, end_frame: 600 },
            ],
            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
            identity,
        );
        let report = audit_temporal_transform_case(&input).unwrap();
        assert_eq!(report.exact_count_preserved, Some(false));
        assert_eq!(
            report.count_gated_f1_delta_after_canonical_mapping,
            Some(-1.0)
        );
    }
}
