//! Multi-annotator temporal-grounding adapter.
//!
//! Each human annotation remains an independent reference. The adapter runs
//! the production scorer once per annotator and reports the resulting score
//! distribution. It does not construct a synthetic consensus interval set,
//! adjudicate which annotator is right, or estimate population-level
//! confidence from a small annotation panel.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::temporal_grounding::{
    AudioArtifactClockV1, FrameIntervalV1, GroundTruthStateV1, TEMPORAL_GROUNDING_SCHEMA_V1,
    TemporalGroundTruthV1, TemporalGroundingOutcomeV1, TemporalGroundingPlanV1,
    TemporalGroundingPredictionV1, TemporalGroundingScoreV1, score_temporal_grounding,
    validate_temporal_grounding_prediction,
};

const CASE_SCHEMA_V1: &str = "mel-multi-annotator-temporal-grounding-v1";
const EVALUATOR_VERSION_V1: &str = "mel-multi-annotator-grounding-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndependentTemporalAnnotationV1 {
    /// Pseudonymous annotator identity commitment; raw personal identifiers
    /// must not be placed in this evaluation artifact.
    pub annotator_id_sha256: String,
    /// Digest of the canonical annotation record, including annotator scope.
    pub annotation_sha256: String,
    pub state: GroundTruthStateV1,
    pub intervals: Vec<FrameIntervalV1>,
    pub status_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiAnnotatorTemporalCaseV1 {
    pub schema: String,
    pub query_id: String,
    pub query_text: String,
    pub artifact: AudioArtifactClockV1,
    pub annotation_policy_version: String,
    pub annotation_manifest_sha256: String,
    /// Minimum number of independent annotations required before evaluation.
    pub minimum_annotator_count: u16,
    pub plan: TemporalGroundingPlanV1,
    pub prediction: TemporalGroundingPredictionV1,
    pub annotations: Vec<IndependentTemporalAnnotationV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnnotatorGroundingResultV1 {
    pub annotator_id_sha256: String,
    pub annotation_sha256: String,
    pub state: GroundTruthStateV1,
    pub score: TemporalGroundingScoreV1,
    pub fully_correct_for_this_annotation: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiAnnotatorGroundingSummaryV1 {
    pub panel_size: usize,
    pub minimum_annotator_count: u16,
    pub scored_annotators: usize,
    pub present_annotators: usize,
    pub absent_annotators: usize,
    pub ambiguous_exclusions: usize,
    pub not_estimable_exclusions: usize,
    pub fully_correct_annotators: usize,
    pub fully_correct_rate: Option<f64>,
    pub unanimous_correct_across_scored_annotators: Option<bool>,
    pub present_mean_f1: Option<f64>,
    pub present_median_f1: Option<f64>,
    pub present_min_f1: Option<f64>,
    pub present_max_f1: Option<f64>,
    pub present_exact_count_accuracy: Option<f64>,
    pub present_mean_count_gated_f1: Option<f64>,
    pub present_count_error_min: Option<usize>,
    pub present_count_error_median: Option<f64>,
    pub present_count_error_max: Option<usize>,
    pub annotated_occurrence_count_min: Option<usize>,
    pub annotated_occurrence_count_median: Option<f64>,
    pub annotated_occurrence_count_max: Option<usize>,
    pub absent_correct_claims: usize,
    pub absent_claim_accuracy: Option<f64>,
    pub start_boundary_mae_frames: Option<f64>,
    pub start_boundary_p95_abs_error_frames: Option<f64>,
    pub end_boundary_mae_frames: Option<f64>,
    pub end_boundary_p95_abs_error_frames: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiAnnotatorGroundingReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub query_id: String,
    pub audio_sha256: String,
    pub annotation_policy_version: String,
    pub annotation_manifest_sha256: String,
    pub plan: TemporalGroundingPlanV1,
    pub prediction: TemporalGroundingPredictionV1,
    pub summary: MultiAnnotatorGroundingSummaryV1,
    /// Scores retain every per-annotator result; exclusions are not discarded.
    pub annotator_results: Vec<AnnotatorGroundingResultV1>,
    /// This adapter preserves disagreement; it does not calculate formal IAA.
    pub pairwise_inter_annotator_agreement_computed: bool,
    /// No single merged or majority-voted interval reference is constructed.
    pub synthetic_consensus_ground_truth_created: bool,
    /// Small panels are descriptive. This is not population-level uncertainty.
    pub population_confidence_interval_computed: bool,
}

pub fn audit_multi_annotator_case_json(
    input: &str,
) -> Result<MultiAnnotatorGroundingReportV1, String> {
    let case: MultiAnnotatorTemporalCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid multi-annotator case JSON: {error}"))?;
    audit_multi_annotator_case(&case)
}

pub fn audit_multi_annotator_case(
    case: &MultiAnnotatorTemporalCaseV1,
) -> Result<MultiAnnotatorGroundingReportV1, String> {
    if case.schema != CASE_SCHEMA_V1 {
        return Err(format!(
            "unsupported multi-annotator schema: {}",
            case.schema
        ));
    }
    require_text(&case.query_id, "query_id")?;
    require_text(&case.query_text, "query_text")?;
    require_text(&case.annotation_policy_version, "annotation_policy_version")?;
    validate_digest(
        &case.annotation_manifest_sha256,
        "annotation_manifest_sha256",
    )?;
    validate_artifact(&case.artifact)?;
    if case.minimum_annotator_count < 2 {
        return Err("minimum_annotator_count must be at least two independent annotators".into());
    }
    if case.annotations.len() < case.minimum_annotator_count as usize {
        return Err(format!(
            "annotation panel has {} entries, below required minimum {}",
            case.annotations.len(),
            case.minimum_annotator_count
        ));
    }
    if case.annotations.len() > u16::MAX as usize {
        return Err("annotation panel exceeds the supported annotator count".into());
    }

    let prediction_issues = validate_temporal_grounding_prediction(&case.prediction);
    if !prediction_issues.is_empty() {
        return Err(format!("invalid model prediction: {prediction_issues:?}"));
    }
    if case.prediction.query_id != case.query_id {
        return Err("prediction query ID differs from the multi-annotator case".into());
    }
    if case.prediction.artifact != case.artifact {
        return Err("prediction artifact differs from the multi-annotator case artifact".into());
    }

    let mut annotator_ids = BTreeSet::new();
    let mut annotation_digests = BTreeSet::new();
    let mut results = Vec::with_capacity(case.annotations.len());

    for (index, annotation) in case.annotations.iter().enumerate() {
        validate_digest(
            &annotation.annotator_id_sha256,
            &format!("annotations[{index}].annotator_id_sha256"),
        )?;
        validate_digest(
            &annotation.annotation_sha256,
            &format!("annotations[{index}].annotation_sha256"),
        )?;
        if !annotator_ids.insert(annotation.annotator_id_sha256.as_str()) {
            return Err(format!(
                "duplicate annotator identity at panel index {index}"
            ));
        }
        if !annotation_digests.insert(annotation.annotation_sha256.as_str()) {
            return Err(format!(
                "duplicate annotation receipt digest at panel index {index}"
            ));
        }

        let truth = TemporalGroundTruthV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: case.query_id.clone(),
            query_text: case.query_text.clone(),
            artifact: case.artifact.clone(),
            annotation_policy_version: case.annotation_policy_version.clone(),
            annotation_set_sha256: annotation.annotation_sha256.clone(),
            annotator_count: 1,
            state: annotation.state,
            intervals: annotation.intervals.clone(),
            status_reason: annotation.status_reason.clone(),
        };

        let score =
            score_temporal_grounding(&truth, &case.prediction, &case.plan).map_err(|issues| {
                format!("annotator panel index {index} rejected by production scorer: {issues:?}")
            })?;
        let fully_correct_for_this_annotation = fully_correct(&score);

        results.push(AnnotatorGroundingResultV1 {
            annotator_id_sha256: annotation.annotator_id_sha256.clone(),
            annotation_sha256: annotation.annotation_sha256.clone(),
            state: annotation.state,
            score,
            fully_correct_for_this_annotation,
        });
    }

    let summary = summarize_results(&results, case.minimum_annotator_count);

    Ok(MultiAnnotatorGroundingReportV1 {
        schema: "mel-multi-annotator-grounding-report-v1",
        evaluator_version: EVALUATOR_VERSION_V1,
        query_id: case.query_id.clone(),
        audio_sha256: case.artifact.audio_sha256.clone(),
        annotation_policy_version: case.annotation_policy_version.clone(),
        annotation_manifest_sha256: case.annotation_manifest_sha256.clone(),
        plan: case.plan.clone(),
        prediction: case.prediction.clone(),
        summary,
        annotator_results: results,
        pairwise_inter_annotator_agreement_computed: false,
        synthetic_consensus_ground_truth_created: false,
        population_confidence_interval_computed: false,
    })
}

fn summarize_results(
    results: &[AnnotatorGroundingResultV1],
    minimum_annotator_count: u16,
) -> MultiAnnotatorGroundingSummaryV1 {
    let scored: Vec<_> = results.iter().filter(|item| item.score.scored).collect();
    let present: Vec<_> = scored
        .iter()
        .copied()
        .filter(|item| item.state == GroundTruthStateV1::Present)
        .collect();
    let absent: Vec<_> = scored
        .iter()
        .copied()
        .filter(|item| item.state == GroundTruthStateV1::Absent)
        .collect();
    let ambiguous_exclusions = results
        .iter()
        .filter(|item| item.state == GroundTruthStateV1::Ambiguous)
        .count();
    let not_estimable_exclusions = results
        .iter()
        .filter(|item| item.state == GroundTruthStateV1::NotEstimable)
        .count();

    let positive_f1: Vec<_> = present.iter().filter_map(|item| item.score.f1).collect();
    let count_errors: Vec<_> = present
        .iter()
        .filter_map(|item| item.score.count_absolute_error)
        .collect();
    let exact_count = present
        .iter()
        .filter(|item| item.score.exact_occurrence_count == Some(true))
        .count();
    let count_gated_f1: Vec<_> = present
        .iter()
        .map(|item| {
            if item.score.exact_occurrence_count == Some(true) {
                item.score.f1.unwrap_or(0.0)
            } else {
                0.0
            }
        })
        .collect();

    let start_errors: Vec<_> = present
        .iter()
        .flat_map(|item| {
            item.score
                .matches
                .iter()
                .map(|matched| matched.start_error_frames.unsigned_abs() as f64)
        })
        .collect();
    let end_errors: Vec<_> = present
        .iter()
        .flat_map(|item| {
            item.score
                .matches
                .iter()
                .map(|matched| matched.end_error_frames.unsigned_abs() as f64)
        })
        .collect();

    let fully_correct_annotators = scored
        .iter()
        .filter(|item| item.fully_correct_for_this_annotation == Some(true))
        .count();
    let absent_correct_claims = absent
        .iter()
        .filter(|item| item.score.correct_absence_claim == Some(true))
        .count();

    let truth_counts: Vec<_> = scored
        .iter()
        .map(|item| item.score.ground_truth_occurrences)
        .collect();

    MultiAnnotatorGroundingSummaryV1 {
        panel_size: results.len(),
        minimum_annotator_count,
        scored_annotators: scored.len(),
        present_annotators: present.len(),
        absent_annotators: absent.len(),
        ambiguous_exclusions,
        not_estimable_exclusions,
        fully_correct_annotators,
        fully_correct_rate: rate(fully_correct_annotators, scored.len()),
        unanimous_correct_across_scored_annotators: (!scored.is_empty())
            .then_some(fully_correct_annotators == scored.len()),
        present_mean_f1: mean(&positive_f1),
        present_median_f1: median(&positive_f1),
        present_min_f1: positive_f1.iter().copied().reduce(f64::min),
        present_max_f1: positive_f1.iter().copied().reduce(f64::max),
        present_exact_count_accuracy: rate(exact_count, present.len()),
        present_mean_count_gated_f1: mean(&count_gated_f1),
        present_count_error_min: count_errors.iter().copied().min(),
        present_count_error_median: median_u64(&count_errors),
        present_count_error_max: count_errors.iter().copied().max(),
        annotated_occurrence_count_min: truth_counts.iter().copied().min(),
        annotated_occurrence_count_median: median_u64(&truth_counts),
        annotated_occurrence_count_max: truth_counts.iter().copied().max(),
        absent_correct_claims,
        absent_claim_accuracy: rate(absent_correct_claims, absent.len()),
        start_boundary_mae_frames: mean(&start_errors),
        start_boundary_p95_abs_error_frames: percentile(&start_errors, 0.95),
        end_boundary_mae_frames: mean(&end_errors),
        end_boundary_p95_abs_error_frames: percentile(&end_errors, 0.95),
    }
}

fn fully_correct(score: &TemporalGroundingScoreV1) -> Option<bool> {
    match score.outcome {
        TemporalGroundingOutcomeV1::ScoredPresentQuery => Some(
            score.f1 == Some(1.0)
                && score.true_positives == score.ground_truth_occurrences
                && score.false_positives == 0
                && score.false_negatives == 0
                && score.exact_occurrence_count == Some(true),
        ),
        TemporalGroundingOutcomeV1::ScoredAbsentQuery => score.correct_absence_claim,
        TemporalGroundingOutcomeV1::ExcludedAmbiguousGroundTruth
        | TemporalGroundingOutcomeV1::ExcludedNotEstimableGroundTruth => None,
    }
}

fn validate_artifact(artifact: &AudioArtifactClockV1) -> Result<(), String> {
    validate_digest(&artifact.audio_sha256, "artifact.audio_sha256")?;
    if artifact.sample_rate_hz == 0 || artifact.channels == 0 || artifact.total_frames == 0 {
        return Err("audio artifact sample rate, channels and frame count must be positive".into());
    }
    Ok(())
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

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

fn median_u64(values: &[usize]) -> Option<f64> {
    median(&values.iter().map(|value| *value as f64).collect::<Vec<_>>())
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_grounding::{
        GroundingPredictionStateV1, TEMPORAL_GROUNDING_EVALUATOR_V1, TEMPORAL_GROUNDING_SCHEMA_V1,
        TemporalGroundingPredictionV1,
    };

    const AUDIO: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MANIFEST: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn artifact() -> AudioArtifactClockV1 {
        AudioArtifactClockV1 {
            audio_sha256: AUDIO.into(),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 48_000 * 60,
        }
    }

    fn plan() -> TemporalGroundingPlanV1 {
        TemporalGroundingPlanV1 {
            evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
            minimum_interval_iou: 0.5,
        }
    }

    fn prediction(intervals: Vec<FrameIntervalV1>) -> TemporalGroundingPredictionV1 {
        TemporalGroundingPredictionV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: "query-piano".into(),
            artifact: artifact(),
            state: if intervals.is_empty() {
                GroundingPredictionStateV1::QueryAbsent
            } else {
                GroundingPredictionStateV1::Occurrences { intervals }
            },
        }
    }

    fn annotation(
        id_char: char,
        receipt_char: char,
        state: GroundTruthStateV1,
        intervals: Vec<FrameIntervalV1>,
        reason: Option<&str>,
    ) -> IndependentTemporalAnnotationV1 {
        IndependentTemporalAnnotationV1 {
            annotator_id_sha256: digest(id_char),
            annotation_sha256: digest(receipt_char),
            state,
            intervals,
            status_reason: reason.map(str::to_owned),
        }
    }

    fn case(
        annotations: Vec<IndependentTemporalAnnotationV1>,
        model_prediction: TemporalGroundingPredictionV1,
    ) -> MultiAnnotatorTemporalCaseV1 {
        MultiAnnotatorTemporalCaseV1 {
            schema: CASE_SCHEMA_V1.into(),
            query_id: "query-piano".into(),
            query_text: "Every piano entrance".into(),
            artifact: artifact(),
            annotation_policy_version: "music-temporal-annotation-v1".into(),
            annotation_manifest_sha256: MANIFEST.into(),
            minimum_annotator_count: 2,
            plan: plan(),
            prediction: model_prediction,
            annotations,
        }
    }

    #[test]
    fn identical_annotators_preserve_independent_records_and_zero_spread() {
        let intervals = vec![
            FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            },
            FrameIntervalV1 {
                start_frame: 500,
                end_frame: 600,
            },
        ];
        let input = case(
            vec![
                annotation(
                    '1',
                    'a',
                    GroundTruthStateV1::Present,
                    intervals.clone(),
                    None,
                ),
                annotation(
                    '2',
                    'b',
                    GroundTruthStateV1::Present,
                    intervals.clone(),
                    None,
                ),
            ],
            prediction(intervals),
        );
        let report = audit_multi_annotator_case(&input).unwrap();
        assert_eq!(report.summary.panel_size, 2);
        assert_eq!(report.summary.scored_annotators, 2);
        assert_eq!(report.summary.present_mean_f1, Some(1.0));
        assert_eq!(report.summary.present_min_f1, Some(1.0));
        assert_eq!(report.summary.present_max_f1, Some(1.0));
        assert_eq!(report.summary.present_exact_count_accuracy, Some(1.0));
        assert_eq!(
            report.summary.unanimous_correct_across_scored_annotators,
            Some(true)
        );
        assert_eq!(report.annotator_results.len(), 2);
        assert_eq!(report.annotator_results[0].score.f1, Some(1.0));
        assert_eq!(report.annotator_results[1].score.f1, Some(1.0));
        assert!(!report.synthetic_consensus_ground_truth_created);
        assert!(!report.pairwise_inter_annotator_agreement_computed);
    }

    #[test]
    fn boundary_disagreement_remains_visible_in_per_annotator_scores() {
        let input = case(
            vec![
                annotation(
                    '1',
                    'a',
                    GroundTruthStateV1::Present,
                    vec![FrameIntervalV1 {
                        start_frame: 100,
                        end_frame: 200,
                    }],
                    None,
                ),
                annotation(
                    '2',
                    'b',
                    GroundTruthStateV1::Present,
                    vec![FrameIntervalV1 {
                        start_frame: 120,
                        end_frame: 220,
                    }],
                    None,
                ),
            ],
            prediction(vec![FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            }]),
        );
        let report = audit_multi_annotator_case(&input).unwrap();
        assert_eq!(report.annotator_results.len(), 2);
        assert_eq!(
            report.annotator_results[0].score.f1,
            report.annotator_results[1].score.f1
        );
        assert_ne!(
            report.annotator_results[0].score.matched_interval_iou_mean,
            report.annotator_results[1].score.matched_interval_iou_mean
        );
        assert_eq!(
            report.summary.present_min_f1,
            report
                .annotator_results
                .iter()
                .map(|item| item.score.f1.unwrap())
                .reduce(f64::min)
        );
        assert_eq!(
            report.summary.present_max_f1,
            report
                .annotator_results
                .iter()
                .map(|item| item.score.f1.unwrap())
                .reduce(f64::max)
        );
        assert!(report.summary.start_boundary_mae_frames.unwrap() > 0.0);
    }

    #[test]
    fn ambiguous_annotation_is_excluded_only_from_its_own_denominator() {
        let intervals = vec![FrameIntervalV1 {
            start_frame: 100,
            end_frame: 200,
        }];
        let input = case(
            vec![
                annotation(
                    '1',
                    'a',
                    GroundTruthStateV1::Present,
                    intervals.clone(),
                    None,
                ),
                annotation(
                    '2',
                    'b',
                    GroundTruthStateV1::Ambiguous,
                    Vec::new(),
                    Some("gradual onset"),
                ),
                annotation('3', 'c', GroundTruthStateV1::Absent, Vec::new(), None),
            ],
            prediction(intervals),
        );
        let report = audit_multi_annotator_case(&input).unwrap();
        assert_eq!(report.summary.panel_size, 3);
        assert_eq!(report.summary.scored_annotators, 2);
        assert_eq!(report.summary.present_annotators, 1);
        assert_eq!(report.summary.absent_annotators, 1);
        assert_eq!(report.summary.ambiguous_exclusions, 1);
        assert_eq!(report.summary.not_estimable_exclusions, 0);
        assert_eq!(report.summary.absent_claim_accuracy, Some(0.0));
        assert_eq!(
            report.summary.unanimous_correct_across_scored_annotators,
            Some(false)
        );
    }

    #[test]
    fn annotator_count_range_and_per_annotator_count_accuracy_are_separate() {
        let input = case(
            vec![
                annotation(
                    '1',
                    'a',
                    GroundTruthStateV1::Present,
                    vec![
                        FrameIntervalV1 {
                            start_frame: 100,
                            end_frame: 200,
                        },
                        FrameIntervalV1 {
                            start_frame: 500,
                            end_frame: 600,
                        },
                    ],
                    None,
                ),
                annotation(
                    '2',
                    'b',
                    GroundTruthStateV1::Present,
                    vec![FrameIntervalV1 {
                        start_frame: 100,
                        end_frame: 200,
                    }],
                    None,
                ),
            ],
            prediction(vec![
                FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                },
                FrameIntervalV1 {
                    start_frame: 500,
                    end_frame: 600,
                },
            ]),
        );
        let report = audit_multi_annotator_case(&input).unwrap();
        assert_eq!(report.summary.present_count_error_min, Some(0));
        assert_eq!(report.summary.present_count_error_max, Some(1));
        assert_eq!(report.summary.annotated_occurrence_count_min, Some(1));
        assert_eq!(report.summary.annotated_occurrence_count_max, Some(2));
        assert_eq!(report.summary.present_exact_count_accuracy, Some(0.5));
    }

    #[test]
    fn insufficient_duplicate_or_mismatched_panel_is_rejected() {
        let intervals = vec![FrameIntervalV1 {
            start_frame: 100,
            end_frame: 200,
        }];
        let one = case(
            vec![annotation(
                '1',
                'a',
                GroundTruthStateV1::Present,
                intervals.clone(),
                None,
            )],
            prediction(intervals.clone()),
        );
        assert!(audit_multi_annotator_case(&one).is_err());

        let duplicate = case(
            vec![
                annotation(
                    '1',
                    'a',
                    GroundTruthStateV1::Present,
                    intervals.clone(),
                    None,
                ),
                annotation(
                    '1',
                    'b',
                    GroundTruthStateV1::Present,
                    intervals.clone(),
                    None,
                ),
            ],
            prediction(intervals.clone()),
        );
        assert!(audit_multi_annotator_case(&duplicate).is_err());

        let bad = case(
            vec![
                annotation(
                    '1',
                    'a',
                    GroundTruthStateV1::Present,
                    intervals.clone(),
                    None,
                ),
                annotation(
                    '2',
                    'b',
                    GroundTruthStateV1::Present,
                    intervals.clone(),
                    None,
                ),
            ],
            prediction(intervals),
        );
        let mut bad = bad;
        bad.annotations[1].annotation_sha256 = bad.annotations[0].annotation_sha256.clone();
        assert!(audit_multi_annotator_case(&bad).is_err());
    }
}
