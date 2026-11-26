//! Corpus-level multi-annotator reliability profile.
//!
//! Disposition reliability uses nominal Krippendorff alpha over item-level
//! annotator states. Temporal interval agreement is reported separately as
//! symmetric production-scorer F1 at frozen IoU thresholds. This module does
//! not calculate unitizing alpha, build a consensus reference, or score a model.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::temporal_consensus::IndependentTemporalAnnotationV1;
use crate::temporal_grounding::{
    score_temporal_grounding, validate_temporal_ground_truth, AudioArtifactClockV1,
    GroundingPredictionStateV1, GroundTruthStateV1,
    TemporalGroundTruthV1, TemporalGroundingPlanV1, TemporalGroundingPredictionV1,
    TEMPORAL_GROUNDING_EVALUATOR_V1, TEMPORAL_GROUNDING_SCHEMA_V1,
};

const CASE_SCHEMA_V1: &str = "mel-temporal-reliability-corpus-v1";
const REPORT_SCHEMA_V1: &str = "mel-temporal-annotation-reliability-report-v1";
const IOU_THRESHOLDS_V1: [f64; 4] = [0.25, 0.50, 0.75, 0.90];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReliabilityCorpusItemV1 {
    pub item_id: String,
    /// Opaque lineage commitment shared by excerpts from the same source work.
    pub source_work_cluster_sha256: String,
    pub query_id: String,
    pub query_text: String,
    pub artifact: AudioArtifactClockV1,
    pub annotation_policy_version: String,
    pub annotations: Vec<IndependentTemporalAnnotationV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReliabilityCorpusCaseV1 {
    pub schema: String,
    pub annotation_manifest_sha256: String,
    pub minimum_annotators_per_item: u16,
    pub items: Vec<ReliabilityCorpusItemV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReliabilityCategoryCountV1 {
    pub state: GroundTruthStateV1,
    pub ratings: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KrippendorffAlphaNominalV1 {
    /// Number of raw item×annotator ratings that entered the coincidence matrix.
    pub ratings: usize,
    pub pairable_items: usize,
    pub categories: Vec<ReliabilityCategoryCountV1>,
    pub observed_disagreement: Option<f64>,
    pub expected_disagreement: Option<f64>,
    pub alpha: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairwiseEventAgreementThresholdV1 {
    pub minimum_interval_iou: f64,
    pub present_present_annotation_pairs: usize,
    pub symmetric_event_f1_mean: Option<f64>,
    pub matched_interval_iou_mean: Option<f64>,
    pub exact_occurrence_count_agreement: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TemporalAnnotationReliabilityReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub item_count: usize,
    pub source_work_clusters: usize,
    pub total_annotation_records: usize,
    pub minimum_annotators_per_item: u16,
    pub disposition_alpha: KrippendorffAlphaNominalV1,
    pub pairwise_by_iou_threshold: Vec<PairwiseEventAgreementThresholdV1>,
    /// The report intentionally does not select or synthesize a single reference.
    pub consensus_ground_truth_created: bool,
    /// No model predictions are supplied to this reliability-only corpus API.
    pub model_performance_scored: bool,
    /// The corpus is a sample; alpha is not a population confidence interval.
    pub population_confidence_interval_computed: bool,
    /// Source-work bootstrap is a separate inferential layer, not implied here.
    pub source_work_bootstrap_computed: bool,
    /// This is nominal disposition alpha, not Krippendorff's unitizing alpha.
    pub unitizing_alpha_computed: bool,
}

pub fn audit_reliability_corpus_json(
    input: &str,
) -> Result<TemporalAnnotationReliabilityReportV1, String> {
    let case: ReliabilityCorpusCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid temporal reliability corpus JSON: {error}"))?;
    audit_reliability_corpus(&case)
}

pub fn audit_reliability_corpus(
    case: &ReliabilityCorpusCaseV1,
) -> Result<TemporalAnnotationReliabilityReportV1, String> {
    if case.schema != CASE_SCHEMA_V1 {
        return Err(format!("unsupported reliability corpus schema: {}", case.schema));
    }
    validate_digest(&case.annotation_manifest_sha256, "annotation_manifest_sha256")?;
    if case.minimum_annotators_per_item < 2 {
        return Err("minimum_annotators_per_item must be at least two".into());
    }
    if case.items.is_empty() {
        return Err("reliability corpus must contain at least one item".into());
    }

    let mut item_ids = BTreeSet::new();
    let mut source_work_clusters = BTreeSet::new();
    let mut total_annotation_records = 0_usize;
    for (item_index, item) in case.items.iter().enumerate() {
        require_text(&item.item_id, &format!("items[{item_index}].item_id"))?;
        require_text(&item.query_id, &format!("items[{item_index}].query_id"))?;
        require_text(&item.query_text, &format!("items[{item_index}].query_text"))?;
        require_text(
            &item.annotation_policy_version,
            &format!("items[{item_index}].annotation_policy_version"),
        )?;
        validate_digest(
            &item.source_work_cluster_sha256,
            &format!("items[{item_index}].source_work_cluster_sha256"),
        )?;
        validate_artifact(&item.artifact, item_index)?;
        if !item_ids.insert(item.item_id.as_str()) {
            return Err(format!("duplicate reliability item_id: {}", item.item_id));
        }
        source_work_clusters.insert(item.source_work_cluster_sha256.as_str());

        if item.annotations.len() < case.minimum_annotators_per_item as usize {
            return Err(format!(
                "item {} has {} annotations, below minimum {}",
                item.item_id,
                item.annotations.len(),
                case.minimum_annotators_per_item
            ));
        }
        if item.annotations.len() > u16::MAX as usize {
            return Err(format!("item {} exceeds supported annotation count", item.item_id));
        }
        total_annotation_records = total_annotation_records
            .checked_add(item.annotations.len())
            .ok_or_else(|| "annotation record count overflow".to_string())?;

        let mut annotator_ids = BTreeSet::new();
        let mut annotation_digests = BTreeSet::new();
        for (annotation_index, annotation) in item.annotations.iter().enumerate() {
            validate_digest(
                &annotation.annotator_id_sha256,
                &format!("items[{item_index}].annotations[{annotation_index}].annotator_id_sha256"),
            )?;
            validate_digest(
                &annotation.annotation_sha256,
                &format!("items[{item_index}].annotations[{annotation_index}].annotation_sha256"),
            )?;
            if !annotator_ids.insert(annotation.annotator_id_sha256.as_str()) {
                return Err(format!(
                    "item {} repeats an annotator identity",
                    item.item_id
                ));
            }
            if !annotation_digests.insert(annotation.annotation_sha256.as_str()) {
                return Err(format!(
                    "item {} repeats an annotation receipt digest",
                    item.item_id
                ));
            }

            let truth = annotation_to_truth(item, annotation);
            let issues = validate_temporal_ground_truth(&truth);
            if !issues.is_empty() {
                return Err(format!(
                    "item {} annotation {} failed production truth validation: {issues:?}",
                    item.item_id, annotation_index
                ));
            }
        }
    }

    let disposition_alpha = calculate_nominal_alpha(&case.items)?;
    let pairwise_by_iou_threshold = calculate_pairwise_event_agreement(&case.items)?;

    Ok(TemporalAnnotationReliabilityReportV1 {
        schema: REPORT_SCHEMA_V1,
        evaluator_version: "mel-temporal-annotation-reliability-v1",
        item_count: case.items.len(),
        source_work_clusters: source_work_clusters.len(),
        total_annotation_records,
        minimum_annotators_per_item: case.minimum_annotators_per_item,
        disposition_alpha,
        pairwise_by_iou_threshold,
        consensus_ground_truth_created: false,
        model_performance_scored: false,
        population_confidence_interval_computed: false,
        source_work_bootstrap_computed: false,
        unitizing_alpha_computed: false,
    })
}

fn calculate_nominal_alpha(
    items: &[ReliabilityCorpusItemV1],
) -> Result<KrippendorffAlphaNominalV1, String> {
    const CATEGORY_COUNT: usize = 4;
    let mut coincidence = [[0.0_f64; CATEGORY_COUNT]; CATEGORY_COUNT];
    let mut category_ratings = [0_usize; CATEGORY_COUNT];
    let mut ratings = 0_usize;
    let mut pairable_items = 0_usize;

    for item in items {
        let annotation_count = item.annotations.len();
        if annotation_count < 2 {
            continue;
        }
        pairable_items += 1;
        ratings = ratings
            .checked_add(annotation_count)
            .ok_or_else(|| "nominal alpha rating count overflow".to_string())?;
        let scale = 1.0 / (annotation_count - 1) as f64;
        let states: Vec<_> = item.annotations.iter().map(|value| value.state).collect();
        for state in &states {
            let category = category_index(*state);
            category_ratings[category] = category_ratings[category]
                .checked_add(1)
                .ok_or_else(|| "nominal alpha category count overflow".to_string())?;
        }
        for (left_index, left) in states.iter().enumerate() {
            for (right_index, right) in states.iter().enumerate() {
                if left_index == right_index {
                    continue;
                }
                coincidence[category_index(*left)][category_index(*right)] += scale;
            }
        }
    }

    if pairable_items == 0 || ratings < 2 {
        return Err("nominal alpha has no pairable ratings".into());
    }

    let total_coincidences: f64 = coincidence.iter().flatten().sum();
    if total_coincidences <= 0.0 {
        return Err("nominal alpha coincidence matrix is empty".into());
    }
    let observed_disagreement = coincidence
        .iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter(|(column, _)| *column != row)
                .map(|(_, value)| *value)
                .sum::<f64>()
        })
        .sum::<f64>()
        / total_coincidences;

    let total_ratings = category_ratings.iter().sum::<usize>();
    let same_category_pairs = category_ratings.iter().try_fold(0_usize, |sum, count| {
        let pairs = count
            .checked_mul(count.saturating_sub(1))
            .ok_or_else(|| "nominal alpha category pair count overflow".to_string())?;
        sum.checked_add(pairs)
            .ok_or_else(|| "nominal alpha category pair total overflow".to_string())
    })?;
    let all_pairs = total_ratings
        .checked_mul(total_ratings.saturating_sub(1))
        .ok_or_else(|| "nominal alpha total pair count overflow".to_string())?;
    let expected_disagreement = (all_pairs > 0)
        .then(|| 1.0 - same_category_pairs as f64 / all_pairs as f64);
    let alpha = expected_disagreement
        .filter(|value| *value > f64::EPSILON)
        .map(|expected| 1.0 - observed_disagreement / expected);

    Ok(KrippendorffAlphaNominalV1 {
        ratings,
        pairable_items,
        categories: [
            GroundTruthStateV1::Present,
            GroundTruthStateV1::Absent,
            GroundTruthStateV1::Ambiguous,
            GroundTruthStateV1::NotEstimable,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, state)| ReliabilityCategoryCountV1 {
            state,
            ratings: category_ratings[index],
        })
        .collect(),
        observed_disagreement: Some(observed_disagreement),
        expected_disagreement,
        alpha,
    })
}

fn calculate_pairwise_event_agreement(
    items: &[ReliabilityCorpusItemV1],
) -> Result<Vec<PairwiseEventAgreementThresholdV1>, String> {
    let mut output = Vec::with_capacity(IOU_THRESHOLDS_V1.len());
    for threshold in IOU_THRESHOLDS_V1 {
        let plan = TemporalGroundingPlanV1 {
            evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
            minimum_interval_iou: threshold,
        };
        let mut symmetric_f1 = Vec::new();
        let mut mean_iou = Vec::new();
        let mut exact_count_agreement = Vec::new();
        for item in items {
            for left_index in 0..item.annotations.len() {
                let left = &item.annotations[left_index];
                if left.state != GroundTruthStateV1::Present {
                    continue;
                }
                for right in item.annotations.iter().skip(left_index + 1) {
                    if right.state != GroundTruthStateV1::Present {
                        continue;
                    }
                    let left_truth = annotation_to_truth(item, left);
                    let right_truth = annotation_to_truth(item, right);
                    let left_prediction = annotation_to_prediction(item, left);
                    let right_prediction = annotation_to_prediction(item, right);
                    let left_as_truth_right_as_prediction =
                        score_temporal_grounding(&left_truth, &right_prediction, &plan)
                            .map_err(|issues| {
                                format!("left-to-right reliability scoring failed: {issues:?}")
                            })?;
                    let right_as_truth_left_as_prediction =
                        score_temporal_grounding(&right_truth, &left_prediction, &plan)
                            .map_err(|issues| {
                                format!("right-to-left reliability scoring failed: {issues:?}")
                            })?;
                    if let (Some(left_f1), Some(right_f1)) = (
                        left_as_truth_right_as_prediction.f1,
                        right_as_truth_left_as_prediction.f1,
                    ) {
                        symmetric_f1.push((left_f1 + right_f1) / 2.0);
                    }
                    if let (Some(left_iou), Some(right_iou)) = (
                        left_as_truth_right_as_prediction.matched_interval_iou_mean,
                        right_as_truth_left_as_prediction.matched_interval_iou_mean,
                    ) {
                        mean_iou.push((left_iou + right_iou) / 2.0);
                    }
                    exact_count_agreement.push(if left.intervals.len() == right.intervals.len() {
                        1.0
                    } else {
                        0.0
                    });
                }
            }
        }
        output.push(PairwiseEventAgreementThresholdV1 {
            minimum_interval_iou: threshold,
            present_present_annotation_pairs: symmetric_f1.len(),
            symmetric_event_f1_mean: mean(&symmetric_f1),
            matched_interval_iou_mean: mean(&mean_iou),
            exact_occurrence_count_agreement: mean(&exact_count_agreement),
        });
    }
    Ok(output)
}

fn annotation_to_truth(
    item: &ReliabilityCorpusItemV1,
    annotation: &IndependentTemporalAnnotationV1,
) -> TemporalGroundTruthV1 {
    TemporalGroundTruthV1 {
        schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
        query_id: item.query_id.clone(),
        query_text: item.query_text.clone(),
        artifact: item.artifact.clone(),
        annotation_policy_version: item.annotation_policy_version.clone(),
        annotation_set_sha256: annotation.annotation_sha256.clone(),
        annotator_count: 1,
        state: annotation.state,
        intervals: annotation.intervals.clone(),
        status_reason: annotation.status_reason.clone(),
    }
}

fn annotation_to_prediction(
    item: &ReliabilityCorpusItemV1,
    annotation: &IndependentTemporalAnnotationV1,
) -> TemporalGroundingPredictionV1 {
    TemporalGroundingPredictionV1 {
        schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
        query_id: item.query_id.clone(),
        artifact: item.artifact.clone(),
        state: GroundingPredictionStateV1::Occurrences {
            intervals: annotation.intervals.clone(),
        },
    }
}

fn category_index(state: GroundTruthStateV1) -> usize {
    match state {
        GroundTruthStateV1::Present => 0,
        GroundTruthStateV1::Absent => 1,
        GroundTruthStateV1::Ambiguous => 2,
        GroundTruthStateV1::NotEstimable => 3,
    }
}

fn validate_artifact(artifact: &AudioArtifactClockV1, item_index: usize) -> Result<(), String> {
    validate_digest(
        &artifact.audio_sha256,
        &format!("items[{item_index}].artifact.audio_sha256"),
    )?;
    if artifact.sample_rate_hz == 0 || artifact.channels == 0 || artifact.total_frames == 0 {
        return Err(format!(
            "items[{item_index}] audio sample rate, channels and frame count must be positive"
        ));
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

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_grounding::FrameIntervalV1;

    fn digest(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn artifact(ch: char) -> AudioArtifactClockV1 {
        AudioArtifactClockV1 {
            audio_sha256: digest(ch),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 480_000,
        }
    }

    fn annotation(
        annotator: char,
        receipt: char,
        state: GroundTruthStateV1,
        intervals: Vec<FrameIntervalV1>,
    ) -> IndependentTemporalAnnotationV1 {
        IndependentTemporalAnnotationV1 {
            annotator_id_sha256: digest(annotator),
            annotation_sha256: digest(receipt),
            state,
            intervals,
            status_reason: if state == GroundTruthStateV1::Ambiguous {
                Some("annotator cannot identify a stable event boundary".into())
            } else if state == GroundTruthStateV1::NotEstimable {
                Some("audio is masked".into())
            } else {
                None
            },
        }
    }

    fn item(
        id: &str,
        source: char,
        audio: char,
        annotations: Vec<IndependentTemporalAnnotationV1>,
    ) -> ReliabilityCorpusItemV1 {
        ReliabilityCorpusItemV1 {
            item_id: id.into(),
            source_work_cluster_sha256: digest(source),
            query_id: format!("query-{id}"),
            query_text: "Identify every piano entrance".into(),
            artifact: artifact(audio),
            annotation_policy_version: "music-temporal-policy-v1".into(),
            annotations,
        }
    }

    fn synthetic_case() -> ReliabilityCorpusCaseV1 {
        ReliabilityCorpusCaseV1 {
            schema: CASE_SCHEMA_V1.into(),
            annotation_manifest_sha256: digest('f'),
            minimum_annotators_per_item: 2,
            items: vec![
                item(
                    "item-1",
                    '1',
                    'a',
                    vec![
                        annotation(
                            '1',
                            'a',
                            GroundTruthStateV1::Present,
                            vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
                        ),
                        annotation(
                            '2',
                            'b',
                            GroundTruthStateV1::Present,
                            vec![FrameIntervalV1 { start_frame: 102, end_frame: 202 }],
                        ),
                        annotation(
                            '3',
                            'c',
                            GroundTruthStateV1::Present,
                            vec![FrameIntervalV1 { start_frame: 110, end_frame: 210 }],
                        ),
                    ],
                ),
                item(
                    "item-2",
                    '1',
                    'b',
                    vec![
                        annotation('1', 'd', GroundTruthStateV1::Absent, Vec::new()),
                        annotation(
                            '2',
                            'e',
                            GroundTruthStateV1::Ambiguous,
                            Vec::new(),
                        ),
                    ],
                ),
                item(
                    "item-3",
                    '2',
                    'c',
                    vec![
                        annotation('1', '6', GroundTruthStateV1::Absent, Vec::new()),
                        annotation('3', '7', GroundTruthStateV1::Ambiguous, Vec::new()),
                    ],
                ),
                item(
                    "item-4",
                    '3',
                    'd',
                    vec![
                        annotation('2', '8', GroundTruthStateV1::Absent, Vec::new()),
                        annotation('3', '9', GroundTruthStateV1::Absent, Vec::new()),
                    ],
                ),
            ],
        }
    }

    #[test]
    fn nominal_alpha_matches_hand_calculated_four_item_nine_rating_case() {
        let report = audit_reliability_corpus(&synthetic_case()).unwrap();
        assert_eq!(report.item_count, 4);
        assert_eq!(report.source_work_clusters, 3);
        assert_eq!(report.total_annotation_records, 9);
        assert_eq!(report.disposition_alpha.ratings, 9);
        assert!((report.disposition_alpha.observed_disagreement.unwrap() - 4.0 / 9.0).abs() < 1e-12);
        assert!((report.disposition_alpha.expected_disagreement.unwrap() - 13.0 / 18.0).abs() < 1e-12);
        assert!((report.disposition_alpha.alpha.unwrap() - 5.0 / 13.0).abs() < 1e-12);
        assert!(!report.unitizing_alpha_computed);
        assert!(!report.consensus_ground_truth_created);
        assert!(!report.model_performance_scored);
        assert!(!report.population_confidence_interval_computed);
    }

    #[test]
    fn pairwise_event_f1_decreases_when_iou_threshold_becomes_stricter() {
        let report = audit_reliability_corpus(&synthetic_case()).unwrap();
        let values: Vec<_> = report
            .pairwise_by_iou_threshold
            .iter()
            .map(|item| item.symmetric_event_f1_mean.unwrap())
            .collect();
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 1.0);
        assert_eq!(values[2], 1.0);
        assert!((values[3] - 1.0 / 3.0).abs() < 1e-12);
        assert!(values.windows(2).all(|pair| pair[1] <= pair[0]));
    }

    #[test]
    fn undefined_nominal_alpha_does_not_invent_zero_or_one() {
        let mut input = synthetic_case();
        for item in &mut input.items {
            for annotation in &mut item.annotations {
                annotation.state = GroundTruthStateV1::Present;
                annotation.intervals = vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }];
            }
        }
        let report = audit_reliability_corpus(&input).unwrap();
        assert_eq!(report.disposition_alpha.observed_disagreement, Some(0.0));
        assert_eq!(report.disposition_alpha.expected_disagreement, Some(0.0));
        assert_eq!(report.disposition_alpha.alpha, None);
    }

    #[test]
    fn duplicate_items_annotators_and_receipts_are_rejected() {
        let mut input = synthetic_case();
        input.items[1].item_id = input.items[0].item_id.clone();
        assert!(audit_reliability_corpus(&input).is_err());

        let mut input = synthetic_case();
        input.items[0].annotations[1].annotator_id_sha256 =
            input.items[0].annotations[0].annotator_id_sha256.clone();
        assert!(audit_reliability_corpus(&input).is_err());

        let mut input = synthetic_case();
        input.items[0].annotations[1].annotation_sha256 =
            input.items[0].annotations[0].annotation_sha256.clone();
        assert!(audit_reliability_corpus(&input).is_err());
    }

    #[test]
    fn corpus_rejects_an_item_with_too_few_raters() {
        let mut input = synthetic_case();
        input.items[0].annotations.pop();
        input.minimum_annotators_per_item = 3;
        assert!(audit_reliability_corpus(&input).is_err());
    }
}
