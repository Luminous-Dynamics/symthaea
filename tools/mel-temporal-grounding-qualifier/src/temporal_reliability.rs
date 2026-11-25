//! Dataset-level, source-work-clustered inter-annotator reliability.
//!
//! Nominal occurrence-state agreement and temporal interval/boundary
//! agreement are deliberately separate. Krippendorff's alpha is calculated
//! only for Present/Absent labels across multiple query items. Present/Present
//! interval sets are compared symmetrically through the production scorer.
//! Ambiguous and NotEstimable labels remain visible but are excluded from
//! the measures for which they are not admissible.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::temporal_consensus::{
    audit_multi_annotator_case, AnnotatorGroundingResultV1,
    IndependentTemporalAnnotationV1, MultiAnnotatorGroundingSummaryV1,
    MultiAnnotatorTemporalCaseV1,
};
use crate::temporal_grounding::{
    score_temporal_grounding, AudioArtifactClockV1, GroundTruthStateV1,
    GroundingPredictionStateV1, TemporalGroundTruthV1, TemporalGroundingPlanV1,
    TemporalGroundingPredictionV1, TemporalGroundingScoreV1, TEMPORAL_GROUNDING_SCHEMA_V1,
};

const DATASET_SCHEMA_V1: &str = "mel-multi-annotator-reliability-dataset-v1";
const RELIABILITY_REPORT_SCHEMA_V1: &str = "mel-multi-annotator-reliability-report-v1";
const RELIABILITY_EVALUATOR_V1: &str = "mel-multi-annotator-reliability-v1";
const CONFIRMATORY_SOURCE_CLUSTER_FLOOR: usize = 30;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiAnnotatorReliabilityItemV1 {
    pub item_id: String,
    /// Same source work/session may contribute multiple query items. All are
    /// resampled together in the source-cluster bootstrap.
    pub source_work_id_sha256: String,
    pub case: MultiAnnotatorTemporalCaseV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiAnnotatorReliabilityDatasetV1 {
    pub schema: String,
    pub dataset_id: String,
    pub annotation_policy_version: String,
    pub minimum_raters_per_item: u16,
    /// One frozen scoring plan is shared by every item in this dataset.
    pub scoring_plan: crate::temporal_grounding::TemporalGroundingPlanV1,
    pub bootstrap_replicates: usize,
    pub bootstrap_seed: u64,
    /// Task-specific tolerances must be preregistered, ascending, and > 0.
    pub boundary_tolerances_ms: Vec<u32>,
    pub items: Vec<MultiAnnotatorReliabilityItemV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairwiseAnnotationReliabilityV1 {
    pub item_id: String,
    pub query_id: String,
    pub source_work_id_sha256: String,
    pub annotator_a_sha256: String,
    pub annotator_b_sha256: String,
    pub state_a: GroundTruthStateV1,
    pub state_b: GroundTruthStateV1,
    /// None when either annotation is ambiguous or not estimable.
    pub state_agreement: Option<bool>,
    /// Present/Present only; mean of both production-scored directions.
    pub interval_f1_symmetric: Option<f64>,
    pub matched_iou_symmetric: Option<f64>,
    pub exact_occurrence_count_agreement: Option<bool>,
    pub occurrence_count_difference: Option<usize>,
    pub start_boundary_mae_ms: Option<f64>,
    pub end_boundary_mae_ms: Option<f64>,
    pub start_boundary_abs_errors_ms: Vec<f64>,
    pub end_boundary_abs_errors_ms: Vec<f64>,
    pub exclusion_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KrippendorffAlphaNominalV1 {
    pub alpha: Option<f64>,
    pub observed_disagreement: Option<f64>,
    pub expected_disagreement: Option<f64>,
    pub eligible_items: usize,
    pub items_without_two_eligible_raters: usize,
    pub eligible_ratings: usize,
    /// Ordered coincidence count actually used in nominal alpha.
    pub ratings_in_alpha: usize,
    pub present_ratings: usize,
    pub absent_ratings: usize,
    pub ambiguous_exclusions: usize,
    pub not_estimable_exclusions: usize,
    pub ordered_coincidences: usize,
    pub unordered_rater_pairs: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BoundaryToleranceAgreementV1 {
    pub tolerance_ms: u32,
    pub start_within_tolerance_rate: Option<f64>,
    pub end_within_tolerance_rate: Option<f64>,
    pub both_boundaries_within_tolerance_rate: Option<f64>,
    pub start_boundary_observations: usize,
    pub end_boundary_observations: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiAnnotatorReliabilitySummaryV1 {
    pub dataset_items: usize,
    pub source_work_clusters: usize,
    pub unique_annotator_identities: usize,
    pub nominal_state_agreement_pairs: usize,
    pub nominal_state_agreement_rate: Option<f64>,
    pub same_state_present_pairs: usize,
    pub same_state_absent_pairs: usize,
    pub present_absent_disagreement_pairs: usize,
    pub ambiguous_or_not_estimable_excluded_pairs: usize,
    pub present_present_interval_pairs: usize,
    pub mean_symmetric_interval_f1: Option<f64>,
    pub mean_symmetric_matched_iou: Option<f64>,
    pub exact_occurrence_count_agreement_rate: Option<f64>,
    pub mean_occurrence_count_difference: Option<f64>,
    pub mean_start_boundary_mae_ms: Option<f64>,
    pub mean_end_boundary_mae_ms: Option<f64>,
    pub boundary_tolerance_agreement: Vec<BoundaryToleranceAgreementV1>,
    pub state_reliability: KrippendorffAlphaNominalV1,
    pub recommended_confirmatory_source_cluster_floor: usize,
    pub confirmatory_source_cluster_floor_met: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PercentileIntervalV1 {
    pub estimate: f64,
    pub percentile_95_low: f64,
    pub percentile_95_high: f64,
    pub valid_replicates: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BoundaryToleranceBootstrapV1 {
    pub tolerance_ms: u32,
    pub start_within_tolerance_rate: Option<PercentileIntervalV1>,
    pub end_within_tolerance_rate: Option<PercentileIntervalV1>,
    pub both_boundaries_within_tolerance_rate: Option<PercentileIntervalV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiAnnotatorReliabilityBootstrapV1 {
    pub schema: &'static str,
    pub source_work_clusters: usize,
    pub bootstrap_replicates: usize,
    pub seed: u64,
    pub recommended_confirmatory_source_cluster_floor: usize,
    pub confirmatory_source_cluster_floor_met: bool,
    pub state_alpha_nominal: Option<PercentileIntervalV1>,
    pub state_agreement_rate: Option<PercentileIntervalV1>,
    pub present_pair_interval_f1: Option<PercentileIntervalV1>,
    pub present_pair_matched_iou: Option<PercentileIntervalV1>,
    pub present_pair_exact_count_agreement: Option<PercentileIntervalV1>,
    pub start_boundary_mae_ms: Option<PercentileIntervalV1>,
    pub end_boundary_mae_ms: Option<PercentileIntervalV1>,
    pub boundary_tolerance_agreement: Vec<BoundaryToleranceBootstrapV1>,
}

/// Per-query model performance evaluated independently against every
/// annotator through the #13560 adapter. This is deliberately separate from
/// the annotator-vs-annotator reliability estimates.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelGroundingItemReliabilityV1 {
    pub item_id: String,
    pub query_id: String,
    pub query_text: String,
    pub source_work_id_sha256: String,
    pub annotation_manifest_sha256: String,
    pub artifact: AudioArtifactClockV1,
    pub plan: TemporalGroundingPlanV1,
    pub prediction: TemporalGroundingPredictionV1,
    pub annotations: Vec<IndependentTemporalAnnotationV1>,
    pub summary: MultiAnnotatorGroundingSummaryV1,
    pub annotator_results: Vec<AnnotatorGroundingResultV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelGroundingDatasetSummaryV1 {
    pub query_items: usize,
    pub source_work_clusters: usize,
    pub present_items: usize,
    pub one_to_many_items: usize,
    pub absent_items: usize,
    pub source_work_macro_present_mean_f1: Option<f64>,
    pub source_work_macro_present_exact_count_accuracy: Option<f64>,
    pub source_work_macro_one_to_many_count_gated_f1: Option<f64>,
    pub source_work_macro_absent_claim_accuracy: Option<f64>,
    pub source_work_macro_prediction_coverage: Option<f64>,
    pub source_work_macro_fully_correct_rate: Option<f64>,
    pub model_confidence_calibration_computed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiAnnotatorReliabilityReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub dataset_id: String,
    pub annotation_policy_version: String,
    pub summary: MultiAnnotatorReliabilitySummaryV1,
    /// Model-vs-each-human scores, kept separate from human-vs-human reliability.
    pub model_grounding_summary: ModelGroundingDatasetSummaryV1,
    pub model_grounding_by_item: Vec<ModelGroundingItemReliabilityV1>,
    pub bootstrap: Option<MultiAnnotatorReliabilityBootstrapV1>,
    pub pairwise_results: Vec<PairwiseAnnotationReliabilityV1>,
    pub item_count: usize,
    /// A per-query panel is one unit; confidence intervals resample source works.
    pub confidence_intervals_are_source_work_clustered: bool,
    pub annotation_receipts_authenticated: bool,
    pub pairwise_state_agreement_computed: bool,
    pub pairwise_interval_agreement_computed: bool,
    pub population_confidence_claim_permitted: bool,
    pub proposed_confirmatory_cluster_floor_met: bool,
}

pub fn audit_multi_annotator_dataset_json(
    input: &str,
) -> Result<MultiAnnotatorReliabilityReportV1, String> {
    let dataset: MultiAnnotatorReliabilityDatasetV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid multi-annotator reliability dataset JSON: {error}"))?;
    audit_multi_annotator_dataset(&dataset)
}

pub fn audit_multi_annotator_dataset(
    dataset: &MultiAnnotatorReliabilityDatasetV1,
) -> Result<MultiAnnotatorReliabilityReportV1, String> {
    validate_dataset(dataset)?;

    let mut unique_annotators = BTreeSet::new();
    let mut pairwise_results = Vec::new();
    let mut model_grounding_by_item = Vec::with_capacity(dataset.items.len());
    for item in &dataset.items {
        for annotation in &item.case.annotations {
            unique_annotators.insert(annotation.annotator_id_sha256.as_str());
        }
        pairwise_results.extend(pairwise_item_results(item)?);
        let model_report = audit_multi_annotator_case(&item.case)
            .map_err(|error| format!("item {} rejected by per-annotator model scorer: {error}", item.item_id))?;
        model_grounding_by_item.push(ModelGroundingItemReliabilityV1 {
            item_id: item.item_id.clone(),
            query_id: item.case.query_id.clone(),
            query_text: item.case.query_text.clone(),
            source_work_id_sha256: item.source_work_id_sha256.clone(),
            annotation_manifest_sha256: item.case.annotation_manifest_sha256.clone(),
            artifact: item.case.artifact.clone(),
            plan: item.case.plan.clone(),
            prediction: item.case.prediction.clone(),
            annotations: item.case.annotations.clone(),
            summary: model_report.summary,
            annotator_results: model_report.annotator_results,
        });
    }

    let model_grounding_summary = summarize_model_grounding(&model_grounding_by_item);
    let alpha = nominal_alpha(&dataset.items);
    let summary = summarize_dataset(
        &dataset.items,
        &pairwise_results,
        unique_annotators.len(),
        alpha,
        &dataset.boundary_tolerances_ms,
    );
    let source_cluster_count = dataset
        .items
        .iter()
        .map(|item| item.source_work_id_sha256.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let bootstrap = if source_cluster_count >= 2 {
        Some(bootstrap_dataset(
            dataset,
            &pairwise_results,
            source_cluster_count,
        )?)
    } else {
        None
    };

    Ok(MultiAnnotatorReliabilityReportV1 {
        schema: RELIABILITY_REPORT_SCHEMA_V1,
        evaluator_version: RELIABILITY_EVALUATOR_V1,
        dataset_id: dataset.dataset_id.clone(),
        annotation_policy_version: dataset.annotation_policy_version.clone(),
        summary,
        model_grounding_summary,
        model_grounding_by_item,
        bootstrap,
        pairwise_results,
        item_count: dataset.items.len(),
        confidence_intervals_are_source_work_clustered: true,
        annotation_receipts_authenticated: false,
        pairwise_state_agreement_computed: true,
        pairwise_interval_agreement_computed: true,
        population_confidence_claim_permitted: false,
        proposed_confirmatory_cluster_floor_met: source_cluster_count
            >= CONFIRMATORY_SOURCE_CLUSTER_FLOOR,
    })
}

fn validate_dataset(dataset: &MultiAnnotatorReliabilityDatasetV1) -> Result<(), String> {
    if dataset.schema != DATASET_SCHEMA_V1 {
        return Err(format!("unsupported reliability dataset schema: {}", dataset.schema));
    }
    require_text(&dataset.dataset_id, "dataset_id")?;
    require_text(&dataset.annotation_policy_version, "annotation_policy_version")?;
    if dataset.minimum_raters_per_item < 2 {
        return Err("minimum_raters_per_item must be at least two".into());
    }
    if dataset.items.is_empty() {
        return Err("reliability dataset must contain at least one item".into());
    }
    if dataset.bootstrap_replicates < 1_000 {
        return Err("source-work bootstrap requires at least 1,000 replicates".into());
    }
    if dataset.bootstrap_seed == 0 {
        return Err("bootstrap seed must be nonzero".into());
    }
    if dataset.boundary_tolerances_ms.is_empty()
        || dataset.boundary_tolerances_ms.iter().any(|value| *value == 0)
        || dataset
            .boundary_tolerances_ms
            .windows(2)
            .any(|window| window[0] >= window[1])
    {
        return Err("boundary tolerances must be positive, unique, and strictly ascending".into());
    }

    let mut item_ids = BTreeSet::new();
    let mut query_ids = BTreeSet::new();
    let mut annotation_digests = BTreeSet::new();
    for item in &dataset.items {
        require_text(&item.item_id, "item_id")?;
        validate_digest(&item.source_work_id_sha256, "source_work_id_sha256")?;
        if !item_ids.insert(item.item_id.as_str()) {
            return Err(format!("duplicate reliability item_id: {}", item.item_id));
        }
        if !query_ids.insert(item.case.query_id.as_str()) {
            return Err(format!("duplicate reliability query_id: {}", item.case.query_id));
        }
        if item.case.minimum_annotator_count != dataset.minimum_raters_per_item {
            return Err(format!(
                "item {} panel minimum differs from dataset-level minimum",
                item.item_id
            ));
        }
        if item.case.plan != dataset.scoring_plan {
            return Err(format!(
                "item {} does not use the frozen dataset scoring plan",
                item.item_id
            ));
        }
        for annotation in &item.case.annotations {
            if !annotation_digests.insert(annotation.annotation_sha256.as_str()) {
                return Err(format!(
                    "duplicate annotation-record digest in dataset: {}",
                    annotation.annotation_sha256
                ));
            }
        }
        if item.case.annotation_policy_version != dataset.annotation_policy_version {
            return Err(format!(
                "item {} has a different annotation policy version",
                item.item_id
            ));
        }
        if item.case.annotations.len() < dataset.minimum_raters_per_item as usize {
            return Err(format!(
                "item {} has {} raters, below dataset minimum {}",
                item.item_id,
                item.case.annotations.len(),
                dataset.minimum_raters_per_item
            ));
        }
    }
    Ok(())
}

fn pairwise_item_results(
    item: &MultiAnnotatorReliabilityItemV1,
) -> Result<Vec<PairwiseAnnotationReliabilityV1>, String> {
    let mut pairs = Vec::new();
    let annotations = &item.case.annotations;
    for a in 0..annotations.len() {
        for b in (a + 1)..annotations.len() {
            let left = &annotations[a];
            let right = &annotations[b];
            let eligible_left = is_nominal_state(left.state);
            let eligible_right = is_nominal_state(right.state);
            let state_agreement = (eligible_left && eligible_right)
                .then_some(left.state == right.state);

            let (f1, iou, exact_count, count_difference, starts, ends, exclusion_reason) =
                if !eligible_left || !eligible_right {
                    (
                        None,
                        None,
                        None,
                        None,
                        Vec::new(),
                        Vec::new(),
                        Some("ambiguous_or_not_estimable_annotation".to_string()),
                    )
                } else if left.state == GroundTruthStateV1::Present
                    && right.state == GroundTruthStateV1::Present
                {
                    let left_score = score_annotation_direction(&item.case, left, right)?;
                    let right_score = score_annotation_direction(&item.case, right, left)?;
                    let left_f1 = left_score.f1.unwrap_or(0.0);
                    let right_f1 = right_score.f1.unwrap_or(0.0);
                    let f1 = (left_f1 + right_f1) / 2.0;
                    let iou = average_optional(
                        left_score.matched_interval_iou_mean,
                        right_score.matched_interval_iou_mean,
                    );
                    let left_count = left.intervals.len();
                    let right_count = right.intervals.len();
                    let mut starts = boundary_errors_ms(
                        &left_score,
                        item.case.artifact.sample_rate_hz,
                        true,
                    );
                    starts.extend(boundary_errors_ms(
                        &right_score,
                        item.case.artifact.sample_rate_hz,
                        true,
                    ));
                    let mut ends = boundary_errors_ms(
                        &left_score,
                        item.case.artifact.sample_rate_hz,
                        false,
                    );
                    ends.extend(boundary_errors_ms(
                        &right_score,
                        item.case.artifact.sample_rate_hz,
                        false,
                    ));
                    (
                        Some(f1),
                        iou,
                        Some(left_count == right_count),
                        Some(left_count.abs_diff(right_count)),
                        starts,
                        ends,
                        None,
                    )
                } else {
                    (None, None, None, None, Vec::new(), Vec::new(), None)
                };

            pairs.push(PairwiseAnnotationReliabilityV1 {
                item_id: item.item_id.clone(),
                query_id: item.case.query_id.clone(),
                source_work_id_sha256: item.source_work_id_sha256.clone(),
                annotator_a_sha256: left.annotator_id_sha256.clone(),
                annotator_b_sha256: right.annotator_id_sha256.clone(),
                state_a: left.state,
                state_b: right.state,
                state_agreement,
                interval_f1_symmetric: f1,
                matched_iou_symmetric: iou,
                exact_occurrence_count_agreement: exact_count,
                occurrence_count_difference: count_difference,
                start_boundary_mae_ms: mean(&starts),
                end_boundary_mae_ms: mean(&ends),
                start_boundary_abs_errors_ms: starts,
                end_boundary_abs_errors_ms: ends,
                exclusion_reason,
            });
        }
    }
    Ok(pairs)
}

fn score_annotation_direction(
    case: &MultiAnnotatorTemporalCaseV1,
    truth_annotation: &IndependentTemporalAnnotationV1,
    prediction_annotation: &IndependentTemporalAnnotationV1,
) -> Result<TemporalGroundingScoreV1, String> {
    let truth = TemporalGroundTruthV1 {
        schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
        query_id: case.query_id.clone(),
        query_text: case.query_text.clone(),
        artifact: case.artifact.clone(),
        annotation_policy_version: case.annotation_policy_version.clone(),
        annotation_set_sha256: truth_annotation.annotation_sha256.clone(),
        annotator_count: 1,
        state: GroundTruthStateV1::Present,
        intervals: truth_annotation.intervals.clone(),
        status_reason: None,
    };
    let prediction = TemporalGroundingPredictionV1 {
        schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
        query_id: case.query_id.clone(),
        artifact: case.artifact.clone(),
        state: GroundingPredictionStateV1::Occurrences {
            intervals: prediction_annotation.intervals.clone(),
        },
    };
    score_temporal_grounding(&truth, &prediction, &case.plan)
        .map_err(|issues| format!("pairwise production scorer rejected a Present/Present pair: {issues:?}"))
}

fn nominal_alpha(items: &[MultiAnnotatorReliabilityItemV1]) -> KrippendorffAlphaNominalV1 {
    let mut coincidences = [[0_u64; 2]; 2];
    let mut eligible_items = 0;
    let mut items_without_two_eligible_raters = 0;
    let mut eligible_ratings = 0;
    let mut present_ratings = 0;
    let mut absent_ratings = 0;
    let mut ambiguous_exclusions = 0;
    let mut not_estimable_exclusions = 0;
    let mut unordered_rater_pairs = 0_u64;

    for item in items {
        let mut counts = [0_u64; 2];
        for annotation in &item.case.annotations {
            match annotation.state {
                GroundTruthStateV1::Present => {
                    counts[0] += 1;
                    present_ratings += 1;
                    eligible_ratings += 1;
                }
                GroundTruthStateV1::Absent => {
                    counts[1] += 1;
                    absent_ratings += 1;
                    eligible_ratings += 1;
                }
                GroundTruthStateV1::Ambiguous => ambiguous_exclusions += 1,
                GroundTruthStateV1::NotEstimable => not_estimable_exclusions += 1,
            }
        }
        let n = counts[0] + counts[1];
        if n < 2 {
            items_without_two_eligible_raters += 1;
            continue;
        }
        eligible_items += 1;
        unordered_rater_pairs += n * (n - 1) / 2;
        for category in 0..2 {
            for other in 0..2 {
                let total = counts[category] * counts[other];
                coincidences[category][other] +=
                    total.saturating_sub(u64::from(category == other) * counts[category]);
            }
        }
    }

    let ordered_coincidences: u64 = coincidences.iter().flatten().sum();
    let observed_disagreement = (ordered_coincidences > 0).then(|| {
        (coincidences[0][1] + coincidences[1][0]) as f64
            / ordered_coincidences as f64
    });
    let marginals = [
        coincidences[0][0] + coincidences[0][1],
        coincidences[1][0] + coincidences[1][1],
    ];
    let marginal_total = marginals[0] + marginals[1];
    let expected_disagreement = (marginal_total > 1).then(|| {
        (marginals[0] * marginals[1] * 2) as f64
            / (marginal_total * (marginal_total - 1)) as f64
    });
    let alpha = observed_disagreement
        .zip(expected_disagreement)
        .filter(|(_, expected)| *expected > 0.0)
        .map(|(observed, expected)| 1.0 - observed / expected);

    KrippendorffAlphaNominalV1 {
        alpha,
        observed_disagreement,
        expected_disagreement,
        eligible_items,
        items_without_two_eligible_raters,
        eligible_ratings,
        ratings_in_alpha: ordered_coincidences as usize,
        present_ratings,
        absent_ratings,
        ambiguous_exclusions,
        not_estimable_exclusions,
        ordered_coincidences: ordered_coincidences as usize,
        unordered_rater_pairs: unordered_rater_pairs as usize,
    }
}

fn summarize_model_grounding(
    items: &[ModelGroundingItemReliabilityV1],
) -> ModelGroundingDatasetSummaryV1 {
    let clusters = items
        .iter()
        .map(|item| item.source_work_id_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let present_items = items
        .iter()
        .filter(|item| item.summary.present_annotators > 0)
        .count();
    let one_to_many_items = items
        .iter()
        .filter(|item| item.summary.one_to_many_annotators > 0)
        .count();
    let absent_items = items
        .iter()
        .filter(|item| item.summary.absent_annotators > 0)
        .count();
    ModelGroundingDatasetSummaryV1 {
        query_items: items.len(),
        source_work_clusters: clusters.len(),
        present_items,
        one_to_many_items,
        absent_items,
        source_work_macro_present_mean_f1: mean_item_metric(
            items,
            |summary| summary.present_mean_f1,
        ),
        source_work_macro_present_exact_count_accuracy: mean_item_metric(
            items,
            |summary| summary.present_exact_count_accuracy,
        ),
        source_work_macro_one_to_many_count_gated_f1: mean_item_metric(
            items,
            |summary| summary.one_to_many_mean_count_gated_f1,
        ),
        source_work_macro_absent_claim_accuracy: mean_item_metric(
            items,
            |summary| summary.absent_claim_accuracy,
        ),
        source_work_macro_prediction_coverage: mean_item_metric(
            items,
            |summary| summary.prediction_coverage,
        ),
        source_work_macro_fully_correct_rate: mean_item_metric(
            items,
            |summary| summary.fully_correct_rate,
        ),
        model_confidence_calibration_computed: false,
    }
}

fn mean_item_metric(
    items: &[ModelGroundingItemReliabilityV1],
    field: impl Fn(&MultiAnnotatorGroundingSummaryV1) -> Option<f64>,
) -> Option<f64> {
    let mut by_source_work: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for item in items {
        if let Some(value) = field(&item.summary) {
            by_source_work
                .entry(item.source_work_id_sha256.as_str())
                .or_default()
                .push(value);
        }
    }
    let source_means: Vec<_> = by_source_work
        .values()
        .filter_map(|values| mean(values))
        .collect();
    mean(&source_means)
}

fn cluster_boundary_tolerance(
    pairs: Option<&[&PairwiseAnnotationReliabilityV1]>,
    tolerance_ms: u32,
) -> Option<(f64, f64, f64)> {
    let pairs = pairs?;
    let mut start_errors = Vec::new();
    let mut end_errors = Vec::new();
    for pair in pairs {
        start_errors.extend(pair.start_boundary_abs_errors_ms.iter().copied());
        end_errors.extend(pair.end_boundary_abs_errors_ms.iter().copied());
    }
    if start_errors.is_empty() || end_errors.is_empty() {
        return None;
    }
    let tolerance = tolerance_ms as f64;
    let start_within = start_errors
        .iter()
        .filter(|error| **error <= tolerance)
        .count();
    let end_within = end_errors
        .iter()
        .filter(|error| **error <= tolerance)
        .count();
    let both_total = start_errors.len().min(end_errors.len());
    let both_within = start_errors
        .iter()
        .zip(&end_errors)
        .filter(|(start, end)| **start <= tolerance && **end <= tolerance)
        .count();
    Some((
        start_within as f64 / start_errors.len() as f64,
        end_within as f64 / end_errors.len() as f64,
        both_within as f64 / both_total as f64,
    ))
}

fn boundary_tolerance_summary(
    pairs: &[PairwiseAnnotationReliabilityV1],
    tolerance_ms: u32,
) -> BoundaryToleranceAgreementV1 {
    let mut by_cluster: BTreeMap<&str, Vec<&PairwiseAnnotationReliabilityV1>> = BTreeMap::new();
    for pair in pairs {
        by_cluster
            .entry(pair.source_work_id_sha256.as_str())
            .or_default()
            .push(pair);
    }
    let cluster_metrics: Vec<_> = by_cluster
        .values()
        .filter_map(|cluster_pairs| {
            cluster_boundary_tolerance(Some(cluster_pairs.as_slice()), tolerance_ms)
        })
        .collect();
    let start_observations: usize = pairs
        .iter()
        .map(|pair| pair.start_boundary_abs_errors_ms.len())
        .sum();
    let end_observations: usize = pairs
        .iter()
        .map(|pair| pair.end_boundary_abs_errors_ms.len())
        .sum();
    BoundaryToleranceAgreementV1 {
        tolerance_ms,
        start_within_tolerance_rate: mean(
            &cluster_metrics.iter().map(|item| item.0).collect::<Vec<_>>(),
        ),
        end_within_tolerance_rate: mean(
            &cluster_metrics.iter().map(|item| item.1).collect::<Vec<_>>(),
        ),
        both_boundaries_within_tolerance_rate: mean(
            &cluster_metrics.iter().map(|item| item.2).collect::<Vec<_>>(),
        ),
        start_boundary_observations: start_observations,
        end_boundary_observations: end_observations,
    }
}

fn summarize_dataset(
    items: &[MultiAnnotatorReliabilityItemV1],
    pairwise: &[PairwiseAnnotationReliabilityV1],
    unique_annotators: usize,
    alpha: KrippendorffAlphaNominalV1,
    tolerances: &[u32],
) -> MultiAnnotatorReliabilitySummaryV1 {
    let state_pairs: Vec<_> = pairwise
        .iter()
        .filter_map(|pair| pair.state_agreement)
        .collect();
    let present_pairs: Vec<_> = pairwise
        .iter()
        .filter(|pair| pair.interval_f1_symmetric.is_some())
        .collect();
    let present_present_pairs = present_pairs.len();
    let same_state_present_pairs = pairwise
        .iter()
        .filter(|pair| {
            pair.state_a == GroundTruthStateV1::Present
                && pair.state_b == GroundTruthStateV1::Present
        })
        .count();
    let same_state_absent_pairs = pairwise
        .iter()
        .filter(|pair| {
            pair.state_a == GroundTruthStateV1::Absent
                && pair.state_b == GroundTruthStateV1::Absent
        })
        .count();
    let present_absent_disagreement_pairs = pairwise
        .iter()
        .filter(|pair| {
            is_nominal_state(pair.state_a)
                && is_nominal_state(pair.state_b)
                && pair.state_a != pair.state_b
        })
        .count();
    let excluded_pairs = pairwise
        .iter()
        .filter(|pair| pair.state_agreement.is_none())
        .count();
    let boundary_tolerance_agreement = tolerances
        .iter()
        .map(|tolerance| boundary_tolerance_summary(pairwise, *tolerance))
        .collect();
    let clusters = items
        .iter()
        .map(|item| item.source_work_id_sha256.as_str())
        .collect::<BTreeSet<_>>()
        .len();

    MultiAnnotatorReliabilitySummaryV1 {
        dataset_items: items.len(),
        source_work_clusters: clusters,
        unique_annotator_identities: unique_annotators,
        nominal_state_agreement_pairs: state_pairs.len(),
        nominal_state_agreement_rate: mean_by_source_cluster(pairwise, |pair| {
            pair.state_agreement.map(bool_to_f64)
        }),
        same_state_present_pairs,
        same_state_absent_pairs,
        present_absent_disagreement_pairs,
        ambiguous_or_not_estimable_excluded_pairs: excluded_pairs,
        present_present_interval_pairs: present_present_pairs,
        mean_symmetric_interval_f1: mean_by_source_cluster(pairwise, |pair| {
            pair.interval_f1_symmetric
        }),
        mean_symmetric_matched_iou: mean_by_source_cluster(pairwise, |pair| {
            pair.matched_iou_symmetric
        }),
        exact_occurrence_count_agreement_rate: mean_by_source_cluster(pairwise, |pair| {
            pair.exact_occurrence_count_agreement.map(bool_to_f64)
        }),
        mean_occurrence_count_difference: mean_by_source_cluster(pairwise, |pair| {
            pair.occurrence_count_difference.map(|value| value as f64)
        }),
        mean_start_boundary_mae_ms: mean_boundary_by_source_cluster(pairwise, true),
        mean_end_boundary_mae_ms: mean_boundary_by_source_cluster(pairwise, false),
        boundary_tolerance_agreement,
        state_reliability: alpha,
        recommended_confirmatory_source_cluster_floor: CONFIRMATORY_SOURCE_CLUSTER_FLOOR,
        confirmatory_source_cluster_floor_met: clusters >= CONFIRMATORY_SOURCE_CLUSTER_FLOOR,
    }
}

fn bootstrap_dataset(
    dataset: &MultiAnnotatorReliabilityDatasetV1,
    pairwise: &[PairwiseAnnotationReliabilityV1],
    source_cluster_count: usize,
) -> Result<MultiAnnotatorReliabilityBootstrapV1, String> {
    if source_cluster_count < 2 {
        return Err("cluster bootstrap requires at least two source-work clusters".into());
    }
    let mut clusters: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, item) in dataset.items.iter().enumerate() {
        clusters
            .entry(item.source_work_id_sha256.as_str())
            .or_default()
            .push(index);
    }
    let cluster_keys: Vec<_> = clusters.keys().copied().collect();
    let mut cluster_pairwise: BTreeMap<&str, Vec<&PairwiseAnnotationReliabilityV1>> = BTreeMap::new();
    for pair in pairwise {
        cluster_pairwise
            .entry(pair.source_work_id_sha256.as_str())
            .or_default()
            .push(pair);
    }

    let cluster_f1 = cluster_keys
        .iter()
        .map(|key| cluster_pair_metric(cluster_pairwise.get(key).map(Vec::as_slice), |pair| pair.interval_f1_symmetric))
        .collect::<Vec<_>>();
    let cluster_iou = cluster_keys
        .iter()
        .map(|key| cluster_pair_metric(cluster_pairwise.get(key).map(Vec::as_slice), |pair| pair.matched_iou_symmetric))
        .collect::<Vec<_>>();
    let cluster_count_agreement = cluster_keys
        .iter()
        .map(|key| cluster_pair_metric(cluster_pairwise.get(key).map(Vec::as_slice), |pair| pair.exact_occurrence_count_agreement.map(bool_to_f64)))
        .collect::<Vec<_>>();
    let cluster_state_agreement = cluster_keys
        .iter()
        .map(|key| cluster_pair_metric(cluster_pairwise.get(key).map(Vec::as_slice), |pair| pair.state_agreement.map(bool_to_f64)))
        .collect::<Vec<_>>();
    let cluster_start_mae = cluster_keys
        .iter()
        .map(|key| cluster_boundary_metric(cluster_pairwise.get(key).map(Vec::as_slice), true))
        .collect::<Vec<_>>();
    let cluster_end_mae = cluster_keys
        .iter()
        .map(|key| cluster_boundary_metric(cluster_pairwise.get(key).map(Vec::as_slice), false))
        .collect::<Vec<_>>();

    let cluster_boundary_tolerances: Vec<Vec<Option<(f64, f64, f64)>>> = dataset
        .boundary_tolerances_ms
        .iter()
        .map(|tolerance| {
            cluster_keys
                .iter()
                .map(|key| {
                    cluster_boundary_tolerance(
                        cluster_pairwise.get(key).map(Vec::as_slice),
                        *tolerance,
                    )
                })
                .collect()
        })
        .collect();
    let point_boundary_tolerances: Vec<_> = dataset
        .boundary_tolerances_ms
        .iter()
        .map(|tolerance| boundary_tolerance_summary(pairwise, *tolerance))
        .collect();

    let point_alpha = nominal_alpha(&dataset.items).alpha;
    let point_state = cluster_mean(&cluster_state_agreement);
    let point_f1 = cluster_mean(&cluster_f1);
    let point_iou = cluster_mean(&cluster_iou);
    let point_count = cluster_mean(&cluster_count_agreement);
    let point_start = cluster_mean(&cluster_start_mae);
    let point_end = cluster_mean(&cluster_end_mae);

    let mut state = dataset.bootstrap_seed;
    let mut alpha_samples = Vec::with_capacity(dataset.bootstrap_replicates);
    let mut state_samples = Vec::with_capacity(dataset.bootstrap_replicates);
    let mut f1_samples = Vec::with_capacity(dataset.bootstrap_replicates);
    let mut iou_samples = Vec::with_capacity(dataset.bootstrap_replicates);
    let mut count_samples = Vec::with_capacity(dataset.bootstrap_replicates);
    let mut start_samples = Vec::with_capacity(dataset.bootstrap_replicates);
    let mut end_samples = Vec::with_capacity(dataset.bootstrap_replicates);
    let mut tolerance_start_samples: Vec<Vec<Option<f64>>> = dataset
        .boundary_tolerances_ms
        .iter()
        .map(|_| Vec::with_capacity(dataset.bootstrap_replicates))
        .collect();
    let mut tolerance_end_samples = tolerance_start_samples.clone();
    let mut tolerance_both_samples = tolerance_start_samples.clone();

    for _ in 0..dataset.bootstrap_replicates {
        let mut sampled_items = Vec::new();
        let mut sampled_state = Vec::new();
        let mut sampled_f1 = Vec::new();
        let mut sampled_iou = Vec::new();
        let mut sampled_count = Vec::new();
        let mut sampled_start = Vec::new();
        let mut sampled_end = Vec::new();
        let mut sampled_tolerances = dataset
            .boundary_tolerances_ms
            .iter()
            .map(|_| (Vec::new(), Vec::new(), Vec::new()))
            .collect::<Vec<_>>();

        for _ in 0..source_cluster_count {
            let index = unbiased_index(&mut state, source_cluster_count);
            let key = cluster_keys[index];
            let item_indices = clusters
                .get(key)
                .ok_or_else(|| "bootstrap source cluster disappeared".to_string())?;
            sampled_items.extend(item_indices.iter().map(|item_index| &dataset.items[*item_index]));
            push_some(&mut sampled_state, cluster_state_agreement[index]);
            push_some(&mut sampled_f1, cluster_f1[index]);
            push_some(&mut sampled_iou, cluster_iou[index]);
            push_some(&mut sampled_count, cluster_count_agreement[index]);
            push_some(&mut sampled_start, cluster_start_mae[index]);
            push_some(&mut sampled_end, cluster_end_mae[index]);
            for (tolerance_index, cluster_values) in
                cluster_boundary_tolerances.iter().enumerate()
            {
                if let Some((start, end, both)) = cluster_values[index] {
                    sampled_tolerances[tolerance_index].0.push(start);
                    sampled_tolerances[tolerance_index].1.push(end);
                    sampled_tolerances[tolerance_index].2.push(both);
                }
            }
        }
        for (index, (start, end, both)) in sampled_tolerances.iter().enumerate() {
            tolerance_start_samples[index].push(mean(start));
            tolerance_end_samples[index].push(mean(end));
            tolerance_both_samples[index].push(mean(both));
        }

        alpha_samples.push(nominal_alpha_refs(&sampled_items).alpha);
        state_samples.push(mean(&sampled_state));
        f1_samples.push(mean(&sampled_f1));
        iou_samples.push(mean(&sampled_iou));
        count_samples.push(mean(&sampled_count));
        start_samples.push(mean(&sampled_start));
        end_samples.push(mean(&sampled_end));
    }

    Ok(MultiAnnotatorReliabilityBootstrapV1 {
        schema: "mel-multi-annotator-reliability-bootstrap-v1",
        source_work_clusters: source_cluster_count,
        bootstrap_replicates: dataset.bootstrap_replicates,
        seed: dataset.bootstrap_seed,
        recommended_confirmatory_source_cluster_floor: CONFIRMATORY_SOURCE_CLUSTER_FLOOR,
        confirmatory_source_cluster_floor_met: source_cluster_count
            >= CONFIRMATORY_SOURCE_CLUSTER_FLOOR,
        state_alpha_nominal: percentile_interval(point_alpha, alpha_samples),
        state_agreement_rate: percentile_interval(point_state, state_samples),
        present_pair_interval_f1: percentile_interval(point_f1, f1_samples),
        present_pair_matched_iou: percentile_interval(point_iou, iou_samples),
        present_pair_exact_count_agreement: percentile_interval(point_count, count_samples),
        start_boundary_mae_ms: percentile_interval(point_start, start_samples),
        end_boundary_mae_ms: percentile_interval(point_end, end_samples),
        boundary_tolerance_agreement: dataset
            .boundary_tolerances_ms
            .iter()
            .enumerate()
            .map(|(index, tolerance)| {
                let point = &point_boundary_tolerances[index];
                BoundaryToleranceBootstrapV1 {
                    tolerance_ms: *tolerance,
                    start_within_tolerance_rate: percentile_interval(
                        point.start_within_tolerance_rate,
                        tolerance_start_samples[index].clone(),
                    ),
                    end_within_tolerance_rate: percentile_interval(
                        point.end_within_tolerance_rate,
                        tolerance_end_samples[index].clone(),
                    ),
                    both_boundaries_within_tolerance_rate: percentile_interval(
                        point.both_boundaries_within_tolerance_rate,
                        tolerance_both_samples[index].clone(),
                    ),
                }
            })
            .collect(),
    })
}

fn nominal_alpha_refs(items: &[&MultiAnnotatorReliabilityItemV1]) -> KrippendorffAlphaNominalV1 {
    // The same coincidence-matrix calculation as nominal_alpha, but permits
    // bootstrap resampling to repeat an entire source-work cluster.
    let owned: Vec<_> = items.iter().map(|item| (*item).clone()).collect();
    nominal_alpha(&owned)
}

fn mean_by_source_cluster(
    pairs: &[PairwiseAnnotationReliabilityV1],
    field: impl Fn(&PairwiseAnnotationReliabilityV1) -> Option<f64>,
) -> Option<f64> {
    let mut clusters: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for pair in pairs {
        if let Some(value) = field(pair) {
            clusters
                .entry(pair.source_work_id_sha256.as_str())
                .or_default()
                .push(value);
        }
    }
    let cluster_means: Vec<_> = clusters
        .values()
        .filter_map(|values| mean(values))
        .collect();
    mean(&cluster_means)
}

fn mean_boundary_by_source_cluster(
    pairs: &[PairwiseAnnotationReliabilityV1],
    start: bool,
) -> Option<f64> {
    let mut clusters: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for pair in pairs {
        let errors = if start {
            &pair.start_boundary_abs_errors_ms
        } else {
            &pair.end_boundary_abs_errors_ms
        };
        clusters
            .entry(pair.source_work_id_sha256.as_str())
            .or_default()
            .extend(errors.iter().copied());
    }
    let cluster_means: Vec<_> = clusters
        .values()
        .filter_map(|values| mean(values))
        .collect();
    mean(&cluster_means)
}

fn cluster_pair_metric(
    pairs: Option<&[&PairwiseAnnotationReliabilityV1]>,
    field: impl Fn(&PairwiseAnnotationReliabilityV1) -> Option<f64>,
) -> Option<f64> {
    let values: Vec<_> = pairs?
        .iter()
        .filter_map(|pair| field(pair))
        .collect();
    mean(&values)
}

fn cluster_boundary_metric(
    pairs: Option<&[&PairwiseAnnotationReliabilityV1]>,
    start: bool,
) -> Option<f64> {
    let values: Vec<_> = pairs?
        .iter()
        .flat_map(|pair| {
            if start {
                pair.start_boundary_abs_errors_ms.iter().copied()
            } else {
                pair.end_boundary_abs_errors_ms.iter().copied()
            }
        })
        .collect();
    mean(&values)
}

fn cluster_mean(values: &[Option<f64>]) -> Option<f64> {
    mean(&values.iter().filter_map(|value| *value).collect::<Vec<_>>())
}

fn percentile_interval(point: Option<f64>, mut samples: Vec<Option<f64>>) -> Option<PercentileIntervalV1> {
    let estimate = point?;
    let mut values: Vec<_> = samples.drain(..).flatten().collect();
    if values.len() < 800 {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let last = values.len() - 1;
    let low_index = ((last as f64) * 0.025).floor() as usize;
    let high_index = ((last as f64) * 0.975).ceil() as usize;
    Some(PercentileIntervalV1 {
        estimate,
        percentile_95_low: values[low_index],
        percentile_95_high: values[high_index.min(last)],
        valid_replicates: values.len(),
    })
}

fn push_some(target: &mut Vec<f64>, value: Option<f64>) {
    if let Some(value) = value {
        target.push(value);
    }
}

fn boundary_errors_ms(
    score: &TemporalGroundingScoreV1,
    sample_rate_hz: u32,
    start: bool,
) -> Vec<f64> {
    score
        .matches
        .iter()
        .map(|matched| {
            let frames = if start {
                matched.start_error_frames.unsigned_abs()
            } else {
                matched.end_error_frames.unsigned_abs()
            };
            frames as f64 * 1_000.0 / sample_rate_hz as f64
        })
        .collect()
}

fn is_nominal_state(state: GroundTruthStateV1) -> bool {
    matches!(state, GroundTruthStateV1::Present | GroundTruthStateV1::Absent)
}

fn average_optional(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    left.zip(right).map(|(left, right)| (left + right) / 2.0)
}

fn bool_to_f64(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn unbiased_index(state: &mut u64, upper_bound: usize) -> usize {
    let bound = upper_bound as u64;
    let limit = u64::MAX - (u64::MAX % bound);
    loop {
        *state = (*state).wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        if value < limit {
            return (value % bound) as usize;
        }
    }
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
        AudioArtifactClockV1, FrameIntervalV1, TemporalGroundingPlanV1,
        TEMPORAL_GROUNDING_EVALUATOR_V1,
    };

    const MANIFEST: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn artifact_for(source_work: char) -> AudioArtifactClockV1 {
        AudioArtifactClockV1 {
            audio_sha256: digest(source_work),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 48_000 * 60,
        }
    }

    fn annotation(
        annotator: char,
        receipt_id: u64,
        state: GroundTruthStateV1,
        intervals: Vec<FrameIntervalV1>,
        reason: Option<&str>,
    ) -> IndependentTemporalAnnotationV1 {
        IndependentTemporalAnnotationV1 {
            annotator_id_sha256: digest(annotator),
            annotation_sha256: format!("{receipt_id:064x}"),
            state,
            intervals,
            status_reason: reason.map(str::to_owned),
        }
    }

    fn case(
        query: &str,
        source_work: char,
        annotations: Vec<IndependentTemporalAnnotationV1>,
        prediction: GroundingPredictionStateV1,
    ) -> MultiAnnotatorTemporalCaseV1 {
        MultiAnnotatorTemporalCaseV1 {
            schema: "mel-multi-annotator-temporal-grounding-v1".into(),
            query_id: query.into(),
            query_text: "Every piano entrance".into(),
            artifact: artifact_for(source_work),
            annotation_policy_version: "musician-annotation-v1".into(),
            annotation_manifest_sha256: MANIFEST.into(),
            minimum_annotator_count: 2,
            plan: TemporalGroundingPlanV1 {
                evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
                minimum_interval_iou: 0.5,
            },
            prediction: TemporalGroundingPredictionV1 {
                schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
                query_id: query.into(),
                artifact: artifact_for(source_work),
                state: prediction,
            },
            annotations,
        }
    }

    fn dataset(items: Vec<MultiAnnotatorReliabilityItemV1>) -> MultiAnnotatorReliabilityDatasetV1 {
        MultiAnnotatorReliabilityDatasetV1 {
            schema: DATASET_SCHEMA_V1.into(),
            dataset_id: "synthetic-reliability-v1".into(),
            annotation_policy_version: "musician-annotation-v1".into(),
            minimum_raters_per_item: 2,
            scoring_plan: TemporalGroundingPlanV1 {
                evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
                minimum_interval_iou: 0.5,
            },
            bootstrap_replicates: 1_000,
            bootstrap_seed: 0x13617,
            boundary_tolerances_ms: vec![25, 50, 100],
            items,
        }
    }

    fn item(
        id: &str,
        cluster: char,
        query: &str,
        state: GroundTruthStateV1,
        intervals: Vec<FrameIntervalV1>,
    ) -> MultiAnnotatorReliabilityItemV1 {
        let item_key = id.bytes().fold(7_u64, |acc, byte| {
            acc.wrapping_mul(257).wrapping_add(byte as u64)
        });
        let annotation_intervals = if state == GroundTruthStateV1::Present {
            intervals.clone()
        } else {
            Vec::new()
        };
        let annotations = vec![
            annotation('1', item_key * 4 + 1, state, annotation_intervals.clone(), None),
            annotation('2', item_key * 4 + 2, state, annotation_intervals.clone(), None),
            annotation('3', item_key * 4 + 3, state, annotation_intervals, None),
        ];
        MultiAnnotatorReliabilityItemV1 {
            item_id: id.into(),
            source_work_id_sha256: digest(cluster),
            case: case(
                query,
                cluster,
                annotations,
                if state == GroundTruthStateV1::Present {
                    GroundingPredictionStateV1::Occurrences { intervals }
                } else {
                    GroundingPredictionStateV1::QueryAbsent
                },
            ),
        }
    }

    fn balanced_agreement_dataset() -> MultiAnnotatorReliabilityDatasetV1 {
        let two_events = vec![
            FrameIntervalV1 {
                start_frame: 48_000,
                end_frame: 52_800,
            },
            FrameIntervalV1 {
                start_frame: 120_000,
                end_frame: 124_800,
            },
        ];
        let mut items = Vec::new();
        for cluster in ['a', 'b', 'c'] {
            items.push(item(&format!("{cluster}-present"), cluster, &format!("{cluster}-q1"), GroundTruthStateV1::Present, two_events.clone()));
            items.push(item(&format!("{cluster}-absent"), cluster, &format!("{cluster}-q2"), GroundTruthStateV1::Absent, Vec::new()));
        }
        dataset(items)
    }

    #[test]
    fn model_performance_and_human_reliability_are_reported_on_separate_axes() {
        let report = audit_multi_annotator_dataset(&balanced_agreement_dataset()).unwrap();
        assert_eq!(report.model_grounding_summary.query_items, 6);
        assert_eq!(report.model_grounding_summary.source_work_clusters, 3);
        assert_eq!(report.model_grounding_summary.present_items, 3);
        assert_eq!(report.model_grounding_summary.one_to_many_items, 3);
        assert_eq!(report.model_grounding_summary.absent_items, 3);
        assert_eq!(
            report.model_grounding_summary.source_work_macro_present_mean_f1,
            Some(1.0)
        );
        assert_eq!(
            report.model_grounding_summary.source_work_macro_one_to_many_count_gated_f1,
            Some(1.0)
        );
        assert_eq!(
            report.model_grounding_summary.source_work_macro_absent_claim_accuracy,
            Some(1.0)
        );
        assert_eq!(
            report.model_grounding_summary.source_work_macro_prediction_coverage,
            Some(1.0)
        );
        assert!(!report.model_grounding_summary.model_confidence_calibration_computed);
        assert_eq!(report.model_grounding_by_item.len(), 6);
        assert_eq!(report.model_grounding_by_item[0].annotations.len(), 3);
        assert_eq!(
            report.model_grounding_by_item[0].prediction.query_id,
            report.model_grounding_by_item[0].query_id
        );
        assert_eq!(
            report.model_grounding_by_item[0].artifact.audio_sha256,
            report.model_grounding_by_item[0].prediction.artifact.audio_sha256
        );
    }

    #[test]
    fn unanimous_states_with_prevalence_diversity_produce_alpha_one() {
        let report = audit_multi_annotator_dataset(&balanced_agreement_dataset()).unwrap();
        assert_eq!(report.summary.state_reliability.alpha, Some(1.0));
        assert_eq!(report.summary.state_reliability.eligible_items, 6);
        assert_eq!(report.summary.state_reliability.present_ratings, 9);
        assert_eq!(report.summary.state_reliability.absent_ratings, 9);
        assert_eq!(report.summary.nominal_state_agreement_rate, Some(1.0));
    }

    #[test]
    fn all_same_category_has_undefined_expected_disagreement_not_alpha_one() {
        let mut data = balanced_agreement_dataset();
        data.items.retain(|item| item.case.annotations[0].state == GroundTruthStateV1::Present);
        let report = audit_multi_annotator_dataset(&data).unwrap();
        assert_eq!(report.summary.state_reliability.alpha, None);
        assert_eq!(report.summary.state_reliability.expected_disagreement, Some(0.0));
    }

    #[test]
    fn present_absent_disagreement_reduces_nominal_alpha_and_is_counted() {
        let mut data = balanced_agreement_dataset();
        data.items[0].case.annotations[2].state = GroundTruthStateV1::Absent;
        data.items[0].case.annotations[2].intervals.clear();
        let report = audit_multi_annotator_dataset(&data).unwrap();
        assert!(report.summary.state_reliability.alpha.unwrap() < 1.0);
        assert_eq!(report.summary.present_absent_disagreement_pairs, 2);
        assert_eq!(
            report.model_grounding_by_item[0].summary.absent_claim_accuracy,
            Some(0.0)
        );
        assert!(report.model_grounding_summary.source_work_macro_absent_claim_accuracy.unwrap() < 1.0);
    }

    #[test]
    fn present_present_interval_pairs_use_symmetric_production_scores() {
        let report = audit_multi_annotator_dataset(&balanced_agreement_dataset()).unwrap();
        let pair = report
            .pairwise_results
            .iter()
            .find(|pair| pair.state_a == GroundTruthStateV1::Present && pair.state_b == GroundTruthStateV1::Present)
            .unwrap();
        assert_eq!(pair.interval_f1_symmetric, Some(1.0));
        assert_eq!(pair.matched_iou_symmetric, Some(1.0));
        assert_eq!(pair.exact_occurrence_count_agreement, Some(true));
        assert_eq!(pair.start_boundary_mae_ms, Some(0.0));
    }

    #[test]
    fn slight_boundary_shift_preserves_count_but_changes_temporal_agreement() {
        let mut data = balanced_agreement_dataset();
        let query_id = data.items[0].case.query_id.clone();
        let annotator_a = data.items[0].case.annotations[0].annotator_id_sha256.clone();
        let annotator_b = data.items[0].case.annotations[1].annotator_id_sha256.clone();
        data.items[0].case.annotations[1].intervals[0].start_frame += 1_200;
        data.items[0].case.annotations[1].intervals[0].end_frame += 1_200;
        let report = audit_multi_annotator_dataset(&data).unwrap();
        let pair = report.pairwise_results.iter().find(|pair| {
            pair.query_id == query_id
                && pair.annotator_a_sha256 == annotator_a
                && pair.annotator_b_sha256 == annotator_b
        }).unwrap();
        assert_eq!(pair.exact_occurrence_count_agreement, Some(true));
        assert_eq!(pair.interval_f1_symmetric, Some(1.0));
        assert!(pair.matched_iou_symmetric.unwrap() < 1.0);
        assert_eq!(pair.start_boundary_mae_ms, Some(25.0));
    }

    #[test]
    fn occurrence_count_disagreement_is_separate_from_state_agreement() {
        let mut data = balanced_agreement_dataset();
        data.items[0].case.annotations[2].intervals.pop();
        let report = audit_multi_annotator_dataset(&data).unwrap();
        let pair = report.pairwise_results.iter().find(|pair| {
            pair.query_id == data.items[0].case.query_id
                && pair.annotator_a_sha256 == data.items[0].case.annotations[0].annotator_id_sha256
                && pair.annotator_b_sha256 == data.items[0].case.annotations[2].annotator_id_sha256
        }).unwrap();
        assert_eq!(pair.state_agreement, Some(true));
        assert_eq!(pair.exact_occurrence_count_agreement, Some(false));
        assert_eq!(pair.occurrence_count_difference, Some(1));
    }

    #[test]
    fn ambiguous_and_not_estimable_are_excluded_not_recoded_absent() {
        let mut data = balanced_agreement_dataset();
        data.items[0].case.annotations[2].state = GroundTruthStateV1::Ambiguous;
        data.items[0].case.annotations[2].intervals.clear();
        data.items[0].case.annotations[2].status_reason = Some("gradual onset".into());
        data.items[1].case.annotations[2].state = GroundTruthStateV1::NotEstimable;
        data.items[1].case.annotations[2].intervals.clear();
        data.items[1].case.annotations[2].status_reason = Some("masking prevents a reliable judgement".into());
        let report = audit_multi_annotator_dataset(&data).unwrap();
        assert_eq!(report.summary.ambiguous_or_not_estimable_excluded_pairs, 4);
        assert_eq!(report.summary.state_reliability.ambiguous_exclusions, 1);
        assert_eq!(report.summary.state_reliability.not_estimable_exclusions, 1);
        assert_eq!(report.summary.state_reliability.eligible_ratings, 16);
        assert!(report.summary.present_absent_disagreement_pairs < report.pairwise_results.len());
    }

    #[test]
    fn source_work_bootstrap_is_deterministic_and_clustered() {
        let mut data = balanced_agreement_dataset();
        data.items[0].case.annotations[1].intervals[0].start_frame += 1_440;
        data.items[0].case.annotations[1].intervals[0].end_frame += 1_440;
        let report_a = audit_multi_annotator_dataset(&data).unwrap();
        let report_b = audit_multi_annotator_dataset(&data).unwrap();
        assert_eq!(report_a.bootstrap, report_b.bootstrap);
        let bootstrap = report_a.bootstrap.unwrap();
        assert_eq!(bootstrap.source_work_clusters, 3);
        assert!(!bootstrap.confirmatory_source_cluster_floor_met);
        assert_eq!(bootstrap.state_alpha_nominal.unwrap().estimate, 1.0);
        assert!(!report_a.population_confidence_claim_permitted);
        assert!(!report_a.proposed_confirmatory_cluster_floor_met);
        assert_eq!(report_a.summary.dataset_items, 6);
        assert_eq!(report_a.summary.source_work_clusters, 3);
        assert_eq!(bootstrap.boundary_tolerance_agreement.len(), 3);
        let tolerance_25ms = bootstrap.boundary_tolerance_agreement[0]
            .both_boundaries_within_tolerance_rate
            .as_ref()
            .unwrap();
        assert!((tolerance_25ms.estimate - (8.0 / 9.0)).abs() < 1e-9);
        assert_eq!(tolerance_25ms.valid_replicates, 1_000);
        let tolerance_50ms = bootstrap.boundary_tolerance_agreement[1]
            .both_boundaries_within_tolerance_rate
            .as_ref()
            .unwrap();
        assert_eq!(tolerance_50ms.estimate, 1.0);
        assert_eq!(tolerance_50ms.valid_replicates, 1_000);
    }

    #[test]
    fn duplicate_items_and_invalid_tolerance_order_are_rejected() {
        let mut data = balanced_agreement_dataset();
        data.items[1].item_id = data.items[0].item_id.clone();
        assert!(audit_multi_annotator_dataset(&data).is_err());

        let mut data = balanced_agreement_dataset();
        data.boundary_tolerances_ms = vec![50, 25, 100];
        assert!(audit_multi_annotator_dataset(&data).is_err());

        let mut data = balanced_agreement_dataset();
        data.items[1].case.query_id = data.items[0].case.query_id.clone();
        assert!(audit_multi_annotator_dataset(&data).is_err());

        let mut data = balanced_agreement_dataset();
        data.items[1].case.annotations[0].annotation_sha256 =
            data.items[0].case.annotations[0].annotation_sha256.clone();
        assert!(audit_multi_annotator_dataset(&data).is_err());

        let mut data = balanced_agreement_dataset();
        data.items[0].case.plan.minimum_interval_iou = 0.75;
        assert!(audit_multi_annotator_dataset(&data).is_err());

        let mut data = balanced_agreement_dataset();
        data.items[0].case.minimum_annotator_count = 3;
        assert!(audit_multi_annotator_dataset(&data).is_err());
    }
}
