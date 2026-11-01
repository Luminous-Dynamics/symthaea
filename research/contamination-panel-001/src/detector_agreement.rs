//! Detector complementarity, selective coverage and disagreement analysis.
//!
//! This module deliberately does not fit an ensemble. It compares independently
//! calibrated detector outputs on the same frozen holdout examples and treats
//! abstention/unavailability as part of the measurement contract.
//!
//! Every detector threshold is fixed on a separately committed calibration
//! cohort. AUROC and average precision use scored examples with determinate
//! labels; class-conditional coverage and worst-case missing-output bounds keep
//! selective coverage visible. Pairwise rank and decision agreement are
//! reported separately because they measure different properties.

use crate::exposure_transfer::TransferRelationV1;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DETECTOR_AGREEMENT_REQUEST_SCHEMA: &str = "symthaea.detector_agreement_request.v1";
pub const DETECTOR_AGREEMENT_REPORT_SCHEMA: &str = "symthaea.detector_agreement_report.v1";
pub const MAX_COMPARATORS: usize = 64;
pub const MIN_COMPARATORS: usize = 2;
pub const MAX_HOLDOUT_SAMPLES: usize = 1_000_000;
pub const MAX_OPAQUE_ID_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GroundTruthLabelV1 {
    Clean,
    Contaminated,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScoreOrientationV1 {
    HigherIsContaminated,
    LowerIsContaminated,
}

/// A score threshold already transformed to the common "higher = contaminated"
/// orientation. No threshold may be selected on the holdout cohort.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FixedThresholdCalibrationV1 {
    pub threshold_fit_cohort_sha256: String,
    /// Independent calibration-validation cohort, not the threshold-fit cohort.
    pub calibration_cohort_sha256: String,
    pub calibration_policy_sha256: String,
    pub threshold_fit_sample_count: u64,
    pub calibration_sample_count: u64,
    pub calibration_clean_count: u64,
    pub calibration_false_positive_count: u64,
    pub target_fpr_permille: u16,
    pub normalized_threshold: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorAgreementSpecV1 {
    pub detector_id: String,
    pub detector_family: String,
    pub detector_version: String,
    pub model_checkpoint_sha256: String,
    pub feature_protocol_sha256: String,
    pub score_orientation: ScoreOrientationV1,
    pub calibration: FixedThresholdCalibrationV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DetectorScoreStatusV1 {
    Scored {
        raw_score: f64,
        score_evidence_sha256: String,
        feature_protocol_sha256: String,
    },
    Abstained {
        reason_code: String,
    },
    Unavailable {
        reason_code: String,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorScoreObservationV1 {
    pub detector_id: String,
    pub result: DetectorScoreStatusV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorAgreementSampleV1 {
    pub sample_id: String,
    pub task_family: String,
    pub difficulty_band: String,
    pub relation: TransferRelationV1,
    pub source_cluster_id: String,
    pub truth: GroundTruthLabelV1,
    /// Exactly one explicit scored/abstained/unavailable status per detector.
    pub detector_results: Vec<DetectorScoreObservationV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorAgreementRequestV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_checkpoint_sha256: String,
    pub holdout_manifest_sha256: String,
    pub detector_specs: Vec<DetectorAgreementSpecV1>,
    pub samples: Vec<DetectorAgreementSampleV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DetectorStratumReportV1 {
    pub detector_id: String,
    pub task_family: String,
    pub difficulty_band: String,
    pub relation: TransferRelationV1,
    pub sample_count: u64,
    pub clean_count: u64,
    pub contaminated_count: u64,
    pub indeterminate_truth_count: u64,
    pub scored_count: u64,
    pub abstained_count: u64,
    pub unavailable_count: u64,
    pub scored_clean_count: u64,
    pub scored_contaminated_count: u64,
    pub coverage_permille: f64,
    pub clean_coverage_permille: Option<f64>,
    pub contaminated_coverage_permille: Option<f64>,
    pub auroc: Option<f64>,
    pub average_precision: Option<f64>,
    pub full_holdout_auroc_lower_bound: Option<f64>,
    pub full_holdout_auroc_upper_bound: Option<f64>,
    pub conditional_fpr_permille: Option<f64>,
    pub conditional_tpr_permille: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DetectorAgreementSummaryV1 {
    pub detector_id: String,
    pub detector_family: String,
    pub detector_version: String,
    pub score_orientation: ScoreOrientationV1,
    pub threshold_fit_cohort_sha256: String,
    pub calibration_cohort_sha256: String,
    pub calibration_policy_sha256: String,
    pub threshold_fit_sample_count: u64,
    pub calibration_sample_count: u64,
    pub calibration_clean_count: u64,
    pub calibration_false_positive_count: u64,
    pub calibration_empirical_fpr_permille: f64,
    pub target_fpr_permille: u16,
    pub calibration_meets_target_empirically: bool,
    pub normalized_threshold: f64,
    pub sample_count: u64,
    pub clean_count: u64,
    pub contaminated_count: u64,
    pub indeterminate_truth_count: u64,
    pub scored_count: u64,
    pub abstained_count: u64,
    pub unavailable_count: u64,
    pub scored_clean_count: u64,
    pub scored_contaminated_count: u64,
    pub scored_indeterminate_count: u64,
    pub predicted_contaminated_count: u64,
    pub true_positive_count: u64,
    pub false_positive_count: u64,
    pub true_negative_count: u64,
    pub false_negative_count: u64,
    pub coverage_permille: f64,
    pub clean_coverage_permille: Option<f64>,
    pub contaminated_coverage_permille: Option<f64>,
    pub auroc: Option<f64>,
    pub average_precision: Option<f64>,
    /// Worst-case AUROC over all determinate holdout positives/negatives.
    pub full_holdout_auroc_lower_bound: Option<f64>,
    pub full_holdout_auroc_upper_bound: Option<f64>,
    /// FPR conditional on the detector returning a score for a clean item.
    pub conditional_fpr_permille: Option<f64>,
    /// TPR conditional on the detector returning a score for a contaminated item.
    pub conditional_tpr_permille: Option<f64>,
    /// Worst case: every clean item without a score is treated as a false positive.
    pub worst_case_fpr_permille: Option<f64>,
    /// Lower bound: every contaminated item without a score is treated as a false negative.
    pub tpr_lower_bound_permille: Option<f64>,
    /// Upper bound: every contaminated item without a score is treated as a true positive.
    pub tpr_upper_bound_permille: Option<f64>,
    pub independent_source_cluster_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairwiseContingencyV1 {
    pub left_positive_right_positive: u64,
    pub left_positive_right_negative: u64,
    pub left_negative_right_positive: u64,
    pub left_negative_right_negative: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PairwiseDetectorAgreementV1 {
    pub left_detector_id: String,
    pub right_detector_id: String,
    /// Common scored rows, including indeterminate ground-truth rows.
    pub common_scored_count: u64,
    pub spearman_rank_correlation: Option<f64>,
    /// Decision table is calculated only on common scored rows with determinate labels.
    pub common_labeled_count: u64,
    pub common_clean_count: u64,
    pub common_contaminated_count: u64,
    pub contingency: PairwiseContingencyV1,
    pub raw_decision_agreement_permille: Option<f64>,
    pub positive_agreement_permille: Option<f64>,
    pub negative_agreement_permille: Option<f64>,
    pub cohen_kappa: Option<f64>,
    pub decision_disagreement_permille: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DetectorAgreementReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_checkpoint_sha256: String,
    pub holdout_manifest_sha256: String,
    pub request_sha256: String,
    pub detector_count: u64,
    pub sample_count: u64,
    pub clean_count: u64,
    pub contaminated_count: u64,
    pub indeterminate_truth_count: u64,
    pub source_cluster_count: u64,
    pub detector_summaries: Vec<DetectorAgreementSummaryV1>,
    pub pairwise_agreement: Vec<PairwiseDetectorAgreementV1>,
    pub family_relation_strata: Vec<DetectorStratumReportV1>,
    pub threshold_selection_used_holdout_labels: bool,
    pub ensemble_fitted: bool,
    pub intervals_multiplicity_adjusted: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone)]
struct PreparedDetector {
    id: String,
    family: String,
    version: String,
    orientation: ScoreOrientationV1,
    calibration: FixedThresholdCalibrationV1,
}

#[derive(Debug, Clone, Copy)]
struct ObservedScore {
    normalized_score: f64,
    predicted_contaminated: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct StatusCounts {
    scored: u64,
    abstained: u64,
    unavailable: u64,
    scored_clean: u64,
    scored_contaminated: u64,
    scored_indeterminate: u64,
    predicted_contaminated: u64,
    true_positive: u64,
    false_positive: u64,
    true_negative: u64,
    false_negative: u64,
}

impl StatusCounts {
    fn total(self) -> u64 {
        self.scored + self.abstained + self.unavailable
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct StratumKey {
    family_key: String,
    difficulty_band: String,
    relation: TransferRelationV1,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn valid_opaque_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_OPAQUE_ID_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

fn increment(counter: &mut u64, label: &str) -> Result<(), String> {
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| format!("{label} count overflows"))?;
    Ok(())
}

fn normalize_score(raw_score: f64, orientation: ScoreOrientationV1) -> f64 {
    match orientation {
        ScoreOrientationV1::HigherIsContaminated => raw_score,
        ScoreOrientationV1::LowerIsContaminated => -raw_score,
    }
}

fn average_ranked(values: &[f64]) -> Result<Vec<f64>, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("rank input cannot be empty or non-finite".to_owned());
    }
    let mut order: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    order.sort_by(|left, right| left.1.total_cmp(&right.1));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && order[end].1 == order[start].1 {
            end += 1;
        }
        let rank = ((start + 1) as f64 + end as f64) / 2.0;
        for item in &order[start..end] {
            ranks[item.0] = rank;
        }
        start = end;
    }
    Ok(ranks)
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let count = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (a, b) in left.iter().zip(right) {
        let da = a - left_mean;
        let db = b - right_mean;
        covariance += da * db;
        left_variance += da * da;
        right_variance += db * db;
    }
    let denominator = (left_variance * right_variance).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        let correlation = covariance / denominator;
        correlation
            .is_finite()
            .then_some(correlation.clamp(-1.0, 1.0))
    }
}

fn spearman_correlation(left: &[f64], right: &[f64]) -> Result<Option<f64>, String> {
    if left.len() != right.len() || left.len() < 2 {
        return Ok(None);
    }
    let left_ranks = average_ranked(left)?;
    let right_ranks = average_ranked(right)?;
    Ok(pearson_correlation(&left_ranks, &right_ranks))
}

fn auc_and_average_precision(scores: &[(f64, bool)]) -> (Option<f64>, Option<f64>) {
    let positives = scores.iter().filter(|item| item.1).count();
    let negatives = scores.len().saturating_sub(positives);
    if positives == 0 || negatives == 0 {
        return (None, None);
    }

    let values: Vec<f64> = scores.iter().map(|item| item.0).collect();
    let positive_flags: Vec<bool> = scores.iter().map(|item| item.1).collect();
    let ranks = match average_ranked(&values) {
        Ok(ranks) => ranks,
        Err(_) => return (None, None),
    };
    let rank_sum = ranks
        .iter()
        .zip(&positive_flags)
        .filter_map(|(rank, positive)| positive.then_some(*rank))
        .sum::<f64>();
    let positive_n = positives as f64;
    let negative_n = negatives as f64;
    let auc = (rank_sum - positive_n * (positive_n + 1.0) / 2.0) / (positive_n * negative_n);

    let mut order: Vec<(f64, bool)> = scores.to_vec();
    order.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut seen = 0_u64;
    let mut true_positive = 0_u64;
    let mut average_precision = 0.0;
    let mut index = 0;
    while index < order.len() {
        let score = order[index].0;
        let mut end = index + 1;
        while end < order.len() && order[end].0 == score {
            end += 1;
        }
        let group_size = (end - index) as u64;
        let group_positive = order[index..end].iter().filter(|item| item.1).count() as u64;
        seen += group_size;
        true_positive += group_positive;
        if group_positive > 0 {
            let precision = true_positive as f64 / seen as f64;
            average_precision += precision * group_positive as f64 / positive_n;
        }
        index = end;
    }
    (
        Some(auc.clamp(0.0, 1.0)),
        Some(average_precision.clamp(0.0, 1.0)),
    )
}

fn full_holdout_auroc_bounds(
    conditional_auroc: Option<f64>,
    scored_positive: u64,
    scored_negative: u64,
    total_positive: u64,
    total_negative: u64,
) -> (Option<f64>, Option<f64>) {
    let total_pairs = total_positive as f64 * total_negative as f64;
    if total_pairs == 0.0 {
        return (None, None);
    }
    let scored_pairs = scored_positive as f64 * scored_negative as f64;
    let observed_concordance = conditional_auroc.unwrap_or(0.0) * scored_pairs;
    let unscored_pairs = (total_pairs - scored_pairs).max(0.0);
    let lower = observed_concordance / total_pairs;
    let upper = (observed_concordance + unscored_pairs) / total_pairs;
    (
        lower.is_finite().then_some(lower.clamp(0.0, 1.0)),
        upper.is_finite().then_some(upper.clamp(0.0, 1.0)),
    )
}

fn permille(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 * 1_000.0 / denominator as f64)
    }
}

fn validate_request(request: &DetectorAgreementRequestV1) -> Result<Vec<PreparedDetector>, String> {
    if request.schema != DETECTOR_AGREEMENT_REQUEST_SCHEMA {
        return Err(format!(
            "request schema must be {DETECTOR_AGREEMENT_REQUEST_SCHEMA}"
        ));
    }
    for (value, label) in [
        (&request.study_id, "study_id"),
        (&request.benchmark_id, "benchmark_id"),
        (&request.benchmark_version, "benchmark_version"),
    ] {
        if !valid_opaque_id(value) {
            return Err(format!(
                "{label} must be a nonempty opaque ASCII identifier"
            ));
        }
    }
    require_digest(&request.model_checkpoint_sha256, "model_checkpoint_sha256")?;
    require_digest(&request.holdout_manifest_sha256, "holdout_manifest_sha256")?;
    if !(MIN_COMPARATORS..=MAX_COMPARATORS).contains(&request.detector_specs.len()) {
        return Err(format!(
            "detector count must be in {MIN_COMPARATORS}..={MAX_COMPARATORS}"
        ));
    }
    if request.samples.is_empty() || request.samples.len() > MAX_HOLDOUT_SAMPLES {
        return Err(format!(
            "holdout sample count must be in 1..={MAX_HOLDOUT_SAMPLES}"
        ));
    }

    let mut detector_ids = BTreeSet::new();
    let mut prepared = Vec::with_capacity(request.detector_specs.len());
    for spec in &request.detector_specs {
        for (value, label) in [
            (&spec.detector_id, "detector_id"),
            (&spec.detector_family, "detector_family"),
            (&spec.detector_version, "detector_version"),
        ] {
            if !valid_opaque_id(value) {
                return Err(format!(
                    "{label} must be a nonempty opaque ASCII identifier"
                ));
            }
        }
        if !detector_ids.insert(spec.detector_id.as_str()) {
            return Err("detector IDs must be unique".to_owned());
        }
        require_digest(
            &spec.model_checkpoint_sha256,
            "detector model_checkpoint_sha256",
        )?;
        require_digest(&spec.feature_protocol_sha256, "feature_protocol_sha256")?;
        if spec.model_checkpoint_sha256 != request.model_checkpoint_sha256 {
            return Err("every detector must score the same declared model checkpoint".to_owned());
        }

        let calibration = &spec.calibration;
        require_digest(
            &calibration.threshold_fit_cohort_sha256,
            "threshold_fit_cohort_sha256",
        )?;
        require_digest(
            &calibration.calibration_cohort_sha256,
            "calibration_validation_cohort_sha256",
        )?;
        require_digest(
            &calibration.calibration_policy_sha256,
            "calibration_policy_sha256",
        )?;
        if calibration.threshold_fit_cohort_sha256 == calibration.calibration_cohort_sha256
            || calibration.threshold_fit_cohort_sha256 == request.holdout_manifest_sha256
            || calibration.calibration_cohort_sha256 == request.holdout_manifest_sha256
        {
            return Err("threshold-fit, calibration-validation and holdout manifests must have distinct digests".to_owned());
        }
        if calibration.threshold_fit_sample_count == 0
            || calibration.calibration_sample_count == 0
            || calibration.calibration_clean_count == 0
            || calibration.calibration_clean_count > calibration.calibration_sample_count
            || calibration.calibration_false_positive_count > calibration.calibration_clean_count
        {
            return Err("calibration cohort counts are inconsistent".to_owned());
        }
        if !(1..=500).contains(&calibration.target_fpr_permille) {
            return Err("target FPR must be in 1..=500 permille (0.1%..50%)".to_owned());
        }
        if !calibration.normalized_threshold.is_finite() {
            return Err("normalized calibration threshold must be finite".to_owned());
        }
        prepared.push(PreparedDetector {
            id: spec.detector_id.clone(),
            family: spec.detector_family.clone(),
            version: spec.detector_version.clone(),
            orientation: spec.score_orientation,
            calibration: calibration.clone(),
        });
    }

    let mut sample_ids = BTreeSet::new();
    for sample in &request.samples {
        for (value, label) in [
            (&sample.sample_id, "sample_id"),
            (&sample.task_family, "task_family"),
            (&sample.difficulty_band, "difficulty_band"),
            (&sample.source_cluster_id, "source_cluster_id"),
        ] {
            if !valid_opaque_id(value) {
                return Err(format!(
                    "{label} must be a nonempty opaque ASCII identifier"
                ));
            }
        }
        if !sample_ids.insert(sample.sample_id.as_str()) {
            return Err("sample IDs must be globally unique".to_owned());
        }
        if sample.detector_results.len() != prepared.len() {
            return Err("every sample must have an explicit status for every detector".to_owned());
        }
        let mut observed_ids = BTreeSet::new();
        for observation in &sample.detector_results {
            if !observed_ids.insert(observation.detector_id.as_str()) {
                return Err("sample contains duplicate detector observations".to_owned());
            }
            let detector = prepared
                .iter()
                .find(|item| item.id == observation.detector_id)
                .ok_or_else(|| {
                    "sample contains an observation for an undeclared detector".to_owned()
                })?;
            match &observation.result {
                DetectorScoreStatusV1::Scored {
                    raw_score,
                    score_evidence_sha256,
                    feature_protocol_sha256,
                } => {
                    if !raw_score.is_finite() {
                        return Err("scored detector values must be finite".to_owned());
                    }
                    require_digest(score_evidence_sha256, "score_evidence_sha256")?;
                    require_digest(feature_protocol_sha256, "score feature_protocol_sha256")?;
                    let spec = request
                        .detector_specs
                        .iter()
                        .find(|spec| spec.detector_id == detector.id)
                        .expect("prepared detector came from request detector_specs");
                    if feature_protocol_sha256 != &spec.feature_protocol_sha256 {
                        return Err(
                            "sample score protocol digest differs from detector specification"
                                .to_owned(),
                        );
                    }
                }
                DetectorScoreStatusV1::Abstained { reason_code }
                | DetectorScoreStatusV1::Unavailable { reason_code } => {
                    if !valid_opaque_id(reason_code) {
                        return Err(
                            "abstain/unavailable reason_code must be a nonempty opaque ID"
                                .to_owned(),
                        );
                    }
                }
            }
        }
        if observed_ids != detector_ids {
            return Err(
                "sample detector status set does not exactly match declared detector IDs"
                    .to_owned(),
            );
        }
    }
    Ok(prepared)
}

/// Analyze detector coverage, discrimination, operating characteristics and
/// agreement. All thresholds are supplied from external calibration artifacts;
/// no holdout label is consulted to choose or retune them.
pub fn analyze_detector_agreement(
    request: &DetectorAgreementRequestV1,
) -> Result<DetectorAgreementReportV1, String> {
    let detectors = validate_request(request)?;
    let request_bytes = serde_json::to_vec(request)
        .map_err(|error| format!("cannot serialize detector agreement request: {error}"))?;
    let request_sha256 = crate::hash_domain(b"detector-agreement-request.v1", &request_bytes);

    let detector_index: BTreeMap<&str, usize> = detectors
        .iter()
        .enumerate()
        .map(|(index, detector)| (detector.id.as_str(), index))
        .collect();
    let mut by_sample: Vec<Vec<Option<ObservedScore>>> =
        vec![vec![None; detectors.len()]; request.samples.len()];
    let mut status_counts = vec![StatusCounts::default(); detectors.len()];
    let mut truth_counts = [0_u64; 3];
    let mut clusters = BTreeSet::new();
    let mut stratum_samples = BTreeMap::<StratumKey, Vec<usize>>::new();

    for (sample_index, sample) in request.samples.iter().enumerate() {
        clusters.insert(sample.source_cluster_id.as_str());
        match sample.truth {
            GroundTruthLabelV1::Clean => increment(&mut truth_counts[0], "clean")?,
            GroundTruthLabelV1::Contaminated => increment(&mut truth_counts[1], "contaminated")?,
            GroundTruthLabelV1::Indeterminate => increment(&mut truth_counts[2], "indeterminate")?,
        }
        let key = StratumKey {
            family_key: sample.task_family.clone(),
            difficulty_band: sample.difficulty_band.clone(),
            relation: sample.relation,
        };
        stratum_samples.entry(key).or_default().push(sample_index);

        for observation in &sample.detector_results {
            let index = *detector_index
                .get(observation.detector_id.as_str())
                .expect("detector IDs validated above");
            match &observation.result {
                DetectorScoreStatusV1::Scored { raw_score, .. } => {
                    let normalized = normalize_score(*raw_score, detectors[index].orientation);
                    let predicted = normalized >= detectors[index].calibration.normalized_threshold;
                    by_sample[sample_index][index] = Some(ObservedScore {
                        normalized_score: normalized,
                        predicted_contaminated: predicted,
                    });
                    increment(&mut status_counts[index].scored, "scored")?;
                    match sample.truth {
                        GroundTruthLabelV1::Clean => {
                            increment(&mut status_counts[index].scored_clean, "scored clean")?;
                            if predicted {
                                increment(
                                    &mut status_counts[index].false_positive,
                                    "false positive",
                                )?;
                            } else {
                                increment(
                                    &mut status_counts[index].true_negative,
                                    "true negative",
                                )?;
                            }
                        }
                        GroundTruthLabelV1::Contaminated => {
                            increment(
                                &mut status_counts[index].scored_contaminated,
                                "scored contaminated",
                            )?;
                            if predicted {
                                increment(
                                    &mut status_counts[index].true_positive,
                                    "true positive",
                                )?;
                            } else {
                                increment(
                                    &mut status_counts[index].false_negative,
                                    "false negative",
                                )?;
                            }
                        }
                        GroundTruthLabelV1::Indeterminate => {
                            increment(
                                &mut status_counts[index].scored_indeterminate,
                                "scored indeterminate",
                            )?;
                        }
                    }
                    if predicted {
                        increment(
                            &mut status_counts[index].predicted_contaminated,
                            "predicted contaminated",
                        )?;
                    }
                }
                DetectorScoreStatusV1::Abstained { .. } => {
                    increment(&mut status_counts[index].abstained, "abstained")?;
                }
                DetectorScoreStatusV1::Unavailable { .. } => {
                    increment(&mut status_counts[index].unavailable, "unavailable")?;
                }
            }
        }
    }

    let mut detector_summaries = Vec::with_capacity(detectors.len());
    for (index, detector) in detectors.iter().enumerate() {
        let counts = status_counts[index];
        let clean = truth_counts[0];
        let contaminated = truth_counts[1];
        let calibration_fpr = permille(
            detector.calibration.calibration_false_positive_count,
            detector.calibration.calibration_clean_count,
        )
        .expect("calibration clean count validated");
        let scored_scores: Vec<(f64, bool)> = request
            .samples
            .iter()
            .enumerate()
            .filter_map(|(sample_index, sample)| {
                by_sample[sample_index][index]
                    .map(|score| match sample.truth {
                        GroundTruthLabelV1::Contaminated => (score.normalized_score, true),
                        GroundTruthLabelV1::Clean => (score.normalized_score, false),
                        GroundTruthLabelV1::Indeterminate => (score.normalized_score, false),
                    })
                    .filter(|_| sample.truth != GroundTruthLabelV1::Indeterminate)
            })
            .collect();
        let (auroc, average_precision) = auc_and_average_precision(&scored_scores);
        let (full_holdout_auroc_lower_bound, full_holdout_auroc_upper_bound) =
            full_holdout_auroc_bounds(
                auroc,
                counts.scored_contaminated,
                counts.scored_clean,
                contaminated,
                clean,
            );
        let missing_clean = clean.saturating_sub(counts.scored_clean);
        let missing_contaminated = contaminated.saturating_sub(counts.scored_contaminated);

        detector_summaries.push(DetectorAgreementSummaryV1 {
            detector_id: detector.id.clone(),
            detector_family: detector.family.clone(),
            detector_version: detector.version.clone(),
            score_orientation: detector.orientation,
            threshold_fit_cohort_sha256: detector.calibration.threshold_fit_cohort_sha256.clone(),
            calibration_cohort_sha256: detector.calibration.calibration_cohort_sha256.clone(),
            calibration_policy_sha256: detector.calibration.calibration_policy_sha256.clone(),
            threshold_fit_sample_count: detector.calibration.threshold_fit_sample_count,
            calibration_sample_count: detector.calibration.calibration_sample_count,
            calibration_clean_count: detector.calibration.calibration_clean_count,
            calibration_false_positive_count: detector.calibration.calibration_false_positive_count,
            calibration_empirical_fpr_permille: calibration_fpr,
            target_fpr_permille: detector.calibration.target_fpr_permille,
            calibration_meets_target_empirically: calibration_fpr
                <= f64::from(detector.calibration.target_fpr_permille),
            normalized_threshold: detector.calibration.normalized_threshold,
            sample_count: request.samples.len() as u64,
            clean_count: clean,
            contaminated_count: contaminated,
            indeterminate_truth_count: truth_counts[2],
            scored_count: counts.scored,
            abstained_count: counts.abstained,
            unavailable_count: counts.unavailable,
            scored_clean_count: counts.scored_clean,
            scored_contaminated_count: counts.scored_contaminated,
            scored_indeterminate_count: counts.scored_indeterminate,
            predicted_contaminated_count: counts.predicted_contaminated,
            true_positive_count: counts.true_positive,
            false_positive_count: counts.false_positive,
            true_negative_count: counts.true_negative,
            false_negative_count: counts.false_negative,
            coverage_permille: permille(counts.scored, counts.total())
                .expect("each detector has one status for each sample"),
            clean_coverage_permille: permille(counts.scored_clean, clean),
            contaminated_coverage_permille: permille(counts.scored_contaminated, contaminated),
            auroc,
            average_precision,
            full_holdout_auroc_lower_bound,
            full_holdout_auroc_upper_bound,
            conditional_fpr_permille: permille(counts.false_positive, counts.scored_clean),
            conditional_tpr_permille: permille(counts.true_positive, counts.scored_contaminated),
            worst_case_fpr_permille: permille(counts.false_positive + missing_clean, clean),
            tpr_lower_bound_permille: permille(counts.true_positive, contaminated),
            tpr_upper_bound_permille: permille(
                counts.true_positive + missing_contaminated,
                contaminated,
            ),
            independent_source_cluster_count: clusters.len() as u64,
        });
    }

    let mut pairwise_agreement = Vec::new();
    for (left_index, left_detector) in detectors.iter().enumerate() {
        for (right_index, right_detector) in detectors.iter().enumerate().skip(left_index + 1) {
            let mut left_scores = Vec::new();
            let mut right_scores = Vec::new();
            let mut contingency = PairwiseContingencyV1 {
                left_positive_right_positive: 0,
                left_positive_right_negative: 0,
                left_negative_right_positive: 0,
                left_negative_right_negative: 0,
            };
            let mut common_labeled_count = 0_u64;
            let mut common_clean_count = 0_u64;
            let mut common_contaminated_count = 0_u64;

            for (sample_index, sample) in request.samples.iter().enumerate() {
                let (Some(left), Some(right)) = (
                    by_sample[sample_index][left_index],
                    by_sample[sample_index][right_index],
                ) else {
                    continue;
                };
                left_scores.push(left.normalized_score);
                right_scores.push(right.normalized_score);
                if sample.truth == GroundTruthLabelV1::Indeterminate {
                    continue;
                }
                increment(&mut common_labeled_count, "pairwise common labeled")?;
                match sample.truth {
                    GroundTruthLabelV1::Clean => {
                        increment(&mut common_clean_count, "pairwise common clean")?
                    }
                    GroundTruthLabelV1::Contaminated => increment(
                        &mut common_contaminated_count,
                        "pairwise common contaminated",
                    )?,
                    GroundTruthLabelV1::Indeterminate => {}
                }
                match (left.predicted_contaminated, right.predicted_contaminated) {
                    (true, true) => increment(
                        &mut contingency.left_positive_right_positive,
                        "pairwise both positive",
                    )?,
                    (true, false) => increment(
                        &mut contingency.left_positive_right_negative,
                        "pairwise left-only positive",
                    )?,
                    (false, true) => increment(
                        &mut contingency.left_negative_right_positive,
                        "pairwise right-only positive",
                    )?,
                    (false, false) => increment(
                        &mut contingency.left_negative_right_negative,
                        "pairwise both negative",
                    )?,
                }
            }

            let both_positive = contingency.left_positive_right_positive as f64;
            let left_only = contingency.left_positive_right_negative as f64;
            let right_only = contingency.left_negative_right_positive as f64;
            let both_negative = contingency.left_negative_right_negative as f64;
            let discordant = left_only + right_only;
            let total = both_positive + left_only + right_only + both_negative;
            let raw_agreement = if total > 0.0 {
                Some((both_positive + both_negative) * 1_000.0 / total)
            } else {
                None
            };
            let positive_denominator = 2.0 * both_positive + left_only + right_only;
            let negative_denominator = 2.0 * both_negative + left_only + right_only;
            let positive_agreement = (positive_denominator > 0.0)
                .then_some(2.0 * both_positive * 1_000.0 / positive_denominator);
            let negative_agreement = (negative_denominator > 0.0)
                .then_some(2.0 * both_negative * 1_000.0 / negative_denominator);
            let left_positive_marginal = both_positive + left_only;
            let right_positive_marginal = both_positive + right_only;
            let expected_agreement = if total > 0.0 {
                ((left_positive_marginal * right_positive_marginal)
                    + ((total - left_positive_marginal) * (total - right_positive_marginal)))
                    / (total * total)
            } else {
                0.0
            };
            let observed_agreement = if total > 0.0 {
                (both_positive + both_negative) / total
            } else {
                0.0
            };
            let kappa = if total > 0.0 && (1.0 - expected_agreement).abs() > f64::EPSILON {
                Some((observed_agreement - expected_agreement) / (1.0 - expected_agreement))
            } else {
                None
            };

            pairwise_agreement.push(PairwiseDetectorAgreementV1 {
                left_detector_id: left_detector.id.clone(),
                right_detector_id: right_detector.id.clone(),
                common_scored_count: left_scores.len() as u64,
                spearman_rank_correlation: spearman_correlation(&left_scores, &right_scores)?,
                common_labeled_count,
                common_clean_count,
                common_contaminated_count,
                contingency,
                raw_decision_agreement_permille: raw_agreement,
                positive_agreement_permille: positive_agreement,
                negative_agreement_permille: negative_agreement,
                cohen_kappa: kappa,
                decision_disagreement_permille: if total > 0.0 {
                    Some(discordant * 1_000.0 / total)
                } else {
                    None
                },
            });
        }
    }

    let mut family_relation_strata = Vec::new();
    for (key, sample_indices) in &stratum_samples {
        for (detector_index, detector) in detectors.iter().enumerate() {
            let mut clean_count = 0_u64;
            let mut contaminated_count = 0_u64;
            let mut indeterminate_count = 0_u64;
            let mut scored_count = 0_u64;
            let mut abstained_count = 0_u64;
            let mut unavailable_count = 0_u64;
            let mut scored_clean_count = 0_u64;
            let mut scored_contaminated_count = 0_u64;
            let mut false_positive = 0_u64;
            let mut true_positive = 0_u64;
            let mut scored_scores = Vec::new();

            for sample_index in sample_indices {
                let sample = &request.samples[*sample_index];
                match sample.truth {
                    GroundTruthLabelV1::Clean => increment(&mut clean_count, "stratum clean")?,
                    GroundTruthLabelV1::Contaminated => {
                        increment(&mut contaminated_count, "stratum contaminated")?
                    }
                    GroundTruthLabelV1::Indeterminate => {
                        increment(&mut indeterminate_count, "stratum indeterminate")?
                    }
                }
                match by_sample[*sample_index][detector_index] {
                    Some(score) => {
                        increment(&mut scored_count, "stratum scored")?;
                        if sample.truth == GroundTruthLabelV1::Clean {
                            increment(&mut scored_clean_count, "stratum scored clean")?;
                            if score.predicted_contaminated {
                                increment(&mut false_positive, "stratum false positive")?;
                            }
                        } else if sample.truth == GroundTruthLabelV1::Contaminated {
                            increment(
                                &mut scored_contaminated_count,
                                "stratum scored contaminated",
                            )?;
                            if score.predicted_contaminated {
                                increment(&mut true_positive, "stratum true positive")?;
                            }
                        }
                        if sample.truth != GroundTruthLabelV1::Indeterminate {
                            scored_scores.push((
                                score.normalized_score,
                                sample.truth == GroundTruthLabelV1::Contaminated,
                            ));
                        }
                    }
                    None => {
                        let observation = request.samples[*sample_index]
                            .detector_results
                            .iter()
                            .find(|item| item.detector_id == detector.id)
                            .expect("complete matrix validated above");
                        match &observation.result {
                            DetectorScoreStatusV1::Abstained { .. } => {
                                increment(&mut abstained_count, "stratum abstained")?
                            }
                            DetectorScoreStatusV1::Unavailable { .. } => {
                                increment(&mut unavailable_count, "stratum unavailable")?
                            }
                            DetectorScoreStatusV1::Scored { .. } => {
                                return Err(
                                    "scored observation missing from normalized score matrix"
                                        .to_owned(),
                                );
                            }
                        }
                    }
                }
            }
            let (auroc, average_precision) = auc_and_average_precision(&scored_scores);
            let (full_holdout_auroc_lower_bound, full_holdout_auroc_upper_bound) =
                full_holdout_auroc_bounds(
                    auroc,
                    scored_contaminated_count,
                    scored_clean_count,
                    contaminated_count,
                    clean_count,
                );
            let total_count = sample_indices.len() as u64;
            family_relation_strata.push(DetectorStratumReportV1 {
                detector_id: detector.id.clone(),
                task_family: key.family_key.clone(),
                difficulty_band: key.difficulty_band.clone(),
                relation: key.relation,
                sample_count: total_count,
                clean_count,
                contaminated_count,
                indeterminate_truth_count: indeterminate_count,
                scored_count,
                abstained_count,
                unavailable_count,
                scored_clean_count,
                scored_contaminated_count,
                coverage_permille: permille(scored_count, total_count).unwrap_or(0.0),
                clean_coverage_permille: permille(scored_clean_count, clean_count),
                contaminated_coverage_permille: permille(
                    scored_contaminated_count,
                    contaminated_count,
                ),
                auroc,
                average_precision,
                full_holdout_auroc_lower_bound,
                full_holdout_auroc_upper_bound,
                conditional_fpr_permille: permille(false_positive, scored_clean_count),
                conditional_tpr_permille: permille(true_positive, scored_contaminated_count),
            });
        }
    }

    let issues = vec![
        "conditional AUROC/AP exclude indeterminate labels and are calculated on each detector's scored subset; class-conditional coverage is reported separately".to_owned(),
        "worst-case FPR treats every clean abstention/unavailable score as a false positive; TPR bounds treat missing contaminated scores as false negative or true positive".to_owned(),
        "95% confidence intervals and multiplicity adjustment are not provided; candidate items may be clustered and are not assumed independent".to_owned(),
        "calibration/holdout digest inequality is a declared separation check, not proof that the underlying sample sets are disjoint".to_owned(),
        "rank correlation and threshold agreement are descriptive complementarity measures, not a detector ensemble or evidence of causal contamination".to_owned(),
    ];

    Ok(DetectorAgreementReportV1 {
        schema: DETECTOR_AGREEMENT_REPORT_SCHEMA.to_owned(),
        study_id: request.study_id.clone(),
        benchmark_id: request.benchmark_id.clone(),
        benchmark_version: request.benchmark_version.clone(),
        model_checkpoint_sha256: request.model_checkpoint_sha256.clone(),
        holdout_manifest_sha256: request.holdout_manifest_sha256.clone(),
        request_sha256,
        detector_count: detectors.len() as u64,
        sample_count: request.samples.len() as u64,
        clean_count: truth_counts[0],
        contaminated_count: truth_counts[1],
        indeterminate_truth_count: truth_counts[2],
        source_cluster_count: clusters.len() as u64,
        detector_summaries,
        pairwise_agreement,
        family_relation_strata,
        threshold_selection_used_holdout_labels: false,
        ensemble_fitted: false,
        intervals_multiplicity_adjusted: false,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: "exploratory_detector_complementarity".to_owned(),
        issues,
        claim_boundary: "This report compares fixed detector scores and thresholds on a caller-declared holdout cohort. It requires an explicit scored/abstained/unavailable status for every sample-detector pair, reports class-conditional coverage and worst-case operating-characteristic bounds, and computes pairwise rank/decision agreement on identical scored examples. Calibration threshold digests are required to differ from the holdout manifest digest, but that does not prove sample disjointness or authenticate calibration. AUROC/AP are conditional on scored samples with determinate labels. This module does not fit an ensemble, establish statistical guarantees, prove training membership, establish that exposure caused an answer, authenticate any model/evaluator receipt, or demonstrate external validity or SOTA superiority.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn detector(
        id: &str,
        orientation: ScoreOrientationV1,
        threshold: f64,
        protocol: char,
    ) -> DetectorAgreementSpecV1 {
        DetectorAgreementSpecV1 {
            detector_id: id.to_owned(),
            detector_family: "output_level".to_owned(),
            detector_version: "v1".to_owned(),
            model_checkpoint_sha256: digest('a'),
            feature_protocol_sha256: digest(protocol),
            score_orientation: orientation,
            calibration: FixedThresholdCalibrationV1 {
                threshold_fit_cohort_sha256: digest(if protocol == '1' { '6' } else { '7' }),
                calibration_cohort_sha256: digest(if protocol == '1' { '2' } else { '3' }),
                calibration_policy_sha256: digest('4'),
                threshold_fit_sample_count: 800,
                calibration_sample_count: 400,
                calibration_clean_count: 300,
                calibration_false_positive_count: 1,
                target_fpr_permille: 10,
                normalized_threshold: threshold,
            },
        }
    }

    fn score(detector_id: &str, raw: f64, protocol: char) -> DetectorScoreObservationV1 {
        DetectorScoreObservationV1 {
            detector_id: detector_id.to_owned(),
            result: DetectorScoreStatusV1::Scored {
                raw_score: raw,
                score_evidence_sha256: digest('5'),
                feature_protocol_sha256: digest(protocol),
            },
        }
    }

    fn sample(
        id: &str,
        truth: GroundTruthLabelV1,
        left: DetectorScoreObservationV1,
        right: DetectorScoreObservationV1,
    ) -> DetectorAgreementSampleV1 {
        DetectorAgreementSampleV1 {
            sample_id: id.to_owned(),
            task_family: "math".to_owned(),
            difficulty_band: "medium".to_owned(),
            relation: TransferRelationV1::SemanticEquivalent,
            source_cluster_id: format!("cluster-{id}"),
            truth,
            detector_results: vec![left, right],
        }
    }

    fn request() -> DetectorAgreementRequestV1 {
        DetectorAgreementRequestV1 {
            schema: DETECTOR_AGREEMENT_REQUEST_SCHEMA.to_owned(),
            study_id: "study-001".to_owned(),
            benchmark_id: "benchmark-001".to_owned(),
            benchmark_version: "v1".to_owned(),
            model_checkpoint_sha256: digest('a'),
            holdout_manifest_sha256: digest('9'),
            detector_specs: vec![
                detector(
                    "detector-a",
                    ScoreOrientationV1::HigherIsContaminated,
                    0.5,
                    '1',
                ),
                detector(
                    "detector-b",
                    ScoreOrientationV1::LowerIsContaminated,
                    0.5,
                    '2',
                ),
            ],
            samples: vec![
                sample(
                    "clean-1",
                    GroundTruthLabelV1::Clean,
                    score("detector-a", 0.1, '1'),
                    score("detector-b", -0.1, '2'),
                ),
                sample(
                    "clean-2",
                    GroundTruthLabelV1::Clean,
                    score("detector-a", 0.2, '1'),
                    score("detector-b", -0.2, '2'),
                ),
                sample(
                    "contam-1",
                    GroundTruthLabelV1::Contaminated,
                    score("detector-a", 0.9, '1'),
                    score("detector-b", -0.9, '2'),
                ),
                sample(
                    "contam-2",
                    GroundTruthLabelV1::Contaminated,
                    score("detector-a", 0.8, '1'),
                    score("detector-b", -0.8, '2'),
                ),
            ],
        }
    }

    #[test]
    fn hand_computable_json_fixture_round_trips_and_scores() {
        let request: DetectorAgreementRequestV1 = serde_json::from_str(include_str!(
            "../fixtures/detector-agreement-request.hand.v1.json"
        ))
        .unwrap();
        let report = analyze_detector_agreement(&request).unwrap();
        assert_eq!(report.sample_count, 4);
        assert_eq!(report.detector_count, 2);
        assert_eq!(
            report.pairwise_agreement[0].spearman_rank_correlation,
            Some(1.0)
        );
        assert_eq!(report.detector_summaries[0].auroc, Some(1.0));
        assert!(!report.ensemble_fitted);
    }

    #[test]
    fn perfectly_ordered_detectors_have_auc_one_and_spearman_one() {
        let report = analyze_detector_agreement(&request()).unwrap();
        assert_eq!(report.detector_summaries[0].auroc, Some(1.0));
        assert_eq!(report.detector_summaries[1].auroc, Some(1.0));
        assert_eq!(
            report.pairwise_agreement[0].spearman_rank_correlation,
            Some(1.0)
        );
    }

    #[test]
    fn inverted_raw_direction_is_normalized_before_analysis() {
        let report = analyze_detector_agreement(&request()).unwrap();
        assert_eq!(
            report.detector_summaries[0].true_positive_count,
            report.detector_summaries[1].true_positive_count
        );
        assert_eq!(
            report.pairwise_agreement[0].decision_disagreement_permille,
            Some(0.0)
        );
    }

    #[test]
    fn abstentions_are_included_in_coverage_and_worst_case_bounds() {
        let mut request = request();
        request.samples[0].detector_results[0].result = DetectorScoreStatusV1::Abstained {
            reason_code: "no_hidden_state".to_owned(),
        };
        let report = analyze_detector_agreement(&request).unwrap();
        let summary = &report.detector_summaries[0];
        assert_eq!(summary.abstained_count, 1);
        assert_eq!(summary.coverage_permille, 750.0);
        assert_eq!(summary.worst_case_fpr_permille, Some(500.0));
    }

    #[test]
    fn full_holdout_auc_bounds_widen_when_a_contaminated_score_is_missing() {
        let mut request = request();
        request.samples[2].detector_results[0].result = DetectorScoreStatusV1::Unavailable {
            reason_code: "backend_timeout".to_owned(),
        };
        let report = analyze_detector_agreement(&request).unwrap();
        let summary = &report.detector_summaries[0];
        assert_eq!(summary.auroc, Some(1.0));
        assert_eq!(summary.full_holdout_auroc_lower_bound, Some(0.5));
        assert_eq!(summary.full_holdout_auroc_upper_bound, Some(1.0));
    }

    #[test]
    fn unavailable_contaminated_score_widens_tpr_bounds() {
        let mut request = request();
        request.samples[2].detector_results[0].result = DetectorScoreStatusV1::Unavailable {
            reason_code: "backend_timeout".to_owned(),
        };
        let report = analyze_detector_agreement(&request).unwrap();
        let summary = &report.detector_summaries[0];
        assert_eq!(summary.tpr_lower_bound_permille, Some(500.0));
        assert_eq!(summary.tpr_upper_bound_permille, Some(1_000.0));
    }

    #[test]
    fn threshold_fit_and_validation_cohorts_must_be_distinct() {
        let mut request = request();
        request.detector_specs[0]
            .calibration
            .threshold_fit_cohort_sha256 = request.detector_specs[0]
            .calibration
            .calibration_cohort_sha256
            .clone();
        assert!(analyze_detector_agreement(&request).is_err());
    }

    #[test]
    fn holdout_calibration_digest_reuse_is_rejected() {
        let mut request = request();
        request.detector_specs[0]
            .calibration
            .calibration_cohort_sha256 = request.holdout_manifest_sha256.clone();
        assert!(analyze_detector_agreement(&request).is_err());
    }

    #[test]
    fn missing_detector_status_is_rejected() {
        let mut request = request();
        request.samples[0].detector_results.pop();
        assert!(analyze_detector_agreement(&request).is_err());
    }

    #[test]
    fn duplicate_sample_detector_status_is_rejected() {
        let mut request = request();
        request.samples[0].detector_results[1].detector_id = "detector-a".to_owned();
        assert!(analyze_detector_agreement(&request).is_err());
    }

    #[test]
    fn protocol_digest_mismatch_is_rejected() {
        let mut request = request();
        if let DetectorScoreStatusV1::Scored {
            feature_protocol_sha256,
            ..
        } = &mut request.samples[0].detector_results[0].result
        {
            *feature_protocol_sha256 = digest('f');
        }
        assert!(analyze_detector_agreement(&request).is_err());
    }

    #[test]
    fn indeterminate_truth_is_not_silently_counted_as_clean() {
        let mut request = request();
        request.samples[0].truth = GroundTruthLabelV1::Indeterminate;
        let report = analyze_detector_agreement(&request).unwrap();
        let summary = &report.detector_summaries[0];
        assert_eq!(summary.indeterminate_truth_count, 1);
        assert_eq!(summary.clean_count, 1);
        assert_eq!(summary.scored_indeterminate_count, 1);
    }

    #[test]
    fn tied_scores_produce_half_auc_and_finite_average_precision() {
        let values = vec![(0.5, true), (0.5, false), (0.5, true), (0.5, false)];
        let (auc, ap) = auc_and_average_precision(&values);
        assert_eq!(auc, Some(0.5));
        assert_eq!(ap, Some(0.5));
    }

    #[test]
    fn zero_variance_rank_inputs_return_no_spearman_value() {
        assert_eq!(
            spearman_correlation(&[1.0, 1.0], &[0.1, 0.2]).unwrap(),
            None
        );
    }

    #[test]
    fn pairwise_agreement_separates_positive_and_negative_agreement() {
        let report = analyze_detector_agreement(&request()).unwrap();
        let pair = &report.pairwise_agreement[0];
        assert_eq!(pair.positive_agreement_permille, Some(1_000.0));
        assert_eq!(pair.negative_agreement_permille, Some(1_000.0));
        assert_eq!(pair.cohen_kappa, Some(1.0));
    }

    #[test]
    fn pairwise_disagreement_is_reported_on_shared_examples() {
        let mut request = request();
        request.samples[0].detector_results[1] = score("detector-b", -0.9, '2');
        let report = analyze_detector_agreement(&request).unwrap();
        let pair = &report.pairwise_agreement[0];
        assert_eq!(pair.common_scored_count, 4);
        assert!(pair.decision_disagreement_permille.unwrap() > 0.0);
    }

    #[test]
    fn report_never_fits_or_authenticates_an_ensemble() {
        let report = analyze_detector_agreement(&request()).unwrap();
        assert!(!report.ensemble_fitted);
        assert!(!report.independently_authenticated);
        assert!(!report.confirmatory_eligible);
        assert!(!report.threshold_selection_used_holdout_labels);
    }

    #[test]
    fn request_ordering_does_not_break_status_lookup() {
        let mut request = request();
        request.samples[0].detector_results.reverse();
        let report = analyze_detector_agreement(&request).unwrap();
        assert_eq!(report.detector_summaries.len(), 2);
    }
}
