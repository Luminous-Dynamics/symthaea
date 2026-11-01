//! Distribution-free calibration and low-FPR operating-point audit for detector panels.
//!
//! Each test score is compared with a disjoint, group-matched clean calibration
//! cohort. The upper-tail conformal p-value is (1 + #calibration_scores >=
//! test_risk_score) / (n_clean + 1). Ties are included and score direction is
//! normalized so larger values always mean greater contamination risk.
//!
//! The finite-sample super-uniform guarantee requires exchangeability of clean
//! calibration and clean test observations within the exact declared group.
//! This module refuses to silently pool across checkpoint, task, language,
//! exposure stage or detector-probe configuration.

use crate::hash_domain;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CALIBRATION_STUDY_SCHEMA: &str = "symthaea.detector_calibration_study.v1";
pub const CALIBRATION_REPORT_SCHEMA: &str = "symthaea.detector_calibration_report.v1";
pub const CALIBRATION_GROUP_SCHEMA: &str = "symthaea.detector_calibration_group.v1";
pub const MAX_CALIBRATION_OBSERVATIONS: usize = 2_000_000;
pub const MAX_CALIBRATION_GROUPS: usize = 10_000;
pub const MAX_DETECTORS: usize = 128;
pub const MAX_TARGETS: usize = 16;
pub const MAX_ID_LENGTH: usize = 128;
pub const DEFAULT_FPR_TARGETS_PERMILLE: [u16; 3] = [10, 50, 100];

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ScoreDirectionV1 {
    HigherMeansContaminated,
    LowerMeansContaminated,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ExposureStageV1 {
    Pretraining,
    SupervisedFineTuning,
    ReinforcementLearningVerifiableReward,
    ReinforcementLearningHumanFeedback,
    SearchTime,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ObservationSplitV1 {
    CleanCalibration,
    LockedTest,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum GroundTruthV1 {
    Clean,
    Contaminated,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationGroupIdentityV1 {
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub task_family: String,
    pub language: String,
    pub exposure_stage: ExposureStageV1,
    pub probe_config_sha256: String,
    /// One independently trained treatment/placebo model-pair replicate.
    pub training_replicate_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationGroupV1 {
    pub schema: String,
    pub group_sha256: String,
    pub identity: CalibrationGroupIdentityV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorDefinitionV1 {
    pub detector_id: String,
    pub detector_version: String,
    pub score_direction: ScoreDirectionV1,
    pub score_semantics: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorObservationV1 {
    pub sample_id: String,
    pub source_cluster_id: String,
    pub calibration_group_sha256: String,
    pub split: ObservationSplitV1,
    pub ground_truth: GroundTruthV1,
    pub detector_id: String,
    pub detector_version: String,
    /// Original detector score, before direction normalization.
    pub raw_score: f64,
    pub inference_calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub wall_time_micros: u64,
    pub peak_memory_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorCalibrationStudyV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub target_fpr_permille: Vec<u16>,
    pub detector_definitions: Vec<DetectorDefinitionV1>,
    pub calibration_groups: Vec<CalibrationGroupV1>,
    pub observations: Vec<DetectorObservationV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConformalPValueV1 {
    /// Exact upper-tail p-value numerator: 1 + number of clean scores >= test score.
    pub numerator: u64,
    /// Exact p-value denominator: n_clean + 1.
    pub denominator: u64,
    /// Ceiling of the rational p-value scaled to permille. Display only.
    pub ceiling_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibratedTestScoreV1 {
    pub calibration_group_sha256: String,
    pub sample_id: String,
    pub source_cluster_id: String,
    pub detector_id: String,
    pub detector_version: String,
    pub ground_truth: GroundTruthV1,
    pub raw_score: f64,
    pub normalized_risk_score: f64,
    pub calibration_clean_support: u64,
    pub conformal_p_value: ConformalPValueV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorCostSummaryV1 {
    pub calibration_group_sha256: String,
    pub detector_id: String,
    pub detector_version: String,
    pub test_observation_count: u64,
    pub total_inference_calls: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub mean_wall_time_micros: f64,
    pub p50_wall_time_micros: u64,
    pub p95_wall_time_micros: u64,
    pub maximum_peak_memory_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OperatingPointMetricsV1 {
    pub target_fpr_permille: u16,
    pub calibration_clean_support: u64,
    pub minimum_attainable_p_value: ConformalPValueV1,
    pub calibration_resolution_sufficient: bool,
    pub test_clean_support: u64,
    pub test_contaminated_support: u64,
    pub false_positive: Option<u64>,
    pub true_positive: Option<u64>,
    pub false_negative: Option<u64>,
    pub true_negative: Option<u64>,
    pub false_positive_rate: Option<f64>,
    pub false_positive_rate_upper_95_one_sided: Option<f64>,
    pub true_positive_rate: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorCalibrationCellV1 {
    pub calibration_group_sha256: String,
    pub task_family: String,
    pub language: String,
    pub exposure_stage: ExposureStageV1,
    pub training_replicate_id: String,
    pub detector_id: String,
    pub detector_version: String,
    pub calibration_clean_support: u64,
    pub locked_test_support: u64,
    pub operating_points: Vec<OperatingPointMetricsV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorDisagreementCellV1 {
    pub calibration_group_sha256: String,
    pub target_fpr_permille: u16,
    pub detector_left: String,
    pub detector_right: String,
    pub comparable_test_support: u64,
    pub disagree_count: u64,
    pub both_flag_contaminated: u64,
    pub left_only_flag_contaminated: u64,
    pub right_only_flag_contaminated: u64,
    pub neither_flag_contaminated: u64,
    pub disagreement_rate: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorMacroCellV1 {
    pub detector_id: String,
    pub detector_version: String,
    pub target_fpr_permille: u16,
    pub eligible_calibration_group_count: u64,
    pub mean_test_false_positive_rate: Option<f64>,
    pub worst_group_fpr_upper_95_one_sided: Option<f64>,
    pub mean_test_true_positive_rate: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorCalibrationReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub study_sha256: String,
    pub calibration_group_count: u64,
    pub detector_count: u64,
    pub observation_count: u64,
    pub scored_locked_test_observation_count: u64,
    pub group_cells: Vec<DetectorCalibrationCellV1>,
    pub scored_test_p_values: Vec<CalibratedTestScoreV1>,
    pub disagreement_matrix: Vec<DetectorDisagreementCellV1>,
    pub detector_costs: Vec<DetectorCostSummaryV1>,
    pub macro_cells: Vec<DetectorMacroCellV1>,
    pub intervals_multiplicity_adjusted: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
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

fn valid_id(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty()
        || value.len() > MAX_ID_LENGTH
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-/".contains(&byte))
    {
        Err(format!("{label} must be a nonempty opaque ASCII identifier <= {MAX_ID_LENGTH} bytes"))
    } else {
        Ok(())
    }
}

fn canonical_json<T: Serialize>(value: &T, label: &str) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|error| format!("cannot serialize {label}: {error}"))
}

pub fn compute_calibration_group_sha256(
    identity: &CalibrationGroupIdentityV1,
) -> Result<String, String> {
    let bytes = canonical_json(identity, "calibration group identity")?;
    Ok(hash_domain(b"detector-calibration-group.v1", &bytes))
}

fn normalize_risk(score: f64, direction: ScoreDirectionV1) -> Result<f64, String> {
    if !score.is_finite() {
        return Err("detector score must be finite".to_owned());
    }
    let risk = match direction {
        ScoreDirectionV1::HigherMeansContaminated => score,
        ScoreDirectionV1::LowerMeansContaminated => -score,
    };
    if risk.is_finite() {
        Ok(risk)
    } else {
        Err("normalized detector risk score is non-finite".to_owned())
    }
}

fn minimum_calibration_support(target_fpr_permille: u16) -> Result<u64, String> {
    if !(1..=1_000).contains(&target_fpr_permille) {
        return Err("target FPR must be in [1, 1000] permille".to_owned());
    }
    let numerator = 1_000_u64;
    let denominator = u64::from(target_fpr_permille);
    let required_denominator = numerator.div_ceil(denominator);
    Ok(required_denominator.saturating_sub(1))
}

fn conformal_p_value(
    clean_scores_sorted: &[f64],
    test_risk_score: f64,
) -> Result<ConformalPValueV1, String> {
    if clean_scores_sorted.is_empty() {
        return Err("conformal calibration requires at least one clean score".to_owned());
    }
    if !test_risk_score.is_finite() {
        return Err("test risk score must be finite".to_owned());
    }
    let first_ge = clean_scores_sorted.partition_point(|score| *score < test_risk_score);
    let at_least_as_extreme = clean_scores_sorted.len() - first_ge;
    let numerator = u64::try_from(at_least_as_extreme)
        .map_err(|_| "conformal numerator overflows u64")?
        .checked_add(1)
        .ok_or_else(|| "conformal numerator overflows u64".to_owned())?;
    let denominator = u64::try_from(clean_scores_sorted.len())
        .map_err(|_| "conformal support overflows u64")?
        .checked_add(1)
        .ok_or_else(|| "conformal denominator overflows u64".to_owned())?;
    let ceiling = numerator
        .checked_mul(1_000)
        .ok_or_else(|| "conformal p-value scaling overflows u64".to_owned())?
        .div_ceil(denominator)
        .min(1_000) as u16;
    Ok(ConformalPValueV1 {
        numerator,
        denominator,
        ceiling_permille: ceiling,
    })
}

fn conformal_rejects(p_value: &ConformalPValueV1, alpha_permille: u16) -> bool {
    u128::from(p_value.numerator) * 1_000
        <= u128::from(alpha_permille) * u128::from(p_value.denominator)
}

/// Lanczos log-gamma approximation; used by the exact-binomial beta quantile.
fn log_gamma(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if value < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * value).sin().ln()
            - log_gamma(1.0 - value);
    }
    let shifted = value - 1.0;
    let mut accumulator = 0.999_999_999_999_809_9;
    for (index, coefficient) in COEFFICIENTS.iter().enumerate() {
        accumulator += coefficient / (shifted + index as f64 + 1.0);
    }
    let t = shifted + 7.5;
    0.918_938_533_204_672_7
        + (shifted + 0.5) * t.ln()
        - t
        + accumulator.ln()
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 256;
    const EPSILON: f64 = 3.0e-14;
    const MIN_POSITIVE: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - (qab * x / qap);
    if d.abs() < MIN_POSITIVE {
        d = MIN_POSITIVE;
    }
    d = 1.0 / d;
    let mut h = d;

    for iteration in 1..=MAX_ITERATIONS {
        let m = iteration as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_POSITIVE {
            d = MIN_POSITIVE;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_POSITIVE {
            c = MIN_POSITIVE;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_POSITIVE {
            d = MIN_POSITIVE;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_POSITIVE {
            c = MIN_POSITIVE;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPSILON {
            break;
        }
    }
    h
}

fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let log_beta_factor = log_gamma(a + b) - log_gamma(a) - log_gamma(b)
        + a * x.ln()
        + b * (-x).ln_1p();
    let factor = log_beta_factor.exp();
    let result = if x < (a + 1.0) / (a + b + 2.0) {
        factor * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b
    };
    result.clamp(0.0, 1.0)
}

fn inverse_regularized_beta(probability: f64, a: f64, b: f64) -> Result<f64, String> {
    if !probability.is_finite()
        || !(0.0..=1.0).contains(&probability)
        || !a.is_finite()
        || !b.is_finite()
        || a <= 0.0
        || b <= 0.0
    {
        return Err("invalid beta-quantile arguments".to_owned());
    }
    if probability == 0.0 {
        return Ok(0.0);
    }
    if probability == 1.0 {
        return Ok(1.0);
    }
    let mut lower = 0.0;
    let mut upper = 1.0;
    for _ in 0..96 {
        let midpoint = (lower + upper) / 2.0;
        if regularized_incomplete_beta(midpoint, a, b) < probability {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    let result = (lower + upper) / 2.0;
    if result.is_finite() {
        Ok(result)
    } else {
        Err("beta quantile is non-finite".to_owned())
    }
}

/// One-sided 95% Clopper-Pearson upper bound for a binomial false-positive rate.
pub fn exact_binomial_upper_95(false_positive: u64, clean_support: u64) -> Result<f64, String> {
    if clean_support == 0 || false_positive > clean_support {
        return Err("binomial false-positive counts require 0 <= FP <= n and n > 0".to_owned());
    }
    if false_positive == clean_support {
        return Ok(1.0);
    }
    inverse_regularized_beta(
        0.95,
        false_positive as f64 + 1.0,
        (clean_support - false_positive) as f64,
    )
}

fn validate_group(group: &CalibrationGroupV1) -> Result<(), String> {
    if group.schema != CALIBRATION_GROUP_SCHEMA {
        return Err(format!("group schema must be {CALIBRATION_GROUP_SCHEMA}"));
    }
    require_digest(&group.identity.model_checkpoint_sha256, "model checkpoint")?;
    require_digest(&group.identity.tokenizer_sha256, "tokenizer")?;
    require_digest(&group.identity.probe_config_sha256, "probe config")?;
    valid_id(&group.identity.task_family, "task_family")?;
    valid_id(&group.identity.language, "language")?;
    valid_id(&group.identity.training_replicate_id, "training_replicate_id")?;
    let expected = compute_calibration_group_sha256(&group.identity)?;
    require_digest(&group.group_sha256, "calibration group")?;
    if group.group_sha256 != expected {
        return Err("calibration group digest does not match its declared identity".to_owned());
    }
    Ok(())
}

fn validate_detector(definition: &DetectorDefinitionV1) -> Result<(), String> {
    valid_id(&definition.detector_id, "detector_id")?;
    valid_id(&definition.detector_version, "detector_version")?;
    if definition.score_semantics.trim().is_empty() || definition.score_semantics.len() > 512 {
        return Err("detector score_semantics must be nonempty and <=512 bytes".to_owned());
    }
    Ok(())
}

fn checked_add(total: &mut u64, value: u64, label: &str) -> Result<(), String> {
    *total = total
        .checked_add(value)
        .ok_or_else(|| format!("{label} counter overflows u64"))?;
    Ok(())
}

fn percentile_nearest_rank(values: &[u64], percentile: u64) -> Result<u64, String> {
    if values.is_empty() || !(1..=100).contains(&percentile) {
        return Err("percentile requires observations and percentile in [1, 100]".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (percentile * sorted.len() as u64).div_ceil(100).max(1);
    Ok(sorted[(rank - 1) as usize])
}

fn make_cost_summary(
    group_sha256: &str,
    definition: &DetectorDefinitionV1,
    observations: &[&DetectorObservationV1],
) -> Result<DetectorCostSummaryV1, String> {
    if observations.is_empty() {
        return Err("detector cost summary requires observations".to_owned());
    }
    let mut calls = 0_u64;
    let mut input_tokens = 0_u64;
    let mut output_tokens = 0_u64;
    let mut total_wall_time = 0_u128;
    let mut maximum_peak_memory = 0_u64;
    let mut wall_times = Vec::with_capacity(observations.len());
    for observation in observations {
        checked_add(&mut calls, observation.inference_calls, "inference calls")?;
        checked_add(&mut input_tokens, observation.input_tokens, "input token")?;
        checked_add(&mut output_tokens, observation.output_tokens, "output token")?;
        total_wall_time = total_wall_time
            .checked_add(u128::from(observation.wall_time_micros))
            .ok_or_else(|| "wall-time counter overflows u128".to_owned())?;
        maximum_peak_memory = maximum_peak_memory.max(observation.peak_memory_bytes);
        wall_times.push(observation.wall_time_micros);
    }
    let mean_wall_time = total_wall_time as f64 / observations.len() as f64;
    Ok(DetectorCostSummaryV1 {
        calibration_group_sha256: group_sha256.to_owned(),
        detector_id: definition.detector_id.clone(),
        detector_version: definition.detector_version.clone(),
        test_observation_count: observations.len() as u64,
        total_inference_calls: calls,
        total_input_tokens: input_tokens,
        total_output_tokens: output_tokens,
        mean_wall_time_micros: mean_wall_time,
        p50_wall_time_micros: percentile_nearest_rank(&wall_times, 50)?,
        p95_wall_time_micros: percentile_nearest_rank(&wall_times, 95)?,
        maximum_peak_memory_bytes: maximum_peak_memory,
    })
}

fn analyze_group(
    group: &CalibrationGroupV1,
    definitions: &[DetectorDefinitionV1],
    observations: &[&DetectorObservationV1],
    targets: &[u16],
    pvalue_output: &mut Vec<CalibratedTestScoreV1>,
    disagreement_output: &mut Vec<DetectorDisagreementCellV1>,
    cost_output: &mut Vec<DetectorCostSummaryV1>,
) -> Result<Vec<DetectorCalibrationCellV1>, String> {
    let mut observations_by_key =
        BTreeMap::<(ObservationSplitV1, &str, &str, &str), &DetectorObservationV1>::new();
    let mut sample_metadata =
        BTreeMap::<(ObservationSplitV1, &str), (GroundTruthV1, &str)>::new();
    for observation in observations {
        let key = (
            observation.split,
            observation.sample_id.as_str(),
            observation.detector_id.as_str(),
            observation.detector_version.as_str(),
        );
        if observations_by_key.insert(key, observation).is_some() {
            return Err("duplicate detector observation for sample/detector/split".to_owned());
        }
        let meta_key = (observation.split, observation.sample_id.as_str());
        let metadata = (
            observation.ground_truth,
            observation.source_cluster_id.as_str(),
        );
        if let Some(previous) = sample_metadata.insert(meta_key, metadata)
            && previous != metadata
        {
            return Err("ground truth or source cluster differs across detectors for the same sample".to_owned());
        }
    }

    let mut calibration_ids = BTreeSet::<&str>::new();
    let mut test_ids = BTreeSet::<&str>::new();
    let mut calibration_clusters = BTreeSet::<&str>::new();
    let mut test_clusters = BTreeSet::<&str>::new();
    for ((split, sample_id), (truth, cluster_id)) in &sample_metadata {
        valid_id(sample_id, "sample_id")?;
        valid_id(cluster_id, "source_cluster_id")?;
        match split {
            ObservationSplitV1::CleanCalibration => {
                if *truth != GroundTruthV1::Clean {
                    return Err("clean calibration split may contain only adjudicated clean observations".to_owned());
                }
                calibration_ids.insert(*sample_id);
                calibration_clusters.insert(*cluster_id);
            }
            ObservationSplitV1::LockedTest => {
                test_ids.insert(*sample_id);
                test_clusters.insert(*cluster_id);
            }
        }
    }
    if calibration_ids.is_empty() || test_ids.is_empty() {
        return Err("each calibration group requires nonempty clean-calibration and locked-test splits".to_owned());
    }
    if calibration_ids.intersection(&test_ids).next().is_some() {
        return Err("sample IDs overlap between clean calibration and locked test".to_owned());
    }
    if calibration_clusters.intersection(&test_clusters).next().is_some() {
        return Err("source clusters overlap between clean calibration and locked test".to_owned());
    }

    for detector in definitions {
        for sample_id in &calibration_ids {
            if !observations_by_key.contains_key(&(
                ObservationSplitV1::CleanCalibration,
                *sample_id,
                detector.detector_id.as_str(),
                detector.detector_version.as_str(),
            )) {
                return Err(format!(
                    "detector {} has an incomplete clean-calibration score grid",
                    detector.detector_id
                ));
            }
        }
        for sample_id in &test_ids {
            if !observations_by_key.contains_key(&(
                ObservationSplitV1::LockedTest,
                *sample_id,
                detector.detector_id.as_str(),
                detector.detector_version.as_str(),
            )) {
                return Err(format!(
                    "detector {} has an incomplete locked-test score grid",
                    detector.detector_id
                ));
            }
        }
    }
    if observations_by_key.len() != (calibration_ids.len() + test_ids.len()) * definitions.len() {
        return Err("observation grid contains unknown detector identities or unmatched samples".to_owned());
    }

    let mut sorted_clean_by_detector =
        BTreeMap::<(&str, &str), Vec<f64>>::new();
    for detector in definitions {
        let clean = sorted_clean_by_detector
            .entry((detector.detector_id.as_str(), detector.detector_version.as_str()))
            .or_default();
        let mut costs = Vec::with_capacity(test_ids.len());
        for sample_id in &calibration_ids {
            let observation = observations_by_key[&(
                ObservationSplitV1::CleanCalibration,
                sample_id,
                detector.detector_id.as_str(),
                detector.detector_version.as_str(),
            )];
            clean.push(normalize_risk(observation.raw_score, detector.score_direction)?);
        }
        clean.sort_by(f64::total_cmp);
        for sample_id in &test_ids {
            let observation = observations_by_key[&(
                ObservationSplitV1::LockedTest,
                sample_id,
                detector.detector_id.as_str(),
                detector.detector_version.as_str(),
            )];
            costs.push(observation);
            let normalized = normalize_risk(observation.raw_score, detector.score_direction)?;
            let p_value = conformal_p_value(clean, normalized)?;
            pvalue_output.push(CalibratedTestScoreV1 {
                calibration_group_sha256: group.group_sha256.clone(),
                sample_id: observation.sample_id.clone(),
                source_cluster_id: observation.source_cluster_id.clone(),
                detector_id: detector.detector_id.clone(),
                detector_version: detector.detector_version.clone(),
                ground_truth: observation.ground_truth,
                raw_score: observation.raw_score,
                normalized_risk_score: normalized,
                calibration_clean_support: clean.len() as u64,
                conformal_p_value: p_value,
            });
        }
        cost_output.push(make_cost_summary(&group.group_sha256, detector, &costs)?);
    }

    let test_pvalues_by_detector: BTreeMap<(&str, &str, &str), &CalibratedTestScoreV1> =
        pvalue_output
            .iter()
            .filter(|value| value.calibration_group_sha256 == group.group_sha256)
            .map(|value| {
                (
                    (
                        value.detector_id.as_str(),
                        value.detector_version.as_str(),
                        value.sample_id.as_str(),
                    ),
                    value,
                )
            })
            .collect();

    let mut cells = Vec::with_capacity(definitions.len() * targets.len());
    let mut predictions = BTreeMap::<(u16, &str, &str, &str), bool>::new();
    for detector in definitions {
        let clean_support = sorted_clean_by_detector
            .get(&(detector.detector_id.as_str(), detector.detector_version.as_str()))
            .map(Vec::len)
            .unwrap_or(0) as u64;
        for target in targets {
            let target_supported = clean_support >= minimum_calibration_support(*target)?;
            let mut test_clean = 0_u64;
            let mut test_contaminated = 0_u64;
            let mut false_positive = 0_u64;
            let mut true_positive = 0_u64;
            let mut false_negative = 0_u64;
            let mut true_negative = 0_u64;
            for sample_id in &test_ids {
                let scored = test_pvalues_by_detector
                    .get(&(
                        detector.detector_id.as_str(),
                        detector.detector_version.as_str(),
                        *sample_id,
                    ))
                    .ok_or_else(|| "calibrated test p-value is missing".to_owned())?;
                let predicted = target_supported
                    && conformal_rejects(&scored.conformal_p_value, *target);
                predictions.insert(
                    (
                        *target,
                        group.group_sha256.as_str(),
                        detector.detector_id.as_str(),
                        *sample_id,
                    ),
                    predicted,
                );
                match scored.ground_truth {
                    GroundTruthV1::Clean => {
                        test_clean += 1;
                        if predicted {
                            false_positive += 1;
                        } else {
                            true_negative += 1;
                        }
                    }
                    GroundTruthV1::Contaminated => {
                        test_contaminated += 1;
                        if predicted {
                            true_positive += 1;
                        } else {
                            false_negative += 1;
                        }
                    }
                }
            }
            let metrics = OperatingPointMetricsV1 {
                target_fpr_permille: *target,
                calibration_clean_support: clean_support,
                minimum_attainable_p_value: ConformalPValueV1 {
                    numerator: 1,
                    denominator: clean_support + 1,
                    ceiling_permille: (1_000_u64.div_ceil(clean_support + 1)).min(1_000) as u16,
                },
                calibration_resolution_sufficient: target_supported,
                test_clean_support: test_clean,
                test_contaminated_support: test_contaminated,
                false_positive: target_supported.then_some(false_positive),
                true_positive: target_supported.then_some(true_positive),
                false_negative: target_supported.then_some(false_negative),
                true_negative: target_supported.then_some(true_negative),
                false_positive_rate: if target_supported && test_clean > 0 {
                    Some(false_positive as f64 / test_clean as f64)
                } else {
                    None
                },
                false_positive_rate_upper_95_one_sided: if target_supported && test_clean > 0 {
                    Some(exact_binomial_upper_95(false_positive, test_clean)?)
                } else {
                    None
                },
                true_positive_rate: if target_supported && test_contaminated > 0 {
                    Some(true_positive as f64 / test_contaminated as f64)
                } else {
                    None
                },
            };
            cells.push(DetectorCalibrationCellV1 {
                calibration_group_sha256: group.group_sha256.clone(),
                task_family: group.identity.task_family.clone(),
                language: group.identity.language.clone(),
                exposure_stage: group.identity.exposure_stage,
                training_replicate_id: group.identity.training_replicate_id.clone(),
                detector_id: detector.detector_id.clone(),
                detector_version: detector.detector_version.clone(),
                calibration_clean_support: clean_support,
                locked_test_support: test_ids.len() as u64,
                operating_points: vec![metrics],
            });
        }
    }

    for target in targets {
        for left_index in 0..definitions.len() {
            for right_index in (left_index + 1)..definitions.len() {
                let left = &definitions[left_index];
                let right = &definitions[right_index];
                let mut disagreement = DetectorDisagreementCellV1 {
                    calibration_group_sha256: group.group_sha256.clone(),
                    target_fpr_permille: *target,
                    detector_left: left.detector_id.clone(),
                    detector_right: right.detector_id.clone(),
                    comparable_test_support: test_ids.len() as u64,
                    disagree_count: 0,
                    both_flag_contaminated: 0,
                    left_only_flag_contaminated: 0,
                    right_only_flag_contaminated: 0,
                    neither_flag_contaminated: 0,
                    disagreement_rate: None,
                };
                for sample_id in &test_ids {
                    let left_flag = predictions[&(
                        *target,
                        group.group_sha256.as_str(),
                        left.detector_id.as_str(),
                        sample_id,
                    )];
                    let right_flag = predictions[&(
                        *target,
                        group.group_sha256.as_str(),
                        right.detector_id.as_str(),
                        sample_id,
                    )];
                    match (left_flag, right_flag) {
                        (true, true) => disagreement.both_flag_contaminated += 1,
                        (true, false) => {
                            disagreement.left_only_flag_contaminated += 1;
                            disagreement.disagree_count += 1;
                        }
                        (false, true) => {
                            disagreement.right_only_flag_contaminated += 1;
                            disagreement.disagree_count += 1;
                        }
                        (false, false) => disagreement.neither_flag_contaminated += 1,
                    }
                }
                if disagreement.comparable_test_support > 0 {
                    disagreement.disagreement_rate = Some(
                        disagreement.disagree_count as f64
                            / disagreement.comparable_test_support as f64,
                    );
                }
                disagreement_output.push(disagreement);
            }
        }
    }

    Ok(cells)
}

pub fn analyze_detector_calibration(
    study: &DetectorCalibrationStudyV1,
) -> Result<DetectorCalibrationReportV1, String> {
    if study.schema != CALIBRATION_STUDY_SCHEMA {
        return Err(format!("study schema must be {CALIBRATION_STUDY_SCHEMA}"));
    }
    valid_id(&study.study_id, "study_id")?;
    valid_id(&study.benchmark_id, "benchmark_id")?;
    if study.benchmark_version.trim().is_empty() || study.benchmark_version.len() > 64 {
        return Err("benchmark_version must be nonempty and <=64 bytes".to_owned());
    }
    if study.detector_definitions.is_empty()
        || study.detector_definitions.len() > MAX_DETECTORS
        || study.calibration_groups.is_empty()
        || study.calibration_groups.len() > MAX_CALIBRATION_GROUPS
        || study.observations.is_empty()
        || study.observations.len() > MAX_CALIBRATION_OBSERVATIONS
        || study.target_fpr_permille.is_empty()
        || study.target_fpr_permille.len() > MAX_TARGETS
    {
        return Err("calibration study has empty or oversized detector/group/observation/target collections".to_owned());
    }
    let mut targets = study.target_fpr_permille.clone();
    targets.sort_unstable();
    targets.dedup();
    if targets != study.target_fpr_permille
        || targets.iter().any(|target| !(1..=1_000).contains(target))
    {
        return Err("target FPR permille values must be unique, ascending and in [1, 1000]".to_owned());
    }

    let mut definitions = BTreeMap::<(&str, &str), &DetectorDefinitionV1>::new();
    for definition in &study.detector_definitions {
        validate_detector(definition)?;
        if definitions
            .insert(
                (definition.detector_id.as_str(), definition.detector_version.as_str()),
                definition,
            )
            .is_some()
        {
            return Err("detector ID/version definitions must be unique".to_owned());
        }
    }

    let mut groups = BTreeMap::<String, &CalibrationGroupV1>::new();
    for group in &study.calibration_groups {
        validate_group(group)?;
        if groups.insert(group.group_sha256.clone(), group).is_some() {
            return Err("calibration group digest is duplicated".to_owned());
        }
    }

    let mut observation_keys = BTreeSet::<ObservationKey>::new();
    let mut observations_by_group = BTreeMap::<String, Vec<&DetectorObservationV1>>::new();
    for observation in &study.observations {
        valid_id(&observation.sample_id, "sample_id")?;
        valid_id(&observation.source_cluster_id, "source_cluster_id")?;
        require_digest(&observation.calibration_group_sha256, "calibration_group_sha256")?;
        valid_id(&observation.detector_id, "detector_id")?;
        valid_id(&observation.detector_version, "detector_version")?;
        if !observation.raw_score.is_finite() {
            return Err("raw detector score must be finite".to_owned());
        }
        if !groups.contains_key(&observation.calibration_group_sha256) {
            return Err("observation references an unknown calibration group".to_owned());
        }
        if !definitions.contains_key(&(
            observation.detector_id.as_str(),
            observation.detector_version.as_str(),
        )) {
            return Err("observation references an unknown detector definition".to_owned());
        }
        let key = ObservationKey {
            group_sha256: observation.calibration_group_sha256.clone(),
            split: observation.split,
            sample_id: observation.sample_id.clone(),
            detector_id: observation.detector_id.clone(),
            detector_version: observation.detector_version.clone(),
        };
        if !observation_keys.insert(key) {
            return Err("duplicate observation key".to_owned());
        }
        observations_by_group
            .entry(observation.calibration_group_sha256.clone())
            .or_default()
            .push(observation);
    }

    let study_bytes = canonical_json(study, "detector calibration study")?;
    let study_sha256 = hash_domain(b"detector-calibration-study.v1", &study_bytes);
    let mut group_cells = Vec::new();
    let mut scored_test_p_values = Vec::new();
    let mut disagreement_matrix = Vec::new();
    let mut detector_costs = Vec::new();

    for group in &study.calibration_groups {
        let observations = observations_by_group
            .get(&group.group_sha256)
            .ok_or_else(|| "calibration group has no observations".to_owned())?;
        let cells = analyze_group(
            group,
            &study.detector_definitions,
            observations,
            &targets,
            &mut scored_test_p_values,
            &mut disagreement_matrix,
            &mut detector_costs,
        )?;
        group_cells.extend(cells);
    }

    // Equal-weight groups, not item counts, so a large domain or one
    // over-sampled checkpoint cannot dominate the macro summary.
    let mut macro_cells = Vec::new();
    for definition in &study.detector_definitions {
        for target in &targets {
            let eligible: Vec<&DetectorCalibrationCellV1> = group_cells
                .iter()
                .filter(|cell| {
                    cell.detector_id == definition.detector_id
                        && cell.detector_version == definition.detector_version
                        && cell
                            .operating_points
                            .iter()
                            .any(|point| point.target_fpr_permille == *target
                                && point.calibration_resolution_sufficient)
                })
                .collect();
            let fprs: Vec<f64> = eligible
                .iter()
                .filter_map(|cell| {
                    cell.operating_points
                        .iter()
                        .find(|point| point.target_fpr_permille == *target)
                        .and_then(|point| point.false_positive_rate)
                })
                .collect();
            let tprs: Vec<f64> = eligible
                .iter()
                .filter_map(|cell| {
                    cell.operating_points
                        .iter()
                        .find(|point| point.target_fpr_permille == *target)
                        .and_then(|point| point.true_positive_rate)
                })
                .collect();
            let upper_bounds: Vec<f64> = eligible
                .iter()
                .filter_map(|cell| {
                    cell.operating_points
                        .iter()
                        .find(|point| point.target_fpr_permille == *target)
                        .and_then(|point| point.false_positive_rate_upper_95_one_sided)
                })
                .collect();
            macro_cells.push(DetectorMacroCellV1 {
                detector_id: definition.detector_id.clone(),
                detector_version: definition.detector_version.clone(),
                target_fpr_permille: *target,
                eligible_calibration_group_count: eligible.len() as u64,
                mean_test_false_positive_rate: mean_optional(&fprs),
                worst_group_fpr_upper_95_one_sided: upper_bounds
                    .iter()
                    .copied()
                    .reduce(f64::max),
                mean_test_true_positive_rate: mean_optional(&tprs),
            });
        }
    }

    let uncalibrated_count = group_cells
        .iter()
        .flat_map(|cell| &cell.operating_points)
        .filter(|point| !point.calibration_resolution_sufficient)
        .count();
    let issues = vec![
        "Conformal p-values are finite-sample super-uniform only under exchangeability of clean calibration and clean test observations inside the exact declared calibration group.".to_owned(),
        "Calibration identity digests are consistency commitments; this module does not authenticate model/checkpoint custody, source manifests, clean-label truth or evaluator access controls.".to_owned(),
        "Held-out FPR intervals are one-sided exact binomial bounds per group and target, not simultaneous or multiplicity-adjusted family-wise intervals.".to_owned(),
        "Groups with insufficient calibration resolution are not estimable at the requested alpha; a zero observed false-positive count is not a 1% guarantee.".to_owned(),
        "Macro summaries weight eligible calibration groups equally; they do not replace independent model-pair replication or a transportability analysis.".to_owned(),
    ];
    let analysis_status = if uncalibrated_count > 0 {
        "exploratory_with_unestimable_operating_points"
    } else {
        "exploratory_calibration_audit"
    };

    Ok(DetectorCalibrationReportV1 {
        schema: CALIBRATION_REPORT_SCHEMA.to_owned(),
        study_id: study.study_id.clone(),
        benchmark_id: study.benchmark_id.clone(),
        benchmark_version: study.benchmark_version.clone(),
        study_sha256,
        calibration_group_count: study.calibration_groups.len() as u64,
        detector_count: study.detector_definitions.len() as u64,
        observation_count: study.observations.len() as u64,
        scored_locked_test_observation_count: scored_test_p_values.len() as u64,
        group_cells,
        scored_test_p_values,
        disagreement_matrix,
        detector_costs,
        macro_cells,
        intervals_multiplicity_adjusted: false,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: analysis_status.to_owned(),
        issues,
        claim_boundary: "This report calculates finite-sample upper-tail conformal p-values against disjoint clean calibration cohorts and evaluates fixed-alpha operating points on a separate locked test split. The conformal super-uniform guarantee requires exchangeability of clean calibration/test observations within each exact model-checkpoint, tokenizer, task-family, language, exposure-stage and detector-probe group. Held-out FPR uncertainty is a one-sided exact binomial bound for each group/target; intervals are not simultaneous or multiplicity-adjusted. The report does not establish exchangeability under distribution shift, authenticate model/data custody, prove training membership, establish causal exposure, or support a universal detector or SOTA claim.".to_owned(),
    })
}

fn mean_optional(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn identity() -> CalibrationGroupIdentityV1 {
        CalibrationGroupIdentityV1 {
            model_checkpoint_sha256: digest('a'),
            tokenizer_sha256: digest('b'),
            task_family: "math".to_owned(),
            language: "en".to_owned(),
            exposure_stage: ExposureStageV1::ReinforcementLearningVerifiableReward,
            probe_config_sha256: digest('c'),
            training_replicate_id: "pair-001".to_owned(),
        }
    }

    fn definition(id: &str, direction: ScoreDirectionV1) -> DetectorDefinitionV1 {
        DetectorDefinitionV1 {
            detector_id: id.to_owned(),
            detector_version: "v1".to_owned(),
            score_direction: direction,
            score_semantics: "risk-like contamination score".to_owned(),
        }
    }

    #[test]
    fn group_digest_binds_model_task_language_stage_and_probe() {
        let a = identity();
        let digest_a = compute_calibration_group_sha256(&a).unwrap();
        let mut b = a.clone();
        b.language = "fr".to_owned();
        let digest_b = compute_calibration_group_sha256(&b).unwrap();
        assert_ne!(digest_a, digest_b);
    }

    #[test]
    fn conformal_p_value_uses_upper_tail_and_plus_one_correction() {
        let clean = vec![0.1, 0.2, 0.3];
        let p = conformal_p_value(&clean, 0.25).unwrap();
        assert_eq!(p.numerator, 2);
        assert_eq!(p.denominator, 4);
        assert_eq!(p.ceiling_permille, 500);
    }

    #[test]
    fn ties_are_included_in_clean_upper_tail() {
        let clean = vec![0.5, 0.5, 1.0];
        let p = conformal_p_value(&clean, 0.5).unwrap();
        assert_eq!(p.numerator, 4);
        assert_eq!(p.denominator, 4);
        assert_eq!(p.ceiling_permille, 1_000);
    }

    #[test]
    fn p_value_resolution_requires_99_clean_calibration_items_for_one_percent() {
        assert_eq!(minimum_calibration_support(10).unwrap(), 99);
        assert_eq!(minimum_calibration_support(50).unwrap(), 19);
        assert_eq!(minimum_calibration_support(100).unwrap(), 9);
    }

    #[test]
    fn alpha_boundary_uses_exact_rational_arithmetic() {
        let p = ConformalPValueV1 {
            numerator: 1,
            denominator: 100,
            ceiling_permille: 10,
        };
        assert!(conformal_rejects(&p, 10));
        let above = ConformalPValueV1 {
            numerator: 2,
            denominator: 199,
            ceiling_permille: 11,
        };
        assert!(!conformal_rejects(&above, 10));
    }

    #[test]
    fn low_is_risk_direction_is_normalized() {
        assert_eq!(
            normalize_risk(-0.8, ScoreDirectionV1::LowerMeansContaminated).unwrap(),
            0.8
        );
        assert_eq!(
            normalize_risk(0.8, ScoreDirectionV1::HigherMeansContaminated).unwrap(),
            0.8
        );
    }

    #[test]
    fn zero_false_positives_at_300_have_about_one_percent_exact_upper_bound() {
        let upper = exact_binomial_upper_95(0, 300).unwrap();
        assert!((upper - 0.009936).abs() < 0.0001);
    }

    #[test]
    fn exact_binomial_upper_bound_is_monotone_in_false_positives() {
        let zero = exact_binomial_upper_95(0, 100).unwrap();
        let one = exact_binomial_upper_95(1, 100).unwrap();
        let two = exact_binomial_upper_95(2, 100).unwrap();
        assert!(zero < one && one < two);
    }

    #[test]
    fn all_false_positives_yield_unit_upper_bound() {
        assert_eq!(exact_binomial_upper_95(10, 10).unwrap(), 1.0);
    }

    #[test]
    fn target_alpha_requires_sorted_unique_permille_values() {
        let mut study = minimal_study();
        study.target_fpr_permille = vec![50, 10];
        assert!(analyze_detector_calibration(&study).is_err());
    }

    #[test]
    fn calibration_split_rejects_contaminated_labels() {
        let mut study = minimal_study();
        study.observations[0].ground_truth = GroundTruthV1::Contaminated;
        assert!(analyze_detector_calibration(&study).is_err());
    }

    #[test]
    fn sample_ids_must_not_overlap_splits() {
        let mut study = minimal_study();
        let mut obs = study.observations[0].clone();
        obs.split = ObservationSplitV1::LockedTest;
        obs.sample_id = study.observations[1].sample_id.clone();
        obs.detector_id = study.observations[1].detector_id.clone();
        obs.detector_version = study.observations[1].detector_version.clone();
        study.observations.push(obs);
        assert!(analyze_detector_calibration(&study).is_err());
    }

    fn minimal_study() -> DetectorCalibrationStudyV1 {
        let identity = identity();
        let group_sha = compute_calibration_group_sha256(&identity).unwrap();
        let group = CalibrationGroupV1 {
            schema: CALIBRATION_GROUP_SCHEMA.to_owned(),
            group_sha256: group_sha.clone(),
            identity,
        };
        let detectors = vec![
            definition("det-a", ScoreDirectionV1::HigherMeansContaminated),
            definition("det-b", ScoreDirectionV1::LowerMeansContaminated),
        ];
        let mut observations = Vec::new();
        for detector in &detectors {
            for index in 0..100 {
                observations.push(DetectorObservationV1 {
                    sample_id: format!("cal-{index:03}"),
                    source_cluster_id: format!("cal-source-{index:03}"),
                    calibration_group_sha256: group_sha.clone(),
                    split: ObservationSplitV1::CleanCalibration,
                    ground_truth: GroundTruthV1::Clean,
                    detector_id: detector.detector_id.clone(),
                    detector_version: detector.detector_version.clone(),
                    raw_score: if detector.score_direction == ScoreDirectionV1::HigherMeansContaminated {
                        f64::from(index) / 100.0
                    } else {
                        -f64::from(index) / 100.0
                    },
                    inference_calls: 1,
                    input_tokens: 100,
                    output_tokens: 10,
                    wall_time_micros: 1_000 + u64::from(index),
                    peak_memory_bytes: 10_000,
                });
            }
            for (index, (truth, score)) in [
                (GroundTruthV1::Clean, 0.2),
                (GroundTruthV1::Clean, 0.8),
                (GroundTruthV1::Contaminated, 1.0),
                (GroundTruthV1::Contaminated, 0.99),
            ]
            .into_iter()
            .enumerate()
            {
                let raw_score = if detector.score_direction == ScoreDirectionV1::HigherMeansContaminated {
                    score
                } else {
                    -score
                };
                observations.push(DetectorObservationV1 {
                    sample_id: format!("test-{index:03}"),
                    source_cluster_id: format!("test-source-{index:03}"),
                    calibration_group_sha256: group_sha.clone(),
                    split: ObservationSplitV1::LockedTest,
                    ground_truth: truth,
                    detector_id: detector.detector_id.clone(),
                    detector_version: detector.detector_version.clone(),
                    raw_score,
                    inference_calls: 2,
                    input_tokens: 120,
                    output_tokens: 16,
                    wall_time_micros: 2_000 + index as u64,
                    peak_memory_bytes: 12_000,
                });
            }
        }
        DetectorCalibrationStudyV1 {
            schema: CALIBRATION_STUDY_SCHEMA.to_owned(),
            study_id: "cal-study-001".to_owned(),
            benchmark_id: "benchmark-001".to_owned(),
            benchmark_version: "v1".to_owned(),
            target_fpr_permille: DEFAULT_FPR_TARGETS_PERMILLE.to_vec(),
            detector_definitions: detectors,
            calibration_groups: vec![group],
            observations,
        }
    }

    #[test]
    fn analyzer_produces_group_macro_cost_and_disagreement_reports() {
        let report = analyze_detector_calibration(&minimal_study()).unwrap();
        assert_eq!(report.calibration_group_count, 1);
        assert_eq!(report.detector_count, 2);
        assert_eq!(report.scored_locked_test_observation_count, 8);
        assert_eq!(report.disagreement_matrix.len(), 3);
        assert_eq!(report.detector_costs.len(), 2);
        assert_eq!(report.macro_cells.len(), 6);
        assert!(!report.confirmatory_eligible);
        assert!(!report.intervals_multiplicity_adjusted);
    }

    #[test]
    fn locked_test_truth_does_not_change_conformal_p_value() {
        let mut study_a = minimal_study();
        let mut study_b = study_a.clone();
        for observation in &mut study_b.observations {
            if observation.split == ObservationSplitV1::LockedTest
                && observation.sample_id == "test-000"
            {
                observation.ground_truth = GroundTruthV1::Contaminated;
            }
        }
        let report_a = analyze_detector_calibration(&study_a).unwrap();
        let report_b = analyze_detector_calibration(&study_b).unwrap();
        let p_a = report_a
            .scored_test_p_values
            .iter()
            .find(|value| value.sample_id == "test-000" && value.detector_id == "det-a")
            .unwrap();
        let p_b = report_b
            .scored_test_p_values
            .iter()
            .find(|value| value.sample_id == "test-000" && value.detector_id == "det-a")
            .unwrap();
        assert_eq!(p_a.conformal_p_value, p_b.conformal_p_value);
    }
}
