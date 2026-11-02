//! Two-stage conformal calibration for heterogeneous contamination detector evidence.
//!
//! Stage A maps each detector score to a conservative empirical upper-tail p-value
//! within a task-family × model-regime group. A fixed sum(-ln(p_d)) forms a fused
//! nonconformity score. Stage B calibrates that *fused score itself* on an
//! independent clean cohort. The final empirical rank therefore does not assume
//! detector independence; validity still requires exchangeability of Stage-B
//! calibration units and future clean test units within each group.
//!
//! Missing detector features produce abstention on evaluation rows; calibration
//! rows must be complete. Calibration and evaluation units are independent
//! clusters, not individual repeated generations or near-duplicate questions.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CONFORMAL_FUSION_REQUEST_SCHEMA: &str = "symthaea.detector_conformal_fusion_request.v1";
pub const CONFORMAL_FUSION_REPORT_SCHEMA: &str = "symthaea.detector_conformal_fusion_report.v1";
pub const FUSION_METHOD_ID: &str = "weighted_sum_negative_log_empirical_detector_pvalues_v2";
pub const MAX_CALIBRATION_ROWS: usize = 1_000_000;
pub const MAX_EVALUATION_ROWS: usize = 1_000_000;
pub const MAX_DETECTORS_PER_ROW: usize = 64;
pub const MIN_DETECTORS_PER_GROUP: usize = 2;
pub const MIN_CALIBRATION_A_UNITS: usize = 2;
pub const MIN_CALIBRATION_B_UNITS: usize = 2;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum DetectorScoreDirectionV1 {
    HigherMeansContaminated,
    LowerMeansContaminated,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorFeatureV1 {
    pub detector_id: String,
    pub direction: DetectorScoreDirectionV1,
    pub score: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorSpecV1 {
    pub detector_id: String,
    pub direction: DetectorScoreDirectionV1,
    /// Fixed, preregistered weight in permille. All detector weights sum to 1000.
    pub weight_permille: u16,
    /// Binds the declared detector feature meaning, extraction and score semantics.
    pub feature_contract_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationFeatureRowV1 {
    /// One independent source/task cluster. Repeated items/generations are nested.
    pub calibration_unit_id: String,
    pub source_cluster_sha256: String,
    pub task_family: String,
    pub model_regime: String,
    pub features: Vec<DetectorFeatureV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationFeatureRowV1 {
    pub evaluation_unit_id: String,
    pub source_cluster_sha256: String,
    pub task_family: String,
    pub model_regime: String,
    pub features: Vec<DetectorFeatureV1>,
    /// Optional label used only for the post-score evaluation summary.
    pub ground_truth_contaminated: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConformalFusionRequestV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub prompt_harness_sha256: String,
    pub runtime_environment_sha256: String,
    pub calibration_protocol_sha256: String,
    pub detector_registry_sha256: String,
    /// Alpha in permille; 10 means 1%, 50 means 5%.
    pub target_alpha_permille: u16,
    pub fusion_method_id: String,
    /// Complete detector set and fixed weights, pinned before calibration.
    pub detector_specs: Vec<DetectorSpecV1>,
    /// Independent clean reference cohort used to form detector-specific ranks.
    pub calibration_a: Vec<CalibrationFeatureRowV1>,
    /// Disjoint clean cohort used to calibrate the fixed fused statistic.
    pub calibration_b: Vec<CalibrationFeatureRowV1>,
    /// Locked evaluation cohort. Labels are used only after scoring.
    pub evaluation: Vec<EvaluationFeatureRowV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SelectiveDecisionV1 {
    FlagContaminated,
    NoContaminationFlag,
    AbstainUnsupportedGroup,
    AbstainDetectorFeatureMismatch,
    AbstainInsufficientCalibrationResolution,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DetectorEmpiricalPValueV1 {
    pub detector_id: String,
    pub direction: DetectorScoreDirectionV1,
    pub raw_score: f64,
    pub rank_numerator: u64,
    pub rank_denominator: u64,
    pub p_value: f64,
    pub weight_permille: u16,
    pub weighted_contribution: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectiveFusionScoreV1 {
    pub evaluation_unit_id: String,
    pub task_family: String,
    pub model_regime: String,
    pub detector_p_values: Vec<DetectorEmpiricalPValueV1>,
    pub fused_nonconformity: Option<f64>,
    pub final_p_rank_numerator: Option<u64>,
    pub final_p_rank_denominator: Option<u64>,
    pub final_p_value: Option<f64>,
    pub by_adjusted_p_value: Option<f64>,
    pub by_family_size: Option<u64>,
    pub minimum_attainable_p_value: Option<f64>,
    pub decision: SelectiveDecisionV1,
    pub ground_truth_contaminated: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConformalGroupSummaryV1 {
    pub task_family: String,
    pub model_regime: String,
    pub detector_ids: Vec<String>,
    pub calibration_a_unit_count: u64,
    pub calibration_b_unit_count: u64,
    pub evaluation_unit_count: u64,
    pub labelled_evaluation_unit_count: u64,
    pub flagged_count: u64,
    pub no_flag_count: u64,
    pub abstained_count: u64,
    pub clean_labelled_count: u64,
    pub contaminated_labelled_count: u64,
    pub false_positive_count: u64,
    pub true_positive_count: u64,
    pub false_negative_count: u64,
    pub empirical_fpr_permille: Option<f64>,
    pub empirical_tpr_permille: Option<f64>,
    pub fpr_one_sided_95_upper_permille: Option<f64>,
    pub tpr_one_sided_95_lower_permille: Option<f64>,
    pub decision_coverage_permille: f64,
    pub minimum_attainable_p_value: f64,
    pub minimum_detector_rank_p_value: f64,
    pub pairwise_rank_correlations: Vec<DetectorDisagreementV1>,
    pub target_alpha_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DetectorDisagreementV1 {
    pub left_detector_id: String,
    pub right_detector_id: String,
    pub paired_unit_count: u64,
    pub spearman_rank_correlation: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConformalFusionReportV1 {
    pub schema: String,
    pub fusion_method_id: String,
    pub detector_specs: Vec<DetectorSpecV1>,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub prompt_harness_sha256: String,
    pub runtime_environment_sha256: String,
    pub calibration_protocol_sha256: String,
    pub detector_registry_sha256: String,
    pub request_sha256: String,
    pub calibration_a_manifest_sha256: String,
    pub calibration_b_manifest_sha256: String,
    pub evaluation_manifest_sha256: String,
    pub target_alpha_permille: u16,
    pub multiple_testing_method: String,
    pub fdr_control_conditional_on_exchangeability: bool,
    pub group_count: u64,
    pub evaluation_unit_count: u64,
    pub flagged_count: u64,
    pub no_flag_count: u64,
    pub abstained_count: u64,
    pub score_rows: Vec<SelectiveFusionScoreV1>,
    pub group_summaries: Vec<ConformalGroupSummaryV1>,
    pub intervals_multiplicity_adjusted: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub assumptions: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    task_family: String,
    model_regime: String,
}

#[derive(Debug, Clone)]
struct GroupCalibrationA {
    detector_ids: Vec<String>,
    directions: BTreeMap<String, DetectorScoreDirectionV1>,
    weights_permille: BTreeMap<String, u16>,
    scores: BTreeMap<String, Vec<f64>>,
}

#[derive(Debug, Clone)]
struct GroupCalibrationB {
    fused_scores: Vec<f64>,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn valid_id(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

fn validate_features(features: &[DetectorFeatureV1], label: &str) -> Result<(), String> {
    if features.len() < MIN_DETECTORS_PER_GROUP || features.len() > MAX_DETECTORS_PER_ROW {
        return Err(format!(
            "{label} requires {MIN_DETECTORS_PER_GROUP}..={MAX_DETECTORS_PER_ROW} detector features"
        ));
    }
    let mut ids = BTreeSet::new();
    for feature in features {
        if !valid_id(&feature.detector_id, 128) {
            return Err(format!("{label} contains an invalid detector ID"));
        }
        if !ids.insert(feature.detector_id.as_str()) {
            return Err(format!("{label} contains duplicate detector IDs"));
        }
        if !feature.score.is_finite() {
            return Err(format!("{label} contains a non-finite detector score"));
        }
    }
    Ok(())
}

fn validate_evaluation_features(features: &[DetectorFeatureV1], label: &str) -> Result<(), String> {
    if features.len() > MAX_DETECTORS_PER_ROW {
        return Err(format!(
            "{label} exceeds {MAX_DETECTORS_PER_ROW} detector features"
        ));
    }
    let mut ids = BTreeSet::new();
    for feature in features {
        if !valid_id(&feature.detector_id, 128) {
            return Err(format!("{label} contains an invalid detector ID"));
        }
        if !ids.insert(feature.detector_id.as_str()) {
            return Err(format!("{label} contains duplicate detector IDs"));
        }
        if !feature.score.is_finite() {
            return Err(format!("{label} contains a non-finite detector score"));
        }
    }
    Ok(())
}

/// Canonical domain-separated digest for the pinned detector registry.
pub fn compute_detector_registry_sha256(specs: &[DetectorSpecV1]) -> Result<String, String> {
    validate_detector_specs(specs)?;
    let mut normalized = specs.to_vec();
    normalized.sort_by(|left, right| left.detector_id.cmp(&right.detector_id));
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize detector registry: {error}"))?;
    Ok(crate::hash_domain(
        b"detector-conformal-registry.v1",
        &bytes,
    ))
}

fn validate_detector_specs(specs: &[DetectorSpecV1]) -> Result<(), String> {
    if !(MIN_DETECTORS_PER_GROUP..=MAX_DETECTORS_PER_ROW).contains(&specs.len()) {
        return Err(format!(
            "detector registry requires {MIN_DETECTORS_PER_GROUP}..={MAX_DETECTORS_PER_ROW} specs"
        ));
    }
    let mut ids = BTreeSet::new();
    let mut total_weight = 0_u32;
    for spec in specs {
        if !valid_id(&spec.detector_id, 128) || !ids.insert(spec.detector_id.as_str()) {
            return Err("detector registry IDs must be unique opaque identifiers".to_owned());
        }
        if spec.weight_permille == 0 || spec.weight_permille > 1_000 {
            return Err("detector weights must be in 1..=1000 permille".to_owned());
        }
        validate_digest(&spec.feature_contract_sha256, "feature_contract_sha256")?;
        total_weight += u32::from(spec.weight_permille);
    }
    if total_weight != 1_000 {
        return Err("fixed detector weights must sum to exactly 1000 permille".to_owned());
    }
    Ok(())
}

fn canonical_calibration_manifest(rows: &[CalibrationFeatureRowV1]) -> Result<Vec<u8>, String> {
    let mut normalized = rows.to_vec();
    normalized.sort_by(|left, right| left.calibration_unit_id.cmp(&right.calibration_unit_id));
    for row in &mut normalized {
        row.features
            .sort_by(|left, right| left.detector_id.cmp(&right.detector_id));
    }
    serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize calibration manifest: {error}"))
}

fn canonical_evaluation_manifest(rows: &[EvaluationFeatureRowV1]) -> Result<Vec<u8>, String> {
    let mut normalized = rows.to_vec();
    normalized.sort_by(|left, right| left.evaluation_unit_id.cmp(&right.evaluation_unit_id));
    for row in &mut normalized {
        row.features
            .sort_by(|left, right| left.detector_id.cmp(&right.detector_id));
    }
    serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize evaluation manifest: {error}"))
}

fn group_key(task_family: &str, model_regime: &str) -> GroupKey {
    GroupKey {
        task_family: task_family.to_owned(),
        model_regime: model_regime.to_owned(),
    }
}

fn validate_calibration_row(
    unit_id: &str,
    task_family: &str,
    model_regime: &str,
    features: &[DetectorFeatureV1],
    label: &str,
) -> Result<(), String> {
    if !valid_id(unit_id, 128) || !valid_id(task_family, 64) || !valid_id(model_regime, 128) {
        return Err(format!("{label} has invalid unit/group IDs"));
    }
    validate_features(features, label)
}

fn detector_rank_p_value(
    calibration_scores: &[f64],
    direction: DetectorScoreDirectionV1,
    score: f64,
) -> Result<(u64, u64, f64), String> {
    if calibration_scores.is_empty() || !score.is_finite() {
        return Err("cannot rank detector score against an empty/non-finite reference".to_owned());
    }
    let extreme_count = calibration_scores
        .iter()
        .filter(|reference| match direction {
            DetectorScoreDirectionV1::HigherMeansContaminated => **reference >= score,
            DetectorScoreDirectionV1::LowerMeansContaminated => **reference <= score,
        })
        .count() as u64;
    let numerator = 1 + extreme_count;
    let denominator = calibration_scores.len() as u64 + 1;
    Ok((
        numerator,
        denominator,
        numerator as f64 / denominator as f64,
    ))
}

fn fused_nonconformity(
    features: &[DetectorFeatureV1],
    calibration: &GroupCalibrationA,
) -> Result<Vec<DetectorEmpiricalPValueV1>, String> {
    let feature_ids: Vec<String> = {
        let mut ids: Vec<String> = features
            .iter()
            .map(|feature| feature.detector_id.clone())
            .collect();
        ids.sort();
        ids
    };
    if feature_ids != calibration.detector_ids {
        return Err(
            "evaluation detector feature set does not match calibration contract".to_owned(),
        );
    }

    let mut p_values = Vec::with_capacity(features.len());
    for feature in features {
        let expected_direction = calibration
            .directions
            .get(&feature.detector_id)
            .ok_or_else(|| "detector direction missing from calibration contract".to_owned())?;
        if expected_direction != &feature.direction {
            return Err(format!(
                "detector {} direction differs from calibration contract",
                feature.detector_id
            ));
        }
        let reference = calibration
            .scores
            .get(&feature.detector_id)
            .ok_or_else(|| "detector calibration reference scores missing".to_owned())?;
        let (rank_numerator, rank_denominator, p_value) =
            detector_rank_p_value(reference, feature.direction, feature.score)?;
        let weight_permille = *calibration
            .weights_permille
            .get(&feature.detector_id)
            .ok_or_else(|| "detector weight missing from fixed registry".to_owned())?;
        let weighted_contribution = f64::from(weight_permille) / 1_000.0 * -p_value.ln();
        p_values.push(DetectorEmpiricalPValueV1 {
            detector_id: feature.detector_id.clone(),
            direction: feature.direction,
            raw_score: feature.score,
            rank_numerator,
            rank_denominator,
            p_value,
            weight_permille,
            weighted_contribution,
        });
    }
    p_values.sort_by(|left, right| left.detector_id.cmp(&right.detector_id));
    Ok(p_values)
}

fn combine_detector_p_values(p_values: &[DetectorEmpiricalPValueV1]) -> Result<f64, String> {
    if p_values.len() < MIN_DETECTORS_PER_GROUP {
        return Err("fused statistic requires at least two detector p-values".to_owned());
    }
    let result = p_values
        .iter()
        .map(|value| value.weighted_contribution)
        .sum::<f64>();
    if result.is_finite() {
        Ok(result)
    } else {
        Err("fused nonconformity statistic is non-finite".to_owned())
    }
}

fn unique_unit_id(set: &mut BTreeSet<String>, value: &str, label: &str) -> Result<(), String> {
    if !set.insert(value.to_owned()) {
        return Err(format!(
            "calibration/evaluation unit ID {value} is repeated across {label} splits"
        ));
    }
    Ok(())
}

fn outer_conformal_p_value(
    calibration_b_fused_scores: &[f64],
    evaluation_fused_score: f64,
) -> Result<(u64, u64, f64), String> {
    if calibration_b_fused_scores.is_empty() || !evaluation_fused_score.is_finite() {
        return Err("outer calibration requires nonempty finite fused scores".to_owned());
    }
    let extreme_count = calibration_b_fused_scores
        .iter()
        .filter(|score| **score >= evaluation_fused_score)
        .count() as u64;
    let numerator = 1 + extreme_count;
    let denominator = calibration_b_fused_scores.len() as u64 + 1;
    Ok((
        numerator,
        denominator,
        numerator as f64 / denominator as f64,
    ))
}

fn ln_gamma_lanczos(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = value - 1.0;
    let mut x = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        x += coefficient / (z + index as f64);
    }
    let t = z + 7.5;
    0.918_938_533_204_672_7 + (z + 0.5) * t.ln() - t + x.ln()
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 300;
    const EPSILON: f64 = 3.0e-14;
    const FLOOR: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FLOOR {
        d = FLOOR;
    }
    d = 1.0 / d;
    let mut h = d;
    for iteration in 1..=MAX_ITERATIONS {
        let m = iteration as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FLOOR {
            d = FLOOR;
        }
        c = 1.0 + aa / c;
        if c.abs() < FLOOR {
            c = FLOOR;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FLOOR {
            d = FLOOR;
        }
        c = 1.0 + aa / c;
        if c.abs() < FLOOR {
            c = FLOOR;
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
    let log_beta_factor = ln_gamma_lanczos(a + b) - ln_gamma_lanczos(a) - ln_gamma_lanczos(b)
        + a * x.ln()
        + b * (-x).ln_1p();
    let beta_factor = log_beta_factor.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        (beta_factor * beta_continued_fraction(a, b, x) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - beta_factor * beta_continued_fraction(b, a, 1.0 - x) / b).clamp(0.0, 1.0)
    }
}

fn inverse_regularized_beta(probability: f64, a: f64, b: f64) -> Result<f64, String> {
    if !(0.0..=1.0).contains(&probability)
        || a <= 0.0
        || b <= 0.0
        || !a.is_finite()
        || !b.is_finite()
    {
        return Err("invalid beta quantile parameters".to_owned());
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
        let middle = (lower + upper) / 2.0;
        if regularized_incomplete_beta(middle, a, b) < probability {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let result = (lower + upper) / 2.0;
    if result.is_finite() {
        Ok(result)
    } else {
        Err("beta quantile calculation is non-finite".to_owned())
    }
}

/// One-sided 95% Clopper-Pearson upper confidence bound for a binomial rate.
fn exact_binomial_upper_95(successes: u64, trials: u64) -> Result<f64, String> {
    if trials == 0 || successes > trials {
        return Err("binomial counts require 0 <= successes <= positive trials".to_owned());
    }
    if successes == trials {
        return Ok(1.0);
    }
    inverse_regularized_beta(0.95, successes as f64 + 1.0, (trials - successes) as f64)
}

/// One-sided 95% Clopper-Pearson lower confidence bound for a binomial rate.
fn exact_binomial_lower_95(successes: u64, trials: u64) -> Result<f64, String> {
    if trials == 0 || successes > trials {
        return Err("binomial counts require 0 <= successes <= positive trials".to_owned());
    }
    if successes == 0 {
        return Ok(0.0);
    }
    inverse_regularized_beta(0.05, successes as f64, (trials - successes) as f64 + 1.0)
}

fn benjamini_yekutieli_adjusted_p_values(
    values: &[(String, f64)],
) -> Result<BTreeMap<String, f64>, String> {
    if values.is_empty() {
        return Ok(BTreeMap::new());
    }
    if values
        .iter()
        .any(|(_, p)| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("BY adjustment requires finite p-values in [0, 1]".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let m = sorted.len();
    let harmonic = (1..=m).map(|rank| 1.0 / rank as f64).sum::<f64>();
    let multiplier = m as f64 * harmonic;
    let mut adjusted = BTreeMap::new();
    let mut reverse_min = 1.0_f64;
    for index in (0..m).rev() {
        let rank = index + 1;
        let candidate = sorted[index].1 * multiplier / rank as f64;
        reverse_min = reverse_min.min(candidate).min(1.0);
        adjusted.insert(sorted[index].0.clone(), reverse_min);
    }
    Ok(adjusted)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]].total_cmp(&values[order[start]]).is_eq() {
            end += 1;
        }
        let average_rank = ((start + 1) as f64 + end as f64) / 2.0;
        for index in &order[start..end] {
            ranks[*index] = average_rank;
        }
        start = end;
    }
    ranks
}

fn spearman_rank_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len()
        || left.len() < 3
        || left.iter().chain(right).any(|value| !value.is_finite())
    {
        return None;
    }
    let left_ranks = average_ranks(left);
    let right_ranks = average_ranks(right);
    let left_mean = left_ranks.iter().sum::<f64>() / left_ranks.len() as f64;
    let right_mean = right_ranks.iter().sum::<f64>() / right_ranks.len() as f64;
    let covariance = left_ranks
        .iter()
        .zip(&right_ranks)
        .map(|(a, b)| (a - left_mean) * (b - right_mean))
        .sum::<f64>();
    let left_variance = left_ranks
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>();
    let right_variance = right_ranks
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>();
    let denominator = (left_variance * right_variance).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some((covariance / denominator).clamp(-1.0, 1.0))
    }
}

fn pairwise_detector_spearman(
    rows: &[&EvaluationFeatureRowV1],
    calibration: &GroupCalibrationA,
) -> Vec<DetectorDisagreementV1> {
    let mut output = Vec::new();
    for left_index in 0..calibration.detector_ids.len() {
        for right_index in (left_index + 1)..calibration.detector_ids.len() {
            let left_id = &calibration.detector_ids[left_index];
            let right_id = &calibration.detector_ids[right_index];
            let mut left_scores = Vec::new();
            let mut right_scores = Vec::new();
            for row in rows {
                let left = row.features.iter().find(|feature| {
                    feature.detector_id == *left_id
                        && calibration.directions.get(left_id) == Some(&feature.direction)
                });
                let right = row.features.iter().find(|feature| {
                    feature.detector_id == *right_id
                        && calibration.directions.get(right_id) == Some(&feature.direction)
                });
                if let (Some(left), Some(right)) = (left, right) {
                    left_scores.push(left.score);
                    right_scores.push(right.score);
                }
            }
            output.push(DetectorDisagreementV1 {
                left_detector_id: left_id.clone(),
                right_detector_id: right_id.clone(),
                paired_unit_count: left_scores.len() as u64,
                spearman_rank_correlation: spearman_rank_correlation(&left_scores, &right_scores),
            });
        }
    }
    output
}

fn apply_by_adjustment(
    score_rows: &mut [SelectiveFusionScoreV1],
    group_summaries: &mut [ConformalGroupSummaryV1],
    alpha_permille: u16,
) -> Result<(), String> {
    for summary in group_summaries.iter_mut() {
        let family_values: Vec<(String, f64)> = score_rows
            .iter()
            .filter(|row| {
                row.task_family == summary.task_family && row.model_regime == summary.model_regime
            })
            .filter_map(|row| {
                if row.decision == SelectiveDecisionV1::NoContaminationFlag {
                    row.final_p_value
                        .map(|p| (row.evaluation_unit_id.clone(), p))
                } else {
                    None
                }
            })
            .collect();
        let family_size = family_values.len() as u64;
        let adjusted = benjamini_yekutieli_adjusted_p_values(&family_values)?;
        for row in score_rows.iter_mut().filter(|row| {
            row.task_family == summary.task_family && row.model_regime == summary.model_regime
        }) {
            if let Some(value) = adjusted.get(&row.evaluation_unit_id).copied() {
                row.by_adjusted_p_value = Some(value);
                row.by_family_size = Some(family_size);
                row.decision = if value * 1_000.0 <= f64::from(alpha_permille) {
                    SelectiveDecisionV1::FlagContaminated
                } else {
                    SelectiveDecisionV1::NoContaminationFlag
                };
            }
        }

        let rows: Vec<&SelectiveFusionScoreV1> = score_rows
            .iter()
            .filter(|row| {
                row.task_family == summary.task_family && row.model_regime == summary.model_regime
            })
            .collect();
        let flagged = rows
            .iter()
            .filter(|row| row.decision == SelectiveDecisionV1::FlagContaminated)
            .count() as u64;
        let no_flag = rows
            .iter()
            .filter(|row| row.decision == SelectiveDecisionV1::NoContaminationFlag)
            .count() as u64;
        let abstained = rows.len() as u64 - flagged - no_flag;
        let clean = rows
            .iter()
            .filter(|row| row.ground_truth_contaminated == Some(false))
            .count() as u64;
        let contaminated = rows
            .iter()
            .filter(|row| row.ground_truth_contaminated == Some(true))
            .count() as u64;
        let labelled = clean + contaminated;
        let fp = rows
            .iter()
            .filter(|row| {
                row.ground_truth_contaminated == Some(false)
                    && row.decision == SelectiveDecisionV1::FlagContaminated
            })
            .count() as u64;
        let tp = rows
            .iter()
            .filter(|row| {
                row.ground_truth_contaminated == Some(true)
                    && row.decision == SelectiveDecisionV1::FlagContaminated
            })
            .count() as u64;
        let fn_count = contaminated - tp;
        summary.labelled_evaluation_unit_count = labelled;
        summary.flagged_count = flagged;
        summary.no_flag_count = no_flag;
        summary.abstained_count = abstained;
        summary.clean_labelled_count = clean;
        summary.contaminated_labelled_count = contaminated;
        summary.false_positive_count = fp;
        summary.true_positive_count = tp;
        summary.false_negative_count = fn_count;
        summary.empirical_fpr_permille = (clean > 0).then_some(fp as f64 * 1_000.0 / clean as f64);
        summary.empirical_tpr_permille =
            (contaminated > 0).then_some(tp as f64 * 1_000.0 / contaminated as f64);
        summary.fpr_one_sided_95_upper_permille = if clean > 0 {
            Some(exact_binomial_upper_95(fp, clean)? * 1_000.0)
        } else {
            None
        };
        summary.tpr_one_sided_95_lower_permille = if contaminated > 0 {
            Some(exact_binomial_lower_95(tp, contaminated)? * 1_000.0)
        } else {
            None
        };
        summary.decision_coverage_permille =
            (flagged + no_flag) as f64 * 1_000.0 / rows.len().max(1) as f64;
    }
    Ok(())
}

pub fn calibrate_detector_panel(
    request: &DetectorConformalFusionRequestV1,
) -> Result<DetectorConformalFusionReportV1, String> {
    if request.schema != CONFORMAL_FUSION_REQUEST_SCHEMA {
        return Err(format!(
            "request schema must be {CONFORMAL_FUSION_REQUEST_SCHEMA}"
        ));
    }
    if !valid_id(&request.study_id, 128)
        || !valid_id(&request.benchmark_id, 128)
        || request.benchmark_version.trim().is_empty()
        || request.benchmark_version.len() > 64
    {
        return Err("study and benchmark identifiers are invalid".to_owned());
    }
    if !(1..=500).contains(&request.target_alpha_permille) {
        return Err("target_alpha_permille must be in [1, 500]".to_owned());
    }
    if request.fusion_method_id != FUSION_METHOD_ID {
        return Err(format!("fusion_method_id must be {FUSION_METHOD_ID}"));
    }
    validate_digest(&request.model_checkpoint_sha256, "model_checkpoint_sha256")?;
    validate_digest(&request.tokenizer_sha256, "tokenizer_sha256")?;
    validate_digest(&request.prompt_harness_sha256, "prompt_harness_sha256")?;
    validate_digest(
        &request.runtime_environment_sha256,
        "runtime_environment_sha256",
    )?;
    validate_digest(
        &request.calibration_protocol_sha256,
        "calibration_protocol_sha256",
    )?;
    validate_digest(
        &request.detector_registry_sha256,
        "detector_registry_sha256",
    )?;
    validate_detector_specs(&request.detector_specs)?;
    if compute_detector_registry_sha256(&request.detector_specs)?
        != request.detector_registry_sha256
    {
        return Err("detector_registry_sha256 does not match the pinned detector_specs".to_owned());
    }
    if request.calibration_a.len() < 2
        || request.calibration_a.len() > MAX_CALIBRATION_ROWS
        || request.calibration_b.len() < 2
        || request.calibration_b.len() > MAX_CALIBRATION_ROWS
        || request.evaluation.is_empty()
        || request.evaluation.len() > MAX_EVALUATION_ROWS
    {
        return Err(
            "calibration A/B and evaluation cohorts must be nonempty and bounded".to_owned(),
        );
    }

    let mut all_unit_ids = BTreeSet::new();
    let mut group_a_rows = BTreeMap::<GroupKey, Vec<&CalibrationFeatureRowV1>>::new();
    let mut group_b_rows = BTreeMap::<GroupKey, Vec<&CalibrationFeatureRowV1>>::new();
    let mut evaluation_rows = BTreeMap::<GroupKey, Vec<&EvaluationFeatureRowV1>>::new();

    let mut source_cluster_digests = BTreeSet::new();
    for row in &request.calibration_a {
        validate_digest(
            &row.source_cluster_sha256,
            "calibration-A source_cluster_sha256",
        )?;
        if !source_cluster_digests.insert(row.source_cluster_sha256.as_str()) {
            return Err("source cluster digests must be disjoint across all splits".to_owned());
        }
        validate_calibration_row(
            &row.calibration_unit_id,
            &row.task_family,
            &row.model_regime,
            &row.features,
            "calibration-A row",
        )?;
        unique_unit_id(&mut all_unit_ids, &row.calibration_unit_id, "calibration-A")?;
        group_a_rows
            .entry(group_key(&row.task_family, &row.model_regime))
            .or_default()
            .push(row);
    }
    for row in &request.calibration_b {
        validate_digest(
            &row.source_cluster_sha256,
            "calibration-B source_cluster_sha256",
        )?;
        if !source_cluster_digests.insert(row.source_cluster_sha256.as_str()) {
            return Err("source cluster digests must be disjoint across all splits".to_owned());
        }
        validate_calibration_row(
            &row.calibration_unit_id,
            &row.task_family,
            &row.model_regime,
            &row.features,
            "calibration-B row",
        )?;
        unique_unit_id(&mut all_unit_ids, &row.calibration_unit_id, "calibration-B")?;
        group_b_rows
            .entry(group_key(&row.task_family, &row.model_regime))
            .or_default()
            .push(row);
    }
    for row in &request.evaluation {
        validate_digest(
            &row.source_cluster_sha256,
            "evaluation source_cluster_sha256",
        )?;
        if !source_cluster_digests.insert(row.source_cluster_sha256.as_str()) {
            return Err("source cluster digests must be disjoint across all splits".to_owned());
        }
        if !valid_id(&row.evaluation_unit_id, 128)
            || !valid_id(&row.task_family, 64)
            || !valid_id(&row.model_regime, 128)
        {
            return Err("evaluation row has invalid unit/group IDs".to_owned());
        }
        validate_evaluation_features(&row.features, "evaluation row")?;
        unique_unit_id(&mut all_unit_ids, &row.evaluation_unit_id, "evaluation")?;
        evaluation_rows
            .entry(group_key(&row.task_family, &row.model_regime))
            .or_default()
            .push(row);
    }

    if group_a_rows.keys().collect::<BTreeSet<_>>() != group_b_rows.keys().collect::<BTreeSet<_>>()
    {
        return Err("calibration-A and calibration-B must cover the same groups".to_owned());
    }
    let mut group_calibrations_a = BTreeMap::<GroupKey, GroupCalibrationA>::new();
    let mut group_calibrations_b = BTreeMap::<GroupKey, GroupCalibrationB>::new();
    for (key, rows_a) in &group_a_rows {
        let rows_b = group_b_rows
            .get(key)
            .ok_or_else(|| "calibration-B group missing".to_owned())?;
        if rows_a.len() < MIN_CALIBRATION_A_UNITS || rows_b.len() < MIN_CALIBRATION_B_UNITS {
            return Err(
                "each group requires at least two independent units in calibration A and B"
                    .to_owned(),
            );
        }

        let mut specs = request.detector_specs.clone();
        specs.sort_by(|left, right| left.detector_id.cmp(&right.detector_id));
        let detector_ids: Vec<String> = specs.iter().map(|spec| spec.detector_id.clone()).collect();
        let expected: BTreeSet<&str> = detector_ids.iter().map(String::as_str).collect();
        let mut directions = BTreeMap::new();
        let mut weights_permille = BTreeMap::new();
        let mut a_scores = BTreeMap::<String, Vec<f64>>::new();
        for spec in &specs {
            directions.insert(spec.detector_id.clone(), spec.direction);
            weights_permille.insert(spec.detector_id.clone(), spec.weight_permille);
            a_scores.insert(spec.detector_id.clone(), Vec::with_capacity(rows_a.len()));
        }

        for row in rows_a {
            let ids: BTreeSet<&str> = row
                .features
                .iter()
                .map(|feature| feature.detector_id.as_str())
                .collect();
            if ids != expected {
                return Err(
                    "calibration-A detector feature sets must exactly match the pinned registry"
                        .to_owned(),
                );
            }
            for feature in &row.features {
                if directions.get(&feature.detector_id) != Some(&feature.direction) {
                    return Err(
                        "calibration-A detector directions must match the pinned registry"
                            .to_owned(),
                    );
                }
                a_scores
                    .get_mut(&feature.detector_id)
                    .ok_or_else(|| "calibration-A detector score vector missing".to_owned())?
                    .push(feature.score);
            }
        }

        let calibration_a = GroupCalibrationA {
            detector_ids,
            directions,
            weights_permille,
            scores: a_scores,
        };

        let mut b_fused = Vec::with_capacity(rows_b.len());
        for row in rows_b {
            let p_values = fused_nonconformity(&row.features, &calibration_a)?;
            b_fused.push(combine_detector_p_values(&p_values)?);
        }
        group_calibrations_a.insert(key.clone(), calibration_a);
        group_calibrations_b.insert(
            key.clone(),
            GroupCalibrationB {
                fused_scores: b_fused,
            },
        );
    }

    let mut score_rows = Vec::with_capacity(request.evaluation.len());
    let mut group_summaries = Vec::new();
    for (key, rows) in &evaluation_rows {
        let Some(calibration_a) = group_calibrations_a.get(key) else {
            let labelled = rows
                .iter()
                .filter(|row| row.ground_truth_contaminated.is_some())
                .count() as u64;
            let clean = rows
                .iter()
                .filter(|row| row.ground_truth_contaminated == Some(false))
                .count() as u64;
            let contaminated = rows
                .iter()
                .filter(|row| row.ground_truth_contaminated == Some(true))
                .count() as u64;
            for row in rows {
                score_rows.push(SelectiveFusionScoreV1 {
                    evaluation_unit_id: row.evaluation_unit_id.clone(),
                    task_family: row.task_family.clone(),
                    model_regime: row.model_regime.clone(),
                    detector_p_values: Vec::new(),
                    fused_nonconformity: None,
                    final_p_rank_numerator: None,
                    final_p_rank_denominator: None,
                    final_p_value: None,
                    by_adjusted_p_value: None,
                    by_family_size: None,
                    minimum_attainable_p_value: None,
                    decision: SelectiveDecisionV1::AbstainUnsupportedGroup,
                    ground_truth_contaminated: row.ground_truth_contaminated,
                });
            }
            group_summaries.push(ConformalGroupSummaryV1 {
                task_family: key.task_family.clone(),
                model_regime: key.model_regime.clone(),
                detector_ids: Vec::new(),
                calibration_a_unit_count: 0,
                calibration_b_unit_count: 0,
                evaluation_unit_count: rows.len() as u64,
                labelled_evaluation_unit_count: labelled,
                flagged_count: 0,
                no_flag_count: 0,
                abstained_count: rows.len() as u64,
                clean_labelled_count: clean,
                contaminated_labelled_count: contaminated,
                false_positive_count: 0,
                true_positive_count: 0,
                false_negative_count: contaminated,
                empirical_fpr_permille: None,
                empirical_tpr_permille: if contaminated == 0 { None } else { Some(0.0) },
                fpr_one_sided_95_upper_permille: None,
                tpr_one_sided_95_lower_permille: None,
                decision_coverage_permille: 0.0,
                minimum_attainable_p_value: 1.0,
                minimum_detector_rank_p_value: 1.0,
                pairwise_rank_correlations: Vec::new(),
                target_alpha_permille: request.target_alpha_permille,
            });
            continue;
        };
        let Some(calibration_b) = group_calibrations_b.get(key) else {
            let labelled = rows
                .iter()
                .filter(|row| row.ground_truth_contaminated.is_some())
                .count() as u64;
            let clean = rows
                .iter()
                .filter(|row| row.ground_truth_contaminated == Some(false))
                .count() as u64;
            let contaminated = rows
                .iter()
                .filter(|row| row.ground_truth_contaminated == Some(true))
                .count() as u64;
            for row in rows {
                score_rows.push(SelectiveFusionScoreV1 {
                    evaluation_unit_id: row.evaluation_unit_id.clone(),
                    task_family: row.task_family.clone(),
                    model_regime: row.model_regime.clone(),
                    detector_p_values: Vec::new(),
                    fused_nonconformity: None,
                    final_p_rank_numerator: None,
                    final_p_rank_denominator: None,
                    final_p_value: None,
                    by_adjusted_p_value: None,
                    by_family_size: None,
                    minimum_attainable_p_value: None,
                    decision: SelectiveDecisionV1::AbstainUnsupportedGroup,
                    ground_truth_contaminated: row.ground_truth_contaminated,
                });
            }
            group_summaries.push(ConformalGroupSummaryV1 {
                task_family: key.task_family.clone(),
                model_regime: key.model_regime.clone(),
                detector_ids: calibration_a.detector_ids.clone(),
                calibration_a_unit_count: group_a_rows.get(key).map_or(0, |v| v.len() as u64),
                calibration_b_unit_count: 0,
                evaluation_unit_count: rows.len() as u64,
                labelled_evaluation_unit_count: labelled,
                flagged_count: 0,
                no_flag_count: 0,
                abstained_count: rows.len() as u64,
                clean_labelled_count: clean,
                contaminated_labelled_count: contaminated,
                false_positive_count: 0,
                true_positive_count: 0,
                false_negative_count: contaminated,
                empirical_fpr_permille: None,
                empirical_tpr_permille: if contaminated == 0 { None } else { Some(0.0) },
                fpr_one_sided_95_upper_permille: None,
                tpr_one_sided_95_lower_permille: None,
                decision_coverage_permille: 0.0,
                minimum_attainable_p_value: 1.0,
                minimum_detector_rank_p_value: 1.0
                    / (calibration_a.scores.values().next().map_or(0, Vec::len) as f64 + 1.0),
                pairwise_rank_correlations: pairwise_detector_spearman(rows, calibration_a),
                target_alpha_permille: request.target_alpha_permille,
            });
            continue;
        };
        let b_count = calibration_b.fused_scores.len() as u64;
        let min_p = 1.0 / (b_count as f64 + 1.0);
        let b_denominator = u128::from(b_count) + 1;
        let alpha_resolvable = u128::from(request.target_alpha_permille) * b_denominator >= 1_000;

        let mut flagged = 0_u64;
        let mut no_flag = 0_u64;
        let mut abstained = 0_u64;
        let mut labelled = 0_u64;
        let mut clean = 0_u64;
        let mut contaminated = 0_u64;
        let mut fp = 0_u64;
        let mut tp = 0_u64;
        let mut fn_count = 0_u64;

        for row in rows {
            let ids: BTreeSet<&str> = row
                .features
                .iter()
                .map(|feature| feature.detector_id.as_str())
                .collect();
            let expected: BTreeSet<&str> = calibration_a
                .detector_ids
                .iter()
                .map(String::as_str)
                .collect();
            let matching_features = ids == expected
                && row.features.iter().all(|feature| {
                    calibration_a.directions.get(&feature.detector_id) == Some(&feature.direction)
                });

            let (p_values, fused_score, final_rank, final_p, decision) = if !matching_features {
                abstained += 1;
                (
                    Vec::new(),
                    None,
                    None,
                    None,
                    SelectiveDecisionV1::AbstainDetectorFeatureMismatch,
                )
            } else {
                let p_values = fused_nonconformity(&row.features, calibration_a)?;
                let fused_score = combine_detector_p_values(&p_values)?;
                let (numerator, denominator, p_value) =
                    outer_conformal_p_value(&calibration_b.fused_scores, fused_score)?;
                if !alpha_resolvable {
                    abstained += 1;
                    (
                        p_values,
                        Some(fused_score),
                        Some((numerator, denominator)),
                        Some(p_value),
                        SelectiveDecisionV1::AbstainInsufficientCalibrationResolution,
                    )
                } else {
                    // Final decisions are assigned after BY correction across
                    // all eligible test units in this group.
                    no_flag += 1;
                    (
                        p_values,
                        Some(fused_score),
                        Some((numerator, denominator)),
                        Some(p_value),
                        SelectiveDecisionV1::NoContaminationFlag,
                    )
                }
            };

            if let Some(label) = row.ground_truth_contaminated {
                labelled += 1;
                if label {
                    contaminated += 1;
                    if decision == SelectiveDecisionV1::FlagContaminated {
                        tp += 1;
                    } else {
                        fn_count += 1;
                    }
                } else {
                    clean += 1;
                    if decision == SelectiveDecisionV1::FlagContaminated {
                        fp += 1;
                    }
                }
            }
            score_rows.push(SelectiveFusionScoreV1 {
                evaluation_unit_id: row.evaluation_unit_id.clone(),
                task_family: row.task_family.clone(),
                model_regime: row.model_regime.clone(),
                detector_p_values: p_values,
                fused_nonconformity: fused_score,
                final_p_rank_numerator: final_rank.map(|rank| rank.0),
                final_p_rank_denominator: final_rank.map(|rank| rank.1),
                final_p_value: final_p,
                by_adjusted_p_value: None,
                by_family_size: None,
                minimum_attainable_p_value: matching_features.then_some(min_p),
                decision,
                ground_truth_contaminated: row.ground_truth_contaminated,
            });
        }

        let fpr = if clean == 0 {
            None
        } else {
            Some(fp as f64 * 1_000.0 / clean as f64)
        };
        let tpr = if contaminated == 0 {
            None
        } else {
            // Abstained and no-flag positive units are both not detected.
            Some((contaminated - fn_count) as f64 * 1_000.0 / contaminated as f64)
        };
        let fpr_one_sided_95_upper_permille = if clean == 0 {
            None
        } else {
            Some(exact_binomial_upper_95(fp, clean)? * 1_000.0)
        };
        let tpr_one_sided_95_lower_permille = if contaminated == 0 {
            None
        } else {
            Some(exact_binomial_lower_95(tp, contaminated)? * 1_000.0)
        };
        group_summaries.push(ConformalGroupSummaryV1 {
            task_family: key.task_family.clone(),
            model_regime: key.model_regime.clone(),
            detector_ids: calibration_a.detector_ids.clone(),
            calibration_a_unit_count: group_a_rows.get(key).map_or(0, |v| v.len() as u64),
            calibration_b_unit_count: b_count,
            evaluation_unit_count: rows.len() as u64,
            labelled_evaluation_unit_count: labelled,
            flagged_count: flagged,
            no_flag_count: no_flag,
            abstained_count: abstained,
            clean_labelled_count: clean,
            contaminated_labelled_count: contaminated,
            false_positive_count: fp,
            true_positive_count: tp,
            false_negative_count: fn_count,
            empirical_fpr_permille: fpr,
            empirical_tpr_permille: tpr,
            fpr_one_sided_95_upper_permille,
            tpr_one_sided_95_lower_permille,
            decision_coverage_permille: (flagged + no_flag) as f64 * 1_000.0 / rows.len() as f64,
            minimum_attainable_p_value: min_p,
            minimum_detector_rank_p_value: 1.0
                / (calibration_a.scores.values().next().map_or(0, Vec::len) as f64 + 1.0),
            pairwise_rank_correlations: pairwise_detector_spearman(rows, calibration_a),
            target_alpha_permille: request.target_alpha_permille,
        });
    }
    apply_by_adjustment(
        &mut score_rows,
        &mut group_summaries,
        request.target_alpha_permille,
    )?;

    score_rows.sort_by(|left, right| {
        (
            &left.task_family,
            &left.model_regime,
            &left.evaluation_unit_id,
        )
            .cmp(&(
                &right.task_family,
                &right.model_regime,
                &right.evaluation_unit_id,
            ))
    });
    group_summaries.sort_by(|left, right| {
        (&left.task_family, &left.model_regime).cmp(&(&right.task_family, &right.model_regime))
    });

    let flagged_count = score_rows
        .iter()
        .filter(|row| row.decision == SelectiveDecisionV1::FlagContaminated)
        .count() as u64;
    let no_flag_count = score_rows
        .iter()
        .filter(|row| row.decision == SelectiveDecisionV1::NoContaminationFlag)
        .count() as u64;
    let abstained_count = score_rows.len() as u64 - flagged_count - no_flag_count;

    let request_bytes = serde_json::to_vec(request)
        .map_err(|error| format!("cannot serialize conformal fusion request: {error}"))?;
    let calibration_a_bytes = canonical_calibration_manifest(&request.calibration_a)?;
    let calibration_b_bytes = canonical_calibration_manifest(&request.calibration_b)?;
    let evaluation_bytes = canonical_evaluation_manifest(&request.evaluation)?;
    let mut detector_specs = request.detector_specs.clone();
    detector_specs.sort_by(|left, right| left.detector_id.cmp(&right.detector_id));
    Ok(DetectorConformalFusionReportV1 {
        schema: CONFORMAL_FUSION_REPORT_SCHEMA.to_owned(),
        fusion_method_id: FUSION_METHOD_ID.to_owned(),
        detector_specs,
        study_id: request.study_id.clone(),
        benchmark_id: request.benchmark_id.clone(),
        benchmark_version: request.benchmark_version.clone(),
        model_checkpoint_sha256: request.model_checkpoint_sha256.clone(),
        tokenizer_sha256: request.tokenizer_sha256.clone(),
        prompt_harness_sha256: request.prompt_harness_sha256.clone(),
        runtime_environment_sha256: request.runtime_environment_sha256.clone(),
        calibration_protocol_sha256: request.calibration_protocol_sha256.clone(),
        detector_registry_sha256: request.detector_registry_sha256.clone(),
        request_sha256: crate::hash_domain(b"detector-conformal-fusion-request.v1", &request_bytes),
        calibration_a_manifest_sha256: crate::hash_domain(
            b"detector-conformal-calibration-a.v1",
            &calibration_a_bytes,
        ),
        calibration_b_manifest_sha256: crate::hash_domain(
            b"detector-conformal-calibration-b.v1",
            &calibration_b_bytes,
        ),
        evaluation_manifest_sha256: crate::hash_domain(
            b"detector-conformal-evaluation.v1",
            &evaluation_bytes,
        ),
        target_alpha_permille: request.target_alpha_permille,
        multiple_testing_method: "benjamini_yekutieli".to_owned(),
        fdr_control_conditional_on_exchangeability: true,
        group_count: group_summaries.len() as u64,
        evaluation_unit_count: score_rows.len() as u64,
        flagged_count,
        no_flag_count,
        abstained_count,
        score_rows,
        group_summaries,
        intervals_multiplicity_adjusted: false,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: "exploratory_conformal_calibration".to_owned(),
        assumptions: vec![
            "Calibration-A is an independent clean reference used to map detector-specific score directions and scales.".to_owned(),
            "Calibration-B is disjoint from Calibration-A and is exchangeable with future clean evaluation units within each task-family × model-regime group.".to_owned(),
            "The final empirical rank is calibrated on the fused statistic itself; detector-feature independence is not assumed.".to_owned(),
            "Benjamini-Yekutieli adjustment is applied within each predeclared task-family × model-regime testing family; its FDR control is conditional on each outer conformal p-value being super-uniform.".to_owned(),
            "Calibration units are independent source/task clusters or other predeclared exchangeability units, not repeated generations or near-duplicate questions.".to_owned(),
            "Arbitrary temporal, language, domain, model or source shift breaks the exchangeability basis; no validity guarantee is claimed under such shift.".to_owned(),
            "Ground-truth labels are used only for evaluation summaries and do not affect calibration, fusion or final p-values.".to_owned(),
        ],
        claim_boundary: "The two-stage empirical-rank fusion produces a group-conditional conformal-style p-value only under the declared clean-cohort and exchangeability assumptions. Benjamini-Yekutieli is applied to eligible test p-values within each predeclared task-family × model-regime family and controls FDR only if each null p-value is super-uniform. It does not prove calibration cohorts are genuinely clean, authenticate detector/model receipts, guarantee validity under arbitrary distribution shift, establish a causal direct effect, or establish simultaneous confidence-interval coverage. Missing features and insufficient calibration resolution lead to abstention. This report remains exploratory.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature(id: &str, score: f64, direction: DetectorScoreDirectionV1) -> DetectorFeatureV1 {
        DetectorFeatureV1 {
            detector_id: id.to_owned(),
            direction,
            score,
        }
    }

    fn row(
        id: &str,
        score_a: f64,
        score_b: f64,
        family: &str,
        regime: &str,
    ) -> CalibrationFeatureRowV1 {
        CalibrationFeatureRowV1 {
            calibration_unit_id: id.to_owned(),
            source_cluster_sha256: crate::hash_domain(b"test-source-cluster.v1", id.as_bytes()),
            task_family: family.to_owned(),
            model_regime: regime.to_owned(),
            features: vec![
                feature(
                    "ftd",
                    score_a,
                    DetectorScoreDirectionV1::HigherMeansContaminated,
                ),
                feature(
                    "dvd",
                    score_b,
                    DetectorScoreDirectionV1::HigherMeansContaminated,
                ),
            ],
        }
    }

    fn evaluation_row(
        id: &str,
        score_a: f64,
        score_b: f64,
        label: Option<bool>,
    ) -> EvaluationFeatureRowV1 {
        EvaluationFeatureRowV1 {
            evaluation_unit_id: id.to_owned(),
            source_cluster_sha256: crate::hash_domain(b"test-source-cluster.v1", id.as_bytes()),
            task_family: "mathematics".to_owned(),
            model_regime: "qwen-rlvr".to_owned(),
            features: vec![
                feature(
                    "ftd",
                    score_a,
                    DetectorScoreDirectionV1::HigherMeansContaminated,
                ),
                feature(
                    "dvd",
                    score_b,
                    DetectorScoreDirectionV1::HigherMeansContaminated,
                ),
            ],
            ground_truth_contaminated: label,
        }
    }

    fn sample_request() -> DetectorConformalFusionRequestV1 {
        let detector_specs = vec![
            DetectorSpecV1 {
                detector_id: "ftd".to_owned(),
                direction: DetectorScoreDirectionV1::HigherMeansContaminated,
                weight_permille: 500,
                feature_contract_sha256: "c".repeat(64),
            },
            DetectorSpecV1 {
                detector_id: "dvd".to_owned(),
                direction: DetectorScoreDirectionV1::HigherMeansContaminated,
                weight_permille: 500,
                feature_contract_sha256: "d".repeat(64),
            },
        ];
        let detector_registry_sha256 = compute_detector_registry_sha256(&detector_specs).unwrap();
        DetectorConformalFusionRequestV1 {
            schema: CONFORMAL_FUSION_REQUEST_SCHEMA.to_owned(),
            study_id: "study-001".to_owned(),
            benchmark_id: "benchmark-001".to_owned(),
            benchmark_version: "v1".to_owned(),
            model_checkpoint_sha256: "e".repeat(64),
            tokenizer_sha256: "f".repeat(64),
            prompt_harness_sha256: "1".repeat(64),
            runtime_environment_sha256: "2".repeat(64),
            calibration_protocol_sha256: "a".repeat(64),
            detector_registry_sha256,
            target_alpha_permille: 500,
            fusion_method_id: FUSION_METHOD_ID.to_owned(),
            detector_specs,
            calibration_a: vec![
                row("a-1", 0.1, 0.1, "mathematics", "qwen-rlvr"),
                row("a-2", 0.2, 0.2, "mathematics", "qwen-rlvr"),
                row("a-3", 0.3, 0.3, "mathematics", "qwen-rlvr"),
                row("a-4", 0.4, 0.4, "mathematics", "qwen-rlvr"),
            ],
            calibration_b: vec![
                row("b-1", 0.1, 0.1, "mathematics", "qwen-rlvr"),
                row("b-2", 0.2, 0.2, "mathematics", "qwen-rlvr"),
                row("b-3", 0.3, 0.3, "mathematics", "qwen-rlvr"),
                row("b-4", 0.4, 0.4, "mathematics", "qwen-rlvr"),
            ],
            evaluation: vec![
                evaluation_row("t-1", 0.9, 0.9, Some(true)),
                evaluation_row("t-2", 0.2, 0.2, Some(false)),
            ],
        }
    }

    #[test]
    fn detector_registry_digest_binds_ids_directions_weights_and_contracts() {
        let request = sample_request();
        assert_eq!(
            compute_detector_registry_sha256(&request.detector_specs).unwrap(),
            request.detector_registry_sha256
        );
        let mut changed = request.detector_specs.clone();
        changed[0].weight_permille = 400;
        changed[1].weight_permille = 600;
        assert_ne!(
            compute_detector_registry_sha256(&changed).unwrap(),
            request.detector_registry_sha256
        );
    }

    #[test]
    fn detector_weights_must_be_fixed_and_sum_to_one_thousand() {
        let mut request = sample_request();
        request.detector_specs[0].weight_permille = 400;
        assert!(compute_detector_registry_sha256(&request.detector_specs).is_err());
        assert!(calibrate_detector_panel(&request).is_err());
    }

    #[test]
    fn benjamini_yekutieli_adjustment_handles_arbitrary_dependence_conservatively() {
        let p = vec![
            ("p1".to_owned(), 0.01),
            ("p2".to_owned(), 0.03),
            ("p3".to_owned(), 0.04),
        ];
        let adjusted = benjamini_yekutieli_adjusted_p_values(&p).unwrap();
        assert!((adjusted["p1"] - 0.055).abs() < 1.0e-12);
        assert!((adjusted["p2"] - (0.04 * 5.5 / 3.0)).abs() < 1.0e-12);
        assert!((adjusted["p3"] - (0.04 * 5.5 / 3.0)).abs() < 1.0e-12);
    }

    #[test]
    fn exact_zero_false_positive_upper_bound_matches_closed_form() {
        let bound = exact_binomial_upper_95(0, 300).unwrap();
        let expected = 1.0 - 0.05_f64.powf(1.0 / 300.0);
        assert!((bound - expected).abs() < 1.0e-10);
    }

    #[test]
    fn exact_all_success_lower_bound_matches_closed_form() {
        let bound = exact_binomial_lower_95(300, 300).unwrap();
        let expected = 0.05_f64.powf(1.0 / 300.0);
        assert!((bound - expected).abs() < 1.0e-10);
    }

    #[test]
    fn exact_binomial_bounds_are_ordered_for_interior_counts() {
        let lower = exact_binomial_lower_95(20, 30).unwrap();
        let upper = exact_binomial_upper_95(20, 30).unwrap();
        assert!(lower < 20.0 / 30.0);
        assert!(upper > 20.0 / 30.0);
        assert!(lower < upper);
    }

    #[test]
    fn spearman_rank_correlation_is_one_for_identical_detector_orderings() {
        let rho = spearman_rank_correlation(&[1.0, 2.0, 2.0, 4.0], &[10.0, 20.0, 20.0, 40.0]);
        assert!((rho.unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn constant_detector_scores_report_undefined_rank_correlation() {
        assert!(spearman_rank_correlation(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]).is_none());
    }

    #[test]
    fn weighted_fusion_preserves_pvalue_contributions() {
        let report = calibrate_detector_panel(&sample_request()).unwrap();
        for row in &report.score_rows {
            if row.detector_p_values.len() == 2 {
                let total_weight: u16 = row
                    .detector_p_values
                    .iter()
                    .map(|p| p.weight_permille)
                    .sum();
                assert_eq!(total_weight, 1_000);
                let sum: f64 = row
                    .detector_p_values
                    .iter()
                    .map(|p| p.weighted_contribution)
                    .sum();
                assert!((sum - row.fused_nonconformity.unwrap()).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn report_applies_by_adjustment_and_records_calibration_manifests() {
        let report = calibrate_detector_panel(&sample_request()).unwrap();
        assert_eq!(report.multiple_testing_method, "benjamini_yekutieli");
        assert!(report.fdr_control_conditional_on_exchangeability);
        assert_eq!(report.calibration_a_manifest_sha256.len(), 64);
        assert_eq!(report.calibration_b_manifest_sha256.len(), 64);
        assert_eq!(report.evaluation_manifest_sha256.len(), 64);
    }

    #[test]
    fn hand_checkable_json_fixture_round_trips_and_calibrates() {
        let request: DetectorConformalFusionRequestV1 = serde_json::from_str(include_str!(
            "../fixtures/detector-conformal-fusion-request.hand.v1.json"
        ))
        .unwrap();
        let report = calibrate_detector_panel(&request).unwrap();
        assert_eq!(report.evaluation_unit_count, 2);
        assert_eq!(report.multiple_testing_method, "benjamini_yekutieli");
        assert!(!report.confirmatory_eligible);
        assert_eq!(report.calibration_a_manifest_sha256.len(), 64);
        assert_eq!(report.calibration_b_manifest_sha256.len(), 64);
        assert_eq!(report.evaluation_manifest_sha256.len(), 64);
    }

    #[test]
    fn outer_calibration_handles_dependent_detector_scores() {
        let report = calibrate_detector_panel(&sample_request()).unwrap();
        assert_eq!(report.score_rows.len(), 2);
        assert_eq!(report.group_summaries.len(), 1);
        assert!(!report.independently_authenticated);
        assert!(!report.confirmatory_eligible);
        assert_eq!(report.score_rows[0].final_p_rank_denominator, Some(5));
    }

    #[test]
    fn higher_and_lower_detector_directions_are_respected() {
        let (n, d, p) = detector_rank_p_value(
            &[0.1, 0.2, 0.3],
            DetectorScoreDirectionV1::LowerMeansContaminated,
            0.05,
        )
        .unwrap();
        assert_eq!(n, 1);
        assert_eq!(d, 4);
        assert_eq!(p, 0.25);
    }

    #[test]
    fn detector_rank_ties_are_conservative() {
        let (n, d, p) = detector_rank_p_value(
            &[0.1, 0.2, 0.2, 0.3],
            DetectorScoreDirectionV1::HigherMeansContaminated,
            0.2,
        )
        .unwrap();
        assert_eq!(n, 4);
        assert_eq!(d, 5);
        assert_eq!(p, 0.8);
    }

    #[test]
    fn duplicate_units_across_splits_are_rejected() {
        let mut request = sample_request();
        request.calibration_b[0].calibration_unit_id = "a-1".to_owned();
        assert!(calibrate_detector_panel(&request).is_err());
    }

    #[test]
    fn missing_calibration_group_is_rejected() {
        let mut request = sample_request();
        request.calibration_b.pop();
        request.calibration_b.iter_mut().for_each(|row| {
            row.task_family = "code".to_owned();
        });
        assert!(calibrate_detector_panel(&request).is_err());
    }

    #[test]
    fn missing_evaluation_feature_abstains_without_zero_imputation() {
        let mut request = sample_request();
        request.evaluation[0].features.pop();
        let report = calibrate_detector_panel(&request).unwrap();
        assert_eq!(
            report.score_rows[0].decision,
            SelectiveDecisionV1::AbstainDetectorFeatureMismatch
        );
        assert!(report.score_rows[0].final_p_value.is_none());
    }

    #[test]
    fn insufficient_calibration_resolution_abstains() {
        let mut request = sample_request();
        request.target_alpha_permille = 100; // 10%, n_B=4 -> minimum p=20%.
        let report = calibrate_detector_panel(&request).unwrap();
        assert_eq!(
            report.score_rows[0].decision,
            SelectiveDecisionV1::AbstainInsufficientCalibrationResolution
        );
    }

    #[test]
    fn unsupported_evaluation_group_abstains_instead_of_using_another_group() {
        let mut request = sample_request();
        request.evaluation[0].task_family = "code".to_owned();
        let report = calibrate_detector_panel(&request).unwrap();
        let row = report
            .score_rows
            .iter()
            .find(|row| row.evaluation_unit_id == "t-1")
            .unwrap();
        assert_eq!(row.decision, SelectiveDecisionV1::AbstainUnsupportedGroup);
        assert!(row.final_p_value.is_none());
    }

    #[test]
    fn abstentions_count_as_false_negatives_for_labelled_contamination() {
        let mut request = sample_request();
        request.evaluation[0].features.pop();
        let report = calibrate_detector_panel(&request).unwrap();
        let summary = &report.group_summaries[0];
        assert_eq!(summary.contaminated_labelled_count, 1);
        assert_eq!(summary.true_positive_count, 0);
        assert_eq!(summary.false_negative_count, 1);
        assert_eq!(summary.empirical_tpr_permille, Some(0.0));
    }

    #[test]
    fn detector_directions_must_match_across_calibration_rows() {
        let mut request = sample_request();
        request.calibration_a[1].features[0].direction =
            DetectorScoreDirectionV1::LowerMeansContaminated;
        assert!(calibrate_detector_panel(&request).is_err());
    }

    #[test]
    fn a_missing_detector_in_calibration_is_a_hard_error() {
        let mut request = sample_request();
        request.calibration_b[0].features.pop();
        assert!(calibrate_detector_panel(&request).is_err());
    }
}
