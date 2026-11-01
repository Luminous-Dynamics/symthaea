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

pub const CONFORMAL_FUSION_REQUEST_SCHEMA: &str =
    "symthaea.detector_conformal_fusion_request.v1";
pub const CONFORMAL_FUSION_REPORT_SCHEMA: &str =
    "symthaea.detector_conformal_fusion_report.v1";
pub const FUSION_METHOD_ID: &str = "sum_negative_log_empirical_detector_pvalues_v1";
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
pub struct CalibrationFeatureRowV1 {
    /// One independent source/task cluster. Repeated items/generations are nested.
    pub calibration_unit_id: String,
    pub task_family: String,
    pub model_regime: String,
    pub features: Vec<DetectorFeatureV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationFeatureRowV1 {
    pub evaluation_unit_id: String,
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
    pub calibration_protocol_sha256: String,
    pub detector_registry_sha256: String,
    /// Alpha in permille; 10 means 1%, 50 means 5%.
    pub target_alpha_permille: u16,
    pub fusion_method_id: String,
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
    pub decision_coverage_permille: f64,
    pub minimum_attainable_p_value: f64,
    pub target_alpha_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConformalFusionReportV1 {
    pub schema: String,
    pub fusion_method_id: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub calibration_protocol_sha256: String,
    pub detector_registry_sha256: String,
    pub request_sha256: String,
    pub target_alpha_permille: u16,
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
    if features.len() < MIN_DETECTORS_PER_GROUP
        || features.len() > MAX_DETECTORS_PER_ROW
    {
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
    if !valid_id(unit_id, 128)
        || !valid_id(task_family, 64)
        || !valid_id(model_regime, 128)
    {
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
    Ok((numerator, denominator, numerator as f64 / denominator as f64))
}

fn fused_nonconformity(
    features: &[DetectorFeatureV1],
    calibration: &GroupCalibrationA,
) -> Result<Vec<DetectorEmpiricalPValueV1>, String> {
    let feature_ids: Vec<String> = {
        let mut ids: Vec<String> = features.iter().map(|feature| feature.detector_id.clone()).collect();
        ids.sort();
        ids
    };
    if feature_ids != calibration.detector_ids {
        return Err("evaluation detector feature set does not match calibration contract".to_owned());
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
        p_values.push(DetectorEmpiricalPValueV1 {
            detector_id: feature.detector_id.clone(),
            direction: feature.direction,
            raw_score: feature.score,
            rank_numerator,
            rank_denominator,
            p_value,
        });
    }
    p_values.sort_by(|left, right| left.detector_id.cmp(&right.detector_id));
    Ok(p_values)
}

fn combine_detector_p_values(p_values: &[DetectorEmpiricalPValueV1]) -> Result<f64, String> {
    if p_values.len() < MIN_DETECTORS_PER_GROUP {
        return Err("fused statistic requires at least two detector p-values".to_owned());
    }
    let result = p_values.iter().map(|value| -value.p_value.ln()).sum::<f64>();
    if result.is_finite() {
        Ok(result)
    } else {
        Err("fused nonconformity statistic is non-finite".to_owned())
    }
}

fn unique_unit_id(
    set: &mut BTreeSet<String>,
    value: &str,
    label: &str,
) -> Result<(), String> {
    if !set.insert(value.to_owned()) {
        return Err(format!("calibration/evaluation unit ID {value} is repeated across {label} splits"));
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
    Ok((numerator, denominator, numerator as f64 / denominator as f64))
}

pub fn calibrate_detector_panel(
    request: &DetectorConformalFusionRequestV1,
) -> Result<DetectorConformalFusionReportV1, String> {
    if request.schema != CONFORMAL_FUSION_REQUEST_SCHEMA {
        return Err(format!("request schema must be {CONFORMAL_FUSION_REQUEST_SCHEMA}"));
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
    validate_digest(&request.calibration_protocol_sha256, "calibration_protocol_sha256")?;
    validate_digest(&request.detector_registry_sha256, "detector_registry_sha256")?;
    if request.calibration_a.len() < 2
        || request.calibration_a.len() > MAX_CALIBRATION_ROWS
        || request.calibration_b.len() < 2
        || request.calibration_b.len() > MAX_CALIBRATION_ROWS
        || request.evaluation.is_empty()
        || request.evaluation.len() > MAX_EVALUATION_ROWS
    {
        return Err("calibration A/B and evaluation cohorts must be nonempty and bounded".to_owned());
    }

    let mut all_unit_ids = BTreeSet::new();
    let mut group_a_rows =
        BTreeMap::<GroupKey, Vec<&CalibrationFeatureRowV1>>::new();
    let mut group_b_rows =
        BTreeMap::<GroupKey, Vec<&CalibrationFeatureRowV1>>::new();
    let mut evaluation_rows =
        BTreeMap::<GroupKey, Vec<&EvaluationFeatureRowV1>>::new();

    for row in &request.calibration_a {
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
        if !valid_id(&row.evaluation_unit_id, 128)
            || !valid_id(&row.task_family, 64)
            || !valid_id(&row.model_regime, 128)
        {
            return Err("evaluation row has invalid unit/group IDs".to_owned());
        }
        validate_features(&row.features, "evaluation row")?;
        unique_unit_id(&mut all_unit_ids, &row.evaluation_unit_id, "evaluation")?;
        evaluation_rows
            .entry(group_key(&row.task_family, &row.model_regime))
            .or_default()
            .push(row);
    }

    if group_a_rows.keys().collect::<BTreeSet<_>>()
        != group_b_rows.keys().collect::<BTreeSet<_>>()
    {
        return Err("calibration-A and calibration-B must cover the same groups".to_owned());
    }
    if group_a_rows.keys().collect::<BTreeSet<_>>()
        != evaluation_rows.keys().collect::<BTreeSet<_>>()
    {
        return Err("every evaluation group must have matching A and B calibration groups; unsupported groups are rejected in v1".to_owned());
    }

    let mut group_calibrations_a = BTreeMap::<GroupKey, GroupCalibrationA>::new();
    let mut group_calibrations_b = BTreeMap::<GroupKey, GroupCalibrationB>::new();
    for (key, rows_a) in &group_a_rows {
        let rows_b = group_b_rows
            .get(key)
            .ok_or_else(|| "calibration-B group missing".to_owned())?;
        if rows_a.len() < MIN_CALIBRATION_A_UNITS
            || rows_b.len() < MIN_CALIBRATION_B_UNITS
        {
            return Err("each group requires at least two independent units in calibration A and B".to_owned());
        }

        let first_features = &rows_a[0].features;
        let mut detector_ids: Vec<String> =
            first_features.iter().map(|feature| feature.detector_id.clone()).collect();
        detector_ids.sort();
        let mut directions = BTreeMap::new();
        let mut a_scores = BTreeMap::<String, Vec<f64>>::new();
        for feature in first_features {
            directions.insert(feature.detector_id.clone(), feature.direction);
            a_scores.insert(feature.detector_id.clone(), Vec::with_capacity(rows_a.len()));
        }

        for row in rows_a {
            let ids: BTreeSet<&str> = row.features.iter().map(|feature| feature.detector_id.as_str()).collect();
            let expected: BTreeSet<&str> = detector_ids.iter().map(String::as_str).collect();
            if ids != expected {
                return Err("calibration-A detector feature sets must match within group".to_owned());
            }
            for feature in &row.features {
                if directions.get(&feature.detector_id) != Some(&feature.direction) {
                    return Err("calibration-A detector directions must be stable within group".to_owned());
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
            scores: a_scores,
        };

        let mut b_fused = Vec::with_capacity(rows_b.len());
        for row in rows_b {
            let p_values = fused_nonconformity(&row.features, &calibration_a)?;
            b_fused.push(combine_detector_p_values(&p_values)?);
        }
        group_calibrations_a.insert(key.clone(), calibration_a);
        group_calibrations_b.insert(key.clone(), GroupCalibrationB {
            fused_scores: b_fused,
        });
    }

    let mut score_rows = Vec::with_capacity(request.evaluation.len());
    let mut group_summaries = Vec::new();
    for (key, rows) in &evaluation_rows {
        let calibration_a = group_calibrations_a
            .get(key)
            .ok_or_else(|| "calibration-A group missing".to_owned())?;
        let calibration_b = group_calibrations_b
            .get(key)
            .ok_or_else(|| "calibration-B group missing".to_owned())?;
        let b_count = calibration_b.fused_scores.len() as u64;
        let min_p = 1.0 / (b_count as f64 + 1.0);
        let alpha_resolvable =
            f64::from(request.target_alpha_permille) / 1_000.0 >= min_p;

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
            let ids: BTreeSet<&str> =
                row.features.iter().map(|feature| feature.detector_id.as_str()).collect();
            let expected: BTreeSet<&str> =
                calibration_a.detector_ids.iter().map(String::as_str).collect();
            let matching_features = ids == expected
                && row.features.iter().all(|feature| {
                    calibration_a.directions.get(&feature.detector_id) == Some(&feature.direction)
                });

            let (p_values, fused_score, final_rank, final_p, decision) =
                if !matching_features {
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
                    } else if numerator as f64 / denominator as f64
                        <= f64::from(request.target_alpha_permille) / 1_000.0
                    {
                        flagged += 1;
                        (
                            p_values,
                            Some(fused_score),
                            Some((numerator, denominator)),
                            Some(p_value),
                            SelectiveDecisionV1::FlagContaminated,
                        )
                    } else {
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
                    if decision == SelectiveDecisionV1::NoContaminationFlag {
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
                minimum_attainable_p_value: matching_features.then_some(min_p),
                decision,
                ground_truth_contaminated: row.ground_truth_contaminated,
            });
        }

        let fpr = if clean == 0 { None } else { Some(fp as f64 * 1_000.0 / clean as f64) };
        let tpr = if contaminated == 0 { None } else {
            // Abstained and no-flag positive units are both not detected.
            Some((contaminated - fn_count) as f64 * 1_000.0 / contaminated as f64)
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
            true_positive_count: contaminated - fn_count,
            false_negative_count: fn_count,
            empirical_fpr_permille: fpr,
            empirical_tpr_permille: tpr,
            decision_coverage_permille: (flagged + no_flag) as f64 * 1_000.0 / rows.len() as f64,
            minimum_attainable_p_value: min_p,
            target_alpha_permille: request.target_alpha_permille,
        });
    }
    score_rows.sort_by(|left, right| {
        (&left.task_family, &left.model_regime, &left.evaluation_unit_id)
            .cmp(&(&right.task_family, &right.model_regime, &right.evaluation_unit_id))
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
    Ok(DetectorConformalFusionReportV1 {
        schema: CONFORMAL_FUSION_REPORT_SCHEMA.to_owned(),
        fusion_method_id: FUSION_METHOD_ID.to_owned(),
        study_id: request.study_id.clone(),
        benchmark_id: request.benchmark_id.clone(),
        benchmark_version: request.benchmark_version.clone(),
        calibration_protocol_sha256: request.calibration_protocol_sha256.clone(),
        detector_registry_sha256: request.detector_registry_sha256.clone(),
        request_sha256: crate::hash_domain(b"detector-conformal-fusion-request.v1", &request_bytes),
        target_alpha_permille: request.target_alpha_permille,
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
            "Calibration units are independent source/task clusters or other predeclared exchangeability units, not repeated generations or near-duplicate questions.".to_owned(),
            "Arbitrary temporal, language, domain, model or source shift breaks the exchangeability basis; no validity guarantee is claimed under such shift.".to_owned(),
            "Ground-truth labels are used only for evaluation summaries and do not affect calibration, fusion or final p-values.".to_owned(),
        ],
        claim_boundary: "The two-stage empirical-rank fusion produces a group-conditional conformal-style p-value only under the declared clean-cohort and exchangeability assumptions. It does not prove calibration cohorts are genuinely clean, authenticate detector/model receipts, guarantee validity under arbitrary distribution shift, establish a causal direct effect, or imply FDR control across multiple test items. Missing features and insufficient calibration resolution lead to abstention. This report is exploratory and multiplicity-adjustment is not applied.".to_owned(),
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
            task_family: family.to_owned(),
            model_regime: regime.to_owned(),
            features: vec![
                feature("ftd", score_a, DetectorScoreDirectionV1::HigherMeansContaminated),
                feature("dvd", score_b, DetectorScoreDirectionV1::HigherMeansContaminated),
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
            task_family: "mathematics".to_owned(),
            model_regime: "qwen-rlvr".to_owned(),
            features: vec![
                feature("ftd", score_a, DetectorScoreDirectionV1::HigherMeansContaminated),
                feature("dvd", score_b, DetectorScoreDirectionV1::HigherMeansContaminated),
            ],
            ground_truth_contaminated: label,
        }
    }

    fn sample_request() -> DetectorConformalFusionRequestV1 {
        DetectorConformalFusionRequestV1 {
            schema: CONFORMAL_FUSION_REQUEST_SCHEMA.to_owned(),
            study_id: "study-001".to_owned(),
            benchmark_id: "benchmark-001".to_owned(),
            benchmark_version: "v1".to_owned(),
            calibration_protocol_sha256: "a".repeat(64),
            detector_registry_sha256: "b".repeat(64),
            target_alpha_permille: 500,
            fusion_method_id: FUSION_METHOD_ID.to_owned(),
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
            0.1,
        )
        .unwrap();
        assert_eq!(n, 4);
        assert_eq!(d, 4);
        assert_eq!(p, 1.0);
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
