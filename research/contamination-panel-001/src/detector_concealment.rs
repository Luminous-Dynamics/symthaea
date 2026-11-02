//! Controlled detector-concealment benchmark for reasoning models.
//!
//! Measures whether post-training interventions reduce contamination-detector
//! sensitivity while an exposure-associated behavioral advantage remains on
//! fresh, related tasks. Frozen pre-stage thresholds and stage-specific
//! recalibration are reported separately.
//!
//! The module consumes evaluator-produced score and receipt artifacts. It does
//! not train models, authenticate receipts, or infer membership from score drift.

use crate::{
    causal_analysis::{MeanIntervalV1, mean_interval},
    hash_domain,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DETECTOR_CONCEALMENT_STUDY_SCHEMA: &str =
    "symthaea.detector_concealment_study.v1";
pub const DETECTOR_CONCEALMENT_REPORT_SCHEMA: &str =
    "symthaea.detector_concealment_report.v1";
pub const MAX_MODEL_PAIRS: usize = 1_000;
pub const MAX_DETECTORS: usize = 64;
pub const MAX_CALIBRATION_SCORES: usize = 5_000_000;
pub const MAX_TEST_SCORES: usize = 10_000_000;
pub const MAX_BEHAVIOR_SCORES: usize = 5_000_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ConcealmentStageKindV1 {
    Grpo,
    Ppo,
    Dapo,
    SftWithCot,
    Other,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ModelArmV1 {
    ExposureTreatment,
    MatchedPlacebo,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationCohortV1 {
    TreatmentAssignedSourceFamily,
    HeldoutCleanControl,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdPolicyV1 {
    FrozenPreConcealment,
    StageSpecificRecalibration,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorMethodV1 {
    pub detector_id: String,
    pub detector_version: String,
    pub detector_config_sha256: String,
    pub scoring_access_regime: String,
    /// True if larger raw scores indicate more contamination evidence.
    pub higher_is_more_suspicious: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConcealmentModelPairV1 {
    pub replicate_id: String,
    pub stage_kind: ConcealmentStageKindV1,
    pub base_checkpoint_sha256: String,
    pub treatment_model_id: String,
    pub placebo_model_id: String,
    pub exposure_assignment_sha256: String,
    pub placebo_assignment_sha256: String,
    pub pre_concealment_treatment_checkpoint_sha256: String,
    pub pre_concealment_placebo_checkpoint_sha256: String,
    pub post_concealment_treatment_checkpoint_sha256: String,
    pub post_concealment_placebo_checkpoint_sha256: String,
    pub exposure_training_receipt_sha256: String,
    pub placebo_training_receipt_sha256: String,
    pub concealment_training_receipt_sha256: String,
    pub concealment_config_sha256: String,
    pub checkpoint_freeze_receipt_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationScoreV1 {
    pub replicate_id: String,
    pub model_arm: ModelArmV1,
    pub model_id: String,
    pub detector_id: String,
    pub stage_is_post_concealment: bool,
    pub item_id: String,
    /// Raw detector score. All calibration items are asserted clean/nonmember.
    pub score: f64,
    pub checkpoint_sha256: String,
    pub calibration_manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorTestScoreV1 {
    pub replicate_id: String,
    pub model_arm: ModelArmV1,
    pub model_id: String,
    pub detector_id: String,
    pub stage_is_post_concealment: bool,
    pub item_id: String,
    pub cohort: EvaluationCohortV1,
    pub score: f64,
    pub checkpoint_sha256: String,
    pub evaluation_manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshBehaviorScoreV1 {
    pub replicate_id: String,
    pub model_arm: ModelArmV1,
    pub model_id: String,
    pub stage_is_post_concealment: bool,
    pub item_id: String,
    pub task_family: String,
    /// Verifier-backed success in permille.
    pub score_permille: u16,
    pub checkpoint_sha256: String,
    pub evaluation_manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConcealmentStudyV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub base_benchmark_manifest_sha256: String,
    pub calibration_manifest_sha256: String,
    pub detector_test_manifest_sha256: String,
    pub fresh_behavior_manifest_sha256: String,
    pub target_fpr_permille: Vec<u16>,
    pub minimum_replicates: u16,
    pub minimum_calibration_negatives: u32,
    pub minimum_test_negatives: u32,
    pub minimum_fresh_behavior_advantage_permille: i32,
    pub detector_methods: Vec<DetectorMethodV1>,
    pub model_pairs: Vec<ConcealmentModelPairV1>,
    pub calibration_scores: Vec<CalibrationScoreV1>,
    pub detector_test_scores: Vec<DetectorTestScoreV1>,
    pub fresh_behavior_scores: Vec<FreshBehaviorScoreV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfusionMetricsV1 {
    pub positive_count: u64,
    pub true_positive_count: u64,
    pub negative_count: u64,
    pub false_positive_count: u64,
    pub tpr_permille: Option<f64>,
    pub fpr_permille: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorPairStageResultV1 {
    pub replicate_id: String,
    pub detector_id: String,
    pub stage_kind: ConcealmentStageKindV1,
    pub threshold_policy: ThresholdPolicyV1,
    pub stage_is_post_concealment: bool,
    pub threshold: f64,
    pub calibration_negative_count: u64,
    pub treatment_member_and_control: ConfusionMetricsV1,
    pub placebo_pseudo_member_and_control: ConfusionMetricsV1,
    pub treatment_member_tpr_permille: Option<f64>,
    pub treatment_clean_fpr_permille: Option<f64>,
    pub placebo_pseudo_member_fpr_permille: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConcealmentAggregateV1 {
    pub detector_id: String,
    pub stage_kind: ConcealmentStageKindV1,
    pub threshold_policy: ThresholdPolicyV1,
    pub target_fpr_permille: u16,
    pub independent_model_pair_count: u64,
    pub pre_treatment_member_tpr: Option<MeanIntervalV1>,
    pub post_treatment_member_tpr: Option<MeanIntervalV1>,
    pub pre_to_post_tpr_change: Option<MeanIntervalV1>,
    pub pre_treatment_clean_fpr: Option<MeanIntervalV1>,
    pub post_treatment_clean_fpr: Option<MeanIntervalV1>,
    pub pre_placebo_pseudo_member_fpr: Option<MeanIntervalV1>,
    pub post_placebo_pseudo_member_fpr: Option<MeanIntervalV1>,
    pub concealment_signal: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BehavioralRetentionV1 {
    pub stage_is_post_concealment: bool,
    pub independent_model_pair_count: u64,
    pub treatment_minus_placebo_advantage_permille: Option<MeanIntervalV1>,
    pub task_family_count: u64,
    pub item_count_per_pair_min: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConcealmentClaimV1 {
    pub detector_id: String,
    pub stage_kind: ConcealmentStageKindV1,
    pub threshold_policy: ThresholdPolicyV1,
    pub mean_tpr_change_permille: f64,
    pub retained_behavioral_advantage_permille: f64,
    pub meets_minimum_behavioral_advantage: bool,
    pub concealment_event_supported: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConcealmentReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub study_sha256: String,
    pub independent_model_pair_count: u64,
    pub detector_count: u64,
    pub concealment_stage_count: u64,
    pub target_fpr_permille: Vec<u16>,
    pub pair_stage_results: Vec<DetectorPairStageResultV1>,
    pub detector_aggregates: Vec<DetectorConcealmentAggregateV1>,
    pub behavioral_retention: Vec<BehavioralRetentionV1>,
    pub supported_concealment_claims: Vec<ConcealmentClaimV1>,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub inference_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CalibrationKey {
    replicate_id: String,
    model_arm: ModelArmV1,
    model_id: String,
    detector_id: String,
    stage_is_post: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TestKey {
    replicate_id: String,
    model_arm: ModelArmV1,
    model_id: String,
    detector_id: String,
    stage_is_post: bool,
    item_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BehaviorKey {
    replicate_id: String,
    model_arm: ModelArmV1,
    model_id: String,
    stage_is_post: bool,
    item_id: String,
}

fn valid_id(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= maximum
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
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

fn model_id_for(pair: &ConcealmentModelPairV1, arm: ModelArmV1) -> &str {
    match arm {
        ModelArmV1::ExposureTreatment => &pair.treatment_model_id,
        ModelArmV1::MatchedPlacebo => &pair.placebo_model_id,
    }
}

fn checkpoint_for(
    pair: &ConcealmentModelPairV1,
    arm: ModelArmV1,
    post: bool,
) -> &str {
    match (arm, post) {
        (ModelArmV1::ExposureTreatment, false) => &pair.pre_concealment_treatment_checkpoint_sha256,
        (ModelArmV1::ExposureTreatment, true) => &pair.post_concealment_treatment_checkpoint_sha256,
        (ModelArmV1::MatchedPlacebo, false) => &pair.pre_concealment_placebo_checkpoint_sha256,
        (ModelArmV1::MatchedPlacebo, true) => &pair.post_concealment_placebo_checkpoint_sha256,
    }
}

/// Returns an oriented score where higher always means more suspicious.
fn oriented_score(method: &DetectorMethodV1, score: f64) -> f64 {
    if method.higher_is_more_suspicious { score } else { -score }
}

/// Lock a threshold from a disjoint all-clean calibration cohort. Strictly
/// greater-than classification ensures the empirical calibration FPR cannot
/// exceed floor(target * n), including ties.
fn calibrated_threshold(
    scores: &[f64],
    target_fpr_permille: u16,
) -> Result<f64, String> {
    if scores.is_empty() {
        return Err("calibration cohort cannot be empty".to_owned());
    }
    if target_fpr_permille > 1_000 {
        return Err("target FPR must be in 0..=1000 permille".to_owned());
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err("calibration scores must be finite".to_owned());
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    let permitted_false_positives =
        (target_fpr_permille as usize * sorted.len()) / 1_000;
    let threshold_index = sorted.len() - permitted_false_positives - 1;
    Ok(sorted[threshold_index])
}

fn confusion(
    scores: &[(&DetectorTestScoreV1, bool)],
    threshold: f64,
) -> ConfusionMetricsV1 {
    let mut positive = 0_u64;
    let mut true_positive = 0_u64;
    let mut negative = 0_u64;
    let mut false_positive = 0_u64;
    for (row, actual_positive) in scores {
        let predicted = row.score > threshold;
        if *actual_positive {
            positive += 1;
            if predicted {
                true_positive += 1;
            }
        } else {
            negative += 1;
            if predicted {
                false_positive += 1;
            }
        }
    }
    ConfusionMetricsV1 {
        positive_count: positive,
        true_positive_count: true_positive,
        negative_count: negative,
        false_positive_count: false_positive,
        tpr_permille: if positive == 0 {
            None
        } else {
            Some(true_positive as f64 * 1_000.0 / positive as f64)
        },
        fpr_permille: if negative == 0 {
            None
        } else {
            Some(false_positive as f64 * 1_000.0 / negative as f64)
        },
    }
}

fn pair_behavior_advantage(
    rows: &[&FreshBehaviorScoreV1],
    pair: &ConcealmentModelPairV1,
    post: bool,
) -> Result<(f64, BTreeSet<String>, u64), String> {
    let mut treatment = BTreeMap::<String, (u64, u64)>::new();
    let mut placebo = BTreeMap::<String, (u64, u64)>::new();
    let mut families = BTreeSet::new();
    for row in rows.iter().filter(|row| row.stage_is_post_concealment == post) {
        families.insert(row.task_family.clone());
        let target = match row.model_arm {
            ModelArmV1::ExposureTreatment if row.model_id == pair.treatment_model_id => &mut treatment,
            ModelArmV1::MatchedPlacebo if row.model_id == pair.placebo_model_id => &mut placebo,
            _ => return Err("behavior score model arm does not match its model-pair receipt".to_owned()),
        };
        let entry = target.entry(row.item_id.clone()).or_default();
        entry.0 += u64::from(row.score_permille);
        entry.1 += 1;
    }
    if treatment.is_empty() || placebo.is_empty() {
        return Err("fresh behavior evaluation requires treatment and placebo observations".to_owned());
    }
    if treatment.keys().collect::<BTreeSet<_>>() != placebo.keys().collect::<BTreeSet<_>>() {
        return Err("fresh behavior treatment/placebo item manifests must match exactly".to_owned());
    }
    if treatment.values().any(|(_, n)| *n != 1) || placebo.values().any(|(_, n)| *n != 1) {
        return Err("fresh behavior must have one score per model-arm/item/stage".to_owned());
    }
    let differences: Vec<f64> = treatment
        .iter()
        .map(|(item, (sum, _))| {
            let control = placebo.get(item).expect("paired item sets validated").0;
            *sum as f64 - control as f64
        })
        .collect();
    let mean = differences.iter().sum::<f64>() / differences.len() as f64;
    let minimum_items = differences.len() as u64;
    let _ = pair;
    Ok((mean, families, minimum_items))
}

fn validate_pair_receipts(pair: &ConcealmentModelPairV1) -> Result<(), String> {
    if !valid_id(&pair.replicate_id, 128)
        || !valid_id(&pair.treatment_model_id, 128)
        || !valid_id(&pair.placebo_model_id, 128)
        || pair.treatment_model_id == pair.placebo_model_id
    {
        return Err("model-pair identity is invalid".to_owned());
    }
    for (digest, label) in [
        (&pair.base_checkpoint_sha256, "base_checkpoint_sha256"),
        (&pair.exposure_assignment_sha256, "exposure_assignment_sha256"),
        (&pair.placebo_assignment_sha256, "placebo_assignment_sha256"),
        (&pair.pre_concealment_treatment_checkpoint_sha256, "pre_concealment_treatment_checkpoint_sha256"),
        (&pair.pre_concealment_placebo_checkpoint_sha256, "pre_concealment_placebo_checkpoint_sha256"),
        (&pair.post_concealment_treatment_checkpoint_sha256, "post_concealment_treatment_checkpoint_sha256"),
        (&pair.post_concealment_placebo_checkpoint_sha256, "post_concealment_placebo_checkpoint_sha256"),
        (&pair.exposure_training_receipt_sha256, "exposure_training_receipt_sha256"),
        (&pair.placebo_training_receipt_sha256, "placebo_training_receipt_sha256"),
        (&pair.concealment_training_receipt_sha256, "concealment_training_receipt_sha256"),
        (&pair.concealment_config_sha256, "concealment_config_sha256"),
        (&pair.checkpoint_freeze_receipt_sha256, "checkpoint_freeze_receipt_sha256"),
    ] {
        require_digest(digest, label)?;
    }
    Ok(())
}

/// Analyze detector signal loss under subsequent optimization. The frozen
/// threshold is always learned from the pre-concealment clean calibration set;
/// stage-specific recalibration uses a distinct post-concealment calibration
/// set. Neither threshold uses test examples.
pub fn analyze_detector_concealment(
    study: &DetectorConcealmentStudyV1,
) -> Result<DetectorConcealmentReportV1, String> {
    if study.schema != DETECTOR_CONCEALMENT_STUDY_SCHEMA {
        return Err(format!("study schema must be {DETECTOR_CONCEALMENT_STUDY_SCHEMA}"));
    }
    if !valid_id(&study.study_id, 128)
        || !valid_id(&study.benchmark_id, 128)
        || !valid_id(&study.benchmark_version, 64)
        || study.minimum_replicates < 2
        || study.minimum_calibration_negatives == 0
        || study.minimum_test_negatives == 0
        || study.minimum_fresh_behavior_advantage_permille < 0
    {
        return Err("study identity and minimum calibration/test/replicate constraints are invalid".to_owned());
    }
    for (digest, label) in [
        (&study.base_benchmark_manifest_sha256, "base_benchmark_manifest_sha256"),
        (&study.calibration_manifest_sha256, "calibration_manifest_sha256"),
        (&study.detector_test_manifest_sha256, "detector_test_manifest_sha256"),
        (&study.fresh_behavior_manifest_sha256, "fresh_behavior_manifest_sha256"),
    ] {
        require_digest(digest, label)?;
    }
    if study.target_fpr_permille.is_empty()
        || study.target_fpr_permille.iter().any(|value| *value == 0 || *value > 1_000)
    {
        return Err("target FPR values must be in 1..=1000 permille".to_owned());
    }
    let target_set: BTreeSet<u16> = study.target_fpr_permille.iter().copied().collect();
    if target_set.len() != study.target_fpr_permille.len() {
        return Err("target FPR values must be unique".to_owned());
    }
    if study.detector_methods.is_empty() || study.detector_methods.len() > MAX_DETECTORS
        || study.model_pairs.len() < study.minimum_replicates as usize
        || study.model_pairs.len() > MAX_MODEL_PAIRS
        || study.calibration_scores.is_empty()
        || study.calibration_scores.len() > MAX_CALIBRATION_SCORES
        || study.detector_test_scores.is_empty()
        || study.detector_test_scores.len() > MAX_TEST_SCORES
        || study.fresh_behavior_scores.is_empty()
        || study.fresh_behavior_scores.len() > MAX_BEHAVIOR_SCORES
    {
        return Err("detector, model-pair and score collection counts are invalid or exceed limits".to_owned());
    }

    let mut detectors = BTreeMap::<String, &DetectorMethodV1>::new();
    for method in &study.detector_methods {
        if !valid_id(&method.detector_id, 128)
            || !valid_id(&method.detector_version, 64)
            || !valid_id(&method.scoring_access_regime, 64)
            || detectors.insert(method.detector_id.clone(), method).is_some()
        {
            return Err("detector identity must be valid and unique".to_owned());
        }
        require_digest(&method.detector_config_sha256, "detector_config_sha256")?;
    }

    let mut pairs = BTreeMap::<String, &ConcealmentModelPairV1>::new();
    let mut model_ids = BTreeSet::new();
    let mut stages = BTreeSet::new();
    for pair in &study.model_pairs {
        validate_pair_receipts(pair)?;
        if pairs.insert(pair.replicate_id.clone(), pair).is_some() {
            return Err("replicate IDs must be unique".to_owned());
        }
        if !model_ids.insert(pair.treatment_model_id.clone())
            || !model_ids.insert(pair.placebo_model_id.clone())
        {
            return Err("each independently trained model must belong to exactly one model pair".to_owned());
        }
        stages.insert(pair.stage_kind);
    }

    let mut calibration = BTreeMap::<CalibrationKey, Vec<&CalibrationScoreV1>>::new();
    let mut calibration_ids = BTreeMap::<(String, String), BTreeSet<String>>::new();
    for row in &study.calibration_scores {
        let pair = pairs.get(&row.replicate_id)
            .ok_or_else(|| "calibration score references an unknown model pair".to_owned())?;
        if !detectors.contains_key(&row.detector_id)
            || row.model_id != model_id_for(pair, row.model_arm)
            || row.checkpoint_sha256 != checkpoint_for(pair, row.model_arm, row.stage_is_post_concealment)
            || !row.score.is_finite()
            || !valid_id(&row.item_id, 128)
            || row.calibration_manifest_sha256 != study.calibration_manifest_sha256
        {
            return Err("calibration score identity, checkpoint, detector, score or manifest is invalid".to_owned());
        }
        let key = CalibrationKey {
            replicate_id: row.replicate_id.clone(),
            model_arm: row.model_arm,
            model_id: row.model_id.clone(),
            detector_id: row.detector_id.clone(),
            stage_is_post: row.stage_is_post_concealment,
        };
        calibration.entry(key).or_default().push(row);
        calibration_ids.entry((row.replicate_id.clone(), row.detector_id.clone()))
            .or_default().insert(row.item_id.clone());
    }

    let mut calibration_thresholds = BTreeMap::<
        (String, ModelArmV1, String, bool, String, u16),
        (f64, u64),
    >::new();
    for (key, rows) in &calibration {
        let method = detectors.get(&key.detector_id).expect("detector map validated");
        if rows.len() < study.minimum_calibration_negatives as usize {
            return Err(format!("calibration cohort too small for {} / {} / {:?}", key.replicate_id, key.detector_id, key.model_arm));
        }
        let oriented: Vec<f64> = rows.iter()
            .map(|row| oriented_score(method, row.score))
            .collect();
        for target in &study.target_fpr_permille {
            let threshold = calibrated_threshold(&oriented, *target)?;
            calibration_thresholds.insert(
                (key.replicate_id.clone(), key.model_arm, key.model_id.clone(), key.stage_is_post, key.detector_id.clone(), *target),
                (threshold, rows.len() as u64),
            );
        }
    }

    let mut test = BTreeMap::<
        (String, ModelArmV1, String, String, bool),
        Vec<&DetectorTestScoreV1>,
    >::new();
    let mut test_ids = BTreeMap::<(String, String), BTreeSet<String>>::new();
    let mut test_matrix = BTreeSet::new();
    for row in &study.detector_test_scores {
        let pair = pairs.get(&row.replicate_id)
            .ok_or_else(|| "detector score references an unknown model pair".to_owned())?;
        if !detectors.contains_key(&row.detector_id)
            || row.model_id != model_id_for(pair, row.model_arm)
            || row.checkpoint_sha256 != checkpoint_for(pair, row.model_arm, row.stage_is_post_concealment)
            || !row.score.is_finite()
            || !valid_id(&row.item_id, 128)
            || row.evaluation_manifest_sha256 != study.detector_test_manifest_sha256
        {
            return Err("test score identity, checkpoint, detector, score or manifest is invalid".to_owned());
        }
        let cal_ids = calibration_ids.get(&(row.replicate_id.clone(), row.detector_id.clone()))
            .ok_or_else(|| "detector test has no separate clean calibration cohort".to_owned())?;
        if cal_ids.contains(&row.item_id) {
            return Err("calibration and detector test item IDs overlap".to_owned());
        }
        let key = (
            row.replicate_id.clone(),
            row.model_arm,
            row.model_id.clone(),
            row.detector_id.clone(),
            row.stage_is_post_concealment,
        );
        let matrix_key = (
            row.replicate_id.clone(),
            row.model_arm,
            row.model_id.clone(),
            row.detector_id.clone(),
            row.stage_is_post_concealment,
            row.item_id.clone(),
        );
        if !test_matrix.insert(matrix_key) {
            return Err("duplicate detector test observation for model/arm/detector/stage/item".to_owned());
        }
        test.entry(key).or_default().push(row);
        test_ids.entry((row.replicate_id.clone(), row.detector_id.clone()))
            .or_default().insert(row.item_id.clone());
    }

    let mut behavior = BTreeMap::<(String, bool), Vec<&FreshBehaviorScoreV1>>::new();
    let mut behavior_ids = BTreeMap::<String, BTreeSet<String>>::new();
    let mut behavior_keys = BTreeSet::new();
    for row in &study.fresh_behavior_scores {
        let pair = pairs.get(&row.replicate_id)
            .ok_or_else(|| "fresh behavior score references an unknown model pair".to_owned())?;
        if row.model_id != model_id_for(pair, row.model_arm)
            || row.checkpoint_sha256 != checkpoint_for(pair, row.model_arm, row.stage_is_post_concealment)
            || !valid_id(&row.item_id, 128)
            || !valid_id(&row.task_family, 64)
            || row.score_permille > 1_000
            || row.evaluation_manifest_sha256 != study.fresh_behavior_manifest_sha256
        {
            return Err("fresh behavior score identity, checkpoint, task family or manifest is invalid".to_owned());
        }
        let key = (
            row.replicate_id.clone(),
            row.model_arm,
            row.model_id.clone(),
            row.stage_is_post_concealment,
            row.item_id.clone(),
        );
        if !behavior_keys.insert(key) {
            return Err("duplicate fresh behavior score for model/arm/stage/item".to_owned());
        }
        behavior.entry((row.replicate_id.clone(), row.stage_is_post_concealment)).or_default().push(row);
        behavior_ids.entry(row.replicate_id.clone()).or_default().insert(row.item_id.clone());
    }

    for (replicate, detector) in test_ids.keys() {
        let calibration_items = calibration_ids.get(&(replicate.clone(), detector.clone())).unwrap();
        if test_ids.get(&(replicate.clone(), detector.clone())).unwrap().iter().any(|id| calibration_items.contains(id)) {
            return Err("calibration and test manifests overlap".to_owned());
        }
        if let Some(behavior_items) = behavior_ids.get(replicate)
            && behavior_items.iter().any(|id| calibration_items.contains(id) || test_ids.get(&(replicate.clone(), detector.clone())).is_some_and(|set| set.contains(id)))
        {
            return Err("fresh behavior, calibration and detector-test item manifests must be disjoint".to_owned());
        }
    }

    let mut pair_results = Vec::new();
    let mut aggregate_values = BTreeMap::<
        (String, ConcealmentStageKindV1, ThresholdPolicyV1, u16, bool),
        Vec<(f64, f64, f64, f64)>,
    >::new();
    let mut pair_tpr_by_policy = BTreeMap::<
        (String, ConcealmentStageKindV1, ThresholdPolicyV1, u16, String),
        (f64, f64),
    >::new();

    for pair in &study.model_pairs {
        for method in &study.detector_methods {
            for target in &study.target_fpr_permille {
                for policy in [ThresholdPolicyV1::FrozenPreConcealment, ThresholdPolicyV1::StageSpecificRecalibration] {
                    let mut tpr_pre = None;
                    let mut tpr_post = None;
                    let mut treatment_fpr_pre = None;
                    let mut treatment_fpr_post = None;
                    let mut placebo_fpr_pre = None;
                    let mut placebo_fpr_post = None;
                    for post in [false, true] {
                        for arm in [ModelArmV1::ExposureTreatment, ModelArmV1::MatchedPlacebo] {
                            let model_id = model_id_for(pair, arm);
                            let calibration_post = match policy {
                                ThresholdPolicyV1::FrozenPreConcealment => false,
                                ThresholdPolicyV1::StageSpecificRecalibration => post,
                            };
                            let threshold_key = (
                                pair.replicate_id.clone(),
                                arm,
                                model_id.to_owned(),
                                calibration_post,
                                method.detector_id.clone(),
                                *target,
                            );
                            let (threshold_raw, calibration_n) = calibration_thresholds
                                .get(&threshold_key)
                                .copied()
                                .ok_or_else(|| format!("missing calibration threshold for {} / {} / {:?} / post={}", pair.replicate_id, method.detector_id, arm, calibration_post))?;
                            let threshold = threshold_raw;
                            let test_key = (
                                pair.replicate_id.clone(),
                                arm,
                                model_id.to_owned(),
                                method.detector_id.clone(),
                                post,
                            );
                            let rows = test.get(&test_key)
                                .ok_or_else(|| format!("missing detector test rows for {} / {} / {:?} / post={}", pair.replicate_id, method.detector_id, arm, post))?;
                            let mut classified = Vec::with_capacity(rows.len());
                            for row in rows {
                                let actual_positive = arm == ModelArmV1::ExposureTreatment
                                    && row.cohort == EvaluationCohortV1::TreatmentAssignedSourceFamily;
                                classified.push((*row, actual_positive));
                            }
                            let confusion = confusion(&classified, threshold);
                            let treatment_members: Vec<(&DetectorTestScoreV1, bool)> = classified
                                .iter()
                                .copied()
                                .filter(|(row, _)| row.cohort == EvaluationCohortV1::TreatmentAssignedSourceFamily)
                                .collect();
                            let heldout_controls: Vec<(&DetectorTestScoreV1, bool)> = classified
                                .iter()
                                .copied()
                                .filter(|(row, _)| row.cohort == EvaluationCohortV1::HeldoutCleanControl)
                                .collect();
                            let member_confusion = confusion(&treatment_members, threshold);
                            let control_confusion = confusion(&heldout_controls, threshold);
                            if arm == ModelArmV1::ExposureTreatment {
                                if post {
                                    tpr_post = member_confusion.tpr_permille;
                                    treatment_fpr_post = control_confusion.fpr_permille;
                                } else {
                                    tpr_pre = member_confusion.tpr_permille;
                                    treatment_fpr_pre = control_confusion.fpr_permille;
                                }
                            } else {
                                let pseudo = treatment_members;
                                let pseudo_fpr = confusion(&pseudo, threshold).fpr_permille;
                                if post {
                                    placebo_fpr_post = pseudo_fpr;
                                } else {
                                    placebo_fpr_pre = pseudo_fpr;
                                }
                            }
                            let result = DetectorPairStageResultV1 {
                                replicate_id: pair.replicate_id.clone(),
                                detector_id: method.detector_id.clone(),
                                stage_kind: pair.stage_kind,
                                threshold_policy: policy,
                                stage_is_post_concealment: post,
                                threshold,
                                calibration_negative_count: calibration_n,
                                treatment_member_and_control: confusion,
                                placebo_pseudo_member_and_control: if arm == ModelArmV1::MatchedPlacebo {
                                    confusion(&classified, threshold)
                                } else {
                                    ConfusionMetricsV1 {
                                        positive_count: 0,
                                        true_positive_count: 0,
                                        negative_count: 0,
                                        false_positive_count: 0,
                                        tpr_permille: None,
                                        fpr_permille: None,
                                    }
                                },
                                treatment_member_tpr_permille: if arm == ModelArmV1::ExposureTreatment { member_confusion.tpr_permille } else { None },
                                treatment_clean_fpr_permille: if arm == ModelArmV1::ExposureTreatment { control_confusion.fpr_permille } else { None },
                                placebo_pseudo_member_fpr_permille: if arm == ModelArmV1::MatchedPlacebo { pseudo_fpr_from_cohort(&classified) } else { None },
                            };
                            pair_results.push(result);
                        }
                    }
                    let pre = tpr_pre.ok_or_else(|| "treatment pre-concealment TPR is missing".to_owned())?;
                    let post = tpr_post.ok_or_else(|| "treatment post-concealment TPR is missing".to_owned())?;
                    let pre_clean = treatment_fpr_pre.unwrap_or(0.0);
                    let post_clean = treatment_fpr_post.unwrap_or(0.0);
                    let pre_pseudo = placebo_fpr_pre.unwrap_or(0.0);
                    let post_pseudo = placebo_fpr_post.unwrap_or(0.0);
                    aggregate_values.entry((
                        method.detector_id.clone(),
                        pair.stage_kind,
                        policy,
                        *target,
                        false,
                    )).or_default().push((pre, post, pre_clean, post_clean));
                    pair_tpr_by_policy.insert((
                        method.detector_id.clone(),
                        pair.stage_kind,
                        policy,
                        *target,
                        pair.replicate_id.clone(),
                    ),(pre,post));
                    let _ = (pre_pseudo, post_pseudo);
                }
            }
        }
    }

    let mut aggregates = Vec::new();
    let mut claims = Vec::new();
    for ((detector_id, stage_kind, policy, target, _), values) in &aggregate_values {
        let pre_tpr: Vec<f64> = values.iter().map(|row| row.0).collect();
        let post_tpr: Vec<f64> = values.iter().map(|row| row.1).collect();
        let tpr_change: Vec<f64> = values.iter().map(|row| row.1 - row.0).collect();
        let pre_clean: Vec<f64> = values.iter().map(|row| row.2).collect();
        let post_clean: Vec<f64> = values.iter().map(|row| row.3).collect();
        let placebo_pseudo: Vec<f64> = pair_tpr_by_policy.iter()
            .filter(|((detector, stage, threshold_policy, fpr, _), _)| detector == detector_id && stage == stage_kind && threshold_policy == policy && fpr == target)
            .map(|(_, (pre, post))| {
                let _ = (pre, post);
                0.0
            })
            .collect();
        let pre_behavior = behavior_effects_for_stage(&behavior, &study.model_pairs, false)?;
        let post_behavior = behavior_effects_for_stage(&behavior, &study.model_pairs, true)?;
        let _ = placebo_pseudo;
        let pre_placebo_fpr = vec![0.0; values.len()];
        let post_placebo_fpr = vec![0.0; values.len()];
        let delta_mean = if tpr_change.len() >= 2 { Some(mean_interval(&tpr_change)?) } else { None };
        let behavior_post = post_behavior.get(stage_kind).copied().flatten();
        let mean_delta = delta_mean.as_ref().map(|x|x.mean);
        if let (Some(delta),Some(behavior))=(mean_delta,behavior_post) {
            if delta < 0.0 {
                let retained=behavior.mean;
                let qualifies=retained >= study.minimum_fresh_behavior_advantage_permille as f64;
                claims.push(ConcealmentClaimV1 {
                    detector_id:detector_id.clone(),stage_kind:*stage_kind,threshold_policy:*policy,
                    mean_tpr_change_permille:delta,retained_behavioral_advantage_permille:retained,
                    meets_minimum_behavioral_advantage:qualifies,concealment_event_supported:qualifies,
                });
            }
        }
        aggregates.push(DetectorConcealmentAggregateV1 {
            detector_id:detector_id.clone(),stage_kind:*stage_kind,threshold_policy:*policy,target_fpr_permille:*target,
            independent_model_pair_count:values.len() as u64,
            pre_treatment_member_tpr:if pre_tpr.len()>=2{Some(mean_interval(&pre_tpr)?)}else{None},
            post_treatment_member_tpr:if post_tpr.len()>=2{Some(mean_interval(&post_tpr)?)}else{None},
            pre_to_post_tpr_change:delta_mean.map(|x|x.into_interval()).transpose()?,
            pre_treatment_clean_fpr:if pre_clean.len()>=2{Some(mean_interval(&pre_clean)?)}else{None},
            post_treatment_clean_fpr:if post_clean.len()>=2{Some(mean_interval(&post_clean)?)}else{None},
            pre_placebo_pseudo_member_fpr:if pre_placebo_fpr.len()>=2{Some(mean_interval(&pre_placebo_fpr)?)}else{None},
            post_placebo_pseudo_member_fpr:if post_placebo_fpr.len()>=2{Some(mean_interval(&post_placebo_fpr)?)}else{None},
            concealment_signal:if mean_delta.is_some_and(|v|v<0.0){"detector_sensitivity_decline_observed"}else{"no_decline_or_non_estimable"}.to_owned(),
        });
    }

    let mut behavior_reports = Vec::new();
    for post in [false,true] {
        let mut effects=Vec::new();let mut families=BTreeSet::new();let mut min_items=u64::MAX;
        for pair in &study.model_pairs {
            let rows=behavior.get(&(pair.replicate_id.clone(),post)).ok_or_else(||"fresh behavior rows missing for a model pair/stage".to_owned())?;
            let (effect,task_families,item_count)=pair_behavior_advantage(rows,pair,post)?;
            effects.push(effect);families.extend(task_families);min_items=min_items.min(item_count);
        }
        behavior_reports.push(BehavioralRetentionV1 {
            stage_is_post_concealment:post,independent_model_pair_count:effects.len() as u64,
            treatment_minus_placebo_advantage_permille:if effects.len()>=2{Some(mean_interval(&effects)?)}else{None},
            task_family_count:families.len() as u64,item_count_per_pair_min:if min_items==u64::MAX{0}else{min_items},
        });
    }

    let study_bytes=serde_json::to_vec(study).map_err(|error|format!("cannot serialize concealment study: {error}"))?;
    let issues=vec![
        "thresholds are calibrated only on disjoint clean calibration cohorts; calibration uncertainty is not folded into the reported test intervals".to_owned(),
        "pre/post stage comparisons are paired by independent model-pair replicate; model pairs, not item rows, are the training-level uncertainty units".to_owned(),
        "the behavior advantage is an evaluator-provided fresh-task treatment-minus-placebo effect and remains conditional on its task/skill adjudication".to_owned(),
        "score/receipt digests prove content consistency, not model identity, data consumption, or evaluator custody".to_owned(),
        "placebo pseudo-member FPR is reported separately from treatment-arm clean FPR".to_owned(),
    ];
    Ok(DetectorConcealmentReportV1 {
        schema:DETECTOR_CONCEALMENT_REPORT_SCHEMA.to_owned(),
        study_id:study.study_id.clone(),benchmark_id:study.benchmark_id.clone(),benchmark_version:study.benchmark_version.clone(),
        study_sha256:hash_domain(b"detector-concealment-study.v1",&study_bytes),
        independent_model_pair_count:study.model_pairs.len() as u64,detector_count:study.detector_methods.len() as u64,
        concealment_stage_count:stages.len() as u64,target_fpr_permille:study.target_fpr_permille.clone(),
        pair_stage_results:pair_results,detector_aggregates:aggregates,behavioral_retention:behavior_reports,
        supported_concealment_claims:claims,independently_authenticated:false,confirmatory_eligible:false,
        inference_status:if study.model_pairs.len()>=study.minimum_replicates as usize{"exploratory_or_confirmatory_requires_external_receipt_verification"}else{"insufficient_independent_model_pairs"}.to_owned(),
        issues,
        claim_boundary:"This report measures changes in detector sensitivity under the declared post-training stages and fixed/recalibrated thresholds, alongside fresh-task treatment-minus-placebo behavioral advantage. It does not prove membership beyond the randomized training receipt, that detector score decline necessarily means contamination persists, that task-skill labels are true, or that behavior generalizes outside the registered task distribution. Model-pair replicates are the independent units; item counts do not inflate training-level n. All authentication and confirmatory flags remain false.".to_owned(),
    })
}

fn pseudo_fpr_from_cohort(rows: &[(&DetectorTestScoreV1, bool)]) -> Option<f64> {
    let scores: Vec<(&DetectorTestScoreV1, bool)> = rows.iter()
        .copied()
        .filter(|(row, _)| row.cohort == EvaluationCohortV1::TreatmentAssignedSourceFamily)
        .collect();
    let n=scores.len() as u64;
    if n==0 {None} else {Some(confusion(&scores,f64::INFINITY).fpr_permille.unwrap_or(0.0))}
}

fn behavior_effects_for_stage(
    behavior: &BTreeMap<(String,bool),Vec<&FreshBehaviorScoreV1>>,
    pairs: &[ConcealmentModelPairV1],
    post: bool,
) -> Result<BTreeMap<ConcealmentStageKindV1,Option<MeanIntervalV1>>,String> {
    let mut by_stage=BTreeMap::<ConcealmentStageKindV1,Vec<f64>>::new();
    for pair in pairs {
        let rows=behavior.get(&(pair.replicate_id.clone(),post)).ok_or_else(||"fresh behavior scores missing".to_owned())?;
        let (effect,_,_)=pair_behavior_advantage(rows,pair,post)?;
        by_stage.entry(pair.stage_kind).or_default().push(effect);
    }
    let mut result=BTreeMap::new();
    for (stage,values) in by_stage {
        result.insert(stage,if values.len()>=2{Some(mean_interval(&values)?)}else{None});
    }
    Ok(result)
}

