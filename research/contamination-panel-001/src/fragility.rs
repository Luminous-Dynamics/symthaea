//! Measure whether detector evidence survives benchmark-contamination concealment.
//!
//! The benchmark pairs contaminated checkpoints with matched clean controls and
//! scores the same frozen candidate universe at every stage. Independent model
//! pairs are the uncertainty units. Calibration negatives select detector
//! thresholds; held-out test data evaluate those thresholds.
//!
//! A positive concealment gap means task-performance inflation survives more
//! strongly than detector separability. This is a failure-mode measurement,
//! not a claim about intent or universal detector evasion.

use crate::{
    causal_analysis::{MeanIntervalV1, mean_interval},
    hash_domain,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FRAGILITY_STUDY_SCHEMA: &str = "symthaea.detector_fragility_study.v1";
pub const FRAGILITY_REPORT_SCHEMA: &str = "symthaea.detector_fragility_report.v1";
pub const MIN_CALIBRATION_NEGATIVES: u64 = 300;
pub const MIN_TEST_NEGATIVES: u64 = 300;
pub const MIN_TEST_MEMBERS: u64 = 100;
pub const MAX_MODEL_PAIR_REPLICATES: usize = 1_000;
pub const MAX_ITEMS_PER_CHECKPOINT: usize = 100_000;
pub const MAX_DETECTORS: usize = 128;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStageV1 {
    SftClean,
    SftContaminated,
    SftCleanThenGrpoClean,
    SftContaminatedThenGrpoClean,
    SftCleanThenRaft,
    SftContaminatedThenRaft,
    SftCleanThenRaftPlusPlus,
    SftContaminatedThenRaftPlusPlus,
    AdvancedLrmClean,
    AdvancedLrmCotContaminated,
}

impl TrainingStageV1 {
    pub const ALL: [Self; 10] = [
        Self::SftClean,
        Self::SftContaminated,
        Self::SftCleanThenGrpoClean,
        Self::SftContaminatedThenGrpoClean,
        Self::SftCleanThenRaft,
        Self::SftContaminatedThenRaft,
        Self::SftCleanThenRaftPlusPlus,
        Self::SftContaminatedThenRaftPlusPlus,
        Self::AdvancedLrmClean,
        Self::AdvancedLrmCotContaminated,
    ];
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSplitV1 {
    Calibration,
    Test,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DetectorDirectionV1 {
    HigherIsMember,
    LowerIsMember,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum InterventionComparisonV1 {
    SftThenCleanGrpo,
    SftThenRaft,
    SftThenRaftPlusPlus,
    AdvancedLrmCotSft,
}

impl InterventionComparisonV1 {
    pub const ALL: [Self; 4] = [
        Self::SftThenCleanGrpo,
        Self::SftThenRaft,
        Self::SftThenRaftPlusPlus,
        Self::AdvancedLrmCotSft,
    ];

    fn stages(
        self,
    ) -> (
        Option<TrainingStageV1>,
        Option<TrainingStageV1>,
        TrainingStageV1,
        TrainingStageV1,
    ) {
        match self {
            Self::SftThenCleanGrpo => (
                Some(TrainingStageV1::SftContaminated),
                Some(TrainingStageV1::SftClean),
                TrainingStageV1::SftContaminatedThenGrpoClean,
                TrainingStageV1::SftCleanThenGrpoClean,
            ),
            Self::SftThenRaft => (
                Some(TrainingStageV1::SftContaminated),
                Some(TrainingStageV1::SftClean),
                TrainingStageV1::SftContaminatedThenRaft,
                TrainingStageV1::SftCleanThenRaft,
            ),
            Self::SftThenRaftPlusPlus => (
                Some(TrainingStageV1::SftContaminated),
                Some(TrainingStageV1::SftClean),
                TrainingStageV1::SftContaminatedThenRaftPlusPlus,
                TrainingStageV1::SftCleanThenRaftPlusPlus,
            ),
            Self::AdvancedLrmCotSft => (
                None,
                None,
                TrainingStageV1::AdvancedLrmCotContaminated,
                TrainingStageV1::AdvancedLrmClean,
            ),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorScoreV1 {
    pub detector_id: String,
    pub detector_spec_sha256: String,
    pub direction: DetectorDirectionV1,
    pub raw_score: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FragilityItemObservationV1 {
    pub item_id: String,
    pub split: EvaluationSplitV1,
    /// Randomized member assignment in the contaminated intervention arm.
    /// The same labels are carried into matched clean-control checkpoints.
    pub intended_member: bool,
    pub correct: bool,
    pub detector_scores: Vec<DetectorScoreV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FragilityCheckpointV1 {
    pub stage: TrainingStageV1,
    pub checkpoint_sha256: String,
    pub parent_checkpoint_sha256: String,
    pub training_receipt_sha256: String,
    pub training_config_sha256: String,
    pub objective_config_sha256: String,
    pub training_data_manifest_sha256: String,
    pub optimizer_steps: u64,
    pub observations: Vec<FragilityItemObservationV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FragilityModelPairV1 {
    pub replicate_id: String,
    pub base_checkpoint_sha256: String,
    pub advanced_lrm_base_checkpoint_sha256: String,
    pub randomization_manifest_sha256: String,
    pub checkpoints: Vec<FragilityCheckpointV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorFragilityStudyV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    /// Permille: 10 = 1% FPR, 50 = 5% FPR.
    pub target_fpr_permille: u16,
    pub minimum_calibration_negatives: u64,
    pub minimum_test_negatives: u64,
    pub minimum_test_members: u64,
    pub model_pair_replicates: Vec<FragilityModelPairV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WilsonIntervalV1 {
    pub lower_95: f64,
    pub upper_95: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DetectorCheckpointMetricsV1 {
    pub detector_id: String,
    pub detector_spec_sha256: String,
    pub direction: DetectorDirectionV1,
    pub auroc: f64,
    pub auprc: f64,
    pub calibration_threshold_oriented: f64,
    pub calibration_negative_count: u64,
    pub calibration_false_positive_count: u64,
    pub test_positive_count: u64,
    pub test_negative_count: u64,
    pub test_true_positive_count: u64,
    pub test_false_positive_count: u64,
    pub test_tpr_permille: f64,
    pub test_fpr_permille: f64,
    pub test_tpr_wilson_95: WilsonIntervalV1,
    pub test_fpr_wilson_95: WilsonIntervalV1,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointPerformanceV1 {
    pub member_accuracy_permille: f64,
    pub nonmember_accuracy_permille: f64,
    pub member_test_count: u64,
    pub nonmember_test_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FragilityCheckpointReportV1 {
    pub stage: TrainingStageV1,
    pub checkpoint_sha256: String,
    pub parent_checkpoint_sha256: String,
    pub training_receipt_sha256: String,
    pub training_config_sha256: String,
    pub objective_config_sha256: String,
    pub training_data_manifest_sha256: String,
    pub optimizer_steps: u64,
    pub performance: CheckpointPerformanceV1,
    pub detectors: Vec<DetectorCheckpointMetricsV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DetectorRetentionV1 {
    pub detector_id: String,
    pub auroc_before: Option<f64>,
    pub auroc_after: f64,
    pub auroc_delta: Option<f64>,
    /// (AUC_after - 0.5) / (AUC_before - 0.5), in permille.
    /// Undefined when the pre-intervention AUROC is <= 0.5.
    pub signal_retention_permille: Option<f64>,
    pub tpr_before_permille: Option<f64>,
    pub tpr_after_permille: f64,
    pub fpr_before_permille: Option<f64>,
    pub fpr_after_permille: f64,
    pub concealment_gap_permille: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FragilityInterventionComparisonV1 {
    pub intervention: InterventionComparisonV1,
    pub before_contaminated_stage: Option<TrainingStageV1>,
    pub before_clean_control_stage: Option<TrainingStageV1>,
    pub after_contaminated_stage: TrainingStageV1,
    pub after_clean_control_stage: TrainingStageV1,
    pub member_performance_inflation_before_permille: Option<f64>,
    pub member_performance_inflation_after_permille: f64,
    pub member_inflation_retention_permille: Option<f64>,
    pub nonmember_generalization_delta_before_permille: Option<f64>,
    pub nonmember_generalization_delta_after_permille: f64,
    pub detectors: Vec<DetectorRetentionV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FragilityPairReportV1 {
    pub replicate_id: String,
    pub checkpoint_reports: Vec<FragilityCheckpointReportV1>,
    pub intervention_comparisons: Vec<FragilityInterventionComparisonV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FragilityAggregateDetectorV1 {
    pub detector_id: String,
    pub signal_retention_permille: Option<MeanIntervalV1>,
    pub concealment_gap_permille: Option<MeanIntervalV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FragilityAggregateInterventionV1 {
    pub intervention: InterventionComparisonV1,
    pub independent_model_pair_count: u64,
    pub member_performance_inflation_before_permille: Option<MeanIntervalV1>,
    pub member_performance_inflation_after_permille: MeanIntervalV1,
    pub member_inflation_retention_permille: Option<MeanIntervalV1>,
    pub nonmember_generalization_delta_after_permille: MeanIntervalV1,
    pub detectors: Vec<FragilityAggregateDetectorV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorFragilityReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub study_sha256: String,
    pub target_fpr_permille: u16,
    pub independent_model_pair_replicate_count: u64,
    pub pair_reports: Vec<FragilityPairReportV1>,
    pub aggregate_interventions: Vec<FragilityAggregateInterventionV1>,
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

fn valid_id(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

fn oriented_score(raw_score: f64, direction: DetectorDirectionV1) -> f64 {
    match direction {
        DetectorDirectionV1::HigherIsMember => raw_score,
        DetectorDirectionV1::LowerIsMember => -raw_score,
    }
}

fn wilson_interval(successes: u64, total: u64) -> Result<WilsonIntervalV1, String> {
    if total == 0 || successes > total {
        return Err("Wilson interval requires 0 <= successes <= positive total".to_owned());
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    Ok(WilsonIntervalV1 {
        lower_95: (center - half).max(0.0),
        upper_95: (center + half).min(1.0),
    })
}

fn validate_score_vector(
    observations: &[FragilityItemObservationV1],
) -> Result<BTreeMap<String, (DetectorDirectionV1, String)>, String> {
    if observations.is_empty() || observations.len() > MAX_ITEMS_PER_CHECKPOINT {
        return Err("checkpoint observation count is empty or exceeds the configured maximum".to_owned());
    }
    let mut ids = BTreeSet::new();
    let first = observations
        .first()
        .ok_or_else(|| "checkpoint observations are empty".to_owned())?;
    if first.detector_scores.is_empty() || first.detector_scores.len() > MAX_DETECTORS {
        return Err("detector count must be in 1..=128".to_owned());
    }
    let mut directions = BTreeMap::new();
    for score in &first.detector_scores {
        if !valid_id(&score.detector_id, 128)
            || !valid_digest(&score.detector_spec_sha256)
            || !score.raw_score.is_finite()
        {
            return Err("detector IDs must be valid and scores finite".to_owned());
        }
        if directions
            .insert(
                score.detector_id.clone(),
                (score.direction, score.detector_spec_sha256.clone()),
            )
            .is_some()
        {
            return Err("detector IDs must be unique within an item".to_owned());
        }
    }
    for observation in observations {
        if !valid_id(&observation.item_id, 128) || !ids.insert(observation.item_id.as_str()) {
            return Err("item IDs must be valid and unique within a checkpoint".to_owned());
        }
        if observation.detector_scores.len() != directions.len() {
            return Err("every item must have exactly one score for every detector".to_owned());
        }
        let mut observed_ids = BTreeSet::new();
        for score in &observation.detector_scores {
            if !score.raw_score.is_finite() {
                return Err("detector scores must be finite".to_owned());
            }
            if !observed_ids.insert(score.detector_id.as_str()) {
                return Err("detector scores contain duplicate IDs".to_owned());
            }
            let contract = directions
                .get(&score.detector_id)
                .ok_or_else(|| "detector ID is not in the checkpoint contract".to_owned())?;
            if contract.0 != score.direction || contract.1 != score.detector_spec_sha256 {
                return Err("detector IDs, directions and specification digests must be identical across all items".to_owned());
            }
        }
    }
    Ok(directions)
}

fn detector_score_for<'a>(
    observation: &'a FragilityItemObservationV1,
    detector_id: &str,
) -> Result<&'a DetectorScoreV1, String> {
    observation
        .detector_scores
        .iter()
        .find(|score| score.detector_id == detector_id)
        .ok_or_else(|| format!("missing detector score for {detector_id}"))
}

fn auc_and_auprc(
    positive_scores: &[(f64, bool)],
) -> Result<(f64, f64), String> {
    let positives = positive_scores.iter().filter(|(_, positive)| *positive).count();
    let negatives = positive_scores.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err("AUROC/AUPRC require both positive and negative test items".to_owned());
    }
    let mut ascending = positive_scores.to_vec();
    ascending.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut positive_rank_sum = 0.0;
    let mut position = 0_usize;
    while position < ascending.len() {
        let mut end = position + 1;
        while end < ascending.len() && ascending[end].0 == ascending[position].0 {
            end += 1;
        }
        let average_rank = ((position + 1 + end) as f64) / 2.0;
        let tied_positives = ascending[position..end]
            .iter()
            .filter(|(_, positive)| *positive)
            .count();
        positive_rank_sum += average_rank * tied_positives as f64;
        position = end;
    }
    let p = positives as f64;
    let n = negatives as f64;
    let auroc = (positive_rank_sum - p * (p + 1.0) / 2.0) / (p * n);

    let mut descending = positive_scores.to_vec();
    descending.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut true_positives = 0_u64;
    let mut false_positives = 0_u64;
    let mut previous_recall = 0.0;
    let mut auprc = 0.0;
    position = 0;
    while position < descending.len() {
        let mut end = position + 1;
        while end < descending.len() && descending[end].0 == descending[position].0 {
            end += 1;
        }
        for (_, positive) in &descending[position..end] {
            if *positive {
                true_positives += 1;
            } else {
                false_positives += 1;
            }
        }
        let recall = true_positives as f64 / p;
        let precision =
            true_positives as f64 / (true_positives + false_positives) as f64;
        auprc += (recall - previous_recall) * precision;
        previous_recall = recall;
        position = end;
    }
    if !auroc.is_finite()
        || !auprc.is_finite()
        || !(0.0..=1.0).contains(&auroc)
        || !(0.0..=1.0).contains(&auprc)
    {
        return Err("AUROC/AUPRC calculation produced an invalid value".to_owned());
    }
    Ok((auroc, auprc))
}

fn checkpoint_detector_metrics(
    checkpoint: &FragilityCheckpointV1,
    directions: &BTreeMap<String, (DetectorDirectionV1, String)>,
    target_fpr_permille: u16,
    minimum_calibration_negatives: u64,
    minimum_test_negatives: u64,
    minimum_test_members: u64,
) -> Result<Vec<DetectorCheckpointMetricsV1>, String> {
    let calibration_negatives: Vec<&FragilityItemObservationV1> = checkpoint
        .observations
        .iter()
        .filter(|item| item.split == EvaluationSplitV1::Calibration && !item.intended_member)
        .collect();
    let test_items: Vec<&FragilityItemObservationV1> = checkpoint
        .observations
        .iter()
        .filter(|item| item.split == EvaluationSplitV1::Test)
        .collect();
    let test_positive_count = test_items.iter().filter(|item| item.intended_member).count() as u64;
    let test_negative_count = test_items.iter().filter(|item| !item.intended_member).count() as u64;
    if calibration_negatives.len() as u64 < minimum_calibration_negatives {
        return Err(format!(
            "{:?} has {} calibration non-members; minimum is {minimum_calibration_negatives}",
            checkpoint.stage,
            calibration_negatives.len()
        ));
    }
    if test_negative_count < minimum_test_negatives || test_positive_count < minimum_test_members {
        return Err(format!(
            "{:?} test split has {test_positive_count} members / {test_negative_count} non-members; minimums are {minimum_test_members}/{minimum_test_negatives}",
            checkpoint.stage
        ));
    }

    let mut output = Vec::with_capacity(directions.len());
    for (detector_id, (direction, detector_spec_sha256)) in directions {
        let mut calibration_scores: Vec<f64> = calibration_negatives
            .iter()
            .map(|item| {
                detector_score_for(item, detector_id)
                    .map(|score| oriented_score(score.raw_score, *direction))
            })
            .collect::<Result<_, _>>()?;
        calibration_scores.sort_by(|left, right| right.total_cmp(left));
        let allowed_false_positives =
            (calibration_scores.len() as u64 * u64::from(target_fpr_permille)) / 1_000;
        if allowed_false_positives >= calibration_scores.len() as u64 {
            return Err("target FPR is too high for the calibration sample size".to_owned());
        }
        let threshold = calibration_scores[allowed_false_positives as usize];

        let calibration_false_positive_count = calibration_scores
            .iter()
            .filter(|score| **score > threshold)
            .count() as u64;
        let mut test_scores = Vec::with_capacity(test_items.len());
        let mut test_true_positives = 0_u64;
        let mut test_false_positives = 0_u64;
        for item in &test_items {
            let score = oriented_score(
                detector_score_for(item, detector_id)?.raw_score,
                *direction,
            );
            test_scores.push((score, item.intended_member));
            if score > threshold {
                if item.intended_member {
                    test_true_positives += 1;
                } else {
                    test_false_positives += 1;
                }
            }
        }
        let (auroc, auprc) = auc_and_auprc(&test_scores)?;
        output.push(DetectorCheckpointMetricsV1 {
            detector_id: detector_id.clone(),
            detector_spec_sha256: detector_spec_sha256.clone(),
            direction: *direction,
            auroc,
            auprc,
            calibration_threshold_oriented: threshold,
            calibration_negative_count: calibration_scores.len() as u64,
            calibration_false_positive_count,
            test_positive_count,
            test_negative_count,
            test_true_positive_count: test_true_positives,
            test_false_positive_count: test_false_positives,
            test_tpr_permille: test_true_positives as f64 * 1_000.0 / test_positive_count as f64,
            test_fpr_permille: test_false_positives as f64 * 1_000.0 / test_negative_count as f64,
            test_tpr_wilson_95: wilson_interval(test_true_positives, test_positive_count)?,
            test_fpr_wilson_95: wilson_interval(test_false_positives, test_negative_count)?,
        });
    }
    Ok(output)
}

fn performance_for_stage(
    checkpoint: &FragilityCheckpointV1,
) -> Result<CheckpointPerformanceV1, String> {
    let test_items: Vec<&FragilityItemObservationV1> = checkpoint
        .observations
        .iter()
        .filter(|item| item.split == EvaluationSplitV1::Test)
        .collect();
    let members: Vec<&FragilityItemObservationV1> =
        test_items.iter().copied().filter(|item| item.intended_member).collect();
    let nonmembers: Vec<&FragilityItemObservationV1> =
        test_items.iter().copied().filter(|item| !item.intended_member).collect();
    if members.is_empty() || nonmembers.is_empty() {
        return Err("performance requires test members and non-members".to_owned());
    }
    let member_accuracy = members.iter().filter(|item| item.correct).count() as f64
        * 1_000.0
        / members.len() as f64;
    let nonmember_accuracy = nonmembers.iter().filter(|item| item.correct).count() as f64
        * 1_000.0
        / nonmembers.len() as f64;
    Ok(CheckpointPerformanceV1 {
        member_accuracy_permille: member_accuracy,
        nonmember_accuracy_permille: nonmember_accuracy,
        member_test_count: members.len() as u64,
        nonmember_test_count: nonmembers.len() as u64,
    })
}

fn retention_ratio(after: f64, before: f64) -> Option<f64> {
    if before > f64::EPSILON {
        Some(after / before * 1_000.0)
    } else {
        None
    }
}

fn detector_metric<'a>(
    checkpoint: &'a FragilityCheckpointReportV1,
    detector_id: &str,
) -> Result<&'a DetectorCheckpointMetricsV1, String> {
    checkpoint
        .detectors
        .iter()
        .find(|metric| metric.detector_id == detector_id)
        .ok_or_else(|| format!("missing detector metrics for {detector_id}"))
}

fn build_intervention_comparison(
    intervention: InterventionComparisonV1,
    reports: &BTreeMap<TrainingStageV1, FragilityCheckpointReportV1>,
) -> Result<FragilityInterventionComparisonV1, String> {
    let (before_contaminated_stage, before_clean_control_stage, after_contaminated_stage, after_clean_control_stage) =
        intervention.stages();
    let after_contaminated = reports
        .get(&after_contaminated_stage)
        .ok_or_else(|| "missing contaminated post-intervention checkpoint report".to_owned())?;
    let after_clean = reports
        .get(&after_clean_control_stage)
        .ok_or_else(|| "missing matched clean-control post-intervention report".to_owned())?;

    let before_performance = if let (Some(before_stage), Some(control_stage)) =
        (before_contaminated_stage, before_clean_control_stage)
    {
        let treatment = reports
            .get(&before_stage)
            .ok_or_else(|| "missing contaminated pre-intervention report".to_owned())?;
        let control = reports
            .get(&control_stage)
            .ok_or_else(|| "missing clean pre-intervention report".to_owned())?;
        Some((
            treatment.performance.member_accuracy_permille
                - control.performance.member_accuracy_permille,
            treatment.performance.nonmember_accuracy_permille
                - control.performance.nonmember_accuracy_permille,
        ))
    } else {
        None
    };
    let after_member_inflation = after_contaminated.performance.member_accuracy_permille
        - after_clean.performance.member_accuracy_permille;
    let after_nonmember_delta = after_contaminated.performance.nonmember_accuracy_permille
        - after_clean.performance.nonmember_accuracy_permille;
    let member_inflation_retention = before_performance
        .and_then(|(before, _)| retention_ratio(after_member_inflation, before));

    let detector_ids: BTreeSet<&str> = after_contaminated
        .detectors
        .iter()
        .map(|metric| metric.detector_id.as_str())
        .collect();
    if detector_ids
        != after_clean
            .detectors
            .iter()
            .map(|metric| metric.detector_id.as_str())
            .collect()
    {
        return Err("matched contaminated and clean-control detector sets differ".to_owned());
    }

    let mut detector_rows = Vec::with_capacity(detector_ids.len());
    for detector_id in detector_ids {
        let after_metric = detector_metric(after_contaminated, detector_id)?;
        let before_metric = if let Some(before_stage) = before_contaminated_stage {
            Some(detector_metric(
                reports
                    .get(&before_stage)
                    .ok_or_else(|| "missing pre-intervention detector report".to_owned())?,
                detector_id,
            )?)
        } else {
            None
        };
        let auc_before = before_metric.map(|metric| metric.auroc);
        let auc_delta = auc_before.map(|before| after_metric.auroc - before);
        let signal_retention = auc_before.and_then(|before| {
            if before > 0.5 + f64::EPSILON {
                Some((after_metric.auroc - 0.5) / (before - 0.5) * 1_000.0)
            } else {
                None
            }
        });
        let tpr_before = before_metric.map(|metric| metric.test_tpr_permille);
        let fpr_before = before_metric.map(|metric| metric.test_fpr_permille);
        let concealment_gap = member_inflation_retention.zip(signal_retention)
            .map(|(inflation, signal)| inflation - signal);
        detector_rows.push(DetectorRetentionV1 {
            detector_id: detector_id.to_owned(),
            auroc_before: auc_before,
            auroc_after: after_metric.auroc,
            auroc_delta,
            signal_retention_permille: signal_retention,
            tpr_before_permille: tpr_before,
            tpr_after_permille: after_metric.test_tpr_permille,
            fpr_before_permille: fpr_before,
            fpr_after_permille: after_metric.test_fpr_permille,
            concealment_gap_permille: concealment_gap,
        });
    }

    Ok(FragilityInterventionComparisonV1 {
        intervention,
        before_contaminated_stage,
        before_clean_control_stage,
        after_contaminated_stage,
        after_clean_control_stage,
        member_performance_inflation_before_permille: before_performance.map(|value| value.0),
        member_performance_inflation_after_permille: after_member_inflation,
        member_inflation_retention_permille: member_inflation_retention,
        nonmember_generalization_delta_before_permille: before_performance.map(|value| value.1),
        nonmember_generalization_delta_after_permille: after_nonmember_delta,
        detectors: detector_rows,
    })
}

fn mean_optional(values: &[Option<f64>]) -> Result<Option<MeanIntervalV1>, String> {
    if values.iter().any(Option::is_none) || values.is_empty() {
        return Ok(None);
    }
    let complete: Vec<f64> = values.iter().filter_map(|value| *value).collect();
    if complete.len() < 2 {
        return Ok(None);
    }
    mean_interval(&complete).map(Some)
}

fn aggregate_interventions(
    pair_reports: &[FragilityPairReportV1],
) -> Result<Vec<FragilityAggregateInterventionV1>, String> {
    let mut output = Vec::new();
    for intervention in InterventionComparisonV1::ALL {
        let comparisons: Vec<&FragilityInterventionComparisonV1> = pair_reports
            .iter()
            .map(|pair| {
                pair.intervention_comparisons
                    .iter()
                    .find(|row| row.intervention == intervention)
                    .ok_or_else(|| "model-pair report is missing an intervention".to_owned())
            })
            .collect::<Result<_, _>>()?;
        let before_values: Vec<Option<f64>> = comparisons
            .iter()
            .map(|row| row.member_performance_inflation_before_permille)
            .collect();
        let after_values: Vec<f64> = comparisons
            .iter()
            .map(|row| row.member_performance_inflation_after_permille)
            .collect();
        let retention_values: Vec<Option<f64>> = comparisons
            .iter()
            .map(|row| row.member_inflation_retention_permille)
            .collect();
        let spillover_values: Vec<f64> = comparisons
            .iter()
            .map(|row| row.nonmember_generalization_delta_after_permille)
            .collect();

        let detector_ids: BTreeSet<String> = comparisons
            .first()
            .ok_or_else(|| "no model-pair comparisons supplied".to_owned())?
            .detectors
            .iter()
            .map(|row| row.detector_id.clone())
            .collect();
        if comparisons.iter().any(|row| {
            row.detectors
                .iter()
                .map(|detector| detector.detector_id.clone())
                .collect::<BTreeSet<_>>()
                != detector_ids
        }) {
            return Err("detector set differs across independent model-pair comparisons".to_owned());
        }
        let mut detector_aggregates = Vec::new();
        for detector_id in detector_ids {
            let rows: Vec<&DetectorRetentionV1> = comparisons
                .iter()
                .map(|comparison| {
                    comparison
                        .detectors
                        .iter()
                        .find(|detector| detector.detector_id == detector_id)
                        .ok_or_else(|| "detector comparison missing across model pairs".to_owned())
                })
                .collect::<Result<_, _>>()?;
            let signal_values: Vec<Option<f64>> = rows
                .iter()
                .map(|row| row.signal_retention_permille)
                .collect();
            let gap_values: Vec<Option<f64>> = rows
                .iter()
                .map(|row| row.concealment_gap_permille)
                .collect();
            detector_aggregates.push(FragilityAggregateDetectorV1 {
                detector_id,
                signal_retention_permille: mean_optional(&signal_values)?,
                concealment_gap_permille: mean_optional(&gap_values)?,
            });
        }
        output.push(FragilityAggregateInterventionV1 {
            intervention,
            independent_model_pair_count: comparisons.len() as u64,
            member_performance_inflation_before_permille: mean_optional(&before_values)?,
            member_performance_inflation_after_permille: mean_interval(&after_values)?,
            member_inflation_retention_permille: mean_optional(&retention_values)?,
            nonmember_generalization_delta_after_permille: mean_interval(&spillover_values)?,
            detectors: detector_aggregates,
        });
    }
    Ok(output)
}

pub fn analyze_detector_fragility(
    study: &DetectorFragilityStudyV1,
) -> Result<DetectorFragilityReportV1, String> {
    if study.schema != FRAGILITY_STUDY_SCHEMA {
        return Err(format!("study schema must be {FRAGILITY_STUDY_SCHEMA}"));
    }
    if !valid_id(&study.study_id, 128)
        || !valid_id(&study.benchmark_id, 128)
        || study.benchmark_version.trim().is_empty()
        || study.benchmark_version.len() > 64
    {
        return Err("study and benchmark identities must be nonempty bounded IDs".to_owned());
    }
    if !(1..=100).contains(&study.target_fpr_permille) {
        return Err("target_fpr_permille must be in [1, 100] (0.1%..10%)".to_owned());
    }
    if study.minimum_calibration_negatives < MIN_CALIBRATION_NEGATIVES
        || study.minimum_test_negatives < MIN_TEST_NEGATIVES
        || study.minimum_test_members < MIN_TEST_MEMBERS
    {
        return Err("minimum sample counts cannot be below the registered low-FPR floors".to_owned());
    }
    if study.model_pair_replicates.len() < 2
        || study.model_pair_replicates.len() > MAX_MODEL_PAIR_REPLICATES
    {
        return Err(format!(
            "independent model-pair replicate count must be in 2..={MAX_MODEL_PAIR_REPLICATES}"
        ));
    }

    let study_bytes = serde_json::to_vec(study)
        .map_err(|error| format!("cannot serialize fragility study: {error}"))?;
    let study_sha256 = hash_domain(b"detector-fragility-study.v1", &study_bytes);

    let mut replicate_ids = BTreeSet::new();
    let mut pair_reports = Vec::with_capacity(study.model_pair_replicates.len());
    let mut reference_item_contract: Option<BTreeMap<String, (EvaluationSplitV1, bool)>> = None;
    let mut reference_detector_contract: Option<
        BTreeMap<String, (DetectorDirectionV1, String)>,
    > = None;

    for pair in &study.model_pair_replicates {
        if !valid_id(&pair.replicate_id, 128) || !replicate_ids.insert(pair.replicate_id.as_str()) {
            return Err("model-pair replicate IDs must be unique valid IDs".to_owned());
        }
        require_digest(&pair.base_checkpoint_sha256, "base_checkpoint_sha256")?;
        require_digest(
            &pair.advanced_lrm_base_checkpoint_sha256,
            "advanced_lrm_base_checkpoint_sha256",
        )?;
        require_digest(
            &pair.randomization_manifest_sha256,
            "randomization_manifest_sha256",
        )?;
        if pair.checkpoints.len() != TrainingStageV1::ALL.len() {
            return Err("every model pair must contain all ten predeclared training stages".to_owned());
        }
        let mut checkpoints = BTreeMap::new();
        let mut pair_detector_contract: Option<
            BTreeMap<String, (DetectorDirectionV1, String)>,
        > = None;
        let mut pair_item_contract: Option<BTreeMap<String, (EvaluationSplitV1, bool)>> = None;
        for checkpoint in &pair.checkpoints {
            if checkpoints.insert(checkpoint.stage, checkpoint).is_some() {
                return Err("training stage occurs more than once in one model pair".to_owned());
            }
            require_digest(&checkpoint.checkpoint_sha256, "checkpoint_sha256")?;
            require_digest(&checkpoint.parent_checkpoint_sha256, "parent_checkpoint_sha256")?;
            require_digest(&checkpoint.training_receipt_sha256, "training_receipt_sha256")?;
            require_digest(&checkpoint.training_config_sha256, "training_config_sha256")?;
            require_digest(&checkpoint.objective_config_sha256, "objective_config_sha256")?;
            require_digest(
                &checkpoint.training_data_manifest_sha256,
                "training_data_manifest_sha256",
            )?;
            let directions = validate_score_vector(&checkpoint.observations)?;
            let item_contract: BTreeMap<String, (EvaluationSplitV1, bool)> = checkpoint
                .observations
                .iter()
                .map(|item| (item.item_id.clone(), (item.split, item.intended_member)))
                .collect();
            if pair_item_contract
                .as_ref()
                .is_some_and(|previous| previous != &item_contract)
            {
                return Err("candidate IDs, calibration/test split and randomized member labels must be identical across checkpoints".to_owned());
            }
            if pair_detector_contract
                .as_ref()
                .is_some_and(|previous| previous != &directions)
            {
                return Err("detector IDs and score directions must be identical across checkpoints".to_owned());
            }
            pair_item_contract = Some(item_contract);
            pair_detector_contract = Some(directions);
        }
        if TrainingStageV1::ALL
            .iter()
            .any(|stage| !checkpoints.contains_key(stage))
        {
            return Err("model pair is missing a required training stage".to_owned());
        }
        let expected_parent = |stage: TrainingStageV1| -> Result<&str, String> {
            match stage {
                TrainingStageV1::SftClean | TrainingStageV1::SftContaminated => {
                    Ok(pair.base_checkpoint_sha256.as_str())
                }
                TrainingStageV1::SftCleanThenGrpoClean => Ok(
                    &checkpoints[&TrainingStageV1::SftClean].checkpoint_sha256,
                ),
                TrainingStageV1::SftContaminatedThenGrpoClean => Ok(
                    &checkpoints[&TrainingStageV1::SftContaminated].checkpoint_sha256,
                ),
                TrainingStageV1::SftCleanThenRaft => Ok(
                    &checkpoints[&TrainingStageV1::SftClean].checkpoint_sha256,
                ),
                TrainingStageV1::SftContaminatedThenRaft => Ok(
                    &checkpoints[&TrainingStageV1::SftContaminated].checkpoint_sha256,
                ),
                TrainingStageV1::SftCleanThenRaftPlusPlus => Ok(
                    &checkpoints[&TrainingStageV1::SftClean].checkpoint_sha256,
                ),
                TrainingStageV1::SftContaminatedThenRaftPlusPlus => Ok(
                    &checkpoints[&TrainingStageV1::SftContaminated].checkpoint_sha256,
                ),
                TrainingStageV1::AdvancedLrmClean
                | TrainingStageV1::AdvancedLrmCotContaminated => {
                    Ok(pair.advanced_lrm_base_checkpoint_sha256.as_str())
                }
            }
        };
        for stage in TrainingStageV1::ALL {
            let checkpoint = checkpoints[&stage];
            if checkpoint.parent_checkpoint_sha256 != expected_parent(stage)? {
                return Err(format!("{stage:?} checkpoint has an unexpected lineage parent"));
            }
        }
        let matched_objective_pairs = [
            (TrainingStageV1::SftClean, TrainingStageV1::SftContaminated),
            (
                TrainingStageV1::SftCleanThenGrpoClean,
                TrainingStageV1::SftContaminatedThenGrpoClean,
            ),
            (TrainingStageV1::SftCleanThenRaft, TrainingStageV1::SftContaminatedThenRaft),
            (
                TrainingStageV1::SftCleanThenRaftPlusPlus,
                TrainingStageV1::SftContaminatedThenRaftPlusPlus,
            ),
            (
                TrainingStageV1::AdvancedLrmClean,
                TrainingStageV1::AdvancedLrmCotContaminated,
            ),
        ];
        for (clean_stage, treatment_stage) in matched_objective_pairs {
            let clean = checkpoints[&clean_stage];
            let treatment = checkpoints[&treatment_stage];
            if clean.objective_config_sha256 != treatment.objective_config_sha256
                || clean.optimizer_steps != treatment.optimizer_steps
            {
                return Err(format!(
                    "matched clean/treatment arms {clean_stage:?}/{treatment_stage:?} must use identical objective config and optimizer-step budgets"
                ));
            }
        }
        if reference_item_contract
            .as_ref()
            .is_some_and(|previous| Some(previous) != pair_item_contract.as_ref())
            || reference_detector_contract
                .as_ref()
                .is_some_and(|previous| Some(previous) != pair_detector_contract.as_ref())
        {
            return Err("candidate split or detector contract differs across independent model pairs".to_owned());
        }
        reference_item_contract = pair_item_contract.clone();
        reference_detector_contract = pair_detector_contract.clone();

        let detector_directions = pair_detector_contract
            .as_ref()
            .ok_or_else(|| "model-pair detector contract is missing".to_owned())?;
        let mut checkpoint_reports = Vec::with_capacity(TrainingStageV1::ALL.len());
        for stage in TrainingStageV1::ALL {
            let checkpoint = checkpoints
                .get(&stage)
                .ok_or_else(|| "required checkpoint stage disappeared".to_owned())?;
            checkpoint_reports.push(FragilityCheckpointReportV1 {
                stage,
                checkpoint_sha256: checkpoint.checkpoint_sha256.clone(),
                parent_checkpoint_sha256: checkpoint.parent_checkpoint_sha256.clone(),
                training_receipt_sha256: checkpoint.training_receipt_sha256.clone(),
                training_config_sha256: checkpoint.training_config_sha256.clone(),
                objective_config_sha256: checkpoint.objective_config_sha256.clone(),
                training_data_manifest_sha256: checkpoint.training_data_manifest_sha256.clone(),
                optimizer_steps: checkpoint.optimizer_steps,
                performance: performance_for_stage(checkpoint)?,
                detectors: checkpoint_detector_metrics(
                    checkpoint,
                    detector_directions,
                    study.target_fpr_permille,
                    study.minimum_calibration_negatives,
                    study.minimum_test_negatives,
                    study.minimum_test_members,
                )?,
            });
        }
        let report_map: BTreeMap<TrainingStageV1, FragilityCheckpointReportV1> =
            checkpoint_reports.iter().cloned().map(|report| (report.stage, report)).collect();
        let comparisons = InterventionComparisonV1::ALL
            .into_iter()
            .map(|intervention| build_intervention_comparison(intervention, &report_map))
            .collect::<Result<Vec<_>, _>>()?;
        pair_reports.push(FragilityPairReportV1 {
            replicate_id: pair.replicate_id.clone(),
            checkpoint_reports,
            intervention_comparisons: comparisons,
        });
    }

    let aggregate_interventions = aggregate_interventions(&pair_reports)?;
    let issues = vec![
        "held-out fixed-FPR rates use Wilson 95% intervals; these are not exact Clopper-Pearson bounds".to_owned(),
        "AUROC/AUPRC and operating-point estimates are conditional on the frozen candidate universe and supplied detector scores".to_owned(),
        "the concealment gap is a derived diagnostic; report detector signal retention and performance-inflation retention separately".to_owned(),
        "per-item membership labels encode randomized exposure assignment, not proof of pretraining absence".to_owned(),
        "checkpoint and training-receipt digests are content commitments, not authenticated custody or proof of data consumption".to_owned(),
        "aggregate intervals are unadjusted across interventions, detectors and domains".to_owned(),
    ];

    Ok(DetectorFragilityReportV1 {
        schema: FRAGILITY_REPORT_SCHEMA.to_owned(),
        study_id: study.study_id.clone(),
        benchmark_id: study.benchmark_id.clone(),
        benchmark_version: study.benchmark_version.clone(),
        study_sha256,
        target_fpr_permille: study.target_fpr_permille,
        independent_model_pair_replicate_count: pair_reports.len() as u64,
        pair_reports,
        aggregate_interventions,
        intervals_multiplicity_adjusted: false,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: "exploratory_detector_fragility".to_owned(),
        issues,
        claim_boundary: "This report quantifies detector separability before and after preregistered training transformations and compares detector-signal retention with member-performance inflation relative to matched clean controls. Candidate items, repeated generations and detector calls are nested within independent model-pair training replicates. The analysis does not prove absence of pretraining exposure, authenticate checkpoint/data custody, prove that a training run consumed the declared data, identify deliberate evasion, establish universal detector failure, or make a multiplicity-adjusted confirmatory claim. Wilson intervals are reported for fixed-threshold test rates; they are not exact binomial confidence bounds.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn score(id: &str, raw: f64) -> DetectorScoreV1 {
        DetectorScoreV1 {
            detector_id: id.to_owned(),
            detector_spec_sha256: digest('a'),
            direction: DetectorDirectionV1::HigherIsMember,
            raw_score: raw,
        }
    }

    #[test]
    fn auc_is_rank_based_and_handles_ties() {
        let scores = vec![
            (0.9, true),
            (0.8, true),
            (0.2, false),
            (0.1, false),
        ];
        let (auc, ap) = auc_and_auprc(&scores).unwrap();
        assert_eq!(auc, 1.0);
        assert_eq!(ap, 1.0);

        let tied = vec![(0.5, true), (0.5, false), (0.5, true), (0.5, false)];
        let (tied_auc, _) = auc_and_auprc(&tied).unwrap();
        assert!((tied_auc - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn lower_is_member_direction_is_normalized_before_metrics() {
        let values = [
            (0.1, true),
            (0.2, true),
            (0.8, false),
            (0.9, false),
        ];
        let oriented: Vec<(f64, bool)> = values
            .iter()
            .map(|(score, member)| (-score, *member))
            .collect();
        let (auc, _) = auc_and_auprc(&oriented).unwrap();
        assert_eq!(auc, 1.0);
    }

    #[test]
    fn wilson_interval_contains_observed_rate() {
        let interval = wilson_interval(10, 100).unwrap();
        assert!(interval.lower_95 <= 0.1);
        assert!(interval.upper_95 >= 0.1);
    }

    #[test]
    fn zero_pre_intervention_signal_has_undefined_retention() {
        assert_eq!(retention_ratio(100.0, 0.0), None);
        assert_eq!(retention_ratio(100.0, 200.0), Some(500.0));
    }

    fn stage_hash(stage: TrainingStageV1) -> char {
        let index = TrainingStageV1::ALL
            .iter()
            .position(|candidate| *candidate == stage)
            .unwrap();
        char::from_digit(index as u32, 10).unwrap()
    }

    fn stage_parent(
        stage: TrainingStageV1,
        base: &str,
        advanced: &str,
    ) -> String {
        match stage {
            TrainingStageV1::SftClean | TrainingStageV1::SftContaminated => base.to_owned(),
            TrainingStageV1::SftCleanThenGrpoClean => digest(stage_hash(TrainingStageV1::SftClean)),
            TrainingStageV1::SftContaminatedThenGrpoClean => digest(stage_hash(TrainingStageV1::SftContaminated)),
            TrainingStageV1::SftCleanThenRaft => digest(stage_hash(TrainingStageV1::SftClean)),
            TrainingStageV1::SftContaminatedThenRaft => digest(stage_hash(TrainingStageV1::SftContaminated)),
            TrainingStageV1::SftCleanThenRaftPlusPlus => digest(stage_hash(TrainingStageV1::SftClean)),
            TrainingStageV1::SftContaminatedThenRaftPlusPlus => digest(stage_hash(TrainingStageV1::SftContaminated)),
            TrainingStageV1::AdvancedLrmClean | TrainingStageV1::AdvancedLrmCotContaminated => advanced.to_owned(),
        }
    }

    fn objective(stage: TrainingStageV1) -> char {
        match stage {
            TrainingStageV1::SftClean | TrainingStageV1::SftContaminated => 'c',
            TrainingStageV1::SftCleanThenGrpoClean | TrainingStageV1::SftContaminatedThenGrpoClean => 'd',
            TrainingStageV1::SftCleanThenRaft | TrainingStageV1::SftContaminatedThenRaft => 'e',
            TrainingStageV1::SftCleanThenRaftPlusPlus | TrainingStageV1::SftContaminatedThenRaftPlusPlus => 'f',
            TrainingStageV1::AdvancedLrmClean | TrainingStageV1::AdvancedLrmCotContaminated => 'g',
        }
    }

    fn steps(stage: TrainingStageV1) -> u64 {
        match stage {
            TrainingStageV1::SftClean | TrainingStageV1::SftContaminated => 100,
            TrainingStageV1::SftCleanThenGrpoClean | TrainingStageV1::SftContaminatedThenGrpoClean => 156,
            TrainingStageV1::SftCleanThenRaft | TrainingStageV1::SftContaminatedThenRaft => 156,
            TrainingStageV1::SftCleanThenRaftPlusPlus | TrainingStageV1::SftContaminatedThenRaftPlusPlus => 156,
            TrainingStageV1::AdvancedLrmClean | TrainingStageV1::AdvancedLrmCotContaminated => 20,
        }
    }

    fn member_accuracy(stage: TrainingStageV1) -> usize {
        match stage {
            TrainingStageV1::SftClean => 70,
            TrainingStageV1::SftContaminated => 90,
            TrainingStageV1::SftCleanThenGrpoClean => 72,
            TrainingStageV1::SftContaminatedThenGrpoClean => 90,
            TrainingStageV1::SftCleanThenRaft => 72,
            TrainingStageV1::SftContaminatedThenRaft => 88,
            TrainingStageV1::SftCleanThenRaftPlusPlus => 72,
            TrainingStageV1::SftContaminatedThenRaftPlusPlus => 87,
            TrainingStageV1::AdvancedLrmClean => 80,
            TrainingStageV1::AdvancedLrmCotContaminated => 90,
        }
    }

    fn member_score(stage: TrainingStageV1, member: bool) -> f64 {
        match stage {
            TrainingStageV1::SftContaminated if member => 0.9,
            TrainingStageV1::SftContaminated => 0.1,
            TrainingStageV1::SftContaminatedThenGrpoClean => 0.5,
            TrainingStageV1::SftContaminatedThenRaft if member => 0.9,
            TrainingStageV1::SftContaminatedThenRaft => 0.2,
            TrainingStageV1::SftContaminatedThenRaftPlusPlus if member => 0.7,
            TrainingStageV1::SftContaminatedThenRaftPlusPlus => 0.6,
            TrainingStageV1::SftClean | TrainingStageV1::SftCleanThenGrpoClean
            | TrainingStageV1::SftCleanThenRaft | TrainingStageV1::SftCleanThenRaftPlusPlus
            | TrainingStageV1::AdvancedLrmClean
            | TrainingStageV1::AdvancedLrmCotContaminated => 0.5,
        }
    }

    fn observations(stage: TrainingStageV1) -> Vec<FragilityItemObservationV1> {
        let mut output = Vec::with_capacity(700);
        for index in 0..300 {
            output.push(FragilityItemObservationV1 {
                item_id: format!("cal-neg-{index:04}"),
                split: EvaluationSplitV1::Calibration,
                intended_member: false,
                correct: false,
                detector_scores: vec![score("min_k", 0.5)],
            });
        }
        for index in 0..100 {
            output.push(FragilityItemObservationV1 {
                item_id: format!("test-member-{index:04}"),
                split: EvaluationSplitV1::Test,
                intended_member: true,
                correct: index < member_accuracy(stage),
                detector_scores: vec![score("min_k", member_score(stage, true))],
            });
        }
        for index in 0..300 {
            output.push(FragilityItemObservationV1 {
                item_id: format!("test-nonmember-{index:04}"),
                split: EvaluationSplitV1::Test,
                intended_member: false,
                correct: index < 165,
                detector_scores: vec![score("min_k", member_score(stage, false))],
            });
        }
        output
    }

    fn synthetic_study() -> DetectorFragilityStudyV1 {
        let base = digest('a');
        let advanced = digest('b');
        let mut pairs = Vec::new();
        for replicate in 0..2 {
            let checkpoints = TrainingStageV1::ALL
                .iter()
                .map(|stage| {
                    let stage_byte = stage_hash(*stage);
                    FragilityCheckpointV1 {
                        stage: *stage,
                        checkpoint_sha256: digest(stage_byte),
                        parent_checkpoint_sha256: stage_parent(*stage, &base, &advanced),
                        training_receipt_sha256: digest('h'),
                        training_config_sha256: digest(char::from_digit(replicate + 1, 10).unwrap()),
                        objective_config_sha256: digest(objective(*stage)),
                        training_data_manifest_sha256: digest(char::from_digit(replicate + 3, 10).unwrap()),
                        optimizer_steps: steps(*stage),
                        observations: observations(*stage),
                    }
                })
                .collect();
            pairs.push(FragilityModelPairV1 {
                replicate_id: format!("pair-{:02}", replicate + 1),
                base_checkpoint_sha256: base.clone(),
                advanced_lrm_base_checkpoint_sha256: advanced.clone(),
                randomization_manifest_sha256: digest(char::from_digit(replicate + 5, 10).unwrap()),
                checkpoints,
            });
        }
        DetectorFragilityStudyV1 {
            schema: FRAGILITY_STUDY_SCHEMA.to_owned(),
            study_id: "fragility-pilot-001".to_owned(),
            benchmark_id: "frozen-benchmark-001".to_owned(),
            benchmark_version: "v1".to_owned(),
            target_fpr_permille: 10,
            minimum_calibration_negatives: 300,
            minimum_test_negatives: 300,
            minimum_test_members: 100,
            model_pair_replicates: pairs,
        }
    }

    #[test]
    fn full_fragility_analysis_detects_inflation_preserved_with_auc_concealment() {
        let report = analyze_detector_fragility(&synthetic_study()).unwrap();
        assert_eq!(report.independent_model_pair_replicate_count, 2);
        assert!(!report.confirmatory_eligible);
        let pair = &report.pair_reports[0];
        let grpo = pair
            .intervention_comparisons
            .iter()
            .find(|row| row.intervention == InterventionComparisonV1::SftThenCleanGrpo)
            .unwrap();
        assert_eq!(
            grpo.member_performance_inflation_before_permille,
            Some(200.0)
        );
        assert_eq!(
            grpo.member_performance_inflation_after_permille,
            180.0
        );
        assert_eq!(grpo.member_inflation_retention_permille, Some(900.0));
        let detector = &grpo.detectors[0];
        assert_eq!(detector.auroc_before, Some(1.0));
        assert_eq!(detector.auroc_after, 0.5);
        assert_eq!(detector.signal_retention_permille, Some(0.0));
        assert_eq!(detector.concealment_gap_permille, Some(900.0));
    }

    #[test]
    fn missing_checkpoint_stage_is_rejected() {
        let mut study = synthetic_study();
        study.model_pair_replicates[0].checkpoints.pop();
        assert!(analyze_detector_fragility(&study).is_err());
    }

    #[test]
    fn stage_contract_contains_all_intervention_controls() {
        assert_eq!(TrainingStageV1::ALL.len(), 10);
        assert_eq!(InterventionComparisonV1::ALL.len(), 4);
        assert!(InterventionComparisonV1::SftThenCleanGrpo
            .stages()
            .0
            .is_some());
        assert!(InterventionComparisonV1::AdvancedLrmCotSft
            .stages()
            .0
            .is_none());
    }
}
