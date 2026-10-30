//! Cross-task transfer qualification and negative-transfer guard for NAS/HPO.
//!
//! This module is an evidence contract, not a Gaussian process or learning-curve predictor.
//! It separates historical source tasks, transfer-calibration tasks and evaluation targets,
//! then decides whether a proposed historical prior is safe and useful enough to apply.
//!
//! The gate is intentionally conservative:
//! - all tasks must share declared objective semantics and metric direction;
//! - calibration tasks and source tasks are disjoint from each other and the target;
//! - each calibration task is evaluated against a target-only baseline with paired seeds;
//! - every calibration task must pass a per-task regression guard;
//! - transfer is applied only when the task-weighted pooled 95% upper confidence bound
//!   demonstrates lower normalized regret than the target-only baseline;
//! - insufficient or ambiguous evidence leads to abstention (prior weight = 0).
//!
//! It does not prove transfer to an unseen task. It prevents a transfer prior from being
//! automatically trusted merely because a similarity model assigned it high relevance.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

use super::tabular_nas_firewall::MetricDirection;

const TASK_TRANSFER_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.nas.task-transfer-calibration-evidence.v0.1";
const TASK_TRANSFER_PROPOSAL_DOMAIN: &[u8] =
    b"symthaea.nas.task-transfer-proposal.v0.1";
const TASK_TRANSFER_RECEIPT_DOMAIN: &[u8] =
    b"symthaea.nas.task-transfer-qualification-receipt.v0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskRole {
    HistoricalSource,
    TransferCalibration,
    EvaluationTarget,
}

/// A task manifest pins the identity and semantics of one NAS/HPO experiment.
/// Search-space and dataset IDs may differ across tasks; objective semantics and direction
/// must match before normalized-regret transfer evidence can be compared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskManifest {
    pub task_id: String,
    pub role: TaskRole,
    pub search_space_id: String,
    pub dataset_id: String,
    pub objective_semantics_id: String,
    pub metric_direction: MetricDirection,
    pub artifact_sha256: String,
    pub split_sha256: String,
    pub fidelity_ladder_sha256: String,
    pub resource_profile_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceTaskWeight {
    pub task_id: String,
    /// Non-negative relevance score in [0, 1].
    pub relevance: f64,
    /// Normalized prior aggregation weight. All source weights must sum to one.
    pub prior_weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransferProposal {
    pub model_sha256: String,
    pub target_task_id: String,
    pub source_tasks: Vec<SourceTaskWeight>,
    pub predicted_target_regret_percent: f64,
    pub predictive_stddev_regret_percent: f64,
    /// The model's requested prior contribution. The gate sets this to zero on abstention.
    pub requested_prior_weight: f64,
}

/// One paired seed comparison on a calibration task. Lower normalized regret is better.
/// Positive delta means transfer made the result worse.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedTransferOutcome {
    pub task_id: String,
    pub seed: u64,
    pub target_only_regret_percent: f64,
    pub transferred_regret_percent: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransferCalibrationEvidence {
    pub model_sha256: String,
    pub calibration_task_ids: Vec<String>,
    pub outcomes: Vec<PairedTransferOutcome>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransferGateConfig {
    pub minimum_calibration_tasks: usize,
    pub minimum_paired_seeds_per_task: usize,
    /// Maximum tolerated per-task regression, in normalized-regret percentage points.
    /// Example: 5.0 permits at most five points of uncertainty-bounded regression.
    pub maximum_task_regression_pp: f64,
    /// Normal quantile used for paired and task-cluster confidence bounds.
    pub confidence_z: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferGateDecision {
    ApplyPrior,
    AbstainInsufficientCalibration,
    AbstainUncertainBenefit,
    AbstainNoDemonstratedBenefit,
    RejectNegativeTransfer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaskTransferStatistics {
    pub task_id: String,
    pub paired_seed_count: usize,
    /// Mean(transferred regret - target-only regret). Positive is harmful.
    pub mean_regret_delta_pp: Option<f64>,
    pub standard_error_pp: Option<f64>,
    pub lower_confidence_bound_pp: Option<f64>,
    pub upper_confidence_bound_pp: Option<f64>,
    pub passed_regression_guard: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransferQualificationReceipt {
    pub target_task_id: String,
    pub target_artifact_sha256: String,
    pub model_sha256: String,
    pub source_task_ids: Vec<String>,
    pub calibration_task_ids: Vec<String>,
    pub source_prior_weight_requested: f64,
    pub source_prior_weight_applied: f64,
    pub predicted_target_regret_percent: f64,
    pub predictive_stddev_regret_percent: f64,
    pub mean_task_regret_delta_pp: Option<f64>,
    pub pooled_lower_confidence_bound_pp: Option<f64>,
    pub pooled_upper_confidence_bound_pp: Option<f64>,
    pub maximum_task_regression_pp: f64,
    pub task_statistics: Vec<TaskTransferStatistics>,
    pub proposal_sha256: String,
    pub calibration_evidence_sha256: String,
    pub decision: TransferGateDecision,
    pub decision_reason: String,
    pub receipt_sha256: String,
}

impl TransferQualificationReceipt {
    /// Recompute the receipt digest so consumers can detect edits to the decision or evidence.
    pub fn verify_digest(&self) -> bool {
        digest_receipt(self) == self.receipt_sha256
    }

    pub fn prior_is_applied(&self) -> bool {
        self.decision == TransferGateDecision::ApplyPrior
            && self.source_prior_weight_applied > 0.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskTransferGateError {
    pub field: String,
    pub message: String,
}

impl TaskTransferGateError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for TaskTransferGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for TaskTransferGateError {}

pub struct CrossTaskTransferGate {
    config: TransferGateConfig,
    tasks: HashMap<String, TaskManifest>,
}

impl CrossTaskTransferGate {
    pub fn new(
        config: TransferGateConfig,
        tasks: Vec<TaskManifest>,
    ) -> Result<Self, TaskTransferGateError> {
        validate_config(&config)?;
        if tasks.is_empty() {
            return Err(TaskTransferGateError::new(
                "tasks",
                "at least one task manifest is required",
            ));
        }
        let mut task_map = HashMap::with_capacity(tasks.len());
        for (index, task) in tasks.into_iter().enumerate() {
            validate_task_manifest(&task, index)?;
            if task_map.insert(task.task_id.clone(), task).is_some() {
                return Err(TaskTransferGateError::new(
                    format!("tasks[{index}].task_id"),
                    "duplicate task ID",
                ));
            }
        }
        if !task_map.values().any(|task| task.role == TaskRole::EvaluationTarget) {
            return Err(TaskTransferGateError::new(
                "tasks",
                "at least one EvaluationTarget task is required",
            ));
        }
        Ok(Self {
            config,
            tasks: task_map,
        })
    }

    /// Qualify a proposed historical prior against paired target-only vs transferred runs
    /// on disjoint calibration tasks. A returned receipt always records Apply or Abstain.
    pub fn qualify(
        &self,
        proposal: &TransferProposal,
        evidence: &TransferCalibrationEvidence,
    ) -> Result<TransferQualificationReceipt, TaskTransferGateError> {
        let target = self.tasks.get(&proposal.target_task_id).ok_or_else(|| {
            TaskTransferGateError::new(
                "proposal.target_task_id",
                "target task is not present in the task manifest registry",
            )
        })?;
        if target.role != TaskRole::EvaluationTarget {
            return Err(TaskTransferGateError::new(
                "proposal.target_task_id",
                "proposal target must have EvaluationTarget role",
            ));
        }
        validate_sha256(&proposal.model_sha256, "proposal.model_sha256")?;
        validate_sha256(&evidence.model_sha256, "evidence.model_sha256")?;
        if proposal.model_sha256 != evidence.model_sha256 {
            return Err(TaskTransferGateError::new(
                "evidence.model_sha256",
                "calibration evidence must qualify the exact proposed transfer model",
            ));
        }
        if !proposal.predicted_target_regret_percent.is_finite()
            || proposal.predicted_target_regret_percent < 0.0
        {
            return Err(TaskTransferGateError::new(
                "proposal.predicted_target_regret_percent",
                "must be finite and non-negative",
            ));
        }
        if !proposal.predictive_stddev_regret_percent.is_finite()
            || proposal.predictive_stddev_regret_percent < 0.0
        {
            return Err(TaskTransferGateError::new(
                "proposal.predictive_stddev_regret_percent",
                "must be finite and non-negative",
            ));
        }
        if !proposal.requested_prior_weight.is_finite()
            || !(0.0..=1.0).contains(&proposal.requested_prior_weight)
        {
            return Err(TaskTransferGateError::new(
                "proposal.requested_prior_weight",
                "must be finite and in [0,1]",
            ));
        }

        let mut source_ids = Vec::with_capacity(proposal.source_tasks.len());
        let mut source_seen = HashSet::with_capacity(proposal.source_tasks.len());
        let mut source_split_sha256s = HashSet::with_capacity(proposal.source_tasks.len());
        let mut source_weight_sum = 0.0;
        for (index, source) in proposal.source_tasks.iter().enumerate() {
            let task = self.tasks.get(&source.task_id).ok_or_else(|| {
                TaskTransferGateError::new(
                    format!("proposal.source_tasks[{index}].task_id"),
                    "source task is not in the task registry",
                )
            })?;
            if task.role != TaskRole::HistoricalSource {
                return Err(TaskTransferGateError::new(
                    format!("proposal.source_tasks[{index}].task_id"),
                    "only HistoricalSource tasks may contribute a transfer prior",
                ));
            }
            if !source_seen.insert(source.task_id.as_str()) {
                return Err(TaskTransferGateError::new(
                    format!("proposal.source_tasks[{index}].task_id"),
                    "duplicate source task",
                ));
            }
            if !source.relevance.is_finite()
                || !(0.0..=1.0).contains(&source.relevance)
            {
                return Err(TaskTransferGateError::new(
                    format!("proposal.source_tasks[{index}].relevance"),
                    "must be finite and in [0,1]",
                ));
            }
            if !source.prior_weight.is_finite()
                || !(0.0..=1.0).contains(&source.prior_weight)
            {
                return Err(TaskTransferGateError::new(
                    format!("proposal.source_tasks[{index}].prior_weight"),
                    "must be finite and in [0,1]",
                ));
            }
            validate_objective_compatibility(task, target, "proposal.source_tasks")?;
            if task.split_sha256 == target.split_sha256 {
                return Err(TaskTransferGateError::new(
                    format!("proposal.source_tasks[{index}].task_id"),
                    "historical source split digest must differ from evaluation target split",
                ));
            }
            source_split_sha256s.insert(task.split_sha256.as_str());
            source_weight_sum += source.prior_weight;
            source_ids.push(source.task_id.clone());
        }
        if proposal.source_tasks.is_empty() {
            return Err(TaskTransferGateError::new(
                "proposal.source_tasks",
                "at least one historical source task is required",
            ));
        }
        if (source_weight_sum - 1.0).abs() > 1e-9 {
            return Err(TaskTransferGateError::new(
                "proposal.source_tasks",
                "historical prior weights must sum to one",
            ));
        }

        let calibration_ids = validate_calibration_task_ids(
            &self.tasks,
            target,
            &source_seen,
            &source_split_sha256s,
            evidence,
        )?;
        let calibration_set: HashSet<&str> =
            calibration_ids.iter().map(String::as_str).collect();

        let mut outcomes_by_task: BTreeMap<String, Vec<&PairedTransferOutcome>> =
            BTreeMap::new();
        let mut observed_pairs: HashSet<(&str, u64)> = HashSet::new();
        for (index, outcome) in evidence.outcomes.iter().enumerate() {
            if !calibration_set.contains(outcome.task_id.as_str()) {
                return Err(TaskTransferGateError::new(
                    format!("evidence.outcomes[{index}].task_id"),
                    "outcome is not from a declared calibration task",
                ));
            }
            if !outcome.target_only_regret_percent.is_finite()
                || outcome.target_only_regret_percent < 0.0
                || !outcome.transferred_regret_percent.is_finite()
                || outcome.transferred_regret_percent < 0.0
            {
                return Err(TaskTransferGateError::new(
                    format!("evidence.outcomes[{index}]"),
                    "normalized regret values must be finite and non-negative",
                ));
            }
            if !observed_pairs.insert((outcome.task_id.as_str(), outcome.seed)) {
                return Err(TaskTransferGateError::new(
                    format!("evidence.outcomes[{index}]"),
                    "duplicate task/seed observation",
                ));
            }
            outcomes_by_task
                .entry(outcome.task_id.clone())
                .or_default()
                .push(outcome);
        }

        let mut task_statistics = Vec::with_capacity(calibration_ids.len());
        let mut task_means = Vec::new();
        let mut insufficient = calibration_ids.len() < self.config.minimum_calibration_tasks;

        for task_id in &calibration_ids {
            let outcomes = outcomes_by_task.get(task_id).map(Vec::as_slice).unwrap_or(&[]);
            let deltas: Vec<f64> = outcomes
                .iter()
                .map(|outcome| {
                    outcome.transferred_regret_percent
                        - outcome.target_only_regret_percent
                })
                .collect();
            let mean = mean(&deltas);
            let standard_error = sample_standard_error(&deltas);
            let half_width = standard_error.map(|se| self.config.confidence_z * se);
            let lower = mean.zip(half_width).map(|(m, h)| m - h);
            let upper = mean.zip(half_width).map(|(m, h)| m + h);
            let enough_seeds = deltas.len() >= self.config.minimum_paired_seeds_per_task;
            insufficient |= !enough_seeds;
            if let Some(mean_delta) = mean {
                task_means.push(mean_delta);
            }
            let passed = upper.is_some_and(|bound| {
                bound <= self.config.maximum_task_regression_pp
            });
            task_statistics.push(TaskTransferStatistics {
                task_id: task_id.clone(),
                paired_seed_count: deltas.len(),
                mean_regret_delta_pp: mean,
                standard_error_pp: standard_error,
                lower_confidence_bound_pp: lower,
                upper_confidence_bound_pp: upper,
                passed_regression_guard: passed,
            });
        }

        let pooled_mean = mean(&task_means);
        let pooled_standard_error = sample_standard_error(&task_means);
        let pooled_half_width =
            pooled_standard_error.map(|se| self.config.confidence_z * se);
        let pooled_lower = pooled_mean.zip(pooled_half_width).map(|(m, h)| m - h);
        let pooled_upper = pooled_mean.zip(pooled_half_width).map(|(m, h)| m + h);

        let any_confirmed_harm = task_statistics.iter().any(|stats| {
            stats.lower_confidence_bound_pp.is_some_and(|lower| {
                lower > self.config.maximum_task_regression_pp
            })
        });
        let any_task_uncertain = task_statistics.iter().any(|stats| {
            stats.upper_confidence_bound_pp.is_none_or(|upper| {
                upper > self.config.maximum_task_regression_pp
            })
        });

        let (decision, reason) = if insufficient {
            (
                TransferGateDecision::AbstainInsufficientCalibration,
                "calibration task/seed minimum not met; historical prior is not applied".to_string(),
            )
        } else if any_confirmed_harm {
            (
                TransferGateDecision::RejectNegativeTransfer,
                "at least one calibration task shows regression above the configured tolerance".to_string(),
            )
        } else if any_task_uncertain {
            (
                TransferGateDecision::AbstainUncertainBenefit,
                "at least one calibration task has an upper confidence bound above the regression tolerance".to_string(),
            )
        } else if pooled_upper.is_some_and(|upper| upper < 0.0) {
            (
                TransferGateDecision::ApplyPrior,
                "each calibration task passes the regression guard and pooled task-weighted regret improves".to_string(),
            )
        } else {
            (
                TransferGateDecision::AbstainNoDemonstratedBenefit,
                "regression guard passes, but pooled calibration does not demonstrate a transfer benefit".to_string(),
            )
        };

        let calibration_evidence_sha256 =
            digest_calibration_evidence(evidence, &calibration_ids);
        let proposal_sha256 = digest_proposal(proposal);
        let mut receipt = TransferQualificationReceipt {
            target_task_id: target.task_id.clone(),
            target_artifact_sha256: target.artifact_sha256.clone(),
            model_sha256: proposal.model_sha256.clone(),
            source_task_ids: source_ids,
            calibration_task_ids: calibration_ids,
            source_prior_weight_requested: proposal.requested_prior_weight,
            source_prior_weight_applied: if decision == TransferGateDecision::ApplyPrior {
                proposal.requested_prior_weight
            } else {
                0.0
            },
            predicted_target_regret_percent: proposal.predicted_target_regret_percent,
            predictive_stddev_regret_percent: proposal.predictive_stddev_regret_percent,
            mean_task_regret_delta_pp: pooled_mean,
            pooled_lower_confidence_bound_pp: pooled_lower,
            pooled_upper_confidence_bound_pp: pooled_upper,
            maximum_task_regression_pp: self.config.maximum_task_regression_pp,
            task_statistics,
            proposal_sha256,
            calibration_evidence_sha256,
            decision,
            decision_reason: reason,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = digest_receipt(&receipt);
        Ok(receipt)
    }
}

fn validate_config(config: &TransferGateConfig) -> Result<(), TaskTransferGateError> {
    if config.minimum_calibration_tasks < 2 {
        return Err(TaskTransferGateError::new(
            "minimum_calibration_tasks",
            "must be at least two for task-level uncertainty estimation",
        ));
    }
    if config.minimum_paired_seeds_per_task < 2 {
        return Err(TaskTransferGateError::new(
            "minimum_paired_seeds_per_task",
            "must be at least two for paired uncertainty estimation",
        ));
    }
    if !config.maximum_task_regression_pp.is_finite()
        || config.maximum_task_regression_pp < 0.0
    {
        return Err(TaskTransferGateError::new(
            "maximum_task_regression_pp",
            "must be finite and non-negative",
        ));
    }
    if !config.confidence_z.is_finite() || config.confidence_z <= 0.0 {
        return Err(TaskTransferGateError::new(
            "confidence_z",
            "must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_task_manifest(
    task: &TaskManifest,
    index: usize,
) -> Result<(), TaskTransferGateError> {
    for (field, value) in [
        ("task_id", task.task_id.as_str()),
        ("search_space_id", task.search_space_id.as_str()),
        ("dataset_id", task.dataset_id.as_str()),
        ("objective_semantics_id", task.objective_semantics_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(TaskTransferGateError::new(
                format!("tasks[{index}].{field}"),
                "must be non-empty",
            ));
        }
    }
    for (field, digest) in [
        ("artifact_sha256", task.artifact_sha256.as_str()),
        ("split_sha256", task.split_sha256.as_str()),
        ("fidelity_ladder_sha256", task.fidelity_ladder_sha256.as_str()),
        (
            "resource_profile_sha256",
            task.resource_profile_sha256.as_str(),
        ),
    ] {
        validate_sha256(digest, &format!("tasks[{index}].{field}"))?;
    }
    Ok(())
}

fn validate_objective_compatibility(
    source: &TaskManifest,
    target: &TaskManifest,
    field: &str,
) -> Result<(), TaskTransferGateError> {
    if source.objective_semantics_id != target.objective_semantics_id
        || source.metric_direction != target.metric_direction
    {
        return Err(TaskTransferGateError::new(
            field,
            format!(
                "task {:?} objective semantics/direction differ from target {:?}",
                source.task_id, target.task_id
            ),
        ));
    }
    Ok(())
}

fn validate_calibration_task_ids(
    tasks: &HashMap<String, TaskManifest>,
    target: &TaskManifest,
    source_seen: &HashSet<&str>,
    source_split_sha256s: &HashSet<&str>,
    evidence: &TransferCalibrationEvidence,
) -> Result<Vec<String>, TaskTransferGateError> {
    if evidence.calibration_task_ids.is_empty() {
        return Err(TaskTransferGateError::new(
            "evidence.calibration_task_ids",
            "at least one calibration task must be declared",
        ));
    }
    let mut ids = BTreeSet::new();
    for (index, task_id) in evidence.calibration_task_ids.iter().enumerate() {
        if !ids.insert(task_id.as_str()) {
            return Err(TaskTransferGateError::new(
                format!("evidence.calibration_task_ids[{index}]"),
                "duplicate calibration task ID",
            ));
        }
        if task_id == &target.task_id || source_seen.contains(task_id.as_str()) {
            return Err(TaskTransferGateError::new(
                format!("evidence.calibration_task_ids[{index}]"),
                "calibration task must be disjoint from target and historical source tasks",
            ));
        }
        let task = tasks.get(task_id).ok_or_else(|| {
            TaskTransferGateError::new(
                format!("evidence.calibration_task_ids[{index}]"),
                "calibration task is not in the task registry",
            )
        })?;
        if task.role != TaskRole::TransferCalibration {
            return Err(TaskTransferGateError::new(
                format!("evidence.calibration_task_ids[{index}]"),
                "calibration ID must have TransferCalibration role",
            ));
        }
        validate_objective_compatibility(task, target, "evidence.calibration_task_ids")?;
        if task.split_sha256 == target.split_sha256
            || source_split_sha256s.contains(task.split_sha256.as_str())
        {
            return Err(TaskTransferGateError::new(
                format!("evidence.calibration_task_ids[{index}]"),
                "calibration split digest must differ from target and all historical source splits",
            ));
        }
    }
    Ok(ids.into_iter().map(str::to_string).collect())
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn sample_standard_error(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let average = mean(values)?;
    let sum_squared = values
        .iter()
        .map(|value| {
            let delta = *value - average;
            delta * delta
        })
        .sum::<f64>();
    let sample_variance = sum_squared / (values.len() - 1) as f64;
    Some((sample_variance / values.len() as f64).sqrt())
}

fn digest_proposal(proposal: &TransferProposal) -> String {
    let mut hasher = Sha256::new();
    hasher.update(TASK_TRANSFER_PROPOSAL_DOMAIN);
    update_string(&mut hasher, &proposal.model_sha256);
    update_string(&mut hasher, &proposal.target_task_id);
    hasher.update(proposal.predicted_target_regret_percent.to_bits().to_be_bytes());
    hasher.update(
        proposal
            .predictive_stddev_regret_percent
            .to_bits()
            .to_be_bytes(),
    );
    hasher.update(proposal.requested_prior_weight.to_bits().to_be_bytes());
    let mut sources: Vec<&SourceTaskWeight> = proposal.source_tasks.iter().collect();
    sources.sort_by(|left, right| left.task_id.cmp(&right.task_id));
    hasher.update((sources.len() as u64).to_be_bytes());
    for source in sources {
        update_string(&mut hasher, &source.task_id);
        hasher.update(source.relevance.to_bits().to_be_bytes());
        hasher.update(source.prior_weight.to_bits().to_be_bytes());
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_calibration_evidence(
    evidence: &TransferCalibrationEvidence,
    calibration_task_ids: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(TASK_TRANSFER_EVIDENCE_DOMAIN);
    update_string(&mut hasher, &evidence.model_sha256);
    hasher.update((calibration_task_ids.len() as u64).to_be_bytes());
    for task_id in calibration_task_ids {
        update_string(&mut hasher, task_id);
    }
    let mut outcomes: Vec<&PairedTransferOutcome> = evidence.outcomes.iter().collect();
    outcomes.sort_by(|left, right| {
        left.task_id
            .cmp(&right.task_id)
            .then_with(|| left.seed.cmp(&right.seed))
    });
    hasher.update((outcomes.len() as u64).to_be_bytes());
    for outcome in outcomes {
        update_string(&mut hasher, &outcome.task_id);
        hasher.update(outcome.seed.to_be_bytes());
        hasher.update(outcome.target_only_regret_percent.to_bits().to_be_bytes());
        hasher.update(outcome.transferred_regret_percent.to_bits().to_be_bytes());
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_receipt(receipt: &TransferQualificationReceipt) -> String {
    let mut hasher = Sha256::new();
    hasher.update(TASK_TRANSFER_RECEIPT_DOMAIN);
    update_string(&mut hasher, &receipt.target_task_id);
    update_string(&mut hasher, &receipt.target_artifact_sha256);
    update_string(&mut hasher, &receipt.model_sha256);
    update_string(&mut hasher, &receipt.proposal_sha256);
    update_string(&mut hasher, &receipt.calibration_evidence_sha256);
    hasher.update(receipt.source_prior_weight_requested.to_bits().to_be_bytes());
    hasher.update(receipt.source_prior_weight_applied.to_bits().to_be_bytes());
    hasher.update(receipt.predicted_target_regret_percent.to_bits().to_be_bytes());
    hasher.update(receipt.predictive_stddev_regret_percent.to_bits().to_be_bytes());
    update_optional_f64(&mut hasher, receipt.mean_task_regret_delta_pp);
    update_optional_f64(&mut hasher, receipt.pooled_lower_confidence_bound_pp);
    update_optional_f64(&mut hasher, receipt.pooled_upper_confidence_bound_pp);
    hasher.update(receipt.maximum_task_regression_pp.to_bits().to_be_bytes());
    hasher.update([decision_byte(receipt.decision)]);
    update_string(&mut hasher, &receipt.decision_reason);
    hasher.update((receipt.source_task_ids.len() as u64).to_be_bytes());
    for source_id in &receipt.source_task_ids {
        update_string(&mut hasher, source_id);
    }
    hasher.update((receipt.calibration_task_ids.len() as u64).to_be_bytes());
    for task_id in &receipt.calibration_task_ids {
        update_string(&mut hasher, task_id);
    }
    hasher.update((receipt.task_statistics.len() as u64).to_be_bytes());
    for stats in &receipt.task_statistics {
        update_string(&mut hasher, &stats.task_id);
        hasher.update((stats.paired_seed_count as u64).to_be_bytes());
        update_optional_f64(&mut hasher, stats.mean_regret_delta_pp);
        update_optional_f64(&mut hasher, stats.standard_error_pp);
        update_optional_f64(&mut hasher, stats.lower_confidence_bound_pp);
        update_optional_f64(&mut hasher, stats.upper_confidence_bound_pp);
        hasher.update([stats.passed_regression_guard as u8]);
    }
    hex_digest(hasher.finalize().as_slice())
}

fn update_optional_f64(hasher: &mut Sha256, value: Option<f64>) {
    match value {
        Some(number) => {
            hasher.update([1]);
            hasher.update(number.to_bits().to_be_bytes());
        }
        None => hasher.update([0]),
    }
}

fn decision_byte(decision: TransferGateDecision) -> u8 {
    match decision {
        TransferGateDecision::ApplyPrior => 0,
        TransferGateDecision::AbstainInsufficientCalibration => 1,
        TransferGateDecision::AbstainUncertainBenefit => 2,
        TransferGateDecision::AbstainNoDemonstratedBenefit => 3,
        TransferGateDecision::RejectNegativeTransfer => 4,
    }
}

fn validate_sha256(value: &str, field: &str) -> Result<(), TaskTransferGateError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(TaskTransferGateError::new(
            field,
            "must be exactly 64 hexadecimal characters",
        ));
    }
    Ok(())
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes.iter().copied() {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(
        task_id: &str,
        role: TaskRole,
        search_space_id: &str,
        dataset_id: &str,
        hash_byte: char,
    ) -> TaskManifest {
        let digest = hash_byte.to_string().repeat(64);
        TaskManifest {
            task_id: task_id.into(),
            role,
            search_space_id: search_space_id.into(),
            dataset_id: dataset_id.into(),
            objective_semantics_id: "validation_error".into(),
            metric_direction: MetricDirection::Minimize,
            artifact_sha256: digest.clone(),
            split_sha256: format!("{}{}", hash_byte, "1".repeat(63)),
            fidelity_ladder_sha256: format!("{}{}", hash_byte, "2".repeat(63)),
            resource_profile_sha256: format!("{}{}", hash_byte, "3".repeat(63)),
        }
    }

    fn task_registry() -> Vec<TaskManifest> {
        vec![
            task("source-nb201-c10", TaskRole::HistoricalSource, "nasbench201", "cifar10", 'a'),
            task("source-nb201-c100", TaskRole::HistoricalSource, "nasbench201", "cifar100", 'b'),
            task("calibration-nb101", TaskRole::TransferCalibration, "nasbench101", "cifar10", 'c'),
            task("calibration-nats", TaskRole::TransferCalibration, "natsbench", "cifar100", 'd'),
            task("calibration-fcnet", TaskRole::TransferCalibration, "fcnet", "protein", 'e'),
            task("target-nb201-imagenet16", TaskRole::EvaluationTarget, "nasbench201", "imagenet16-120", 'f'),
        ]
    }

    fn config() -> TransferGateConfig {
        TransferGateConfig {
            minimum_calibration_tasks: 3,
            minimum_paired_seeds_per_task: 30,
            maximum_task_regression_pp: 5.0,
            confidence_z: 1.96,
        }
    }

    fn proposal() -> TransferProposal {
        TransferProposal {
            model_sha256: "9".repeat(64),
            target_task_id: "target-nb201-imagenet16".into(),
            source_tasks: vec![
                SourceTaskWeight {
                    task_id: "source-nb201-c10".into(),
                    relevance: 0.8,
                    prior_weight: 0.6,
                },
                SourceTaskWeight {
                    task_id: "source-nb201-c100".into(),
                    relevance: 0.7,
                    prior_weight: 0.4,
                },
            ],
            predicted_target_regret_percent: 7.0,
            predictive_stddev_regret_percent: 1.5,
            requested_prior_weight: 0.5,
        }
    }

    fn evidence(delta_by_task: [f64; 3]) -> TransferCalibrationEvidence {
        let task_ids = [
            "calibration-nb101",
            "calibration-nats",
            "calibration-fcnet",
        ];
        let mut outcomes = Vec::new();
        for (task_index, task_id) in task_ids.iter().enumerate() {
            for seed in 0..30 {
                outcomes.push(PairedTransferOutcome {
                    task_id: (*task_id).into(),
                    seed,
                    target_only_regret_percent: 10.0,
                    transferred_regret_percent: 10.0 + delta_by_task[task_index],
                });
            }
        }
        TransferCalibrationEvidence {
            model_sha256: "9".repeat(64),
            calibration_task_ids: task_ids.iter().map(|value| (*value).into()).collect(),
            outcomes,
        }
    }

    fn gate() -> CrossTaskTransferGate {
        CrossTaskTransferGate::new(config(), task_registry()).unwrap()
    }

    #[test]
    fn applies_transfer_only_when_each_calibration_task_passes_and_pooled_benefit_is_clear() {
        let receipt = gate().qualify(&proposal(), &evidence([-2.0, -1.5, -1.0])).unwrap();
        assert_eq!(receipt.decision, TransferGateDecision::ApplyPrior);
        assert_eq!(receipt.source_prior_weight_applied, 0.5);
        assert!(receipt.pooled_upper_confidence_bound_pp.unwrap() < 0.0);
        assert!(receipt.verify_digest());
    }

    #[test]
    fn rejects_transfer_when_one_calibration_task_shows_material_harm() {
        let receipt = gate().qualify(&proposal(), &evidence([-2.0, -1.0, 8.0])).unwrap();
        assert_eq!(receipt.decision, TransferGateDecision::RejectNegativeTransfer);
        assert_eq!(receipt.source_prior_weight_applied, 0.0);
        assert!(receipt.task_statistics[2].lower_confidence_bound_pp.unwrap() > 5.0);
    }

    #[test]
    fn abstains_when_calibration_tasks_are_too_few() {
        let mut e = evidence([-2.0, -2.0, -2.0]);
        e.calibration_task_ids.pop();
        e.outcomes.retain(|outcome| outcome.task_id != "calibration-fcnet");
        let receipt = gate().qualify(&proposal(), &e).unwrap();
        assert_eq!(receipt.decision, TransferGateDecision::AbstainInsufficientCalibration);
        assert_eq!(receipt.source_prior_weight_applied, 0.0);
    }

    #[test]
    fn abstains_when_paired_seed_count_is_below_minimum() {
        let mut e = evidence([-2.0, -2.0, -2.0]);
        e.outcomes.retain(|outcome| {
            outcome.task_id != "calibration-fcnet" || outcome.seed < 10
        });
        let receipt = gate().qualify(&proposal(), &e).unwrap();
        assert_eq!(receipt.decision, TransferGateDecision::AbstainInsufficientCalibration);
        assert_eq!(receipt.task_statistics[0].paired_seed_count, 10);
    }

    #[test]
    fn abstains_when_transfer_is_not_demonstrably_better() {
        let receipt = gate().qualify(&proposal(), &evidence([0.0, 0.0, 0.0])).unwrap();
        assert_eq!(receipt.decision, TransferGateDecision::AbstainNoDemonstratedBenefit);
        assert_eq!(receipt.source_prior_weight_applied, 0.0);
    }

    #[test]
    fn abstains_when_regression_confidence_interval_overlaps_tolerance() {
        let mut e = evidence([-2.0, -2.0, 5.0]);
        for outcome in e
            .outcomes
            .iter_mut()
            .filter(|outcome| outcome.task_id == "calibration-fcnet")
        {
            let offset = if outcome.seed % 2 == 0 { -2.5 } else { 2.5 };
            outcome.transferred_regret_percent = 15.0 + offset;
        }
        let receipt = gate().qualify(&proposal(), &e).unwrap();
        assert_eq!(receipt.decision, TransferGateDecision::AbstainUncertainBenefit);
        assert_eq!(receipt.source_prior_weight_applied, 0.0);
    }

    #[test]
    fn rejects_target_as_historical_source() {
        let mut invalid = proposal();
        invalid.source_tasks[0].task_id = invalid.target_task_id.clone();
        assert!(gate().qualify(&invalid, &evidence([-2.0, -2.0, -2.0])).is_err());
    }

    #[test]
    fn rejects_calibration_role_mismatch() {
        let mut e = evidence([-2.0, -2.0, -2.0]);
        e.calibration_task_ids[0] = "source-nb201-c10".into();
        assert!(gate().qualify(&proposal(), &e).is_err());
    }

    #[test]
    fn rejects_metric_semantics_or_direction_mismatch() {
        let mut tasks = task_registry();
        let index = tasks
            .iter()
            .position(|task| task.task_id == "source-nb201-c10")
            .unwrap();
        tasks[index].metric_direction = MetricDirection::Maximize;
        let gate = CrossTaskTransferGate::new(config(), tasks).unwrap();
        assert!(gate.qualify(&proposal(), &evidence([-2.0, -2.0, -2.0])).is_err());
    }

    #[test]
    fn rejects_unbalanced_prior_weights() {
        let mut invalid = proposal();
        invalid.source_tasks[0].prior_weight = 0.7;
        assert!(gate().qualify(&invalid, &evidence([-2.0, -2.0, -2.0])).is_err());
    }

    #[test]
    fn rejects_source_or_calibration_split_reuse_with_target_or_source() {
        let mut tasks = task_registry();
        let source_split = tasks
            .iter()
            .find(|task| task.task_id == "source-nb201-c10")
            .unwrap()
            .split_sha256
            .clone();
        let calibration = tasks
            .iter_mut()
            .find(|task| task.task_id == "calibration-nb101")
            .unwrap();
        calibration.split_sha256 = source_split;
        let gate = CrossTaskTransferGate::new(config(), tasks).unwrap();
        assert!(gate
            .qualify(&proposal(), &evidence([-2.0, -2.0, -2.0]))
            .is_err());

        let mut tasks = task_registry();
        let target_split = tasks
            .iter()
            .find(|task| task.task_id == "target-nb201-imagenet16")
            .unwrap()
            .split_sha256
            .clone();
        let source = tasks
            .iter_mut()
            .find(|task| task.task_id == "source-nb201-c10")
            .unwrap();
        source.split_sha256 = target_split;
        let gate = CrossTaskTransferGate::new(config(), tasks).unwrap();
        assert!(gate
            .qualify(&proposal(), &evidence([-2.0, -2.0, -2.0]))
            .is_err());
    }

    #[test]
    fn rejects_source_and_calibration_overlap() {
        let mut e = evidence([-2.0, -2.0, -2.0]);
        e.calibration_task_ids[0] = "source-nb201-c10".into();
        assert!(gate().qualify(&proposal(), &e).is_err());
    }

    #[test]
    fn rejects_duplicate_task_seed_pair_and_model_digest_mismatch() {
        let mut e = evidence([-2.0, -2.0, -2.0]);
        e.outcomes.push(e.outcomes[0].clone());
        assert!(gate().qualify(&proposal(), &e).is_err());

        let mut e = evidence([-2.0, -2.0, -2.0]);
        e.model_sha256 = "8".repeat(64);
        assert!(gate().qualify(&proposal(), &e).is_err());
    }

    #[test]
    fn receipt_digest_detects_decision_or_statistic_tampering() {
        let mut receipt = gate().qualify(&proposal(), &evidence([-2.0, -2.0, -2.0])).unwrap();
        assert!(receipt.verify_digest());
        receipt.source_prior_weight_applied = 0.0;
        assert!(!receipt.verify_digest());
    }
}
