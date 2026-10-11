// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-domain reasoning capability matrix.
//!
//! This module is deliberately a measurement/index layer, not a reasoning mechanism and not a
//! global intelligence score. Each lane remains independently auditable and binds an exact
//! subject revision, configuration, benchmark lineage, holdout/contamination policy, resource
//! budget, episodes, evaluator receipts, abstention profile, and optional baseline observations.

use super::reasoning_evaluator::{
    aggregate_receipts, CapabilitySlice, ReasoningEvaluatorError,
};
use super::reasoning_qualification::{
    AbstentionReason, QualificationValidationError, ReasoningDomain, ReasoningEpisode,
    ReasoningOutcome, ReasoningQualificationReceipt, ResourceUsage,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

pub const REASONING_CAPABILITY_MATRIX_SCHEMA_VERSION: u32 = 1;

/// How strongly the task partition is isolated from development/tuning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HoldoutPolicy {
    /// Frozen before the evaluated subject/configuration was selected and not used for tuning.
    FrozenUnseen,
    /// Public evaluation tasks. Useful evidence, but exposure cannot be excluded by design.
    PublicEvaluation,
    /// Development/training probe. Measurement-only, never fresh holdout evidence.
    DevelopmentProbe,
    /// Domain-specific policy whose exact semantics are externally versioned.
    Custom(String),
}

/// Exposure status is explicit and never inferred from benchmark naming alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContaminationStatus {
    /// The declared contamination controls found no known exposure for this subject lineage.
    Controlled,
    /// Exposure/tuning on the evaluated task material is known.
    Exposed,
    /// Exposure cannot currently be established either way.
    Unknown,
}

/// Maximum resources an individual episode in this lane may consume.
/// `None` means that resource dimension is recorded but this lane does not impose a maximum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub wall_time_us: Option<u64>,
    pub deliberation_steps: Option<u64>,
    pub tool_calls: Option<u64>,
    pub model_tokens: Option<u64>,
}

/// Immutable identity and policy of one capability lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityLaneDescriptor {
    pub lane_id: String,
    pub domain: ReasoningDomain,
    pub benchmark: String,
    pub benchmark_version: String,
    pub split: String,
    pub holdout_policy: HoldoutPolicy,
    /// Content/version identity of the policy or exposure ledger used for contamination review.
    pub contamination_policy_id: String,
    pub contamination_status: ContaminationStatus,
    pub resource_budget: ResourceBudget,
}

/// One externally sourced comparison point. Baselines remain metric-specific and are never
/// combined into a universal human/AI score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineMetric {
    pub baseline_id: String,
    pub metric_name: String,
    pub value: f64,
    pub unit: String,
    pub evidence_ref: String,
}

/// Observed resource envelope over all episodes in one lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservedResources {
    pub total: ResourceUsage,
    pub maximum: ResourceUsage,
    pub mean_wall_time_us: f64,
    pub mean_deliberation_steps: f64,
    pub mean_tool_calls: f64,
    pub mean_model_tokens: f64,
}

/// Fully bound report for one domain/benchmark lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityLaneReport {
    pub descriptor: CapabilityLaneDescriptor,
    pub subject_revision: String,
    pub configuration_id: String,
    pub capability: CapabilitySlice,
    pub resources: ObservedResources,
    pub abstentions: BTreeMap<String, usize>,
    pub baselines: Vec<BaselineMetric>,
    pub episode_ids: Vec<String>,
    pub receipt_ids: Vec<String>,
}

/// A capability matrix is a vector of independently interpretable lanes. There is intentionally
/// no `overall_score`, `agi_score`, or other scalar that can hide domain failures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningCapabilityMatrix {
    pub schema_version: u32,
    pub subject_revision: String,
    pub lanes: Vec<CapabilityLaneReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CapabilityMatrixError {
    EmptyField(String),
    EmptyLane,
    EmptyMatrix,
    EpisodeReceiptCountMismatch { episodes: usize, receipts: usize },
    SubjectRevisionMismatch { expected: String, found: String },
    ConfigurationMismatch { expected: String, found: String },
    DomainMismatch,
    BenchmarkMismatch { field: &'static str, expected: String, found: String },
    ReceiptEpisodeMismatch(String),
    DuplicateEpisode(String),
    DuplicateLane(String),
    DuplicateBaseline { baseline_id: String, metric_name: String },
    NonFiniteBaseline { baseline_id: String, metric_name: String },
    ResourceBudgetExceeded {
        problem_id: String,
        resource: &'static str,
        observed: u64,
        limit: u64,
    },
    Evaluator(ReasoningEvaluatorError),
    Qualification(QualificationValidationError),
}

impl fmt::Display for CapabilityMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::EmptyLane => write!(f, "capability lane must contain at least one episode"),
            Self::EmptyMatrix => write!(f, "capability matrix must contain at least one lane"),
            Self::EpisodeReceiptCountMismatch { episodes, receipts } => write!(
                f,
                "capability lane has {episodes} episodes but {receipts} receipts"
            ),
            Self::SubjectRevisionMismatch { expected, found } => write!(
                f,
                "subject revision mismatch: expected `{expected}`, found `{found}`"
            ),
            Self::ConfigurationMismatch { expected, found } => write!(
                f,
                "configuration mismatch: expected `{expected}`, found `{found}`"
            ),
            Self::DomainMismatch => write!(f, "episode domain does not match lane domain"),
            Self::BenchmarkMismatch { field, expected, found } => write!(
                f,
                "benchmark {field} mismatch: expected `{expected}`, found `{found}`"
            ),
            Self::ReceiptEpisodeMismatch(id) => {
                write!(f, "receipt references episode `{id}` outside the lane")
            }
            Self::DuplicateEpisode(id) => write!(f, "episode `{id}` appears twice in the lane"),
            Self::DuplicateLane(id) => write!(f, "capability lane `{id}` appears twice"),
            Self::DuplicateBaseline { baseline_id, metric_name } => write!(
                f,
                "baseline `{baseline_id}` repeats metric `{metric_name}`"
            ),
            Self::NonFiniteBaseline { baseline_id, metric_name } => write!(
                f,
                "baseline `{baseline_id}` metric `{metric_name}` is not finite"
            ),
            Self::ResourceBudgetExceeded { problem_id, resource, observed, limit } => write!(
                f,
                "episode `{problem_id}` exceeds {resource} budget: {observed} > {limit}"
            ),
            Self::Evaluator(err) => write!(f, "reasoning evaluator rejected lane: {err}"),
            Self::Qualification(err) => write!(f, "reasoning evidence rejected lane: {err}"),
        }
    }
}

impl std::error::Error for CapabilityMatrixError {}

impl From<ReasoningEvaluatorError> for CapabilityMatrixError {
    fn from(value: ReasoningEvaluatorError) -> Self {
        Self::Evaluator(value)
    }
}

impl From<QualificationValidationError> for CapabilityMatrixError {
    fn from(value: QualificationValidationError) -> Self {
        Self::Qualification(value)
    }
}

impl CapabilityLaneDescriptor {
    pub fn validate(&self) -> Result<(), CapabilityMatrixError> {
        require_nonempty("lane_id", &self.lane_id)?;
        require_nonempty("benchmark", &self.benchmark)?;
        require_nonempty("benchmark_version", &self.benchmark_version)?;
        require_nonempty("split", &self.split)?;
        require_nonempty("contamination_policy_id", &self.contamination_policy_id)?;
        if let HoldoutPolicy::Custom(name) = &self.holdout_policy {
            require_nonempty("holdout_policy.custom", name)?;
        }
        Ok(())
    }
}

impl BaselineMetric {
    pub fn validate(&self) -> Result<(), CapabilityMatrixError> {
        require_nonempty("baseline_id", &self.baseline_id)?;
        require_nonempty("baseline.metric_name", &self.metric_name)?;
        require_nonempty("baseline.unit", &self.unit)?;
        require_nonempty("baseline.evidence_ref", &self.evidence_ref)?;
        if !self.value.is_finite() {
            return Err(CapabilityMatrixError::NonFiniteBaseline {
                baseline_id: self.baseline_id.clone(),
                metric_name: self.metric_name.clone(),
            });
        }
        Ok(())
    }
}

/// Build one homogeneous capability lane from exact episodes and their evaluator receipts.
///
/// The function rejects subject/configuration/domain/benchmark drift, budget violations,
/// receipt/episode mismatches, duplicate episodes, malformed receipts, and duplicate baselines.
pub fn build_capability_lane(
    descriptor: CapabilityLaneDescriptor,
    episodes: &[ReasoningEpisode],
    receipts: &[ReasoningQualificationReceipt],
    baselines: Vec<BaselineMetric>,
) -> Result<CapabilityLaneReport, CapabilityMatrixError> {
    descriptor.validate()?;
    if episodes.is_empty() {
        return Err(CapabilityMatrixError::EmptyLane);
    }
    if episodes.len() != receipts.len() {
        return Err(CapabilityMatrixError::EpisodeReceiptCountMismatch {
            episodes: episodes.len(),
            receipts: receipts.len(),
        });
    }

    validate_baselines(&baselines)?;

    let subject_revision = episodes[0].subject_revision.clone();
    let configuration_id = episodes[0].configuration_id.clone();
    require_nonempty("subject_revision", &subject_revision)?;
    require_nonempty("configuration_id", &configuration_id)?;

    let mut episode_by_id: HashMap<String, &ReasoningEpisode> =
        HashMap::with_capacity(episodes.len());
    let mut episode_ids = Vec::with_capacity(episodes.len());
    let mut total = ResourceUsage::default();
    let mut maximum = ResourceUsage::default();
    let mut abstentions = BTreeMap::new();

    for episode in episodes {
        episode.validate()?;
        if episode.subject_revision != subject_revision {
            return Err(CapabilityMatrixError::SubjectRevisionMismatch {
                expected: subject_revision.clone(),
                found: episode.subject_revision.clone(),
            });
        }
        if episode.configuration_id != configuration_id {
            return Err(CapabilityMatrixError::ConfigurationMismatch {
                expected: configuration_id.clone(),
                found: episode.configuration_id.clone(),
            });
        }
        if episode.domain != descriptor.domain {
            return Err(CapabilityMatrixError::DomainMismatch);
        }
        require_benchmark_match("name", &descriptor.benchmark, &episode.problem.benchmark)?;
        require_benchmark_match(
            "version",
            &descriptor.benchmark_version,
            &episode.problem.benchmark_version,
        )?;
        require_benchmark_match("split", &descriptor.split, &episode.problem.split)?;
        enforce_budget(&descriptor.resource_budget, episode)?;

        let id = episode.id()?.0;
        if episode_by_id.insert(id.clone(), episode).is_some() {
            return Err(CapabilityMatrixError::DuplicateEpisode(id));
        }
        episode_ids.push(id);
        accumulate_resources(&mut total, &mut maximum, episode.resources);
        if let ReasoningOutcome::Abstained { reason, .. } = &episode.outcome {
            *abstentions.entry(abstention_label(reason)).or_insert(0) += 1;
        }
    }

    let mut receipt_ids = Vec::with_capacity(receipts.len());
    let mut seen_receipt_episodes = HashSet::with_capacity(receipts.len());
    for receipt in receipts {
        receipt.validate()?;
        let id = receipt.episode_id.0.clone();
        if !episode_by_id.contains_key(&id) {
            return Err(CapabilityMatrixError::ReceiptEpisodeMismatch(id));
        }
        if !seen_receipt_episodes.insert(id.clone()) {
            return Err(CapabilityMatrixError::DuplicateEpisode(id));
        }
        receipt_ids.push(receipt.receipt_id.clone());
    }

    // Count equality plus uniqueness plus membership proves every lane episode has exactly one
    // receipt. The evaluator adds an independent receipt-integrity/duplicate check.
    let capability = aggregate_receipts(receipts)?;
    let count = episodes.len() as f64;
    let resources = ObservedResources {
        total,
        maximum,
        mean_wall_time_us: total.wall_time_us as f64 / count,
        mean_deliberation_steps: total.deliberation_steps as f64 / count,
        mean_tool_calls: total.tool_calls as f64 / count,
        mean_model_tokens: total.model_tokens as f64 / count,
    };

    Ok(CapabilityLaneReport {
        descriptor,
        subject_revision,
        configuration_id,
        capability,
        resources,
        abstentions,
        baselines,
        episode_ids,
        receipt_ids,
    })
}

/// Build a matrix whose lanes all qualify the same code revision.
pub fn build_capability_matrix(
    subject_revision: impl Into<String>,
    lanes: Vec<CapabilityLaneReport>,
) -> Result<ReasoningCapabilityMatrix, CapabilityMatrixError> {
    let subject_revision = subject_revision.into();
    require_nonempty("matrix.subject_revision", &subject_revision)?;
    if lanes.is_empty() {
        return Err(CapabilityMatrixError::EmptyMatrix);
    }

    let mut lane_ids = HashSet::with_capacity(lanes.len());
    for lane in &lanes {
        if lane.subject_revision != subject_revision {
            return Err(CapabilityMatrixError::SubjectRevisionMismatch {
                expected: subject_revision.clone(),
                found: lane.subject_revision.clone(),
            });
        }
        if !lane_ids.insert(lane.descriptor.lane_id.as_str()) {
            return Err(CapabilityMatrixError::DuplicateLane(
                lane.descriptor.lane_id.clone(),
            ));
        }
    }

    Ok(ReasoningCapabilityMatrix {
        schema_version: REASONING_CAPABILITY_MATRIX_SCHEMA_VERSION,
        subject_revision,
        lanes,
    })
}

fn validate_baselines(baselines: &[BaselineMetric]) -> Result<(), CapabilityMatrixError> {
    let mut seen = HashSet::with_capacity(baselines.len());
    for baseline in baselines {
        baseline.validate()?;
        let key = (baseline.baseline_id.as_str(), baseline.metric_name.as_str());
        if !seen.insert(key) {
            return Err(CapabilityMatrixError::DuplicateBaseline {
                baseline_id: baseline.baseline_id.clone(),
                metric_name: baseline.metric_name.clone(),
            });
        }
    }
    Ok(())
}

fn enforce_budget(
    budget: &ResourceBudget,
    episode: &ReasoningEpisode,
) -> Result<(), CapabilityMatrixError> {
    check_limit(
        budget.wall_time_us,
        episode.resources.wall_time_us,
        "wall_time_us",
        &episode.problem.problem_id,
    )?;
    check_limit(
        budget.deliberation_steps,
        episode.resources.deliberation_steps,
        "deliberation_steps",
        &episode.problem.problem_id,
    )?;
    check_limit(
        budget.tool_calls,
        episode.resources.tool_calls,
        "tool_calls",
        &episode.problem.problem_id,
    )?;
    check_limit(
        budget.model_tokens,
        episode.resources.model_tokens,
        "model_tokens",
        &episode.problem.problem_id,
    )
}

fn check_limit(
    limit: Option<u64>,
    observed: u64,
    resource: &'static str,
    problem_id: &str,
) -> Result<(), CapabilityMatrixError> {
    if let Some(limit) = limit {
        if observed > limit {
            return Err(CapabilityMatrixError::ResourceBudgetExceeded {
                problem_id: problem_id.into(),
                resource,
                observed,
                limit,
            });
        }
    }
    Ok(())
}

fn accumulate_resources(total: &mut ResourceUsage, maximum: &mut ResourceUsage, value: ResourceUsage) {
    total.wall_time_us = total.wall_time_us.saturating_add(value.wall_time_us);
    total.deliberation_steps = total
        .deliberation_steps
        .saturating_add(value.deliberation_steps);
    total.tool_calls = total.tool_calls.saturating_add(value.tool_calls);
    total.model_tokens = total.model_tokens.saturating_add(value.model_tokens);

    maximum.wall_time_us = maximum.wall_time_us.max(value.wall_time_us);
    maximum.deliberation_steps = maximum.deliberation_steps.max(value.deliberation_steps);
    maximum.tool_calls = maximum.tool_calls.max(value.tool_calls);
    maximum.model_tokens = maximum.model_tokens.max(value.model_tokens);
}

fn require_benchmark_match(
    field: &'static str,
    expected: &str,
    found: &str,
) -> Result<(), CapabilityMatrixError> {
    if expected == found {
        Ok(())
    } else {
        Err(CapabilityMatrixError::BenchmarkMismatch {
            field,
            expected: expected.into(),
            found: found.into(),
        })
    }
}

fn require_nonempty(field: &str, value: &str) -> Result<(), CapabilityMatrixError> {
    if value.trim().is_empty() {
        Err(CapabilityMatrixError::EmptyField(field.into()))
    } else {
        Ok(())
    }
}

fn abstention_label(reason: &AbstentionReason) -> String {
    match reason {
        AbstentionReason::InsufficientEvidence => "insufficient_evidence".into(),
        AbstentionReason::Unidentified => "unidentified".into(),
        AbstentionReason::OutOfDistribution => "out_of_distribution".into(),
        AbstentionReason::ResourceBudgetExceeded => "resource_budget_exceeded".into(),
        AbstentionReason::ConflictingEvidence => "conflicting_evidence".into(),
        AbstentionReason::UnsafeToConclude => "unsafe_to_conclude".into(),
        AbstentionReason::Other(value) => format!("other:{value}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligence::reasoning_evaluator::{evaluate_episode, EpisodeJudgment};
    use crate::intelligence::reasoning_qualification::{ReasoningProblemRef, ReasoningOutcome};

    fn descriptor() -> CapabilityLaneDescriptor {
        CapabilityLaneDescriptor {
            lane_id: "logic-exact-v1".into(),
            domain: ReasoningDomain::Logic,
            benchmark: "unit-logic".into(),
            benchmark_version: "v1".into(),
            split: "holdout".into(),
            holdout_policy: HoldoutPolicy::FrozenUnseen,
            contamination_policy_id: "exposure-ledger:v1".into(),
            contamination_status: ContaminationStatus::Controlled,
            resource_budget: ResourceBudget {
                wall_time_us: Some(10_000),
                deliberation_steps: Some(100),
                tool_calls: Some(2),
                model_tokens: Some(1_000),
            },
        }
    }

    fn episode(problem: &str, outcome: ReasoningOutcome, steps: u64) -> ReasoningEpisode {
        match ReasoningEpisode::new(
            "subject-a",
            "config-a",
            ReasoningDomain::Logic,
            ReasoningProblemRef {
                benchmark: "unit-logic".into(),
                benchmark_version: "v1".into(),
                split: "holdout".into(),
                problem_id: problem.into(),
                problem_hash: format!("blake3:{problem}"),
            },
            vec![],
            vec![],
            vec![],
            outcome,
            ResourceUsage {
                wall_time_us: 100,
                deliberation_steps: steps,
                tool_calls: 0,
                model_tokens: 0,
            },
        ) {
            Ok(value) => value,
            Err(err) => panic!("test episode must validate: {err}"),
        }
    }

    fn receipt(episode: &ReasoningEpisode, correct: bool) -> ReasoningQualificationReceipt {
        match evaluate_episode(
            episode,
            &format!("eval:{}", episode.problem.problem_id),
            &EpisodeJudgment {
                exact_correct: Some(correct),
                task_score: None,
            },
        ) {
            Ok(value) => value,
            Err(err) => panic!("test receipt must validate: {err}"),
        }
    }

    #[test]
    fn valid_lane_preserves_accuracy_coverage_abstention_and_resources() {
        let answered = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "true".into(),
                confidence: 0.8,
            },
            4,
        );
        let abstained = episode(
            "p2",
            ReasoningOutcome::Abstained {
                reason: AbstentionReason::Unidentified,
                answerability: 0.1,
            },
            6,
        );
        let episodes = vec![answered, abstained];
        let receipts = vec![receipt(&episodes[0], true), receipt(&episodes[1], true)];

        let report = build_capability_lane(descriptor(), &episodes, &receipts, vec![]);
        let report = match report {
            Ok(value) => value,
            Err(err) => panic!("lane should qualify: {err}"),
        };
        assert_eq!(report.capability.episodes, 2);
        assert_eq!(report.capability.coverage, 0.5);
        assert_eq!(report.capability.exact_accuracy, Some(0.5));
        assert_eq!(report.capability.selective_accuracy, Some(1.0));
        assert_eq!(report.abstentions.get("unidentified"), Some(&1));
        assert_eq!(report.resources.total.deliberation_steps, 10);
        assert_eq!(report.resources.maximum.deliberation_steps, 6);
    }

    #[test]
    fn rejects_subject_drift_inside_one_lane() {
        let first = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "a".into(),
                confidence: 0.5,
            },
            1,
        );
        let mut second = episode(
            "p2",
            ReasoningOutcome::Asserted {
                value: "b".into(),
                confidence: 0.5,
            },
            1,
        );
        second.subject_revision = "subject-b".into();
        let episodes = vec![first, second];
        let receipts = vec![receipt(&episodes[0], true), receipt(&episodes[1], true)];
        assert!(matches!(
            build_capability_lane(descriptor(), &episodes, &receipts, vec![]),
            Err(CapabilityMatrixError::SubjectRevisionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_receipt_for_episode_outside_lane() {
        let inside = episode(
            "inside",
            ReasoningOutcome::Asserted {
                value: "a".into(),
                confidence: 0.5,
            },
            1,
        );
        let outside = episode(
            "outside",
            ReasoningOutcome::Asserted {
                value: "b".into(),
                confidence: 0.5,
            },
            1,
        );
        let receipts = vec![receipt(&outside, true)];
        assert!(matches!(
            build_capability_lane(descriptor(), &[inside], &receipts, vec![]),
            Err(CapabilityMatrixError::ReceiptEpisodeMismatch(_))
        ));
    }

    #[test]
    fn rejects_episode_that_exceeds_declared_compute_budget() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "a".into(),
                confidence: 0.5,
            },
            101,
        );
        let receipts = vec![receipt(&subject, true)];
        assert!(matches!(
            build_capability_lane(descriptor(), &[subject], &receipts, vec![]),
            Err(CapabilityMatrixError::ResourceBudgetExceeded {
                resource: "deliberation_steps",
                ..
            })
        ));
    }

    #[test]
    fn rejects_non_finite_or_duplicate_baseline_metrics() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "a".into(),
                confidence: 0.5,
            },
            1,
        );
        let receipts = vec![receipt(&subject, true)];
        let bad = BaselineMetric {
            baseline_id: "human-expert".into(),
            metric_name: "exact_accuracy".into(),
            value: f64::NAN,
            unit: "fraction".into(),
            evidence_ref: "paper:v1".into(),
        };
        assert!(matches!(
            build_capability_lane(descriptor(), &[subject], &receipts, vec![bad]),
            Err(CapabilityMatrixError::NonFiniteBaseline { .. })
        ));
    }

    #[test]
    fn matrix_rejects_duplicate_lane_identity_and_subject_drift() {
        let subject = episode(
            "p1",
            ReasoningOutcome::Asserted {
                value: "a".into(),
                confidence: 0.5,
            },
            1,
        );
        let receipts = vec![receipt(&subject, true)];
        let lane = match build_capability_lane(descriptor(), &[subject], &receipts, vec![]) {
            Ok(value) => value,
            Err(err) => panic!("lane should qualify: {err}"),
        };

        assert!(matches!(
            build_capability_matrix("subject-a", vec![lane.clone(), lane]),
            Err(CapabilityMatrixError::DuplicateLane(_))
        ));
    }
}
