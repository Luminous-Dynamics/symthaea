//! Repeatability and replication views over algorithm evaluation receipts.
//!
//! This module deliberately separates three claims:
//!
//! ```text
//! repeatability != replication != independence
//! ```
//!
//! Repeatability means repeated measurements under one exact evaluation context.
//! Replication means the same exact implementation has passed under multiple exact contexts.
//! Metadata diversity does not prove organizational or methodological independence, so
//! `IndependenceStatus` remains `NotEstablished` in this crate.

use crate::evaluation::{
    CorrectnessVerdict, EvaluationError, EvaluationReceipt, ObjectiveDirection,
    ObjectiveMeasurement,
};
use crate::{ContentId, ImplementationId, ProblemId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ReplicationError {
    #[error("at least two receipts are required for repeatability")]
    TooFewRepeatabilityReceipts,
    #[error("evaluation receipt {index} is invalid: {source}")]
    InvalidReceipt {
        index: usize,
        source: EvaluationError,
    },
    #[error("all receipts must refer to the same problem")]
    ProblemMismatch,
    #[error("all receipts must refer to the same implementation")]
    ImplementationMismatch,
    #[error("repeatability requires one exact evaluation context")]
    ContextMismatch,
    #[error("repeatability requires one exact objective schema")]
    ObjectiveSchemaMismatch,
    #[error("repeatability/replication requires correctness-passed receipts")]
    CorrectnessNotPassed,
    #[error("a receipt lacks an explicit evidence/measurement run identity")]
    MissingRunIdentity,
    #[error("duplicate evidence/measurement run identity: {0}")]
    DuplicateRunIdentity(String),
    #[error("duplicate evaluation receipt identity")]
    DuplicateReceipt,
    #[error("replication policy thresholds must be positive")]
    InvalidPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IndependenceStatus {
    NotEstablished,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveSampleSummary {
    pub name: String,
    pub direction: ObjectiveDirection,
    pub unit: String,
    pub samples: Vec<f64>,
    pub minimum: f64,
    pub median: f64,
    pub maximum: f64,
}

/// Repeated measurements under one exact evaluator/oracle/input/environment/toolchain context.
#[derive(Debug, Clone, PartialEq)]
pub struct RepeatabilityCohort {
    pub problem_id: ProblemId,
    pub implementation_id: ImplementationId,
    pub context_id: ContentId,
    pub receipt_ids: Vec<ContentId>,
    pub summaries: Vec<ObjectiveSampleSummary>,
}

impl RepeatabilityCohort {
    pub fn new(receipts: &[EvaluationReceipt]) -> Result<Self, ReplicationError> {
        if receipts.len() < 2 {
            return Err(ReplicationError::TooFewRepeatabilityReceipts);
        }

        validate_common_receipts(receipts)?;
        let first = &receipts[0];
        let expected_context = first.context.content_id();
        let expected_schema = objective_schema(&first.objectives);

        for receipt in &receipts[1..] {
            if receipt.context.content_id() != expected_context {
                return Err(ReplicationError::ContextMismatch);
            }
            if objective_schema(&receipt.objectives) != expected_schema {
                return Err(ReplicationError::ObjectiveSchemaMismatch);
            }
        }

        let mut receipt_ids: Vec<_> = receipts.iter().map(|receipt| receipt.id.clone()).collect();
        receipt_ids.sort();

        let summaries = expected_schema
            .iter()
            .enumerate()
            .map(|(objective_index, (name, direction, unit))| {
                let mut samples: Vec<f64> = receipts
                    .iter()
                    .map(|receipt| receipt.objectives[objective_index].value)
                    .collect();
                samples.sort_by(f64::total_cmp);
                let minimum = samples[0];
                let maximum = samples[samples.len() - 1];
                let median = if samples.len() % 2 == 1 {
                    samples[samples.len() / 2]
                } else {
                    let upper = samples.len() / 2;
                    samples[upper - 1] / 2.0 + samples[upper] / 2.0
                };
                ObjectiveSampleSummary {
                    name: name.clone(),
                    direction: *direction,
                    unit: unit.clone(),
                    samples,
                    minimum,
                    median,
                    maximum,
                }
            })
            .collect();

        Ok(Self {
            problem_id: first.problem_id.clone(),
            implementation_id: first.implementation_id.clone(),
            context_id: expected_context,
            receipt_ids,
            summaries,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationPolicy {
    pub min_receipts: usize,
    pub min_distinct_contexts: usize,
    pub min_distinct_environments: usize,
    pub min_distinct_evaluators: usize,
    pub min_distinct_input_profiles: usize,
}

impl Default for ReplicationPolicy {
    fn default() -> Self {
        Self {
            min_receipts: 3,
            min_distinct_contexts: 2,
            min_distinct_environments: 2,
            min_distinct_evaluators: 1,
            min_distinct_input_profiles: 1,
        }
    }
}

impl ReplicationPolicy {
    pub fn validate(&self) -> Result<(), ReplicationError> {
        let thresholds = [
            self.min_receipts,
            self.min_distinct_contexts,
            self.min_distinct_environments,
            self.min_distinct_evaluators,
            self.min_distinct_input_profiles,
        ];
        if thresholds.iter().any(|&value| value == 0)
            || self.min_distinct_contexts > self.min_receipts
            || self.min_distinct_environments > self.min_receipts
            || self.min_distinct_evaluators > self.min_receipts
            || self.min_distinct_input_profiles > self.min_receipts
        {
            return Err(ReplicationError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationCoverageAssessment {
    pub id: ContentId,
    pub problem_id: ProblemId,
    pub implementation_id: ImplementationId,
    pub receipt_ids: Vec<ContentId>,
    pub distinct_contexts: usize,
    pub distinct_environments: usize,
    pub distinct_evaluators: usize,
    pub distinct_input_profiles: usize,
    pub distinct_toolchain_profiles: usize,
    pub distinct_target_profiles: usize,
    pub policy: ReplicationPolicy,
    pub policy_satisfied: bool,
    pub independence: IndependenceStatus,
}

impl ReplicationCoverageAssessment {
    pub fn assess(
        receipts: &[EvaluationReceipt],
        policy: ReplicationPolicy,
    ) -> Result<Self, ReplicationError> {
        policy.validate()?;
        validate_common_receipts(receipts)?;

        let first = receipts
            .first()
            .ok_or(ReplicationError::TooFewRepeatabilityReceipts)?;
        let mut receipt_ids: Vec<_> = receipts.iter().map(|receipt| receipt.id.clone()).collect();
        receipt_ids.sort();

        let distinct_contexts = receipts
            .iter()
            .map(|receipt| receipt.context.content_id())
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_environments = receipts
            .iter()
            .map(|receipt| receipt.context.environment_id.clone())
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_evaluators = receipts
            .iter()
            .map(|receipt| receipt.context.evaluator_id.clone())
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_input_profiles = receipts
            .iter()
            .map(|receipt| receipt.context.input_profile_id.clone())
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_toolchain_profiles = receipts
            .iter()
            .map(|receipt| receipt.context.toolchain_profile.clone())
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_target_profiles = receipts
            .iter()
            .map(|receipt| receipt.context.target_profile.clone())
            .collect::<BTreeSet<_>>()
            .len();

        let policy_satisfied = receipts.len() >= policy.min_receipts
            && distinct_contexts >= policy.min_distinct_contexts
            && distinct_environments >= policy.min_distinct_environments
            && distinct_evaluators >= policy.min_distinct_evaluators
            && distinct_input_profiles >= policy.min_distinct_input_profiles;

        let id = derive_assessment_id(
            &first.problem_id,
            &first.implementation_id,
            &receipt_ids,
            policy,
            policy_satisfied,
        );

        Ok(Self {
            id,
            problem_id: first.problem_id.clone(),
            implementation_id: first.implementation_id.clone(),
            receipt_ids,
            distinct_contexts,
            distinct_environments,
            distinct_evaluators,
            distinct_input_profiles,
            distinct_toolchain_profiles,
            distinct_target_profiles,
            policy,
            policy_satisfied,
            independence: IndependenceStatus::NotEstablished,
        })
    }
}

fn validate_common_receipts(receipts: &[EvaluationReceipt]) -> Result<(), ReplicationError> {
    if receipts.is_empty() {
        return Err(ReplicationError::TooFewRepeatabilityReceipts);
    }

    let first_problem = receipts[0].problem_id.clone();
    let first_implementation = receipts[0].implementation_id.clone();
    let mut receipt_ids = BTreeSet::new();
    let mut run_ids = BTreeSet::new();

    for (index, receipt) in receipts.iter().enumerate() {
        receipt
            .validate()
            .map_err(|source| ReplicationError::InvalidReceipt { index, source })?;
        if receipt.problem_id != first_problem {
            return Err(ReplicationError::ProblemMismatch);
        }
        if receipt.implementation_id != first_implementation {
            return Err(ReplicationError::ImplementationMismatch);
        }
        if receipt.correctness != CorrectnessVerdict::Passed {
            return Err(ReplicationError::CorrectnessNotPassed);
        }
        if !receipt_ids.insert(receipt.id.clone()) {
            return Err(ReplicationError::DuplicateReceipt);
        }
        let Some(run_id) = receipt.evidence_run_id.as_ref() else {
            return Err(ReplicationError::MissingRunIdentity);
        };
        if run_id.trim().is_empty() {
            return Err(ReplicationError::MissingRunIdentity);
        }
        if !run_ids.insert(run_id.clone()) {
            return Err(ReplicationError::DuplicateRunIdentity(run_id.clone()));
        }
    }

    Ok(())
}

fn objective_schema(
    objectives: &[ObjectiveMeasurement],
) -> Vec<(String, ObjectiveDirection, String)> {
    objectives
        .iter()
        .map(|objective| {
            (
                objective.name.clone(),
                objective.direction,
                objective.unit.clone(),
            )
        })
        .collect()
}

fn derive_assessment_id(
    problem_id: &ProblemId,
    implementation_id: &ImplementationId,
    receipt_ids: &[ContentId],
    policy: ReplicationPolicy,
    policy_satisfied: bool,
) -> ContentId {
    let mut parts: Vec<Vec<u8>> = vec![
        problem_id.as_content_id().as_str().as_bytes().to_vec(),
        implementation_id.as_content_id().as_str().as_bytes().to_vec(),
        policy.min_receipts.to_be_bytes().to_vec(),
        policy.min_distinct_contexts.to_be_bytes().to_vec(),
        policy.min_distinct_environments.to_be_bytes().to_vec(),
        policy.min_distinct_evaluators.to_be_bytes().to_vec(),
        policy.min_distinct_input_profiles.to_be_bytes().to_vec(),
        vec![u8::from(policy_satisfied)],
    ];
    parts.extend(
        receipt_ids
            .iter()
            .map(|id| id.as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.algorithm-replication-assessment.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Group receipts by exact context without averaging across machines/toolchains.
pub fn repeatability_groups<'a>(
    receipts: &'a [EvaluationReceipt],
) -> BTreeMap<ContentId, Vec<&'a EvaluationReceipt>> {
    let mut groups: BTreeMap<ContentId, Vec<&EvaluationReceipt>> = BTreeMap::new();
    for receipt in receipts {
        groups
            .entry(receipt.context.content_id())
            .or_default()
            .push(receipt);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{EvaluationContext, ObjectiveMeasurement};

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn receipt(run: &str, environment: &str, latency: f64) -> EvaluationReceipt {
        EvaluationReceipt::new(
            ProblemId(cid("problem", "hamming")),
            ImplementationId(cid("impl", "candidate")),
            EvaluationContext::new(
                cid("evaluator", "criterion-v1"),
                cid("oracle", "bitwise"),
                cid("inputs", "seeded-v1"),
                cid("environment", environment),
                "source-rev",
                "rust-1.96.0",
                environment,
                vec![1, 2, 3],
            )
            .unwrap(),
            CorrectnessVerdict::Passed,
            cid("correctness", "pass"),
            vec![
                ObjectiveMeasurement::new(
                    "latency",
                    ObjectiveDirection::Minimize,
                    latency,
                    "ns/op",
                )
                .unwrap(),
            ],
            Some(run.into()),
        )
        .unwrap()
    }

    #[test]
    fn repeatability_requires_same_exact_context() {
        let a = receipt("run-a", "machine-a", 10.0);
        let b = receipt("run-b", "machine-b", 11.0);
        assert_eq!(
            RepeatabilityCohort::new(&[a, b]).unwrap_err(),
            ReplicationError::ContextMismatch
        );
    }

    #[test]
    fn repeatability_summarizes_without_cross_context_averaging() {
        let receipts = vec![
            receipt("run-a", "machine-a", 12.0),
            receipt("run-b", "machine-a", 10.0),
            receipt("run-c", "machine-a", 11.0),
        ];
        let cohort = RepeatabilityCohort::new(&receipts).unwrap();
        assert_eq!(cohort.summaries[0].samples, vec![10.0, 11.0, 12.0]);
        assert_eq!(cohort.summaries[0].median, 11.0);
    }

    #[test]
    fn duplicate_run_identity_is_not_replication() {
        let a = receipt("same-run", "machine-a", 10.0);
        let b = receipt("same-run", "machine-a", 11.0);
        assert_eq!(
            RepeatabilityCohort::new(&[a, b]).unwrap_err(),
            ReplicationError::DuplicateRunIdentity("same-run".into())
        );
    }

    #[test]
    fn default_replication_requires_multiple_environments() {
        let same_machine = vec![
            receipt("run-a", "machine-a", 10.0),
            receipt("run-b", "machine-a", 11.0),
            receipt("run-c", "machine-a", 9.0),
        ];
        let assessment = ReplicationCoverageAssessment::assess(
            &same_machine,
            ReplicationPolicy::default(),
        )
        .unwrap();
        assert!(!assessment.policy_satisfied);
        assert_eq!(assessment.distinct_environments, 1);
        assert_eq!(assessment.independence, IndependenceStatus::NotEstablished);
    }

    #[test]
    fn cross_environment_receipts_can_satisfy_replication_policy() {
        let receipts = vec![
            receipt("run-a", "machine-a", 10.0),
            receipt("run-b", "machine-a", 11.0),
            receipt("run-c", "machine-b", 13.0),
        ];
        let assessment = ReplicationCoverageAssessment::assess(
            &receipts,
            ReplicationPolicy::default(),
        )
        .unwrap();
        assert!(assessment.policy_satisfied);
        assert_eq!(assessment.distinct_contexts, 2);
        assert_eq!(assessment.distinct_environments, 2);
        assert_eq!(assessment.independence, IndependenceStatus::NotEstablished);
    }

    #[test]
    fn metadata_diversity_never_claims_independence() {
        let receipts = vec![
            receipt("run-a", "machine-a", 10.0),
            receipt("run-b", "machine-b", 11.0),
            receipt("run-c", "machine-c", 12.0),
        ];
        let assessment = ReplicationCoverageAssessment::assess(
            &receipts,
            ReplicationPolicy::default(),
        )
        .unwrap();
        assert_eq!(assessment.independence, IndependenceStatus::NotEstablished);
    }
}
