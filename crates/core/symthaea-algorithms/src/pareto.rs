//! Finite-safe, context-coherent Pareto comparison for algorithm evaluations.
//!
//! This module deliberately refuses to compare receipts collected under different problem,
//! evaluator/environment, or objective schemas. Cross-environment performance is evidence for
//! separate frontiers unless a higher-level normalization theorem is supplied elsewhere.

use crate::evaluation::{
    CorrectnessVerdict, EvaluationError, EvaluationReceipt, ObjectiveDirection,
    ObjectiveMeasurement,
};
use crate::{ContentId, ProblemId};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ParetoError {
    #[error("at least one evaluation receipt is required")]
    Empty,
    #[error("evaluation receipt {index} is invalid: {source}")]
    InvalidReceipt {
        index: usize,
        source: EvaluationError,
    },
    #[error("evaluation receipt {index} did not pass correctness")]
    CorrectnessNotPassed { index: usize },
    #[error("evaluation receipt {index} is for a different problem")]
    ProblemMismatch { index: usize },
    #[error("evaluation receipt {index} is from a different evaluation context")]
    ContextMismatch { index: usize },
    #[error("evaluation receipt {index} has a different objective schema")]
    ObjectiveSchemaMismatch { index: usize },
    #[error("non-finite objective reached Pareto comparison: {name}={value}")]
    NonFinite { name: String, value: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectiveSchemaEntry {
    name: String,
    direction: ObjectiveDirection,
    unit: String,
}

fn schema(objectives: &[ObjectiveMeasurement]) -> Vec<ObjectiveSchemaEntry> {
    objectives
        .iter()
        .map(|objective| ObjectiveSchemaEntry {
            name: objective.name.clone(),
            direction: objective.direction,
            unit: objective.unit.clone(),
        })
        .collect()
}

/// A validated comparison cohort.
///
/// Construction proves every receipt:
/// - self-validates;
/// - has `CorrectnessVerdict::Passed`;
/// - names the same problem;
/// - was measured under the same exact evaluation context;
/// - has the same ordered objective schema;
/// - contains only finite objective values.
#[derive(Debug)]
pub struct ParetoCohort<'a> {
    receipts: &'a [EvaluationReceipt],
    problem_id: ProblemId,
    context_id: ContentId,
    objective_schema: Vec<ObjectiveSchemaEntry>,
}

impl<'a> ParetoCohort<'a> {
    pub fn new(receipts: &'a [EvaluationReceipt]) -> Result<Self, ParetoError> {
        let Some(first) = receipts.first() else {
            return Err(ParetoError::Empty);
        };
        first
            .validate()
            .map_err(|source| ParetoError::InvalidReceipt { index: 0, source })?;
        if first.correctness != CorrectnessVerdict::Passed {
            return Err(ParetoError::CorrectnessNotPassed { index: 0 });
        }
        validate_finite(first)?;

        let problem_id = first.problem_id.clone();
        let context_id = first.context.content_id();
        let objective_schema = schema(&first.objectives);

        for (index, receipt) in receipts.iter().enumerate().skip(1) {
            receipt
                .validate()
                .map_err(|source| ParetoError::InvalidReceipt { index, source })?;
            if receipt.correctness != CorrectnessVerdict::Passed {
                return Err(ParetoError::CorrectnessNotPassed { index });
            }
            if receipt.problem_id != problem_id {
                return Err(ParetoError::ProblemMismatch { index });
            }
            if receipt.context.content_id() != context_id {
                return Err(ParetoError::ContextMismatch { index });
            }
            if schema(&receipt.objectives) != objective_schema {
                return Err(ParetoError::ObjectiveSchemaMismatch { index });
            }
            validate_finite(receipt)?;
        }

        Ok(Self {
            receipts,
            problem_id,
            context_id,
            objective_schema,
        })
    }

    pub fn problem_id(&self) -> &ProblemId {
        &self.problem_id
    }

    pub fn context_id(&self) -> &ContentId {
        &self.context_id
    }

    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }

    /// Return indices of all non-dominated receipts, preserving input order.
    pub fn frontier_indices(&self) -> Vec<usize> {
        (0..self.receipts.len())
            .filter(|&candidate| {
                !(0..self.receipts.len()).any(|other| {
                    other != candidate
                        && dominates(
                            &self.receipts[other].objectives,
                            &self.receipts[candidate].objectives,
                        )
                })
            })
            .collect()
    }

    pub fn frontier(&self) -> Vec<&'a EvaluationReceipt> {
        self.frontier_indices()
            .into_iter()
            .map(|index| &self.receipts[index])
            .collect()
    }

    pub fn objective_count(&self) -> usize {
        self.objective_schema.len()
    }
}

fn validate_finite(receipt: &EvaluationReceipt) -> Result<(), ParetoError> {
    for objective in &receipt.objectives {
        if !objective.value.is_finite() {
            return Err(ParetoError::NonFinite {
                name: objective.name.clone(),
                value: objective.value,
            });
        }
    }
    Ok(())
}

fn dominates(a: &[ObjectiveMeasurement], b: &[ObjectiveMeasurement]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut strictly_better = false;

    for (left, right) in a.iter().zip(b) {
        let ordering = match left.direction {
            ObjectiveDirection::Maximize => left.value.partial_cmp(&right.value),
            ObjectiveDirection::Minimize => right.value.partial_cmp(&left.value),
        };
        match ordering {
            Some(std::cmp::Ordering::Less) => return false,
            Some(std::cmp::Ordering::Greater) => strictly_better = true,
            Some(std::cmp::Ordering::Equal) => {}
            None => return false, // unreachable after cohort validation; fail closed regardless.
        }
    }

    strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{EvaluationContext, ObjectiveMeasurement};
    use crate::{ContentId, ImplementationId, ProblemId};

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn context(target: &str) -> EvaluationContext {
        EvaluationContext::new(
            cid("evaluator", "criterion-v1"),
            cid("oracle", "reference-v1"),
            cid("inputs", "seeded-v1"),
            cid("environment", target),
            "abc123",
            "rust-1.96.0",
            target,
            vec![1, 2, 3],
        )
        .unwrap()
    }

    fn receipt(name: &str, latency: f64, memory: f64) -> EvaluationReceipt {
        EvaluationReceipt::new(
            ProblemId(cid("problem", "hdc-hamming")),
            ImplementationId(cid("implementation", name)),
            context("x86_64-test"),
            CorrectnessVerdict::Passed,
            cid("correctness", name),
            vec![
                ObjectiveMeasurement::new(
                    "latency",
                    ObjectiveDirection::Minimize,
                    latency,
                    "ns/op",
                )
                .unwrap(),
                ObjectiveMeasurement::new(
                    "memory",
                    ObjectiveDirection::Minimize,
                    memory,
                    "bytes",
                )
                .unwrap(),
            ],
            Some(format!("run-{name}")),
        )
        .unwrap()
    }

    #[test]
    fn frontier_preserves_tradeoffs() {
        let receipts = vec![
            receipt("fast-large", 5.0, 20.0),
            receipt("slow-small", 10.0, 10.0),
            receipt("dominated", 11.0, 21.0),
        ];
        let cohort = ParetoCohort::new(&receipts).unwrap();
        assert_eq!(cohort.frontier_indices(), vec![0, 1]);
    }

    #[test]
    fn universally_better_candidate_dominates() {
        let receipts = vec![receipt("winner", 5.0, 10.0), receipt("loser", 6.0, 11.0)];
        let cohort = ParetoCohort::new(&receipts).unwrap();
        assert_eq!(cohort.frontier_indices(), vec![0]);
    }

    #[test]
    fn refuses_cross_environment_comparison() {
        let first = receipt("first", 5.0, 10.0);
        let mut second = receipt("second", 6.0, 11.0);
        second.context = context("aarch64-test");
        // Re-mint so the receipt itself is valid for the changed environment.
        let second = EvaluationReceipt::new(
            second.problem_id,
            second.implementation_id,
            second.context,
            second.correctness,
            second.correctness_evidence_id,
            second.objectives,
            second.evidence_run_id,
        )
        .unwrap();
        let receipts = vec![first, second];
        assert_eq!(
            ParetoCohort::new(&receipts).unwrap_err(),
            ParetoError::ContextMismatch { index: 1 }
        );
    }

    #[test]
    fn correctness_failure_is_not_a_bad_objective_value() {
        let first = receipt("first", 5.0, 10.0);
        let failed = EvaluationReceipt::new(
            first.problem_id.clone(),
            ImplementationId(cid("implementation", "failed")),
            first.context.clone(),
            CorrectnessVerdict::Failed,
            cid("correctness", "failed"),
            first.objectives.clone(),
            Some("run-failed".into()),
        )
        .unwrap();
        let receipts = vec![first, failed];
        assert_eq!(
            ParetoCohort::new(&receipts).unwrap_err(),
            ParetoError::CorrectnessNotPassed { index: 1 }
        );
    }
}
