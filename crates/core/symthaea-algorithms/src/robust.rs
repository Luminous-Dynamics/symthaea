//! Conservative observed-range dominance over repeated algorithm measurements.
//!
//! Point-estimate Pareto ranking can reward benchmark noise. This module adds a deliberately
//! stronger descriptive relation using repeatability cohorts collected under one exact evaluation
//! context:
//!
//! ```text
//! observed-range separation != confidence interval != statistical significance != promotion
//! ```
//!
//! Candidate A range-dominates candidate B only when A's *worst observed* value is at least as
//! good as B's *best observed* value on every objective and strictly better on at least one.
//! Overlapping ranges therefore remain incomparable rather than being forced into a winner.

use crate::evaluation::{EvaluationReceipt, ObjectiveDirection};
use crate::replication::{ObjectiveSampleSummary, RepeatabilityCohort, ReplicationError};
use crate::{ContentId, ImplementationId, ProblemId};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ObservedRangeError {
    #[error("at least one candidate repeatability set is required")]
    Empty,
    #[error("candidate {index} repeatability evidence is invalid: {source}")]
    InvalidRepeatability {
        index: usize,
        source: ReplicationError,
    },
    #[error("candidate {index} is for a different problem")]
    ProblemMismatch { index: usize },
    #[error("candidate {index} was measured under a different exact evaluation context")]
    ContextMismatch { index: usize },
    #[error("candidate {index} has a different objective schema")]
    ObjectiveSchemaMismatch { index: usize },
    #[error("duplicate implementation in observed-range cohort")]
    DuplicateImplementation,
    #[error("non-finite range evidence reached robust comparison: {objective}")]
    NonFinite { objective: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservedObjectiveRange {
    name: String,
    direction: ObjectiveDirection,
    unit: String,
    minimum: f64,
    median: f64,
    maximum: f64,
    sample_count: usize,
}

impl ObservedObjectiveRange {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn direction(&self) -> ObjectiveDirection {
        self.direction
    }

    pub fn unit(&self) -> &str {
        &self.unit
    }

    pub fn minimum(&self) -> f64 {
        self.minimum
    }

    pub fn median(&self) -> f64 {
        self.median
    }

    pub fn maximum(&self) -> f64 {
        self.maximum
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservedRangeCandidate {
    implementation_id: ImplementationId,
    receipt_ids: Vec<ContentId>,
    objectives: Vec<ObservedObjectiveRange>,
}

impl ObservedRangeCandidate {
    pub fn implementation_id(&self) -> &ImplementationId {
        &self.implementation_id
    }

    pub fn receipt_ids(&self) -> &[ContentId] {
        &self.receipt_ids
    }

    pub fn objectives(&self) -> &[ObservedObjectiveRange] {
        &self.objectives
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectiveSchemaEntry {
    name: String,
    direction: ObjectiveDirection,
    unit: String,
}

/// A comparison set whose candidates each have at least two same-context measurements.
///
/// Construction re-derives every [`RepeatabilityCohort`] from raw receipts rather than accepting
/// caller-constructed summaries, so observed ranges cannot be fabricated by directly filling
/// public summary fields.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedRangeCohort {
    problem_id: ProblemId,
    context_id: ContentId,
    schema: Vec<ObjectiveSchemaEntry>,
    candidates: Vec<ObservedRangeCandidate>,
}

impl ObservedRangeCohort {
    pub fn new(candidate_receipts: &[Vec<EvaluationReceipt>]) -> Result<Self, ObservedRangeError> {
        if candidate_receipts.is_empty() {
            return Err(ObservedRangeError::Empty);
        }

        let mut repeatability = Vec::with_capacity(candidate_receipts.len());
        for (index, receipts) in candidate_receipts.iter().enumerate() {
            let cohort = RepeatabilityCohort::new(receipts).map_err(|source| {
                ObservedRangeError::InvalidRepeatability { index, source }
            })?;
            repeatability.push(cohort);
        }

        let first = &repeatability[0];
        let problem_id = first.problem_id.clone();
        let context_id = first.context_id.clone();
        let schema = schema_from_summaries(&first.summaries);
        let mut implementation_ids = BTreeSet::new();
        let mut candidates = Vec::with_capacity(repeatability.len());

        for (index, cohort) in repeatability.into_iter().enumerate() {
            if cohort.problem_id != problem_id {
                return Err(ObservedRangeError::ProblemMismatch { index });
            }
            if cohort.context_id != context_id {
                return Err(ObservedRangeError::ContextMismatch { index });
            }
            if schema_from_summaries(&cohort.summaries) != schema {
                return Err(ObservedRangeError::ObjectiveSchemaMismatch { index });
            }
            if !implementation_ids.insert(cohort.implementation_id.clone()) {
                return Err(ObservedRangeError::DuplicateImplementation);
            }

            let objectives = cohort
                .summaries
                .iter()
                .map(range_from_summary)
                .collect::<Result<Vec<_>, _>>()?;
            candidates.push(ObservedRangeCandidate {
                implementation_id: cohort.implementation_id,
                receipt_ids: cohort.receipt_ids,
                objectives,
            });
        }

        Ok(Self {
            problem_id,
            context_id,
            schema,
            candidates,
        })
    }

    pub fn problem_id(&self) -> &ProblemId {
        &self.problem_id
    }

    pub fn context_id(&self) -> &ContentId {
        &self.context_id
    }

    pub fn candidates(&self) -> &[ObservedRangeCandidate] {
        &self.candidates
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn objective_count(&self) -> usize {
        self.schema.len()
    }

    /// Indices of candidates not observed-range-dominated by any other candidate.
    pub fn frontier_indices(&self) -> Vec<usize> {
        (0..self.candidates.len())
            .filter(|&candidate| {
                !(0..self.candidates.len()).any(|other| {
                    other != candidate
                        && range_dominates(
                            &self.candidates[other].objectives,
                            &self.candidates[candidate].objectives,
                        )
                })
            })
            .collect()
    }

    pub fn frontier(&self) -> Vec<&ObservedRangeCandidate> {
        self.frontier_indices()
            .into_iter()
            .map(|index| &self.candidates[index])
            .collect()
    }

    pub fn range_dominates(&self, left: usize, right: usize) -> Option<bool> {
        let left = self.candidates.get(left)?;
        let right = self.candidates.get(right)?;
        Some(range_dominates(&left.objectives, &right.objectives))
    }
}

fn range_from_summary(
    summary: &ObjectiveSampleSummary,
) -> Result<ObservedObjectiveRange, ObservedRangeError> {
    if !summary.minimum.is_finite()
        || !summary.median.is_finite()
        || !summary.maximum.is_finite()
        || summary.samples.iter().any(|sample| !sample.is_finite())
    {
        return Err(ObservedRangeError::NonFinite {
            objective: summary.name.clone(),
        });
    }
    Ok(ObservedObjectiveRange {
        name: summary.name.clone(),
        direction: summary.direction,
        unit: summary.unit.clone(),
        minimum: summary.minimum,
        median: summary.median,
        maximum: summary.maximum,
        sample_count: summary.samples.len(),
    })
}

fn schema_from_summaries(summaries: &[ObjectiveSampleSummary]) -> Vec<ObjectiveSchemaEntry> {
    summaries
        .iter()
        .map(|summary| ObjectiveSchemaEntry {
            name: summary.name.clone(),
            direction: summary.direction,
            unit: summary.unit.clone(),
        })
        .collect()
}

fn range_dominates(left: &[ObservedObjectiveRange], right: &[ObservedObjectiveRange]) -> bool {
    debug_assert_eq!(left.len(), right.len());
    let mut strictly_separated = false;

    for (a, b) in left.iter().zip(right) {
        match a.direction {
            ObjectiveDirection::Minimize => {
                if a.maximum > b.minimum {
                    return false;
                }
                if a.maximum < b.minimum {
                    strictly_separated = true;
                }
            }
            ObjectiveDirection::Maximize => {
                if a.minimum < b.maximum {
                    return false;
                }
                if a.minimum > b.maximum {
                    strictly_separated = true;
                }
            }
        }
    }

    strictly_separated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{
        CorrectnessVerdict, EvaluationContext, EvaluationReceipt, ObjectiveMeasurement,
    };

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn context() -> EvaluationContext {
        EvaluationContext::new(
            cid("evaluator", "criterion-v1"),
            cid("oracle", "reference-v1"),
            cid("inputs", "fixed-corpus"),
            cid("environment", "machine-a"),
            "source-rev",
            "rust-1.96.0",
            "x86_64-test",
            vec![1, 2, 3],
        )
        .unwrap()
    }

    fn receipt(
        implementation: &str,
        run: &str,
        latency: f64,
        memory: Option<f64>,
    ) -> EvaluationReceipt {
        let mut objectives = vec![
            ObjectiveMeasurement::new(
                "latency",
                ObjectiveDirection::Minimize,
                latency,
                "ns/op",
            )
            .unwrap(),
        ];
        if let Some(memory) = memory {
            objectives.push(
                ObjectiveMeasurement::new(
                    "memory",
                    ObjectiveDirection::Minimize,
                    memory,
                    "bytes",
                )
                .unwrap(),
            );
        }
        EvaluationReceipt::new(
            ProblemId(cid("problem", "hamming")),
            ImplementationId(cid("implementation", implementation)),
            context(),
            CorrectnessVerdict::Passed,
            cid("correctness", implementation),
            objectives,
            Some(run.into()),
        )
        .unwrap()
    }

    #[test]
    fn separated_ranges_produce_robust_winner() {
        let fast = vec![
            receipt("fast", "fast-a", 8.0, None),
            receipt("fast", "fast-b", 9.0, None),
            receipt("fast", "fast-c", 8.5, None),
        ];
        let slow = vec![
            receipt("slow", "slow-a", 11.0, None),
            receipt("slow", "slow-b", 12.0, None),
            receipt("slow", "slow-c", 11.5, None),
        ];
        let cohort = ObservedRangeCohort::new(&[fast, slow]).unwrap();
        assert_eq!(cohort.frontier_indices(), vec![0]);
        assert_eq!(cohort.range_dominates(0, 1), Some(true));
        assert_eq!(cohort.range_dominates(1, 0), Some(false));
    }

    #[test]
    fn overlapping_ranges_refuse_point_estimate_winner() {
        let a = vec![
            receipt("a", "a-1", 9.0, None),
            receipt("a", "a-2", 11.0, None),
        ];
        let b = vec![
            receipt("b", "b-1", 10.0, None),
            receipt("b", "b-2", 10.5, None),
        ];
        let cohort = ObservedRangeCohort::new(&[a, b]).unwrap();
        assert_eq!(cohort.frontier_indices(), vec![0, 1]);
        assert_eq!(cohort.range_dominates(0, 1), Some(false));
        assert_eq!(cohort.range_dominates(1, 0), Some(false));
    }

    #[test]
    fn multiobjective_tradeoff_remains_on_frontier() {
        let fast_large = vec![
            receipt("fast-large", "fl-1", 8.0, Some(20.0)),
            receipt("fast-large", "fl-2", 9.0, Some(21.0)),
        ];
        let slow_small = vec![
            receipt("slow-small", "ss-1", 11.0, Some(10.0)),
            receipt("slow-small", "ss-2", 12.0, Some(11.0)),
        ];
        let cohort = ObservedRangeCohort::new(&[fast_large, slow_small]).unwrap();
        assert_eq!(cohort.frontier_indices(), vec![0, 1]);
    }

    #[test]
    fn cross_context_candidate_sets_are_rejected() {
        let a = vec![
            receipt("a", "a-1", 8.0, None),
            receipt("a", "a-2", 8.5, None),
        ];
        let mut b1 = receipt("b", "b-1", 10.0, None);
        let mut b2 = receipt("b", "b-2", 10.5, None);
        b1.context.environment_id = cid("environment", "machine-b");
        b2.context.environment_id = cid("environment", "machine-b");
        // Re-mint valid receipts in the changed context instead of relying on tampered evidence.
        let changed_context = EvaluationContext::new(
            cid("evaluator", "criterion-v1"),
            cid("oracle", "reference-v1"),
            cid("inputs", "fixed-corpus"),
            cid("environment", "machine-b"),
            "source-rev",
            "rust-1.96.0",
            "x86_64-test",
            vec![1, 2, 3],
        )
        .unwrap();
        b1 = EvaluationReceipt::new(
            b1.problem_id,
            b1.implementation_id,
            changed_context.clone(),
            CorrectnessVerdict::Passed,
            b1.correctness_evidence_id,
            b1.objectives,
            b1.evidence_run_id,
        )
        .unwrap();
        b2 = EvaluationReceipt::new(
            b2.problem_id,
            b2.implementation_id,
            changed_context,
            CorrectnessVerdict::Passed,
            b2.correctness_evidence_id,
            b2.objectives,
            b2.evidence_run_id,
        )
        .unwrap();
        assert!(matches!(
            ObservedRangeCohort::new(&[a, vec![b1, b2]]),
            Err(ObservedRangeError::ContextMismatch { index: 1 })
        ));
    }

    #[test]
    fn objective_schema_mismatch_is_rejected() {
        let a = vec![
            receipt("a", "a-1", 8.0, None),
            receipt("a", "a-2", 8.5, None),
        ];
        let b = vec![
            receipt("b", "b-1", 10.0, Some(10.0)),
            receipt("b", "b-2", 10.5, Some(11.0)),
        ];
        assert!(matches!(
            ObservedRangeCohort::new(&[a, b]),
            Err(ObservedRangeError::ObjectiveSchemaMismatch { index: 1 })
        ));
    }
}
