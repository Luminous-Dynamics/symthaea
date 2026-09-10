// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-context aggregation for repeated Forge searches.
//!
//! v1 deliberately permits only the narrowest defensible cross-run comparison: same semantic
//! problem, same baseline implementation, and the exact same `EvaluationContext`. Distinct
//! discovery runs (normally distinct search seeds) may contribute repeated search attempts. A
//! single `DiscoveryRun` may appear only once, so alternate histories of one run cannot be double
//! counted as independent searches.
//!
//! These are descriptive search-memory statistics. Forge-local selection is not replicated
//! performance evidence and this module grants no promotion or runtime authority.

use crate::learning::{ForgeLearningError, ForgeTrialBatch, TransformationOutcomeTable};
use crate::trace::ForgeTraceEvent;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::discovery::DiscoveryRun;
use symthaea_algorithms::evaluation::{EvaluationContext, EvaluationError};
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{
    ContentId, ImplementationId, ImplementationRecord, ProblemId, TransformationId,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeContextLearningError {
    #[error(transparent)]
    Learning(#[from] ForgeLearningError),
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
    #[error("evaluation context source revision does not match discovery-run baseline revision")]
    SourceRevisionMismatch,
    #[error("exact-context cohort requires at least one batch")]
    EmptyCohort,
    #[error("context-bound batch identity does not match canonical fields")]
    BatchIdentityMismatch,
    #[error("cohort contains a duplicate context-bound batch")]
    DuplicateBatch,
    #[error("cohort contains more than one history for the same DiscoveryRun")]
    DuplicateRun,
    #[error("cohort mixes semantic problems")]
    ProblemMismatch,
    #[error("cohort mixes baseline implementations")]
    BaselineMismatch,
    #[error("cohort mixes evaluation contexts")]
    ContextMismatch,
    #[error("transformation outcome count overflow")]
    CountOverflow,
    #[error("contextual outcome row is internally inconsistent")]
    InvalidCounts,
    #[error("contextual outcome row identity does not match canonical fields")]
    RowIdentityMismatch,
    #[error("contextual outcome table identity does not match canonical fields")]
    TableIdentityMismatch,
}

/// One run-bound Forge learning batch paired with the exact evaluation context that constrains
/// later comparison. The context is a compatibility key here, not a claim that Forge emitted an
/// evidence-grade `EvaluationReceipt`.
#[derive(Debug, Clone, Serialize)]
pub struct ContextBoundForgeBatch {
    id: ContentId,
    context_id: ContentId,
    context: EvaluationContext,
    batch: ForgeTrialBatch,
    table: TransformationOutcomeTable,
}

impl ContextBoundForgeBatch {
    pub fn from_trace(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        context: EvaluationContext,
        trace: &[ForgeTraceEvent],
        observations: &ObservationStore,
    ) -> Result<Self, ForgeContextLearningError> {
        run.validate().map_err(ForgeLearningError::from)?;
        baseline.validate().map_err(ForgeLearningError::from)?;
        context.validate()?;
        if context.source_revision != run.baseline_revision {
            return Err(ForgeContextLearningError::SourceRevisionMismatch);
        }
        let batch = ForgeTrialBatch::from_trace(run, baseline, trace, observations)?;
        let table = TransformationOutcomeTable::from_batch(&batch)?;
        let context_id = context.content_id();
        let id = ContentId::derive(
            "symthaea.forge-context-bound-batch.v1",
            [
                batch.id().as_str().as_bytes(),
                context_id.as_str().as_bytes(),
                table.id().as_str().as_bytes(),
            ],
        );
        Ok(Self {
            id,
            context_id,
            context,
            batch,
            table,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn run_id(&self) -> &ContentId {
        self.batch.run_id()
    }
    pub fn problem_id(&self) -> &ProblemId {
        self.batch.problem_id()
    }
    pub fn baseline_implementation_id(&self) -> &ImplementationId {
        self.batch.baseline_implementation_id()
    }
    pub fn context_id(&self) -> &ContentId {
        &self.context_id
    }
    pub fn table(&self) -> &TransformationOutcomeTable {
        &self.table
    }

    pub fn validate(&self) -> Result<(), ForgeContextLearningError> {
        self.context.validate()?;
        if self.context.content_id() != self.context_id {
            return Err(ForgeContextLearningError::BatchIdentityMismatch);
        }
        let expected = ContentId::derive(
            "symthaea.forge-context-bound-batch.v1",
            [
                self.batch.id().as_str().as_bytes(),
                self.context_id.as_str().as_bytes(),
                self.table.id().as_str().as_bytes(),
            ],
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeContextLearningError::BatchIdentityMismatch)
        }
    }
}

/// Canonical cohort of distinct discovery runs whose learning context is exactly equal.
#[derive(Debug, Clone, Serialize)]
pub struct ExactContextForgeCohort {
    id: ContentId,
    problem_id: ProblemId,
    baseline_implementation_id: ImplementationId,
    context_id: ContentId,
    batches: Vec<ContextBoundForgeBatch>,
}

impl ExactContextForgeCohort {
    pub fn new(
        mut batches: Vec<ContextBoundForgeBatch>,
    ) -> Result<Self, ForgeContextLearningError> {
        if batches.is_empty() {
            return Err(ForgeContextLearningError::EmptyCohort);
        }
        for batch in &batches {
            batch.validate()?;
        }
        batches.sort_by(|a, b| a.id.cmp(&b.id));

        let mut batch_ids = BTreeSet::new();
        let mut run_ids = BTreeSet::new();
        for batch in &batches {
            if !batch_ids.insert(batch.id.as_str().to_string()) {
                return Err(ForgeContextLearningError::DuplicateBatch);
            }
            if !run_ids.insert(batch.run_id().as_str().to_string()) {
                return Err(ForgeContextLearningError::DuplicateRun);
            }
        }

        let first = &batches[0];
        let problem_id = first.problem_id().clone();
        let baseline_implementation_id = first.baseline_implementation_id().clone();
        let context_id = first.context_id.clone();
        for batch in &batches[1..] {
            if batch.problem_id() != &problem_id {
                return Err(ForgeContextLearningError::ProblemMismatch);
            }
            if batch.baseline_implementation_id() != &baseline_implementation_id {
                return Err(ForgeContextLearningError::BaselineMismatch);
            }
            if batch.context_id != context_id {
                return Err(ForgeContextLearningError::ContextMismatch);
            }
        }

        let id = derive_cohort_id(
            &problem_id,
            &baseline_implementation_id,
            &context_id,
            &batches,
        );
        Ok(Self {
            id,
            problem_id,
            baseline_implementation_id,
            context_id,
            batches,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }
    pub fn problem_id(&self) -> &ProblemId {
        &self.problem_id
    }
    pub fn baseline_implementation_id(&self) -> &ImplementationId {
        &self.baseline_implementation_id
    }
    pub fn context_id(&self) -> &ContentId {
        &self.context_id
    }
    pub fn batches(&self) -> &[ContextBoundForgeBatch] {
        &self.batches
    }

    pub fn validate(&self) -> Result<(), ForgeContextLearningError> {
        let rebuilt = Self::new(self.batches.clone())?;
        if rebuilt.id == self.id
            && rebuilt.problem_id == self.problem_id
            && rebuilt.baseline_implementation_id == self.baseline_implementation_id
            && rebuilt.context_id == self.context_id
        {
            Ok(())
        } else {
            Err(ForgeContextLearningError::BatchIdentityMismatch)
        }
    }
}

fn derive_cohort_id(
    problem_id: &ProblemId,
    baseline_id: &ImplementationId,
    context_id: &ContentId,
    batches: &[ContextBoundForgeBatch],
) -> ContentId {
    let count = (batches.len() as u64).to_be_bytes();
    let mut parts = vec![
        problem_id.as_content_id().as_str().as_bytes().to_vec(),
        baseline_id.as_content_id().as_str().as_bytes().to_vec(),
        context_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        batches
            .iter()
            .map(|batch| batch.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-exact-context-cohort.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[derive(Debug, Clone, Default)]
struct Counts {
    trials: u64,
    compile: u64,
    correctness: u64,
    evaluation: u64,
    not_selected: u64,
    selected: u64,
    interrupted: u64,
}

impl Counts {
    fn add_row(
        &mut self,
        row: &crate::learning::TransformationOutcomeStats,
    ) -> Result<(), ForgeContextLearningError> {
        self.trials = add(self.trials, row.trials())?;
        self.compile = add(self.compile, row.rejected_compilation())?;
        self.correctness = add(self.correctness, row.rejected_correctness())?;
        self.evaluation = add(self.evaluation, row.rejected_evaluation())?;
        self.not_selected = add(self.not_selected, row.valid_not_selected())?;
        self.selected = add(self.selected, row.selected_for_continuation())?;
        self.interrupted = add(self.interrupted, row.interrupted())?;
        Ok(())
    }
}

fn add(left: u64, right: u64) -> Result<u64, ForgeContextLearningError> {
    left.checked_add(right)
        .ok_or(ForgeContextLearningError::CountOverflow)
}

/// Counts for one transformation across distinct searches in exactly one cohort.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContextualTransformationStats {
    id: ContentId,
    transformation_id: TransformationId,
    counts: [u64; 7],
}

impl ContextualTransformationStats {
    fn from_counts(transformation_id: TransformationId, counts: Counts) -> Result<Self, ForgeContextLearningError> {
        let values = [
            counts.trials,
            counts.compile,
            counts.correctness,
            counts.evaluation,
            counts.not_selected,
            counts.selected,
            counts.interrupted,
        ];
        let mut row = Self {
            id: ContentId::derive("symthaea.forge-context-row.uninitialized", [b"v1".as_slice()]),
            transformation_id,
            counts: values,
        };
        row.id = derive_row_id(&row);
        row.validate()?;
        Ok(row)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn transformation_id(&self) -> &TransformationId { &self.transformation_id }
    pub fn trials(&self) -> u64 { self.counts[0] }
    pub fn rejected_compilation(&self) -> u64 { self.counts[1] }
    pub fn rejected_correctness(&self) -> u64 { self.counts[2] }
    pub fn rejected_evaluation(&self) -> u64 { self.counts[3] }
    pub fn valid_not_selected(&self) -> u64 { self.counts[4] }
    pub fn selected_for_continuation(&self) -> u64 { self.counts[5] }
    pub fn interrupted(&self) -> u64 { self.counts[6] }
    pub fn completed_trials(&self) -> u64 { self.trials() - self.interrupted() }

    pub fn compile_survival_rate(&self) -> Option<f64> {
        let completed = self.completed_trials();
        ratio(completed - self.rejected_compilation(), completed)
    }

    pub fn correctness_survival_rate(&self) -> Option<f64> {
        let entered = self.completed_trials() - self.rejected_compilation();
        ratio(entered - self.rejected_correctness(), entered)
    }

    pub fn local_selection_rate(&self) -> Option<f64> {
        ratio(
            self.selected_for_continuation(),
            self.valid_not_selected() + self.selected_for_continuation(),
        )
    }

    pub fn interruption_rate(&self) -> Option<f64> {
        ratio(self.interrupted(), self.trials())
    }

    pub fn validate(&self) -> Result<(), ForgeContextLearningError> {
        let classified = self.counts[1..]
            .iter()
            .try_fold(0u64, |sum, value| add(sum, *value))?;
        if classified != self.trials() || self.interrupted() > self.trials() {
            return Err(ForgeContextLearningError::InvalidCounts);
        }
        if derive_row_id(self) == self.id {
            Ok(())
        } else {
            Err(ForgeContextLearningError::RowIdentityMismatch)
        }
    }
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn derive_row_id(row: &ContextualTransformationStats) -> ContentId {
    let mut parts = vec![
        row.transformation_id
            .as_content_id()
            .as_str()
            .as_bytes()
            .to_vec(),
    ];
    parts.extend(row.counts.iter().map(|value| value.to_be_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-contextual-transformation-row.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[derive(Debug, Clone, Serialize)]
pub struct ExactContextTransformationTable {
    id: ContentId,
    cohort_id: ContentId,
    rows: Vec<ContextualTransformationStats>,
}

impl ExactContextTransformationTable {
    pub fn from_cohort(
        cohort: &ExactContextForgeCohort,
    ) -> Result<Self, ForgeContextLearningError> {
        cohort.validate()?;
        let mut grouped = BTreeMap::<TransformationId, Counts>::new();
        for batch in cohort.batches() {
            for row in batch.table().rows() {
                grouped
                    .entry(row.transformation_id().clone())
                    .or_default()
                    .add_row(row)?;
            }
        }
        let rows = grouped
            .into_iter()
            .map(|(transformation, counts)| ContextualTransformationStats::from_counts(transformation, counts))
            .collect::<Result<Vec<_>, _>>()?;
        let id = derive_table_id(cohort.id(), &rows);
        let table = Self {
            id,
            cohort_id: cohort.id().clone(),
            rows,
        };
        table.validate()?;
        Ok(table)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn cohort_id(&self) -> &ContentId { &self.cohort_id }
    pub fn rows(&self) -> &[ContextualTransformationStats] { &self.rows }

    pub fn get(&self, transformation: &TransformationId) -> Option<&ContextualTransformationStats> {
        self.rows
            .binary_search_by(|row| row.transformation_id.cmp(transformation))
            .ok()
            .map(|index| &self.rows[index])
    }

    pub fn validate(&self) -> Result<(), ForgeContextLearningError> {
        let mut previous: Option<&TransformationId> = None;
        for row in &self.rows {
            row.validate()?;
            if previous.is_some_and(|prior| prior >= &row.transformation_id) {
                return Err(ForgeContextLearningError::TableIdentityMismatch);
            }
            previous = Some(&row.transformation_id);
        }
        if derive_table_id(&self.cohort_id, &self.rows) == self.id {
            Ok(())
        } else {
            Err(ForgeContextLearningError::TableIdentityMismatch)
        }
    }
}

fn derive_table_id(cohort_id: &ContentId, rows: &[ContextualTransformationStats]) -> ContentId {
    let count = (rows.len() as u64).to_be_bytes();
    let mut parts = vec![cohort_id.as_str().as_bytes().to_vec(), count.to_vec()];
    parts.extend(rows.iter().map(|row| row.id().as_str().as_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-exact-context-transformation-table.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{full_source_artifact_id, MutationRecord};
    use crate::fitness::{Gate, GateResult};
    use crate::observations as forge_observations;
    use crate::trace::{ForgeAttemptId, ForgeTraceEvent};
    use std::time::Duration;
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::ledger::DiscoveryEventKind;
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, DeterminismRequirement, DiscoveryRisk, ProblemSpec,
        SemanticGuarantee,
    };

    fn setup(seed: u64, environment: &str) -> (
        DiscoveryRun,
        ImplementationRecord,
        EvaluationContext,
        Vec<ForgeTraceEvent>,
        ObservationStore,
        TransformationId,
    ) {
        let problem = ProblemSpec::new(
            "context-learning-test",
            "Return exact reference value.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        ).unwrap();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "baseline",
            "test baseline",
            AlgorithmProvenance::HumanAuthored,
        ).unwrap();
        let baseline_artifact = full_source_artifact_id("fn f() -> i32 { 1 }\n");
        let baseline = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id,
            "repo://src/f.rs",
            baseline_artifact.clone(),
            None,
        ).unwrap();
        let run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            ContentId::derive("generator", [b"forge".as_slice()]),
            "abc123",
            SearchBudget::new(10, 1, 10).unwrap(),
            seed,
        ).unwrap();
        let context = EvaluationContext::new(
            ContentId::derive("evaluator", [b"criterion".as_slice()]),
            ContentId::derive("oracle", [b"reference".as_slice()]),
            ContentId::derive("inputs", [b"fixed".as_slice()]),
            ContentId::derive("environment", [environment.as_bytes()]),
            "abc123",
            "rustc-1.96.0",
            "x86_64-v3",
            vec![1, 2, 3],
        ).unwrap();
        let candidate = full_source_artifact_id("fn f() -> i32 { 2 }\n");
        let attempt = ForgeAttemptId::derive(&baseline_artifact, seed, 0, 0);
        let mutation = MutationRecord::new(0, "literal", "1 -> 2", baseline_artifact, candidate);
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let gates = vec![
            GateResult { gate: Gate::Compile, passed: true, output_tail: String::new(), duration: Duration::from_nanos(1) },
            GateResult { gate: Gate::Test, passed: false, output_tail: "counterexample".into(), duration: Duration::from_nanos(1) },
        ];
        let rejected = forge_observations::gates(&attempt, &mutation, &gates).unwrap();
        let summary = forge_observations::search_summary(1, 0, 0, 1, 0, 0, 0, None, None).unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt.clone(), 0, DiscoveryEventKind::CandidateGenerated,
                mutation.candidate_artifact_id.clone(), generated.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt, 0, DiscoveryEventKind::RejectedCorrectness,
                mutation.candidate_artifact_id.clone(), rejected.id().clone(),
            ),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![generated, rejected, summary]).unwrap();
        (run, baseline, context, trace, observations, mutation.transformation_id)
    }

    fn bound(seed: u64, environment: &str) -> (ContextBoundForgeBatch, TransformationId) {
        let (run, baseline, context, trace, observations, transformation) = setup(seed, environment);
        let batch = ContextBoundForgeBatch::from_trace(
            &run, &baseline, context, &trace, &observations,
        ).unwrap();
        (batch, transformation)
    }

    #[test]
    fn distinct_search_runs_aggregate_only_under_identical_context() {
        let (a, transformation) = bound(7, "machine-a");
        let (b, _) = bound(8, "machine-a");
        assert_ne!(a.run_id(), b.run_id());
        let cohort = ExactContextForgeCohort::new(vec![a, b]).unwrap();
        let table = ExactContextTransformationTable::from_cohort(&cohort).unwrap();
        let row = table.get(&transformation).unwrap();
        assert_eq!(row.trials(), 2);
        assert_eq!(row.rejected_correctness(), 2);
        assert_eq!(row.correctness_survival_rate(), Some(0.0));
    }

    #[test]
    fn same_run_cannot_be_double_counted() {
        let (a, _) = bound(7, "machine-a");
        let duplicate = a.clone();
        assert!(matches!(
            ExactContextForgeCohort::new(vec![a, duplicate]),
            Err(ForgeContextLearningError::DuplicateBatch)
        ));
    }

    #[test]
    fn environment_change_blocks_aggregation() {
        let (a, _) = bound(7, "machine-a");
        let (b, _) = bound(8, "machine-b");
        assert!(matches!(
            ExactContextForgeCohort::new(vec![a, b]),
            Err(ForgeContextLearningError::ContextMismatch)
        ));
    }

    #[test]
    fn context_revision_must_equal_frozen_run_revision() {
        let (run, baseline, context, trace, observations, _) = setup(7, "machine-a");
        let wrong = EvaluationContext::new(
            context.evaluator_id.clone(),
            context.oracle_id.clone(),
            context.input_profile_id.clone(),
            context.environment_id.clone(),
            "different-revision",
            context.toolchain_profile.clone(),
            context.target_profile.clone(),
            context.seeds.clone(),
        ).unwrap();
        assert!(matches!(
            ContextBoundForgeBatch::from_trace(&run, &baseline, wrong, &trace, &observations),
            Err(ForgeContextLearningError::SourceRevisionMismatch)
        ));
    }
}
