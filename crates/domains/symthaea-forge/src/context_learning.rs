// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-context cross-run aggregation for Forge transformation search memory.
//!
//! This module deliberately starts with the narrowest defensible transfer rule: batches may be
//! combined only when they describe the same semantic problem, the same baseline implementation,
//! and the exact same [`EvaluationContext`]. Different discovery-run seeds may contribute repeated
//! searches; different evaluator/oracle/input/environment/toolchain/target/source contexts may not.
//! Local Forge selection remains search behavior, not replicated performance evidence.

use crate::learning::{
    ForgeLearningError, ForgeTrialBatch, TransformationOutcomeTable,
};
use crate::trace::ForgeTraceEvent;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::discovery::DiscoveryRun;
use symthaea_algorithms::evaluation::{EvaluationContext, EvaluationError};
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{ContentId, ImplementationId, ImplementationRecord, ProblemId, TransformationId};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeContextLearningError {
    #[error(transparent)]
    Learning(#[from] ForgeLearningError),
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
    #[error("evaluation context source revision does not match the discovery-run baseline revision")]
    SourceRevisionMismatch,
    #[error("exact-context cohort requires at least one batch")]
    EmptyCohort,
    #[error("context-bound batch identity does not match its canonical fields")]
    BatchIdentityMismatch,
    #[error("cohort contains a duplicate run-bound batch")]
    DuplicateBatch,
    #[error("cohort mixes semantic problem identities")]
    ProblemMismatch,
    #[error("cohort mixes baseline implementation identities")]
    BaselineMismatch,
    #[error("cohort mixes evaluation contexts")]
    ContextMismatch,
    #[error("transformation outcome count overflow")]
    CountOverflow,
    #[error("cross-run outcome row identity does not match its canonical fields")]
    RowIdentityMismatch,
    #[error("cross-run outcome table identity does not match its canonical fields")]
    TableIdentityMismatch,
}

/// One Forge search batch explicitly bound to the evaluation context under which future
/// evidence-grade comparison would occur.
///
/// This does not claim that Forge's own local benchmark is an `EvaluationReceipt`. It only prevents
/// learning consumers from aggregating search histories whose experimental contexts differ.
#[derive(Debug, Clone, Serialize)]
pub struct ContextBoundForgeBatch {
    id: ContentId,
    context_id: ContentId,
    context: EvaluationContext,
    batch: ForgeTrialBatch,
    outcome_table: TransformationOutcomeTable,
}

impl ContextBoundForgeBatch {
    #[allow(clippy::too_many_arguments)]
    pub fn from_trace(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        context: EvaluationContext,
        trace: &[ForgeTraceEvent],
        observations: &ObservationStore,
    ) -> Result<Self, ForgeContextLearningError> {
        run.validate().map_err(ForgeLearningError::from)?;
        context.validate()?;
        if context.source_revision != run.baseline_revision {
            return Err(ForgeContextLearningError::SourceRevisionMismatch);
        }
        let batch = ForgeTrialBatch::from_trace(run, baseline, trace, observations)?;
        let outcome_table = TransformationOutcomeTable::from_batch(&batch)?;
        let context_id = context.content_id();
        let id = derive_context_bound_batch_id(batch.id(), &context_id, outcome_table.id());
        Ok(Self {
            id,
            context_id,
            context,
            batch,
            outcome_table,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn context_id(&self) -> &ContentId {
        &self.context_id
    }

    pub fn context(&self) -> &EvaluationContext {
        &self.context
    }

    pub fn batch(&self) -> &ForgeTrialBatch {
        &self.batch
    }

    pub fn outcome_table(&self) -> &TransformationOutcomeTable {
        &self.outcome_table
    }

    pub fn validate(&self) -> Result<(), ForgeContextLearningError> {
        self.context.validate()?;
        if self.context.content_id() != self.context_id {
            return Err(ForgeContextLearningError::BatchIdentityMismatch);
        }
        let expected = derive_context_bound_batch_id(
            self.batch.id(),
            &self.context_id,
            self.outcome_table.id(),
        );
        if expected != self.id {
            return Err(ForgeContextLearningError::BatchIdentityMismatch);
        }
        Ok(())
    }
}

fn derive_context_bound_batch_id(
    batch_id: &ContentId,
    context_id: &ContentId,
    table_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-context-bound-batch.v1",
        [
            batch_id.as_str().as_bytes(),
            context_id.as_str().as_bytes(),
            table_id.as_str().as_bytes(),
        ],
    )
}

/// Canonical set of repeated search batches that are exactly comparable by the conservative v1
/// context rule.
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
        let mut seen = BTreeSet::new();
        for batch in &batches {
            if !seen.insert(batch.id.as_str()) {
                return Err(ForgeContextLearningError::DuplicateBatch);
            }
        }

        let first = &batches[0];
        let problem_id = first.batch.problem_id().clone();
        let baseline_implementation_id = first.batch.baseline_implementation_id().clone();
        let context_id = first.context_id.clone();
        for batch in &batches[1..] {
            if batch.batch.problem_id() != &problem_id {
                return Err(ForgeContextLearningError::ProblemMismatch);
            }
            if batch.batch.baseline_implementation_id() != &baseline_implementation_id {
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
    baseline_implementation_id: &ImplementationId,
    context_id: &ContentId,
    batches: &[ContextBoundForgeBatch],
) -> ContentId {
    let count = (batches.len() as u64).to_be_bytes();
    let mut parts = vec![
        problem_id.as_content_id().as_str().as_bytes().to_vec(),
        baseline_implementation_id
            .as_content_id()
            .as_str()
            .as_bytes()
            .to_vec(),
        context_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        batches
            .iter()
            .map(|batch| batch.id.as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-exact-context-cohort.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[derive(Debug, Clone, Default)]
struct Counts {
    trials: u64,
    rejected_compilation: u64,
    rejected_correctness: u64,
    rejected_evaluation: u64,
    valid_not_selected: u64,
    selected_for_continuation: u64,
    interrupted: u64,
}

impl Counts {
    fn add(
        &mut self,
        row: &crate::learning::TransformationOutcomeStats,
    ) -> Result<(), ForgeContextLearningError> {
        self.trials = checked(self.trials, row.trials())?;
        self.rejected_compilation = checked(self.rejected_compilation, row.rejected_compilation())?;
        self.rejected_correctness = checked(self.rejected_correctness, row.rejected_correctness())?;
        self.rejected_evaluation = checked(self.rejected_evaluation, row.rejected_evaluation())?;
        self.valid_not_selected = checked(self.valid_not_selected, row.valid_not_selected())?;
        self.selected_for_continuation = checked(
            self.selected_for_continuation,
            row.selected_for_continuation(),
        )?;
        self.interrupted = checked(self.interrupted, row.interrupted())?;
        Ok(())
    }
}

fn checked(a: u64, b: u64) -> Result<u64, ForgeContextLearningError> {
    a.checked_add(b)
        .ok_or(ForgeContextLearningError::CountOverflow)
}

/// Aggregate counts for one transformation across repeated searches under exactly one context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContextualTransformationStats {
    id: ContentId,
    transformation_id: TransformationId,
    trials: u64,
    rejected_compilation: u64,
    rejected_correctness: u64,
    rejected_evaluation: u64,
    valid_not_selected: u64,
    selected_for_continuation: u64,
    interrupted: u64,
}

impl ContextualTransformationStats {
    fn from_counts(transformation_id: TransformationId, counts: Counts) -> Self {
        let mut row = Self {
            id: ContentId::derive(
                "symthaea.forge-contextual-transformation-row.uninitialized",
                [b"v1".as_slice()],
            ),
            transformation_id,
            trials: counts.trials,
            rejected_compilation: counts.rejected_compilation,
            rejected_correctness: counts.rejected_correctness,
            rejected_evaluation: counts.rejected_evaluation,
            valid_not_selected: counts.valid_not_selected,
            selected_for_continuation: counts.selected_for_continuation,
            interrupted: counts.interrupted,
        };
        row.id = derive_contextual_row_id(&row);
        row
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn transformation_id(&self) -> &TransformationId {
        &self.transformation_id
    }

    pub fn trials(&self) -> u64 {
        self.trials
    }

    pub fn completed_trials(&self) -> u64 {
        self.trials - self.interrupted
    }

    pub fn rejected_compilation(&self) -> u64 {
        self.rejected_compilation
    }

    pub fn rejected_correctness(&self) -> u64 {
        self.rejected_correctness
    }

    pub fn rejected_evaluation(&self) -> u64 {
        self.rejected_evaluation
    }

    pub fn valid_not_selected(&self) -> u64 {
        self.valid_not_selected
    }

    pub fn selected_for_continuation(&self) -> u64 {
        self.selected_for_continuation
    }

    pub fn interrupted(&self) -> u64 {
        self.interrupted
    }

    pub fn compile_survival_rate(&self) -> Option<f64> {
        let completed = self.completed_trials();
        ratio(completed - self.rejected_compilation, completed)
    }

    pub fn correctness_survival_rate(&self) -> Option<f64> {
        let entered = self.completed_trials() - self.rejected_compilation;
        ratio(entered - self.rejected_correctness, entered)
    }

    pub fn local_selection_rate(&self) -> Option<f64> {
        ratio(
            self.selected_for_continuation,
            self.valid_not_selected + self.selected_for_continuation,
        )
    }

    pub fn interruption_rate(&self) -> Option<f64> {
        ratio(self.interrupted, self.trials)
    }

    pub fn validate(&self) -> Result<(), ForgeContextLearningError> {
        let classified = self
            .rejected_compilation
            .checked_add(self.rejected_correctness)
            .and_then(|value| value.checked_add(self.rejected_evaluation))
            .and_then(|value| value.checked_add(self.valid_not_selected))
            .and_then(|value| value.checked_add(self.selected_for_continuation))
            .and_then(|value| value.checked_add(self.interrupted))
            .ok_or(ForgeContextLearningError::CountOverflow)?;
        if classified != self.trials || self.interrupted > self.trials {
            return Err(ForgeContextLearningError::RowIdentityMismatch);
        }
        if derive_contextual_row_id(self) != self.id {
            return Err(ForgeContextLearningError::RowIdentityMismatch);
        }
        Ok(())
    }
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn derive_contextual_row_id(row: &ContextualTransformationStats) -> ContentId {
    let counts = [
        row.trials,
        row.rejected_compilation,
        row.rejected_correctness,
        row.rejected_evaluation,
        row.valid_not_selected,
        row.selected_for_continuation,
        row.interrupted,
    ];
    let mut parts = vec![
        row.transformation_id
            .as_content_id()
            .as_str()
            .as_bytes()
            .to_vec(),
    ];
    parts.extend(counts.into_iter().map(|count| count.to_be_bytes().to_vec()));
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
            for row in batch.outcome_table().rows() {
                grouped
                    .entry(row.transformation_id().clone())
                    .or_default()
                    .add(row)?;
            }
        }
        let rows = grouped
            .into_iter()
            .map(|(transformation_id, counts)| {
                ContextualTransformationStats::from_counts(transformation_id, counts)
            })
            .collect::<Vec<_>>();
        for row in &rows {
            row.validate()?;
        }
        let id = derive_contextual_table_id(cohort.id(), &rows);
        Ok(Self {
            id,
            cohort_id: cohort.id().clone(),
            rows,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn cohort_id(&self) -> &ContentId {
        &self.cohort_id
    }

    pub fn rows(&self) -> &[ContextualTransformationStats] {
        &self.rows
    }

    pub fn get(&self, transformation_id: &TransformationId) -> Option<&ContextualTransformationStats> {
        self.rows
            .binary_search_by(|row| row.transformation_id.cmp(transformation_id))
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
        if derive_contextual_table_id(&self.cohort_id, &self.rows) != self.id {
            return Err(ForgeContextLearningError::TableIdentityMismatch);
        }
        Ok(())
    }
}

fn derive_contextual_table_id(
    cohort_id: &ContentId,
    rows: &[ContextualTransformationStats],
) -> ContentId {
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
    use symthaea_algorithms::observation::ObservationStore;
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, DeterminismRequirement, DiscoveryRisk, ProblemSpec,
        SemanticGuarantee,
    };

    fn setup(seed: u64) -> (
        DiscoveryRun,
        ImplementationRecord,
        EvaluationContext,
        Vec<ForgeTraceEvent>,
        ObservationStore,
        TransformationId,
    ) {
        let problem = ProblemSpec::new(
            "cross-run-test",
            "Return exact reference value.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "baseline",
            "test baseline",
            AlgorithmProvenance::HumanAuthored,
        )
        .unwrap();
        let baseline_artifact = full_source_artifact_id("fn f() -> i32 { 1 }\n");
        let baseline = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id,
            "repo://src/f.rs",
            baseline_artifact.clone(),
            None,
        )
        .unwrap();
        let run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            ContentId::derive("generator", [b"forge".as_slice()]),
            "abc123",
            SearchBudget::new(10, 1, 10).unwrap(),
            seed,
        )
        .unwrap();
        let context = EvaluationContext::new(
            ContentId::derive("evaluator", [b"criterion".as_slice()]),
            ContentId::derive("oracle", [b"reference".as_slice()]),
            ContentId::derive("input", [b"fixed-corpus".as_slice()]),
            ContentId::derive("environment", [b"machine-a".as_slice()]),
            "abc123",
            "rustc-1.96.0",
            "x86_64-v3",
            vec![1, 2, 3],
        )
        .unwrap();
        let candidate_artifact = full_source_artifact_id("fn f() -> i32 { 2 }\n");
        let attempt = ForgeAttemptId::derive(&baseline_artifact, seed, 0, 0);
        let mutation = MutationRecord::new(
            0,
            "literal",
            "1 -> 2",
            baseline_artifact,
            candidate_artifact,
        );
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let gates = vec![
            GateResult {
                gate: Gate::Compile,
                passed: true,
                output_tail: String::new(),
                duration: Duration::from_nanos(1),
            },
            GateResult {
                gate: Gate::Test,
                passed: false,
                output_tail: "counterexample".into(),
                duration: Duration::from_nanos(1),
            },
        ];
        let rejected = forge_observations::gates(&attempt, &mutation, &gates).unwrap();
        let summary = forge_observations::search_summary(1, 0, 0, 1, 0, 0, 0, None, None).unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt.clone(),
                0,
                DiscoveryEventKind::CandidateGenerated,
                mutation.candidate_artifact_id.clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt,
                0,
                DiscoveryEventKind::RejectedCorrectness,
                mutation.candidate_artifact_id.clone(),
                rejected.id().clone(),
            ),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![generated, rejected, summary]).unwrap();
        (
            run,
            baseline,
            context,
            trace,
            observations,
            mutation.transformation_id,
        )
    }

    #[test]
    fn repeated_search_seeds_can_aggregate_under_identical_context() {
        let (run_a, baseline_a, context_a, trace_a, observations_a, transformation) = setup(7);
        let (run_b, baseline_b, context_b, trace_b, observations_b, _) = setup(8);
        assert_eq!(baseline_a.id, baseline_b.id);
        assert_eq!(context_a.content_id(), context_b.content_id());
        assert_ne!(run_a.id, run_b.id);

        let a = ContextBoundForgeBatch::from_trace(
            &run_a,
            &baseline_a,
            context_a,
            &trace_a,
            &observations_a,
        )
        .unwrap();
        let b = ContextBoundForgeBatch::from_trace(
            &run_b,
            &baseline_b,
            context_b,
            &trace_b,
            &observations_b,
        )
        .unwrap();
        let cohort = ExactContextForgeCohort::new(vec![a, b]).unwrap();
        let table = ExactContextTransformationTable::from_cohort(&cohort).unwrap();
        let row = table.get(&transformation).unwrap();
        assert_eq!(row.trials(), 2);
        assert_eq!(row.rejected_correctness(), 2);
        assert_eq!(row.correctness_survival_rate(), Some(0.0));
        assert!(table.validate().is_ok());
    }

    #[test]
    fn environment_change_prevents_cross_run_aggregation() {
        let (run_a, baseline_a, context_a, trace_a, observations_a, _) = setup(7);
        let (run_b, baseline_b, mut context_b, trace_b, observations_b, _) = setup(8);
        context_b = EvaluationContext::new(
            context_b.evaluator_id.clone(),
            context_b.oracle_id.clone(),
            context_b.input_profile_id.clone(),
            ContentId::derive("environment", [b"machine-b".as_slice()]),
            context_b.source_revision.clone(),
            context_b.toolchain_profile.clone(),
            context_b.target_profile.clone(),
            context_b.seeds.clone(),
        )
        .unwrap();

        let a = ContextBoundForgeBatch::from_trace(
            &run_a,
            &baseline_a,
            context_a,
            &trace_a,
            &observations_a,
        )
        .unwrap();
        let b = ContextBoundForgeBatch::from_trace(
            &run_b,
            &baseline_b,
            context_b,
            &trace_b,
            &observations_b,
        )
        .unwrap();
        assert!(matches!(
            ExactContextForgeCohort::new(vec![a, b]),
            Err(ForgeContextLearningError::ContextMismatch)
        ));
    }

    #[test]
    fn evaluation_context_must_describe_the_frozen_source_revision() {
        let (run, baseline, context, trace, observations, _) = setup(7);
        let wrong = EvaluationContext::new(
            context.evaluator_id.clone(),
            context.oracle_id.clone(),
            context.input_profile_id.clone(),
            context.environment_id.clone(),
            "different-revision",
            context.toolchain_profile.clone(),
            context.target_profile.clone(),
            context.seeds.clone(),
        )
        .unwrap();
        assert!(matches!(
            ContextBoundForgeBatch::from_trace(&run, &baseline, wrong, &trace, &observations),
            Err(ForgeContextLearningError::SourceRevisionMismatch)
        ));
    }
}
