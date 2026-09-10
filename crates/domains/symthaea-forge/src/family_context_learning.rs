// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-context repeated-search aggregation for Forge transformation families.
//!
//! This layer composes the concrete exact-context theorem with generator-scoped family identity.
//! A family cohort requires at least two distinct `DiscoveryRun`s and refuses to mix semantic
//! problem, baseline implementation, evaluation context, or generator identity. Family rows record
//! both trial counts and distinct-run coverage so repeated attempts inside one run cannot masquerade
//! as breadth across independent search seeds.
//!
//! These records remain descriptive search memory. Forge-local continuation selection is not
//! replicated superiority evidence, promotion eligibility, or runtime authority.

use crate::context_learning::{ContextBoundForgeBatch, ExactContextForgeCohort, ForgeContextLearningError};
use crate::family_learning::{
    ForgeFamilyLearningError, ForgeFamilyOutcomeStats, ForgeFamilyOutcomeTable, ForgeFamilyTrialSet,
    ForgeTransformationFamilyId,
};
use crate::trace::ForgeTraceEvent;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::discovery::DiscoveryRun;
use symthaea_algorithms::evaluation::EvaluationContext;
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{ContentId, ImplementationId, ImplementationRecord, ProblemId};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeFamilyContextError {
    #[error(transparent)]
    Context(#[from] ForgeContextLearningError),
    #[error(transparent)]
    Family(#[from] ForgeFamilyLearningError),
    #[error("family cohort requires at least two distinct discovery runs")]
    InsufficientRuns,
    #[error("context-bound family batch identity does not match canonical fields")]
    BatchIdentityMismatch,
    #[error("family cohort contains a duplicate family batch")]
    DuplicateBatch,
    #[error("family cohort mixes generator identities")]
    GeneratorMismatch,
    #[error("family cohort identity does not match canonical fields")]
    CohortIdentityMismatch,
    #[error("family cohort outcome count overflow")]
    CountOverflow,
    #[error("family cohort outcome row is internally inconsistent")]
    InvalidCounts,
    #[error("family cohort outcome row identity does not match canonical fields")]
    RowIdentityMismatch,
    #[error("family cohort outcome table identity does not match canonical fields")]
    TableIdentityMismatch,
}

/// One exact-context Forge batch plus its generator-scoped family projection.
#[derive(Debug, Clone, Serialize)]
pub struct ContextBoundForgeFamilyBatch {
    id: ContentId,
    generator_id: ContentId,
    concrete: ContextBoundForgeBatch,
    family_trials: ForgeFamilyTrialSet,
    family_table: ForgeFamilyOutcomeTable,
}

impl ContextBoundForgeFamilyBatch {
    pub fn from_trace(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        context: EvaluationContext,
        trace: &[ForgeTraceEvent],
        observations: &ObservationStore,
    ) -> Result<Self, ForgeFamilyContextError> {
        let concrete = ContextBoundForgeBatch::from_trace(
            run,
            baseline,
            context,
            trace,
            observations,
        )?;
        let family_trials = ForgeFamilyTrialSet::from_trace(run, baseline, trace, observations)?;
        let family_table = ForgeFamilyOutcomeTable::from_trials(&family_trials)?;
        let generator_id = run.generator_id.clone();
        let id = derive_batch_id(
            concrete.id(),
            &generator_id,
            family_trials.id(),
            family_table.id(),
        );
        let batch = Self {
            id,
            generator_id,
            concrete,
            family_trials,
            family_table,
        };
        batch.validate()?;
        Ok(batch)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { self.concrete.run_id() }
    pub fn problem_id(&self) -> &ProblemId { self.concrete.problem_id() }
    pub fn baseline_implementation_id(&self) -> &ImplementationId {
        self.concrete.baseline_implementation_id()
    }
    pub fn context_id(&self) -> &ContentId { self.concrete.context_id() }
    pub fn generator_id(&self) -> &ContentId { &self.generator_id }
    pub fn concrete(&self) -> &ContextBoundForgeBatch { &self.concrete }
    pub fn family_trials(&self) -> &ForgeFamilyTrialSet { &self.family_trials }
    pub fn family_table(&self) -> &ForgeFamilyOutcomeTable { &self.family_table }

    pub fn validate(&self) -> Result<(), ForgeFamilyContextError> {
        self.concrete.validate()?;
        self.family_trials.validate()?;
        self.family_table.validate()?;
        if self.family_trials.run_id() != self.concrete.run_id()
            || self.family_trials.generator_id() != &self.generator_id
            || self.family_table.family_trial_set_id() != self.family_trials.id()
        {
            return Err(ForgeFamilyContextError::BatchIdentityMismatch);
        }
        let expected = derive_batch_id(
            self.concrete.id(),
            &self.generator_id,
            self.family_trials.id(),
            self.family_table.id(),
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeFamilyContextError::BatchIdentityMismatch)
        }
    }
}

fn derive_batch_id(
    concrete_batch_id: &ContentId,
    generator_id: &ContentId,
    family_trial_set_id: &ContentId,
    family_table_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-context-bound-family-batch.v1",
        [
            concrete_batch_id.as_str().as_bytes(),
            generator_id.as_str().as_bytes(),
            family_trial_set_id.as_str().as_bytes(),
            family_table_id.as_str().as_bytes(),
        ],
    )
}

/// Canonical cohort of at least two distinct runs under one exact comparison context and generator.
#[derive(Debug, Clone, Serialize)]
pub struct ExactContextForgeFamilyCohort {
    id: ContentId,
    context_cohort_id: ContentId,
    problem_id: ProblemId,
    baseline_implementation_id: ImplementationId,
    context_id: ContentId,
    generator_id: ContentId,
    batches: Vec<ContextBoundForgeFamilyBatch>,
}

impl ExactContextForgeFamilyCohort {
    pub fn new(
        mut batches: Vec<ContextBoundForgeFamilyBatch>,
    ) -> Result<Self, ForgeFamilyContextError> {
        if batches.len() < 2 {
            return Err(ForgeFamilyContextError::InsufficientRuns);
        }
        for batch in &batches {
            batch.validate()?;
        }
        batches.sort_by(|a, b| a.id.cmp(&b.id));
        let mut seen = BTreeSet::new();
        for batch in &batches {
            if !seen.insert(batch.id.as_str().to_string()) {
                return Err(ForgeFamilyContextError::DuplicateBatch);
            }
        }

        // Reuse the lower-level theorem for exact problem/baseline/context equality and unique runs.
        let concrete_cohort = ExactContextForgeCohort::new(
            batches.iter().map(|batch| batch.concrete.clone()).collect(),
        )?;
        let generator_id = batches[0].generator_id.clone();
        if batches
            .iter()
            .any(|batch| batch.generator_id != generator_id)
        {
            return Err(ForgeFamilyContextError::GeneratorMismatch);
        }

        let problem_id = concrete_cohort.problem_id().clone();
        let baseline_implementation_id = concrete_cohort.baseline_implementation_id().clone();
        let context_id = concrete_cohort.context_id().clone();
        let context_cohort_id = concrete_cohort.id().clone();
        let id = derive_cohort_id(
            &context_cohort_id,
            &problem_id,
            &baseline_implementation_id,
            &context_id,
            &generator_id,
            &batches,
        );
        let cohort = Self {
            id,
            context_cohort_id,
            problem_id,
            baseline_implementation_id,
            context_id,
            generator_id,
            batches,
        };
        cohort.validate()?;
        Ok(cohort)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn context_cohort_id(&self) -> &ContentId { &self.context_cohort_id }
    pub fn problem_id(&self) -> &ProblemId { &self.problem_id }
    pub fn baseline_implementation_id(&self) -> &ImplementationId {
        &self.baseline_implementation_id
    }
    pub fn context_id(&self) -> &ContentId { &self.context_id }
    pub fn generator_id(&self) -> &ContentId { &self.generator_id }
    pub fn batches(&self) -> &[ContextBoundForgeFamilyBatch] { &self.batches }
    pub fn run_count(&self) -> u64 { self.batches.len() as u64 }

    pub fn validate(&self) -> Result<(), ForgeFamilyContextError> {
        let rebuilt = Self::new(self.batches.clone())?;
        if rebuilt.id == self.id
            && rebuilt.context_cohort_id == self.context_cohort_id
            && rebuilt.problem_id == self.problem_id
            && rebuilt.baseline_implementation_id == self.baseline_implementation_id
            && rebuilt.context_id == self.context_id
            && rebuilt.generator_id == self.generator_id
        {
            Ok(())
        } else {
            Err(ForgeFamilyContextError::CohortIdentityMismatch)
        }
    }
}

fn derive_cohort_id(
    context_cohort_id: &ContentId,
    problem_id: &ProblemId,
    baseline_id: &ImplementationId,
    context_id: &ContentId,
    generator_id: &ContentId,
    batches: &[ContextBoundForgeFamilyBatch],
) -> ContentId {
    let count = (batches.len() as u64).to_be_bytes();
    let mut parts = vec![
        context_cohort_id.as_str().as_bytes().to_vec(),
        problem_id.as_content_id().as_str().as_bytes().to_vec(),
        baseline_id.as_content_id().as_str().as_bytes().to_vec(),
        context_id.as_str().as_bytes().to_vec(),
        generator_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        batches
            .iter()
            .map(|batch| batch.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-exact-context-family-cohort.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[derive(Debug, Clone, Default)]
struct Counts {
    runs_observed: u64,
    trials: u64,
    compile: u64,
    correctness: u64,
    evaluation: u64,
    not_selected: u64,
    selected: u64,
    interrupted: u64,
}

impl Counts {
    fn add_run_row(&mut self, row: &ForgeFamilyOutcomeStats) -> Result<(), ForgeFamilyContextError> {
        self.runs_observed = add(self.runs_observed, 1)?;
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

fn add(left: u64, right: u64) -> Result<u64, ForgeFamilyContextError> {
    left.checked_add(right)
        .ok_or(ForgeFamilyContextError::CountOverflow)
}

/// Outcome counts for one generator-scoped family across an exact-context repeated-search cohort.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContextualForgeFamilyStats {
    id: ContentId,
    family_id: ForgeTransformationFamilyId,
    cohort_runs: u64,
    counts: [u64; 8],
}

impl ContextualForgeFamilyStats {
    fn from_counts(
        family_id: ForgeTransformationFamilyId,
        cohort_runs: u64,
        counts: Counts,
    ) -> Result<Self, ForgeFamilyContextError> {
        let mut row = Self {
            id: ContentId::derive("symthaea.forge-context-family-row.uninitialized", [b"v1".as_slice()]),
            family_id,
            cohort_runs,
            counts: [
                counts.runs_observed,
                counts.trials,
                counts.compile,
                counts.correctness,
                counts.evaluation,
                counts.not_selected,
                counts.selected,
                counts.interrupted,
            ],
        };
        row.id = derive_row_id(&row);
        row.validate()?;
        Ok(row)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn family_id(&self) -> &ForgeTransformationFamilyId { &self.family_id }
    pub fn cohort_runs(&self) -> u64 { self.cohort_runs }
    pub fn runs_observed(&self) -> u64 { self.counts[0] }
    pub fn trials(&self) -> u64 { self.counts[1] }
    pub fn rejected_compilation(&self) -> u64 { self.counts[2] }
    pub fn rejected_correctness(&self) -> u64 { self.counts[3] }
    pub fn rejected_evaluation(&self) -> u64 { self.counts[4] }
    pub fn valid_not_selected(&self) -> u64 { self.counts[5] }
    pub fn selected_for_continuation(&self) -> u64 { self.counts[6] }
    pub fn interrupted(&self) -> u64 { self.counts[7] }
    pub fn completed_trials(&self) -> u64 { self.trials() - self.interrupted() }

    pub fn run_coverage_rate(&self) -> Option<f64> {
        ratio(self.runs_observed(), self.cohort_runs)
    }

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

    pub fn validate(&self) -> Result<(), ForgeFamilyContextError> {
        self.family_id.validate()?;
        let classified = self.counts[2..]
            .iter()
            .try_fold(0u64, |sum, value| add(sum, *value))?;
        if self.cohort_runs < 2
            || self.runs_observed() == 0
            || self.runs_observed() > self.cohort_runs
            || classified != self.trials()
            || self.interrupted() > self.trials()
        {
            return Err(ForgeFamilyContextError::InvalidCounts);
        }
        if derive_row_id(self) == self.id {
            Ok(())
        } else {
            Err(ForgeFamilyContextError::RowIdentityMismatch)
        }
    }
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn derive_row_id(row: &ContextualForgeFamilyStats) -> ContentId {
    let mut parts = vec![
        row.family_id.as_content_id().as_str().as_bytes().to_vec(),
        row.cohort_runs.to_be_bytes().to_vec(),
    ];
    parts.extend(row.counts.iter().map(|value| value.to_be_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-context-family-outcome-row.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Family-level statistics across distinct search runs that satisfy the exact compatibility theorem.
#[derive(Debug, Clone, Serialize)]
pub struct ExactContextForgeFamilyOutcomeTable {
    id: ContentId,
    cohort_id: ContentId,
    cohort_runs: u64,
    rows: Vec<ContextualForgeFamilyStats>,
}

impl ExactContextForgeFamilyOutcomeTable {
    pub fn from_cohort(
        cohort: &ExactContextForgeFamilyCohort,
    ) -> Result<Self, ForgeFamilyContextError> {
        cohort.validate()?;
        let mut grouped = BTreeMap::<ForgeTransformationFamilyId, Counts>::new();
        for batch in cohort.batches() {
            for row in batch.family_table().rows() {
                grouped
                    .entry(row.family_id().clone())
                    .or_default()
                    .add_run_row(row)?;
            }
        }
        let cohort_runs = cohort.run_count();
        let rows = grouped
            .into_iter()
            .map(|(family, counts)| ContextualForgeFamilyStats::from_counts(family, cohort_runs, counts))
            .collect::<Result<Vec<_>, _>>()?;
        let id = derive_table_id(cohort.id(), cohort_runs, &rows);
        let table = Self {
            id,
            cohort_id: cohort.id().clone(),
            cohort_runs,
            rows,
        };
        table.validate()?;
        Ok(table)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn cohort_id(&self) -> &ContentId { &self.cohort_id }
    pub fn cohort_runs(&self) -> u64 { self.cohort_runs }
    pub fn rows(&self) -> &[ContextualForgeFamilyStats] { &self.rows }

    pub fn get(&self, family: &ForgeTransformationFamilyId) -> Option<&ContextualForgeFamilyStats> {
        self.rows
            .binary_search_by(|row| row.family_id().cmp(family))
            .ok()
            .map(|index| &self.rows[index])
    }

    pub fn validate(&self) -> Result<(), ForgeFamilyContextError> {
        if self.cohort_runs < 2 {
            return Err(ForgeFamilyContextError::InvalidCounts);
        }
        let mut previous: Option<&ForgeTransformationFamilyId> = None;
        for row in &self.rows {
            row.validate()?;
            if row.cohort_runs() != self.cohort_runs
                || previous.is_some_and(|prior| prior >= row.family_id())
            {
                return Err(ForgeFamilyContextError::TableIdentityMismatch);
            }
            previous = Some(row.family_id());
        }
        if derive_table_id(&self.cohort_id, self.cohort_runs, &self.rows) == self.id {
            Ok(())
        } else {
            Err(ForgeFamilyContextError::TableIdentityMismatch)
        }
    }
}

fn derive_table_id(
    cohort_id: &ContentId,
    cohort_runs: u64,
    rows: &[ContextualForgeFamilyStats],
) -> ContentId {
    let count = (rows.len() as u64).to_be_bytes();
    let runs = cohort_runs.to_be_bytes();
    let mut parts = vec![
        cohort_id.as_str().as_bytes().to_vec(),
        runs.to_vec(),
        count.to_vec(),
    ];
    parts.extend(rows.iter().map(|row| row.id().as_str().as_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-exact-context-family-outcome-table.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{full_source_artifact_id, MutationRecord};
    use crate::fitness::{Gate, GateResult};
    use crate::observations as forge_observations;
    use crate::trace::ForgeAttemptId;
    use std::time::Duration;
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::ledger::DiscoveryEventKind;
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, DeterminismRequirement, DiscoveryRisk, ProblemSpec,
        SemanticGuarantee,
    };

    struct Fixture {
        problem: ProblemSpec,
        baseline: ImplementationRecord,
        generator: ContentId,
        context: EvaluationContext,
    }

    fn fixture() -> Fixture {
        let problem = ProblemSpec::new(
            "family-context-test",
            "Return exact reference value.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        ).unwrap();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "baseline",
            "family context baseline",
            AlgorithmProvenance::HumanAuthored,
        ).unwrap();
        let baseline_artifact = full_source_artifact_id("fn f() -> i32 { 1 }\n");
        let baseline = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id,
            "repo://src/f.rs",
            baseline_artifact,
            Some("x86_64-test".into()),
        ).unwrap();
        let generator = ContentId::derive("generator", [b"forge-v1".as_slice()]);
        let context = EvaluationContext::new(
            ContentId::derive("evaluator", [b"v1".as_slice()]),
            ContentId::derive("oracle", [b"v1".as_slice()]),
            ContentId::derive("inputs", [b"v1".as_slice()]),
            ContentId::derive("environment", [b"nix-v1".as_slice()]),
            "abc123",
            "rust-1.96.0",
            "x86_64-test",
            vec![1, 2, 3],
        ).unwrap();
        Fixture { problem, baseline, generator, context }
    }

    fn run_and_trace(
        fixture: &Fixture,
        seed: u64,
        candidate_value: i32,
        generator: ContentId,
    ) -> (DiscoveryRun, Vec<ForgeTraceEvent>, ObservationStore) {
        let run = DiscoveryRun::new(
            &fixture.problem,
            DiscoveryPolicy::default(),
            generator,
            "abc123",
            SearchBudget::new(10, 1, 10).unwrap(),
            seed,
        ).unwrap();
        let candidate_source = format!("fn f() -> i32 {{ {candidate_value} }}\n");
        let candidate_artifact = full_source_artifact_id(&candidate_source);
        let mutation = MutationRecord::new(
            0,
            "NumericLiteralPerturb",
            format!("1 -> {candidate_value}"),
            fixture.baseline.artifact_id.clone(),
            candidate_artifact.clone(),
        );
        let attempt = ForgeAttemptId::derive(&fixture.baseline.artifact_id, seed, 0, 0);
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let gates = vec![GateResult {
            gate: Gate::Compile,
            passed: false,
            output_tail: "compile error".into(),
            duration: Duration::from_nanos(1),
        }];
        let rejected = forge_observations::gates(&attempt, &mutation, &gates).unwrap();
        let summary = forge_observations::search_summary(1, 0, 1, 0, 0, 0, 0, None, None).unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt.clone(),
                0,
                DiscoveryEventKind::CandidateGenerated,
                candidate_artifact.clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt,
                0,
                DiscoveryEventKind::RejectedCompilation,
                candidate_artifact,
                rejected.id().clone(),
            ),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![generated, rejected, summary]).unwrap();
        (run, trace, observations)
    }

    #[test]
    fn same_family_across_distinct_search_seeds_aggregates_with_run_coverage() {
        let f = fixture();
        let (run_a, trace_a, obs_a) = run_and_trace(&f, 7, 2, f.generator.clone());
        let (run_b, trace_b, obs_b) = run_and_trace(&f, 8, 3, f.generator.clone());
        let batch_a = ContextBoundForgeFamilyBatch::from_trace(
            &run_a, &f.baseline, f.context.clone(), &trace_a, &obs_a,
        ).unwrap();
        let batch_b = ContextBoundForgeFamilyBatch::from_trace(
            &run_b, &f.baseline, f.context.clone(), &trace_b, &obs_b,
        ).unwrap();
        let cohort = ExactContextForgeFamilyCohort::new(vec![batch_a, batch_b]).unwrap();
        let table = ExactContextForgeFamilyOutcomeTable::from_cohort(&cohort).unwrap();
        assert_eq!(cohort.run_count(), 2);
        assert_eq!(table.rows().len(), 1);
        let row = &table.rows()[0];
        assert_eq!(row.runs_observed(), 2);
        assert_eq!(row.cohort_runs(), 2);
        assert_eq!(row.trials(), 2);
        assert_eq!(row.rejected_compilation(), 2);
        assert_eq!(row.run_coverage_rate(), Some(1.0));
        assert!(table.validate().is_ok());
    }

    #[test]
    fn one_run_is_not_a_repeated_search_cohort() {
        let f = fixture();
        let (run, trace, obs) = run_and_trace(&f, 7, 2, f.generator.clone());
        let batch = ContextBoundForgeFamilyBatch::from_trace(
            &run, &f.baseline, f.context.clone(), &trace, &obs,
        ).unwrap();
        assert!(matches!(
            ExactContextForgeFamilyCohort::new(vec![batch]),
            Err(ForgeFamilyContextError::InsufficientRuns)
        ));
    }

    #[test]
    fn generator_versions_cannot_share_family_cohort() {
        let f = fixture();
        let other_generator = ContentId::derive("generator", [b"forge-v2".as_slice()]);
        let (run_a, trace_a, obs_a) = run_and_trace(&f, 7, 2, f.generator.clone());
        let (run_b, trace_b, obs_b) = run_and_trace(&f, 8, 3, other_generator);
        let batch_a = ContextBoundForgeFamilyBatch::from_trace(
            &run_a, &f.baseline, f.context.clone(), &trace_a, &obs_a,
        ).unwrap();
        let batch_b = ContextBoundForgeFamilyBatch::from_trace(
            &run_b, &f.baseline, f.context.clone(), &trace_b, &obs_b,
        ).unwrap();
        assert!(matches!(
            ExactContextForgeFamilyCohort::new(vec![batch_a, batch_b]),
            Err(ForgeFamilyContextError::GeneratorMismatch)
        ));
    }
}
