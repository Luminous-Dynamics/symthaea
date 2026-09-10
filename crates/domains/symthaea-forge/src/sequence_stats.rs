// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-context statistics over generation-conditioned Forge transformation-family trials.
//!
//! This module is descriptive only. It aggregates sequence memory across repeated searches after
//! the exact problem/baseline/evaluation-context/generator compatibility theorem has passed. The
//! history order is explicit and content-addressed so first-order and higher-order projections
//! cannot be mixed accidentally.

use crate::family_context_learning::{
    ContextBoundForgeFamilyBatch, ExactContextForgeFamilyCohort, ForgeFamilyContextError,
};
use crate::family_learning::ForgeTransformationFamilyId;
use crate::sequence_learning::{
    ForgeConditionedFamilyTrial, ForgeFamilyHistory, ForgeSearchFamilySequence,
    ForgeSequenceLearningError,
};
use crate::trace::ForgeTraceEvent;
use crate::trials::ForgeTrialOutcome;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::discovery::DiscoveryRun;
use symthaea_algorithms::evaluation::EvaluationContext;
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{ContentId, ImplementationRecord};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeSequenceStatsError {
    #[error(transparent)]
    FamilyContext(#[from] ForgeFamilyContextError),
    #[error(transparent)]
    Sequence(#[from] ForgeSequenceLearningError),
    #[error("history order must be greater than zero")]
    ZeroHistoryOrder,
    #[error("context-bound sequence batch identity does not match canonical fields")]
    BatchIdentityMismatch,
    #[error("sequence cohort contains a duplicate batch")]
    DuplicateBatch,
    #[error("sequence cohort identity does not match canonical fields")]
    CohortIdentityMismatch,
    #[error("conditional family key mixes generator identities")]
    KeyGeneratorMismatch,
    #[error("conditional family key identity does not match canonical fields")]
    KeyIdentityMismatch,
    #[error("conditional family outcome count overflow")]
    CountOverflow,
    #[error("conditional family outcome row is internally inconsistent")]
    InvalidCounts,
    #[error("conditional family outcome row identity does not match canonical fields")]
    RowIdentityMismatch,
    #[error("conditional family outcome table identity does not match canonical fields")]
    TableIdentityMismatch,
}

/// Explicit finite-order history projection used for conditional family statistics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeHistoryOrder {
    id: ContentId,
    depth: u16,
}

impl ForgeHistoryOrder {
    pub fn new(depth: u16) -> Result<Self, ForgeSequenceStatsError> {
        if depth == 0 {
            return Err(ForgeSequenceStatsError::ZeroHistoryOrder);
        }
        let bytes = depth.to_be_bytes();
        Ok(Self {
            id: ContentId::derive(
                "symthaea.forge-history-order.v1",
                [bytes.as_slice()],
            ),
            depth,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn depth(&self) -> u16 { self.depth }

    pub fn validate(&self) -> Result<(), ForgeSequenceStatsError> {
        let rebuilt = Self::new(self.depth)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeSequenceStatsError::KeyIdentityMismatch)
        }
    }
}

/// Exact-context family batch paired with its generation-boundary sequence reconstruction.
#[derive(Debug, Clone, Serialize)]
pub struct ContextBoundForgeSequenceBatch {
    id: ContentId,
    family_batch: ContextBoundForgeFamilyBatch,
    sequence: ForgeSearchFamilySequence,
}

impl ContextBoundForgeSequenceBatch {
    pub fn from_trace(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        context: EvaluationContext,
        trace: &[ForgeTraceEvent],
        observations: &ObservationStore,
    ) -> Result<Self, ForgeSequenceStatsError> {
        let family_batch = ContextBoundForgeFamilyBatch::from_trace(
            run,
            baseline,
            context,
            trace,
            observations,
        )?;
        let sequence = ForgeSearchFamilySequence::from_trace(run, baseline, trace, observations)?;
        if family_batch.run_id() != sequence.run_id()
            || family_batch.generator_id() != sequence.generator_id()
            || family_batch.family_trials().id() != sequence.source_family_trial_set_id()
            || family_batch.family_trials().source_batch_id() != sequence.source_batch_id()
        {
            return Err(ForgeSequenceStatsError::BatchIdentityMismatch);
        }
        let id = derive_sequence_batch_id(family_batch.id(), sequence.id());
        let batch = Self {
            id,
            family_batch,
            sequence,
        };
        batch.validate()?;
        Ok(batch)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { self.family_batch.run_id() }
    pub fn family_batch(&self) -> &ContextBoundForgeFamilyBatch { &self.family_batch }
    pub fn sequence(&self) -> &ForgeSearchFamilySequence { &self.sequence }

    pub fn validate(&self) -> Result<(), ForgeSequenceStatsError> {
        self.family_batch.validate()?;
        self.sequence.validate()?;
        if self.family_batch.run_id() != self.sequence.run_id()
            || self.family_batch.generator_id() != self.sequence.generator_id()
            || self.family_batch.family_trials().id() != self.sequence.source_family_trial_set_id()
            || self.family_batch.family_trials().source_batch_id() != self.sequence.source_batch_id()
            || derive_sequence_batch_id(self.family_batch.id(), self.sequence.id()) != self.id
        {
            Err(ForgeSequenceStatsError::BatchIdentityMismatch)
        } else {
            Ok(())
        }
    }
}

fn derive_sequence_batch_id(
    family_batch_id: &ContentId,
    sequence_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-context-bound-sequence-batch.v1",
        [
            family_batch_id.as_str().as_bytes(),
            sequence_id.as_str().as_bytes(),
        ],
    )
}

/// Exact-context repeated-search cohort whose members also have validated sequence memory.
#[derive(Debug, Clone, Serialize)]
pub struct ExactContextForgeSequenceCohort {
    id: ContentId,
    family_cohort_id: ContentId,
    batches: Vec<ContextBoundForgeSequenceBatch>,
}

impl ExactContextForgeSequenceCohort {
    pub fn new(
        mut batches: Vec<ContextBoundForgeSequenceBatch>,
    ) -> Result<Self, ForgeSequenceStatsError> {
        for batch in &batches {
            batch.validate()?;
        }
        batches.sort_by(|a, b| a.id.cmp(&b.id));
        let mut seen = BTreeSet::new();
        for batch in &batches {
            if !seen.insert(batch.id.as_str().to_string()) {
                return Err(ForgeSequenceStatsError::DuplicateBatch);
            }
        }
        let family_cohort = ExactContextForgeFamilyCohort::new(
            batches
                .iter()
                .map(|batch| batch.family_batch.clone())
                .collect(),
        )?;
        let family_cohort_id = family_cohort.id().clone();
        let id = derive_sequence_cohort_id(&family_cohort_id, &batches);
        Ok(Self {
            id,
            family_cohort_id,
            batches,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn family_cohort_id(&self) -> &ContentId { &self.family_cohort_id }
    pub fn batches(&self) -> &[ContextBoundForgeSequenceBatch] { &self.batches }
    pub fn run_count(&self) -> u64 { self.batches.len() as u64 }

    pub fn validate(&self) -> Result<(), ForgeSequenceStatsError> {
        let rebuilt = Self::new(self.batches.clone())?;
        if rebuilt.id == self.id && rebuilt.family_cohort_id == self.family_cohort_id {
            Ok(())
        } else {
            Err(ForgeSequenceStatsError::CohortIdentityMismatch)
        }
    }
}

fn derive_sequence_cohort_id(
    family_cohort_id: &ContentId,
    batches: &[ContextBoundForgeSequenceBatch],
) -> ContentId {
    let count = (batches.len() as u64).to_be_bytes();
    let mut parts = vec![
        family_cohort_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        batches
            .iter()
            .map(|batch| batch.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-exact-context-sequence-cohort.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Statistical key `(history suffix, candidate family)` under one explicit history order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeConditionalFamilyKey {
    id: ContentId,
    order: ForgeHistoryOrder,
    history: ForgeFamilyHistory,
    candidate_family: ForgeTransformationFamilyId,
}

impl ForgeConditionalFamilyKey {
    pub fn new(
        order: &ForgeHistoryOrder,
        full_history: &ForgeFamilyHistory,
        candidate_family: ForgeTransformationFamilyId,
    ) -> Result<Self, ForgeSequenceStatsError> {
        order.validate()?;
        full_history.validate()?;
        candidate_family.validate().map_err(ForgeSequenceLearningError::from)?;
        if candidate_family.generator_id() != full_history.generator_id() {
            return Err(ForgeSequenceStatsError::KeyGeneratorMismatch);
        }
        let history = full_history.suffix(usize::from(order.depth()))?;
        let id = derive_key_id(order, &history, &candidate_family);
        Ok(Self {
            id,
            order: order.clone(),
            history,
            candidate_family,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn order(&self) -> &ForgeHistoryOrder { &self.order }
    pub fn history(&self) -> &ForgeFamilyHistory { &self.history }
    pub fn candidate_family(&self) -> &ForgeTransformationFamilyId { &self.candidate_family }

    pub fn validate(&self) -> Result<(), ForgeSequenceStatsError> {
        self.order.validate()?;
        self.history.validate()?;
        self.candidate_family
            .validate()
            .map_err(ForgeSequenceLearningError::from)?;
        if self.candidate_family.generator_id() != self.history.generator_id() {
            return Err(ForgeSequenceStatsError::KeyGeneratorMismatch);
        }
        if derive_key_id(&self.order, &self.history, &self.candidate_family) == self.id {
            Ok(())
        } else {
            Err(ForgeSequenceStatsError::KeyIdentityMismatch)
        }
    }
}

fn derive_key_id(
    order: &ForgeHistoryOrder,
    history: &ForgeFamilyHistory,
    family: &ForgeTransformationFamilyId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-conditional-family-key.v1",
        [
            order.id().as_str().as_bytes(),
            history.id().as_str().as_bytes(),
            family.as_content_id().as_str().as_bytes(),
        ],
    )
}

#[derive(Debug, Clone, Default)]
struct Counts {
    runs_observed: u64,
    runs_selected: u64,
    trials: u64,
    compile: u64,
    correctness: u64,
    evaluation: u64,
    not_selected: u64,
    selected: u64,
    interrupted: u64,
}

impl Counts {
    fn observe_trial(
        &mut self,
        trial: &ForgeConditionedFamilyTrial,
    ) -> Result<(), ForgeSequenceStatsError> {
        self.trials = add(self.trials, 1)?;
        let target = match trial.outcome() {
            ForgeTrialOutcome::RejectedCompilation => &mut self.compile,
            ForgeTrialOutcome::RejectedCorrectness => &mut self.correctness,
            ForgeTrialOutcome::RejectedEvaluation => &mut self.evaluation,
            ForgeTrialOutcome::ValidNotSelected => &mut self.not_selected,
            ForgeTrialOutcome::SelectedForContinuation => &mut self.selected,
            ForgeTrialOutcome::Interrupted => &mut self.interrupted,
        };
        *target = add(*target, 1)?;
        Ok(())
    }

    fn observe_run(&mut self) -> Result<(), ForgeSequenceStatsError> {
        self.runs_observed = add(self.runs_observed, 1)?;
        Ok(())
    }

    fn observe_selected_run(&mut self) -> Result<(), ForgeSequenceStatsError> {
        self.runs_selected = add(self.runs_selected, 1)?;
        Ok(())
    }
}

fn add(left: u64, right: u64) -> Result<u64, ForgeSequenceStatsError> {
    left.checked_add(right)
        .ok_or(ForgeSequenceStatsError::CountOverflow)
}

/// Descriptive outcomes for one `(history suffix, candidate family)` key across compatible runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContextualFamilyTransitionStats {
    id: ContentId,
    key: ForgeConditionalFamilyKey,
    cohort_runs: u64,
    counts: [u64; 9],
}

impl ContextualFamilyTransitionStats {
    fn from_counts(
        key: ForgeConditionalFamilyKey,
        cohort_runs: u64,
        counts: Counts,
    ) -> Result<Self, ForgeSequenceStatsError> {
        let mut row = Self {
            id: ContentId::derive(
                "symthaea.forge-conditional-family-row.uninitialized",
                [b"v1".as_slice()],
            ),
            key,
            cohort_runs,
            counts: [
                counts.runs_observed,
                counts.runs_selected,
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
    pub fn key(&self) -> &ForgeConditionalFamilyKey { &self.key }
    pub fn cohort_runs(&self) -> u64 { self.cohort_runs }
    pub fn runs_observed(&self) -> u64 { self.counts[0] }
    pub fn runs_selected(&self) -> u64 { self.counts[1] }
    pub fn trials(&self) -> u64 { self.counts[2] }
    pub fn rejected_compilation(&self) -> u64 { self.counts[3] }
    pub fn rejected_correctness(&self) -> u64 { self.counts[4] }
    pub fn rejected_evaluation(&self) -> u64 { self.counts[5] }
    pub fn valid_not_selected(&self) -> u64 { self.counts[6] }
    pub fn selected_for_continuation(&self) -> u64 { self.counts[7] }
    pub fn interrupted(&self) -> u64 { self.counts[8] }
    pub fn completed_trials(&self) -> u64 { self.trials() - self.interrupted() }

    pub fn run_coverage_rate(&self) -> Option<f64> {
        ratio(self.runs_observed(), self.cohort_runs)
    }

    pub fn run_selection_rate(&self) -> Option<f64> {
        ratio(self.runs_selected(), self.runs_observed())
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

    pub fn validate(&self) -> Result<(), ForgeSequenceStatsError> {
        self.key.validate()?;
        let classified = self.counts[3..]
            .iter()
            .try_fold(0u64, |sum, value| add(sum, *value))?;
        if self.cohort_runs < 2
            || self.runs_observed() == 0
            || self.runs_observed() > self.cohort_runs
            || self.runs_observed() > self.trials()
            || self.runs_selected() > self.runs_observed()
            || self.runs_selected() > self.selected_for_continuation()
            || classified != self.trials()
        {
            return Err(ForgeSequenceStatsError::InvalidCounts);
        }
        if derive_row_id(self) == self.id {
            Ok(())
        } else {
            Err(ForgeSequenceStatsError::RowIdentityMismatch)
        }
    }
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn derive_row_id(row: &ContextualFamilyTransitionStats) -> ContentId {
    let mut parts = vec![
        row.key.id().as_str().as_bytes().to_vec(),
        row.cohort_runs.to_be_bytes().to_vec(),
    ];
    parts.extend(row.counts.iter().map(|value| value.to_be_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-conditional-family-outcome-row.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Exact-context table of family outcomes conditioned on a bounded accepted-history suffix.
#[derive(Debug, Clone, Serialize)]
pub struct ExactContextConditionalFamilyTable {
    id: ContentId,
    cohort_id: ContentId,
    order: ForgeHistoryOrder,
    cohort_runs: u64,
    rows: Vec<ContextualFamilyTransitionStats>,
}

impl ExactContextConditionalFamilyTable {
    pub fn from_cohort(
        cohort: &ExactContextForgeSequenceCohort,
        order: ForgeHistoryOrder,
    ) -> Result<Self, ForgeSequenceStatsError> {
        cohort.validate()?;
        order.validate()?;
        let mut grouped = BTreeMap::<String, (ForgeConditionalFamilyKey, Counts)>::new();

        for batch in cohort.batches() {
            let mut seen_in_run = BTreeSet::<String>::new();
            let mut selected_in_run = BTreeSet::<String>::new();
            for trial in batch.sequence().conditioned_trials() {
                let key = ForgeConditionalFamilyKey::new(
                    &order,
                    trial.history_before(),
                    trial.family_id().clone(),
                )?;
                let key_id = key.id().as_str().to_string();
                grouped
                    .entry(key_id.clone())
                    .or_insert_with(|| (key, Counts::default()))
                    .1
                    .observe_trial(trial)?;
                seen_in_run.insert(key_id.clone());
                if trial.outcome() == ForgeTrialOutcome::SelectedForContinuation {
                    selected_in_run.insert(key_id);
                }
            }
            for key_id in seen_in_run {
                grouped
                    .get_mut(&key_id)
                    .expect("key inserted while observing run")
                    .1
                    .observe_run()?;
            }
            for key_id in selected_in_run {
                grouped
                    .get_mut(&key_id)
                    .expect("selected key inserted while observing run")
                    .1
                    .observe_selected_run()?;
            }
        }

        let cohort_runs = cohort.run_count();
        let rows = grouped
            .into_values()
            .map(|(key, counts)| ContextualFamilyTransitionStats::from_counts(key, cohort_runs, counts))
            .collect::<Result<Vec<_>, _>>()?;
        let id = derive_table_id(cohort.id(), &order, cohort_runs, &rows);
        let table = Self {
            id,
            cohort_id: cohort.id().clone(),
            order,
            cohort_runs,
            rows,
        };
        table.validate()?;
        Ok(table)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn cohort_id(&self) -> &ContentId { &self.cohort_id }
    pub fn order(&self) -> &ForgeHistoryOrder { &self.order }
    pub fn cohort_runs(&self) -> u64 { self.cohort_runs }
    pub fn rows(&self) -> &[ContextualFamilyTransitionStats] { &self.rows }

    pub fn get(
        &self,
        key: &ForgeConditionalFamilyKey,
    ) -> Option<&ContextualFamilyTransitionStats> {
        self.rows
            .binary_search_by(|row| row.key().id().cmp(key.id()))
            .ok()
            .map(|index| &self.rows[index])
    }

    pub fn validate(&self) -> Result<(), ForgeSequenceStatsError> {
        self.order.validate()?;
        if self.cohort_runs < 2 {
            return Err(ForgeSequenceStatsError::InvalidCounts);
        }
        let mut previous: Option<&ContentId> = None;
        for row in &self.rows {
            row.validate()?;
            if row.cohort_runs() != self.cohort_runs
                || row.key().order() != &self.order
                || previous.is_some_and(|prior| prior >= row.key().id())
            {
                return Err(ForgeSequenceStatsError::TableIdentityMismatch);
            }
            previous = Some(row.key().id());
        }
        if derive_table_id(&self.cohort_id, &self.order, self.cohort_runs, &self.rows) == self.id {
            Ok(())
        } else {
            Err(ForgeSequenceStatsError::TableIdentityMismatch)
        }
    }
}

fn derive_table_id(
    cohort_id: &ContentId,
    order: &ForgeHistoryOrder,
    cohort_runs: u64,
    rows: &[ContextualFamilyTransitionStats],
) -> ContentId {
    let runs = cohort_runs.to_be_bytes();
    let count = (rows.len() as u64).to_be_bytes();
    let mut parts = vec![
        cohort_id.as_str().as_bytes().to_vec(),
        order.id().as_str().as_bytes().to_vec(),
        runs.to_vec(),
        count.to_vec(),
    ];
    parts.extend(rows.iter().map(|row| row.id().as_str().as_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-exact-context-conditional-family-table.v1",
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
    use symthaea_algorithms::observation::{ObservationEncoding, ObservationObject};
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
            "conditional-sequence-test",
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
            "conditional sequence baseline",
            AlgorithmProvenance::HumanAuthored,
        )
        .unwrap();
        let baseline_artifact = full_source_artifact_id("fn f() -> i32 { 1 }\n");
        let baseline = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id,
            "repo://src/f.rs",
            baseline_artifact,
            None,
        )
        .unwrap();
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
        )
        .unwrap();
        Fixture {
            problem,
            baseline,
            generator,
            context,
        }
    }

    fn decision(
        attempt: &ForgeAttemptId,
        mutation: &MutationRecord,
        decision: &str,
    ) -> ObservationObject {
        ObservationObject::new(
            "symthaea.forge.candidate-decision.v2",
            ObservationEncoding::Json,
            serde_json::to_vec(&serde_json::json!({
                "attempt_id": attempt.as_content_id().as_str(),
                "candidate_artifact_id": mutation.candidate_artifact_id.as_str(),
                "transformation_id": mutation.transformation_id.as_content_id().as_str(),
                "decision": decision,
                "metric_name": null,
                "candidate_score_bits": null
            }))
            .unwrap(),
        )
        .unwrap()
    }

    fn run_trace(
        f: &Fixture,
        seed: u64,
    ) -> (DiscoveryRun, Vec<ForgeTraceEvent>, ObservationStore) {
        let run = DiscoveryRun::new(
            &f.problem,
            DiscoveryPolicy::default(),
            f.generator.clone(),
            "abc123",
            SearchBudget::new(10, 2, 10).unwrap(),
            seed,
        )
        .unwrap();
        let a_artifact = full_source_artifact_id("fn f() -> i32 { 2 }\n");
        let b_artifact = full_source_artifact_id("fn f() -> i32 { 3 }\n");
        let mutation_a = MutationRecord::new(
            0,
            "A",
            "1 -> 2",
            f.baseline.artifact_id.clone(),
            a_artifact.clone(),
        );
        let mutation_b = MutationRecord::new(
            1,
            "B",
            "2 -> 3",
            a_artifact.clone(),
            b_artifact.clone(),
        );
        let attempt_a = ForgeAttemptId::derive(&f.baseline.artifact_id, seed, 0, 0);
        let attempt_b = ForgeAttemptId::derive(&f.baseline.artifact_id, seed, 1, 1);
        let generated_a = forge_observations::candidate_generated(&attempt_a, &mutation_a).unwrap();
        let selected_a = decision(&attempt_a, &mutation_a, "selected-for-next-generation");
        let generated_b = forge_observations::candidate_generated(&attempt_b, &mutation_b).unwrap();
        let rejected_b = forge_observations::gates(
            &attempt_b,
            &mutation_b,
            &[GateResult {
                gate: Gate::Compile,
                passed: false,
                output_tail: "compile error".into(),
                duration: Duration::from_nanos(1),
            }],
        )
        .unwrap();
        let summary =
            forge_observations::search_summary(2, 0, 1, 0, 0, 1, 1, None, None).unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt_a.clone(),
                0,
                DiscoveryEventKind::CandidateGenerated,
                a_artifact.clone(),
                generated_a.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_a,
                0,
                DiscoveryEventKind::SelectedForContinuation,
                a_artifact,
                selected_a.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_b.clone(),
                1,
                DiscoveryEventKind::CandidateGenerated,
                b_artifact.clone(),
                generated_b.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_b,
                1,
                DiscoveryEventKind::RejectedCompilation,
                b_artifact,
                rejected_b.id().clone(),
            ),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![
            generated_a,
            selected_a,
            generated_b,
            rejected_b,
            summary,
        ])
        .unwrap();
        (run, trace, observations)
    }

    #[test]
    fn first_order_transition_is_counted_across_distinct_runs() {
        let f = fixture();
        let (run_a, trace_a, obs_a) = run_trace(&f, 7);
        let (run_b, trace_b, obs_b) = run_trace(&f, 8);
        let batch_a = ContextBoundForgeSequenceBatch::from_trace(
            &run_a,
            &f.baseline,
            f.context.clone(),
            &trace_a,
            &obs_a,
        )
        .unwrap();
        let batch_b = ContextBoundForgeSequenceBatch::from_trace(
            &run_b,
            &f.baseline,
            f.context.clone(),
            &trace_b,
            &obs_b,
        )
        .unwrap();
        let cohort = ExactContextForgeSequenceCohort::new(vec![batch_a, batch_b]).unwrap();
        let table = ExactContextConditionalFamilyTable::from_cohort(
            &cohort,
            ForgeHistoryOrder::new(1).unwrap(),
        )
        .unwrap();
        let row = table
            .rows()
            .iter()
            .find(|row| row.key().candidate_family().operator() == "B")
            .unwrap();
        assert_eq!(row.key().history().families().len(), 1);
        assert_eq!(row.key().history().families()[0].operator(), "A");
        assert_eq!(row.runs_observed(), 2);
        assert_eq!(row.cohort_runs(), 2);
        assert_eq!(row.trials(), 2);
        assert_eq!(row.rejected_compilation(), 2);
        assert_eq!(row.run_coverage_rate(), Some(1.0));
        assert!(table.validate().is_ok());
    }

    #[test]
    fn history_order_is_part_of_the_conditional_key() {
        let generator = ContentId::derive("generator", [b"forge-v1".as_slice()]);
        let a = ForgeTransformationFamilyId::new(generator.clone(), "A").unwrap();
        let b = ForgeTransformationFamilyId::new(generator.clone(), "B").unwrap();
        let history = ForgeFamilyHistory::new(generator, vec![a]).unwrap();
        let one = ForgeConditionalFamilyKey::new(
            &ForgeHistoryOrder::new(1).unwrap(),
            &history,
            b.clone(),
        )
        .unwrap();
        let two = ForgeConditionalFamilyKey::new(
            &ForgeHistoryOrder::new(2).unwrap(),
            &history,
            b,
        )
        .unwrap();
        assert_ne!(one.id(), two.id());
    }
}
