// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run-bound descriptive learning summaries over canonical Forge transformation trials.
//!
//! These summaries intentionally remain per discovery run. They do not combine heterogeneous
//! machines, problems, baselines, or evaluator contexts, and local search selection is not treated
//! as replicated performance evidence.

use crate::candidate::{ledger_from_forge_trace, ForgeProposalError};
use crate::trace::ForgeTraceEvent;
use crate::trials::{
    extract_transformation_trials, ForgeTrialError, ForgeTrialOutcome, ForgeTrialSet,
};
use serde::Serialize;
use std::collections::BTreeMap;
use symthaea_algorithms::discovery::{DiscoveryError, DiscoveryRun};
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{
    ContentId, ImplementationId, ImplementationRecord, ProblemId, RegistryError, TransformationId,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeLearningError {
    #[error(transparent)]
    Proposal(#[from] ForgeProposalError),
    #[error(transparent)]
    Trial(#[from] ForgeTrialError),
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error("baseline implementation does not describe the discovery-run problem")]
    BaselineProblemMismatch,
    #[error("Forge trial attempt seed does not match the discovery-run seed")]
    SeedMismatch,
    #[error("Forge trial attempt baseline artifact does not match the supplied baseline implementation")]
    BaselineArtifactMismatch,
    #[error("Forge trial generation exceeds the discovery-run generation budget")]
    GenerationBudgetMismatch,
    #[error("transformation outcome row is internally inconsistent")]
    InvalidOutcomeCounts,
    #[error("transformation outcome row identity does not match its canonical fields")]
    RowIdentityMismatch,
    #[error("transformation outcome table is not in canonical transformation order")]
    NonCanonicalRowOrder,
    #[error("transformation outcome table contains a duplicate transformation")]
    DuplicateTransformation,
    #[error("transformation outcome table identity does not match its canonical fields")]
    TableIdentityMismatch,
}

/// Exact semantic binding between one Forge trial set and one generic discovery run.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeTrialBatch {
    id: ContentId,
    run_id: ContentId,
    problem_id: ProblemId,
    baseline_implementation_id: ImplementationId,
    baseline_artifact_id: ContentId,
    ledger_snapshot_id: ContentId,
    trials: ForgeTrialSet,
}

impl ForgeTrialBatch {
    /// Replay the trace against the generic discovery contract and extract strictly validated
    /// Forge trials in one operation. This is the preferred boundary for learning consumers.
    pub fn from_trace(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        trace: &[ForgeTraceEvent],
        observations: &ObservationStore,
    ) -> Result<Self, ForgeLearningError> {
        run.validate()?;
        baseline.validate()?;
        if baseline.problem_id != run.problem_id {
            return Err(ForgeLearningError::BaselineProblemMismatch);
        }

        let ledger = ledger_from_forge_trace(run, trace, observations)?;
        let ledger_snapshot_id = ledger.snapshot_id(run).map_err(ForgeProposalError::from)?;
        let trials = extract_transformation_trials(trace, observations)?;
        trials.validate()?;

        for trial in trials.trials() {
            if trial.attempt_id().seed() != run.seed {
                return Err(ForgeLearningError::SeedMismatch);
            }
            if trial.attempt_id().baseline_artifact_id() != &baseline.artifact_id {
                return Err(ForgeLearningError::BaselineArtifactMismatch);
            }
            if trial.generation() >= run.budget.max_generations {
                return Err(ForgeLearningError::GenerationBudgetMismatch);
            }
        }

        let id = derive_batch_id(
            &run.id,
            &baseline.id,
            &baseline.artifact_id,
            &ledger_snapshot_id,
            trials.id(),
        );
        Ok(Self {
            id,
            run_id: run.id.clone(),
            problem_id: run.problem_id.clone(),
            baseline_implementation_id: baseline.id.clone(),
            baseline_artifact_id: baseline.artifact_id.clone(),
            ledger_snapshot_id,
            trials,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn run_id(&self) -> &ContentId {
        &self.run_id
    }

    pub fn problem_id(&self) -> &ProblemId {
        &self.problem_id
    }

    pub fn baseline_implementation_id(&self) -> &ImplementationId {
        &self.baseline_implementation_id
    }

    pub fn baseline_artifact_id(&self) -> &ContentId {
        &self.baseline_artifact_id
    }

    pub fn ledger_snapshot_id(&self) -> &ContentId {
        &self.ledger_snapshot_id
    }

    pub fn trials(&self) -> &ForgeTrialSet {
        &self.trials
    }
}

fn derive_batch_id(
    run_id: &ContentId,
    baseline_implementation_id: &ImplementationId,
    baseline_artifact_id: &ContentId,
    ledger_snapshot_id: &ContentId,
    trial_set_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-trial-batch.v1",
        [
            run_id.as_str().as_bytes(),
            baseline_implementation_id.as_content_id().as_str().as_bytes(),
            baseline_artifact_id.as_str().as_bytes(),
            ledger_snapshot_id.as_str().as_bytes(),
            trial_set_id.as_str().as_bytes(),
        ],
    )
}

/// Counts only. Derived rates are convenience views and never participate in content identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransformationOutcomeStats {
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

impl TransformationOutcomeStats {
    fn from_trials(
        transformation_id: TransformationId,
        trials: impl IntoIterator<Item = ForgeTrialOutcome>,
    ) -> Result<Self, ForgeLearningError> {
        let mut row = Self {
            id: ContentId::derive(
                "symthaea.forge-transformation-outcome-row.uninitialized",
                [b"v1".as_slice()],
            ),
            transformation_id,
            trials: 0,
            rejected_compilation: 0,
            rejected_correctness: 0,
            rejected_evaluation: 0,
            valid_not_selected: 0,
            selected_for_continuation: 0,
            interrupted: 0,
        };
        for outcome in trials {
            row.trials = row.trials.saturating_add(1);
            match outcome {
                ForgeTrialOutcome::RejectedCompilation => {
                    row.rejected_compilation = row.rejected_compilation.saturating_add(1)
                }
                ForgeTrialOutcome::RejectedCorrectness => {
                    row.rejected_correctness = row.rejected_correctness.saturating_add(1)
                }
                ForgeTrialOutcome::RejectedEvaluation => {
                    row.rejected_evaluation = row.rejected_evaluation.saturating_add(1)
                }
                ForgeTrialOutcome::ValidNotSelected => {
                    row.valid_not_selected = row.valid_not_selected.saturating_add(1)
                }
                ForgeTrialOutcome::SelectedForContinuation => {
                    row.selected_for_continuation = row.selected_for_continuation.saturating_add(1)
                }
                ForgeTrialOutcome::Interrupted => {
                    row.interrupted = row.interrupted.saturating_add(1)
                }
            }
        }
        row.id = derive_row_id(&row);
        row.validate()?;
        Ok(row)
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
        self.trials.saturating_sub(self.interrupted)
    }

    pub fn interrupted(&self) -> u64 {
        self.interrupted
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

    /// Among non-interrupted trials, fraction not rejected at compilation.
    pub fn compile_survival_rate(&self) -> Option<f64> {
        ratio(
            self.completed_trials().saturating_sub(self.rejected_compilation),
            self.completed_trials(),
        )
    }

    /// Among completed trials that survived compilation, fraction not rejected for correctness.
    pub fn correctness_survival_rate(&self) -> Option<f64> {
        let entered = self.completed_trials().saturating_sub(self.rejected_compilation);
        ratio(entered.saturating_sub(self.rejected_correctness), entered)
    }

    /// Among completed correctness-surviving trials, fraction that reached a valid local selection
    /// decision rather than being rejected by the configured evaluation heuristic.
    pub fn evaluation_validity_rate(&self) -> Option<f64> {
        let entered = self
            .completed_trials()
            .saturating_sub(self.rejected_compilation)
            .saturating_sub(self.rejected_correctness);
        ratio(
            self.valid_not_selected
                .saturating_add(self.selected_for_continuation),
            entered,
        )
    }

    /// Among locally valid candidates, fraction Forge selected as the next continuation parent.
    /// This is search behavior, not replicated superiority evidence.
    pub fn local_selection_rate(&self) -> Option<f64> {
        ratio(
            self.selected_for_continuation,
            self.valid_not_selected
                .saturating_add(self.selected_for_continuation),
        )
    }

    pub fn interruption_rate(&self) -> Option<f64> {
        ratio(self.interrupted, self.trials)
    }

    pub fn validate(&self) -> Result<(), ForgeLearningError> {
        let classified = self
            .rejected_compilation
            .saturating_add(self.rejected_correctness)
            .saturating_add(self.rejected_evaluation)
            .saturating_add(self.valid_not_selected)
            .saturating_add(self.selected_for_continuation)
            .saturating_add(self.interrupted);
        if classified != self.trials {
            return Err(ForgeLearningError::InvalidOutcomeCounts);
        }
        if derive_row_id(self) != self.id {
            return Err(ForgeLearningError::RowIdentityMismatch);
        }
        Ok(())
    }
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn derive_row_id(row: &TransformationOutcomeStats) -> ContentId {
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
        "symthaea.forge-transformation-outcome-row.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransformationOutcomeTable {
    id: ContentId,
    batch_id: ContentId,
    rows: Vec<TransformationOutcomeStats>,
}

impl TransformationOutcomeTable {
    pub fn from_batch(batch: &ForgeTrialBatch) -> Result<Self, ForgeLearningError> {
        let mut grouped = BTreeMap::<TransformationId, Vec<ForgeTrialOutcome>>::new();
        for trial in batch.trials().trials() {
            grouped
                .entry(trial.transformation_id().clone())
                .or_default()
                .push(trial.outcome());
        }
        let rows = grouped
            .into_iter()
            .map(|(transformation, outcomes)| {
                TransformationOutcomeStats::from_trials(transformation, outcomes)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let id = derive_table_id(batch.id(), &rows);
        let table = Self {
            id,
            batch_id: batch.id().clone(),
            rows,
        };
        table.validate()?;
        Ok(table)
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn batch_id(&self) -> &ContentId {
        &self.batch_id
    }

    pub fn rows(&self) -> &[TransformationOutcomeStats] {
        &self.rows
    }

    pub fn get(&self, transformation_id: &TransformationId) -> Option<&TransformationOutcomeStats> {
        self.rows
            .binary_search_by(|row| row.transformation_id.cmp(transformation_id))
            .ok()
            .map(|index| &self.rows[index])
    }

    pub fn validate(&self) -> Result<(), ForgeLearningError> {
        let mut previous: Option<&TransformationId> = None;
        for row in &self.rows {
            row.validate()?;
            if let Some(previous) = previous {
                match previous.cmp(&row.transformation_id) {
                    std::cmp::Ordering::Less => {}
                    std::cmp::Ordering::Equal => {
                        return Err(ForgeLearningError::DuplicateTransformation)
                    }
                    std::cmp::Ordering::Greater => {
                        return Err(ForgeLearningError::NonCanonicalRowOrder)
                    }
                }
            }
            previous = Some(&row.transformation_id);
        }
        if derive_table_id(&self.batch_id, &self.rows) != self.id {
            return Err(ForgeLearningError::TableIdentityMismatch);
        }
        Ok(())
    }
}

fn derive_table_id(batch_id: &ContentId, rows: &[TransformationOutcomeStats]) -> ContentId {
    let count = (rows.len() as u64).to_be_bytes();
    let mut parts = vec![batch_id.as_str().as_bytes().to_vec(), count.to_vec()];
    parts.extend(rows.iter().map(|row| row.id().as_str().as_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-transformation-outcome-table.v1",
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

    fn fixture() -> (
        DiscoveryRun,
        ImplementationRecord,
        ForgeAttemptId,
        MutationRecord,
    ) {
        let problem = ProblemSpec::new(
            "learning-test",
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
            "test algorithm",
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
            SearchBudget::new(10, 2, 10).unwrap(),
            9,
        )
        .unwrap();
        let candidate = full_source_artifact_id("fn f() -> i32 { 2 }\n");
        let attempt = ForgeAttemptId::derive(&baseline_artifact, run.seed, 0, 0);
        let mutation = MutationRecord::new(
            0,
            "literal",
            "1 -> 2",
            baseline_artifact,
            candidate,
        );
        (run, baseline, attempt, mutation)
    }

    fn correctness_failure() -> Vec<GateResult> {
        vec![
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
        ]
    }

    #[test]
    fn run_bound_table_counts_rejection_without_calling_it_interruption() {
        let (run, baseline, attempt, mutation) = fixture();
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let rejected = forge_observations::gates(&attempt, &mutation, &correctness_failure()).unwrap();
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
        let batch = ForgeTrialBatch::from_trace(&run, &baseline, &trace, &observations).unwrap();
        let table = TransformationOutcomeTable::from_batch(&batch).unwrap();
        let row = table.get(&mutation.transformation_id).expect("one transformation row");
        assert_eq!(row.trials(), 1);
        assert_eq!(row.rejected_correctness(), 1);
        assert_eq!(row.interrupted(), 0);
        assert_eq!(row.compile_survival_rate(), Some(1.0));
        assert_eq!(row.correctness_survival_rate(), Some(0.0));
        assert!(table.validate().is_ok());
    }

    #[test]
    fn interrupted_trial_is_excluded_from_completed_survival_denominators() {
        let (run, baseline, attempt, mutation) = fixture();
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let abort = forge_observations::search_abort(
            "candidate-gate-apparatus",
            "runner unavailable",
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
        )
        .unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt,
                0,
                DiscoveryEventKind::CandidateGenerated,
                mutation.candidate_artifact_id.clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::aborted(abort.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![generated, abort]).unwrap();
        let batch = ForgeTrialBatch::from_trace(&run, &baseline, &trace, &observations).unwrap();
        let table = TransformationOutcomeTable::from_batch(&batch).unwrap();
        let row = table.rows().first().unwrap();
        assert_eq!(row.trials(), 1);
        assert_eq!(row.interrupted(), 1);
        assert_eq!(row.completed_trials(), 0);
        assert_eq!(row.compile_survival_rate(), None);
        assert_eq!(row.interruption_rate(), Some(1.0));
    }
}
