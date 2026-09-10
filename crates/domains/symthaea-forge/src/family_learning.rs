// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generator-scoped transformation-family learning for Forge.
//!
//! A concrete Forge `TransformationId` commits one exact rewrite instance, including generation,
//! detail, parent artifact, and child artifact. That identity is correct for lineage but too
//! specific for statistical learning. This module introduces a separate family identity derived
//! from `(generator_id, operator)` while preserving the concrete transformation identity on every
//! trial.
//!
//! Family-level outcomes remain descriptive search memory. They do not establish independent
//! correctness, replicated performance, promotion eligibility, or runtime authority.

use crate::certificate::MutationRecord;
use crate::learning::{ForgeLearningError, ForgeTrialBatch};
use crate::trace::ForgeAttemptId;
use crate::trials::{ForgeTrialError, ForgeTrialOutcome, TransformationTrial};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use symthaea_algorithms::discovery::DiscoveryRun;
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{ContentId, ImplementationRecord, TransformationId};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeFamilyLearningError {
    #[error(transparent)]
    Learning(#[from] ForgeLearningError),
    #[error(transparent)]
    Trial(#[from] ForgeTrialError),
    #[error("transformation-family operator must be non-empty canonical single-line text")]
    InvalidOperator,
    #[error("generated observation is missing from the observation store")]
    MissingGeneratedObservation,
    #[error("generated observation payload is not valid JSON")]
    InvalidGeneratedObservation,
    #[error("generated observation does not contain the required mutation fields")]
    MissingMutationField,
    #[error("generated observation mutation identity disagrees with the canonical transformation trial")]
    MutationIdentityMismatch,
    #[error("trial generation cannot be represented as a Forge mutation generation")]
    GenerationOverflow,
    #[error("transformation-family identity does not match canonical fields")]
    FamilyIdentityMismatch,
    #[error("family trial identity does not match canonical fields")]
    FamilyTrialIdentityMismatch,
    #[error("family trial set contains mixed discovery runs")]
    RunMismatch,
    #[error("family trial set contains mixed generator identities")]
    GeneratorMismatch,
    #[error("family outcome counts are internally inconsistent")]
    InvalidOutcomeCounts,
    #[error("family outcome row identity does not match canonical fields")]
    RowIdentityMismatch,
    #[error("family outcome table identity does not match canonical fields")]
    TableIdentityMismatch,
    #[error("family outcome count overflow")]
    CountOverflow,
}

/// Stable identity of one Forge mutation operator under one exact generator implementation.
///
/// If Forge's generator changes, its `generator_id` must change and family evidence does not
/// silently transfer. A future explicitly reviewed compatibility theorem may bridge generator
/// versions; v1 does not.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ForgeTransformationFamilyId {
    id: ContentId,
    generator_id: ContentId,
    operator: String,
}

impl ForgeTransformationFamilyId {
    pub fn new(
        generator_id: ContentId,
        operator: impl Into<String>,
    ) -> Result<Self, ForgeFamilyLearningError> {
        let operator = operator.into();
        validate_operator(&operator)?;
        let id = ContentId::derive(
            "symthaea.forge-transformation-family.v1",
            [generator_id.as_str().as_bytes(), operator.as_bytes()],
        );
        Ok(Self {
            id,
            generator_id,
            operator,
        })
    }

    pub fn as_content_id(&self) -> &ContentId {
        &self.id
    }

    pub fn generator_id(&self) -> &ContentId {
        &self.generator_id
    }

    pub fn operator(&self) -> &str {
        &self.operator
    }

    pub fn validate(&self) -> Result<(), ForgeFamilyLearningError> {
        let rebuilt = Self::new(self.generator_id.clone(), self.operator.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeFamilyLearningError::FamilyIdentityMismatch)
        }
    }
}

fn validate_operator(operator: &str) -> Result<(), ForgeFamilyLearningError> {
    if operator.trim().is_empty()
        || operator.trim() != operator
        || operator.chars().any(char::is_control)
    {
        Err(ForgeFamilyLearningError::InvalidOperator)
    } else {
        Ok(())
    }
}

/// One exact concrete transformation trial projected onto its reusable generator-scoped family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeFamilyTrial {
    id: ContentId,
    run_id: ContentId,
    attempt_id: ForgeAttemptId,
    family_id: ForgeTransformationFamilyId,
    concrete_transformation_id: TransformationId,
    concrete_trial_id: ContentId,
    outcome: ForgeTrialOutcome,
}

impl ForgeFamilyTrial {
    fn from_trial(
        run: &DiscoveryRun,
        trial: &TransformationTrial,
        observations: &ObservationStore,
    ) -> Result<Self, ForgeFamilyLearningError> {
        let object = observations
            .get(trial.generated_observation_id())
            .ok_or(ForgeFamilyLearningError::MissingGeneratedObservation)?;
        let payload: Value = serde_json::from_slice(object.payload())
            .map_err(|_| ForgeFamilyLearningError::InvalidGeneratedObservation)?;

        let operator = required_string(&payload, "operator")?;
        let detail = required_string(&payload, "detail")?;
        let parent = parse_content_id(&payload, "parent_artifact_id")?;
        let candidate = parse_content_id(&payload, "candidate_artifact_id")?;
        let observed_transformation = parse_content_id(&payload, "transformation_id")?;
        let generation = usize::try_from(trial.generation())
            .map_err(|_| ForgeFamilyLearningError::GenerationOverflow)?;

        let reconstructed = MutationRecord::new(
            generation,
            operator,
            detail,
            parent,
            candidate,
        );
        if reconstructed.transformation_id.as_content_id() != &observed_transformation
            || &reconstructed.transformation_id != trial.transformation_id()
            || &reconstructed.parent_artifact_id != trial.parent_artifact_id()
            || &reconstructed.candidate_artifact_id != trial.candidate_artifact_id()
        {
            return Err(ForgeFamilyLearningError::MutationIdentityMismatch);
        }

        let family_id = ForgeTransformationFamilyId::new(
            run.generator_id.clone(),
            reconstructed.operator,
        )?;
        let id = derive_family_trial_id(
            &run.id,
            trial.attempt_id(),
            &family_id,
            trial.transformation_id(),
            trial.id(),
            trial.outcome(),
        );
        Ok(Self {
            id,
            run_id: run.id.clone(),
            attempt_id: trial.attempt_id().clone(),
            family_id,
            concrete_transformation_id: trial.transformation_id().clone(),
            concrete_trial_id: trial.id().clone(),
            outcome: trial.outcome(),
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn run_id(&self) -> &ContentId {
        &self.run_id
    }

    pub fn attempt_id(&self) -> &ForgeAttemptId {
        &self.attempt_id
    }

    pub fn family_id(&self) -> &ForgeTransformationFamilyId {
        &self.family_id
    }

    pub fn concrete_transformation_id(&self) -> &TransformationId {
        &self.concrete_transformation_id
    }

    pub fn concrete_trial_id(&self) -> &ContentId {
        &self.concrete_trial_id
    }

    pub fn outcome(&self) -> ForgeTrialOutcome {
        self.outcome
    }

    pub fn validate(&self) -> Result<(), ForgeFamilyLearningError> {
        self.family_id.validate()?;
        self.attempt_id
            .validate()
            .map_err(|_| ForgeFamilyLearningError::FamilyTrialIdentityMismatch)?;
        let expected = derive_family_trial_id(
            &self.run_id,
            &self.attempt_id,
            &self.family_id,
            &self.concrete_transformation_id,
            &self.concrete_trial_id,
            self.outcome,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeFamilyLearningError::FamilyTrialIdentityMismatch)
        }
    }
}

fn derive_family_trial_id(
    run_id: &ContentId,
    attempt_id: &ForgeAttemptId,
    family_id: &ForgeTransformationFamilyId,
    transformation_id: &TransformationId,
    concrete_trial_id: &ContentId,
    outcome: ForgeTrialOutcome,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-family-trial.v1",
        [
            run_id.as_str().as_bytes(),
            attempt_id.as_content_id().as_str().as_bytes(),
            family_id.as_content_id().as_str().as_bytes(),
            transformation_id.as_content_id().as_str().as_bytes(),
            concrete_trial_id.as_str().as_bytes(),
            outcome_tag(outcome),
        ],
    )
}

fn outcome_tag(outcome: ForgeTrialOutcome) -> &'static [u8] {
    match outcome {
        ForgeTrialOutcome::RejectedCompilation => b"rejected-compilation",
        ForgeTrialOutcome::RejectedCorrectness => b"rejected-correctness",
        ForgeTrialOutcome::RejectedEvaluation => b"rejected-evaluation",
        ForgeTrialOutcome::ValidNotSelected => b"valid-not-selected",
        ForgeTrialOutcome::SelectedForContinuation => b"selected-for-continuation",
        ForgeTrialOutcome::Interrupted => b"interrupted",
    }
}

fn required_string(
    payload: &Value,
    field: &str,
) -> Result<String, ForgeFamilyLearningError> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or(ForgeFamilyLearningError::MissingMutationField)
}

fn parse_content_id(
    payload: &Value,
    field: &str,
) -> Result<ContentId, ForgeFamilyLearningError> {
    let raw = required_string(payload, field)?;
    ContentId::parse(raw).map_err(|_| ForgeFamilyLearningError::MissingMutationField)
}

/// Canonical family projection for one exact semantic discovery run.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeFamilyTrialSet {
    id: ContentId,
    run_id: ContentId,
    generator_id: ContentId,
    source_batch_id: ContentId,
    trials: Vec<ForgeFamilyTrial>,
}

impl ForgeFamilyTrialSet {
    pub fn from_trace(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        trace: &[crate::trace::ForgeTraceEvent],
        observations: &ObservationStore,
    ) -> Result<Self, ForgeFamilyLearningError> {
        let batch = ForgeTrialBatch::from_trace(run, baseline, trace, observations)?;
        let mut trials = batch
            .trials()
            .trials()
            .iter()
            .map(|trial| ForgeFamilyTrial::from_trial(run, trial, observations))
            .collect::<Result<Vec<_>, _>>()?;
        trials.sort_by_key(|trial| trial.attempt_id().ordinal());
        let id = derive_family_set_id(&run.id, &run.generator_id, batch.id(), &trials);
        let set = Self {
            id,
            run_id: run.id.clone(),
            generator_id: run.generator_id.clone(),
            source_batch_id: batch.id().clone(),
            trials,
        };
        set.validate()?;
        Ok(set)
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn run_id(&self) -> &ContentId {
        &self.run_id
    }

    pub fn generator_id(&self) -> &ContentId {
        &self.generator_id
    }

    pub fn source_batch_id(&self) -> &ContentId {
        &self.source_batch_id
    }

    pub fn trials(&self) -> &[ForgeFamilyTrial] {
        &self.trials
    }

    pub fn validate(&self) -> Result<(), ForgeFamilyLearningError> {
        let mut previous = None;
        for trial in &self.trials {
            trial.validate()?;
            if trial.run_id() != &self.run_id {
                return Err(ForgeFamilyLearningError::RunMismatch);
            }
            if trial.family_id().generator_id() != &self.generator_id {
                return Err(ForgeFamilyLearningError::GeneratorMismatch);
            }
            if previous.is_some_and(|ordinal| trial.attempt_id().ordinal() <= ordinal) {
                return Err(ForgeFamilyLearningError::FamilyTrialIdentityMismatch);
            }
            previous = Some(trial.attempt_id().ordinal());
        }
        let expected = derive_family_set_id(
            &self.run_id,
            &self.generator_id,
            &self.source_batch_id,
            &self.trials,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeFamilyLearningError::FamilyTrialIdentityMismatch)
        }
    }
}

fn derive_family_set_id(
    run_id: &ContentId,
    generator_id: &ContentId,
    batch_id: &ContentId,
    trials: &[ForgeFamilyTrial],
) -> ContentId {
    let count = (trials.len() as u64).to_be_bytes();
    let mut parts = vec![
        run_id.as_str().as_bytes().to_vec(),
        generator_id.as_str().as_bytes().to_vec(),
        batch_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(trials.iter().map(|trial| trial.id().as_str().as_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-family-trial-set.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[derive(Debug, Clone, Default)]
struct OutcomeCounts {
    total: u64,
    compile: u64,
    correctness: u64,
    evaluation: u64,
    not_selected: u64,
    selected: u64,
    interrupted: u64,
}

impl OutcomeCounts {
    fn observe(&mut self, outcome: ForgeTrialOutcome) -> Result<(), ForgeFamilyLearningError> {
        self.total = add(self.total, 1)?;
        let counter = match outcome {
            ForgeTrialOutcome::RejectedCompilation => &mut self.compile,
            ForgeTrialOutcome::RejectedCorrectness => &mut self.correctness,
            ForgeTrialOutcome::RejectedEvaluation => &mut self.evaluation,
            ForgeTrialOutcome::ValidNotSelected => &mut self.not_selected,
            ForgeTrialOutcome::SelectedForContinuation => &mut self.selected,
            ForgeTrialOutcome::Interrupted => &mut self.interrupted,
        };
        *counter = add(*counter, 1)?;
        Ok(())
    }
}

fn add(left: u64, right: u64) -> Result<u64, ForgeFamilyLearningError> {
    left.checked_add(right)
        .ok_or(ForgeFamilyLearningError::CountOverflow)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeFamilyOutcomeStats {
    id: ContentId,
    family_id: ForgeTransformationFamilyId,
    counts: [u64; 7],
}

impl ForgeFamilyOutcomeStats {
    fn from_counts(
        family_id: ForgeTransformationFamilyId,
        counts: OutcomeCounts,
    ) -> Result<Self, ForgeFamilyLearningError> {
        let mut row = Self {
            id: ContentId::derive("symthaea.forge-family-row.uninitialized", [b"v1".as_slice()]),
            family_id,
            counts: [
                counts.total,
                counts.compile,
                counts.correctness,
                counts.evaluation,
                counts.not_selected,
                counts.selected,
                counts.interrupted,
            ],
        };
        row.id = derive_family_row_id(&row);
        row.validate()?;
        Ok(row)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn family_id(&self) -> &ForgeTransformationFamilyId { &self.family_id }
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

    pub fn validate(&self) -> Result<(), ForgeFamilyLearningError> {
        self.family_id.validate()?;
        let classified = self.counts[1..]
            .iter()
            .try_fold(0u64, |sum, value| add(sum, *value))?;
        if classified != self.trials() || self.interrupted() > self.trials() {
            return Err(ForgeFamilyLearningError::InvalidOutcomeCounts);
        }
        if derive_family_row_id(self) == self.id {
            Ok(())
        } else {
            Err(ForgeFamilyLearningError::RowIdentityMismatch)
        }
    }
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn derive_family_row_id(row: &ForgeFamilyOutcomeStats) -> ContentId {
    let mut parts = vec![row.family_id.as_content_id().as_str().as_bytes().to_vec()];
    parts.extend(row.counts.iter().map(|value| value.to_be_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-family-outcome-row.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Per-run statistics keyed by reusable generator-scoped family rather than concrete rewrite ID.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeFamilyOutcomeTable {
    id: ContentId,
    family_trial_set_id: ContentId,
    rows: Vec<ForgeFamilyOutcomeStats>,
}

impl ForgeFamilyOutcomeTable {
    pub fn from_trials(set: &ForgeFamilyTrialSet) -> Result<Self, ForgeFamilyLearningError> {
        set.validate()?;
        let mut grouped = BTreeMap::<ForgeTransformationFamilyId, OutcomeCounts>::new();
        for trial in set.trials() {
            grouped
                .entry(trial.family_id().clone())
                .or_default()
                .observe(trial.outcome())?;
        }
        let rows = grouped
            .into_iter()
            .map(|(family, counts)| ForgeFamilyOutcomeStats::from_counts(family, counts))
            .collect::<Result<Vec<_>, _>>()?;
        let id = derive_family_table_id(set.id(), &rows);
        let table = Self {
            id,
            family_trial_set_id: set.id().clone(),
            rows,
        };
        table.validate()?;
        Ok(table)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn family_trial_set_id(&self) -> &ContentId { &self.family_trial_set_id }
    pub fn rows(&self) -> &[ForgeFamilyOutcomeStats] { &self.rows }

    pub fn get(&self, family: &ForgeTransformationFamilyId) -> Option<&ForgeFamilyOutcomeStats> {
        self.rows
            .binary_search_by(|row| row.family_id.cmp(family))
            .ok()
            .map(|index| &self.rows[index])
    }

    pub fn validate(&self) -> Result<(), ForgeFamilyLearningError> {
        let mut previous: Option<&ForgeTransformationFamilyId> = None;
        for row in &self.rows {
            row.validate()?;
            if previous.is_some_and(|prior| prior >= &row.family_id) {
                return Err(ForgeFamilyLearningError::TableIdentityMismatch);
            }
            previous = Some(&row.family_id);
        }
        if derive_family_table_id(&self.family_trial_set_id, &self.rows) == self.id {
            Ok(())
        } else {
            Err(ForgeFamilyLearningError::TableIdentityMismatch)
        }
    }
}

fn derive_family_table_id(
    set_id: &ContentId,
    rows: &[ForgeFamilyOutcomeStats],
) -> ContentId {
    let count = (rows.len() as u64).to_be_bytes();
    let mut parts = vec![set_id.as_str().as_bytes().to_vec(), count.to_vec()];
    parts.extend(rows.iter().map(|row| row.id().as_str().as_bytes().to_vec()));
    ContentId::derive(
        "symthaea.forge-family-outcome-table.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::full_source_artifact_id;
    use crate::fitness::{Gate, GateResult};
    use crate::observations as forge_observations;
    use crate::trace::ForgeTraceEvent;
    use std::time::Duration;
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::ledger::DiscoveryEventKind;
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, DeterminismRequirement, DiscoveryRisk, ProblemSpec,
        SemanticGuarantee,
    };

    fn setup() -> (
        DiscoveryRun,
        ImplementationRecord,
        Vec<crate::trace::ForgeTraceEvent>,
        ObservationStore,
        ForgeTransformationFamilyId,
        Vec<TransformationId>,
    ) {
        let problem = ProblemSpec::new(
            "family-test",
            "Return exact reference value.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        ).unwrap();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "baseline",
            "family learning baseline",
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
        let generator = ContentId::derive("generator", [b"forge-v1".as_slice()]);
        let run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            generator.clone(),
            "abc123",
            SearchBudget::new(10, 1, 10).unwrap(),
            7,
        ).unwrap();
        let family = ForgeTransformationFamilyId::new(generator, "NumericLiteralPerturb").unwrap();

        let candidate_a = full_source_artifact_id("fn f() -> i32 { 2 }\n");
        let candidate_b = full_source_artifact_id("fn f() -> i32 { 3 }\n");
        let mutation_a = MutationRecord::new(
            0, "NumericLiteralPerturb", "1 -> 2", baseline_artifact.clone(), candidate_a,
        );
        let mutation_b = MutationRecord::new(
            0, "NumericLiteralPerturb", "1 -> 3", baseline_artifact.clone(), candidate_b,
        );
        assert_ne!(mutation_a.transformation_id, mutation_b.transformation_id);

        let attempt_a = ForgeAttemptId::derive(&baseline_artifact, run.seed, 0, 0);
        let attempt_b = ForgeAttemptId::derive(&baseline_artifact, run.seed, 1, 0);
        let generated_a = forge_observations::candidate_generated(&attempt_a, &mutation_a).unwrap();
        let generated_b = forge_observations::candidate_generated(&attempt_b, &mutation_b).unwrap();
        let compile_fail = vec![GateResult {
            gate: Gate::Compile,
            passed: false,
            output_tail: "compile error".into(),
            duration: Duration::from_nanos(1),
        }];
        let correctness_fail = vec![
            GateResult { gate: Gate::Compile, passed: true, output_tail: String::new(), duration: Duration::from_nanos(1) },
            GateResult { gate: Gate::Test, passed: false, output_tail: "counterexample".into(), duration: Duration::from_nanos(1) },
        ];
        let rejected_a = forge_observations::gates(&attempt_a, &mutation_a, &compile_fail).unwrap();
        let rejected_b = forge_observations::gates(&attempt_b, &mutation_b, &correctness_fail).unwrap();
        let summary = forge_observations::search_summary(2, 0, 1, 1, 0, 0, 0, None, None).unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt_a.clone(), 0, DiscoveryEventKind::CandidateGenerated,
                mutation_a.candidate_artifact_id.clone(), generated_a.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_a, 0, DiscoveryEventKind::RejectedCompilation,
                mutation_a.candidate_artifact_id.clone(), rejected_a.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_b.clone(), 0, DiscoveryEventKind::CandidateGenerated,
                mutation_b.candidate_artifact_id.clone(), generated_b.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_b, 0, DiscoveryEventKind::RejectedCorrectness,
                mutation_b.candidate_artifact_id.clone(), rejected_b.id().clone(),
            ),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![
            generated_a, rejected_a, generated_b, rejected_b, summary,
        ]).unwrap();
        (
            run,
            baseline,
            trace,
            observations,
            family,
            vec![mutation_a.transformation_id, mutation_b.transformation_id],
        )
    }

    #[test]
    fn distinct_concrete_rewrites_share_one_family_row() {
        let (run, baseline, trace, observations, family, concrete) = setup();
        assert_ne!(concrete[0], concrete[1]);
        let trials = ForgeFamilyTrialSet::from_trace(&run, &baseline, &trace, &observations).unwrap();
        assert_eq!(trials.trials().len(), 2);
        assert!(trials
            .trials()
            .iter()
            .all(|trial| trial.family_id() == &family));
        let table = ForgeFamilyOutcomeTable::from_trials(&trials).unwrap();
        assert_eq!(table.rows().len(), 1);
        let row = table.get(&family).unwrap();
        assert_eq!(row.trials(), 2);
        assert_eq!(row.rejected_compilation(), 1);
        assert_eq!(row.rejected_correctness(), 1);
    }

    #[test]
    fn generator_change_changes_family_identity() {
        let (_, _, _, _, family, _) = setup();
        let other = ForgeTransformationFamilyId::new(
            ContentId::derive("generator", [b"forge-v2".as_slice()]),
            family.operator(),
        ).unwrap();
        assert_ne!(family, other);
    }
}
