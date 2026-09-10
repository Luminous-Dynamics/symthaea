// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generation-boundary sequence memory for Forge transformation-family search.
//!
//! Forge evaluates sibling candidates against one current parent and chooses a generation winner
//! only after those siblings have been evaluated. Attempt ordinal therefore must not be interpreted
//! as immediate state transition order. This module reconstructs the accepted lineage at generation
//! boundaries and conditions every trial in generation `g` on the same history entering `g`.
//!
//! These records are descriptive history only. They do not steer search, establish correctness or
//! superiority, authorize promotion, or make discovered code callable.

use crate::family_learning::{
    ForgeFamilyLearningError, ForgeFamilyTrial, ForgeFamilyTrialSet, ForgeTransformationFamilyId,
};
use crate::learning::{ForgeLearningError, ForgeTrialBatch};
use crate::trace::{ForgeAttemptId, ForgeTraceEvent};
use crate::trials::{ForgeTrialOutcome, TransformationTrial};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::discovery::DiscoveryRun;
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{ContentId, ImplementationRecord, TransformationId};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeSequenceLearningError {
    #[error(transparent)]
    Learning(#[from] ForgeLearningError),
    #[error(transparent)]
    Family(#[from] ForgeFamilyLearningError),
    #[error("family trial set was not derived from the same run-bound concrete trial batch")]
    SourceBatchMismatch,
    #[error("family trial has no exact concrete transformation trial")]
    MissingConcreteTrial,
    #[error("family and concrete trial identities disagree")]
    ConcreteTrialMismatch,
    #[error("concrete trial appears more than once in the run-bound batch")]
    DuplicateConcreteTrial,
    #[error("attempt generations are not monotone in canonical attempt order")]
    GenerationOrderMismatch,
    #[error("candidate parent artifact does not equal the accepted parent entering its generation")]
    GenerationParentMismatch,
    #[error("a Forge generation contains more than one SelectedForContinuation trial")]
    MultipleSelectionsPerGeneration,
    #[error("family history contains a family from a different generator")]
    HistoryGeneratorMismatch,
    #[error("family history identity does not match canonical fields")]
    HistoryIdentityMismatch,
    #[error("conditioned family trial identity does not match canonical fields")]
    ConditionedTrialIdentityMismatch,
    #[error("accepted family step identity does not match canonical fields")]
    AcceptedStepIdentityMismatch,
    #[error("search family sequence identity or continuity is invalid")]
    SequenceIdentityMismatch,
}

/// Ordered accepted transformation-family history under one exact generator implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeFamilyHistory {
    id: ContentId,
    generator_id: ContentId,
    families: Vec<ForgeTransformationFamilyId>,
}

impl ForgeFamilyHistory {
    pub fn new(
        generator_id: ContentId,
        families: Vec<ForgeTransformationFamilyId>,
    ) -> Result<Self, ForgeSequenceLearningError> {
        if families
            .iter()
            .any(|family| family.generator_id() != &generator_id)
        {
            return Err(ForgeSequenceLearningError::HistoryGeneratorMismatch);
        }
        for family in &families {
            family.validate()?;
        }
        let id = derive_history_id(&generator_id, &families);
        Ok(Self {
            id,
            generator_id,
            families,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn generator_id(&self) -> &ContentId { &self.generator_id }
    pub fn families(&self) -> &[ForgeTransformationFamilyId] { &self.families }
    pub fn len(&self) -> usize { self.families.len() }
    pub fn is_empty(&self) -> bool { self.families.is_empty() }

    /// Return the last `depth` accepted families as a separately content-addressed history key.
    /// This is a view for later bounded-order models; it grants no search authority.
    pub fn suffix(&self, depth: usize) -> Result<Self, ForgeSequenceLearningError> {
        let start = self.families.len().saturating_sub(depth);
        Self::new(self.generator_id.clone(), self.families[start..].to_vec())
    }

    pub fn validate(&self) -> Result<(), ForgeSequenceLearningError> {
        let rebuilt = Self::new(self.generator_id.clone(), self.families.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeSequenceLearningError::HistoryIdentityMismatch)
        }
    }
}

fn derive_history_id(
    generator_id: &ContentId,
    families: &[ForgeTransformationFamilyId],
) -> ContentId {
    let count = (families.len() as u64).to_be_bytes();
    let mut parts = vec![generator_id.as_str().as_bytes().to_vec(), count.to_vec()];
    parts.extend(
        families
            .iter()
            .map(|family| family.as_content_id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-family-history.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// One exact family trial conditioned on the accepted family history entering its generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeConditionedFamilyTrial {
    id: ContentId,
    run_id: ContentId,
    attempt_id: ForgeAttemptId,
    generation: u64,
    family_trial_id: ContentId,
    concrete_trial_id: ContentId,
    family_id: ForgeTransformationFamilyId,
    parent_artifact_id: ContentId,
    history_before: ForgeFamilyHistory,
    outcome: ForgeTrialOutcome,
}

impl ForgeConditionedFamilyTrial {
    fn new(
        run_id: ContentId,
        family_trial: &ForgeFamilyTrial,
        concrete: &TransformationTrial,
        history_before: ForgeFamilyHistory,
    ) -> Result<Self, ForgeSequenceLearningError> {
        if family_trial.attempt_id() != concrete.attempt_id()
            || family_trial.concrete_trial_id() != concrete.id()
            || family_trial.concrete_transformation_id() != concrete.transformation_id()
            || family_trial.outcome() != concrete.outcome()
            || family_trial.family_id().generator_id() != history_before.generator_id()
        {
            return Err(ForgeSequenceLearningError::ConcreteTrialMismatch);
        }
        let id = derive_conditioned_trial_id(
            &run_id,
            family_trial,
            concrete,
            &history_before,
        );
        Ok(Self {
            id,
            run_id,
            attempt_id: family_trial.attempt_id().clone(),
            generation: concrete.generation(),
            family_trial_id: family_trial.id().clone(),
            concrete_trial_id: concrete.id().clone(),
            family_id: family_trial.family_id().clone(),
            parent_artifact_id: concrete.parent_artifact_id().clone(),
            history_before,
            outcome: concrete.outcome(),
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn attempt_id(&self) -> &ForgeAttemptId { &self.attempt_id }
    pub fn generation(&self) -> u64 { self.generation }
    pub fn family_trial_id(&self) -> &ContentId { &self.family_trial_id }
    pub fn concrete_trial_id(&self) -> &ContentId { &self.concrete_trial_id }
    pub fn family_id(&self) -> &ForgeTransformationFamilyId { &self.family_id }
    pub fn parent_artifact_id(&self) -> &ContentId { &self.parent_artifact_id }
    pub fn history_before(&self) -> &ForgeFamilyHistory { &self.history_before }
    pub fn outcome(&self) -> ForgeTrialOutcome { self.outcome }

    pub fn validate(&self) -> Result<(), ForgeSequenceLearningError> {
        self.attempt_id
            .validate()
            .map_err(|_| ForgeSequenceLearningError::ConditionedTrialIdentityMismatch)?;
        self.family_id.validate()?;
        self.history_before.validate()?;
        if self.attempt_id.generation() != self.generation
            || self.family_id.generator_id() != self.history_before.generator_id()
        {
            return Err(ForgeSequenceLearningError::ConditionedTrialIdentityMismatch);
        }
        let expected = derive_conditioned_trial_id_fields(
            &self.run_id,
            &self.attempt_id,
            self.generation,
            &self.family_trial_id,
            &self.concrete_trial_id,
            &self.family_id,
            &self.parent_artifact_id,
            &self.history_before,
            self.outcome,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeSequenceLearningError::ConditionedTrialIdentityMismatch)
        }
    }
}

fn derive_conditioned_trial_id(
    run_id: &ContentId,
    family_trial: &ForgeFamilyTrial,
    concrete: &TransformationTrial,
    history: &ForgeFamilyHistory,
) -> ContentId {
    derive_conditioned_trial_id_fields(
        run_id,
        family_trial.attempt_id(),
        concrete.generation(),
        family_trial.id(),
        concrete.id(),
        family_trial.family_id(),
        concrete.parent_artifact_id(),
        history,
        concrete.outcome(),
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_conditioned_trial_id_fields(
    run_id: &ContentId,
    attempt_id: &ForgeAttemptId,
    generation: u64,
    family_trial_id: &ContentId,
    concrete_trial_id: &ContentId,
    family_id: &ForgeTransformationFamilyId,
    parent_artifact_id: &ContentId,
    history: &ForgeFamilyHistory,
    outcome: ForgeTrialOutcome,
) -> ContentId {
    let generation = generation.to_be_bytes();
    ContentId::derive(
        "symthaea.forge-conditioned-family-trial.v1",
        [
            run_id.as_str().as_bytes(),
            attempt_id.as_content_id().as_str().as_bytes(),
            generation.as_slice(),
            family_trial_id.as_str().as_bytes(),
            concrete_trial_id.as_str().as_bytes(),
            family_id.as_content_id().as_str().as_bytes(),
            parent_artifact_id.as_str().as_bytes(),
            history.id().as_str().as_bytes(),
            outcome_tag(outcome),
        ],
    )
}

/// Exact accepted state transition at one generation boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeAcceptedFamilyStep {
    id: ContentId,
    run_id: ContentId,
    generation: u64,
    family_trial_id: ContentId,
    concrete_trial_id: ContentId,
    family_id: ForgeTransformationFamilyId,
    concrete_transformation_id: TransformationId,
    parent_artifact_id: ContentId,
    candidate_artifact_id: ContentId,
}

impl ForgeAcceptedFamilyStep {
    fn new(
        run_id: ContentId,
        family_trial: &ForgeFamilyTrial,
        concrete: &TransformationTrial,
    ) -> Result<Self, ForgeSequenceLearningError> {
        if family_trial.outcome() != ForgeTrialOutcome::SelectedForContinuation
            || concrete.outcome() != ForgeTrialOutcome::SelectedForContinuation
            || family_trial.attempt_id() != concrete.attempt_id()
            || family_trial.concrete_trial_id() != concrete.id()
            || family_trial.concrete_transformation_id() != concrete.transformation_id()
        {
            return Err(ForgeSequenceLearningError::ConcreteTrialMismatch);
        }
        let id = derive_step_id(
            &run_id,
            concrete.generation(),
            family_trial.id(),
            concrete.id(),
            family_trial.family_id(),
            concrete.transformation_id(),
            concrete.parent_artifact_id(),
            concrete.candidate_artifact_id(),
        );
        Ok(Self {
            id,
            run_id,
            generation: concrete.generation(),
            family_trial_id: family_trial.id().clone(),
            concrete_trial_id: concrete.id().clone(),
            family_id: family_trial.family_id().clone(),
            concrete_transformation_id: concrete.transformation_id().clone(),
            parent_artifact_id: concrete.parent_artifact_id().clone(),
            candidate_artifact_id: concrete.candidate_artifact_id().clone(),
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn generation(&self) -> u64 { self.generation }
    pub fn family_trial_id(&self) -> &ContentId { &self.family_trial_id }
    pub fn concrete_trial_id(&self) -> &ContentId { &self.concrete_trial_id }
    pub fn family_id(&self) -> &ForgeTransformationFamilyId { &self.family_id }
    pub fn concrete_transformation_id(&self) -> &TransformationId { &self.concrete_transformation_id }
    pub fn parent_artifact_id(&self) -> &ContentId { &self.parent_artifact_id }
    pub fn candidate_artifact_id(&self) -> &ContentId { &self.candidate_artifact_id }

    pub fn validate(&self) -> Result<(), ForgeSequenceLearningError> {
        self.family_id.validate()?;
        let expected = derive_step_id(
            &self.run_id,
            self.generation,
            &self.family_trial_id,
            &self.concrete_trial_id,
            &self.family_id,
            &self.concrete_transformation_id,
            &self.parent_artifact_id,
            &self.candidate_artifact_id,
        );
        if self.parent_artifact_id == self.candidate_artifact_id || expected != self.id {
            Err(ForgeSequenceLearningError::AcceptedStepIdentityMismatch)
        } else {
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_step_id(
    run_id: &ContentId,
    generation: u64,
    family_trial_id: &ContentId,
    concrete_trial_id: &ContentId,
    family_id: &ForgeTransformationFamilyId,
    transformation_id: &TransformationId,
    parent_artifact_id: &ContentId,
    candidate_artifact_id: &ContentId,
) -> ContentId {
    let generation = generation.to_be_bytes();
    ContentId::derive(
        "symthaea.forge-accepted-family-step.v1",
        [
            run_id.as_str().as_bytes(),
            generation.as_slice(),
            family_trial_id.as_str().as_bytes(),
            concrete_trial_id.as_str().as_bytes(),
            family_id.as_content_id().as_str().as_bytes(),
            transformation_id.as_content_id().as_str().as_bytes(),
            parent_artifact_id.as_str().as_bytes(),
            candidate_artifact_id.as_str().as_bytes(),
        ],
    )
}

#[derive(Clone)]
struct PairedTrial {
    family: ForgeFamilyTrial,
    concrete: TransformationTrial,
}

/// Canonical reconstruction of Forge's accepted family lineage and generation-conditioned trials.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeSearchFamilySequence {
    id: ContentId,
    run_id: ContentId,
    generator_id: ContentId,
    baseline_artifact_id: ContentId,
    source_batch_id: ContentId,
    source_family_trial_set_id: ContentId,
    conditioned_trials: Vec<ForgeConditionedFamilyTrial>,
    accepted_steps: Vec<ForgeAcceptedFamilyStep>,
    final_artifact_id: ContentId,
}

impl ForgeSearchFamilySequence {
    pub fn from_trace(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        trace: &[ForgeTraceEvent],
        observations: &ObservationStore,
    ) -> Result<Self, ForgeSequenceLearningError> {
        let batch = ForgeTrialBatch::from_trace(run, baseline, trace, observations)?;
        let family_set = ForgeFamilyTrialSet::from_trace(run, baseline, trace, observations)?;
        if family_set.source_batch_id() != batch.id() {
            return Err(ForgeSequenceLearningError::SourceBatchMismatch);
        }

        let mut concrete_by_id = BTreeMap::<String, TransformationTrial>::new();
        for trial in batch.trials().trials() {
            if concrete_by_id
                .insert(trial.id().as_str().to_string(), trial.clone())
                .is_some()
            {
                return Err(ForgeSequenceLearningError::DuplicateConcreteTrial);
            }
        }

        let mut paired = Vec::with_capacity(family_set.trials().len());
        let mut previous_generation = None;
        for family in family_set.trials() {
            let concrete = concrete_by_id
                .get(family.concrete_trial_id().as_str())
                .cloned()
                .ok_or(ForgeSequenceLearningError::MissingConcreteTrial)?;
            if family.attempt_id() != concrete.attempt_id()
                || family.concrete_transformation_id() != concrete.transformation_id()
                || family.outcome() != concrete.outcome()
            {
                return Err(ForgeSequenceLearningError::ConcreteTrialMismatch);
            }
            if previous_generation.is_some_and(|generation| concrete.generation() < generation) {
                return Err(ForgeSequenceLearningError::GenerationOrderMismatch);
            }
            previous_generation = Some(concrete.generation());
            paired.push(PairedTrial {
                family: family.clone(),
                concrete,
            });
        }

        let mut by_generation = BTreeMap::<u64, Vec<PairedTrial>>::new();
        for pair in paired {
            by_generation
                .entry(pair.concrete.generation())
                .or_default()
                .push(pair);
        }

        let mut current_parent = baseline.artifact_id.clone();
        let mut accepted_families = Vec::<ForgeTransformationFamilyId>::new();
        let mut conditioned_trials = Vec::new();
        let mut accepted_steps = Vec::new();

        for (generation, generation_trials) in by_generation {
            let history = ForgeFamilyHistory::new(
                run.generator_id.clone(),
                accepted_families.clone(),
            )?;
            let selected = generation_trials
                .iter()
                .filter(|pair| pair.family.outcome() == ForgeTrialOutcome::SelectedForContinuation)
                .collect::<Vec<_>>();
            if selected.len() > 1 {
                return Err(ForgeSequenceLearningError::MultipleSelectionsPerGeneration);
            }

            for pair in &generation_trials {
                if pair.concrete.parent_artifact_id() != &current_parent {
                    return Err(ForgeSequenceLearningError::GenerationParentMismatch);
                }
                conditioned_trials.push(ForgeConditionedFamilyTrial::new(
                    run.id.clone(),
                    &pair.family,
                    &pair.concrete,
                    history.clone(),
                )?);
            }

            if let Some(winner) = selected.first() {
                let step = ForgeAcceptedFamilyStep::new(
                    run.id.clone(),
                    &winner.family,
                    &winner.concrete,
                )?;
                if step.generation() != generation || step.parent_artifact_id() != &current_parent {
                    return Err(ForgeSequenceLearningError::GenerationParentMismatch);
                }
                current_parent = step.candidate_artifact_id().clone();
                accepted_families.push(step.family_id().clone());
                accepted_steps.push(step);
            }
        }

        conditioned_trials.sort_by_key(|trial| trial.attempt_id().ordinal());
        accepted_steps.sort_by_key(|step| step.generation());
        let id = derive_sequence_id(
            &run.id,
            &run.generator_id,
            &baseline.artifact_id,
            batch.id(),
            family_set.id(),
            &conditioned_trials,
            &accepted_steps,
            &current_parent,
        );
        let sequence = Self {
            id,
            run_id: run.id.clone(),
            generator_id: run.generator_id.clone(),
            baseline_artifact_id: baseline.artifact_id.clone(),
            source_batch_id: batch.id().clone(),
            source_family_trial_set_id: family_set.id().clone(),
            conditioned_trials,
            accepted_steps,
            final_artifact_id: current_parent,
        };
        sequence.validate()?;
        Ok(sequence)
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn generator_id(&self) -> &ContentId { &self.generator_id }
    pub fn baseline_artifact_id(&self) -> &ContentId { &self.baseline_artifact_id }
    pub fn source_batch_id(&self) -> &ContentId { &self.source_batch_id }
    pub fn source_family_trial_set_id(&self) -> &ContentId { &self.source_family_trial_set_id }
    pub fn conditioned_trials(&self) -> &[ForgeConditionedFamilyTrial] { &self.conditioned_trials }
    pub fn accepted_steps(&self) -> &[ForgeAcceptedFamilyStep] { &self.accepted_steps }
    pub fn final_artifact_id(&self) -> &ContentId { &self.final_artifact_id }

    pub fn validate(&self) -> Result<(), ForgeSequenceLearningError> {
        let mut seen_attempts = BTreeSet::new();
        let mut previous_ordinal = None;
        let mut previous_generation = None;
        let mut current_parent = self.baseline_artifact_id.clone();
        let mut accepted = Vec::<ForgeTransformationFamilyId>::new();
        let mut selected_by_generation = BTreeMap::<u64, &ForgeAcceptedFamilyStep>::new();

        for step in &self.accepted_steps {
            step.validate()?;
            if step.run_id() != &self.run_id || step.family_id().generator_id() != &self.generator_id {
                return Err(ForgeSequenceLearningError::SequenceIdentityMismatch);
            }
            if selected_by_generation.insert(step.generation(), step).is_some() {
                return Err(ForgeSequenceLearningError::MultipleSelectionsPerGeneration);
            }
        }

        let mut grouped = BTreeMap::<u64, Vec<&ForgeConditionedFamilyTrial>>::new();
        for trial in &self.conditioned_trials {
            trial.validate()?;
            if trial.run_id() != &self.run_id
                || trial.family_id().generator_id() != &self.generator_id
                || !seen_attempts.insert(trial.attempt_id().as_content_id().as_str().to_string())
                || previous_ordinal.is_some_and(|ordinal| trial.attempt_id().ordinal() <= ordinal)
                || previous_generation.is_some_and(|generation| trial.generation() < generation)
            {
                return Err(ForgeSequenceLearningError::SequenceIdentityMismatch);
            }
            previous_ordinal = Some(trial.attempt_id().ordinal());
            previous_generation = Some(trial.generation());
            grouped.entry(trial.generation()).or_default().push(trial);
        }

        let mut selected_count = 0usize;
        for (generation, trials) in grouped {
            let expected_history = ForgeFamilyHistory::new(self.generator_id.clone(), accepted.clone())?;
            let selected_trials = trials
                .iter()
                .filter(|trial| trial.outcome() == ForgeTrialOutcome::SelectedForContinuation)
                .collect::<Vec<_>>();
            if selected_trials.len() > 1 {
                return Err(ForgeSequenceLearningError::MultipleSelectionsPerGeneration);
            }
            for trial in &trials {
                if trial.parent_artifact_id() != &current_parent
                    || trial.history_before() != &expected_history
                {
                    return Err(ForgeSequenceLearningError::GenerationParentMismatch);
                }
            }
            match (selected_trials.first(), selected_by_generation.get(&generation)) {
                (None, None) => {}
                (Some(trial), Some(step)) => {
                    if step.family_trial_id() != trial.family_trial_id()
                        || step.concrete_trial_id() != trial.concrete_trial_id()
                        || step.family_id() != trial.family_id()
                        || step.parent_artifact_id() != &current_parent
                    {
                        return Err(ForgeSequenceLearningError::SequenceIdentityMismatch);
                    }
                    selected_count += 1;
                    current_parent = step.candidate_artifact_id().clone();
                    accepted.push(step.family_id().clone());
                }
                _ => return Err(ForgeSequenceLearningError::SequenceIdentityMismatch),
            }
        }
        if selected_count != self.accepted_steps.len() || current_parent != self.final_artifact_id {
            return Err(ForgeSequenceLearningError::SequenceIdentityMismatch);
        }

        let expected = derive_sequence_id(
            &self.run_id,
            &self.generator_id,
            &self.baseline_artifact_id,
            &self.source_batch_id,
            &self.source_family_trial_set_id,
            &self.conditioned_trials,
            &self.accepted_steps,
            &self.final_artifact_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeSequenceLearningError::SequenceIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_sequence_id(
    run_id: &ContentId,
    generator_id: &ContentId,
    baseline_artifact_id: &ContentId,
    source_batch_id: &ContentId,
    source_family_trial_set_id: &ContentId,
    conditioned_trials: &[ForgeConditionedFamilyTrial],
    accepted_steps: &[ForgeAcceptedFamilyStep],
    final_artifact_id: &ContentId,
) -> ContentId {
    let trial_count = (conditioned_trials.len() as u64).to_be_bytes();
    let step_count = (accepted_steps.len() as u64).to_be_bytes();
    let mut parts = vec![
        run_id.as_str().as_bytes().to_vec(),
        generator_id.as_str().as_bytes().to_vec(),
        baseline_artifact_id.as_str().as_bytes().to_vec(),
        source_batch_id.as_str().as_bytes().to_vec(),
        source_family_trial_set_id.as_str().as_bytes().to_vec(),
        trial_count.to_vec(),
    ];
    parts.extend(
        conditioned_trials
            .iter()
            .map(|trial| trial.id().as_str().as_bytes().to_vec()),
    );
    parts.push(step_count.to_vec());
    parts.extend(
        accepted_steps
            .iter()
            .map(|step| step.id().as_str().as_bytes().to_vec()),
    );
    parts.push(final_artifact_id.as_str().as_bytes().to_vec());
    ContentId::derive(
        "symthaea.forge-search-family-sequence.v1",
        parts.iter().map(Vec::as_slice),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{full_source_artifact_id, MutationRecord};
    use crate::fitness::{Gate, GateResult};
    use crate::observations as forge_observations;
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::ledger::DiscoveryEventKind;
    use symthaea_algorithms::observation::{ObservationEncoding, ObservationObject};
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, DeterminismRequirement, DiscoveryRisk, ProblemSpec,
        SemanticGuarantee,
    };
    use std::time::Duration;

    fn candidate_decision(
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

    fn local_rejection(
        attempt: &ForgeAttemptId,
        mutation: &MutationRecord,
    ) -> ObservationObject {
        forge_observations::selection(
            attempt,
            mutation,
            None,
            None,
            "not-better-than-parent",
        )
        .unwrap()
    }

    fn compile_rejection(
        attempt: &ForgeAttemptId,
        mutation: &MutationRecord,
    ) -> ObservationObject {
        forge_observations::gates(
            attempt,
            mutation,
            &[GateResult {
                gate: Gate::Compile,
                passed: false,
                output_tail: "compile error".into(),
                duration: Duration::from_nanos(1),
            }],
        )
        .unwrap()
    }

    #[test]
    fn sibling_trials_share_pre_generation_history_and_next_generation_sees_winner() {
        let problem = ProblemSpec::new(
            "sequence-test",
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
            "sequence baseline",
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
        let generator = ContentId::derive("generator", [b"forge-v1".as_slice()]);
        let run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            generator.clone(),
            "abc123",
            SearchBudget::new(10, 2, 10).unwrap(),
            7,
        )
        .unwrap();

        let a_artifact = full_source_artifact_id("fn f() -> i32 { 2 }\n");
        let b_artifact = full_source_artifact_id("fn f() -> i32 { 3 }\n");
        let c_artifact = full_source_artifact_id("fn f() -> i32 { 4 }\n");
        let mutation_a = MutationRecord::new(
            0,
            "NumericLiteralPerturb",
            "1 -> 2",
            baseline_artifact.clone(),
            a_artifact.clone(),
        );
        let mutation_b = MutationRecord::new(
            0,
            "AlternateLiteral",
            "1 -> 3",
            baseline_artifact.clone(),
            b_artifact.clone(),
        );
        let mutation_c = MutationRecord::new(
            1,
            "NumericLiteralPerturb",
            "2 -> 4",
            a_artifact.clone(),
            c_artifact.clone(),
        );
        let attempt_a = ForgeAttemptId::derive(&baseline_artifact, 7, 0, 0);
        let attempt_b = ForgeAttemptId::derive(&baseline_artifact, 7, 1, 0);
        let attempt_c = ForgeAttemptId::derive(&baseline_artifact, 7, 2, 1);

        let generated_a = forge_observations::candidate_generated(&attempt_a, &mutation_a).unwrap();
        let generated_b = forge_observations::candidate_generated(&attempt_b, &mutation_b).unwrap();
        let generated_c = forge_observations::candidate_generated(&attempt_c, &mutation_c).unwrap();
        let rejected_b = local_rejection(&attempt_b, &mutation_b);
        let selected_a = candidate_decision(&attempt_a, &mutation_a, "selected-for-next-generation");
        let rejected_c = compile_rejection(&attempt_c, &mutation_c);
        let summary = forge_observations::search_summary(3, 0, 1, 0, 0, 2, 1, None, None).unwrap();

        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt_a.clone(), 0, DiscoveryEventKind::CandidateGenerated,
                a_artifact.clone(), generated_a.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_b.clone(), 0, DiscoveryEventKind::CandidateGenerated,
                b_artifact.clone(), generated_b.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_b, 0, DiscoveryEventKind::ValidNotSelected,
                b_artifact, rejected_b.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_a, 0, DiscoveryEventKind::SelectedForContinuation,
                a_artifact.clone(), selected_a.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_c.clone(), 1, DiscoveryEventKind::CandidateGenerated,
                c_artifact.clone(), generated_c.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_c, 1, DiscoveryEventKind::RejectedCompilation,
                c_artifact, rejected_c.id().clone(),
            ),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![
            generated_a,
            generated_b,
            rejected_b,
            selected_a,
            generated_c,
            rejected_c,
            summary,
        ])
        .unwrap();

        let sequence = ForgeSearchFamilySequence::from_trace(
            &run,
            &baseline,
            &trace,
            &observations,
        )
        .unwrap();
        assert_eq!(sequence.conditioned_trials().len(), 3);
        assert!(sequence.conditioned_trials()[0].history_before().is_empty());
        assert!(sequence.conditioned_trials()[1].history_before().is_empty());
        assert_eq!(sequence.conditioned_trials()[2].history_before().len(), 1);
        assert_eq!(
            sequence.conditioned_trials()[2].history_before().families()[0].operator(),
            "NumericLiteralPerturb"
        );
        assert_eq!(sequence.conditioned_trials()[2].parent_artifact_id(), &a_artifact);
        assert_eq!(sequence.accepted_steps().len(), 1);
        assert_eq!(sequence.final_artifact_id(), &a_artifact);
        assert!(sequence.validate().is_ok());
    }

    #[test]
    fn family_history_suffix_is_generator_bound_and_ordered() {
        let generator = ContentId::derive("generator", [b"forge-v1".as_slice()]);
        let a = ForgeTransformationFamilyId::new(generator.clone(), "A").unwrap();
        let b = ForgeTransformationFamilyId::new(generator.clone(), "B").unwrap();
        let c = ForgeTransformationFamilyId::new(generator.clone(), "C").unwrap();
        let history = ForgeFamilyHistory::new(generator, vec![a, b.clone(), c.clone()]).unwrap();
        let suffix = history.suffix(2).unwrap();
        assert_eq!(suffix.families(), &[b, c]);
        assert_ne!(suffix.id(), history.id());
    }
}
