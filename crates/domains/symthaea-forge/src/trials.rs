// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical per-transformation trials derived from strongly validated Forge history.
//!
//! These records are descriptive learning inputs only. A local Forge outcome is not independent
//! correctness/performance evidence and does not grant promotion or runtime authority.

use crate::trace::{ForgeAttemptId, ForgeTraceEvent};
use crate::trial_semantics::{validate_forge_trial_semantics, ForgeTrialSemanticError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::ledger::DiscoveryEventKind;
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{ContentId, TransformationId};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeTrialOutcome {
    RejectedCompilation,
    RejectedCorrectness,
    RejectedEvaluation,
    ValidNotSelected,
    SelectedForContinuation,
    Interrupted,
}

impl ForgeTrialOutcome {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::RejectedCompilation => b"rejected-compilation",
            Self::RejectedCorrectness => b"rejected-correctness",
            Self::RejectedEvaluation => b"rejected-evaluation",
            Self::ValidNotSelected => b"valid-not-selected",
            Self::SelectedForContinuation => b"selected-for-continuation",
            Self::Interrupted => b"interrupted",
        }
    }
}

#[derive(Debug, Error)]
pub enum ForgeTrialError {
    #[error(transparent)]
    Semantics(#[from] ForgeTrialSemanticError),
    #[error("Forge trial observation payload is invalid JSON")]
    InvalidJson,
    #[error("Forge trial observation is missing a canonical content identity")]
    InvalidContentId,
    #[error("Forge trial has no matching generated candidate")]
    MissingGeneratedCandidate,
    #[error("Forge trial identity does not match its canonical fields")]
    TrialIdentityMismatch,
    #[error("Forge trial parent and candidate artifacts must differ")]
    NoArtifactChange,
    #[error("Forge trial attempt generation does not match the trial generation")]
    GenerationMismatch,
    #[error("non-interrupted Forge trial must bind a terminal attempt observation")]
    MissingTerminalObservation,
    #[error("interrupted Forge trial must not invent a terminal attempt observation")]
    InterruptedHasTerminalObservation,
    #[error("Forge trial set contains a duplicate attempt")]
    DuplicateAttempt,
    #[error("Forge trial set is not in strictly increasing attempt order")]
    NonCanonicalOrder,
    #[error("Forge trial set mixes baseline artifacts or search seeds")]
    MixedSearchIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformationTrial {
    id: ContentId,
    attempt_id: ForgeAttemptId,
    generation: u64,
    transformation_id: TransformationId,
    parent_artifact_id: ContentId,
    candidate_artifact_id: ContentId,
    outcome: ForgeTrialOutcome,
    generated_observation_id: ContentId,
    terminal_observation_id: Option<ContentId>,
}

impl TransformationTrial {
    #[allow(clippy::too_many_arguments)]
    fn new(
        attempt_id: ForgeAttemptId,
        generation: u64,
        transformation_id: TransformationId,
        parent_artifact_id: ContentId,
        candidate_artifact_id: ContentId,
        outcome: ForgeTrialOutcome,
        generated_observation_id: ContentId,
        terminal_observation_id: Option<ContentId>,
    ) -> Result<Self, ForgeTrialError> {
        if attempt_id.generation() != generation {
            return Err(ForgeTrialError::GenerationMismatch);
        }
        if parent_artifact_id == candidate_artifact_id {
            return Err(ForgeTrialError::NoArtifactChange);
        }
        match (outcome, terminal_observation_id.is_some()) {
            (ForgeTrialOutcome::Interrupted, true) => {
                return Err(ForgeTrialError::InterruptedHasTerminalObservation)
            }
            (ForgeTrialOutcome::Interrupted, false) => {}
            (_, false) => return Err(ForgeTrialError::MissingTerminalObservation),
            (_, true) => {}
        }
        let id = derive_trial_id(
            &attempt_id,
            generation,
            &transformation_id,
            &parent_artifact_id,
            &candidate_artifact_id,
            outcome,
            &generated_observation_id,
            terminal_observation_id.as_ref(),
        );
        Ok(Self {
            id,
            attempt_id,
            generation,
            transformation_id,
            parent_artifact_id,
            candidate_artifact_id,
            outcome,
            generated_observation_id,
            terminal_observation_id,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn attempt_id(&self) -> &ForgeAttemptId {
        &self.attempt_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn transformation_id(&self) -> &TransformationId {
        &self.transformation_id
    }

    pub fn parent_artifact_id(&self) -> &ContentId {
        &self.parent_artifact_id
    }

    pub fn candidate_artifact_id(&self) -> &ContentId {
        &self.candidate_artifact_id
    }

    pub fn outcome(&self) -> ForgeTrialOutcome {
        self.outcome
    }

    pub fn generated_observation_id(&self) -> &ContentId {
        &self.generated_observation_id
    }

    pub fn terminal_observation_id(&self) -> Option<&ContentId> {
        self.terminal_observation_id.as_ref()
    }

    pub fn validate(&self) -> Result<(), ForgeTrialError> {
        self.attempt_id
            .validate()
            .map_err(|error| ForgeTrialError::Semantics(ForgeTrialSemanticError::Trace(error.to_string())))?;
        let rebuilt = Self::new(
            self.attempt_id.clone(),
            self.generation,
            self.transformation_id.clone(),
            self.parent_artifact_id.clone(),
            self.candidate_artifact_id.clone(),
            self.outcome,
            self.generated_observation_id.clone(),
            self.terminal_observation_id.clone(),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeTrialError::TrialIdentityMismatch)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeTrialSet {
    id: ContentId,
    trials: Vec<TransformationTrial>,
}

impl ForgeTrialSet {
    pub fn from_trials(trials: Vec<TransformationTrial>) -> Result<Self, ForgeTrialError> {
        validate_trial_sequence(&trials)?;
        let id = derive_trial_set_id(&trials);
        Ok(Self { id, trials })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn trials(&self) -> &[TransformationTrial] {
        &self.trials
    }

    pub fn validate(&self) -> Result<(), ForgeTrialError> {
        validate_trial_sequence(&self.trials)?;
        if derive_trial_set_id(&self.trials) == self.id {
            Ok(())
        } else {
            Err(ForgeTrialError::TrialIdentityMismatch)
        }
    }
}

#[derive(Debug)]
struct OpenTrial {
    attempt_id: ForgeAttemptId,
    generation: u64,
    transformation_id: TransformationId,
    parent_artifact_id: ContentId,
    candidate_artifact_id: ContentId,
    generated_observation_id: ContentId,
}

/// Extract canonical concrete transformation trials from one Forge search history.
///
/// `GeneratorNoOp` attempts are intentionally omitted because no transformation was applied. If
/// the search aborts while a generated candidate remains open, the corresponding trial is retained
/// with `Interrupted` rather than receiving a fabricated rejection outcome.
pub fn extract_transformation_trials(
    trace: &[ForgeTraceEvent],
    observations: &ObservationStore,
) -> Result<ForgeTrialSet, ForgeTrialError> {
    validate_forge_trial_semantics(trace, observations)?;

    let mut open = BTreeMap::<String, OpenTrial>::new();
    let mut finished = BTreeMap::<u64, TransformationTrial>::new();

    for event in trace {
        match event.kind {
            DiscoveryEventKind::CandidateGenerated => {
                let attempt = event
                    .attempt_id
                    .as_ref()
                    .ok_or(ForgeTrialError::MissingGeneratedCandidate)?;
                let object = observations
                    .get(&event.observation_id)
                    .ok_or(ForgeTrialError::MissingGeneratedCandidate)?;
                let payload: Value = serde_json::from_slice(object.payload())
                    .map_err(|_| ForgeTrialError::InvalidJson)?;
                let transformation = parse_content_id(&payload, "transformation_id")?;
                let parent = parse_content_id(&payload, "parent_artifact_id")?;
                let candidate = event
                    .candidate_artifact_id
                    .as_ref()
                    .ok_or(ForgeTrialError::MissingGeneratedCandidate)?
                    .clone();
                open.insert(
                    attempt.as_content_id().as_str().to_string(),
                    OpenTrial {
                        attempt_id: attempt.clone(),
                        generation: event
                            .generation
                            .ok_or(ForgeTrialError::GenerationMismatch)?,
                        transformation_id: TransformationId(transformation),
                        parent_artifact_id: parent,
                        candidate_artifact_id: candidate,
                        generated_observation_id: event.observation_id.clone(),
                    },
                );
            }
            DiscoveryEventKind::RejectedCompilation
            | DiscoveryEventKind::RejectedCorrectness
            | DiscoveryEventKind::RejectedEvaluation
            | DiscoveryEventKind::ValidNotSelected
            | DiscoveryEventKind::SelectedForContinuation => {
                let attempt = event
                    .attempt_id
                    .as_ref()
                    .ok_or(ForgeTrialError::MissingGeneratedCandidate)?;
                let partial = open
                    .remove(attempt.as_content_id().as_str())
                    .ok_or(ForgeTrialError::MissingGeneratedCandidate)?;
                let outcome = match event.kind {
                    DiscoveryEventKind::RejectedCompilation => ForgeTrialOutcome::RejectedCompilation,
                    DiscoveryEventKind::RejectedCorrectness => ForgeTrialOutcome::RejectedCorrectness,
                    DiscoveryEventKind::RejectedEvaluation => ForgeTrialOutcome::RejectedEvaluation,
                    DiscoveryEventKind::ValidNotSelected => ForgeTrialOutcome::ValidNotSelected,
                    DiscoveryEventKind::SelectedForContinuation => {
                        ForgeTrialOutcome::SelectedForContinuation
                    }
                    _ => unreachable!("match arm is constrained to candidate terminal kinds"),
                };
                let trial = TransformationTrial::new(
                    partial.attempt_id,
                    partial.generation,
                    partial.transformation_id,
                    partial.parent_artifact_id,
                    partial.candidate_artifact_id,
                    outcome,
                    partial.generated_observation_id,
                    Some(event.observation_id.clone()),
                )?;
                finished.insert(trial.attempt_id().ordinal(), trial);
            }
            DiscoveryEventKind::SearchAborted => {
                for (_, partial) in std::mem::take(&mut open) {
                    let trial = TransformationTrial::new(
                        partial.attempt_id,
                        partial.generation,
                        partial.transformation_id,
                        partial.parent_artifact_id,
                        partial.candidate_artifact_id,
                        ForgeTrialOutcome::Interrupted,
                        partial.generated_observation_id,
                        None,
                    )?;
                    finished.insert(trial.attempt_id().ordinal(), trial);
                }
            }
            DiscoveryEventKind::GeneratorNoOp | DiscoveryEventKind::SearchCompleted => {}
            DiscoveryEventKind::CandidateArchived => {
                return Err(ForgeTrialError::MissingGeneratedCandidate)
            }
        }
    }

    ForgeTrialSet::from_trials(finished.into_values().collect())
}

fn parse_content_id(payload: &Value, field: &str) -> Result<ContentId, ForgeTrialError> {
    let raw = payload
        .get(field)
        .and_then(Value::as_str)
        .ok_or(ForgeTrialError::InvalidContentId)?;
    ContentId::parse(raw.to_string()).map_err(|_| ForgeTrialError::InvalidContentId)
}

#[allow(clippy::too_many_arguments)]
fn derive_trial_id(
    attempt_id: &ForgeAttemptId,
    generation: u64,
    transformation_id: &TransformationId,
    parent_artifact_id: &ContentId,
    candidate_artifact_id: &ContentId,
    outcome: ForgeTrialOutcome,
    generated_observation_id: &ContentId,
    terminal_observation_id: Option<&ContentId>,
) -> ContentId {
    let generation_bytes = generation.to_be_bytes();
    ContentId::derive(
        "symthaea.forge-transformation-trial.v1",
        [
            attempt_id.as_content_id().as_str().as_bytes(),
            generation_bytes.as_slice(),
            transformation_id.as_content_id().as_str().as_bytes(),
            parent_artifact_id.as_str().as_bytes(),
            candidate_artifact_id.as_str().as_bytes(),
            outcome.tag(),
            generated_observation_id.as_str().as_bytes(),
            terminal_observation_id
                .map(ContentId::as_str)
                .unwrap_or("")
                .as_bytes(),
        ],
    )
}

fn derive_trial_set_id(trials: &[TransformationTrial]) -> ContentId {
    let count = (trials.len() as u64).to_be_bytes();
    let mut parts = vec![count.to_vec()];
    parts.extend(
        trials
            .iter()
            .map(|trial| trial.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-transformation-trial-set.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn validate_trial_sequence(trials: &[TransformationTrial]) -> Result<(), ForgeTrialError> {
    let mut seen = BTreeSet::new();
    let mut previous_ordinal = None;
    let mut search_identity: Option<(&ContentId, u64)> = None;

    for trial in trials {
        trial.validate()?;
        let attempt = trial.attempt_id();
        if !seen.insert(attempt.as_content_id().as_str()) {
            return Err(ForgeTrialError::DuplicateAttempt);
        }
        if previous_ordinal.is_some_and(|previous| attempt.ordinal() <= previous) {
            return Err(ForgeTrialError::NonCanonicalOrder);
        }
        previous_ordinal = Some(attempt.ordinal());
        match search_identity {
            Some((baseline, seed))
                if baseline != attempt.baseline_artifact_id() || seed != attempt.seed() =>
            {
                return Err(ForgeTrialError::MixedSearchIdentity)
            }
            None => {
                search_identity = Some((attempt.baseline_artifact_id(), attempt.seed()));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{full_source_artifact_id, MutationRecord};
    use crate::fitness::{Gate, GateResult};
    use crate::observations as forge_observations;
    use std::time::Duration;

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
                duration: Duration::from_nanos(2),
            },
        ]
    }

    fn generated_fixture() -> (ForgeAttemptId, MutationRecord) {
        let baseline = full_source_artifact_id("fn f() -> i32 { 1 }\n");
        let candidate = full_source_artifact_id("fn f() -> i32 { 2 }\n");
        (
            ForgeAttemptId::derive(&baseline, 9, 0, 0),
            MutationRecord::new(0, "literal", "1 -> 2", baseline, candidate),
        )
    }

    #[test]
    fn rejected_candidate_becomes_one_canonical_trial() {
        let (attempt, mutation) = generated_fixture();
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
        let set = extract_transformation_trials(&trace, &observations).unwrap();
        assert_eq!(set.trials().len(), 1);
        assert_eq!(set.trials()[0].outcome(), ForgeTrialOutcome::RejectedCorrectness);
        assert!(set.trials()[0].terminal_observation_id().is_some());
        assert!(set.validate().is_ok());
    }

    #[test]
    fn aborted_open_candidate_becomes_interrupted_not_rejected() {
        let (attempt, mutation) = generated_fixture();
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let abort = forge_observations::search_abort(
            "candidate-gate-apparatus",
            "runner lost",
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
                attempt.clone(),
                0,
                DiscoveryEventKind::CandidateGenerated,
                mutation.candidate_artifact_id.clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::aborted(abort.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![generated, abort]).unwrap();
        let set = extract_transformation_trials(&trace, &observations).unwrap();
        assert_eq!(set.trials().len(), 1);
        assert_eq!(set.trials()[0].outcome(), ForgeTrialOutcome::Interrupted);
        assert!(set.trials()[0].terminal_observation_id().is_none());
    }
}
