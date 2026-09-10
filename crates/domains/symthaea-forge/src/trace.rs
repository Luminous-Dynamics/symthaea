// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generator-local trace events that can later be bound to a semantic `DiscoveryRun`.
//!
//! Forge itself does not know the semantic problem contract, so it cannot directly mint a
//! `DiscoveryLedger`. Every local search-loop attempt has a deterministic, self-validating
//! [`ForgeAttemptId`]. Candidate generation and its terminal outcome must carry the same attempt ID,
//! removing ambiguity when identical candidate bytes are generated more than once.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use symthaea_algorithms::ledger::DiscoveryEventKind;
use symthaea_algorithms::observation::{ObservationEncoding, ObservationStore};
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeAttemptId {
    id: ContentId,
    baseline_artifact_id: ContentId,
    seed: u64,
    ordinal: u64,
    generation: u64,
}

impl ForgeAttemptId {
    pub fn derive(
        baseline_artifact_id: &ContentId,
        seed: u64,
        ordinal: u64,
        generation: u64,
    ) -> Self {
        let seed_bytes = seed.to_be_bytes();
        let ordinal_bytes = ordinal.to_be_bytes();
        let generation_bytes = generation.to_be_bytes();
        let id = ContentId::derive(
            "symthaea.forge-attempt.v1",
            [
                baseline_artifact_id.as_str().as_bytes(),
                seed_bytes.as_slice(),
                ordinal_bytes.as_slice(),
                generation_bytes.as_slice(),
            ],
        );
        Self {
            id,
            baseline_artifact_id: baseline_artifact_id.clone(),
            seed,
            ordinal,
            generation,
        }
    }

    pub fn as_content_id(&self) -> &ContentId {
        &self.id
    }

    pub fn baseline_artifact_id(&self) -> &ContentId {
        &self.baseline_artifact_id
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn ordinal(&self) -> u64 {
        self.ordinal
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn validate(&self) -> Result<(), ForgeTraceError> {
        let rebuilt = Self::derive(
            &self.baseline_artifact_id,
            self.seed,
            self.ordinal,
            self.generation,
        );
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeTraceError::AttemptIdentityMismatch)
        }
    }
}

impl fmt::Display for ForgeAttemptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.id.fmt(f)
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ForgeTraceError {
    #[error("Forge trace event has an invalid local shape")]
    InvalidEventShape,
    #[error("Forge attempt identity does not match its canonical fields")]
    AttemptIdentityMismatch,
    #[error("Forge trace attempt baseline/seed changed during one search")]
    AttemptSearchIdentityMismatch,
    #[error("Forge trace attempt ordinal is not the next canonical search ordinal")]
    AttemptOrdinalMismatch,
    #[error("candidate terminal event references an unknown or already-closed attempt")]
    UnknownAttemptId,
    #[error("candidate terminal event generation disagrees with its attempt")]
    AttemptGenerationMismatch,
    #[error("candidate terminal event artifact disagrees with CandidateGenerated")]
    AttemptArtifactMismatch,
    #[error("attempt-scoped observation does not bind the same attempt identity")]
    ObservationAttemptMismatch,
    #[error("SearchCompleted appeared before all generated candidate attempts were closed")]
    CompletionWithOpenCandidates,
    #[error("Forge trace contains an event after a terminal search event")]
    EventAfterCompletion,
    #[error("Forge trace is not terminated by SearchCompleted or SearchAborted")]
    MissingCompletion,
    #[error("CandidateArchived is not emitted by Forge's local search trace")]
    UnsupportedArchivedEvent,
    #[error("Forge trace references an observation absent from the supplied store: {0}")]
    MissingObservation(String),
    #[error("Forge observation store failed self-validation: {0}")]
    InvalidObservationStore(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeTraceEvent {
    pub attempt_id: Option<ForgeAttemptId>,
    pub generation: Option<u64>,
    pub kind: DiscoveryEventKind,
    pub candidate_artifact_id: Option<ContentId>,
    pub observation_id: ContentId,
}

impl ForgeTraceEvent {
    pub fn candidate(
        attempt_id: ForgeAttemptId,
        generation: u64,
        kind: DiscoveryEventKind,
        candidate_artifact_id: ContentId,
        observation_id: ContentId,
    ) -> Self {
        Self {
            attempt_id: Some(attempt_id),
            generation: Some(generation),
            kind,
            candidate_artifact_id: Some(candidate_artifact_id),
            observation_id,
        }
    }

    pub fn no_candidate(
        attempt_id: ForgeAttemptId,
        generation: u64,
        observation_id: ContentId,
    ) -> Self {
        Self {
            attempt_id: Some(attempt_id),
            generation: Some(generation),
            kind: DiscoveryEventKind::GeneratorNoOp,
            candidate_artifact_id: None,
            observation_id,
        }
    }

    pub fn completed(observation_id: ContentId) -> Self {
        Self {
            attempt_id: None,
            generation: None,
            kind: DiscoveryEventKind::SearchCompleted,
            candidate_artifact_id: None,
            observation_id,
        }
    }

    pub fn aborted(observation_id: ContentId) -> Self {
        Self {
            attempt_id: None,
            generation: None,
            kind: DiscoveryEventKind::SearchAborted,
            candidate_artifact_id: None,
            observation_id,
        }
    }
}

/// Validate Forge's generator-local candidate lifecycle before semantic replay.
///
/// New attempt occurrences must appear in exact global ordinal order under one pristine baseline
/// and search seed. `GeneratorNoOp` closes its attempt in one event. A concrete candidate opens at
/// `CandidateGenerated` and must close with exactly one candidate terminal event carrying the same
/// attempt, generation, and artifact. Abort may retain open candidates as interrupted/unknown.
pub fn validate_forge_trace(trace: &[ForgeTraceEvent]) -> Result<(), ForgeTraceError> {
    let mut open = BTreeMap::<String, (ForgeAttemptId, ContentId)>::new();
    let mut expected_ordinal = 0u64;
    let mut search_identity: Option<(ContentId, u64)> = None;
    let mut terminal_seen = false;

    for event in trace {
        if terminal_seen {
            return Err(ForgeTraceError::EventAfterCompletion);
        }

        match event.kind {
            DiscoveryEventKind::CandidateGenerated => {
                let (Some(attempt), Some(generation), Some(artifact)) = (
                    event.attempt_id.as_ref(),
                    event.generation,
                    event.candidate_artifact_id.as_ref(),
                ) else {
                    return Err(ForgeTraceError::InvalidEventShape);
                };
                validate_new_attempt(
                    attempt,
                    generation,
                    &mut expected_ordinal,
                    &mut search_identity,
                )?;
                open.insert(
                    attempt.as_content_id().as_str().to_string(),
                    (attempt.clone(), artifact.clone()),
                );
            }
            DiscoveryEventKind::RejectedCompilation
            | DiscoveryEventKind::RejectedCorrectness
            | DiscoveryEventKind::RejectedEvaluation
            | DiscoveryEventKind::ValidNotSelected
            | DiscoveryEventKind::SelectedForContinuation => {
                let (Some(attempt), Some(generation), Some(artifact)) = (
                    event.attempt_id.as_ref(),
                    event.generation,
                    event.candidate_artifact_id.as_ref(),
                ) else {
                    return Err(ForgeTraceError::InvalidEventShape);
                };
                attempt.validate()?;
                if attempt.generation() != generation {
                    return Err(ForgeTraceError::AttemptGenerationMismatch);
                }
                let key = attempt.as_content_id().as_str();
                let Some((generated_attempt, generated_artifact)) = open.get(key) else {
                    return Err(ForgeTraceError::UnknownAttemptId);
                };
                if generated_attempt != attempt {
                    return Err(ForgeTraceError::AttemptIdentityMismatch);
                }
                if generated_artifact != artifact {
                    return Err(ForgeTraceError::AttemptArtifactMismatch);
                }
                open.remove(key);
            }
            DiscoveryEventKind::GeneratorNoOp => {
                let (Some(attempt), Some(generation)) =
                    (event.attempt_id.as_ref(), event.generation)
                else {
                    return Err(ForgeTraceError::InvalidEventShape);
                };
                if event.candidate_artifact_id.is_some() {
                    return Err(ForgeTraceError::InvalidEventShape);
                }
                validate_new_attempt(
                    attempt,
                    generation,
                    &mut expected_ordinal,
                    &mut search_identity,
                )?;
            }
            DiscoveryEventKind::CandidateArchived => {
                return Err(ForgeTraceError::UnsupportedArchivedEvent);
            }
            DiscoveryEventKind::SearchCompleted => {
                validate_search_terminal_shape(event)?;
                if !open.is_empty() {
                    return Err(ForgeTraceError::CompletionWithOpenCandidates);
                }
                terminal_seen = true;
            }
            DiscoveryEventKind::SearchAborted => {
                validate_search_terminal_shape(event)?;
                terminal_seen = true;
            }
        }
    }

    if terminal_seen {
        Ok(())
    } else {
        Err(ForgeTraceError::MissingCompletion)
    }
}

fn validate_new_attempt(
    attempt: &ForgeAttemptId,
    generation: u64,
    expected_ordinal: &mut u64,
    search_identity: &mut Option<(ContentId, u64)>,
) -> Result<(), ForgeTraceError> {
    attempt.validate()?;
    if attempt.generation() != generation {
        return Err(ForgeTraceError::AttemptGenerationMismatch);
    }
    if attempt.ordinal() != *expected_ordinal {
        return Err(ForgeTraceError::AttemptOrdinalMismatch);
    }
    match search_identity {
        Some((baseline, seed))
            if baseline != attempt.baseline_artifact_id() || *seed != attempt.seed() =>
        {
            return Err(ForgeTraceError::AttemptSearchIdentityMismatch);
        }
        None => {
            *search_identity = Some((attempt.baseline_artifact_id().clone(), attempt.seed()));
        }
        Some(_) => {}
    }
    *expected_ordinal = expected_ordinal
        .checked_add(1)
        .ok_or(ForgeTraceError::AttemptOrdinalMismatch)?;
    Ok(())
}

fn validate_search_terminal_shape(event: &ForgeTraceEvent) -> Result<(), ForgeTraceError> {
    if event.attempt_id.is_some()
        || event.generation.is_some()
        || event.candidate_artifact_id.is_some()
    {
        Err(ForgeTraceError::InvalidEventShape)
    } else {
        Ok(())
    }
}

/// Prove that every event reference resolves to an exact canonical object and every attempt-scoped
/// observation independently names the same attempt as its trace event.
pub fn validate_forge_trace_observations(
    trace: &[ForgeTraceEvent],
    observations: &ObservationStore,
) -> Result<(), ForgeTraceError> {
    validate_forge_trace(trace)?;
    observations
        .validate()
        .map_err(|error| ForgeTraceError::InvalidObservationStore(error.to_string()))?;
    for event in trace {
        let object = observations.get(&event.observation_id).ok_or_else(|| {
            ForgeTraceError::MissingObservation(event.observation_id.as_str().to_string())
        })?;
        if let Some(attempt) = event.attempt_id.as_ref() {
            if object.encoding() != ObservationEncoding::Json {
                return Err(ForgeTraceError::ObservationAttemptMismatch);
            }
            let value: Value = serde_json::from_slice(object.payload())
                .map_err(|_| ForgeTraceError::ObservationAttemptMismatch)?;
            if value.get("attempt_id").and_then(Value::as_str)
                != Some(attempt.as_content_id().as_str())
            {
                return Err(ForgeTraceError::ObservationAttemptMismatch);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_algorithms::observation::{ObservationEncoding, ObservationObject};

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn attempt(ordinal: u64) -> ForgeAttemptId {
        ForgeAttemptId::derive(&cid("baseline", "a"), 7, ordinal, 0)
    }

    fn attempt_object(id: &ForgeAttemptId, label: &str) -> ObservationObject {
        ObservationObject::new(
            "forge.test.attempt.v1",
            ObservationEncoding::Json,
            serde_json::to_vec(&serde_json::json!({
                "attempt_id": id.as_content_id().as_str(),
                "label": label
            }))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn attempt_identity_is_self_validating() {
        let id = attempt(0);
        assert!(id.validate().is_ok());
        assert_eq!(id.ordinal(), 0);
        assert_eq!(id.generation(), 0);
    }

    #[test]
    fn repeated_identical_artifacts_are_unambiguous_by_attempt() {
        let artifact = cid("artifact", "same");
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt(0),
                0,
                DiscoveryEventKind::CandidateGenerated,
                artifact.clone(),
                cid("obs", "g1"),
            ),
            ForgeTraceEvent::candidate(
                attempt(1),
                0,
                DiscoveryEventKind::CandidateGenerated,
                artifact.clone(),
                cid("obs", "g2"),
            ),
            ForgeTraceEvent::candidate(
                attempt(1),
                0,
                DiscoveryEventKind::RejectedCorrectness,
                artifact.clone(),
                cid("obs", "r2"),
            ),
            ForgeTraceEvent::candidate(
                attempt(0),
                0,
                DiscoveryEventKind::ValidNotSelected,
                artifact,
                cid("obs", "r1"),
            ),
            ForgeTraceEvent::completed(cid("summary", "done")),
        ];
        assert!(validate_forge_trace(&trace).is_ok());
    }

    #[test]
    fn skipped_attempt_ordinal_is_rejected() {
        let trace = vec![
            ForgeTraceEvent::no_candidate(attempt(0), 0, cid("obs", "zero")),
            ForgeTraceEvent::no_candidate(attempt(2), 0, cid("obs", "two")),
            ForgeTraceEvent::completed(cid("summary", "done")),
        ];
        assert_eq!(
            validate_forge_trace(&trace).unwrap_err(),
            ForgeTraceError::AttemptOrdinalMismatch
        );
    }

    #[test]
    fn terminal_cannot_borrow_another_attempts_artifact() {
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt(0),
                0,
                DiscoveryEventKind::CandidateGenerated,
                cid("artifact", "generated"),
                cid("obs", "g"),
            ),
            ForgeTraceEvent::candidate(
                attempt(0),
                0,
                DiscoveryEventKind::RejectedCorrectness,
                cid("artifact", "substituted"),
                cid("obs", "r"),
            ),
            ForgeTraceEvent::aborted(cid("summary", "abort")),
        ];
        assert_eq!(
            validate_forge_trace(&trace).unwrap_err(),
            ForgeTraceError::AttemptArtifactMismatch
        );
    }

    #[test]
    fn completion_requires_all_concrete_attempts_closed() {
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt(0),
                0,
                DiscoveryEventKind::CandidateGenerated,
                cid("artifact", "candidate"),
                cid("obs", "generated"),
            ),
            ForgeTraceEvent::completed(cid("summary", "done")),
        ];
        assert_eq!(
            validate_forge_trace(&trace).unwrap_err(),
            ForgeTraceError::CompletionWithOpenCandidates
        );
    }

    #[test]
    fn abortion_may_preserve_open_attempt_as_interrupted() {
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt(0),
                0,
                DiscoveryEventKind::CandidateGenerated,
                cid("artifact", "candidate"),
                cid("obs", "generated"),
            ),
            ForgeTraceEvent::aborted(cid("summary", "runner-lost")),
        ];
        assert!(validate_forge_trace(&trace).is_ok());
    }

    #[test]
    fn observation_must_name_same_attempt() {
        let id = attempt(0);
        let wrong = attempt(1);
        let object = attempt_object(&wrong, "generated");
        let terminal = ObservationObject::utf8("forge.test.terminal.v1", "aborted").unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                id,
                0,
                DiscoveryEventKind::CandidateGenerated,
                cid("artifact", "candidate"),
                object.id().clone(),
            ),
            ForgeTraceEvent::aborted(terminal.id().clone()),
        ];
        let store = ObservationStore::from_objects(vec![object, terminal]).unwrap();
        assert_eq!(
            validate_forge_trace_observations(&trace, &store).unwrap_err(),
            ForgeTraceError::ObservationAttemptMismatch
        );
    }

    #[test]
    fn observation_coverage_and_attempt_binding_can_pass() {
        let id = attempt(0);
        let generated = attempt_object(&id, "generated");
        let rejected = attempt_object(&id, "rejected");
        let completed = ObservationObject::utf8("forge.test.terminal.v1", "complete").unwrap();
        let artifact = cid("artifact", "candidate");
        let trace = vec![
            ForgeTraceEvent::candidate(
                id.clone(),
                0,
                DiscoveryEventKind::CandidateGenerated,
                artifact.clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                id,
                0,
                DiscoveryEventKind::RejectedCorrectness,
                artifact,
                rejected.id().clone(),
            ),
            ForgeTraceEvent::completed(completed.id().clone()),
        ];
        let store = ObservationStore::from_objects(vec![generated, rejected, completed]).unwrap();
        assert!(validate_forge_trace_observations(&trace, &store).is_ok());
    }
}
