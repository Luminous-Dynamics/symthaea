// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generator-local trace events that can later be bound to a semantic `DiscoveryRun`.
//!
//! Forge itself does not know the semantic problem contract, so it cannot directly mint a
//! `DiscoveryLedger`. Every local search-loop attempt has a deterministic content-addressed
//! [`ForgeAttemptId`]. Candidate generation and its terminal outcome must carry the same attempt ID,
//! removing ambiguity when identical candidate bytes are generated more than once.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use symthaea_algorithms::ledger::DiscoveryEventKind;
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeAttemptId(ContentId);

impl ForgeAttemptId {
    /// Deterministic identity of one Forge inner-loop attempt.
    ///
    /// `ordinal` is global within the search, while `generation` is retained independently so an
    /// impossible scheduler/replay mismatch cannot silently reuse the same ordinal in another
    /// generation. The pristine baseline + seed bind the attempt to one reproducible search plan.
    pub fn derive(
        baseline_artifact_id: &ContentId,
        seed: u64,
        ordinal: u64,
        generation: u64,
    ) -> Self {
        let seed_bytes = seed.to_be_bytes();
        let ordinal_bytes = ordinal.to_be_bytes();
        let generation_bytes = generation.to_be_bytes();
        Self(ContentId::derive(
            "symthaea.forge-attempt.v1",
            [
                baseline_artifact_id.as_str().as_bytes(),
                seed_bytes.as_slice(),
                ordinal_bytes.as_slice(),
                generation_bytes.as_slice(),
            ],
        ))
    }

    pub fn as_content_id(&self) -> &ContentId {
        &self.0
    }
}

impl fmt::Display for ForgeAttemptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ForgeTraceError {
    #[error("Forge trace event has an invalid local shape")]
    InvalidEventShape,
    #[error("Forge trace reuses one attempt identity")]
    DuplicateAttemptId,
    #[error("candidate terminal event references an unknown or already-closed attempt")]
    UnknownAttemptId,
    #[error("candidate terminal event generation disagrees with CandidateGenerated")]
    AttemptGenerationMismatch,
    #[error("candidate terminal event artifact disagrees with CandidateGenerated")]
    AttemptArtifactMismatch,
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
    /// Generator-local occurrence identity. Present for every inner-loop attempt event and absent
    /// only for search-level terminal events.
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
/// Each inner-loop attempt identity is globally unique. `GeneratorNoOp` closes its attempt in one
/// event. A concrete candidate opens at `CandidateGenerated` and must close with exactly one
/// candidate terminal event carrying the same attempt ID, generation, and artifact. Search abort
/// may retain open attempts as interrupted/unknown; search completion may not.
pub fn validate_forge_trace(trace: &[ForgeTraceEvent]) -> Result<(), ForgeTraceError> {
    let mut seen_attempts = BTreeSet::<String>::new();
    let mut open = BTreeMap::<String, (u64, ContentId)>::new();
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
                let key = attempt.as_content_id().as_str().to_string();
                if !seen_attempts.insert(key.clone()) {
                    return Err(ForgeTraceError::DuplicateAttemptId);
                }
                open.insert(key, (generation, artifact.clone()));
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
                let key = attempt.as_content_id().as_str();
                let Some((generated_generation, generated_artifact)) = open.get(key) else {
                    return Err(ForgeTraceError::UnknownAttemptId);
                };
                if *generated_generation != generation {
                    return Err(ForgeTraceError::AttemptGenerationMismatch);
                }
                if generated_artifact != artifact {
                    return Err(ForgeTraceError::AttemptArtifactMismatch);
                }
                open.remove(key);
            }
            DiscoveryEventKind::GeneratorNoOp => {
                let (Some(attempt), Some(_generation)) =
                    (event.attempt_id.as_ref(), event.generation)
                else {
                    return Err(ForgeTraceError::InvalidEventShape);
                };
                if event.candidate_artifact_id.is_some() {
                    return Err(ForgeTraceError::InvalidEventShape);
                }
                let key = attempt.as_content_id().as_str().to_string();
                if !seen_attempts.insert(key) {
                    return Err(ForgeTraceError::DuplicateAttemptId);
                }
            }
            DiscoveryEventKind::CandidateArchived => {
                return Err(ForgeTraceError::UnsupportedArchivedEvent);
            }
            DiscoveryEventKind::SearchCompleted => {
                if event.attempt_id.is_some()
                    || event.generation.is_some()
                    || event.candidate_artifact_id.is_some()
                {
                    return Err(ForgeTraceError::InvalidEventShape);
                }
                if !open.is_empty() {
                    return Err(ForgeTraceError::CompletionWithOpenCandidates);
                }
                terminal_seen = true;
            }
            DiscoveryEventKind::SearchAborted => {
                if event.attempt_id.is_some()
                    || event.generation.is_some()
                    || event.candidate_artifact_id.is_some()
                {
                    return Err(ForgeTraceError::InvalidEventShape);
                }
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

pub fn validate_forge_trace_observations(
    trace: &[ForgeTraceEvent],
    observations: &ObservationStore,
) -> Result<(), ForgeTraceError> {
    validate_forge_trace(trace)?;
    observations
        .validate()
        .map_err(|error| ForgeTraceError::InvalidObservationStore(error.to_string()))?;
    for event in trace {
        if !observations.contains(&event.observation_id) {
            return Err(ForgeTraceError::MissingObservation(
                event.observation_id.as_str().to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_algorithms::observation::ObservationObject;

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn attempt(ordinal: u64) -> ForgeAttemptId {
        ForgeAttemptId::derive(&cid("baseline", "a"), 7, ordinal, 0)
    }

    #[test]
    fn attempt_identity_changes_with_ordinal_or_generation() {
        let baseline = cid("baseline", "a");
        assert_ne!(
            ForgeAttemptId::derive(&baseline, 7, 0, 0),
            ForgeAttemptId::derive(&baseline, 7, 1, 0)
        );
        assert_ne!(
            ForgeAttemptId::derive(&baseline, 7, 0, 0),
            ForgeAttemptId::derive(&baseline, 7, 0, 1)
        );
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
    fn terminal_cannot_borrow_another_attempts_artifact() {
        let generated = cid("artifact", "generated");
        let substituted = cid("artifact", "substituted");
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt(0),
                0,
                DiscoveryEventKind::CandidateGenerated,
                generated,
                cid("obs", "g"),
            ),
            ForgeTraceEvent::candidate(
                attempt(0),
                0,
                DiscoveryEventKind::RejectedCorrectness,
                substituted,
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
    fn no_op_attempt_identity_cannot_be_reused() {
        let id = attempt(0);
        let trace = vec![
            ForgeTraceEvent::no_candidate(id.clone(), 0, cid("obs", "noop")),
            ForgeTraceEvent::no_candidate(id, 0, cid("obs", "noop-again")),
            ForgeTraceEvent::completed(cid("summary", "done")),
        ];
        assert_eq!(
            validate_forge_trace(&trace).unwrap_err(),
            ForgeTraceError::DuplicateAttemptId
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
    fn observation_coverage_is_required() {
        let object = ObservationObject::utf8("forge.test.v1", "complete").unwrap();
        let trace = vec![ForgeTraceEvent::completed(object.id().clone())];
        let full = ObservationStore::from_objects(vec![object]).unwrap();
        assert!(validate_forge_trace_observations(&trace, &full).is_ok());

        let empty = ObservationStore::new();
        assert!(matches!(
            validate_forge_trace_observations(&trace, &empty),
            Err(ForgeTraceError::MissingObservation(_))
        ));
    }
}
