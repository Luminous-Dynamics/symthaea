// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generator-local trace events that can later be bound to a semantic `DiscoveryRun`.
//!
//! Forge itself does not know the semantic problem contract, so it cannot directly mint a
//! `DiscoveryLedger`. It records only what it actually observed: generation, event class,
//! candidate artifact identity (when one exists), and a content-addressed observation identity.
//!
//! A Forge trace has a stronger local lifecycle than the generic ledger: every generated candidate
//! occurrence must eventually close with exactly one rejection/non-selection/continuation outcome.
//! Repeated generation of identical artifact bytes is allowed and counted as distinct occurrences.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_algorithms::ledger::DiscoveryEventKind;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ForgeTraceError {
    #[error("Forge trace event has an invalid local shape")]
    InvalidEventShape,
    #[error("candidate terminal event has no unmatched CandidateGenerated occurrence")]
    TerminalWithoutGeneration,
    #[error("SearchCompleted appeared before all generated candidate occurrences were closed")]
    CompletionWithOpenCandidates,
    #[error("Forge trace contains an event after SearchCompleted")]
    EventAfterCompletion,
    #[error("Forge trace is not terminated by SearchCompleted")]
    MissingCompletion,
    #[error("CandidateArchived is not emitted by Forge's local search trace")]
    UnsupportedArchivedEvent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeTraceEvent {
    pub generation: Option<u64>,
    pub kind: DiscoveryEventKind,
    pub candidate_artifact_id: Option<ContentId>,
    pub observation_id: ContentId,
}

impl ForgeTraceEvent {
    pub fn candidate(
        generation: u64,
        kind: DiscoveryEventKind,
        candidate_artifact_id: ContentId,
        observation_id: ContentId,
    ) -> Self {
        Self {
            generation: Some(generation),
            kind,
            candidate_artifact_id: Some(candidate_artifact_id),
            observation_id,
        }
    }

    pub fn no_candidate(generation: u64, observation_id: ContentId) -> Self {
        Self {
            generation: Some(generation),
            kind: DiscoveryEventKind::GeneratorNoOp,
            candidate_artifact_id: None,
            observation_id,
        }
    }

    pub fn completed(observation_id: ContentId) -> Self {
        Self {
            generation: None,
            kind: DiscoveryEventKind::SearchCompleted,
            candidate_artifact_id: None,
            observation_id,
        }
    }
}

/// Validate Forge's generator-local candidate lifecycle before semantic replay.
///
/// The key includes generation + artifact identity, while the value is an occurrence count. This
/// means the same exact artifact may be generated multiple times without collapsing observations.
pub fn validate_forge_trace(trace: &[ForgeTraceEvent]) -> Result<(), ForgeTraceError> {
    let mut open: BTreeMap<(u64, ContentId), u64> = BTreeMap::new();
    let mut completed = false;

    for event in trace {
        if completed {
            return Err(ForgeTraceError::EventAfterCompletion);
        }

        match event.kind {
            DiscoveryEventKind::CandidateGenerated => {
                let (Some(generation), Some(artifact)) =
                    (event.generation, event.candidate_artifact_id.as_ref())
                else {
                    return Err(ForgeTraceError::InvalidEventShape);
                };
                *open.entry((generation, artifact.clone())).or_default() += 1;
            }
            DiscoveryEventKind::RejectedCompilation
            | DiscoveryEventKind::RejectedCorrectness
            | DiscoveryEventKind::RejectedEvaluation
            | DiscoveryEventKind::ValidNotSelected
            | DiscoveryEventKind::SelectedForContinuation => {
                let (Some(generation), Some(artifact)) =
                    (event.generation, event.candidate_artifact_id.as_ref())
                else {
                    return Err(ForgeTraceError::InvalidEventShape);
                };
                let key = (generation, artifact.clone());
                let remove_key = {
                    let Some(count) = open.get_mut(&key) else {
                        return Err(ForgeTraceError::TerminalWithoutGeneration);
                    };
                    if *count == 0 {
                        return Err(ForgeTraceError::TerminalWithoutGeneration);
                    }
                    *count -= 1;
                    *count == 0
                };
                if remove_key {
                    open.remove(&key);
                }
            }
            DiscoveryEventKind::GeneratorNoOp => {
                if event.generation.is_none() || event.candidate_artifact_id.is_some() {
                    return Err(ForgeTraceError::InvalidEventShape);
                }
            }
            DiscoveryEventKind::CandidateArchived => {
                return Err(ForgeTraceError::UnsupportedArchivedEvent);
            }
            DiscoveryEventKind::SearchCompleted => {
                if event.generation.is_some() || event.candidate_artifact_id.is_some() {
                    return Err(ForgeTraceError::InvalidEventShape);
                }
                if !open.is_empty() {
                    return Err(ForgeTraceError::CompletionWithOpenCandidates);
                }
                completed = true;
            }
        }
    }

    if completed {
        Ok(())
    } else {
        Err(ForgeTraceError::MissingCompletion)
    }
}

pub fn forge_observation_id<'a>(
    domain: &str,
    parts: impl IntoIterator<Item = &'a [u8]>,
) -> ContentId {
    ContentId::derive(domain, parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    #[test]
    fn event_shape_is_explicit_before_semantic_binding() {
        let artifact = cid("artifact", "candidate");
        let observation = cid("observation", "generated");
        let event = ForgeTraceEvent::candidate(
            3,
            DiscoveryEventKind::CandidateGenerated,
            artifact.clone(),
            observation.clone(),
        );
        assert_eq!(event.generation, Some(3));
        assert_eq!(event.candidate_artifact_id, Some(artifact));
        assert_eq!(event.observation_id, observation);
    }

    #[test]
    fn completion_has_no_candidate_or_generation() {
        let event = ForgeTraceEvent::completed(cid("summary", "complete"));
        assert_eq!(event.kind, DiscoveryEventKind::SearchCompleted);
        assert!(event.generation.is_none());
        assert!(event.candidate_artifact_id.is_none());
    }

    #[test]
    fn repeated_identical_artifact_occurrences_are_counted_separately() {
        let artifact = cid("artifact", "same");
        let trace = vec![
            ForgeTraceEvent::candidate(
                0,
                DiscoveryEventKind::CandidateGenerated,
                artifact.clone(),
                cid("obs", "g1"),
            ),
            ForgeTraceEvent::candidate(
                0,
                DiscoveryEventKind::CandidateGenerated,
                artifact.clone(),
                cid("obs", "g2"),
            ),
            ForgeTraceEvent::candidate(
                0,
                DiscoveryEventKind::RejectedCorrectness,
                artifact.clone(),
                cid("obs", "r1"),
            ),
            ForgeTraceEvent::candidate(
                0,
                DiscoveryEventKind::ValidNotSelected,
                artifact,
                cid("obs", "r2"),
            ),
            ForgeTraceEvent::completed(cid("summary", "done")),
        ];
        assert!(validate_forge_trace(&trace).is_ok());
    }

    #[test]
    fn terminal_without_generated_occurrence_is_rejected() {
        let trace = vec![
            ForgeTraceEvent::candidate(
                0,
                DiscoveryEventKind::RejectedCompilation,
                cid("artifact", "candidate"),
                cid("obs", "reject"),
            ),
            ForgeTraceEvent::completed(cid("summary", "done")),
        ];
        assert_eq!(
            validate_forge_trace(&trace).unwrap_err(),
            ForgeTraceError::TerminalWithoutGeneration
        );
    }

    #[test]
    fn completion_with_open_candidate_is_rejected() {
        let trace = vec![
            ForgeTraceEvent::candidate(
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
    fn event_after_completion_is_rejected() {
        let trace = vec![
            ForgeTraceEvent::completed(cid("summary", "done")),
            ForgeTraceEvent::no_candidate(0, cid("obs", "late")),
        ];
        assert_eq!(
            validate_forge_trace(&trace).unwrap_err(),
            ForgeTraceError::EventAfterCompletion
        );
    }
}
