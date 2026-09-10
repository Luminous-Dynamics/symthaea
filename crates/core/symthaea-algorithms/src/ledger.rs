// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only memory for algorithm-discovery search outcomes.
//!
//! A discovery ledger records what a bounded search observed, including negative results. It does
//! not execute candidates, rank them, promote them, or grant runtime authority. Generator-specific
//! details live outside this core contract and are referenced by content identity.

use crate::discovery::{DiscoveryError, DiscoveryRun};
use crate::ContentId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LedgerError {
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error("ledger event sequence does not match append order")]
    SequenceMismatch,
    #[error("ledger event previous identity does not match the prior event")]
    PreviousEventMismatch,
    #[error("ledger event identity does not match its canonical fields")]
    EventIdentityMismatch,
    #[error("ledger event run does not match this discovery run")]
    RunMismatch,
    #[error("event generation {generation} exceeds run generation budget {maximum}")]
    GenerationBudgetExceeded { generation: u64, maximum: u64 },
    #[error("candidate-generation budget exhausted: maximum {maximum}")]
    CandidateBudgetExceeded { maximum: u64 },
    #[error("cached candidate-generation count does not match the validated event history")]
    CandidateCountMismatch,
    #[error("event kind has an invalid generation/artifact shape")]
    InvalidEventShape,
    #[error("search ledger is sealed by a terminal event")]
    LedgerSealed,
    #[error("a terminal search event may appear only as the final ledger event")]
    CompletionNotFinal,
}

/// Stable semantic class of one discovery observation.
///
/// Detailed compiler output, counterexamples, benchmark errors, mutation descriptions, and similar
/// generator-specific payloads are deliberately not embedded here. `observation_id` binds those
/// details without coupling core contracts to a particular search engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiscoveryEventKind {
    CandidateGenerated,
    GeneratorNoOp,
    RejectedCompilation,
    RejectedCorrectness,
    RejectedEvaluation,
    ValidNotSelected,
    SelectedForContinuation,
    CandidateArchived,
    /// The search apparatus reached its intended bounded end.
    SearchCompleted,
    /// The search apparatus stopped early. This says nothing about candidate quality.
    SearchAborted,
}

impl DiscoveryEventKind {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::CandidateGenerated => b"candidate-generated",
            Self::GeneratorNoOp => b"generator-no-op",
            Self::RejectedCompilation => b"rejected-compilation",
            Self::RejectedCorrectness => b"rejected-correctness",
            Self::RejectedEvaluation => b"rejected-evaluation",
            Self::ValidNotSelected => b"valid-not-selected",
            Self::SelectedForContinuation => b"selected-for-continuation",
            Self::CandidateArchived => b"candidate-archived",
            Self::SearchCompleted => b"search-completed",
            Self::SearchAborted => b"search-aborted",
        }
    }

    fn expects_candidate(self) -> bool {
        !matches!(
            self,
            Self::GeneratorNoOp | Self::SearchCompleted | Self::SearchAborted
        )
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::SearchCompleted | Self::SearchAborted)
    }
}

/// One immutable observation in a hash-chained discovery history.
///
/// Fields are private so callers cannot mutate a live ledger entry. Persisted events may be
/// deserialized, but must re-enter a ledger through [`DiscoveryLedger::from_events`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryEvent {
    id: ContentId,
    run_id: ContentId,
    sequence: u64,
    previous_event_id: ContentId,
    generation: Option<u64>,
    kind: DiscoveryEventKind,
    candidate_artifact_id: Option<ContentId>,
    observation_id: ContentId,
}

impl DiscoveryEvent {
    #[allow(clippy::too_many_arguments)]
    fn new(
        run: &DiscoveryRun,
        sequence: u64,
        previous_event_id: ContentId,
        generation: Option<u64>,
        kind: DiscoveryEventKind,
        candidate_artifact_id: Option<ContentId>,
        observation_id: ContentId,
    ) -> Result<Self, LedgerError> {
        validate_event_shape(run, generation, kind, candidate_artifact_id.as_ref())?;
        let id = derive_event_id(
            &run.id,
            sequence,
            &previous_event_id,
            generation,
            kind,
            candidate_artifact_id.as_ref(),
            &observation_id,
        );
        Ok(Self {
            id,
            run_id: run.id.clone(),
            sequence,
            previous_event_id,
            generation,
            kind,
            candidate_artifact_id,
            observation_id,
        })
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn run_id(&self) -> &ContentId {
        &self.run_id
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn previous_event_id(&self) -> &ContentId {
        &self.previous_event_id
    }

    pub fn generation(&self) -> Option<u64> {
        self.generation
    }

    pub fn kind(&self) -> DiscoveryEventKind {
        self.kind
    }

    pub fn candidate_artifact_id(&self) -> Option<&ContentId> {
        self.candidate_artifact_id.as_ref()
    }

    pub fn observation_id(&self) -> &ContentId {
        &self.observation_id
    }

    fn validate_for(
        &self,
        run: &DiscoveryRun,
        expected_sequence: u64,
        expected_previous: &ContentId,
    ) -> Result<(), LedgerError> {
        if self.run_id != run.id {
            return Err(LedgerError::RunMismatch);
        }
        if self.sequence != expected_sequence {
            return Err(LedgerError::SequenceMismatch);
        }
        if &self.previous_event_id != expected_previous {
            return Err(LedgerError::PreviousEventMismatch);
        }
        validate_event_shape(
            run,
            self.generation,
            self.kind,
            self.candidate_artifact_id.as_ref(),
        )?;
        let expected = derive_event_id(
            &self.run_id,
            self.sequence,
            &self.previous_event_id,
            self.generation,
            self.kind,
            self.candidate_artifact_id.as_ref(),
            &self.observation_id,
        );
        if expected != self.id {
            return Err(LedgerError::EventIdentityMismatch);
        }
        Ok(())
    }
}

fn validate_event_shape(
    run: &DiscoveryRun,
    generation: Option<u64>,
    kind: DiscoveryEventKind,
    candidate_artifact_id: Option<&ContentId>,
) -> Result<(), LedgerError> {
    if kind.is_terminal() {
        if generation.is_some() || candidate_artifact_id.is_some() {
            return Err(LedgerError::InvalidEventShape);
        }
        return Ok(());
    }

    let Some(generation) = generation else {
        return Err(LedgerError::InvalidEventShape);
    };
    if generation >= run.budget.max_generations {
        return Err(LedgerError::GenerationBudgetExceeded {
            generation,
            maximum: run.budget.max_generations,
        });
    }
    if kind.expects_candidate() != candidate_artifact_id.is_some() {
        return Err(LedgerError::InvalidEventShape);
    }
    Ok(())
}

fn derive_event_id(
    run_id: &ContentId,
    sequence: u64,
    previous_event_id: &ContentId,
    generation: Option<u64>,
    kind: DiscoveryEventKind,
    candidate_artifact_id: Option<&ContentId>,
    observation_id: &ContentId,
) -> ContentId {
    let sequence_bytes = sequence.to_be_bytes();
    let generation_bytes = generation.unwrap_or(u64::MAX).to_be_bytes();
    ContentId::derive(
        "symthaea.discovery-ledger-event.v1",
        [
            run_id.as_str().as_bytes(),
            sequence_bytes.as_slice(),
            previous_event_id.as_str().as_bytes(),
            generation_bytes.as_slice(),
            kind.tag(),
            candidate_artifact_id
                .map(ContentId::as_str)
                .unwrap_or("")
                .as_bytes(),
            observation_id.as_str().as_bytes(),
        ],
    )
}

fn genesis_id(run: &DiscoveryRun) -> ContentId {
    ContentId::derive(
        "symthaea.discovery-ledger-genesis.v1",
        [run.id.as_str().as_bytes()],
    )
}

fn validate_events(
    run: &DiscoveryRun,
    run_id: &ContentId,
    events: &[DiscoveryEvent],
) -> Result<u64, LedgerError> {
    run.validate()?;
    if run_id != &run.id {
        return Err(LedgerError::RunMismatch);
    }
    let mut previous = genesis_id(run);
    let mut candidate_generated_count = 0u64;
    for (index, event) in events.iter().enumerate() {
        event.validate_for(run, index as u64, &previous)?;
        if event.kind == DiscoveryEventKind::CandidateGenerated {
            candidate_generated_count = candidate_generated_count
                .checked_add(1)
                .ok_or(LedgerError::CandidateBudgetExceeded {
                    maximum: run.budget.max_candidates,
                })?;
            if candidate_generated_count > run.budget.max_candidates {
                return Err(LedgerError::CandidateBudgetExceeded {
                    maximum: run.budget.max_candidates,
                });
            }
        }
        if event.kind.is_terminal() && index + 1 != events.len() {
            return Err(LedgerError::CompletionNotFinal);
        }
        previous = event.id.clone();
    }
    Ok(candidate_generated_count)
}

/// Append-only, self-validating observation history for exactly one [`DiscoveryRun`].
///
/// The ledger itself is intentionally serialization-only. Persisted event vectors must be
/// rehydrated through [`Self::from_events`], which validates the complete chain once. This keeps
/// normal append O(1) rather than revalidating an ever-growing history on every event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveryLedger {
    run_id: ContentId,
    events: Vec<DiscoveryEvent>,
    #[serde(skip)]
    candidate_generated_count: u64,
}

impl DiscoveryLedger {
    pub fn new(run: &DiscoveryRun) -> Result<Self, LedgerError> {
        run.validate()?;
        Ok(Self {
            run_id: run.id.clone(),
            events: Vec::new(),
            candidate_generated_count: 0,
        })
    }

    /// Rehydrate persisted events only after validating the full hash chain, shapes, terminal
    /// placement and run-level candidate budget in one linear pass.
    pub fn from_events(
        run: &DiscoveryRun,
        events: Vec<DiscoveryEvent>,
    ) -> Result<Self, LedgerError> {
        let count = validate_events(run, &run.id, &events)?;
        Ok(Self {
            run_id: run.id.clone(),
            events,
            candidate_generated_count: count,
        })
    }

    pub fn append(
        &mut self,
        run: &DiscoveryRun,
        generation: Option<u64>,
        kind: DiscoveryEventKind,
        candidate_artifact_id: Option<ContentId>,
        observation_id: ContentId,
    ) -> Result<&DiscoveryEvent, LedgerError> {
        run.validate()?;
        if self.run_id != run.id {
            return Err(LedgerError::RunMismatch);
        }
        if self.is_sealed() {
            return Err(LedgerError::LedgerSealed);
        }
        if kind == DiscoveryEventKind::CandidateGenerated
            && self.candidate_generated_count >= run.budget.max_candidates
        {
            return Err(LedgerError::CandidateBudgetExceeded {
                maximum: run.budget.max_candidates,
            });
        }

        let sequence = self.events.len() as u64;
        let previous = self
            .events
            .last()
            .map(|event| event.id.clone())
            .unwrap_or_else(|| genesis_id(run));
        let event = DiscoveryEvent::new(
            run,
            sequence,
            previous,
            generation,
            kind,
            candidate_artifact_id,
            observation_id,
        )?;
        self.events.push(event);
        if kind == DiscoveryEventKind::CandidateGenerated {
            self.candidate_generated_count += 1;
        }
        Ok(self.events.last().expect("event was just appended"))
    }

    pub fn complete(
        &mut self,
        run: &DiscoveryRun,
        summary_observation_id: ContentId,
    ) -> Result<&DiscoveryEvent, LedgerError> {
        self.append(
            run,
            None,
            DiscoveryEventKind::SearchCompleted,
            None,
            summary_observation_id,
        )
    }

    pub fn abort(
        &mut self,
        run: &DiscoveryRun,
        abort_observation_id: ContentId,
    ) -> Result<&DiscoveryEvent, LedgerError> {
        self.append(
            run,
            None,
            DiscoveryEventKind::SearchAborted,
            None,
            abort_observation_id,
        )
    }

    pub fn validate_for(&self, run: &DiscoveryRun) -> Result<(), LedgerError> {
        let count = validate_events(run, &self.run_id, &self.events)?;
        if count != self.candidate_generated_count {
            return Err(LedgerError::CandidateCountMismatch);
        }
        Ok(())
    }

    pub fn run_id(&self) -> &ContentId {
        &self.run_id
    }

    pub fn events(&self) -> impl Iterator<Item = &DiscoveryEvent> {
        self.events.iter()
    }

    /// Clone the event stream for persistence. Rehydration must use [`Self::from_events`].
    pub fn to_events(&self) -> Vec<DiscoveryEvent> {
        self.events.clone()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn is_sealed(&self) -> bool {
        self.terminal_kind().is_some()
    }

    pub fn terminal_kind(&self) -> Option<DiscoveryEventKind> {
        self.events
            .last()
            .map(DiscoveryEvent::kind)
            .filter(|kind| kind.is_terminal())
    }

    pub fn candidate_generated_count(&self) -> u64 {
        self.candidate_generated_count
    }

    pub fn head_id(&self, run: &DiscoveryRun) -> Result<ContentId, LedgerError> {
        self.validate_for(run)?;
        Ok(self
            .events
            .last()
            .map(|event| event.id.clone())
            .unwrap_or_else(|| genesis_id(run)))
    }

    /// Content identity of the complete current ledger snapshot.
    pub fn snapshot_id(&self, run: &DiscoveryRun) -> Result<ContentId, LedgerError> {
        let head = self.head_id(run)?;
        let count = (self.events.len() as u64).to_be_bytes();
        Ok(ContentId::derive(
            "symthaea.discovery-ledger-snapshot.v1",
            [
                run.id.as_str().as_bytes(),
                head.as_str().as_bytes(),
                count.as_slice(),
            ],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{DiscoveryPolicy, SearchBudget};
    use crate::{DeterminismRequirement, DiscoveryRisk, ProblemSpec, SemanticGuarantee};

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn run_with_candidates(max_candidates: u64) -> DiscoveryRun {
        let problem = ProblemSpec::new(
            "ledger-test",
            "Return the exact reference value.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap();
        DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            cid("generator", "test"),
            "abc123",
            SearchBudget::new(max_candidates, 2, 10).unwrap(),
            7,
        )
        .unwrap()
    }

    fn run() -> DiscoveryRun {
        run_with_candidates(10)
    }

    #[test]
    fn negative_results_are_preserved_in_hash_chain() {
        let run = run();
        let artifact = cid("artifact", "candidate-a");
        let mut ledger = DiscoveryLedger::new(&run).unwrap();
        ledger
            .append(
                &run,
                Some(0),
                DiscoveryEventKind::CandidateGenerated,
                Some(artifact.clone()),
                cid("observation", "generated"),
            )
            .unwrap();
        ledger
            .append(
                &run,
                Some(0),
                DiscoveryEventKind::RejectedCorrectness,
                Some(artifact),
                cid("observation", "counterexample-17"),
            )
            .unwrap();
        ledger.complete(&run, cid("summary", "no-winner")).unwrap();

        assert!(ledger.is_sealed());
        assert_eq!(ledger.terminal_kind(), Some(DiscoveryEventKind::SearchCompleted));
        assert_eq!(ledger.len(), 3);
        assert_eq!(ledger.candidate_generated_count(), 1);
        assert!(ledger.validate_for(&run).is_ok());
        assert!(ledger
            .events()
            .any(|event| event.kind() == DiscoveryEventKind::RejectedCorrectness));
    }

    #[test]
    fn identical_runs_and_observations_have_identical_snapshot_identity() {
        let run = run();
        let build = || {
            let mut ledger = DiscoveryLedger::new(&run).unwrap();
            ledger
                .append(
                    &run,
                    Some(0),
                    DiscoveryEventKind::GeneratorNoOp,
                    None,
                    cid("observation", "no-sites"),
                )
                .unwrap();
            ledger.complete(&run, cid("summary", "complete")).unwrap();
            ledger
        };
        let a = build();
        let b = build();
        assert_eq!(a.snapshot_id(&run).unwrap(), b.snapshot_id(&run).unwrap());
    }

    #[test]
    fn event_order_changes_snapshot_identity() {
        let run = run();
        let artifact = cid("artifact", "candidate");
        let mut a = DiscoveryLedger::new(&run).unwrap();
        a.append(
            &run,
            Some(0),
            DiscoveryEventKind::CandidateGenerated,
            Some(artifact.clone()),
            cid("observation", "generated"),
        )
        .unwrap();
        a.append(
            &run,
            Some(0),
            DiscoveryEventKind::ValidNotSelected,
            Some(artifact.clone()),
            cid("observation", "not-selected"),
        )
        .unwrap();

        let mut b = DiscoveryLedger::new(&run).unwrap();
        b.append(
            &run,
            Some(0),
            DiscoveryEventKind::ValidNotSelected,
            Some(artifact.clone()),
            cid("observation", "not-selected"),
        )
        .unwrap();
        b.append(
            &run,
            Some(0),
            DiscoveryEventKind::CandidateGenerated,
            Some(artifact),
            cid("observation", "generated"),
        )
        .unwrap();

        assert_ne!(a.snapshot_id(&run).unwrap(), b.snapshot_id(&run).unwrap());
    }

    #[test]
    fn completion_seals_ledger() {
        let run = run();
        let mut ledger = DiscoveryLedger::new(&run).unwrap();
        ledger.complete(&run, cid("summary", "complete")).unwrap();
        assert_eq!(
            ledger
                .append(
                    &run,
                    Some(0),
                    DiscoveryEventKind::GeneratorNoOp,
                    None,
                    cid("observation", "late"),
                )
                .unwrap_err(),
            LedgerError::LedgerSealed
        );
    }

    #[test]
    fn abort_seals_ledger_without_claiming_completion() {
        let run = run();
        let mut ledger = DiscoveryLedger::new(&run).unwrap();
        ledger.abort(&run, cid("abort", "runner-lost")).unwrap();
        assert!(ledger.is_sealed());
        assert_eq!(ledger.terminal_kind(), Some(DiscoveryEventKind::SearchAborted));
        assert!(ledger.validate_for(&run).is_ok());
        assert_eq!(
            ledger
                .complete(&run, cid("summary", "late-completion"))
                .unwrap_err(),
            LedgerError::LedgerSealed
        );
    }

    #[test]
    fn completed_and_aborted_runs_have_distinct_snapshot_identity() {
        let run = run();
        let observation = cid("terminal", "same-payload-id");
        let mut completed = DiscoveryLedger::new(&run).unwrap();
        completed.complete(&run, observation.clone()).unwrap();
        let mut aborted = DiscoveryLedger::new(&run).unwrap();
        aborted.abort(&run, observation).unwrap();
        assert_ne!(
            completed.snapshot_id(&run).unwrap(),
            aborted.snapshot_id(&run).unwrap()
        );
    }

    #[test]
    fn candidate_event_requires_artifact_and_noop_forbids_it() {
        let run = run();
        let mut ledger = DiscoveryLedger::new(&run).unwrap();
        assert_eq!(
            ledger
                .append(
                    &run,
                    Some(0),
                    DiscoveryEventKind::RejectedCompilation,
                    None,
                    cid("observation", "compile"),
                )
                .unwrap_err(),
            LedgerError::InvalidEventShape
        );
        assert_eq!(
            ledger
                .append(
                    &run,
                    Some(0),
                    DiscoveryEventKind::GeneratorNoOp,
                    Some(cid("artifact", "impossible")),
                    cid("observation", "noop"),
                )
                .unwrap_err(),
            LedgerError::InvalidEventShape
        );
        assert_eq!(
            ledger
                .append(
                    &run,
                    Some(0),
                    DiscoveryEventKind::SearchAborted,
                    None,
                    cid("observation", "bad-terminal-shape"),
                )
                .unwrap_err(),
            LedgerError::InvalidEventShape
        );
    }

    #[test]
    fn generation_budget_is_enforced() {
        let run = run();
        let mut ledger = DiscoveryLedger::new(&run).unwrap();
        assert_eq!(
            ledger
                .append(
                    &run,
                    Some(2),
                    DiscoveryEventKind::GeneratorNoOp,
                    None,
                    cid("observation", "out-of-range"),
                )
                .unwrap_err(),
            LedgerError::GenerationBudgetExceeded {
                generation: 2,
                maximum: 2,
            }
        );
    }

    #[test]
    fn candidate_budget_is_enforced_during_append() {
        let run = run_with_candidates(1);
        let mut ledger = DiscoveryLedger::new(&run).unwrap();
        ledger
            .append(
                &run,
                Some(0),
                DiscoveryEventKind::CandidateGenerated,
                Some(cid("artifact", "one")),
                cid("observation", "one"),
            )
            .unwrap();
        assert_eq!(
            ledger
                .append(
                    &run,
                    Some(0),
                    DiscoveryEventKind::CandidateGenerated,
                    Some(cid("artifact", "two")),
                    cid("observation", "two"),
                )
                .unwrap_err(),
            LedgerError::CandidateBudgetExceeded { maximum: 1 }
        );
    }

    #[test]
    fn from_events_recomputes_candidate_count() {
        let run = run_with_candidates(2);
        let mut ledger = DiscoveryLedger::new(&run).unwrap();
        for name in ["one", "two"] {
            ledger
                .append(
                    &run,
                    Some(0),
                    DiscoveryEventKind::CandidateGenerated,
                    Some(cid("artifact", name)),
                    cid("observation", name),
                )
                .unwrap();
        }
        let rehydrated = DiscoveryLedger::from_events(&run, ledger.to_events()).unwrap();
        assert_eq!(rehydrated.candidate_generated_count(), 2);
    }

    #[test]
    fn tampering_breaks_validation() {
        let run = run();
        let mut ledger = DiscoveryLedger::new(&run).unwrap();
        ledger
            .append(
                &run,
                Some(0),
                DiscoveryEventKind::GeneratorNoOp,
                None,
                cid("observation", "no-sites"),
            )
            .unwrap();
        ledger.events[0].observation_id = cid("observation", "tampered");
        assert_eq!(
            ledger.validate_for(&run).unwrap_err(),
            LedgerError::EventIdentityMismatch
        );
    }
}
