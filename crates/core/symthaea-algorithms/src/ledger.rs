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
    #[error("event kind has an invalid generation/artifact shape")]
    InvalidEventShape,
    #[error("search ledger is sealed by SearchCompleted")]
    LedgerSealed,
    #[error("SearchCompleted may appear only as the final ledger event")]
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
    /// A concrete candidate artifact was generated and became available for evaluation.
    CandidateGenerated,
    /// The generator could not produce a candidate for this attempt/generation.
    GeneratorNoOp,
    /// Candidate failed to compile/build under the declared gate.
    RejectedCompilation,
    /// Candidate compiled but violated semantic/correctness requirements.
    RejectedCorrectness,
    /// Candidate passed correctness but its configured evaluation failed or was invalid.
    RejectedEvaluation,
    /// Candidate was valid/evaluable but was not selected by the search heuristic/frontier.
    ValidNotSelected,
    /// Candidate was selected as the continuation parent for subsequent search.
    SelectedForContinuation,
    /// Candidate entered an authority-free candidate archive.
    CandidateArchived,
    /// Terminal summary observation. Once appended, the ledger is sealed.
    SearchCompleted,
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
        }
    }

    fn expects_candidate(self) -> Option<bool> {
        match self {
            Self::GeneratorNoOp | Self::SearchCompleted => Some(false),
            Self::CandidateGenerated
            | Self::RejectedCompilation
            | Self::RejectedCorrectness
            | Self::RejectedEvaluation
            | Self::ValidNotSelected
            | Self::SelectedForContinuation
            | Self::CandidateArchived => Some(true),
        }
    }
}

/// One immutable observation in a hash-chained discovery history.
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
    match kind {
        DiscoveryEventKind::SearchCompleted => {
            if generation.is_some() || candidate_artifact_id.is_some() {
                return Err(LedgerError::InvalidEventShape);
            }
        }
        _ => {
            let Some(generation) = generation else {
                return Err(LedgerError::InvalidEventShape);
            };
            if generation >= run.budget.max_generations {
                return Err(LedgerError::GenerationBudgetExceeded {
                    generation,
                    maximum: run.budget.max_generations,
                });
            }
            if kind.expects_candidate() == Some(candidate_artifact_id.is_some()) {
                // expected shape
            } else {
                return Err(LedgerError::InvalidEventShape);
            }
        }
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
            candidate_artifact_id.map(ContentId::as_str).unwrap_or("").as_bytes(),
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

/// Append-only, self-validating observation history for exactly one [`DiscoveryRun`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryLedger {
    run_id: ContentId,
    events: Vec<DiscoveryEvent>,
}

impl DiscoveryLedger {
    pub fn new(run: &DiscoveryRun) -> Result<Self, LedgerError> {
        run.validate()?;
        Ok(Self {
            run_id: run.id.clone(),
            events: Vec::new(),
        })
    }

    /// Rehydrate a persisted ledger only if the full hash chain and event shapes validate.
    pub fn from_events(
        run: &DiscoveryRun,
        events: Vec<DiscoveryEvent>,
    ) -> Result<Self, LedgerError> {
        let ledger = Self {
            run_id: run.id.clone(),
            events,
        };
        ledger.validate_for(run)?;
        Ok(ledger)
    }

    pub fn append(
        &mut self,
        run: &DiscoveryRun,
        generation: Option<u64>,
        kind: DiscoveryEventKind,
        candidate_artifact_id: Option<ContentId>,
        observation_id: ContentId,
    ) -> Result<&DiscoveryEvent, LedgerError> {
        self.validate_for(run)?;
        if self.is_sealed() {
            return Err(LedgerError::LedgerSealed);
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

    pub fn validate_for(&self, run: &DiscoveryRun) -> Result<(), LedgerError> {
        run.validate()?;
        if self.run_id != run.id {
            return Err(LedgerError::RunMismatch);
        }
        let mut previous = genesis_id(run);
        for (index, event) in self.events.iter().enumerate() {
            event.validate_for(run, index as u64, &previous)?;
            if event.kind == DiscoveryEventKind::SearchCompleted && index + 1 != self.events.len() {
                return Err(LedgerError::CompletionNotFinal);
            }
            previous = event.id.clone();
        }
        Ok(())
    }

    pub fn run_id(&self) -> &ContentId {
        &self.run_id
    }

    pub fn events(&self) -> impl Iterator<Item = &DiscoveryEvent> {
        self.events.iter()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn is_sealed(&self) -> bool {
        self.events
            .last()
            .is_some_and(|event| event.kind == DiscoveryEventKind::SearchCompleted)
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
    use crate::{
        DeterminismRequirement, DiscoveryRisk, ProblemSpec, SemanticGuarantee,
    };

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn run() -> DiscoveryRun {
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
            SearchBudget::new(10, 2, 10).unwrap(),
            7,
        )
        .unwrap()
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
        assert_eq!(ledger.len(), 3);
        assert!(ledger.validate_for(&run).is_ok());
        assert!(ledger.events().any(|event| {
            event.kind() == DiscoveryEventKind::RejectedCorrectness
        }));
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
