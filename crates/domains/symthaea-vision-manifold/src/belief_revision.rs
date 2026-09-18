// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Append-only belief revision ledger for VIS-004C.
//!
//! Each revision carries the full previous and next belief distributions. The ledger validates
//! exact continuity so a history cannot silently skip from one belief state to another.

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

use crate::competing_beliefs::{SemanticClassBeliefSet, TrackIdentityBeliefSet};
use crate::epistemic::{VisualEvidence, VisualOrigin};

const MAX_LEDGER_CAPACITY: usize = 4096;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "state", rename_all = "snake_case")]
pub enum VisualBeliefSnapshot {
    SemanticClass(SemanticClassBeliefSet),
    TrackIdentity(TrackIdentityBeliefSet),
}

impl VisualBeliefSnapshot {
    pub fn same_question(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::SemanticClass(a), Self::SemanticClass(b)) => {
                a.subject() == b.subject() && a.vocabulary() == b.vocabulary()
            }
            (Self::TrackIdentity(a), Self::TrackIdentity(b)) => a.subject() == b.subject(),
            _ => false,
        }
    }

    pub fn is_complete_abstention(&self) -> bool {
        match self {
            Self::SemanticClass(set) => {
                set.candidates().is_empty() && set.unassigned_mass().get() == 1.0
            }
            Self::TrackIdentity(set) => {
                set.candidates().is_empty() && set.unassigned_mass().get() == 1.0
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefRevisionOperation {
    Initialized,
    EvidenceAssimilated,
    EvidenceRetracted,
    Reweighted,
    ResetToUnknown,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BeliefRevision {
    revision: u64,
    operation: BeliefRevisionOperation,
    before: Option<VisualBeliefSnapshot>,
    after: VisualBeliefSnapshot,
    cause: VisualEvidence,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct BeliefRevisionWire {
    revision: u64,
    operation: BeliefRevisionOperation,
    before: Option<VisualBeliefSnapshot>,
    after: VisualBeliefSnapshot,
    cause: VisualEvidence,
}

impl BeliefRevision {
    fn new(
        revision: u64,
        operation: BeliefRevisionOperation,
        before: Option<VisualBeliefSnapshot>,
        after: VisualBeliefSnapshot,
        cause: VisualEvidence,
    ) -> Result<Self, BeliefRevisionError> {
        let record = Self {
            revision,
            operation,
            before,
            after,
            cause,
        };
        record.validate_local()?;
        Ok(record)
    }

    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub const fn operation(&self) -> BeliefRevisionOperation {
        self.operation
    }

    pub fn before(&self) -> Option<&VisualBeliefSnapshot> {
        self.before.as_ref()
    }

    pub fn after(&self) -> &VisualBeliefSnapshot {
        &self.after
    }

    pub fn cause(&self) -> &VisualEvidence {
        &self.cause
    }

    fn validate_local(&self) -> Result<(), BeliefRevisionError> {
        if self.revision == 0 {
            return Err(BeliefRevisionError::ZeroRevision);
        }
        validate_historical_cause(&self.cause)?;

        match (&self.before, self.operation) {
            (None, BeliefRevisionOperation::Initialized) => {}
            (None, _) => return Err(BeliefRevisionError::MissingBeforeState),
            (Some(_), BeliefRevisionOperation::Initialized) => {
                return Err(BeliefRevisionError::RepeatedInitialization)
            }
            (Some(before), _) => {
                if !before.same_question(&self.after) {
                    return Err(BeliefRevisionError::QuestionChangedWithinRevision);
                }
                if before == &self.after {
                    return Err(BeliefRevisionError::NoOpRevision);
                }
            }
        }

        if self.operation == BeliefRevisionOperation::ResetToUnknown
            && !self.after.is_complete_abstention()
        {
            return Err(BeliefRevisionError::ResetMustFullyAbstain);
        }

        Ok(())
    }
}

impl<'de> Deserialize<'de> for BeliefRevision {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = BeliefRevisionWire::deserialize(deserializer)?;
        Self::new(
            wire.revision,
            wire.operation,
            wire.before,
            wire.after,
            wire.cause,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BeliefRevisionLedger {
    capacity: usize,
    entries: Vec<BeliefRevision>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct BeliefRevisionLedgerWire {
    capacity: usize,
    entries: Vec<BeliefRevision>,
}

impl BeliefRevisionLedger {
    pub fn new(capacity: usize) -> Result<Self, BeliefRevisionError> {
        if capacity == 0 || capacity > MAX_LEDGER_CAPACITY {
            return Err(BeliefRevisionError::InvalidCapacity);
        }
        Ok(Self {
            capacity,
            entries: Vec::with_capacity(capacity),
        })
    }

    pub fn initialize(
        &mut self,
        initial: VisualBeliefSnapshot,
        cause: VisualEvidence,
    ) -> Result<&BeliefRevision, BeliefRevisionError> {
        if !self.entries.is_empty() {
            return Err(BeliefRevisionError::AlreadyInitialized);
        }
        let revision = BeliefRevision::new(
            1,
            BeliefRevisionOperation::Initialized,
            None,
            initial,
            cause,
        )?;
        self.entries.push(revision);
        Ok(self.entries.last().expect("just pushed revision"))
    }

    pub fn append(
        &mut self,
        operation: BeliefRevisionOperation,
        after: VisualBeliefSnapshot,
        cause: VisualEvidence,
    ) -> Result<&BeliefRevision, BeliefRevisionError> {
        if self.entries.is_empty() {
            return Err(BeliefRevisionError::NotInitialized);
        }
        if self.entries.len() >= self.capacity {
            return Err(BeliefRevisionError::CapacityExceeded);
        }
        if operation == BeliefRevisionOperation::Initialized {
            return Err(BeliefRevisionError::RepeatedInitialization);
        }

        let previous = self
            .entries
            .last()
            .expect("non-empty ledger checked above")
            .after
            .clone();
        if !previous.same_question(&after) {
            return Err(BeliefRevisionError::QuestionChangedAcrossLedger);
        }
        let revision_number = self
            .entries
            .last()
            .expect("non-empty ledger checked above")
            .revision
            .checked_add(1)
            .ok_or(BeliefRevisionError::RevisionOverflow)?;
        let revision = BeliefRevision::new(
            revision_number,
            operation,
            Some(previous),
            after,
            cause,
        )?;
        self.entries.push(revision);
        Ok(self.entries.last().expect("just pushed revision"))
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn entries(&self) -> &[BeliefRevision] {
        &self.entries
    }

    pub fn current(&self) -> Option<&VisualBeliefSnapshot> {
        self.entries.last().map(BeliefRevision::after)
    }

    fn validate_history(&self) -> Result<(), BeliefRevisionError> {
        if self.capacity == 0 || self.capacity > MAX_LEDGER_CAPACITY {
            return Err(BeliefRevisionError::InvalidCapacity);
        }
        if self.entries.len() > self.capacity {
            return Err(BeliefRevisionError::CapacityExceeded);
        }
        if self.entries.is_empty() {
            return Ok(());
        }

        let first = &self.entries[0];
        if first.revision != 1
            || first.operation != BeliefRevisionOperation::Initialized
            || first.before.is_some()
        {
            return Err(BeliefRevisionError::InvalidInitialRevision);
        }

        for (index, entry) in self.entries.iter().enumerate() {
            entry.validate_local()?;
            let expected_revision = (index as u64)
                .checked_add(1)
                .ok_or(BeliefRevisionError::RevisionOverflow)?;
            if entry.revision != expected_revision {
                return Err(BeliefRevisionError::NonContiguousRevision);
            }
            if index == 0 {
                continue;
            }
            let previous = &self.entries[index - 1];
            if entry.before.as_ref() != Some(&previous.after) {
                return Err(BeliefRevisionError::BrokenStateContinuity);
            }
            if !previous.after.same_question(&entry.after) {
                return Err(BeliefRevisionError::QuestionChangedAcrossLedger);
            }
        }

        Ok(())
    }
}

impl<'de> Deserialize<'de> for BeliefRevisionLedger {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = BeliefRevisionLedgerWire::deserialize(deserializer)?;
        let ledger = Self {
            capacity: wire.capacity,
            entries: wire.entries,
        };
        ledger.validate_history().map_err(serde::de::Error::custom)?;
        Ok(ledger)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeliefRevisionError {
    InvalidCapacity,
    CapacityExceeded,
    ZeroRevision,
    RevisionOverflow,
    AlreadyInitialized,
    NotInitialized,
    MissingBeforeState,
    RepeatedInitialization,
    InvalidInitialRevision,
    NonContiguousRevision,
    BrokenStateContinuity,
    QuestionChangedWithinRevision,
    QuestionChangedAcrossLedger,
    NoOpRevision,
    ObservedRevisionCause,
    GenerativeRevisionCause,
    ResetMustFullyAbstain,
}

impl fmt::Display for BeliefRevisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidCapacity => "belief revision ledger capacity must be within 1..=4096",
            Self::CapacityExceeded => "belief revision ledger capacity exceeded; history is not evicted silently",
            Self::ZeroRevision => "belief revision numbers start at one",
            Self::RevisionOverflow => "belief revision number overflow",
            Self::AlreadyInitialized => "belief revision ledger is already initialized",
            Self::NotInitialized => "belief revision ledger must be initialized before appending",
            Self::MissingBeforeState => "non-initial belief revision requires a before state",
            Self::RepeatedInitialization => "initialized operation is valid only for the first revision",
            Self::InvalidInitialRevision => "first ledger entry must be revision 1 with Initialized and no before state",
            Self::NonContiguousRevision => "belief revision numbers must be contiguous",
            Self::BrokenStateContinuity => "revision before-state must exactly equal the prior revision after-state",
            Self::QuestionChangedWithinRevision => "a belief revision cannot switch to a different belief question",
            Self::QuestionChangedAcrossLedger => "one revision ledger cannot mix different belief questions",
            Self::NoOpRevision => "belief revision must change the stored belief state",
            Self::ObservedRevisionCause => "belief revision is caused by inference from observations, not raw observation itself",
            Self::GenerativeRevisionCause => "predicted/simulated/counterfactual state cannot rewrite historical beliefs",
            Self::ResetMustFullyAbstain => "ResetToUnknown must produce a fully unassigned belief state",
        };
        f.write_str(message)
    }
}

impl std::error::Error for BeliefRevisionError {}

fn validate_historical_cause(cause: &VisualEvidence) -> Result<(), BeliefRevisionError> {
    match cause.origin() {
        VisualOrigin::Inferred | VisualOrigin::Remembered => Ok(()),
        VisualOrigin::Observed => Err(BeliefRevisionError::ObservedRevisionCause),
        VisualOrigin::Predicted | VisualOrigin::Simulated | VisualOrigin::Counterfactual => {
            Err(BeliefRevisionError::GenerativeRevisionCause)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::competing_beliefs::{BeliefMass, SemanticClassCandidate};
    use crate::entity_identity::VisualEntityHypothesisRef;
    use crate::epistemic::{VisualCaptureClock, VisualObservationRef, VisualStreamRef};

    fn observation(frame: u64) -> VisualObservationRef {
        VisualObservationRef::new(
            VisualStreamRef::new(12, 3).unwrap(),
            frame,
            1_000 + frame,
            VisualCaptureClock::StreamMonotonic,
        )
    }

    fn inferred(frame: u64) -> VisualEvidence {
        VisualEvidence::inferred(vec![observation(frame)], 0.8).unwrap()
    }

    fn semantic(cup: f32, container: f32, unknown: f32) -> VisualBeliefSnapshot {
        let subject = VisualEntityHypothesisRef::new(21, 1).unwrap();
        let mut candidates = Vec::new();
        if cup > 0.0 {
            candidates.push(
                SemanticClassCandidate::new(
                    "cup",
                    BeliefMass::new(cup).unwrap(),
                    inferred(1),
                )
                .unwrap(),
            );
        }
        if container > 0.0 {
            candidates.push(
                SemanticClassCandidate::new(
                    "container",
                    BeliefMass::new(container).unwrap(),
                    inferred(1),
                )
                .unwrap(),
            );
        }
        VisualBeliefSnapshot::SemanticClass(
            SemanticClassBeliefSet::new(
                subject,
                "open-vocabulary-v1",
                candidates,
                BeliefMass::new(unknown).unwrap(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn ledger_preserves_exact_before_after_continuity() {
        let mut ledger = BeliefRevisionLedger::new(8).unwrap();
        ledger.initialize(semantic(0.2, 0.1, 0.7), inferred(1)).unwrap();
        ledger
            .append(
                BeliefRevisionOperation::EvidenceAssimilated,
                semantic(0.5, 0.2, 0.3),
                inferred(2),
            )
            .unwrap();
        assert_eq!(ledger.entries().len(), 2);
        assert_eq!(
            ledger.entries()[1].before(),
            Some(ledger.entries()[0].after())
        );
    }

    #[test]
    fn no_op_revision_is_rejected() {
        let initial = semantic(0.2, 0.1, 0.7);
        let mut ledger = BeliefRevisionLedger::new(8).unwrap();
        ledger.initialize(initial.clone(), inferred(1)).unwrap();
        assert_eq!(
            ledger
                .append(BeliefRevisionOperation::Reweighted, initial, inferred(2))
                .unwrap_err(),
            BeliefRevisionError::NoOpRevision
        );
    }

    #[test]
    fn reset_to_unknown_requires_complete_abstention() {
        let mut ledger = BeliefRevisionLedger::new(8).unwrap();
        ledger.initialize(semantic(0.4, 0.2, 0.4), inferred(1)).unwrap();
        assert_eq!(
            ledger
                .append(
                    BeliefRevisionOperation::ResetToUnknown,
                    semantic(0.1, 0.0, 0.9),
                    inferred(2),
                )
                .unwrap_err(),
            BeliefRevisionError::ResetMustFullyAbstain
        );
        ledger
            .append(
                BeliefRevisionOperation::ResetToUnknown,
                semantic(0.0, 0.0, 1.0),
                inferred(2),
            )
            .unwrap();
    }

    #[test]
    fn generative_state_cannot_rewrite_historical_belief() {
        let mut ledger = BeliefRevisionLedger::new(8).unwrap();
        ledger.initialize(semantic(0.2, 0.1, 0.7), inferred(1)).unwrap();
        let predicted = VisualEvidence::predicted(vec![observation(2)], 0.9).unwrap();
        assert_eq!(
            ledger
                .append(
                    BeliefRevisionOperation::EvidenceAssimilated,
                    semantic(0.6, 0.1, 0.3),
                    predicted,
                )
                .unwrap_err(),
            BeliefRevisionError::GenerativeRevisionCause
        );
    }

    #[test]
    fn capacity_refuses_history_loss() {
        let mut ledger = BeliefRevisionLedger::new(1).unwrap();
        ledger.initialize(semantic(0.2, 0.1, 0.7), inferred(1)).unwrap();
        assert_eq!(
            ledger
                .append(
                    BeliefRevisionOperation::EvidenceAssimilated,
                    semantic(0.5, 0.2, 0.3),
                    inferred(2),
                )
                .unwrap_err(),
            BeliefRevisionError::CapacityExceeded
        );
    }

    #[test]
    fn deserialization_rejects_broken_continuity() {
        let mut ledger = BeliefRevisionLedger::new(8).unwrap();
        ledger.initialize(semantic(0.2, 0.1, 0.7), inferred(1)).unwrap();
        ledger
            .append(
                BeliefRevisionOperation::EvidenceAssimilated,
                semantic(0.5, 0.2, 0.3),
                inferred(2),
            )
            .unwrap();
        let mut value = serde_json::to_value(&ledger).unwrap();
        value["entries"][1]["before"] = serde_json::to_value(semantic(0.1, 0.1, 0.8)).unwrap();
        assert!(serde_json::from_value::<BeliefRevisionLedger>(value).is_err());
    }
}
