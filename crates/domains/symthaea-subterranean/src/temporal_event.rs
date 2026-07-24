// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded causal-event ordering under uncertain timestamps.
//!
//! Timestamp order is not treated as causal proof. Events identify explicit
//! dependencies, and the ledger rejects replay, unknown dependencies, cycles,
//! and intervals that make an asserted dependency impossible.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const TEMPORAL_EVENT_SCHEMA_VERSION: u16 = 1;
pub const MAX_CAUSAL_EVENTS: usize = 128;
pub const MAX_EVENT_DEPENDENCIES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CausalEventId {
    pub source: u16,
    pub boot_epoch: u64,
    pub sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalEventKind {
    SensorObservation,
    CommandIssued,
    ActuatorResponse,
    HazardDetected,
    PlanCreated,
    PlanInvalidated,
    OperatorDecision,
    EvidenceRecorded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeInterval {
    pub earliest_ns: u64,
    pub latest_ns: u64,
}

impl TimeInterval {
    pub const fn point(time_ns: u64) -> Self {
        Self {
            earliest_ns: time_ns,
            latest_ns: time_ns,
        }
    }

    pub const fn validate(self) -> bool {
        self.earliest_ns <= self.latest_ns
    }

    pub const fn definitely_before(self, other: Self) -> bool {
        self.latest_ns < other.earliest_ns
    }

    pub const fn impossible_predecessor_of(self, other: Self) -> bool {
        self.earliest_ns > other.latest_ns
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalEvent {
    pub id: CausalEventId,
    pub kind: CausalEventKind,
    pub interval: TimeInterval,
    pub observed_step: u64,
    pub dependencies: Vec<CausalEventId>,
    pub state_revision: u64,
    pub payload_digest: u64,
}

impl CausalEvent {
    pub fn validate(&self) -> bool {
        self.interval.validate()
            && self.dependencies.len() <= MAX_EVENT_DEPENDENCIES
            && self.dependencies.iter().all(|dependency| *dependency != self.id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventOrdering {
    Ordered,
    Concurrent,
    Late,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventAppendAssessment {
    pub ordering: EventOrdering,
    pub ambiguous_dependencies: usize,
    pub dropped_oldest: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventAppendError {
    Malformed,
    Replay,
    UnknownDependency,
    DependencyContradiction,
    SourceEpochRegression,
    SourceSequenceReplay,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEventLedger {
    schema_version: u16,
    events: VecDeque<CausalEvent>,
    dropped_events: u64,
    rejected_events: u64,
    contradictory_events: u64,
}

impl Default for CausalEventLedger {
    fn default() -> Self {
        Self {
            schema_version: TEMPORAL_EVENT_SCHEMA_VERSION,
            events: VecDeque::with_capacity(MAX_CAUSAL_EVENTS),
            dropped_events: 0,
            rejected_events: 0,
            contradictory_events: 0,
        }
    }
}

impl CausalEventLedger {
    pub fn validate(&self) -> bool {
        self.schema_version == TEMPORAL_EVENT_SCHEMA_VERSION
            && self.events.len() <= MAX_CAUSAL_EVENTS
            && self.events.iter().all(CausalEvent::validate)
            && self.events.iter().enumerate().all(|(index, event)| {
                self.events
                    .iter()
                    .skip(index + 1)
                    .all(|other| other.id != event.id)
            })
    }

    pub fn append(
        &mut self,
        event: CausalEvent,
    ) -> Result<EventAppendAssessment, EventAppendError> {
        if !event.validate() {
            self.rejected_events = self.rejected_events.saturating_add(1);
            return Err(EventAppendError::Malformed);
        }
        if self.events.iter().any(|existing| existing.id == event.id) {
            self.rejected_events = self.rejected_events.saturating_add(1);
            return Err(EventAppendError::Replay);
        }
        if let Some(last) = self
            .events
            .iter()
            .rev()
            .find(|existing| existing.id.source == event.id.source)
        {
            if event.id.boot_epoch < last.id.boot_epoch {
                self.rejected_events = self.rejected_events.saturating_add(1);
                return Err(EventAppendError::SourceEpochRegression);
            }
            if event.id.boot_epoch == last.id.boot_epoch && event.id.sequence <= last.id.sequence {
                self.rejected_events = self.rejected_events.saturating_add(1);
                return Err(EventAppendError::SourceSequenceReplay);
            }
        }

        let mut ambiguous_dependencies = 0usize;
        for dependency_id in &event.dependencies {
            let Some(dependency) = self
                .events
                .iter()
                .find(|candidate| candidate.id == *dependency_id)
            else {
                self.rejected_events = self.rejected_events.saturating_add(1);
                return Err(EventAppendError::UnknownDependency);
            };
            if dependency.interval.impossible_predecessor_of(event.interval) {
                self.rejected_events = self.rejected_events.saturating_add(1);
                self.contradictory_events = self.contradictory_events.saturating_add(1);
                return Err(EventAppendError::DependencyContradiction);
            }
            if !dependency.interval.definitely_before(event.interval) {
                ambiguous_dependencies = ambiguous_dependencies.saturating_add(1);
            }
        }

        let ordering = match self.events.back() {
            Some(last) if event.interval.latest_ns < last.interval.earliest_ns => {
                EventOrdering::Late
            }
            Some(last) if !last.interval.definitely_before(event.interval) => {
                EventOrdering::Concurrent
            }
            _ => EventOrdering::Ordered,
        };
        let dropped_oldest = self.events.len() == MAX_CAUSAL_EVENTS;
        if dropped_oldest {
            self.events.pop_front();
            self.dropped_events = self.dropped_events.saturating_add(1);
        }
        self.events.push_back(event);
        Ok(EventAppendAssessment {
            ordering,
            ambiguous_dependencies,
            dropped_oldest,
        })
    }

    pub fn contains(&self, id: CausalEventId) -> bool {
        self.events.iter().any(|event| event.id == id)
    }

    pub fn records(&self) -> Vec<CausalEvent> {
        self.events.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub const fn dropped_events(&self) -> u64 {
        self.dropped_events
    }

    pub const fn rejected_events(&self) -> u64 {
        self.rejected_events
    }

    pub const fn contradictory_events(&self) -> u64 {
        self.contradictory_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(sequence: u64, time: u64, dependencies: Vec<CausalEventId>) -> CausalEvent {
        CausalEvent {
            id: CausalEventId {
                source: 1,
                boot_epoch: 1,
                sequence,
            },
            kind: CausalEventKind::SensorObservation,
            interval: TimeInterval::point(time),
            observed_step: sequence,
            dependencies,
            state_revision: sequence,
            payload_digest: sequence,
        }
    }

    #[test]
    fn explicit_dependency_must_exist_and_be_temporally_possible() {
        let mut ledger = CausalEventLedger::default();
        let first = event(1, 100, vec![]);
        let first_id = first.id;
        ledger.append(first).unwrap();
        assert!(matches!(
            ledger.append(event(2, 90, vec![first_id])),
            Err(EventAppendError::DependencyContradiction)
        ));
        assert_eq!(ledger.contradictory_events(), 1);
    }

    #[test]
    fn late_independent_event_is_retained_without_rewriting_causality() {
        let mut ledger = CausalEventLedger::default();
        ledger.append(event(1, 100, vec![])).unwrap();
        let mut late = event(1, 90, vec![]);
        late.id.source = 2;
        let assessment = ledger.append(late).unwrap();
        assert_eq!(assessment.ordering, EventOrdering::Late);
        assert_eq!(ledger.len(), 2);
    }

    #[test]
    fn source_sequence_replay_is_rejected_even_with_new_payload() {
        let mut ledger = CausalEventLedger::default();
        ledger.append(event(1, 100, vec![])).unwrap();
        let mut replay = event(1, 110, vec![]);
        replay.payload_digest = 99;
        assert!(matches!(
            ledger.append(replay),
            Err(EventAppendError::SourceSequenceReplay | EventAppendError::Replay)
        ));
    }
}
