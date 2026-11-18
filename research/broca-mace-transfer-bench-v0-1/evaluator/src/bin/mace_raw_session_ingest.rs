// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Raw-session ingestion and bitemporal replay slice for Broca MACE.
//!
//! This harness starts from append-only session events, not prebuilt Experience
//! records. It reconstructs what was valid at V and known by T before exposing
//! qualified experiences to the candidate policy.

use std::collections::{BTreeMap, BTreeSet};

use broca_mace_policy::{
    decide, Action, Experience, Objective, Policy, Query, Regime, Scope,
};

#[derive(Clone, Copy, Debug)]
enum RawPayload {
    Outcome {
        action: Action,
        observed_utility: f64,
        measurement_error_bound: f64,
        intervention: bool,
        prediction_pre_registered: bool,
        outcome_reconciled: bool,
        source_trusted: bool,
        confounders_controlled: bool,
        delayed_horizon_complete: bool,
    },
    Retraction {
        target_event_id: u32,
    },
}

#[derive(Clone, Copy, Debug)]
struct RawSessionEvent {
    event_id: u32,
    session_id: u8,
    source_family: u16,
    source_root_id: u32,
    scope: Scope,
    valid_from: i64,
    valid_to: Option<i64>,
    /// Transaction/knowledge time: when this event entered the ledger.
    recorded_at: i64,
    payload: RawPayload,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IngestError {
    DuplicateEventId(u32),
    InvalidValidInterval(u32),
    InvalidOutcome(u32),
    UnknownRetractionTarget(u32),
    RetractionTargetsNonOutcome(u32),
    RetractionPrecedesSource(u32),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateEventId(id) => write!(f, "duplicate event ID {id}"),
            Self::InvalidValidInterval(id) => write!(f, "invalid valid-time interval {id}"),
            Self::InvalidOutcome(id) => write!(f, "invalid outcome payload {id}"),
            Self::UnknownRetractionTarget(id) => {
                write!(f, "unknown retraction target {id}")
            }
            Self::RetractionTargetsNonOutcome(id) => {
                write!(f, "retraction target is not an outcome {id}")
            }
            Self::RetractionPrecedesSource(id) => {
                write!(f, "retraction predates source event {id}")
            }
        }
    }
}

#[derive(Clone, Debug)]
struct MemorySnapshot {
    valid_at: i64,
    known_at: i64,
    experiences: Vec<Experience>,
    source_event_ids: Vec<u32>,
    applied_retraction_ids: Vec<u32>,
    contributing_sessions: Vec<u8>,
}

impl MemorySnapshot {
    fn event_ids(&self) -> Vec<u32> {
        self.experiences.iter().map(|item| item.event_id).collect()
    }
}

fn validate_ledger(events: &[RawSessionEvent]) -> Result<(), IngestError> {
    let mut by_id = BTreeMap::new();

    for event in events {
        if by_id.insert(event.event_id, event).is_some() {
            return Err(IngestError::DuplicateEventId(event.event_id));
        }

        if let RawPayload::Outcome {
            observed_utility,
            measurement_error_bound,
            ..
        } = event.payload
        {
            if event.valid_to.is_some_and(|end| end <= event.valid_from) {
                return Err(IngestError::InvalidValidInterval(event.event_id));
            }
            if !observed_utility.is_finite()
                || !(0.0..=1.0).contains(&observed_utility)
                || !measurement_error_bound.is_finite()
                || !(0.0..=1.0).contains(&measurement_error_bound)
            {
                return Err(IngestError::InvalidOutcome(event.event_id));
            }
        }
    }

    for event in events {
        if let RawPayload::Retraction { target_event_id } = event.payload {
            let target = by_id
                .get(&target_event_id)
                .ok_or(IngestError::UnknownRetractionTarget(target_event_id))?;
            if !matches!(target.payload, RawPayload::Outcome { .. }) {
                return Err(IngestError::RetractionTargetsNonOutcome(
                    target_event_id,
                ));
            }
            if event.recorded_at < target.recorded_at {
                return Err(IngestError::RetractionPrecedesSource(
                    target_event_id,
                ));
            }
        }
    }

    Ok(())
}

/// Reconstructs a memory view for world-valid time V and ledger-known time T.
///
/// - Valid intervals are half-open: [valid_from, valid_to).
/// - Events recorded after T are not visible, even if valid at V.
/// - A retraction recorded by T removes its target from this as-known view.
/// - The append-only input ledger is never mutated.
fn consolidate_as_known(
    events: &[RawSessionEvent],
    valid_at: i64,
    known_at: i64,
) -> Result<MemorySnapshot, IngestError> {
    validate_ledger(events)?;

    let mut retracted_ids = BTreeSet::new();
    let mut applied_retraction_ids = Vec::new();
    let mut sessions = BTreeSet::new();

    for event in events {
        if event.recorded_at > known_at {
            continue;
        }
        if let RawPayload::Retraction { target_event_id } = event.payload {
            retracted_ids.insert(target_event_id);
            applied_retraction_ids.push(event.event_id);
            sessions.insert(event.session_id);
        }
    }

    let mut experiences = Vec::new();
    let mut source_event_ids = Vec::new();

    for event in events {
        if event.recorded_at > known_at
            || event.valid_from > valid_at
            || event.valid_to.is_some_and(|end| valid_at >= end)
            || retracted_ids.contains(&event.event_id)
        {
            continue;
        }

        if let RawPayload::Outcome {
            action,
            observed_utility,
            measurement_error_bound,
            intervention,
            prediction_pre_registered,
            outcome_reconciled,
            source_trusted,
            confounders_controlled,
            delayed_horizon_complete,
        } = event.payload
        {
            experiences.push(Experience {
                event_id: event.event_id,
                source_family: event.source_family,
                source_root_id: event.source_root_id,
                scope: event.scope,
                action,
                observed_utility,
                measurement_error_bound,
                intervention,
                prediction_pre_registered,
                outcome_reconciled,
                source_trusted,
                source_retracted: false,
                confounders_controlled,
                delayed_horizon_complete,
            });
            source_event_ids.push(event.event_id);
            sessions.insert(event.session_id);
        }
    }

    experiences.sort_by_key(|item| item.event_id);
    source_event_ids.sort_unstable();
    applied_retraction_ids.sort_unstable();

    Ok(MemorySnapshot {
        valid_at,
        known_at,
        experiences,
        source_event_ids,
        applied_retraction_ids,
        contributing_sessions: sessions.into_iter().collect(),
    })
}

fn outcome(
    event_id: u32,
    session_id: u8,
    source_family: u16,
    source_root_id: u32,
    scope: Scope,
    action: Action,
    utility: f64,
    recorded_at: i64,
) -> RawSessionEvent {
    RawSessionEvent {
        event_id,
        session_id,
        source_family,
        source_root_id,
        scope,
        valid_from: 0,
        valid_to: Some(100),
        recorded_at,
        payload: RawPayload::Outcome {
            action,
            observed_utility: utility,
            measurement_error_bound: 0.01,
            intervention: true,
            prediction_pre_registered: true,
            outcome_reconciled: true,
            source_trusted: true,
            confounders_controlled: true,
            delayed_horizon_complete: true,
        },
    }
}

fn retraction(event_id: u32, target_event_id: u32, recorded_at: i64) -> RawSessionEvent {
    RawSessionEvent {
        event_id,
        session_id: 3,
        source_family: 0,
        source_root_id: 0,
        scope: Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        },
        valid_from: recorded_at,
        valid_to: None,
        recorded_at,
        payload: RawPayload::Retraction { target_event_id },
    }
}

fn session_chain() -> Vec<RawSessionEvent> {
    let quality = Scope {
        regime: Regime::Baseline,
        objective: Objective::Quality,
    };
    vec![
        outcome(1, 1, 10, 100, quality, Action::A, 0.92, 10),
        outcome(2, 1, 11, 101, quality, Action::A, 0.90, 10),
        outcome(3, 1, 20, 200, quality, Action::B, 0.32, 10),
        outcome(4, 1, 21, 201, quality, Action::B, 0.34, 10),
        retraction(5, 3, 30),
        retraction(6, 4, 31),
        // Valid in the old world, but arrived after Session B.
        outcome(7, 2, 30, 300, quality, Action::A, 0.95, 40),
    ]
}

fn run_demo() -> Result<(), IngestError> {
    let events = session_chain();
    let query = Query {
        scope: Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        },
    };

    let before_correction = consolidate_as_known(&events, 20, 29)?;
    let b_decision = decide(
        Policy::CausalQualified,
        &before_correction.experiences,
        query,
    );

    let after_correction = consolidate_as_known(&events, 20, 31)?;
    let c_decision = decide(
        Policy::CausalQualified,
        &after_correction.experiences,
        query,
    );
    let active = decide(Policy::ActiveProbe, &after_correction.experiences, query);
    let after_late_arrival = consolidate_as_known(&events, 20, 40)?;

    println!("MACE raw-session bitemporal replay v0.1");
    println!("raw_events={}", events.len());
    println!(
        "V={}, T={}: B_action={:?}, events={:?}, sessions={:?}",
        before_correction.valid_at,
        before_correction.known_at,
        b_decision.action,
        before_correction.source_event_ids,
        before_correction.contributing_sessions
    );
    println!(
        "V={}, T={}: C_action={:?}, events={:?}, retractions={:?}, probe={}",
        after_correction.valid_at,
        after_correction.known_at,
        c_decision.action,
        after_correction.source_event_ids,
        after_correction.applied_retraction_ids,
        active.probe.is_some()
    );
    println!(
        "late_event_7_visible_at_T29={}, visible_at_T40={}",
        before_correction.source_event_ids.contains(&7),
        after_late_arrival.source_event_ids.contains(&7)
    );
    Ok(())
}

fn main() {
    if let Err(error) = run_demo() {
        eprintln!("raw-session replay failed closed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_a_b_c_replay_does_not_backdate_a_late_retraction() {
        let events = session_chain();
        let quality = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let query = Query { scope: quality };

        let at_b = consolidate_as_known(&events, 20, 29).unwrap();
        let b_decision = decide(Policy::CausalQualified, &at_b.experiences, query);
        assert_eq!(b_decision.action, Some(Action::A));
        assert_eq!(at_b.event_ids(), vec![1, 2, 3, 4]);
        assert!(at_b.applied_retraction_ids.is_empty());

        let at_c = consolidate_as_known(&events, 20, 31).unwrap();
        let c_decision = decide(Policy::CausalQualified, &at_c.experiences, query);
        assert_eq!(c_decision.action, None);
        assert_eq!(at_c.event_ids(), vec![3, 4]);
        assert_eq!(at_c.applied_retraction_ids, vec![5, 6]);

        let active = decide(Policy::ActiveProbe, &at_c.experiences, query);
        assert!(active.probe.is_some());
        assert_eq!(active.action, None);
    }

    #[test]
    fn late_arriving_past_event_does_not_leak_into_earlier_knowledge() {
        let events = session_chain();
        let before_arrival = consolidate_as_known(&events, 20, 39).unwrap();
        let after_arrival = consolidate_as_known(&events, 20, 40).unwrap();

        assert!(!before_arrival.event_ids().contains(&7));
        assert!(after_arrival.event_ids().contains(&7));
        assert_eq!(before_arrival.valid_at, after_arrival.valid_at);
        assert_ne!(before_arrival.known_at, after_arrival.known_at);
    }

    #[test]
    fn valid_time_interval_is_half_open() {
        let events = session_chain();
        let inside = consolidate_as_known(&events, 99, 29).unwrap();
        let outside = consolidate_as_known(&events, 100, 29).unwrap();

        assert!(inside.event_ids().contains(&1));
        assert!(!outside.event_ids().contains(&1));
    }

    #[test]
    fn duplicate_event_ids_fail_closed() {
        let mut events = session_chain();
        events.push(events[0]);
        assert_eq!(
            consolidate_as_known(&events, 20, 29).unwrap_err(),
            IngestError::DuplicateEventId(1)
        );
    }

    #[test]
    fn unknown_retraction_targets_fail_closed() {
        let mut events = session_chain();
        events.push(retraction(8, 999, 50));
        assert_eq!(
            consolidate_as_known(&events, 20, 50).unwrap_err(),
            IngestError::UnknownRetractionTarget(999)
        );
    }

    #[test]
    fn retractions_cannot_predate_the_source_transaction() {
        let mut events = session_chain();
        events.push(retraction(8, 1, 9));
        assert_eq!(
            consolidate_as_known(&events, 20, 50).unwrap_err(),
            IngestError::RetractionPrecedesSource(1)
        );
    }

    #[test]
    fn invalid_measurement_bounds_fail_closed_at_ingest() {
        let mut events = session_chain();
        if let RawPayload::Outcome {
            measurement_error_bound,
            ..
        } = &mut events[0].payload
        {
            *measurement_error_bound = f64::NAN;
        }
        assert_eq!(
            consolidate_as_known(&events, 20, 29).unwrap_err(),
            IngestError::InvalidOutcome(1)
        );
    }

    #[test]
    fn replay_is_deterministic_and_input_ledger_is_unchanged() {
        let events = session_chain();
        let original_len = events.len();
        let first = consolidate_as_known(&events, 20, 31).unwrap();
        let second = consolidate_as_known(&events, 20, 31).unwrap();

        assert_eq!(first.event_ids(), second.event_ids());
        assert_eq!(first.applied_retraction_ids, second.applied_retraction_ids);
        assert_eq!(first.contributing_sessions, second.contributing_sessions);
        assert_eq!(events.len(), original_len);
        assert_eq!(events[0].recorded_at, 10);
    }
}
