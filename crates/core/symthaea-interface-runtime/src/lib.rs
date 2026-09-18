// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded real-time runtime-plane primitives for Symthaea interfaces.
//!
//! The state plane is intentionally **latest-wins**. A slow terminal, dashboard,
//! phone client, or remote observer must not accumulate an unbounded queue of old
//! cognitive snapshots and then render stale history while claiming to be live.
//!
//! This crate therefore provides a single shared latest-state slot with independent
//! receivers. Publication is O(1) memory: each publish replaces the prior shared
//! value, while receivers retain only an `Arc` to a snapshot they are actively
//! inspecting. Per-receiver transport revisions make skipped publications explicit.
//!
//! Transport revision is not semantic runtime ordering. Semantic order remains the
//! `RuntimeCursor`/`EventSeq` carried by `symthaea-interface-types`.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use symthaea_interface_types::{RuntimeCursor, RuntimeState};

/// Non-zero transport-local revision for the latest-wins state slot.
///
/// This is deliberately distinct from `EventSeq`: replacing an old snapshot in a
/// UI transport is a flow-control event, not a semantic runtime event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateRevision(u64);

impl StateRevision {
    pub fn new(value: u64) -> Option<Self> {
        (value != 0).then_some(Self(value))
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for StateRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Failure of the bounded state-plane primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatePlaneError {
    /// The publication revision counter reached `u64::MAX`.
    RevisionExhausted,
    /// Another thread panicked while holding the state-slot mutex.
    Poisoned,
}

impl fmt::Display for StatePlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RevisionExhausted => write!(f, "state-plane revision counter exhausted"),
            Self::Poisoned => write!(f, "state-plane slot mutex poisoned"),
        }
    }
}

impl std::error::Error for StatePlaneError {}

#[derive(Debug)]
struct StateSlot<T> {
    revision: u64,
    current: Option<Arc<T>>,
}

impl<T> Default for StateSlot<T> {
    fn default() -> Self {
        Self {
            revision: 0,
            current: None,
        }
    }
}

fn lock_slot<T>(
    shared: &Arc<Mutex<StateSlot<T>>>,
) -> Result<MutexGuard<'_, StateSlot<T>>, StatePlaneError> {
    shared.lock().map_err(|_| StatePlaneError::Poisoned)
}

/// Producer half of a bounded latest-wins state plane.
///
/// Cloning the publisher does not allocate another queue; all publishers replace
/// the same single shared slot.
#[derive(Debug, Clone)]
pub struct LatestStatePublisher<T> {
    shared: Arc<Mutex<StateSlot<T>>>,
}

impl<T> LatestStatePublisher<T> {
    /// Replace the current state value and advance the transport-local revision.
    pub fn publish(&self, value: T) -> Result<StateRevision, StatePlaneError> {
        let mut slot = lock_slot(&self.shared)?;
        let revision = slot
            .revision
            .checked_add(1)
            .ok_or(StatePlaneError::RevisionExhausted)?;
        let revision = StateRevision::new(revision).ok_or(StatePlaneError::RevisionExhausted)?;
        slot.revision = revision.get();
        slot.current = Some(Arc::new(value));
        Ok(revision)
    }

    /// Create another independent observer.
    ///
    /// If state already exists, the new observer sees that current snapshot once
    /// with `skipped_revisions == 0`; publications that predate subscription are
    /// not reported as lag.
    pub fn subscribe(&self) -> Result<LatestStateReceiver<T>, StatePlaneError> {
        let slot = lock_slot(&self.shared)?;
        let last_seen_revision = if slot.current.is_some() {
            slot.revision.saturating_sub(1)
        } else {
            slot.revision
        };
        Ok(LatestStateReceiver {
            shared: Arc::clone(&self.shared),
            last_seen_revision,
        })
    }

    /// Current publication revision, or `None` before the first publish.
    pub fn current_revision(&self) -> Result<Option<StateRevision>, StatePlaneError> {
        let slot = lock_slot(&self.shared)?;
        Ok(StateRevision::new(slot.revision))
    }

    /// Whether any state has ever been published into this slot.
    pub fn is_initialized(&self) -> Result<bool, StatePlaneError> {
        let slot = lock_slot(&self.shared)?;
        Ok(slot.current.is_some())
    }
}

/// One observation from the latest-wins state plane.
#[derive(Debug, Clone)]
pub struct LatestStateSample<T> {
    /// Revision of the returned state publication.
    pub revision: StateRevision,
    /// Number of intermediate state publications this receiver did not observe.
    ///
    /// This is transport lag, not a semantic `EventSeq` gap.
    pub skipped_revisions: u64,
    /// Immutable shared snapshot. A later publish cannot mutate this value.
    pub value: Arc<T>,
}

/// Consumer half of a bounded latest-wins state plane.
///
/// Receivers are independent: a slow Bevy dashboard does not hold back a terminal,
/// and neither causes the publisher to retain an unbounded history.
#[derive(Debug, Clone)]
pub struct LatestStateReceiver<T> {
    shared: Arc<Mutex<StateSlot<T>>>,
    last_seen_revision: u64,
}

impl<T> LatestStateReceiver<T> {
    /// Return the newest state only if this receiver has not observed its revision.
    ///
    /// If several publishes happened since the previous read, only the newest
    /// snapshot is returned and `skipped_revisions` reports how many were replaced.
    pub fn latest_if_changed(
        &mut self,
    ) -> Result<Option<LatestStateSample<T>>, StatePlaneError> {
        let (revision, value) = {
            let slot = lock_slot(&self.shared)?;
            if slot.revision == 0
                || slot.revision == self.last_seen_revision
                || slot.current.is_none()
            {
                return Ok(None);
            }
            (
                slot.revision,
                Arc::clone(slot.current.as_ref().expect("checked current state")),
            )
        };

        let skipped_revisions = revision
            .saturating_sub(self.last_seen_revision)
            .saturating_sub(1);
        self.last_seen_revision = revision;

        Ok(Some(LatestStateSample {
            revision: StateRevision::new(revision).expect("published revision is non-zero"),
            skipped_revisions,
            value,
        }))
    }

    /// Peek at the newest state without advancing this receiver's observation point.
    pub fn peek_latest(&self) -> Result<Option<LatestStateSample<T>>, StatePlaneError> {
        let slot = lock_slot(&self.shared)?;
        let Some(value) = slot.current.as_ref() else {
            return Ok(None);
        };
        let revision = StateRevision::new(slot.revision).expect("published revision is non-zero");
        let skipped_revisions = slot
            .revision
            .saturating_sub(self.last_seen_revision)
            .saturating_sub(1);
        Ok(Some(LatestStateSample {
            revision,
            skipped_revisions,
            value: Arc::clone(value),
        }))
    }

    /// Last revision this receiver advanced past, if any.
    pub fn last_seen_revision(&self) -> Option<StateRevision> {
        StateRevision::new(self.last_seen_revision)
    }

    /// Number of not-yet-observed publications currently ahead of this receiver.
    ///
    /// Because the slot stores only the newest value, this is useful for telemetry
    /// about client lag without allocating historical snapshots.
    pub fn pending_revision_distance(&self) -> Result<u64, StatePlaneError> {
        let slot = lock_slot(&self.shared)?;
        Ok(slot.revision.saturating_sub(self.last_seen_revision))
    }
}

/// Create one bounded state slot with an initial receiver.
///
/// The channel starts empty. The initial receiver treats publications that happen
/// before its first read as real lag and reports the overwritten count.
pub fn latest_state_channel<T>() -> (LatestStatePublisher<T>, LatestStateReceiver<T>) {
    let shared = Arc::new(Mutex::new(StateSlot::default()));
    (
        LatestStatePublisher {
            shared: Arc::clone(&shared),
        },
        LatestStateReceiver {
            shared,
            last_seen_revision: 0,
        },
    )
}

/// State-plane aliases for provenance-bearing Symthaea runtime state.
pub type LatestRuntimeStatePublisher<T> = LatestStatePublisher<RuntimeState<T>>;
pub type LatestRuntimeStateReceiver<T> = LatestStateReceiver<RuntimeState<T>>;
pub type LatestRuntimeStateSample<T> = LatestStateSample<RuntimeState<T>>;

/// Create a latest-wins channel specialized for `RuntimeState<T>`.
pub fn latest_runtime_state_channel<T>() -> (
    LatestRuntimeStatePublisher<T>,
    LatestRuntimeStateReceiver<T>,
) {
    latest_state_channel()
}

/// Relationship between two semantic runtime cursors.
///
/// This classification is deliberately separate from state-slot revisions. It can
/// be used by clients to distinguish ordinary latest-wins state coalescing from a
/// real gap or discontinuity in the authoritative runtime event sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorContinuity {
    /// No prior semantic cursor has been observed by this consumer.
    FirstObservation,
    /// Same runtime and exactly the next semantic event sequence.
    ImmediateSuccessor,
    /// Same runtime, forward progress, but one or more semantic events are missing.
    Gap { missed_events: u64 },
    /// Runtime identity changed; continuity must not be inferred across the boundary.
    RuntimeChanged,
    /// Same runtime but the new cursor is duplicate or moves backwards.
    DuplicateOrReordered,
}

/// Classify semantic continuity without consulting wall-clock time.
pub fn classify_cursor_continuity(
    previous: Option<&RuntimeCursor>,
    current: &RuntimeCursor,
) -> CursorContinuity {
    let Some(previous) = previous else {
        return CursorContinuity::FirstObservation;
    };

    if previous.runtime_id() != current.runtime_id() {
        return CursorContinuity::RuntimeChanged;
    }

    if current.is_immediate_successor_of(previous) {
        return CursorContinuity::ImmediateSuccessor;
    }

    let previous_seq = previous.seq().get();
    let current_seq = current.seq().get();
    match current_seq.checked_sub(previous_seq) {
        Some(delta) if delta > 1 => CursorContinuity::Gap {
            missed_events: delta - 1,
        },
        _ => CursorContinuity::DuplicateOrReordered,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interface_types::{EventSeq, RuntimeId, RuntimeState, StateProvenance};

    fn cursor(runtime: &str, seq: u64) -> RuntimeCursor {
        RuntimeCursor::new(
            RuntimeId::new(runtime).unwrap(),
            EventSeq::new(seq).unwrap(),
        )
    }

    #[test]
    fn channel_starts_empty() {
        let (publisher, mut receiver) = latest_state_channel::<u64>();
        assert!(!publisher.is_initialized().unwrap());
        assert!(publisher.current_revision().unwrap().is_none());
        assert!(receiver.latest_if_changed().unwrap().is_none());
        assert!(receiver.peek_latest().unwrap().is_none());
    }

    #[test]
    fn slow_receiver_observes_only_newest_value_and_exact_lag() {
        let (publisher, mut receiver) = latest_state_channel();
        for value in 1_u64..=10_000 {
            publisher.publish(value).unwrap();
        }

        let sample = receiver.latest_if_changed().unwrap().unwrap();
        assert_eq!(*sample.value, 10_000);
        assert_eq!(sample.revision.get(), 10_000);
        assert_eq!(sample.skipped_revisions, 9_999);
        assert!(receiver.latest_if_changed().unwrap().is_none());
    }

    #[test]
    fn multiple_receivers_progress_independently() {
        let (publisher, mut first) = latest_state_channel();
        publisher.publish("one").unwrap();
        let mut second = publisher.subscribe().unwrap();

        let first_sample = first.latest_if_changed().unwrap().unwrap();
        let second_sample = second.latest_if_changed().unwrap().unwrap();
        assert_eq!(*first_sample.value, "one");
        assert_eq!(*second_sample.value, "one");
        assert_eq!(second_sample.skipped_revisions, 0);

        publisher.publish("two").unwrap();
        publisher.publish("three").unwrap();

        let first_sample = first.latest_if_changed().unwrap().unwrap();
        assert_eq!(*first_sample.value, "three");
        assert_eq!(first_sample.skipped_revisions, 1);

        let second_sample = second.latest_if_changed().unwrap().unwrap();
        assert_eq!(*second_sample.value, "three");
        assert_eq!(second_sample.skipped_revisions, 1);
    }

    #[test]
    fn previous_arc_remains_immutable_after_replacement() {
        let (publisher, mut receiver) = latest_state_channel();
        publisher.publish(String::from("old")).unwrap();
        let old = receiver.latest_if_changed().unwrap().unwrap().value;
        publisher.publish(String::from("new")).unwrap();
        let new = receiver.latest_if_changed().unwrap().unwrap().value;

        assert_eq!(old.as_str(), "old");
        assert_eq!(new.as_str(), "new");
    }

    #[test]
    fn peek_does_not_advance_receiver() {
        let (publisher, mut receiver) = latest_state_channel();
        publisher.publish(7_u8).unwrap();
        let peeked = receiver.peek_latest().unwrap().unwrap();
        assert_eq!(*peeked.value, 7);
        assert!(receiver.last_seen_revision().is_none());

        let observed = receiver.latest_if_changed().unwrap().unwrap();
        assert_eq!(observed.revision, peeked.revision);
    }

    #[test]
    fn runtime_state_preserves_structural_unknown() {
        let (publisher, mut receiver) = latest_runtime_state_channel::<u64>();
        publisher.publish(RuntimeState::Unknown).unwrap();
        let observed = receiver.latest_if_changed().unwrap().unwrap();
        assert_eq!(observed.value.provenance(), StateProvenance::Unknown);
        assert!(observed.value.value().is_none());
        assert!(observed.value.cursor().is_none());
    }

    #[test]
    fn cursor_continuity_detects_first_immediate_gap_runtime_change_and_reorder() {
        let a1 = cursor("runtime-a", 1);
        let a2 = cursor("runtime-a", 2);
        let a5 = cursor("runtime-a", 5);
        let b1 = cursor("runtime-b", 1);

        assert_eq!(
            classify_cursor_continuity(None, &a1),
            CursorContinuity::FirstObservation
        );
        assert_eq!(
            classify_cursor_continuity(Some(&a1), &a2),
            CursorContinuity::ImmediateSuccessor
        );
        assert_eq!(
            classify_cursor_continuity(Some(&a2), &a5),
            CursorContinuity::Gap { missed_events: 2 }
        );
        assert_eq!(
            classify_cursor_continuity(Some(&a5), &b1),
            CursorContinuity::RuntimeChanged
        );
        assert_eq!(
            classify_cursor_continuity(Some(&a5), &a2),
            CursorContinuity::DuplicateOrReordered
        );
        assert_eq!(
            classify_cursor_continuity(Some(&a2), &a2),
            CursorContinuity::DuplicateOrReordered
        );
    }

    #[test]
    fn pending_distance_reports_backpressure_without_history_allocation() {
        let (publisher, mut receiver) = latest_state_channel();
        for value in 0..128_u16 {
            publisher.publish(value).unwrap();
        }
        assert_eq!(receiver.pending_revision_distance().unwrap(), 128);
        let sample = receiver.latest_if_changed().unwrap().unwrap();
        assert_eq!(*sample.value, 127);
        assert_eq!(receiver.pending_revision_distance().unwrap(), 0);
    }
}
