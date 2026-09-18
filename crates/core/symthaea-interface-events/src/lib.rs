// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded ordered semantic-event delivery for Symthaea interfaces.
//!
//! Unlike the state plane, semantic events are never latest-wins. They represent
//! ordered interaction facts. The publisher therefore accepts only one contiguous
//! `RuntimeCursor` lineage and retains a finite ring. A slow subscriber may read
//! retained events in order; once it falls behind retention it receives an explicit
//! `ReplayRequired` result rather than silently skipping semantic history.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use symthaea_interface_types::{EventSeq, RuntimeCursor, RuntimeEvent, RuntimeId};

/// Event-plane construction/publication failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventPlaneError {
    ZeroCapacity,
    Poisoned,
    RuntimeChanged {
        expected: RuntimeId,
        found: RuntimeId,
    },
    NonContiguous {
        expected: EventSeq,
        found: EventSeq,
    },
    SequenceExhausted,
    ZeroReadLimit,
}

impl fmt::Display for EventPlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "semantic event retention capacity must be non-zero"),
            Self::Poisoned => write!(f, "semantic event ring mutex poisoned"),
            Self::RuntimeChanged { expected, found } => write!(
                f,
                "semantic event runtime changed from {expected} to {found}"
            ),
            Self::NonContiguous { expected, found } => write!(
                f,
                "non-contiguous semantic event sequence: expected {expected}, found {found}"
            ),
            Self::SequenceExhausted => write!(f, "semantic event sequence exhausted"),
            Self::ZeroReadLimit => write!(f, "semantic event read limit must be non-zero"),
        }
    }
}

impl std::error::Error for EventPlaneError {}

#[derive(Debug)]
struct EventRing {
    capacity: usize,
    events: VecDeque<Arc<RuntimeEvent>>,
    /// First cursor ever published into this lineage. It survives retention eviction
    /// so subscribers that existed before the first event can detect falling behind.
    first_cursor: Option<RuntimeCursor>,
    /// Last published cursor is retained even when its event later ages out.
    last_cursor: Option<RuntimeCursor>,
}

fn lock_ring(shared: &Arc<Mutex<EventRing>>) -> Result<MutexGuard<'_, EventRing>, EventPlaneError> {
    shared.lock().map_err(|_| EventPlaneError::Poisoned)
}

/// Start position for a new semantic-event subscriber.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubscribeFrom {
    /// Receive only events published after subscription.
    Tail,
    /// Consume the oldest event still retained in the finite ring.
    OldestRetained,
    /// Resume after an explicitly acknowledged semantic cursor.
    After(RuntimeCursor),
}

#[derive(Debug, Clone)]
pub struct SemanticEventPublisher {
    shared: Arc<Mutex<EventRing>>,
}

impl SemanticEventPublisher {
    /// Publish one event into the ordered ring.
    ///
    /// After the first event establishes runtime identity and sequence position,
    /// every later event must be the exact semantic successor. Runtime changes,
    /// duplicates, reordering, and forward gaps fail closed.
    pub fn publish(&self, event: RuntimeEvent) -> Result<(), EventPlaneError> {
        let mut ring = lock_ring(&self.shared)?;

        if let Some(previous) = ring.last_cursor.as_ref() {
            if event.cursor().runtime_id() != previous.runtime_id() {
                return Err(EventPlaneError::RuntimeChanged {
                    expected: previous.runtime_id().clone(),
                    found: event.cursor().runtime_id().clone(),
                });
            }

            let expected = previous
                .seq()
                .checked_next()
                .ok_or(EventPlaneError::SequenceExhausted)?;
            if event.cursor().seq() != expected {
                return Err(EventPlaneError::NonContiguous {
                    expected,
                    found: event.cursor().seq(),
                });
            }
        } else {
            ring.first_cursor = Some(event.cursor().clone());
        }

        ring.last_cursor = Some(event.cursor().clone());
        ring.events.push_back(Arc::new(event));
        while ring.events.len() > ring.capacity {
            ring.events.pop_front();
        }
        Ok(())
    }

    pub fn retained_len(&self) -> Result<usize, EventPlaneError> {
        Ok(lock_ring(&self.shared)?.events.len())
    }

    pub fn capacity(&self) -> Result<usize, EventPlaneError> {
        Ok(lock_ring(&self.shared)?.capacity)
    }

    pub fn newest_cursor(&self) -> Result<Option<RuntimeCursor>, EventPlaneError> {
        Ok(lock_ring(&self.shared)?.last_cursor.clone())
    }

    pub fn oldest_retained_cursor(&self) -> Result<Option<RuntimeCursor>, EventPlaneError> {
        Ok(lock_ring(&self.shared)?
            .events
            .front()
            .map(|event| event.cursor().clone()))
    }

    /// Create an independent subscriber without allocating a per-client queue.
    pub fn subscribe(
        &self,
        start: SubscribeFrom,
    ) -> Result<SemanticEventSubscriber, EventPlaneError> {
        let ring = lock_ring(&self.shared)?;
        let state = SubscriberState::from_start(&ring, start)?;
        Ok(SemanticEventSubscriber {
            shared: Arc::clone(&self.shared),
            state,
        })
    }
}

#[derive(Debug, Clone)]
enum SubscriberState {
    /// Subscriber existed before the lineage's first event. Once the lineage starts,
    /// it expects that exact first cursor even if the event has since aged out.
    AwaitFirst,
    /// Next semantic cursor this subscriber expects to consume.
    Next(RuntimeCursor),
    /// Sequence space ended at u64::MAX; no successor can exist.
    Exhausted { runtime_id: RuntimeId },
}

impl SubscriberState {
    fn from_start(ring: &EventRing, start: SubscribeFrom) -> Result<Self, EventPlaneError> {
        match start {
            SubscribeFrom::Tail => match ring.last_cursor.as_ref() {
                None => Ok(Self::AwaitFirst),
                Some(last) => match last.seq().checked_next() {
                    Some(next) => Ok(Self::Next(RuntimeCursor::new(
                        last.runtime_id().clone(),
                        next,
                    ))),
                    None => Ok(Self::Exhausted {
                        runtime_id: last.runtime_id().clone(),
                    }),
                },
            },
            SubscribeFrom::OldestRetained => {
                if let Some(oldest) = ring.events.front() {
                    Ok(Self::Next(oldest.cursor().clone()))
                } else {
                    Ok(Self::AwaitFirst)
                }
            }
            SubscribeFrom::After(cursor) => {
                if let Some(last) = ring.last_cursor.as_ref()
                    && cursor.runtime_id() != last.runtime_id()
                {
                    return Err(EventPlaneError::RuntimeChanged {
                        expected: last.runtime_id().clone(),
                        found: cursor.runtime_id().clone(),
                    });
                }
                match cursor.seq().checked_next() {
                    Some(next) => Ok(Self::Next(RuntimeCursor::new(
                        cursor.runtime_id().clone(),
                        next,
                    ))),
                    None => Ok(Self::Exhausted {
                        runtime_id: cursor.runtime_id().clone(),
                    }),
                }
            }
        }
    }
}

/// Ordered event batch returned from the retained ring.
#[derive(Debug, Clone)]
pub struct EventBatch {
    pub events: Vec<Arc<RuntimeEvent>>,
}

impl EventBatch {
    pub fn first_cursor(&self) -> Option<&RuntimeCursor> {
        self.events.first().map(|event| event.cursor())
    }

    pub fn last_cursor(&self) -> Option<&RuntimeCursor> {
        self.events.last().map(|event| event.cursor())
    }
}

/// Non-error result of polling one semantic-event subscriber.
#[derive(Debug, Clone)]
pub enum EventRead {
    /// No event is currently available at this subscriber's next cursor.
    NoEvents,
    /// One or more retained events, always in semantic cursor order.
    Events(EventBatch),
    /// The subscriber's next required cursor has already aged out of retention.
    ///
    /// The caller must replay from durable evidence/audit storage or explicitly
    /// resubscribe at a newer position; the event plane never silently skips.
    ReplayRequired {
        requested_next: RuntimeCursor,
        oldest_retained: RuntimeCursor,
        newest_retained: RuntimeCursor,
    },
    /// The retained ring belongs to another runtime identity.
    RuntimeChanged {
        expected: RuntimeId,
        current: RuntimeId,
    },
}

/// Independent cursor over the shared finite semantic-event ring.
#[derive(Debug, Clone)]
pub struct SemanticEventSubscriber {
    shared: Arc<Mutex<EventRing>>,
    state: SubscriberState,
}

impl SemanticEventSubscriber {
    /// Read at most `limit` contiguous retained events and advance this subscriber.
    pub fn read_available(&mut self, limit: usize) -> Result<EventRead, EventPlaneError> {
        if limit == 0 {
            return Err(EventPlaneError::ZeroReadLimit);
        }

        let ring = lock_ring(&self.shared)?;
        let Some(oldest) = ring.events.front() else {
            return Ok(EventRead::NoEvents);
        };
        let newest = ring
            .events
            .back()
            .expect("non-empty ring has newest event");
        let current_runtime = newest.cursor().runtime_id().clone();

        if matches!(&self.state, SubscriberState::AwaitFirst) {
            let first = ring
                .first_cursor
                .as_ref()
                .expect("non-empty event lineage has first cursor");
            self.state = SubscriberState::Next(first.clone());
        }

        let expected = match &self.state {
            SubscriberState::AwaitFirst => unreachable!("initialized above"),
            SubscriberState::Exhausted { runtime_id } => {
                if runtime_id != &current_runtime {
                    return Ok(EventRead::RuntimeChanged {
                        expected: runtime_id.clone(),
                        current: current_runtime,
                    });
                }
                return Ok(EventRead::NoEvents);
            }
            SubscriberState::Next(cursor) => cursor.clone(),
        };

        if expected.runtime_id() != &current_runtime {
            return Ok(EventRead::RuntimeChanged {
                expected: expected.runtime_id().clone(),
                current: current_runtime,
            });
        }

        let expected_seq = expected.seq().get();
        let oldest_seq = oldest.cursor().seq().get();
        let newest_seq = newest.cursor().seq().get();

        if expected_seq < oldest_seq {
            return Ok(EventRead::ReplayRequired {
                requested_next: expected,
                oldest_retained: oldest.cursor().clone(),
                newest_retained: newest.cursor().clone(),
            });
        }
        if expected_seq > newest_seq {
            return Ok(EventRead::NoEvents);
        }

        let offset = usize::try_from(expected_seq - oldest_seq)
            .expect("retained event offset fits usize because ring length does");
        let events: Vec<Arc<RuntimeEvent>> = ring
            .events
            .iter()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect();

        let last = events
            .last()
            .expect("expected sequence inside retained range yields an event");
        self.state = match last.cursor().seq().checked_next() {
            Some(next) => SubscriberState::Next(RuntimeCursor::new(
                current_runtime,
                next,
            )),
            None => SubscriberState::Exhausted {
                runtime_id: current_runtime,
            },
        };

        Ok(EventRead::Events(EventBatch { events }))
    }

    /// The next semantic cursor this subscriber requires, when sequence space remains.
    pub fn next_cursor(&self) -> Option<&RuntimeCursor> {
        match &self.state {
            SubscriberState::Next(cursor) => Some(cursor),
            SubscriberState::AwaitFirst | SubscriberState::Exhausted { .. } => None,
        }
    }
}

/// Create a bounded event plane plus an initial tail subscriber.
pub fn semantic_event_channel(
    capacity: usize,
) -> Result<(SemanticEventPublisher, SemanticEventSubscriber), EventPlaneError> {
    if capacity == 0 {
        return Err(EventPlaneError::ZeroCapacity);
    }
    let shared = Arc::new(Mutex::new(EventRing {
        capacity,
        events: VecDeque::new(),
        first_cursor: None,
        last_cursor: None,
    }));
    Ok((
        SemanticEventPublisher {
            shared: Arc::clone(&shared),
        },
        SemanticEventSubscriber {
            shared,
            state: SubscriberState::AwaitFirst,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interface_types::{RuntimeEventKind, SessionId};

    fn runtime(name: &str) -> RuntimeId {
        RuntimeId::new(name).unwrap()
    }

    fn cursor(runtime_name: &str, seq: u64) -> RuntimeCursor {
        RuntimeCursor::new(runtime(runtime_name), EventSeq::new(seq).unwrap())
    }

    fn event(runtime_name: &str, seq: u64) -> RuntimeEvent {
        RuntimeEvent::new(
            cursor(runtime_name, seq),
            Some(SessionId::new("session-1").unwrap()),
            RuntimeEventKind::ResponseStarted,
        )
        .unwrap()
    }

    fn seqs(read: EventRead) -> Vec<u64> {
        match read {
            EventRead::Events(batch) => batch
                .events
                .iter()
                .map(|event| event.cursor().seq().get())
                .collect(),
            other => panic!("expected events, got {other:?}"),
        }
    }

    #[test]
    fn zero_capacity_and_zero_read_limit_fail_closed() {
        assert_eq!(
            semantic_event_channel(0).unwrap_err(),
            EventPlaneError::ZeroCapacity
        );
        let (_, mut subscriber) = semantic_event_channel(2).unwrap();
        assert_eq!(
            subscriber.read_available(0).unwrap_err(),
            EventPlaneError::ZeroReadLimit
        );
    }

    #[test]
    fn publisher_accepts_only_one_contiguous_runtime_lineage() {
        let (publisher, _) = semantic_event_channel(4).unwrap();
        publisher.publish(event("runtime-a", 10)).unwrap();
        publisher.publish(event("runtime-a", 11)).unwrap();

        assert_eq!(
            publisher.publish(event("runtime-a", 11)).unwrap_err(),
            EventPlaneError::NonContiguous {
                expected: EventSeq::new(12).unwrap(),
                found: EventSeq::new(11).unwrap(),
            }
        );
        assert_eq!(
            publisher.publish(event("runtime-a", 13)).unwrap_err(),
            EventPlaneError::NonContiguous {
                expected: EventSeq::new(12).unwrap(),
                found: EventSeq::new(13).unwrap(),
            }
        );
        assert_eq!(
            publisher.publish(event("runtime-b", 12)).unwrap_err(),
            EventPlaneError::RuntimeChanged {
                expected: runtime("runtime-a"),
                found: runtime("runtime-b"),
            }
        );
    }

    #[test]
    fn retention_is_strictly_bounded() {
        let (publisher, _) = semantic_event_channel(3).unwrap();
        for seq in 1..=5 {
            publisher.publish(event("runtime-a", seq)).unwrap();
        }
        assert_eq!(publisher.capacity().unwrap(), 3);
        assert_eq!(publisher.retained_len().unwrap(), 3);
        assert_eq!(
            publisher.oldest_retained_cursor().unwrap().unwrap().seq(),
            EventSeq::new(3).unwrap()
        );
        assert_eq!(
            publisher.newest_cursor().unwrap().unwrap().seq(),
            EventSeq::new(5).unwrap()
        );
    }

    #[test]
    fn subscriber_created_before_first_event_receives_first_event() {
        let (publisher, mut subscriber) = semantic_event_channel(4).unwrap();
        assert!(matches!(
            subscriber.read_available(8).unwrap(),
            EventRead::NoEvents
        ));
        publisher.publish(event("runtime-a", 7)).unwrap();
        assert_eq!(seqs(subscriber.read_available(8).unwrap()), vec![7]);
    }

    #[test]
    fn subscriber_created_before_first_event_requires_replay_if_first_ages_out() {
        let (publisher, mut subscriber) = semantic_event_channel(2).unwrap();
        publisher.publish(event("runtime-a", 7)).unwrap();
        publisher.publish(event("runtime-a", 8)).unwrap();
        publisher.publish(event("runtime-a", 9)).unwrap();

        match subscriber.read_available(8).unwrap() {
            EventRead::ReplayRequired {
                requested_next,
                oldest_retained,
                newest_retained,
            } => {
                assert_eq!(requested_next.seq(), EventSeq::new(7).unwrap());
                assert_eq!(oldest_retained.seq(), EventSeq::new(8).unwrap());
                assert_eq!(newest_retained.seq(), EventSeq::new(9).unwrap());
            }
            other => panic!("expected replay-required, got {other:?}"),
        }
    }

    #[test]
    fn tail_subscriber_receives_only_future_events() {
        let (publisher, _) = semantic_event_channel(4).unwrap();
        for seq in 1..=3 {
            publisher.publish(event("runtime-a", seq)).unwrap();
        }
        let mut subscriber = publisher.subscribe(SubscribeFrom::Tail).unwrap();
        assert!(matches!(
            subscriber.read_available(8).unwrap(),
            EventRead::NoEvents
        ));
        publisher.publish(event("runtime-a", 4)).unwrap();
        assert_eq!(seqs(subscriber.read_available(8).unwrap()), vec![4]);
    }

    #[test]
    fn oldest_retained_subscriber_reads_in_order_and_respects_batch_limit() {
        let (publisher, _) = semantic_event_channel(8).unwrap();
        for seq in 1..=5 {
            publisher.publish(event("runtime-a", seq)).unwrap();
        }
        let mut subscriber = publisher
            .subscribe(SubscribeFrom::OldestRetained)
            .unwrap();
        assert_eq!(seqs(subscriber.read_available(2).unwrap()), vec![1, 2]);
        assert_eq!(seqs(subscriber.read_available(2).unwrap()), vec![3, 4]);
        assert_eq!(seqs(subscriber.read_available(2).unwrap()), vec![5]);
        assert!(matches!(
            subscriber.read_available(2).unwrap(),
            EventRead::NoEvents
        ));
    }

    #[test]
    fn lag_beyond_retention_requires_replay_instead_of_skipping() {
        let (publisher, _) = semantic_event_channel(3).unwrap();
        publisher.publish(event("runtime-a", 1)).unwrap();
        let mut subscriber = publisher
            .subscribe(SubscribeFrom::After(cursor("runtime-a", 1)))
            .unwrap();

        for seq in 2..=5 {
            publisher.publish(event("runtime-a", seq)).unwrap();
        }

        match subscriber.read_available(8).unwrap() {
            EventRead::ReplayRequired {
                requested_next,
                oldest_retained,
                newest_retained,
            } => {
                assert_eq!(requested_next.seq(), EventSeq::new(2).unwrap());
                assert_eq!(oldest_retained.seq(), EventSeq::new(3).unwrap());
                assert_eq!(newest_retained.seq(), EventSeq::new(5).unwrap());
            }
            other => panic!("expected replay-required, got {other:?}"),
        }
    }

    #[test]
    fn subscribers_are_independent_without_per_client_event_queues() {
        let (publisher, _) = semantic_event_channel(6).unwrap();
        let mut fast = publisher.subscribe(SubscribeFrom::Tail).unwrap();
        let mut slow = publisher.subscribe(SubscribeFrom::Tail).unwrap();

        for seq in 1..=3 {
            publisher.publish(event("runtime-a", seq)).unwrap();
            assert_eq!(seqs(fast.read_available(8).unwrap()), vec![seq]);
        }
        assert_eq!(seqs(slow.read_available(8).unwrap()), vec![1, 2, 3]);
    }

    #[test]
    fn resume_runtime_mismatch_fails_at_subscription() {
        let (publisher, _) = semantic_event_channel(4).unwrap();
        publisher.publish(event("runtime-a", 1)).unwrap();
        let error = publisher
            .subscribe(SubscribeFrom::After(cursor("runtime-b", 1)))
            .unwrap_err();
        assert_eq!(
            error,
            EventPlaneError::RuntimeChanged {
                expected: runtime("runtime-a"),
                found: runtime("runtime-b"),
            }
        );
    }
}
