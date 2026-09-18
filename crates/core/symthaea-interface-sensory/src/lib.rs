// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded latency-first sensory transport for Symthaea interfaces.
//!
//! Sensory traffic differs from both state and semantic events. It must stay
//! bounded under slow consumers, may intentionally prefer freshness over history,
//! and must expose overload rather than silently accumulating latency.
//!
//! The lane below is single-consumer with cloneable producers. Both publication
//! and consumption use `Mutex::try_lock`, so the latency-sensitive API never waits
//! for lane ownership. This is still **not** a hard-real-time device-callback
//! primitive: audio callbacks should keep using lock-free/ring-buffer transport and
//! forward frames here from a worker thread.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};

/// Overflow behavior for a full sensory lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensoryOverflowPolicy {
    /// Evict the oldest retained frame and admit the new frame.
    DropOldest,
    /// Preserve the admitted window and reject the incoming frame.
    DropNewest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensoryPlaneError {
    ZeroCapacity,
    Poisoned,
}

impl fmt::Display for SensoryPlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "sensory lane capacity must be non-zero"),
            Self::Poisoned => write!(f, "sensory lane mutex poisoned"),
        }
    }
}

impl std::error::Error for SensoryPlaneError {}

/// Result of a non-blocking publication attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishOutcome {
    Enqueued { depth: usize },
    DroppedOldest { depth: usize },
    DroppedNewest { depth: usize },
    /// Lane ownership was contended; the producer did not wait.
    Contended,
}

#[derive(Debug, Clone)]
pub enum ReadOutcome<T> {
    Item(Arc<T>),
    Empty,
    Contended,
}

#[derive(Debug, Clone)]
pub struct LatestSensorySample<T> {
    pub value: Arc<T>,
    /// Older queued frames intentionally discarded by this newest-only read.
    pub coalesced_before: usize,
}

#[derive(Debug, Clone)]
pub enum LatestRead<T> {
    Item(LatestSensorySample<T>),
    Empty,
    Contended,
}

/// Transport diagnostics only; none of these counters establish semantic order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SensoryStatsSnapshot {
    pub accepted: u64,
    pub dropped_oldest: u64,
    pub dropped_newest: u64,
    pub producer_contentions: u64,
    pub consumer_contentions: u64,
    pub consumed: u64,
    pub consumer_coalesced: u64,
    pub high_watermark: u64,
}

#[derive(Debug, Default)]
struct SensoryStats {
    accepted: AtomicU64,
    dropped_oldest: AtomicU64,
    dropped_newest: AtomicU64,
    producer_contentions: AtomicU64,
    consumer_contentions: AtomicU64,
    consumed: AtomicU64,
    consumer_coalesced: AtomicU64,
    high_watermark: AtomicU64,
}

impl SensoryStats {
    fn record_high_watermark(&self, depth: usize) {
        let depth = depth as u64;
        let mut observed = self.high_watermark.load(Ordering::Relaxed);
        while depth > observed {
            match self.high_watermark.compare_exchange_weak(
                observed,
                depth,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => observed = current,
            }
        }
    }

    fn snapshot(&self) -> SensoryStatsSnapshot {
        SensoryStatsSnapshot {
            accepted: self.accepted.load(Ordering::Relaxed),
            dropped_oldest: self.dropped_oldest.load(Ordering::Relaxed),
            dropped_newest: self.dropped_newest.load(Ordering::Relaxed),
            producer_contentions: self.producer_contentions.load(Ordering::Relaxed),
            consumer_contentions: self.consumer_contentions.load(Ordering::Relaxed),
            consumed: self.consumed.load(Ordering::Relaxed),
            consumer_coalesced: self.consumer_coalesced.load(Ordering::Relaxed),
            high_watermark: self.high_watermark.load(Ordering::Relaxed),
        }
    }
}

struct SensoryLane<T> {
    capacity: usize,
    policy: SensoryOverflowPolicy,
    queue: Mutex<VecDeque<Arc<T>>>,
    stats: SensoryStats,
}

/// Cloneable producer for one finite sensory lane.
pub struct SensoryPublisher<T> {
    shared: Arc<SensoryLane<T>>,
}

impl<T> Clone for SensoryPublisher<T> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<T> SensoryPublisher<T> {
    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }

    pub fn overflow_policy(&self) -> SensoryOverflowPolicy {
        self.shared.policy
    }

    /// Attempt to publish without blocking on lane ownership.
    pub fn try_publish(&self, value: T) -> Result<PublishOutcome, SensoryPlaneError> {
        let mut queue = match self.shared.queue.try_lock() {
            Ok(queue) => queue,
            Err(TryLockError::WouldBlock) => {
                self.shared
                    .stats
                    .producer_contentions
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(PublishOutcome::Contended);
            }
            Err(TryLockError::Poisoned(_)) => return Err(SensoryPlaneError::Poisoned),
        };

        if queue.len() >= self.shared.capacity {
            match self.shared.policy {
                SensoryOverflowPolicy::DropOldest => {
                    let _ = queue.pop_front();
                    self.shared
                        .stats
                        .dropped_oldest
                        .fetch_add(1, Ordering::Relaxed);
                    queue.push_back(Arc::new(value));
                    let depth = queue.len();
                    self.shared.stats.accepted.fetch_add(1, Ordering::Relaxed);
                    self.shared.stats.record_high_watermark(depth);
                    Ok(PublishOutcome::DroppedOldest { depth })
                }
                SensoryOverflowPolicy::DropNewest => {
                    self.shared
                        .stats
                        .dropped_newest
                        .fetch_add(1, Ordering::Relaxed);
                    Ok(PublishOutcome::DroppedNewest { depth: queue.len() })
                }
            }
        } else {
            queue.push_back(Arc::new(value));
            let depth = queue.len();
            self.shared.stats.accepted.fetch_add(1, Ordering::Relaxed);
            self.shared.stats.record_high_watermark(depth);
            Ok(PublishOutcome::Enqueued { depth })
        }
    }

    pub fn stats(&self) -> SensoryStatsSnapshot {
        self.shared.stats.snapshot()
    }
}

/// Single consumer for one finite sensory lane.
///
/// The receiver is deliberately not `Clone`: fan-out should create independent
/// lanes with explicit client/modality policy instead of making consumers race to
/// remove frames from one shared FIFO.
pub struct SensoryReceiver<T> {
    shared: Arc<SensoryLane<T>>,
}

impl<T> SensoryReceiver<T> {
    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }

    pub fn overflow_policy(&self) -> SensoryOverflowPolicy {
        self.shared.policy
    }

    pub fn try_pop(&mut self) -> Result<ReadOutcome<T>, SensoryPlaneError> {
        let mut queue = match self.shared.queue.try_lock() {
            Ok(queue) => queue,
            Err(TryLockError::WouldBlock) => {
                self.shared
                    .stats
                    .consumer_contentions
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(ReadOutcome::Contended);
            }
            Err(TryLockError::Poisoned(_)) => return Err(SensoryPlaneError::Poisoned),
        };

        match queue.pop_front() {
            Some(value) => {
                self.shared.stats.consumed.fetch_add(1, Ordering::Relaxed);
                Ok(ReadOutcome::Item(value))
            }
            None => Ok(ReadOutcome::Empty),
        }
    }

    /// Consume only the newest retained frame and explicitly coalesce older ones.
    ///
    /// Visual/HDC clients can call this once per render/update so producer bursts
    /// cannot turn into arbitrary per-frame backlog work.
    pub fn try_take_latest(&mut self) -> Result<LatestRead<T>, SensoryPlaneError> {
        let mut queue = match self.shared.queue.try_lock() {
            Ok(queue) => queue,
            Err(TryLockError::WouldBlock) => {
                self.shared
                    .stats
                    .consumer_contentions
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(LatestRead::Contended);
            }
            Err(TryLockError::Poisoned(_)) => return Err(SensoryPlaneError::Poisoned),
        };

        let Some(value) = queue.pop_back() else {
            return Ok(LatestRead::Empty);
        };
        let coalesced_before = queue.len();
        queue.clear();
        self.shared.stats.consumed.fetch_add(1, Ordering::Relaxed);
        self.shared
            .stats
            .consumer_coalesced
            .fetch_add(coalesced_before as u64, Ordering::Relaxed);

        Ok(LatestRead::Item(LatestSensorySample {
            value,
            coalesced_before,
        }))
    }

    /// Return queue depth only when lane ownership is immediately available.
    pub fn try_depth(&self) -> Result<Option<usize>, SensoryPlaneError> {
        match self.shared.queue.try_lock() {
            Ok(queue) => Ok(Some(queue.len())),
            Err(TryLockError::WouldBlock) => {
                self.shared
                    .stats
                    .consumer_contentions
                    .fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
            Err(TryLockError::Poisoned(_)) => Err(SensoryPlaneError::Poisoned),
        }
    }

    pub fn stats(&self) -> SensoryStatsSnapshot {
        self.shared.stats.snapshot()
    }
}

pub fn sensory_channel<T>(
    capacity: usize,
    policy: SensoryOverflowPolicy,
) -> Result<(SensoryPublisher<T>, SensoryReceiver<T>), SensoryPlaneError> {
    if capacity == 0 {
        return Err(SensoryPlaneError::ZeroCapacity);
    }

    let shared = Arc::new(SensoryLane {
        capacity,
        policy,
        queue: Mutex::new(VecDeque::new()),
        stats: SensoryStats::default(),
    });

    Ok((
        SensoryPublisher {
            shared: Arc::clone(&shared),
        },
        SensoryReceiver { shared },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item<T>(read: ReadOutcome<T>) -> Arc<T> {
        match read {
            ReadOutcome::Item(value) => value,
            ReadOutcome::Empty => panic!("expected item, got empty"),
            ReadOutcome::Contended => panic!("expected item, got contention"),
        }
    }

    #[test]
    fn zero_capacity_fails_closed() {
        let result = sensory_channel::<u8>(0, SensoryOverflowPolicy::DropOldest);
        assert!(matches!(result, Err(SensoryPlaneError::ZeroCapacity)));
    }

    #[test]
    fn drop_oldest_keeps_freshest_bounded_window() {
        let (publisher, mut receiver) =
            sensory_channel(3, SensoryOverflowPolicy::DropOldest).unwrap();

        for value in 1_u8..=5 {
            publisher.try_publish(value).unwrap();
        }

        assert_eq!(*item(receiver.try_pop().unwrap()), 3);
        assert_eq!(*item(receiver.try_pop().unwrap()), 4);
        assert_eq!(*item(receiver.try_pop().unwrap()), 5);
        assert!(matches!(receiver.try_pop().unwrap(), ReadOutcome::Empty));

        let stats = receiver.stats();
        assert_eq!(stats.accepted, 5);
        assert_eq!(stats.dropped_oldest, 2);
        assert_eq!(stats.dropped_newest, 0);
        assert_eq!(stats.high_watermark, 3);
    }

    #[test]
    fn drop_newest_preserves_admitted_window() {
        let (publisher, mut receiver) =
            sensory_channel(2, SensoryOverflowPolicy::DropNewest).unwrap();

        assert_eq!(
            publisher.try_publish(1_u8).unwrap(),
            PublishOutcome::Enqueued { depth: 1 }
        );
        assert_eq!(
            publisher.try_publish(2_u8).unwrap(),
            PublishOutcome::Enqueued { depth: 2 }
        );
        assert_eq!(
            publisher.try_publish(3_u8).unwrap(),
            PublishOutcome::DroppedNewest { depth: 2 }
        );

        assert_eq!(*item(receiver.try_pop().unwrap()), 1);
        assert_eq!(*item(receiver.try_pop().unwrap()), 2);
        let stats = receiver.stats();
        assert_eq!(stats.accepted, 2);
        assert_eq!(stats.dropped_newest, 1);
    }

    #[test]
    fn latest_read_bounds_render_work_and_counts_coalescing() {
        let (publisher, mut receiver) =
            sensory_channel(16, SensoryOverflowPolicy::DropOldest).unwrap();

        for value in 1_u64..=10 {
            publisher.try_publish(value).unwrap();
        }

        let latest = match receiver.try_take_latest().unwrap() {
            LatestRead::Item(sample) => sample,
            LatestRead::Empty => panic!("expected newest frame"),
            LatestRead::Contended => panic!("unexpected contention"),
        };
        assert_eq!(*latest.value, 10);
        assert_eq!(latest.coalesced_before, 9);
        assert_eq!(receiver.try_depth().unwrap(), Some(0));

        let stats = receiver.stats();
        assert_eq!(stats.consumed, 1);
        assert_eq!(stats.consumer_coalesced, 9);
    }

    #[test]
    fn ten_thousand_publications_remain_capacity_bounded() {
        let (publisher, receiver) =
            sensory_channel(8, SensoryOverflowPolicy::DropOldest).unwrap();

        for value in 0_u64..10_000 {
            publisher.try_publish(value).unwrap();
        }

        assert_eq!(receiver.try_depth().unwrap(), Some(8));
        let stats = receiver.stats();
        assert_eq!(stats.accepted, 10_000);
        assert_eq!(stats.dropped_oldest, 9_992);
        assert_eq!(stats.high_watermark, 8);
    }

    #[test]
    fn publisher_clones_share_one_bounded_lane() {
        let (publisher, mut receiver) =
            sensory_channel(2, SensoryOverflowPolicy::DropOldest).unwrap();
        let other = publisher.clone();

        publisher.try_publish("a").unwrap();
        other.try_publish("b").unwrap();
        other.try_publish("c").unwrap();

        assert_eq!(*item(receiver.try_pop().unwrap()), "b");
        assert_eq!(*item(receiver.try_pop().unwrap()), "c");
        assert_eq!(publisher.stats().dropped_oldest, 1);
    }
}
