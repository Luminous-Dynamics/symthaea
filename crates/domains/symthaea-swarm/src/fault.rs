//! Deterministic network-fault scheduling for integration witnesses.
//!
//! This module does not intercept Iroh internally. Adapters place encoded
//! packets into [`DeterministicFaultInjector`] before a send or after a receive,
//! then drain packets at simulation ticks. The same seed and profile produce the
//! same loss, delay, duplication, reordering, and corruption schedule.

use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaultProfile {
    pub fixed_delay_ticks: u64,
    pub jitter_ticks: u64,
    pub drop_every: Option<u64>,
    pub duplicate_every: Option<u64>,
    pub corrupt_every: Option<u64>,
    pub maximum_queued_packets: usize,
    pub maximum_queued_bytes: usize,
}

impl FaultProfile {
    pub const CLEAN: Self = Self {
        fixed_delay_ticks: 0,
        jitter_ticks: 0,
        drop_every: None,
        duplicate_every: None,
        corrupt_every: None,
        maximum_queued_packets: 65_536,
        maximum_queued_bytes: 64 * 1024 * 1024,
    };

    pub const HOSTILE_DATAGRAMS: Self = Self {
        fixed_delay_ticks: 2,
        jitter_ticks: 6,
        drop_every: Some(7),
        duplicate_every: Some(11),
        corrupt_every: Some(29),
        maximum_queued_packets: 65_536,
        maximum_queued_bytes: 64 * 1024 * 1024,
    };

    pub fn validate(self) -> Result<Self, FaultConfigError> {
        if self.maximum_queued_packets == 0 {
            return Err(FaultConfigError::ZeroPacketCapacity);
        }
        if self.maximum_queued_bytes == 0 {
            return Err(FaultConfigError::ZeroByteCapacity);
        }
        for (name, interval) in [
            ("drop_every", self.drop_every),
            ("duplicate_every", self.duplicate_every),
            ("corrupt_every", self.corrupt_every),
        ] {
            if interval == Some(0) {
                return Err(FaultConfigError::ZeroInterval { name });
            }
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FaultMetrics {
    pub submitted: u64,
    pub delivered: u64,
    pub dropped: u64,
    pub duplicated: u64,
    pub corrupted: u64,
    pub queue_overflow: u64,
    pub queued_packets: usize,
    pub queued_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledPacket {
    pub ordinal: u64,
    pub deliver_at_tick: u64,
    pub duplicate: bool,
    pub corrupted: bool,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionOutcome {
    Scheduled {
        copies: usize,
        first_delivery_tick: u64,
    },
    Dropped,
}

#[derive(Debug)]
pub struct DeterministicFaultInjector {
    profile: FaultProfile,
    seed: u64,
    ordinal: u64,
    queue: BTreeMap<u64, VecDeque<ScheduledPacket>>,
    metrics: FaultMetrics,
}

impl DeterministicFaultInjector {
    pub fn new(profile: FaultProfile, seed: u64) -> Result<Self, FaultConfigError> {
        Ok(Self {
            profile: profile.validate()?,
            seed,
            ordinal: 0,
            queue: BTreeMap::new(),
            metrics: FaultMetrics::default(),
        })
    }

    pub fn profile(&self) -> FaultProfile {
        self.profile
    }

    pub fn metrics(&self) -> FaultMetrics {
        self.metrics
    }

    pub fn submit(
        &mut self,
        now_tick: u64,
        bytes: Vec<u8>,
    ) -> Result<SubmissionOutcome, FaultQueueError> {
        self.ordinal = self
            .ordinal
            .checked_add(1)
            .ok_or(FaultQueueError::OrdinalExhausted)?;
        let ordinal = self.ordinal;
        self.metrics.submitted = self.metrics.submitted.saturating_add(1);
        if interval_matches(self.profile.drop_every, ordinal) {
            self.metrics.dropped = self.metrics.dropped.saturating_add(1);
            return Ok(SubmissionOutcome::Dropped);
        }

        let duplicated = interval_matches(self.profile.duplicate_every, ordinal);
        let copies = if duplicated { 2 } else { 1 };
        let corrupted = !bytes.is_empty() && interval_matches(self.profile.corrupt_every, ordinal);
        let total_bytes = bytes.len().saturating_mul(copies);
        if self.metrics.queued_packets.saturating_add(copies) > self.profile.maximum_queued_packets
            || self.metrics.queued_bytes.saturating_add(total_bytes)
                > self.profile.maximum_queued_bytes
        {
            self.metrics.queue_overflow = self.metrics.queue_overflow.saturating_add(1);
            return Err(FaultQueueError::CapacityExceeded {
                queued_packets: self.metrics.queued_packets,
                queued_bytes: self.metrics.queued_bytes,
                incoming_packets: copies,
                incoming_bytes: total_bytes,
            });
        }

        let mut first_delivery_tick = u64::MAX;
        for copy in 0..copies {
            let deliver_at_tick = now_tick.saturating_add(self.delay_for(ordinal, copy));
            first_delivery_tick = first_delivery_tick.min(deliver_at_tick);
            let mut packet_bytes = bytes.clone();
            if corrupted {
                let index = mixed_index(self.seed, ordinal, copy, packet_bytes.len());
                packet_bytes[index] ^= 0x80;
            }
            self.queue
                .entry(deliver_at_tick)
                .or_default()
                .push_back(ScheduledPacket {
                    ordinal,
                    deliver_at_tick,
                    duplicate: copy > 0,
                    corrupted,
                    bytes: packet_bytes,
                });
        }
        if duplicated {
            self.metrics.duplicated = self.metrics.duplicated.saturating_add(1);
        }
        if corrupted {
            self.metrics.corrupted = self.metrics.corrupted.saturating_add(copies as u64);
        }
        self.metrics.queued_packets += copies;
        self.metrics.queued_bytes += total_bytes;
        Ok(SubmissionOutcome::Scheduled {
            copies,
            first_delivery_tick,
        })
    }

    /// Drain all packets due at or before `current_tick` in deterministic order.
    pub fn drain_ready(&mut self, current_tick: u64) -> Vec<ScheduledPacket> {
        let ready_ticks = self
            .queue
            .range(..=current_tick)
            .map(|(tick, _)| *tick)
            .collect::<Vec<_>>();
        let mut ready = Vec::new();
        for tick in ready_ticks {
            if let Some(mut packets) = self.queue.remove(&tick) {
                while let Some(packet) = packets.pop_front() {
                    self.metrics.queued_packets = self.metrics.queued_packets.saturating_sub(1);
                    self.metrics.queued_bytes =
                        self.metrics.queued_bytes.saturating_sub(packet.bytes.len());
                    self.metrics.delivered = self.metrics.delivered.saturating_add(1);
                    ready.push(packet);
                }
            }
        }
        ready
    }

    pub fn clear(&mut self) {
        self.queue.clear();
        self.metrics.queued_packets = 0;
        self.metrics.queued_bytes = 0;
    }

    fn delay_for(&self, ordinal: u64, copy: usize) -> u64 {
        if self.profile.jitter_ticks == 0 {
            return self.profile.fixed_delay_ticks.saturating_add(copy as u64);
        }
        let mixed = splitmix64(
            self.seed ^ ordinal.rotate_left(17) ^ (copy as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        self.profile
            .fixed_delay_ticks
            .saturating_add(mixed % self.profile.jitter_ticks.saturating_add(1))
    }
}

fn interval_matches(interval: Option<u64>, ordinal: u64) -> bool {
    interval.is_some_and(|value| ordinal % value == 0)
}

fn mixed_index(seed: u64, ordinal: u64, copy: usize, len: usize) -> usize {
    let mixed = splitmix64(seed ^ ordinal ^ (copy as u64).rotate_left(31));
    (mixed as usize) % len
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum FaultConfigError {
    #[error("fault injector packet capacity must be greater than zero")]
    ZeroPacketCapacity,
    #[error("fault injector byte capacity must be greater than zero")]
    ZeroByteCapacity,
    #[error("fault interval {name} must not be zero")]
    ZeroInterval { name: &'static str },
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum FaultQueueError {
    #[error("fault injector packet ordinal is exhausted")]
    OrdinalExhausted,
    #[error(
        "fault queue capacity exceeded: queued {queued_packets} packets/{queued_bytes} bytes; incoming {incoming_packets} packets/{incoming_bytes} bytes"
    )]
    CapacityExceeded {
        queued_packets: usize,
        queued_bytes: usize,
        incoming_packets: usize,
        incoming_bytes: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_seed_and_profile_produce_identical_schedule() {
        let profile = FaultProfile::HOSTILE_DATAGRAMS;
        let mut left = DeterministicFaultInjector::new(profile, 7).unwrap();
        let mut right = DeterministicFaultInjector::new(profile, 7).unwrap();
        for value in 0..64u8 {
            assert_eq!(
                left.submit(10, vec![value; 8]).unwrap(),
                right.submit(10, vec![value; 8]).unwrap()
            );
        }
        assert_eq!(left.drain_ready(100), right.drain_ready(100));
        assert_eq!(left.metrics(), right.metrics());
    }

    #[test]
    fn queue_capacity_fails_without_partial_scheduling() {
        let profile = FaultProfile {
            maximum_queued_packets: 1,
            maximum_queued_bytes: 4,
            ..FaultProfile::CLEAN
        };
        let mut injector = DeterministicFaultInjector::new(profile, 1).unwrap();
        injector.submit(0, vec![1, 2, 3, 4]).unwrap();
        assert!(matches!(
            injector.submit(0, vec![5]),
            Err(FaultQueueError::CapacityExceeded { .. })
        ));
        assert_eq!(injector.metrics().queued_packets, 1);
        assert_eq!(injector.drain_ready(0).len(), 1);
    }

    #[test]
    fn hostile_profile_exercises_loss_duplication_and_corruption() {
        let mut injector =
            DeterministicFaultInjector::new(FaultProfile::HOSTILE_DATAGRAMS, 99).unwrap();
        for _ in 0..64 {
            let _ = injector.submit(0, vec![0; 16]);
        }
        let metrics = injector.metrics();
        assert!(metrics.dropped > 0);
        assert!(metrics.duplicated > 0);
        assert!(metrics.corrupted > 0);
    }
    #[test]
    fn submission_receipt_reports_earliest_copy_and_empty_payload_is_not_corrupted() {
        let profile = FaultProfile {
            jitter_ticks: 10,
            duplicate_every: Some(1),
            corrupt_every: Some(1),
            ..FaultProfile::CLEAN
        };
        let mut injector = DeterministicFaultInjector::new(profile, 123).unwrap();
        let outcome = injector.submit(5, Vec::new()).unwrap();
        let ready = injector.drain_ready(u64::MAX);
        let earliest = ready
            .iter()
            .map(|packet| packet.deliver_at_tick)
            .min()
            .unwrap();
        assert_eq!(
            outcome,
            SubmissionOutcome::Scheduled {
                copies: 2,
                first_delivery_tick: earliest,
            }
        );
        assert_eq!(injector.metrics().corrupted, 0);
        assert!(ready.iter().all(|packet| !packet.corrupted));
    }
}
