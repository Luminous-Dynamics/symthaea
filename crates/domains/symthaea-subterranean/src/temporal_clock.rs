// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded clock-discipline and timestamp-trust contracts.
//!
//! This module does not synchronize clocks and does not authenticate clock
//! sources. It accepts externally bound source identities and independently
//! enforces epoch monotonicity, replay resistance, bounded future skew, bounded
//! age, uncertainty limits, and sequence-gap visibility.

use serde::{Deserialize, Serialize};

pub const TEMPORAL_CLOCK_SCHEMA_VERSION: u16 = 1;
pub const MAX_CLOCK_SOURCES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClockSourceId(pub u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockDomain {
    Control,
    Sensor,
    Operator,
    Peer,
    Evidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockSample {
    pub source: ClockSourceId,
    pub domain: ClockDomain,
    pub boot_epoch: u64,
    pub sequence: u64,
    pub event_time_ns: u64,
    pub uncertainty_ns: u64,
    pub received_step: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockPolicy {
    pub max_future_skew_ns: u64,
    pub max_age_ns: u64,
    pub max_uncertainty_ns: u64,
    pub max_sequence_gap: u64,
}

impl Default for ClockPolicy {
    fn default() -> Self {
        Self {
            max_future_skew_ns: 20_000_000,
            max_age_ns: 500_000_000,
            max_uncertainty_ns: 50_000_000,
            max_sequence_gap: 8,
        }
    }
}

impl ClockPolicy {
    pub fn validate(self) -> bool {
        self.max_future_skew_ns <= self.max_age_ns
            && self.max_uncertainty_ns <= self.max_age_ns
            && self.max_sequence_gap > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockDisposition {
    Accepted,
    Degraded,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockIssue {
    None,
    SourceCapacity,
    EpochRegression,
    SequenceReplay,
    SequenceGap,
    TimeRegression,
    FutureTimestamp,
    StaleTimestamp,
    ExcessiveUncertainty,
}

impl ClockIssue {
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SourceCapacity => "source_capacity",
            Self::EpochRegression => "epoch_regression",
            Self::SequenceReplay => "sequence_replay",
            Self::SequenceGap => "sequence_gap",
            Self::TimeRegression => "time_regression",
            Self::FutureTimestamp => "future_timestamp",
            Self::StaleTimestamp => "stale_timestamp",
            Self::ExcessiveUncertainty => "excessive_uncertainty",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockAssessment {
    pub disposition: ClockDisposition,
    pub issue: ClockIssue,
    pub age_ns: u64,
    pub sequence_gap: u64,
}

impl ClockAssessment {
    pub const fn nominal() -> Self {
        Self {
            disposition: ClockDisposition::Accepted,
            issue: ClockIssue::None,
            age_ns: 0,
            sequence_gap: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct SourceClockState {
    source: ClockSourceId,
    domain: ClockDomain,
    boot_epoch: u64,
    last_sequence: u64,
    last_event_time_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalClockSupervisor {
    schema_version: u16,
    policy: ClockPolicy,
    sources: Vec<SourceClockState>,
    accepted_samples: u64,
    degraded_samples: u64,
    rejected_samples: u64,
    last: ClockAssessment,
}

impl TemporalClockSupervisor {
    pub fn new(policy: ClockPolicy) -> Self {
        Self {
            schema_version: TEMPORAL_CLOCK_SCHEMA_VERSION,
            policy,
            sources: Vec::with_capacity(MAX_CLOCK_SOURCES),
            accepted_samples: 0,
            degraded_samples: 0,
            rejected_samples: 0,
            last: ClockAssessment::nominal(),
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == TEMPORAL_CLOCK_SCHEMA_VERSION
            && self.policy.validate()
            && self.sources.len() <= MAX_CLOCK_SOURCES
            && self.sources.iter().enumerate().all(|(index, state)| {
                self.sources[..index]
                    .iter()
                    .all(|other| other.source != state.source)
            })
    }

    pub fn observe(&mut self, control_time_ns: u64, sample: ClockSample) -> ClockAssessment {
        let source_index = self
            .sources
            .iter()
            .position(|state| state.source == sample.source);
        let age_ns = control_time_ns.saturating_sub(sample.event_time_ns);
        let future_ns = sample.event_time_ns.saturating_sub(control_time_ns);

        let assessment = if future_ns > self.policy.max_future_skew_ns {
            rejected(ClockIssue::FutureTimestamp, age_ns, 0)
        } else if age_ns > self.policy.max_age_ns {
            rejected(ClockIssue::StaleTimestamp, age_ns, 0)
        } else if sample.uncertainty_ns > self.policy.max_uncertainty_ns {
            rejected(ClockIssue::ExcessiveUncertainty, age_ns, 0)
        } else if let Some(index) = source_index {
            let previous = self.sources[index];
            if sample.domain != previous.domain {
                rejected(ClockIssue::EpochRegression, age_ns, 0)
            } else if sample.boot_epoch < previous.boot_epoch {
                rejected(ClockIssue::EpochRegression, age_ns, 0)
            } else if sample.boot_epoch == previous.boot_epoch
                && sample.sequence <= previous.last_sequence
            {
                rejected(ClockIssue::SequenceReplay, age_ns, 0)
            } else if sample.boot_epoch == previous.boot_epoch
                && sample.event_time_ns < previous.last_event_time_ns
            {
                rejected(ClockIssue::TimeRegression, age_ns, 0)
            } else {
                let gap = if sample.boot_epoch > previous.boot_epoch {
                    0
                } else {
                    sample
                        .sequence
                        .saturating_sub(previous.last_sequence)
                        .saturating_sub(1)
                };
                self.sources[index] = SourceClockState {
                    source: sample.source,
                    domain: sample.domain,
                    boot_epoch: sample.boot_epoch,
                    last_sequence: sample.sequence,
                    last_event_time_ns: sample.event_time_ns,
                };
                if gap > self.policy.max_sequence_gap {
                    degraded(ClockIssue::SequenceGap, age_ns, gap)
                } else {
                    ClockAssessment {
                        disposition: ClockDisposition::Accepted,
                        issue: ClockIssue::None,
                        age_ns,
                        sequence_gap: gap,
                    }
                }
            }
        } else if self.sources.len() >= MAX_CLOCK_SOURCES {
            rejected(ClockIssue::SourceCapacity, age_ns, 0)
        } else {
            self.sources.push(SourceClockState {
                source: sample.source,
                domain: sample.domain,
                boot_epoch: sample.boot_epoch,
                last_sequence: sample.sequence,
                last_event_time_ns: sample.event_time_ns,
            });
            ClockAssessment {
                disposition: ClockDisposition::Accepted,
                issue: ClockIssue::None,
                age_ns,
                sequence_gap: 0,
            }
        };

        match assessment.disposition {
            ClockDisposition::Accepted => {
                self.accepted_samples = self.accepted_samples.saturating_add(1)
            }
            ClockDisposition::Degraded => {
                self.degraded_samples = self.degraded_samples.saturating_add(1)
            }
            ClockDisposition::Rejected => {
                self.rejected_samples = self.rejected_samples.saturating_add(1)
            }
        }
        self.last = assessment;
        assessment
    }

    pub const fn last(&self) -> ClockAssessment {
        self.last
    }

    pub const fn accepted_samples(&self) -> u64 {
        self.accepted_samples
    }

    pub const fn degraded_samples(&self) -> u64 {
        self.degraded_samples
    }

    pub const fn rejected_samples(&self) -> u64 {
        self.rejected_samples
    }
}

impl Default for TemporalClockSupervisor {
    fn default() -> Self {
        Self::new(ClockPolicy::default())
    }
}

const fn rejected(issue: ClockIssue, age_ns: u64, sequence_gap: u64) -> ClockAssessment {
    ClockAssessment {
        disposition: ClockDisposition::Rejected,
        issue,
        age_ns,
        sequence_gap,
    }
}

const fn degraded(issue: ClockIssue, age_ns: u64, sequence_gap: u64) -> ClockAssessment {
    ClockAssessment {
        disposition: ClockDisposition::Degraded,
        issue,
        age_ns,
        sequence_gap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(sequence: u64, time: u64) -> ClockSample {
        ClockSample {
            source: ClockSourceId(1),
            domain: ClockDomain::Sensor,
            boot_epoch: 1,
            sequence,
            event_time_ns: time,
            uncertainty_ns: 1_000,
            received_step: sequence,
        }
    }

    #[test]
    fn replay_and_time_regression_are_rejected() {
        let mut clock = TemporalClockSupervisor::default();
        assert_eq!(
            clock.observe(100_000_000, sample(1, 100_000_000)).disposition,
            ClockDisposition::Accepted
        );
        assert_eq!(
            clock.observe(110_000_000, sample(1, 110_000_000)).issue,
            ClockIssue::SequenceReplay
        );
        assert_eq!(
            clock.observe(120_000_000, sample(2, 90_000_000)).issue,
            ClockIssue::TimeRegression
        );
    }

    #[test]
    fn large_sequence_gap_is_visible_but_not_replayed() {
        let mut clock = TemporalClockSupervisor::default();
        clock.observe(100, sample(1, 100));
        let assessment = clock.observe(200, sample(20, 200));
        assert_eq!(assessment.disposition, ClockDisposition::Degraded);
        assert_eq!(assessment.issue, ClockIssue::SequenceGap);
        assert_eq!(assessment.sequence_gap, 18);
    }

    #[test]
    fn new_epoch_may_restart_sequence_without_regressing_epoch() {
        let mut clock = TemporalClockSupervisor::default();
        clock.observe(100, sample(10, 100));
        let mut reboot = sample(1, 110);
        reboot.boot_epoch = 2;
        assert_eq!(
            clock.observe(110, reboot).disposition,
            ClockDisposition::Accepted
        );
        reboot.boot_epoch = 1;
        reboot.sequence = 11;
        assert_eq!(
            clock.observe(120, reboot).issue,
            ClockIssue::EpochRegression
        );
    }
}
