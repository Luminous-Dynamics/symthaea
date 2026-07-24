// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded redundant-sensor fusion with replay resistance and critical quorum.
//!
//! This module does not authenticate transport identities. Callers must bind a
//! `SensorSourceId` to an authenticated physical sensor path before ingestion.
//! The crate independently enforces source bounds, monotonic sequences,
//! physically valid values, disagreement penalties, and fail-closed quorum.

use crate::observation_quality::CRITICAL_SENSOR_CHANNELS;
use crate::types::{NUM_STATE_CHANNELS, STATE_CHANNEL_RANGES, SubterraneanState};
use serde::{Deserialize, Serialize};

pub const MAX_SENSOR_SOURCES: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorSourceId(pub u8);

impl SensorSourceId {
    pub const fn index(self) -> Option<usize> {
        if (self.0 as usize) < MAX_SENSOR_SOURCES {
            Some(self.0 as usize)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensorSourceObservation {
    pub source: SensorSourceId,
    pub sequence: u64,
    pub channels: [f64; NUM_STATE_CHANNELS],
    pub valid: [bool; NUM_STATE_CHANNELS],
}

impl SensorSourceObservation {
    pub fn from_state(source: SensorSourceId, sequence: u64, state: &SubterraneanState) -> Self {
        Self {
            source,
            sequence,
            channels: state.channels,
            valid: [true; NUM_STATE_CHANNELS],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RedundantSensorFrame {
    pub observations: Vec<SensorSourceObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorFusionPolicy {
    pub disagreement_threshold: f64,
    pub source_isolation_threshold: f64,
    pub reliability_recovery_rate: f64,
    pub reliability_penalty_rate: f64,
}

impl Default for SensorFusionPolicy {
    fn default() -> Self {
        Self {
            disagreement_threshold: 0.08,
            source_isolation_threshold: 0.2,
            reliability_recovery_rate: 0.02,
            reliability_penalty_rate: 0.15,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorFusionReport {
    pub declared_sources: usize,
    pub accepted_sources: usize,
    pub rejected_sources: usize,
    pub replay_rejections: usize,
    pub isolated_sources: usize,
    pub channels_without_quorum: usize,
    pub critical_channels_without_quorum: usize,
    pub maximum_normalized_disagreement: f64,
    pub minimum_source_reliability: f64,
}

impl SensorFusionReport {
    pub const fn nominal() -> Self {
        Self {
            declared_sources: 1,
            accepted_sources: 1,
            rejected_sources: 0,
            replay_rejections: 0,
            isolated_sources: 0,
            channels_without_quorum: 0,
            critical_channels_without_quorum: 0,
            maximum_normalized_disagreement: 0.0,
            minimum_source_reliability: 1.0,
        }
    }

    pub const fn requires_fail_closed(self) -> bool {
        self.accepted_sources == 0 || self.critical_channels_without_quorum > 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorFusionSupervisor {
    policy: SensorFusionPolicy,
    reliability: [f64; MAX_SENSOR_SOURCES],
    last_sequence: [u64; MAX_SENSOR_SOURCES],
    seen: [bool; MAX_SENSOR_SOURCES],
    last_report: SensorFusionReport,
}

impl SensorFusionSupervisor {
    pub fn new(policy: SensorFusionPolicy) -> Self {
        Self {
            policy,
            reliability: [1.0; MAX_SENSOR_SOURCES],
            last_sequence: [0; MAX_SENSOR_SOURCES],
            seen: [false; MAX_SENSOR_SOURCES],
            last_report: SensorFusionReport::nominal(),
        }
    }

    pub fn validate(&self) -> bool {
        self.policy.disagreement_threshold.is_finite()
            && self.policy.disagreement_threshold > 0.0
            && self.policy.source_isolation_threshold.is_finite()
            && (0.0..=1.0).contains(&self.policy.source_isolation_threshold)
            && self.policy.reliability_recovery_rate.is_finite()
            && (0.0..=1.0).contains(&self.policy.reliability_recovery_rate)
            && self.policy.reliability_penalty_rate.is_finite()
            && (0.0..=1.0).contains(&self.policy.reliability_penalty_rate)
            && self
                .reliability
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    }

    pub fn reliability(&self, source: SensorSourceId) -> Option<f64> {
        source.index().map(|index| self.reliability[index])
    }

    pub fn report(&self) -> SensorFusionReport {
        self.last_report
    }

    pub fn fuse_local_state(
        &mut self,
        state: &SubterraneanState,
    ) -> (SubterraneanState, SensorFusionReport) {
        let sequence = if self.seen[0] {
            self.last_sequence[0].saturating_add(1)
        } else {
            1
        };
        self.fuse(
            &RedundantSensorFrame {
                observations: vec![SensorSourceObservation::from_state(
                    SensorSourceId(0),
                    sequence,
                    state,
                )],
            },
            state,
        )
    }

    pub fn fuse(
        &mut self,
        frame: &RedundantSensorFrame,
        fallback: &SubterraneanState,
    ) -> (SubterraneanState, SensorFusionReport) {
        let declared_sources = frame.observations.len().min(MAX_SENSOR_SOURCES);
        let mut accepted: Vec<&SensorSourceObservation> = Vec::with_capacity(MAX_SENSOR_SOURCES);
        let mut rejected_sources = frame.observations.len().saturating_sub(MAX_SENSOR_SOURCES);
        let mut replay_rejections = 0usize;
        let mut source_present = [false; MAX_SENSOR_SOURCES];

        for observation in frame.observations.iter().take(MAX_SENSOR_SOURCES) {
            let Some(index) = observation.source.index() else {
                rejected_sources = rejected_sources.saturating_add(1);
                continue;
            };
            if source_present[index]
                || (self.seen[index] && observation.sequence <= self.last_sequence[index])
            {
                rejected_sources = rejected_sources.saturating_add(1);
                replay_rejections = replay_rejections.saturating_add(1);
                continue;
            }
            source_present[index] = true;
            self.seen[index] = true;
            self.last_sequence[index] = observation.sequence;
            if self.reliability[index] <= self.policy.source_isolation_threshold {
                rejected_sources = rejected_sources.saturating_add(1);
                continue;
            }
            accepted.push(observation);
        }

        let quorum_required = declared_sources >= 2;
        let mut fused = fallback.clone();
        let mut channels_without_quorum = 0usize;
        let mut critical_channels_without_quorum = 0usize;
        let mut maximum_normalized_disagreement = 0.0f64;
        let mut per_source_penalty = [0.0f64; MAX_SENSOR_SOURCES];
        let mut per_source_samples = [0usize; MAX_SENSOR_SOURCES];

        for channel in 0..NUM_STATE_CHANNELS {
            let (minimum, maximum) = STATE_CHANNEL_RANGES[channel];
            let span = (maximum - minimum).max(f64::EPSILON);
            let mut values: Vec<(usize, f64)> = accepted
                .iter()
                .filter_map(|observation| {
                    let index = observation.source.index()?;
                    let value = observation.channels[channel];
                    (observation.valid[channel]
                        && value.is_finite()
                        && (minimum..=maximum).contains(&value))
                    .then_some((index, value))
                })
                .collect();
            values.sort_by(|left, right| left.1.total_cmp(&right.1));
            let required = if quorum_required { 2 } else { 1 };
            if values.len() < required {
                channels_without_quorum = channels_without_quorum.saturating_add(1);
                if CRITICAL_SENSOR_CHANNELS.contains(&channel) {
                    critical_channels_without_quorum =
                        critical_channels_without_quorum.saturating_add(1);
                }
                continue;
            }

            let median = if values.len() % 2 == 1 {
                values[values.len() / 2].1
            } else {
                let upper = values.len() / 2;
                (values[upper - 1].1 + values[upper].1) * 0.5
            };
            fused.channels[channel] = median;
            let minimum_value = values.first().map_or(median, |value| value.1);
            let maximum_value = values.last().map_or(median, |value| value.1);
            let disagreement = ((maximum_value - minimum_value) / span).clamp(0.0, 1.0);
            maximum_normalized_disagreement = maximum_normalized_disagreement.max(disagreement);
            for (source, value) in values {
                per_source_penalty[source] += ((value - median).abs() / span).clamp(0.0, 1.0);
                per_source_samples[source] = per_source_samples[source].saturating_add(1);
            }
        }

        for source in 0..MAX_SENSOR_SOURCES {
            if !source_present[source] {
                continue;
            }
            let average_penalty = if per_source_samples[source] == 0 {
                1.0
            } else {
                per_source_penalty[source] / per_source_samples[source] as f64
            };
            if average_penalty > self.policy.disagreement_threshold {
                self.reliability[source] = (self.reliability[source]
                    - self.policy.reliability_penalty_rate * average_penalty)
                    .clamp(0.0, 1.0);
            } else {
                self.reliability[source] = (self.reliability[source]
                    + self.policy.reliability_recovery_rate)
                    .clamp(0.0, 1.0);
            }
        }

        let isolated_sources = self
            .reliability
            .iter()
            .filter(|value| **value <= self.policy.source_isolation_threshold)
            .count();
        let minimum_source_reliability = source_present
            .iter()
            .enumerate()
            .filter_map(|(index, present)| present.then_some(self.reliability[index]))
            .fold(1.0f64, f64::min);
        self.last_report = SensorFusionReport {
            declared_sources,
            accepted_sources: accepted.len(),
            rejected_sources,
            replay_rejections,
            isolated_sources,
            channels_without_quorum,
            critical_channels_without_quorum,
            maximum_normalized_disagreement,
            minimum_source_reliability,
        };
        (fused, self.last_report)
    }

    pub fn reset_runtime(&mut self) {
        self.last_report = SensorFusionReport::nominal();
    }
}

impl Default for SensorFusionSupervisor {
    fn default() -> Self {
        Self::new(SensorFusionPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BATTERY_RATIO, GAS_RISK};

    #[test]
    fn median_fusion_rejects_single_outlier() {
        let fallback = SubterraneanState::home();
        let mut a = fallback.clone();
        let mut b = fallback.clone();
        let mut c = fallback.clone();
        a.channels[GAS_RISK] = 0.2;
        b.channels[GAS_RISK] = 0.21;
        c.channels[GAS_RISK] = 0.95;
        let frame = RedundantSensorFrame {
            observations: vec![
                SensorSourceObservation::from_state(SensorSourceId(0), 1, &a),
                SensorSourceObservation::from_state(SensorSourceId(1), 1, &b),
                SensorSourceObservation::from_state(SensorSourceId(2), 1, &c),
            ],
        };
        let (fused, report) = SensorFusionSupervisor::default().fuse(&frame, &fallback);
        assert!((fused.channels[GAS_RISK] - 0.21).abs() < 1e-9);
        assert_eq!(report.accepted_sources, 3);
    }

    #[test]
    fn critical_channel_without_quorum_fails_closed() {
        let fallback = SubterraneanState::home();
        let mut invalid = SensorSourceObservation::from_state(SensorSourceId(1), 1, &fallback);
        invalid.valid[BATTERY_RATIO] = false;
        let frame = RedundantSensorFrame {
            observations: vec![
                SensorSourceObservation::from_state(SensorSourceId(0), 1, &fallback),
                invalid,
            ],
        };
        let (_, report) = SensorFusionSupervisor::default().fuse(&frame, &fallback);
        assert!(report.requires_fail_closed());
        assert!(report.critical_channels_without_quorum > 0);
    }

    #[test]
    fn replayed_source_sequence_is_rejected() {
        let fallback = SubterraneanState::home();
        let frame = RedundantSensorFrame {
            observations: vec![SensorSourceObservation::from_state(
                SensorSourceId(0),
                1,
                &fallback,
            )],
        };
        let mut supervisor = SensorFusionSupervisor::default();
        supervisor.fuse(&frame, &fallback);
        let (_, report) = supervisor.fuse(&frame, &fallback);
        assert_eq!(report.replay_rejections, 1);
        assert!(report.requires_fail_closed());
    }
}
