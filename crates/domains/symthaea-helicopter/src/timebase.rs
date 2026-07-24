// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic sensor-clock discipline and timestamp normalization.
//!
//! Physical sensor clocks do not necessarily share the host monotonic epoch.
//! This module estimates a bounded affine mapping from a source clock into the
//! host time domain, rejects implausible drift/offset jumps, and exposes a
//! freshness-safe corrected timestamp for downstream estimators and evidence.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockLockState {
    Uninitialized,
    Synchronizing,
    Locked,
    Faulted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimebaseError {
    InvalidConfiguration,
    NonFiniteTimestamp,
    SourceTimeWentBackwards,
    HostTimeWentBackwards,
    OffsetJumpExceeded,
    DriftExceeded,
    NotSynchronized,
    CorrectedTimeInFuture,
    SampleStale,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClockDisciplineConfig {
    /// Number of accepted observations required before timestamps are trusted.
    pub minimum_lock_samples: u32,
    /// Exponential update gain for offset and scale estimates.
    pub adaptation_gain: f64,
    /// Maximum accepted instantaneous offset innovation, seconds.
    pub maximum_offset_jump_s: f64,
    /// Maximum source clock-rate error relative to host, parts per million.
    pub maximum_drift_ppm: f64,
    /// Corrected timestamps may lead host receive time only by this tolerance.
    pub maximum_future_skew_s: f64,
    /// Maximum corrected sample age at the point of use.
    pub maximum_sample_age_s: f64,
}

impl Default for ClockDisciplineConfig {
    fn default() -> Self {
        Self {
            minimum_lock_samples: 4,
            adaptation_gain: 0.1,
            maximum_offset_jump_s: 0.250,
            maximum_drift_ppm: 500.0,
            maximum_future_skew_s: 0.005,
            maximum_sample_age_s: 0.100,
        }
    }
}

impl ClockDisciplineConfig {
    pub fn validate(&self) -> bool {
        self.minimum_lock_samples > 0
            && self.adaptation_gain.is_finite()
            && (0.0..=1.0).contains(&self.adaptation_gain)
            && self.adaptation_gain > 0.0
            && self.maximum_offset_jump_s.is_finite()
            && self.maximum_offset_jump_s > 0.0
            && self.maximum_drift_ppm.is_finite()
            && self.maximum_drift_ppm > 0.0
            && self.maximum_future_skew_s.is_finite()
            && self.maximum_future_skew_s >= 0.0
            && self.maximum_sample_age_s.is_finite()
            && self.maximum_sample_age_s > 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClockObservation {
    pub source_time_s: f64,
    pub host_receive_time_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CorrectedTimestamp {
    pub source_time_s: f64,
    pub corrected_host_time_s: f64,
    pub age_s: f64,
    pub estimated_offset_s: f64,
    pub estimated_drift_ppm: f64,
    pub lock_state: ClockLockState,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClockDisciplineEvidence {
    pub lock_state: ClockLockState,
    pub accepted_samples: u64,
    pub rejected_samples: u64,
    pub estimated_offset_s: f64,
    pub estimated_drift_ppm: f64,
    pub last_source_time_s: Option<f64>,
    pub last_host_time_s: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SensorClockDiscipline {
    config: ClockDisciplineConfig,
    lock_state: ClockLockState,
    accepted_samples: u64,
    rejected_samples: u64,
    estimated_offset_s: f64,
    estimated_scale: f64,
    last_source_time_s: Option<f64>,
    last_host_time_s: Option<f64>,
}

impl SensorClockDiscipline {
    pub fn new(config: ClockDisciplineConfig) -> Result<Self, TimebaseError> {
        if !config.validate() {
            return Err(TimebaseError::InvalidConfiguration);
        }
        Ok(Self {
            config,
            lock_state: ClockLockState::Uninitialized,
            accepted_samples: 0,
            rejected_samples: 0,
            estimated_offset_s: 0.0,
            estimated_scale: 1.0,
            last_source_time_s: None,
            last_host_time_s: None,
        })
    }

    pub fn evidence(&self) -> ClockDisciplineEvidence {
        ClockDisciplineEvidence {
            lock_state: self.lock_state,
            accepted_samples: self.accepted_samples,
            rejected_samples: self.rejected_samples,
            estimated_offset_s: self.estimated_offset_s,
            estimated_drift_ppm: (self.estimated_scale - 1.0) * 1.0e6,
            last_source_time_s: self.last_source_time_s,
            last_host_time_s: self.last_host_time_s,
        }
    }

    pub fn observe(&mut self, observation: ClockObservation) -> Result<(), TimebaseError> {
        if !observation.source_time_s.is_finite() || !observation.host_receive_time_s.is_finite() {
            return self.reject(TimebaseError::NonFiniteTimestamp, false);
        }
        if self
            .last_source_time_s
            .is_some_and(|previous| observation.source_time_s <= previous)
        {
            return self.reject(TimebaseError::SourceTimeWentBackwards, true);
        }
        if self
            .last_host_time_s
            .is_some_and(|previous| observation.host_receive_time_s < previous)
        {
            return self.reject(TimebaseError::HostTimeWentBackwards, true);
        }

        let raw_offset_s = observation.host_receive_time_s - observation.source_time_s;
        if self.accepted_samples == 0 {
            self.estimated_offset_s = raw_offset_s;
            self.estimated_scale = 1.0;
        } else {
            let previous_source = self.last_source_time_s.unwrap_or(observation.source_time_s);
            let previous_host = self
                .last_host_time_s
                .unwrap_or(observation.host_receive_time_s);
            let source_delta = observation.source_time_s - previous_source;
            let host_delta = observation.host_receive_time_s - previous_host;
            if source_delta <= 0.0 {
                return self.reject(TimebaseError::SourceTimeWentBackwards, true);
            }
            let observed_scale = host_delta / source_delta;
            let observed_drift_ppm = (observed_scale - 1.0) * 1.0e6;
            if !observed_scale.is_finite()
                || observed_drift_ppm.abs() > self.config.maximum_drift_ppm
            {
                return self.reject(TimebaseError::DriftExceeded, true);
            }
            let predicted_host =
                self.estimated_offset_s + self.estimated_scale * observation.source_time_s;
            let innovation_s = observation.host_receive_time_s - predicted_host;
            if innovation_s.abs() > self.config.maximum_offset_jump_s {
                return self.reject(TimebaseError::OffsetJumpExceeded, true);
            }
            let gain = self.config.adaptation_gain;
            self.estimated_scale += gain * (observed_scale - self.estimated_scale);
            self.estimated_offset_s += gain * innovation_s;
        }

        self.accepted_samples = self.accepted_samples.saturating_add(1);
        self.last_source_time_s = Some(observation.source_time_s);
        self.last_host_time_s = Some(observation.host_receive_time_s);
        self.lock_state = if self.accepted_samples >= self.config.minimum_lock_samples as u64 {
            ClockLockState::Locked
        } else {
            ClockLockState::Synchronizing
        };
        Ok(())
    }

    pub fn correct(
        &self,
        source_time_s: f64,
        host_now_s: f64,
    ) -> Result<CorrectedTimestamp, TimebaseError> {
        if !source_time_s.is_finite() || !host_now_s.is_finite() {
            return Err(TimebaseError::NonFiniteTimestamp);
        }
        if self.lock_state != ClockLockState::Locked {
            return Err(TimebaseError::NotSynchronized);
        }
        let corrected_host_time_s = self.estimated_offset_s + self.estimated_scale * source_time_s;
        if corrected_host_time_s > host_now_s + self.config.maximum_future_skew_s {
            return Err(TimebaseError::CorrectedTimeInFuture);
        }
        let age_s = (host_now_s - corrected_host_time_s).max(0.0);
        if age_s > self.config.maximum_sample_age_s {
            return Err(TimebaseError::SampleStale);
        }
        Ok(CorrectedTimestamp {
            source_time_s,
            corrected_host_time_s,
            age_s,
            estimated_offset_s: self.estimated_offset_s,
            estimated_drift_ppm: (self.estimated_scale - 1.0) * 1.0e6,
            lock_state: self.lock_state,
        })
    }

    pub fn reset(&mut self) {
        self.lock_state = ClockLockState::Uninitialized;
        self.accepted_samples = 0;
        self.rejected_samples = 0;
        self.estimated_offset_s = 0.0;
        self.estimated_scale = 1.0;
        self.last_source_time_s = None;
        self.last_host_time_s = None;
    }

    fn reject(&mut self, error: TimebaseError, fault: bool) -> Result<(), TimebaseError> {
        self.rejected_samples = self.rejected_samples.saturating_add(1);
        if fault {
            self.lock_state = ClockLockState::Faulted;
        }
        Err(error)
    }
}

impl Default for SensorClockDiscipline {
    fn default() -> Self {
        Self::new(ClockDisciplineConfig::default())
            .expect("default clock-discipline configuration is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock(clock: &mut SensorClockDiscipline) {
        for i in 0..4 {
            let source = i as f64 * 0.01;
            clock
                .observe(ClockObservation {
                    source_time_s: source,
                    host_receive_time_s: source + 10.0,
                })
                .unwrap();
        }
    }

    #[test]
    fn offset_epoch_is_normalized_after_lock() {
        let mut clock = SensorClockDiscipline::default();
        lock(&mut clock);
        let corrected = clock.correct(0.03, 10.031).unwrap();
        assert!((corrected.corrected_host_time_s - 10.03).abs() < 1.0e-9);
        assert!(corrected.age_s <= 0.0011);
        assert_eq!(corrected.lock_state, ClockLockState::Locked);
    }

    #[test]
    fn stale_corrected_sample_fails_closed() {
        let mut clock = SensorClockDiscipline::default();
        lock(&mut clock);
        assert_eq!(clock.correct(0.03, 10.5), Err(TimebaseError::SampleStale));
    }

    #[test]
    fn drift_beyond_policy_faults_clock() {
        let mut clock = SensorClockDiscipline::default();
        clock
            .observe(ClockObservation {
                source_time_s: 0.0,
                host_receive_time_s: 0.0,
            })
            .unwrap();
        assert_eq!(
            clock.observe(ClockObservation {
                source_time_s: 1.0,
                host_receive_time_s: 1.01,
            }),
            Err(TimebaseError::DriftExceeded)
        );
        assert_eq!(clock.evidence().lock_state, ClockLockState::Faulted);
    }

    #[test]
    fn reset_clears_fault_and_evidence() {
        let mut clock = SensorClockDiscipline::default();
        let _ = clock.observe(ClockObservation {
            source_time_s: f64::NAN,
            host_receive_time_s: 0.0,
        });
        clock.reset();
        assert_eq!(clock.evidence().lock_state, ClockLockState::Uninitialized);
        assert_eq!(clock.evidence().rejected_samples, 0);
    }
}
