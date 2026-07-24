// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Control-loop deadline, jitter, and sensor-to-actuator latency evidence.
//!
//! Simulation correctness does not establish that a physical control loop met
//! real-time deadlines. This monitor consumes explicit scheduled/actual timing
//! observations, latches unsafe deadline streaks, and exposes qualification
//! metrics without depending on wall-clock APIs inside the controller.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealtimeHealth {
    Nominal,
    Degraded,
    Unsafe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimeMonitorError {
    InvalidConfiguration,
    NonFiniteObservation,
    InvalidTimingOrder,
    SequenceDidNotIncrease,
    TimeWentBackwards,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RealtimeMonitorConfig {
    pub nominal_period_s: f64,
    pub maximum_start_jitter_s: f64,
    pub maximum_sensor_to_actuator_latency_s: f64,
    pub maximum_consecutive_deadline_misses: u32,
}

impl Default for RealtimeMonitorConfig {
    fn default() -> Self {
        Self {
            nominal_period_s: 1.0 / 300.0,
            maximum_start_jitter_s: 0.001,
            maximum_sensor_to_actuator_latency_s: 0.010,
            maximum_consecutive_deadline_misses: 3,
        }
    }
}

impl RealtimeMonitorConfig {
    pub fn validate(&self) -> bool {
        self.nominal_period_s.is_finite()
            && self.nominal_period_s > 0.0
            && self.maximum_start_jitter_s.is_finite()
            && self.maximum_start_jitter_s >= 0.0
            && self.maximum_sensor_to_actuator_latency_s.is_finite()
            && self.maximum_sensor_to_actuator_latency_s > 0.0
            && self.maximum_consecutive_deadline_misses > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControlCycleTiming {
    pub sequence: u64,
    pub scheduled_start_s: f64,
    pub actual_start_s: f64,
    pub sensor_sample_time_s: f64,
    pub command_commit_time_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControlCycleAssessment {
    pub sequence: u64,
    pub start_jitter_s: f64,
    pub execution_time_s: f64,
    pub sensor_to_actuator_latency_s: f64,
    pub deadline_missed: bool,
    pub latency_exceeded: bool,
    pub health: RealtimeHealth,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RealtimeEvidence {
    pub health: RealtimeHealth,
    pub observed_cycles: u64,
    pub deadline_misses: u64,
    pub latency_violations: u64,
    pub consecutive_deadline_misses: u32,
    pub maximum_abs_start_jitter_s: f64,
    pub maximum_execution_time_s: f64,
    pub maximum_sensor_to_actuator_latency_s: f64,
    pub mean_abs_start_jitter_s: f64,
}

#[derive(Debug, Clone)]
pub struct RealtimeControlMonitor {
    config: RealtimeMonitorConfig,
    evidence: RealtimeEvidence,
    last_sequence: Option<u64>,
    last_scheduled_start_s: Option<f64>,
}

impl RealtimeControlMonitor {
    pub fn new(config: RealtimeMonitorConfig) -> Result<Self, RealtimeMonitorError> {
        if !config.validate() {
            return Err(RealtimeMonitorError::InvalidConfiguration);
        }
        Ok(Self {
            config,
            evidence: RealtimeEvidence {
                health: RealtimeHealth::Nominal,
                observed_cycles: 0,
                deadline_misses: 0,
                latency_violations: 0,
                consecutive_deadline_misses: 0,
                maximum_abs_start_jitter_s: 0.0,
                maximum_execution_time_s: 0.0,
                maximum_sensor_to_actuator_latency_s: 0.0,
                mean_abs_start_jitter_s: 0.0,
            },
            last_sequence: None,
            last_scheduled_start_s: None,
        })
    }

    pub fn observe(
        &mut self,
        timing: ControlCycleTiming,
    ) -> Result<ControlCycleAssessment, RealtimeMonitorError> {
        let values = [
            timing.scheduled_start_s,
            timing.actual_start_s,
            timing.sensor_sample_time_s,
            timing.command_commit_time_s,
        ];
        if !values.iter().all(|value| value.is_finite()) {
            return Err(RealtimeMonitorError::NonFiniteObservation);
        }
        if timing.sensor_sample_time_s > timing.command_commit_time_s
            || timing.actual_start_s > timing.command_commit_time_s
        {
            return Err(RealtimeMonitorError::InvalidTimingOrder);
        }
        if self
            .last_sequence
            .is_some_and(|previous| timing.sequence <= previous)
        {
            return Err(RealtimeMonitorError::SequenceDidNotIncrease);
        }
        if self
            .last_scheduled_start_s
            .is_some_and(|previous| timing.scheduled_start_s <= previous)
        {
            return Err(RealtimeMonitorError::TimeWentBackwards);
        }

        let start_jitter_s = timing.actual_start_s - timing.scheduled_start_s;
        let execution_time_s = timing.command_commit_time_s - timing.actual_start_s;
        let sensor_to_actuator_latency_s =
            timing.command_commit_time_s - timing.sensor_sample_time_s;
        let deadline_missed =
            timing.command_commit_time_s > timing.scheduled_start_s + self.config.nominal_period_s;
        let latency_exceeded =
            sensor_to_actuator_latency_s > self.config.maximum_sensor_to_actuator_latency_s;
        let jitter_exceeded = start_jitter_s.abs() > self.config.maximum_start_jitter_s;

        self.evidence.observed_cycles = self.evidence.observed_cycles.saturating_add(1);
        let count = self.evidence.observed_cycles as f64;
        let abs_jitter = start_jitter_s.abs();
        self.evidence.mean_abs_start_jitter_s +=
            (abs_jitter - self.evidence.mean_abs_start_jitter_s) / count;
        self.evidence.maximum_abs_start_jitter_s =
            self.evidence.maximum_abs_start_jitter_s.max(abs_jitter);
        self.evidence.maximum_execution_time_s =
            self.evidence.maximum_execution_time_s.max(execution_time_s);
        self.evidence.maximum_sensor_to_actuator_latency_s = self
            .evidence
            .maximum_sensor_to_actuator_latency_s
            .max(sensor_to_actuator_latency_s);

        if deadline_missed {
            self.evidence.deadline_misses = self.evidence.deadline_misses.saturating_add(1);
            self.evidence.consecutive_deadline_misses =
                self.evidence.consecutive_deadline_misses.saturating_add(1);
        } else {
            self.evidence.consecutive_deadline_misses = 0;
        }
        if latency_exceeded {
            self.evidence.latency_violations = self.evidence.latency_violations.saturating_add(1);
        }
        self.evidence.health = if self.evidence.consecutive_deadline_misses
            >= self.config.maximum_consecutive_deadline_misses
        {
            RealtimeHealth::Unsafe
        } else if deadline_missed || latency_exceeded || jitter_exceeded {
            RealtimeHealth::Degraded
        } else {
            RealtimeHealth::Nominal
        };
        self.last_sequence = Some(timing.sequence);
        self.last_scheduled_start_s = Some(timing.scheduled_start_s);

        Ok(ControlCycleAssessment {
            sequence: timing.sequence,
            start_jitter_s,
            execution_time_s,
            sensor_to_actuator_latency_s,
            deadline_missed,
            latency_exceeded,
            health: self.evidence.health,
        })
    }

    pub fn evidence(&self) -> RealtimeEvidence {
        self.evidence
    }

    pub fn reset(&mut self) {
        let config = self.config;
        *self = Self::new(config).expect("validated real-time configuration remains valid");
    }
}

impl Default for RealtimeControlMonitor {
    fn default() -> Self {
        Self::new(RealtimeMonitorConfig::default())
            .expect("default real-time monitor configuration is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timing(sequence: u64, scheduled: f64, commit_delay_s: f64) -> ControlCycleTiming {
        ControlCycleTiming {
            sequence,
            scheduled_start_s: scheduled,
            actual_start_s: scheduled + 0.0002,
            sensor_sample_time_s: scheduled,
            command_commit_time_s: scheduled + commit_delay_s,
        }
    }

    #[test]
    fn nominal_cycle_records_bounded_latency() {
        let mut monitor = RealtimeControlMonitor::default();
        let assessment = monitor.observe(timing(1, 0.0, 0.002)).unwrap();
        assert_eq!(assessment.health, RealtimeHealth::Nominal);
        assert_eq!(monitor.evidence().deadline_misses, 0);
    }

    #[test]
    fn repeated_deadline_misses_latch_unsafe_health() {
        let mut monitor = RealtimeControlMonitor::default();
        for sequence in 1..=3 {
            monitor
                .observe(timing(sequence, (sequence - 1) as f64 / 300.0, 0.02))
                .unwrap();
        }
        assert_eq!(monitor.evidence().health, RealtimeHealth::Unsafe);
        assert_eq!(monitor.evidence().consecutive_deadline_misses, 3);
    }

    #[test]
    fn sequence_regression_is_rejected() {
        let mut monitor = RealtimeControlMonitor::default();
        monitor.observe(timing(2, 0.0, 0.002)).unwrap();
        assert_eq!(
            monitor.observe(timing(1, 1.0 / 300.0, 0.002)),
            Err(RealtimeMonitorError::SequenceDidNotIncrease)
        );
    }
}
