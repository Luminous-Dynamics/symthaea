// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic sensor fault and degradation models for SIL/HIL campaigns.
//!
//! Truth samples must not be passed directly into the estimator when evaluating
//! fault tolerance. This module applies explicit bias, scale, stuck, dropout,
//! white-noise, and random-walk faults while retaining replayable evidence.

use serde::{Deserialize, Serialize};

use crate::sensor_bus::{SensorKind, SensorVector, TimedSensorMeasurement};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SensorFaultMode {
    Healthy,
    Bias { offset: SensorVector },
    Scale { factor: f64 },
    StuckAt { value: SensorVector },
    Dropout,
    WhiteNoise { standard_deviation: f64 },
    RandomWalk { standard_deviation_per_sqrt_s: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorFaultConfig {
    pub kind: SensorKind,
    pub mode: SensorFaultMode,
    pub start_time_s: f64,
    pub end_time_s: Option<f64>,
    pub seed: u64,
}

impl SensorFaultConfig {
    pub fn validate(&self) -> Result<(), SensorFaultError> {
        if !self.start_time_s.is_finite() || self.start_time_s < 0.0 {
            return Err(SensorFaultError::InvalidConfiguration);
        }
        if self
            .end_time_s
            .is_some_and(|end| !end.is_finite() || end <= self.start_time_s)
        {
            return Err(SensorFaultError::InvalidConfiguration);
        }
        match self.mode {
            SensorFaultMode::Scale { factor } => {
                if !factor.is_finite() || factor < 0.0 {
                    return Err(SensorFaultError::InvalidConfiguration);
                }
            }
            SensorFaultMode::WhiteNoise { standard_deviation }
            | SensorFaultMode::RandomWalk {
                standard_deviation_per_sqrt_s: standard_deviation,
            } => {
                if !standard_deviation.is_finite() || standard_deviation < 0.0 {
                    return Err(SensorFaultError::InvalidConfiguration);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn active_at(&self, time_s: f64) -> bool {
        time_s >= self.start_time_s && self.end_time_s.is_none_or(|end| time_s < end)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorFaultError {
    InvalidConfiguration,
    WrongSensorKind,
    TimeWentBackwards,
    DimensionMismatch,
    NonFiniteOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorFaultEvidence {
    pub accepted_truth_samples: u64,
    pub faulted_samples: u64,
    pub dropped_samples: u64,
    pub inactive_passthrough_samples: u64,
    pub replay_resets: u64,
}

#[derive(Debug, Clone)]
pub struct SensorFaultModel {
    config: SensorFaultConfig,
    initial_seed: u64,
    rng_state: u64,
    random_walk: [f64; 6],
    last_time_s: Option<f64>,
    evidence: SensorFaultEvidence,
}

impl SensorFaultModel {
    pub fn new(config: SensorFaultConfig) -> Result<Self, SensorFaultError> {
        config.validate()?;
        let seed = nonzero_seed(config.seed);
        Ok(Self {
            config,
            initial_seed: seed,
            rng_state: seed,
            random_walk: [0.0; 6],
            last_time_s: None,
            evidence: SensorFaultEvidence {
                accepted_truth_samples: 0,
                faulted_samples: 0,
                dropped_samples: 0,
                inactive_passthrough_samples: 0,
                replay_resets: 0,
            },
        })
    }

    pub fn config(&self) -> SensorFaultConfig {
        self.config
    }

    pub fn evidence(&self) -> SensorFaultEvidence {
        self.evidence
    }

    pub fn apply(
        &mut self,
        measurement: TimedSensorMeasurement,
    ) -> Result<Option<TimedSensorMeasurement>, SensorFaultError> {
        self.config.validate()?;
        if measurement.kind != self.config.kind {
            return Err(SensorFaultError::WrongSensorKind);
        }
        if self
            .last_time_s
            .is_some_and(|last| measurement.monotonic_time_s < last)
        {
            return Err(SensorFaultError::TimeWentBackwards);
        }
        let dt_s = self
            .last_time_s
            .map(|last| measurement.monotonic_time_s - last)
            .unwrap_or(0.0);
        self.last_time_s = Some(measurement.monotonic_time_s);
        self.evidence.accepted_truth_samples =
            self.evidence.accepted_truth_samples.saturating_add(1);

        if !self.config.active_at(measurement.monotonic_time_s)
            || self.config.mode == SensorFaultMode::Healthy
        {
            self.evidence.inactive_passthrough_samples =
                self.evidence.inactive_passthrough_samples.saturating_add(1);
            return Ok(Some(measurement));
        }

        if self.config.mode == SensorFaultMode::Dropout {
            self.evidence.faulted_samples = self.evidence.faulted_samples.saturating_add(1);
            self.evidence.dropped_samples = self.evidence.dropped_samples.saturating_add(1);
            return Ok(None);
        }

        let mut output = measurement;
        let dimension = output.vector.dimension as usize;
        match self.config.mode {
            SensorFaultMode::Healthy | SensorFaultMode::Dropout => unreachable!(),
            SensorFaultMode::Bias { offset } => {
                require_same_dimension(output.vector, offset)?;
                for index in 0..dimension {
                    output.vector.values[index] += offset.values[index];
                }
            }
            SensorFaultMode::Scale { factor } => {
                for index in 0..dimension {
                    output.vector.values[index] *= factor;
                }
            }
            SensorFaultMode::StuckAt { value } => {
                require_same_dimension(output.vector, value)?;
                output.vector = value;
            }
            SensorFaultMode::WhiteNoise { standard_deviation } => {
                for index in 0..dimension {
                    output.vector.values[index] += standard_deviation * self.standard_normal();
                }
            }
            SensorFaultMode::RandomWalk {
                standard_deviation_per_sqrt_s,
            } => {
                let scale = standard_deviation_per_sqrt_s * dt_s.max(0.0).sqrt();
                for index in 0..dimension {
                    self.random_walk[index] += scale * self.standard_normal();
                    output.vector.values[index] += self.random_walk[index];
                }
            }
        }

        if !output
            .vector
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(SensorFaultError::NonFiniteOutput);
        }
        self.evidence.faulted_samples = self.evidence.faulted_samples.saturating_add(1);
        Ok(Some(output))
    }

    /// Reset stochastic state so the same truth stream replays identically.
    pub fn reset_replay(&mut self) {
        self.rng_state = self.initial_seed;
        self.random_walk = [0.0; 6];
        self.last_time_s = None;
        self.evidence = SensorFaultEvidence {
            replay_resets: self.evidence.replay_resets.saturating_add(1),
            accepted_truth_samples: 0,
            faulted_samples: 0,
            dropped_samples: 0,
            inactive_passthrough_samples: 0,
        };
    }

    fn next_unit(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x as f64) / (u64::MAX as f64)
    }

    /// Bounded central-limit approximation suitable for deterministic fault campaigns.
    fn standard_normal(&mut self) -> f64 {
        let mut sum = 0.0;
        for _ in 0..12 {
            sum += self.next_unit();
        }
        sum - 6.0
    }
}

fn nonzero_seed(seed: u64) -> u64 {
    if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    }
}

fn require_same_dimension(a: SensorVector, b: SensorVector) -> Result<(), SensorFaultError> {
    if a.dimension != b.dimension {
        Err(SensorFaultError::DimensionMismatch)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement(time_s: f64, values: &[f64]) -> TimedSensorMeasurement {
        TimedSensorMeasurement {
            kind: SensorKind::Imu,
            monotonic_time_s: time_s,
            sequence: (time_s * 1000.0) as u64,
            vector: SensorVector::new(values).unwrap(),
        }
    }

    #[test]
    fn bias_is_applied_only_inside_active_window() {
        let mut model = SensorFaultModel::new(SensorFaultConfig {
            kind: SensorKind::Imu,
            mode: SensorFaultMode::Bias {
                offset: SensorVector::new(&[1.0, -2.0, 0.5]).unwrap(),
            },
            start_time_s: 1.0,
            end_time_s: Some(2.0),
            seed: 1,
        })
        .unwrap();
        let before = model
            .apply(measurement(0.5, &[0.0, 0.0, 0.0]))
            .unwrap()
            .unwrap();
        let active = model
            .apply(measurement(1.5, &[0.0, 0.0, 0.0]))
            .unwrap()
            .unwrap();
        assert_eq!(before.vector.as_slice(), &[0.0, 0.0, 0.0]);
        assert_eq!(active.vector.as_slice(), &[1.0, -2.0, 0.5]);
    }

    #[test]
    fn dropout_returns_no_measurement_and_records_evidence() {
        let mut model = SensorFaultModel::new(SensorFaultConfig {
            kind: SensorKind::Imu,
            mode: SensorFaultMode::Dropout,
            start_time_s: 0.0,
            end_time_s: None,
            seed: 2,
        })
        .unwrap();
        assert!(model.apply(measurement(0.0, &[1.0])).unwrap().is_none());
        assert_eq!(model.evidence().dropped_samples, 1);
    }

    #[test]
    fn noise_replays_exactly_after_reset() {
        let config = SensorFaultConfig {
            kind: SensorKind::Imu,
            mode: SensorFaultMode::WhiteNoise {
                standard_deviation: 0.1,
            },
            start_time_s: 0.0,
            end_time_s: None,
            seed: 42,
        };
        let mut model = SensorFaultModel::new(config).unwrap();
        let first = model.apply(measurement(0.0, &[0.0, 0.0])).unwrap().unwrap();
        model.reset_replay();
        let replay = model.apply(measurement(0.0, &[0.0, 0.0])).unwrap().unwrap();
        assert_eq!(first.vector, replay.vector);
    }

    #[test]
    fn random_walk_rejects_backward_time() {
        let mut model = SensorFaultModel::new(SensorFaultConfig {
            kind: SensorKind::Imu,
            mode: SensorFaultMode::RandomWalk {
                standard_deviation_per_sqrt_s: 0.01,
            },
            start_time_s: 0.0,
            end_time_s: None,
            seed: 7,
        })
        .unwrap();
        model.apply(measurement(1.0, &[0.0])).unwrap();
        assert_eq!(
            model.apply(measurement(0.9, &[0.0])).unwrap_err(),
            SensorFaultError::TimeWentBackwards
        );
    }

    #[test]
    fn dimension_mismatch_fails_closed() {
        let mut model = SensorFaultModel::new(SensorFaultConfig {
            kind: SensorKind::Imu,
            mode: SensorFaultMode::StuckAt {
                value: SensorVector::new(&[1.0, 2.0]).unwrap(),
            },
            start_time_s: 0.0,
            end_time_s: None,
            seed: 9,
        })
        .unwrap();
        assert_eq!(
            model.apply(measurement(0.0, &[0.0])).unwrap_err(),
            SensorFaultError::DimensionMismatch
        );
    }
}
