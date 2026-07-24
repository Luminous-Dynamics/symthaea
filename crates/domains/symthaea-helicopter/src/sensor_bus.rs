// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic multi-rate sensor buffering and snapshot assembly.
//!
//! Sensors arrive at different rates and latencies. The bus preserves each
//! source timeline, rejects regressions, performs bounded interpolation, and
//! refuses to manufacture a control snapshot when a required channel is stale
//! or absent.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorKind {
    Imu,
    Gnss,
    VisualOdometry,
    RadarAltimeter,
    RotorTachometer,
    Powertrain,
}

impl SensorKind {
    const COUNT: usize = 6;

    const fn index(self) -> usize {
        match self {
            Self::Imu => 0,
            Self::Gnss => 1,
            Self::VisualOdometry => 2,
            Self::RadarAltimeter => 3,
            Self::RotorTachometer => 4,
            Self::Powertrain => 5,
        }
    }

    const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Imu,
            1 => Self::Gnss,
            2 => Self::VisualOdometry,
            3 => Self::RadarAltimeter,
            4 => Self::RotorTachometer,
            _ => Self::Powertrain,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorBusError {
    InvalidConfiguration,
    NonFiniteMeasurement,
    InvalidDimension,
    TimeWentBackwards,
    SnapshotTimeBeforeHistory,
    MissingRequiredSensor,
    StaleRequiredSensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorVector {
    pub values: [f64; 6],
    pub dimension: u8,
}

impl SensorVector {
    pub fn new(values: &[f64]) -> Result<Self, SensorBusError> {
        if values.is_empty() || values.len() > 6 {
            return Err(SensorBusError::InvalidDimension);
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err(SensorBusError::NonFiniteMeasurement);
        }
        let mut fixed = [0.0; 6];
        fixed[..values.len()].copy_from_slice(values);
        Ok(Self {
            values: fixed,
            dimension: values.len() as u8,
        })
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.values[..self.dimension as usize]
    }

    fn interpolate(self, other: Self, alpha: f64) -> Result<Self, SensorBusError> {
        if self.dimension != other.dimension {
            return Err(SensorBusError::InvalidDimension);
        }
        let mut values = [0.0; 6];
        for index in 0..self.dimension as usize {
            values[index] = self.values[index] + alpha * (other.values[index] - self.values[index]);
        }
        Ok(Self {
            values,
            dimension: self.dimension,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimedSensorMeasurement {
    pub kind: SensorKind,
    /// Timestamp already corrected into the host monotonic time domain.
    pub monotonic_time_s: f64,
    pub sequence: u64,
    pub vector: SensorVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorChannelPolicy {
    pub kind: SensorKind,
    pub required: bool,
    pub maximum_age_s: f64,
    pub maximum_interpolation_gap_s: f64,
}

impl SensorChannelPolicy {
    fn validate(&self) -> bool {
        self.maximum_age_s.is_finite()
            && self.maximum_age_s > 0.0
            && self.maximum_interpolation_gap_s.is_finite()
            && self.maximum_interpolation_gap_s >= 0.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorBusConfig {
    pub channels: Vec<SensorChannelPolicy>,
    pub maximum_samples_per_channel: usize,
}

impl Default for SensorBusConfig {
    fn default() -> Self {
        Self {
            channels: vec![
                SensorChannelPolicy {
                    kind: SensorKind::Imu,
                    required: true,
                    maximum_age_s: 0.020,
                    maximum_interpolation_gap_s: 0.010,
                },
                SensorChannelPolicy {
                    kind: SensorKind::Gnss,
                    required: false,
                    maximum_age_s: 1.0,
                    maximum_interpolation_gap_s: 0.5,
                },
                SensorChannelPolicy {
                    kind: SensorKind::VisualOdometry,
                    required: false,
                    maximum_age_s: 0.2,
                    maximum_interpolation_gap_s: 0.1,
                },
                SensorChannelPolicy {
                    kind: SensorKind::RadarAltimeter,
                    required: true,
                    maximum_age_s: 0.1,
                    maximum_interpolation_gap_s: 0.05,
                },
                SensorChannelPolicy {
                    kind: SensorKind::RotorTachometer,
                    required: true,
                    maximum_age_s: 0.050,
                    maximum_interpolation_gap_s: 0.025,
                },
                SensorChannelPolicy {
                    kind: SensorKind::Powertrain,
                    required: true,
                    maximum_age_s: 0.1,
                    maximum_interpolation_gap_s: 0.05,
                },
            ],
            maximum_samples_per_channel: 64,
        }
    }
}

impl SensorBusConfig {
    pub fn validate(&self) -> bool {
        self.maximum_samples_per_channel >= 2
            && self.channels.len() == SensorKind::COUNT
            && self.channels.iter().all(SensorChannelPolicy::validate)
            && (0..SensorKind::COUNT).all(|index| {
                self.channels
                    .iter()
                    .filter(|policy| policy.kind.index() == index)
                    .count()
                    == 1
            })
    }

    fn policy(&self, kind: SensorKind) -> &SensorChannelPolicy {
        self.channels
            .iter()
            .find(|policy| policy.kind == kind)
            .expect("validated sensor-bus config contains every channel")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensorSnapshotChannel {
    pub kind: SensorKind,
    pub sample_time_s: f64,
    pub age_s: f64,
    pub interpolated: bool,
    pub vector: SensorVector,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorSnapshot {
    pub monotonic_time_s: f64,
    pub channels: Vec<SensorSnapshotChannel>,
}

impl SensorSnapshot {
    pub fn channel(&self, kind: SensorKind) -> Option<&SensorSnapshotChannel> {
        self.channels.iter().find(|channel| channel.kind == kind)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorBusEvidence {
    pub accepted_measurements: u64,
    pub rejected_measurements: u64,
    pub dropped_for_capacity: u64,
    pub snapshots_built: u64,
    pub incomplete_snapshots: u64,
}

#[derive(Debug, Clone)]
pub struct MultiRateSensorBus {
    config: SensorBusConfig,
    buffers: [VecDeque<TimedSensorMeasurement>; SensorKind::COUNT],
    evidence: SensorBusEvidence,
}

impl MultiRateSensorBus {
    pub fn new(config: SensorBusConfig) -> Result<Self, SensorBusError> {
        if !config.validate() {
            return Err(SensorBusError::InvalidConfiguration);
        }
        Ok(Self {
            config,
            buffers: std::array::from_fn(|_| VecDeque::new()),
            evidence: SensorBusEvidence {
                accepted_measurements: 0,
                rejected_measurements: 0,
                dropped_for_capacity: 0,
                snapshots_built: 0,
                incomplete_snapshots: 0,
            },
        })
    }

    pub fn evidence(&self) -> SensorBusEvidence {
        self.evidence
    }

    pub fn push(&mut self, measurement: TimedSensorMeasurement) -> Result<(), SensorBusError> {
        if !measurement.monotonic_time_s.is_finite()
            || !measurement
                .vector
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        {
            self.evidence.rejected_measurements =
                self.evidence.rejected_measurements.saturating_add(1);
            return Err(SensorBusError::NonFiniteMeasurement);
        }
        let buffer = &mut self.buffers[measurement.kind.index()];
        if let Some(previous) = buffer.back() {
            if measurement.monotonic_time_s <= previous.monotonic_time_s
                || measurement.sequence <= previous.sequence
            {
                self.evidence.rejected_measurements =
                    self.evidence.rejected_measurements.saturating_add(1);
                return Err(SensorBusError::TimeWentBackwards);
            }
            if measurement.vector.dimension != previous.vector.dimension {
                self.evidence.rejected_measurements =
                    self.evidence.rejected_measurements.saturating_add(1);
                return Err(SensorBusError::InvalidDimension);
            }
        }
        buffer.push_back(measurement);
        while buffer.len() > self.config.maximum_samples_per_channel {
            buffer.pop_front();
            self.evidence.dropped_for_capacity =
                self.evidence.dropped_for_capacity.saturating_add(1);
        }
        self.evidence.accepted_measurements = self.evidence.accepted_measurements.saturating_add(1);
        Ok(())
    }

    pub fn snapshot_at(&mut self, monotonic_time_s: f64) -> Result<SensorSnapshot, SensorBusError> {
        if !monotonic_time_s.is_finite() {
            self.evidence.incomplete_snapshots =
                self.evidence.incomplete_snapshots.saturating_add(1);
            return Err(SensorBusError::NonFiniteMeasurement);
        }
        let mut channels = Vec::with_capacity(SensorKind::COUNT);
        for index in 0..SensorKind::COUNT {
            let kind = SensorKind::from_index(index);
            let policy = *self.config.policy(kind);
            match Self::sample_channel(&self.buffers[index], policy, monotonic_time_s) {
                Ok(Some(channel)) => channels.push(channel),
                Ok(None) if policy.required => {
                    self.evidence.incomplete_snapshots =
                        self.evidence.incomplete_snapshots.saturating_add(1);
                    return Err(SensorBusError::MissingRequiredSensor);
                }
                Err(SensorBusError::StaleRequiredSensor) if !policy.required => {}
                Err(error) => {
                    self.evidence.incomplete_snapshots =
                        self.evidence.incomplete_snapshots.saturating_add(1);
                    return Err(error);
                }
                Ok(None) => {}
            }
        }
        self.evidence.snapshots_built = self.evidence.snapshots_built.saturating_add(1);
        Ok(SensorSnapshot {
            monotonic_time_s,
            channels,
        })
    }

    fn sample_channel(
        buffer: &VecDeque<TimedSensorMeasurement>,
        policy: SensorChannelPolicy,
        at_s: f64,
    ) -> Result<Option<SensorSnapshotChannel>, SensorBusError> {
        if buffer.is_empty() {
            return Ok(None);
        }
        if buffer
            .front()
            .is_some_and(|first| at_s < first.monotonic_time_s)
        {
            return if policy.required {
                Err(SensorBusError::SnapshotTimeBeforeHistory)
            } else {
                Ok(None)
            };
        }

        let mut before = None;
        let mut after = None;
        for measurement in buffer {
            if measurement.monotonic_time_s <= at_s {
                before = Some(*measurement);
            } else {
                after = Some(*measurement);
                break;
            }
        }
        let Some(before) = before else {
            return Ok(None);
        };
        let age_s = at_s - before.monotonic_time_s;
        if age_s > policy.maximum_age_s {
            return if policy.required {
                Err(SensorBusError::StaleRequiredSensor)
            } else {
                Ok(None)
            };
        }

        if let Some(after) = after {
            let gap_s = after.monotonic_time_s - before.monotonic_time_s;
            let interpolation_tolerance =
                f64::EPSILON * policy.maximum_interpolation_gap_s.abs().max(1.0) * 8.0;
            if gap_s > 0.0
                && gap_s <= policy.maximum_interpolation_gap_s + interpolation_tolerance
                && at_s > before.monotonic_time_s
            {
                let alpha = (at_s - before.monotonic_time_s) / gap_s;
                return Ok(Some(SensorSnapshotChannel {
                    kind: policy.kind,
                    sample_time_s: at_s,
                    age_s: 0.0,
                    interpolated: true,
                    vector: before.vector.interpolate(after.vector, alpha)?,
                }));
            }
        }
        Ok(Some(SensorSnapshotChannel {
            kind: policy.kind,
            sample_time_s: before.monotonic_time_s,
            age_s,
            interpolated: false,
            vector: before.vector,
        }))
    }

    pub fn clear(&mut self) {
        for buffer in &mut self.buffers {
            buffer.clear();
        }
        self.evidence = SensorBusEvidence {
            accepted_measurements: 0,
            rejected_measurements: 0,
            dropped_for_capacity: 0,
            snapshots_built: 0,
            incomplete_snapshots: 0,
        };
    }
}

impl Default for MultiRateSensorBus {
    fn default() -> Self {
        Self::new(SensorBusConfig::default())
            .expect("default multi-rate sensor-bus configuration is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_required(bus: &mut MultiRateSensorBus, time_s: f64, sequence: u64) {
        for kind in [
            SensorKind::Imu,
            SensorKind::RadarAltimeter,
            SensorKind::RotorTachometer,
            SensorKind::Powertrain,
        ] {
            bus.push(TimedSensorMeasurement {
                kind,
                monotonic_time_s: time_s,
                sequence,
                vector: SensorVector::new(&[sequence as f64]).unwrap(),
            })
            .unwrap();
        }
    }

    #[test]
    fn required_channels_build_snapshot() {
        let mut bus = MultiRateSensorBus::default();
        push_required(&mut bus, 1.0, 1);
        let snapshot = bus.snapshot_at(1.005).unwrap();
        assert_eq!(snapshot.channels.len(), 4);
        assert!(snapshot.channel(SensorKind::Imu).is_some());
    }

    #[test]
    fn bounded_interpolation_uses_bracketing_samples() {
        let mut bus = MultiRateSensorBus::default();
        push_required(&mut bus, 1.0, 1);
        push_required(&mut bus, 1.01, 2);
        let snapshot = bus.snapshot_at(1.005).unwrap();
        let imu = snapshot.channel(SensorKind::Imu).unwrap();
        assert!(imu.interpolated);
        assert!((imu.vector.as_slice()[0] - 1.5).abs() < 1.0e-9);
    }

    #[test]
    fn stale_required_channel_fails_closed() {
        let mut bus = MultiRateSensorBus::default();
        push_required(&mut bus, 0.0, 1);
        assert_eq!(
            bus.snapshot_at(1.0),
            Err(SensorBusError::StaleRequiredSensor)
        );
    }

    #[test]
    fn sequence_or_time_regression_is_rejected() {
        let mut bus = MultiRateSensorBus::default();
        let measurement = TimedSensorMeasurement {
            kind: SensorKind::Imu,
            monotonic_time_s: 1.0,
            sequence: 1,
            vector: SensorVector::new(&[0.0; 6]).unwrap(),
        };
        bus.push(measurement).unwrap();
        assert_eq!(
            bus.push(measurement),
            Err(SensorBusError::TimeWentBackwards)
        );
    }
}
