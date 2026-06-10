// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sensor recording and replay for offline debugging.
//!
//! - [`RecordingAdapter`]: wraps any [`HalSensorAdapter`], transparently
//!   recording all `read_raw()` calls with timestamps.
//! - [`ReplayAdapter`]: plays back a `Vec<SensorRecording>` in order.
//! - [`SensorRecording`]: a single timestamped sensor reading.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::sensor::HalSensorAdapter;

// ============================================================================
// SENSOR RECORDING
// ============================================================================

/// A single recorded sensor reading with a microsecond timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorRecording {
    /// Sensor name at time of recording.
    pub name: String,
    /// Microseconds since the [`RecordingAdapter`] was created.
    pub timestamp_us: u64,
    /// The sensor reading (`None` if the sensor returned no data).
    pub data: Option<Vec<f32>>,
}

// ============================================================================
// RECORDING ADAPTER
// ============================================================================

/// Wraps any [`HalSensorAdapter`], transparently recording all reads.
pub struct RecordingAdapter {
    inner: Box<dyn HalSensorAdapter>,
    recordings: Vec<SensorRecording>,
    start: Instant,
}

impl RecordingAdapter {
    /// Create a new recording adapter wrapping an existing sensor.
    pub fn new(inner: Box<dyn HalSensorAdapter>) -> Self {
        Self {
            inner,
            recordings: Vec::new(),
            start: Instant::now(),
        }
    }

    /// Take all recordings, resetting the buffer.
    pub fn take_recordings(&mut self) -> Vec<SensorRecording> {
        std::mem::take(&mut self.recordings)
    }

    /// Borrow the current recordings (non-consuming).
    pub fn recordings(&self) -> &[SensorRecording] {
        &self.recordings
    }

    /// Number of recordings captured so far.
    pub fn recording_count(&self) -> usize {
        self.recordings.len()
    }
}

impl HalSensorAdapter for RecordingAdapter {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn read_raw(&mut self) -> Option<Vec<f32>> {
        let data = self.inner.read_raw();
        let timestamp_us = self.start.elapsed().as_micros() as u64;
        self.recordings.push(SensorRecording {
            name: self.inner.name().to_string(),
            timestamp_us,
            data: data.clone(),
        });
        data
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }
}

// ============================================================================
// REPLAY ADAPTER
// ============================================================================

/// Replays a sequence of [`SensorRecording`]s as a virtual sensor.
pub struct ReplayAdapter {
    name: String,
    recordings: Vec<SensorRecording>,
    index: usize,
}

impl ReplayAdapter {
    /// Create a new replay adapter from a sequence of recordings.
    ///
    /// The `name` is used for the `HalSensorAdapter::name()` method.
    pub fn new(name: impl Into<String>, recordings: Vec<SensorRecording>) -> Self {
        Self {
            name: name.into(),
            recordings,
            index: 0,
        }
    }

    /// Number of recordings remaining to be replayed.
    pub fn remaining(&self) -> usize {
        self.recordings.len().saturating_sub(self.index)
    }
}

impl HalSensorAdapter for ReplayAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn read_raw(&mut self) -> Option<Vec<f32>> {
        if self.index < self.recordings.len() {
            let rec = &self.recordings[self.index];
            self.index += 1;
            rec.data.clone()
        } else {
            None
        }
    }

    fn is_available(&self) -> bool {
        self.index < self.recordings.len()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockHalSensor;

    #[test]
    fn test_recording_adapter_captures_reads() {
        let sensor = MockHalSensor::new("imu", vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let mut rec = RecordingAdapter::new(Box::new(sensor));

        let r1 = rec.read_raw().unwrap();
        assert_eq!(r1, vec![1.0, 2.0]);
        let r2 = rec.read_raw().unwrap();
        assert_eq!(r2, vec![3.0, 4.0]);

        assert_eq!(rec.recording_count(), 2);
        let recs = rec.recordings();
        assert_eq!(recs[0].name, "imu");
        assert!(recs[1].timestamp_us >= recs[0].timestamp_us);
        assert_eq!(recs[0].data, Some(vec![1.0, 2.0]));
    }

    #[test]
    fn test_recording_adapter_captures_none() {
        let sensor = MockHalSensor::new("imu", vec![]);
        let mut rec = RecordingAdapter::new(Box::new(sensor));

        let result = rec.read_raw();
        assert!(result.is_none());
        assert_eq!(rec.recording_count(), 1);
        assert!(rec.recordings()[0].data.is_none());
    }

    #[test]
    fn test_recording_adapter_delegation() {
        let sensor = MockHalSensor::new("accel", vec![vec![1.0]]);
        let rec = RecordingAdapter::new(Box::new(sensor));
        assert_eq!(rec.name(), "accel");
        assert!(rec.is_available());
    }

    #[test]
    fn test_take_recordings_resets_buffer() {
        let sensor = MockHalSensor::new("imu", vec![vec![1.0], vec![2.0]]);
        let mut rec = RecordingAdapter::new(Box::new(sensor));
        rec.read_raw();
        rec.read_raw();

        let taken = rec.take_recordings();
        assert_eq!(taken.len(), 2);
        assert_eq!(rec.recording_count(), 0);
    }

    #[test]
    fn test_replay_adapter_plays_back() {
        let recordings = vec![
            SensorRecording {
                name: "imu".into(),
                timestamp_us: 0,
                data: Some(vec![1.0, 2.0]),
            },
            SensorRecording {
                name: "imu".into(),
                timestamp_us: 1000,
                data: None,
            },
            SensorRecording {
                name: "imu".into(),
                timestamp_us: 2000,
                data: Some(vec![5.0, 6.0]),
            },
        ];
        let mut replay = ReplayAdapter::new("imu", recordings);
        assert_eq!(replay.remaining(), 3);

        assert_eq!(replay.read_raw(), Some(vec![1.0, 2.0]));
        assert_eq!(replay.remaining(), 2);

        assert_eq!(replay.read_raw(), None); // replayed None
        assert_eq!(replay.remaining(), 1);

        assert_eq!(replay.read_raw(), Some(vec![5.0, 6.0]));
        assert_eq!(replay.remaining(), 0);

        // Exhausted
        assert_eq!(replay.read_raw(), None);
        assert!(!replay.is_available());
    }

    #[test]
    fn test_record_serialize_deserialize_replay_roundtrip() {
        let sensor = MockHalSensor::new("accel", vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let mut rec = RecordingAdapter::new(Box::new(sensor));
        rec.read_raw();
        rec.read_raw();

        // Serialize to JSON
        let json = serde_json::to_string(rec.recordings()).unwrap();

        // Deserialize
        let deserialized: Vec<SensorRecording> = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.len(), 2);

        // Replay
        let mut replay = ReplayAdapter::new("accel", deserialized);
        assert_eq!(replay.read_raw(), Some(vec![1.0, 2.0]));
        assert_eq!(replay.read_raw(), Some(vec![3.0, 4.0]));
        assert!(replay.read_raw().is_none());
    }

    #[test]
    fn test_sensor_recording_json_fields() {
        let rec = SensorRecording {
            name: "test".into(),
            timestamp_us: 42,
            data: Some(vec![1.0]),
        };
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains("\"name\""));
        assert!(json.contains("\"timestamp_us\""));
        assert!(json.contains("\"data\""));
        assert!(json.contains("42"));
    }
}
