// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SensorTrait — Physical Nerves for the Sage.
//!
//! Converts raw sensor protocol data (Zigbee, Bluetooth, Matter, GPIO) into
//! HDC [`ContinuousHV`] vectors for the cognitive loop.  Each sensor has an
//! intrinsic urgency class: a smoke alarm demands `Critical` routing while a
//! thermometer drifts in at `Cruise` pace.
//!
//! # Architecture
//!
//! ```text
//!   ┌──────────────┐     poll()     ┌─────────────────┐     encode      ┌──────────────┐
//!   │ Zigbee HAT   │ ──────────────►│ SensorRegistry  │ ──────────────► │ ContinuousHV │
//!   │ BLE Radio    │                │                 │                 │ (perception) │
//!   │ GPIO pins    │                │ SensorInput[]   │                 └──────────────┘
//!   └──────────────┘                └─────────────────┘
//! ```
//!
//! Protocol-specific adapters (Bluetooth, Zigbee, Matter, GPIO) implement
//! [`SensorInput`].  The [`SensorRegistry`] polls all sensors each tick
//! and encodes readings into HDC space for the cognitive loop.

use super::MeshUrgency;
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;

// ============================================================================
// SENSOR TRAIT
// ============================================================================

/// A physical sensor that converts environmental data into HDC-compatible
/// inputs for the cognitive loop.
///
/// Implemented by protocol-specific adapters (Bluetooth RSSI, Zigbee sensors,
/// Matter devices, GPIO, etc.)
pub trait SensorInput: Send + Sync {
    /// Human-readable sensor name (e.g., `"zigbee::smoke_alarm"`, `"ble::presence"`).
    fn name(&self) -> &str;

    /// Poll for new readings.  Returns `None` if no new data.
    fn poll(&mut self) -> Option<SensorReading>;

    /// The default urgency class of this sensor's data.
    ///
    /// Smoke alarm → Critical.  Thermometer → Cruise.  Presence → Normal.
    fn urgency_class(&self) -> MeshUrgency;

    /// Whether this sensor is currently operational.
    fn is_available(&self) -> bool;
}

// ============================================================================
// SENSOR READING
// ============================================================================

/// A single sensor reading, ready for HDC encoding.
#[derive(Debug, Clone)]
pub struct SensorReading {
    /// Sensor identifier (matches [`SensorInput::name()`]).
    pub sensor_id: String,
    /// Raw measurement as f32 values (protocol-specific interpretation).
    pub values: Vec<f32>,
    /// Topic tags for memory routing (e.g., interoception, power).
    pub tags: Vec<String>,
    /// Additional metadata for persistence (stringified values).
    pub metadata: HashMap<String, String>,
    /// Urgency override (`Some` = sensor demands specific urgency,
    /// `None` = use the sensor's default class).
    pub urgency_override: Option<MeshUrgency>,
    /// Unix timestamp of the reading (seconds).
    pub timestamp_s: u32,
}

// ============================================================================
// SENSOR REGISTRY
// ============================================================================

/// Registry of attached sensors, polled each tick.
pub struct SensorRegistry {
    sensors: Vec<Box<dyn SensorInput>>,
    /// HDC encoder dimension (must match ContinuousMind dimension).
    dimension: usize,
}

impl SensorRegistry {
    /// Create a new registry with the given HDC dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            sensors: Vec::new(),
            dimension,
        }
    }

    /// Register a sensor.
    pub fn register(&mut self, sensor: Box<dyn SensorInput>) {
        self.sensors.push(sensor);
    }

    /// Poll all sensors and return readings with their effective urgency.
    pub fn poll_all(&mut self) -> Vec<(SensorReading, MeshUrgency)> {
        let mut results = Vec::new();
        for sensor in &mut self.sensors {
            if !sensor.is_available() {
                continue;
            }
            if let Some(reading) = sensor.poll() {
                let urgency = reading
                    .urgency_override
                    .unwrap_or_else(|| sensor.urgency_class());
                results.push((reading, urgency));
            }
        }
        results
    }

    /// Number of registered sensors.
    pub fn sensor_count(&self) -> usize {
        self.sensors.len()
    }

    /// Names of currently available sensors.
    pub fn available_sensors(&self) -> Vec<&str> {
        self.sensors
            .iter()
            .filter(|s| s.is_available())
            .map(|s| s.name())
            .collect()
    }

    /// Encode a sensor reading into an HDC hypervector.
    ///
    /// Uses position-value encoding: each value index gets a unique basis
    /// vector (deterministic hash of `sensor_id + index`), scaled by the
    /// reading's magnitude.  The result is a [`ContinuousHV`] that the
    /// cognitive loop can directly perceive.
    pub fn encode_reading(&self, reading: &SensorReading) -> ContinuousHV {
        let mut values = vec![0.0f32; self.dimension];

        for (i, &val) in reading.values.iter().enumerate() {
            // Deterministic basis: hash(sensor_id, index) → spread across dim
            let hash = {
                let mut h = 0x9e3779b9u64; // golden ratio fractional
                for b in reading.sensor_id.bytes() {
                    h = h.wrapping_mul(31).wrapping_add(b as u64);
                }
                h = h.wrapping_mul(31).wrapping_add(i as u64);
                h
            };

            // Scatter across multiple dimensions for rich encoding
            for k in 0u64..8 {
                let idx = ((hash.wrapping_mul(k + 1).wrapping_add(k * 0x517cc1b7)) as usize)
                    % self.dimension;
                // Alternate sign based on k parity for orthogonality
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                values[idx] += val * sign;
            }
        }

        // Normalize to unit length
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > f32::EPSILON {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }
}

// ============================================================================
// MOCK SENSOR (TESTING)
// ============================================================================

/// Mock sensor that produces configurable readings (for integration tests).
pub struct MockSensor {
    name: String,
    urgency: MeshUrgency,
    readings: VecDeque<Vec<f32>>,
    available: bool,
}

impl MockSensor {
    /// Create a mock sensor with queued readings.
    pub fn new(name: impl Into<String>, urgency: MeshUrgency, readings: Vec<Vec<f32>>) -> Self {
        Self {
            name: name.into(),
            urgency,
            readings: readings.into(),
            available: true,
        }
    }

    /// Set availability.
    pub fn set_available(&mut self, available: bool) {
        self.available = available;
    }
}

impl SensorInput for MockSensor {
    fn name(&self) -> &str {
        &self.name
    }

    fn poll(&mut self) -> Option<SensorReading> {
        let values = self.readings.pop_front()?;
        Some(SensorReading {
            sensor_id: self.name.clone(),
            values,
            tags: Vec::new(),
            metadata: HashMap::new(),
            urgency_override: None,
            timestamp_s: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as u32,
        })
    }

    fn urgency_class(&self) -> MeshUrgency {
        self.urgency
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

// ============================================================================
// HAL SENSOR BRIDGE (hardware → SensorInput)
// ============================================================================

/// Bridge from a [`HalSensorAdapter`](symthaea_hal::HalSensorAdapter) to [`SensorInput`].
///
/// Wraps any HAL sensor (real hardware or mock) as a `SensorInput` that
/// the [`SensorRegistry`] can poll and encode into HDC space.
#[cfg(feature = "hal")]
pub struct HalSensorBridge<T: symthaea_hal::HalSensorAdapter> {
    adapter: T,
    urgency: MeshUrgency,
}

#[cfg(feature = "hal")]
impl<T: symthaea_hal::HalSensorAdapter> HalSensorBridge<T> {
    /// Create a bridge with a given urgency class.
    pub fn new(adapter: T, urgency: MeshUrgency) -> Self {
        Self { adapter, urgency }
    }
}

#[cfg(feature = "hal")]
impl<T: symthaea_hal::HalSensorAdapter + Sync> SensorInput for HalSensorBridge<T> {
    fn name(&self) -> &str {
        self.adapter.name()
    }

    fn poll(&mut self) -> Option<SensorReading> {
        let values = self.adapter.read_raw()?;
        Some(SensorReading {
            sensor_id: self.adapter.name().to_string(),
            values,
            tags: Vec::new(),
            metadata: HashMap::new(),
            urgency_override: None,
            timestamp_s: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as u32,
        })
    }

    fn urgency_class(&self) -> MeshUrgency {
        self.urgency
    }

    fn is_available(&self) -> bool {
        self.adapter.is_available()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_registry_creation() {
        let mut registry = SensorRegistry::new(512);
        let temp = MockSensor::new("zigbee::temperature", MeshUrgency::Cruise, vec![vec![22.5]]);
        let smoke = MockSensor::new(
            "zigbee::smoke_alarm",
            MeshUrgency::Critical,
            vec![vec![1.0]],
        );
        registry.register(Box::new(temp));
        registry.register(Box::new(smoke));
        assert_eq!(registry.sensor_count(), 2);
        assert_eq!(registry.available_sensors().len(), 2);
    }

    #[test]
    fn test_sensor_poll_and_encode() {
        let mut registry = SensorRegistry::new(512);
        let temp = MockSensor::new(
            "zigbee::temperature",
            MeshUrgency::Cruise,
            vec![vec![22.5, 45.0]],
        );
        registry.register(Box::new(temp));

        let results = registry.poll_all();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, MeshUrgency::Cruise);

        let hv = registry.encode_reading(&results[0].0);
        assert!(
            hv.norm() > 0.9,
            "Encoded HV should be normalized: norm={}",
            hv.norm()
        );
    }

    #[test]
    fn test_sensor_critical_triggers_urgency() {
        let mut registry = SensorRegistry::new(512);
        let smoke = MockSensor::new(
            "zigbee::smoke_alarm",
            MeshUrgency::Critical,
            vec![vec![1.0]],
        );
        registry.register(Box::new(smoke));

        let results = registry.poll_all();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, MeshUrgency::Critical);
    }

    #[test]
    fn test_sensor_encoding_distinct_per_sensor() {
        let registry = SensorRegistry::new(512);

        let reading_a = SensorReading {
            sensor_id: "sensor_a".to_string(),
            values: vec![1.0],
            tags: Vec::new(),
            metadata: HashMap::new(),
            urgency_override: None,
            timestamp_s: 0,
        };
        let reading_b = SensorReading {
            sensor_id: "sensor_b".to_string(),
            values: vec![1.0],
            tags: Vec::new(),
            metadata: HashMap::new(),
            urgency_override: None,
            timestamp_s: 0,
        };

        let hv_a = registry.encode_reading(&reading_a);
        let hv_b = registry.encode_reading(&reading_b);

        // Different sensor IDs with same values should produce different HVs
        let sim = hv_a.similarity(&hv_b);
        assert!(
            sim < 0.9,
            "Different sensors should produce dissimilar HVs: sim={sim}"
        );
    }

    #[test]
    fn test_sensor_poll_exhausts_queue() {
        let mut registry = SensorRegistry::new(512);
        let sensor = MockSensor::new("test", MeshUrgency::Normal, vec![vec![1.0], vec![2.0]]);
        registry.register(Box::new(sensor));

        // First poll: one reading
        let r1 = registry.poll_all();
        assert_eq!(r1.len(), 1);
        assert!((r1[0].0.values[0] - 1.0).abs() < 1e-6);

        // Second poll: next reading
        let r2 = registry.poll_all();
        assert_eq!(r2.len(), 1);
        assert!((r2[0].0.values[0] - 2.0).abs() < 1e-6);

        // Third poll: empty
        let r3 = registry.poll_all();
        assert!(r3.is_empty());
    }

    #[test]
    fn test_sensor_unavailable_skipped() {
        let mut registry = SensorRegistry::new(512);
        let mut sensor = MockSensor::new("offline", MeshUrgency::Normal, vec![vec![1.0]]);
        sensor.set_available(false);
        registry.register(Box::new(sensor));

        let results = registry.poll_all();
        assert!(results.is_empty());
    }
}
