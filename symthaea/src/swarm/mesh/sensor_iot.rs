// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IoT Sensor Adapter — MQTT/CoAP Bridge to Mesh Consciousness
//!
//! Translates IoT sensor readings from common home automation platforms
//! (Zigbee2MQTT, Tasmota, ESPHome) into the mesh `SensorReading` format.
//! Enables physical infrastructure monitoring (water, power, food storage)
//! without centralized cloud dependencies.
//!
//! # Supported Formats
//! - Zigbee2MQTT JSON: `{"temperature": 22.5, "humidity": 45}`
//! - Tasmota JSON: `{"ENERGY": {"Power": 150, "Voltage": 230}}`
//! - ESPHome JSON: `{"id": "sensor.temp", "value": 22.5, "state": "22.5°C"}`
//!
//! # Design
//! This is a sync adapter (no tokio) that processes pre-received messages.
//! The actual MQTT/CoAP transport is handled externally (behind `survival-mqtt` feature).

use std::collections::HashMap;

/// Maximum sensor readings buffered before oldest are dropped.
const MAX_BUFFERED_READINGS: usize = 256;

/// A parsed IoT sensor reading.
#[derive(Debug, Clone)]
pub struct IoTReading {
    /// Sensor identifier (e.g., "zigbee2mqtt/living_room/temperature").
    pub sensor_id: String,
    /// Parsed numeric values.
    pub values: HashMap<String, f64>,
    /// Source platform.
    pub platform: IoTPlatform,
    /// Timestamp (Unix seconds).
    pub timestamp_secs: u64,
    /// Raw JSON payload (for debugging).
    pub raw_payload: String,
}

/// Supported IoT platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoTPlatform {
    /// Zigbee2MQTT (zigbee devices via MQTT).
    Zigbee2Mqtt,
    /// Tasmota (ESP8266/ESP32 firmware).
    Tasmota,
    /// ESPHome (ESP device framework).
    EspHome,
    /// Generic JSON with numeric fields.
    Generic,
}

/// Resource type classification from sensor readings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// Water level, flow, quality.
    Water,
    /// Electrical power, voltage, current.
    Power,
    /// Temperature (food storage, environment).
    Temperature,
    /// Humidity (food preservation).
    Humidity,
    /// Gas/air quality.
    AirQuality,
    /// General/unclassified.
    General,
}

impl ResourceType {
    /// Classify from a sensor key name.
    pub fn classify(key: &str) -> Self {
        let lower = key.to_lowercase();
        if lower.contains("water")
            || lower.contains("flow")
            || lower.contains("tank")
            || lower.contains("purity")
            || lower.contains("level")
        {
            Self::Water
        } else if lower.contains("power")
            || lower.contains("voltage")
            || lower.contains("current")
            || lower.contains("energy")
            || lower.contains("watt")
        {
            Self::Power
        } else if lower.contains("temp") {
            Self::Temperature
        } else if lower.contains("humid") {
            Self::Humidity
        } else if lower.contains("co2")
            || lower.contains("air")
            || lower.contains("gas")
            || lower.contains("pm25")
            || lower.contains("voc")
        {
            Self::AirQuality
        } else {
            Self::General
        }
    }
}

/// Resource alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational — normal fluctuation.
    Info,
    /// Warning — approaching threshold.
    Warning,
    /// Critical — immediate attention needed.
    Critical,
    /// Emergency — system failure / danger.
    Emergency,
}

/// A resource alert generated from sensor anomaly detection.
#[derive(Debug, Clone)]
pub struct ResourceAlert {
    /// Sensor that triggered the alert.
    pub sensor_id: String,
    /// Resource type.
    pub resource_type: ResourceType,
    /// Alert severity.
    pub severity: AlertSeverity,
    /// Human-readable description.
    pub description: String,
    /// Alert timestamp.
    pub timestamp_secs: u64,
    /// Value that triggered the alert.
    pub trigger_value: f64,
    /// Threshold that was exceeded.
    pub threshold: f64,
}

/// Alert thresholds for different resource types.
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Water level below this triggers warning (fraction 0-1).
    pub water_low_warning: f64,
    /// Water level below this triggers critical.
    pub water_low_critical: f64,
    /// Power above this triggers warning (watts).
    pub power_high_warning: f64,
    /// Power above this triggers critical.
    pub power_high_critical: f64,
    /// Temperature above this triggers warning (°C).
    pub temp_high_warning: f64,
    /// Temperature below this triggers warning.
    pub temp_low_warning: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            water_low_warning: 0.3,
            water_low_critical: 0.1,
            power_high_warning: 3000.0,
            power_high_critical: 5000.0,
            temp_high_warning: 35.0,
            temp_low_warning: 5.0,
        }
    }
}

/// IoT sensor adapter that processes incoming messages.
pub struct IoTSensorAdapter {
    /// Buffered readings waiting to be processed.
    readings: Vec<IoTReading>,
    /// Alert thresholds.
    thresholds: AlertThresholds,
    /// Generated alerts (ring buffer).
    alerts: Vec<ResourceAlert>,
    /// Maximum alerts retained.
    max_alerts: usize,
    /// Sensor value history for anomaly detection (sensor_id → recent values).
    history: HashMap<String, Vec<f64>>,
    /// Maximum history per sensor.
    max_history_per_sensor: usize,
    /// Total readings processed.
    total_processed: u64,
}

impl IoTSensorAdapter {
    /// Create a new adapter with default thresholds.
    pub fn new() -> Self {
        Self {
            readings: Vec::new(),
            thresholds: AlertThresholds::default(),
            alerts: Vec::new(),
            max_alerts: 64,
            history: HashMap::new(),
            max_history_per_sensor: 100,
            total_processed: 0,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(thresholds: AlertThresholds) -> Self {
        let mut adapter = Self::new();
        adapter.thresholds = thresholds;
        adapter
    }

    /// Parse a JSON message from an IoT platform.
    pub fn parse_message(
        &self,
        topic: &str,
        payload: &str,
        timestamp_secs: u64,
    ) -> Option<IoTReading> {
        let parsed: serde_json::Value = serde_json::from_str(payload).ok()?;
        let obj = parsed.as_object()?;

        let platform = self.detect_platform(topic, obj);
        let mut values = HashMap::new();

        match platform {
            IoTPlatform::Tasmota => {
                // Tasmota nests under "ENERGY", "BME280", etc.
                for (_, section) in obj {
                    if let Some(section_obj) = section.as_object() {
                        for (key, val) in section_obj {
                            if let Some(n) = val.as_f64() {
                                values.insert(key.clone(), n);
                            }
                        }
                    }
                }
                // Also capture top-level numerics
                for (key, val) in obj {
                    if let Some(n) = val.as_f64() {
                        values.insert(key.clone(), n);
                    }
                }
            }
            IoTPlatform::EspHome => {
                if let Some(val) = obj.get("value").and_then(|v| v.as_f64()) {
                    let id = obj.get("id").and_then(|v| v.as_str()).unwrap_or("value");
                    values.insert(id.to_string(), val);
                }
            }
            _ => {
                // Zigbee2MQTT and Generic: all top-level numeric fields
                for (key, val) in obj {
                    if let Some(n) = val.as_f64() {
                        values.insert(key.clone(), n);
                    }
                }
            }
        }

        if values.is_empty() {
            return None;
        }

        Some(IoTReading {
            sensor_id: topic.to_string(),
            values,
            platform,
            timestamp_secs,
            raw_payload: payload.to_string(),
        })
    }

    /// Ingest a reading and check for alerts.
    pub fn ingest(&mut self, reading: IoTReading) -> Vec<ResourceAlert> {
        let mut new_alerts = Vec::new();

        for (key, &value) in &reading.values {
            let resource_type = ResourceType::classify(key);

            // Update history
            let hist = self
                .history
                .entry(format!("{}:{}", reading.sensor_id, key))
                .or_default();
            if hist.len() >= self.max_history_per_sensor {
                hist.remove(0);
            }
            hist.push(value);

            // Check thresholds
            if let Some(alert) = self.check_threshold(
                &reading.sensor_id,
                resource_type,
                key,
                value,
                reading.timestamp_secs,
            ) {
                new_alerts.push(alert);
            }
        }

        // Buffer the reading
        if self.readings.len() >= MAX_BUFFERED_READINGS {
            self.readings.remove(0);
        }
        self.readings.push(reading);
        self.total_processed += 1;

        // Buffer alerts
        for alert in &new_alerts {
            if self.alerts.len() >= self.max_alerts {
                self.alerts.remove(0);
            }
            self.alerts.push(alert.clone());
        }

        new_alerts
    }

    /// Drain buffered readings.
    pub fn drain_readings(&mut self) -> Vec<IoTReading> {
        std::mem::take(&mut self.readings)
    }

    /// Get recent alerts.
    pub fn alerts(&self) -> &[ResourceAlert] {
        &self.alerts
    }

    /// Total readings processed.
    pub fn total_processed(&self) -> u64 {
        self.total_processed
    }

    /// Number of unique sensors seen.
    pub fn sensor_count(&self) -> usize {
        self.history.len()
    }

    fn detect_platform(
        &self,
        topic: &str,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> IoTPlatform {
        if topic.starts_with("zigbee2mqtt/") {
            IoTPlatform::Zigbee2Mqtt
        } else if obj.contains_key("ENERGY")
            || obj.contains_key("BME280")
            || topic.starts_with("tele/")
        {
            IoTPlatform::Tasmota
        } else if obj.contains_key("id") && obj.contains_key("value") {
            IoTPlatform::EspHome
        } else {
            IoTPlatform::Generic
        }
    }

    fn check_threshold(
        &self,
        sensor_id: &str,
        resource_type: ResourceType,
        _key: &str,
        value: f64,
        timestamp: u64,
    ) -> Option<ResourceAlert> {
        let (severity, threshold, description) = match resource_type {
            ResourceType::Water => {
                if value < self.thresholds.water_low_critical {
                    (
                        AlertSeverity::Critical,
                        self.thresholds.water_low_critical,
                        format!("Water level critically low: {:.1}%", value * 100.0),
                    )
                } else if value < self.thresholds.water_low_warning {
                    (
                        AlertSeverity::Warning,
                        self.thresholds.water_low_warning,
                        format!("Water level low: {:.1}%", value * 100.0),
                    )
                } else {
                    return None;
                }
            }
            ResourceType::Power => {
                if value > self.thresholds.power_high_critical {
                    (
                        AlertSeverity::Critical,
                        self.thresholds.power_high_critical,
                        format!("Power consumption critical: {:.0}W", value),
                    )
                } else if value > self.thresholds.power_high_warning {
                    (
                        AlertSeverity::Warning,
                        self.thresholds.power_high_warning,
                        format!("Power consumption high: {:.0}W", value),
                    )
                } else {
                    return None;
                }
            }
            ResourceType::Temperature => {
                if value > self.thresholds.temp_high_warning {
                    (
                        AlertSeverity::Warning,
                        self.thresholds.temp_high_warning,
                        format!("High temperature: {:.1}°C", value),
                    )
                } else if value < self.thresholds.temp_low_warning {
                    (
                        AlertSeverity::Warning,
                        self.thresholds.temp_low_warning,
                        format!("Low temperature: {:.1}°C", value),
                    )
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        Some(ResourceAlert {
            sensor_id: sensor_id.to_string(),
            resource_type,
            severity,
            description,
            timestamp_secs: timestamp,
            trigger_value: value,
            threshold,
        })
    }
}

impl Default for IoTSensorAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_zigbee2mqtt() {
        let adapter = IoTSensorAdapter::new();
        let reading = adapter
            .parse_message(
                "zigbee2mqtt/living_room",
                r#"{"temperature": 22.5, "humidity": 45, "battery": 80}"#,
                1000,
            )
            .unwrap();
        assert_eq!(reading.platform, IoTPlatform::Zigbee2Mqtt);
        assert!((reading.values["temperature"] - 22.5).abs() < 0.01);
        assert!((reading.values["humidity"] - 45.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_tasmota() {
        let adapter = IoTSensorAdapter::new();
        let reading = adapter
            .parse_message(
                "tele/plug1/SENSOR",
                r#"{"ENERGY": {"Power": 150, "Voltage": 230, "Current": 0.65}}"#,
                1000,
            )
            .unwrap();
        assert_eq!(reading.platform, IoTPlatform::Tasmota);
        assert!((reading.values["Power"] - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_esphome() {
        let adapter = IoTSensorAdapter::new();
        let reading = adapter
            .parse_message(
                "esphome/sensor",
                r#"{"id": "sensor.temp", "value": 22.5, "state": "22.5°C"}"#,
                1000,
            )
            .unwrap();
        assert_eq!(reading.platform, IoTPlatform::EspHome);
        assert!((reading.values["sensor.temp"] - 22.5).abs() < 0.01);
    }

    #[test]
    fn test_water_alert() {
        let mut adapter = IoTSensorAdapter::new();
        let reading = IoTReading {
            sensor_id: "tank/water_level".to_string(),
            values: HashMap::from([("water_level".to_string(), 0.05)]),
            platform: IoTPlatform::Generic,
            timestamp_secs: 1000,
            raw_payload: String::new(),
        };
        let alerts = adapter.ingest(reading);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);
        assert_eq!(alerts[0].resource_type, ResourceType::Water);
    }

    #[test]
    fn test_power_alert() {
        let mut adapter = IoTSensorAdapter::new();
        let reading = IoTReading {
            sensor_id: "grid/power".to_string(),
            values: HashMap::from([("power".to_string(), 4000.0)]),
            platform: IoTPlatform::Generic,
            timestamp_secs: 1000,
            raw_payload: String::new(),
        };
        let alerts = adapter.ingest(reading);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_no_alert_normal_values() {
        let mut adapter = IoTSensorAdapter::new();
        let reading = IoTReading {
            sensor_id: "room/temp".to_string(),
            values: HashMap::from([("temperature".to_string(), 22.0)]),
            platform: IoTPlatform::Generic,
            timestamp_secs: 1000,
            raw_payload: String::new(),
        };
        let alerts = adapter.ingest(reading);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_resource_type_classification() {
        assert_eq!(ResourceType::classify("water_level"), ResourceType::Water);
        assert_eq!(ResourceType::classify("Power"), ResourceType::Power);
        assert_eq!(
            ResourceType::classify("temperature"),
            ResourceType::Temperature
        );
        assert_eq!(ResourceType::classify("humidity"), ResourceType::Humidity);
        assert_eq!(ResourceType::classify("co2"), ResourceType::AirQuality);
        assert_eq!(ResourceType::classify("random"), ResourceType::General);
    }

    #[test]
    fn test_drain_readings() {
        let mut adapter = IoTSensorAdapter::new();
        adapter.ingest(IoTReading {
            sensor_id: "s1".to_string(),
            values: HashMap::from([("x".to_string(), 1.0)]),
            platform: IoTPlatform::Generic,
            timestamp_secs: 0,
            raw_payload: String::new(),
        });
        let drained = adapter.drain_readings();
        assert_eq!(drained.len(), 1);
        assert!(adapter.drain_readings().is_empty());
    }

    #[test]
    fn test_sensor_count() {
        let mut adapter = IoTSensorAdapter::new();
        adapter.ingest(IoTReading {
            sensor_id: "s1".to_string(),
            values: HashMap::from([("temp".to_string(), 20.0), ("humidity".to_string(), 50.0)]),
            platform: IoTPlatform::Generic,
            timestamp_secs: 0,
            raw_payload: String::new(),
        });
        assert_eq!(adapter.sensor_count(), 2); // s1:temp + s1:humidity
    }

    #[test]
    fn test_default() {
        let adapter = IoTSensorAdapter::default();
        assert_eq!(adapter.total_processed(), 0);
    }

    #[test]
    fn test_invalid_json() {
        let adapter = IoTSensorAdapter::new();
        assert!(adapter.parse_message("topic", "not json", 0).is_none());
    }

    #[test]
    fn test_empty_values_rejected() {
        let adapter = IoTSensorAdapter::new();
        assert!(
            adapter
                .parse_message("topic", r#"{"text": "hello"}"#, 0)
                .is_none()
        );
    }
}
