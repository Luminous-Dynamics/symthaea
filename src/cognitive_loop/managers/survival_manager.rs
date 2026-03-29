// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Survival Manager — Infrastructure Monitoring CognitiveSubsystem
//!
//! **Requires**: `feature = "survival"` (implies `mesh`)
//!
//! Aggregates sensor readings from IoT devices, detects anomalies
//! (pipe burst, power outage), and triggers emergency responses.
//!
//! # Science
//! - Maslow, A. (1943). A Theory of Human Motivation — hierarchy of needs
//! - WHO water guidelines — minimum 50L/person/day
//!
//! # Interval
//! 47 — co-prime with {7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 53, 67}

use crate::cognitive_loop::subsystem_trait::{CognitiveSubsystem, CycleSnapshot, SubsystemOutput};
use crate::swarm::mesh::sensor_forecast::{DemandForecast, DemandForecaster};
use crate::swarm::mesh::sensor_iot::{
    AlertSeverity, IoTReading, IoTSensorAdapter, ResourceAlert, ResourceType,
};

/// Neuromodulatory gain: scarcity → NE + cortisol (stress response).
/// Basis: Sapolsky (2004) — resource scarcity triggers HPA axis.
const SCARCITY_NE_GAIN: f64 = 0.06;

/// Neuromodulatory gain: abundance → 5-HT (security).
const ABUNDANCE_5HT_GAIN: f64 = 0.02;

/// Neuromodulatory gain: sharing → oxytocin.
const SHARING_OXY_GAIN: f64 = 0.03;

/// Scarcity threshold (fraction of normal resource level).
const SCARCITY_THRESHOLD: f64 = 0.3;

/// Survival telemetry snapshot.
#[derive(Debug, Clone, Default)]
pub struct SurvivalTelemetry {
    /// Water availability (0.0-1.0 fraction).
    pub water_pct: f64,
    /// Current power consumption (kW).
    pub power_kw: f64,
    /// Estimated food supply (days).
    pub food_days: f64,
    /// Whether an emergency is currently active.
    pub emergency_active: bool,
    /// Number of active sensors.
    pub sensor_count: usize,
    /// Number of active alerts.
    pub alert_count: usize,
    /// Number of active forecasts.
    pub forecast_count: usize,
}

/// A survival event from the sensor network.
#[derive(Debug, Clone)]
pub enum SurvivalEvent {
    /// New IoT sensor reading.
    SensorReading(IoTReading),
    /// Emergency declared.
    EmergencyDeclared { description: String },
    /// Emergency resolved.
    EmergencyResolved,
    /// Resource sharing event (community mutual aid).
    ResourceShared {
        resource_type: String,
        quantity: f64,
    },
}

/// Survival Manager — monitors physical infrastructure via IoT sensors.
pub struct SurvivalManager {
    /// IoT sensor adapter.
    sensor_adapter: IoTSensorAdapter,
    /// Demand forecaster.
    forecaster: DemandForecaster,
    /// Pending events.
    pending_events: Vec<SurvivalEvent>,
    /// Whether enabled.
    enabled: bool,
    /// Current resource levels.
    resource_levels: std::collections::HashMap<String, f64>,
    /// Whether an emergency is active.
    emergency_active: bool,
    /// Recent alerts from sensors.
    recent_alerts: Vec<ResourceAlert>,
    /// Last telemetry snapshot.
    last_telemetry: SurvivalTelemetry,
}

impl SurvivalManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 47;

    /// Create a new SurvivalManager.
    pub fn new(enabled: bool) -> Self {
        Self {
            sensor_adapter: IoTSensorAdapter::new(),
            forecaster: DemandForecaster::new(),
            pending_events: Vec::new(),
            enabled,
            resource_levels: std::collections::HashMap::new(),
            emergency_active: false,
            recent_alerts: Vec::new(),
            last_telemetry: SurvivalTelemetry::default(),
        }
    }

    /// Inject a survival event.
    pub fn inject_event(&mut self, event: SurvivalEvent) {
        if self.enabled {
            self.pending_events.push(event);
        }
    }

    /// Access the IoT adapter.
    pub fn sensor_adapter(&self) -> &IoTSensorAdapter {
        &self.sensor_adapter
    }

    /// Access the forecaster.
    pub fn forecaster(&self) -> &DemandForecaster {
        &self.forecaster
    }

    /// Get last telemetry snapshot.
    pub fn telemetry(&self) -> &SurvivalTelemetry {
        &self.last_telemetry
    }

    /// Whether an emergency is active.
    pub fn is_emergency(&self) -> bool {
        self.emergency_active
    }

    /// Get recent alerts.
    pub fn recent_alerts(&self) -> &[ResourceAlert] {
        &self.recent_alerts
    }

    /// Generate a forecast for a resource type.
    pub fn forecast(&self, resource_type: &str, horizon_hours: u32) -> DemandForecast {
        let hour = chrono::Utc::now().hour() as usize;
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.forecaster
            .forecast(resource_type, horizon_hours, hour, now_secs)
    }

    fn process_events(&mut self) {
        self.recent_alerts.clear();
        for event in self.pending_events.drain(..) {
            match event {
                SurvivalEvent::SensorReading(reading) => {
                    // Feed forecaster
                    let hour = chrono::Utc::now().hour() as usize;
                    for (key, &value) in &reading.values {
                        let resource_type =
                            crate::swarm::mesh::sensor_iot::ResourceType::classify(key);
                        let type_name = format!("{:?}", resource_type).to_lowercase();
                        self.forecaster.record_consumption(&type_name, value, hour);

                        // Update resource levels
                        self.resource_levels.insert(type_name, value);
                    }

                    // Check for alerts
                    let alerts = self.sensor_adapter.ingest(reading);
                    for alert in &alerts {
                        if alert.severity >= AlertSeverity::Critical {
                            self.emergency_active = true;
                        }
                    }
                    self.recent_alerts.extend(alerts);
                }
                SurvivalEvent::EmergencyDeclared { .. } => {
                    self.emergency_active = true;
                }
                SurvivalEvent::EmergencyResolved => {
                    self.emergency_active = false;
                }
                SurvivalEvent::ResourceShared {
                    resource_type,
                    quantity,
                } => {
                    // Community sharing — update levels
                    let current = self
                        .resource_levels
                        .get(&resource_type)
                        .copied()
                        .unwrap_or(0.0);
                    self.resource_levels
                        .insert(resource_type, current + quantity);
                }
            }
        }
    }

    fn update_telemetry(&mut self) {
        self.last_telemetry = SurvivalTelemetry {
            water_pct: self.resource_levels.get("water").copied().unwrap_or(0.0),
            power_kw: self.resource_levels.get("power").copied().unwrap_or(0.0) / 1000.0,
            food_days: self
                .resource_levels
                .get("temperature")
                .copied()
                .unwrap_or(0.0), // Proxy
            emergency_active: self.emergency_active,
            sensor_count: self.sensor_adapter.sensor_count(),
            alert_count: self.recent_alerts.len(),
            forecast_count: self.forecaster.resource_count(),
        };
    }
}

impl CognitiveSubsystem for SurvivalManager {
    fn name(&self) -> &'static str {
        "survival_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, _snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        if !self.enabled {
            return output;
        }

        self.process_events();

        // Neuromod: emergency → high arousal (NE + cortisol)
        if self.emergency_active {
            output.arousal_delta += SCARCITY_NE_GAIN as f32;
            output.valence_delta -= 0.03;
            output.flags |= crate::cognitive_loop::subsystem_trait::output_flags::ESCALATE_URGENCY;
        }

        // Neuromod: critical alerts → arousal spike
        let critical_count = self
            .recent_alerts
            .iter()
            .filter(|a| a.severity >= AlertSeverity::Critical)
            .count();
        if critical_count > 0 {
            output.arousal_delta += (critical_count as f64 * 0.02).min(0.08) as f32;
        }

        // Neuromod: resource scarcity → stress
        for (_, &level) in &self.resource_levels {
            if level < SCARCITY_THRESHOLD && level > 0.0 {
                output.arousal_delta += 0.02;
                output.valence_delta -= 0.01;
            }
        }

        // Neuromod: stable resources → calm
        if !self.emergency_active
            && self.recent_alerts.is_empty()
            && !self.resource_levels.is_empty()
        {
            output.valence_delta += ABUNDANCE_5HT_GAIN as f32;
        }

        self.update_telemetry();
        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Layout: [emergency_active: u8 = 1][enabled: u8 = 1]
        // Total: 2 bytes
        //
        // Note: resource_levels (HashMap), sensor_adapter, and forecaster
        // contain complex state that cannot be cheaply serialized as raw bytes.
        // The emergency flag is the critical state for restart resilience.
        let mut data = Vec::with_capacity(2);
        data.push(self.emergency_active as u8);
        data.push(self.enabled as u8);
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        const MIN_SIZE: usize = 2;
        if data.len() < MIN_SIZE {
            return Err(format!(
                "SurvivalManager checkpoint too short: {} < {}",
                data.len(),
                MIN_SIZE
            ));
        }
        self.emergency_active = data[0] != 0;
        self.enabled = data[1] != 0;
        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;
    use std::collections::HashMap;

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot::default()
    }

    #[test]
    fn test_survival_manager_creation() {
        let mgr = SurvivalManager::new(true);
        assert_eq!(mgr.name(), "survival_manager");
        assert_eq!(mgr.interval(), 47);
    }

    #[test]
    fn test_disabled_returns_neutral() {
        let mut mgr = SurvivalManager::new(false);
        let output = mgr.process(&default_snapshot());
        assert_eq!(output.arousal_delta, 0.0);
    }

    #[test]
    fn test_emergency_triggers_arousal() {
        let mut mgr = SurvivalManager::new(true);
        mgr.inject_event(SurvivalEvent::EmergencyDeclared {
            description: "Power outage".to_string(),
        });
        let output = mgr.process(&default_snapshot());
        assert!(output.arousal_delta > 0.0);
        assert!(mgr.is_emergency());
    }

    #[test]
    fn test_emergency_resolution() {
        let mut mgr = SurvivalManager::new(true);
        mgr.inject_event(SurvivalEvent::EmergencyDeclared {
            description: "test".to_string(),
        });
        mgr.process(&default_snapshot());
        assert!(mgr.is_emergency());

        mgr.inject_event(SurvivalEvent::EmergencyResolved);
        mgr.process(&default_snapshot());
        assert!(!mgr.is_emergency());
    }

    #[test]
    fn test_sensor_reading_processing() {
        let mut mgr = SurvivalManager::new(true);
        mgr.inject_event(SurvivalEvent::SensorReading(IoTReading {
            sensor_id: "tank".to_string(),
            values: HashMap::from([("water_level".to_string(), 0.05)]),
            platform: crate::swarm::mesh::sensor_iot::IoTPlatform::Generic,
            timestamp_secs: 1000,
            raw_payload: String::new(),
        }));
        mgr.process(&default_snapshot());
        assert!(!mgr.recent_alerts().is_empty());
    }

    #[test]
    fn test_resource_sharing() {
        let mut mgr = SurvivalManager::new(true);
        mgr.inject_event(SurvivalEvent::ResourceShared {
            resource_type: "water".to_string(),
            quantity: 50.0,
        });
        mgr.process(&default_snapshot());
        let telem = mgr.telemetry();
        assert!(telem.water_pct > 0.0);
    }

    #[test]
    fn test_telemetry_updates() {
        let mut mgr = SurvivalManager::new(true);
        mgr.process(&default_snapshot());
        let t = mgr.telemetry();
        assert!(!t.emergency_active);
    }

    #[test]
    fn test_disabled_ignores_events() {
        let mut mgr = SurvivalManager::new(false);
        mgr.inject_event(SurvivalEvent::EmergencyDeclared {
            description: "test".to_string(),
        });
        mgr.process(&default_snapshot());
        assert!(!mgr.is_emergency());
    }
}
