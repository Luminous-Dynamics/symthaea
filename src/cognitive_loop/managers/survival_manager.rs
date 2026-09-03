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
//! # IoT trust boundary
//!
//! Parsed MQTT/IoT input is **untrusted telemetry**. It may be retained for
//! observability, but it cannot directly mutate authoritative resource state,
//! forecasting, emergency state, or neuromodulation. Only a
//! [`VerifiedIoTReading`] minted by a crate-owned trusted ingress boundary may
//! enter those paths.
//!
//! Verified and untrusted telemetry use physically separate [`IoTSensorAdapter`]
//! instances. Unauthenticated sensor-series churn therefore cannot poison or fill
//! the history/buffer state used to interpret verified readings.
//!
//! `VerifiedIoTReading` is not itself cryptographic proof. Its constructor is
//! crate-private so a future Xenia/HAL ingress adapter can mint it only after
//! source authentication and device-appropriate evidence verification. External
//! callers cannot self-label arbitrary parser output as verified.
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
    AlertSeverity, IoTReading, IoTSensorAdapter, MAX_IOT_FIELD_NAME_BYTES,
    MAX_IOT_NUMERIC_FIELDS, MAX_IOT_PAYLOAD_BYTES, MAX_IOT_TOPIC_BYTES, ResourceAlert,
    ResourceType,
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
    /// Number of verified sensor series retained by the authoritative adapter.
    pub sensor_count: usize,
    /// Number of untrusted sensor series retained for observation-only diagnostics.
    pub untrusted_sensor_count: usize,
    /// Number of untrusted readings rejected by the parser/ingest resource envelope.
    pub untrusted_rejected_readings: u64,
    /// Number of active authoritative alerts.
    pub alert_count: usize,
    /// Number of active forecasts derived from verified telemetry.
    pub forecast_count: usize,
}

/// IoT reading admitted by a crate-owned trusted ingress boundary.
///
/// This wrapper intentionally has no public constructor. A transport/parser cannot
/// upgrade its own bytes to authoritative evidence merely by setting a flag.
#[derive(Debug, Clone)]
pub struct VerifiedIoTReading {
    reading: IoTReading,
    source_principal: String,
    evidence_digest: [u8; 32],
    sequence: u64,
}

impl VerifiedIoTReading {
    /// Construct a verified reading after an owning ingress layer has completed its
    /// authentication/attestation checks.
    ///
    /// This function is crate-private by design. It does not perform cryptography;
    /// the caller owns that proof obligation. The reading must also satisfy the
    /// same basic allocation/value bounds enforced at the parser boundary.
    pub(crate) fn from_trusted_ingress(
        reading: IoTReading,
        source_principal: impl Into<String>,
        evidence_digest: [u8; 32],
        sequence: u64,
    ) -> Option<Self> {
        let source_principal = source_principal.into();
        if source_principal.is_empty()
            || evidence_digest == [0; 32]
            || sequence == 0
            || !reading_within_ingress_bounds(&reading)
        {
            return None;
        }
        Some(Self {
            reading,
            source_principal,
            evidence_digest,
            sequence,
        })
    }

    /// Original parsed reading.
    pub fn reading(&self) -> &IoTReading {
        &self.reading
    }

    /// Authenticated/attested source identity supplied by the trusted ingress.
    pub fn source_principal(&self) -> &str {
        &self.source_principal
    }

    /// Commitment to the external verification evidence.
    pub fn evidence_digest(&self) -> [u8; 32] {
        self.evidence_digest
    }

    /// Monotonic sequence in the verified source domain.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

/// A survival event from the sensor network.
#[derive(Debug, Clone)]
pub enum SurvivalEvent {
    /// Parser-originated IoT telemetry. Observation-only and non-authoritative.
    SensorReading(IoTReading),
    /// IoT telemetry admitted by a crate-owned trusted ingress boundary.
    VerifiedSensorReading(VerifiedIoTReading),
    /// Emergency declared by an explicit higher-authority path.
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
    /// Adapter state for verified telemetry only.
    verified_sensor_adapter: IoTSensorAdapter,
    /// Physically separate adapter state for untrusted observation-only telemetry.
    untrusted_sensor_adapter: IoTSensorAdapter,
    /// Demand forecaster. Only verified telemetry is admitted.
    forecaster: DemandForecaster,
    /// Pending events.
    pending_events: Vec<SurvivalEvent>,
    /// Whether enabled.
    enabled: bool,
    /// Current authoritative resource levels.
    resource_levels: std::collections::HashMap<String, f64>,
    /// Whether an emergency is active.
    emergency_active: bool,
    /// Recent authoritative alerts from verified sensors.
    recent_alerts: Vec<ResourceAlert>,
    /// Recent observation-only alerts from untrusted parser input.
    untrusted_alerts: Vec<ResourceAlert>,
    /// Highest admitted sequence per verified source+sensor domain.
    verified_sequences: std::collections::HashMap<String, u64>,
    /// Last telemetry snapshot.
    last_telemetry: SurvivalTelemetry,
}

impl SurvivalManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 47;

    /// Create a new SurvivalManager.
    pub fn new(enabled: bool) -> Self {
        Self {
            verified_sensor_adapter: IoTSensorAdapter::new(),
            untrusted_sensor_adapter: IoTSensorAdapter::new(),
            forecaster: DemandForecaster::new(),
            pending_events: Vec::new(),
            enabled,
            resource_levels: std::collections::HashMap::new(),
            emergency_active: false,
            recent_alerts: Vec::new(),
            untrusted_alerts: Vec::new(),
            verified_sequences: std::collections::HashMap::new(),
            last_telemetry: SurvivalTelemetry::default(),
        }
    }

    /// Inject a survival event.
    pub fn inject_event(&mut self, event: SurvivalEvent) {
        if self.enabled {
            self.pending_events.push(event);
        }
    }

    /// Access the authoritative/verified sensor adapter.
    ///
    /// Parsing is stateless and may still be performed through this reference, but
    /// only `VerifiedSensorReading` events can mutate this adapter's retained state.
    pub fn sensor_adapter(&self) -> &IoTSensorAdapter {
        &self.verified_sensor_adapter
    }

    /// Access observation-only untrusted sensor state.
    pub fn untrusted_sensor_adapter(&self) -> &IoTSensorAdapter {
        &self.untrusted_sensor_adapter
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

    /// Get recent authoritative alerts from verified telemetry.
    pub fn recent_alerts(&self) -> &[ResourceAlert] {
        &self.recent_alerts
    }

    /// Get recent observation-only alerts from untrusted telemetry.
    pub fn untrusted_alerts(&self) -> &[ResourceAlert] {
        &self.untrusted_alerts
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

    fn ingest_verified_reading(&mut self, verified: VerifiedIoTReading) {
        let replay_domain = format!(
            "{}\0{}",
            verified.source_principal, verified.reading.sensor_id
        );
        if self
            .verified_sequences
            .get(&replay_domain)
            .is_some_and(|last| verified.sequence <= *last)
        {
            return;
        }

        // The trusted-ingress constructor already enforces the basic resource
        // envelope. Advance replay state before any authoritative cognition so a
        // duplicate cannot retrigger those effects.
        self.verified_sequences
            .insert(replay_domain, verified.sequence);

        // Only verified telemetry may influence forecasting/resource state.
        let hour = chrono::Utc::now().hour() as usize;
        for (key, &value) in &verified.reading.values {
            let resource_type = ResourceType::classify(key);
            let type_name = format!("{:?}", resource_type).to_lowercase();
            self.forecaster.record_consumption(&type_name, value, hour);
            self.resource_levels.insert(type_name, value);
        }

        let alerts = self.verified_sensor_adapter.ingest(verified.reading);
        for alert in &alerts {
            if alert.severity >= AlertSeverity::Critical {
                self.emergency_active = true;
            }
        }
        self.recent_alerts.extend(alerts);
    }

    fn process_events(&mut self) {
        self.recent_alerts.clear();
        self.untrusted_alerts.clear();
        let events = std::mem::take(&mut self.pending_events);
        for event in events {
            match event {
                SurvivalEvent::SensorReading(reading) => {
                    // Parser-originated data remains useful for observability, but it
                    // cannot mutate verified adapter history, forecasts, resources,
                    // emergency state, or neuromodulation.
                    let alerts = self.untrusted_sensor_adapter.ingest(reading);
                    self.untrusted_alerts.extend(alerts);
                }
                SurvivalEvent::VerifiedSensorReading(verified) => {
                    self.ingest_verified_reading(verified);
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
                    // Community sharing — update levels through its separate explicit path.
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
            sensor_count: self.verified_sensor_adapter.sensor_count(),
            untrusted_sensor_count: self.untrusted_sensor_adapter.sensor_count(),
            untrusted_rejected_readings: self.untrusted_sensor_adapter.rejected_readings(),
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

        // Neuromod: critical *verified* alerts → arousal spike.
        let critical_count = self
            .recent_alerts
            .iter()
            .filter(|a| a.severity >= AlertSeverity::Critical)
            .count();
        if critical_count > 0 {
            output.arousal_delta += (critical_count as f64 * 0.02).min(0.08) as f32;
        }

        // Neuromod: verified resource scarcity → stress.
        for &level in self.resource_levels.values() {
            if level < SCARCITY_THRESHOLD && level > 0.0 {
                output.arousal_delta += 0.02;
                output.valence_delta -= 0.01;
            }
        }

        // Neuromod: stable verified resources → calm.
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
        // Note: resource_levels, verified/untrusted adapter state, forecaster, and
        // verified sequence state contain complex state not serialized here.
        // Until a durable ingress checkpoint lands, a restart must reacquire
        // fresh verified sequence evidence before consequential use.
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

fn reading_within_ingress_bounds(reading: &IoTReading) -> bool {
    !reading.sensor_id.is_empty()
        && reading.sensor_id.len() <= MAX_IOT_TOPIC_BYTES
        && reading.raw_payload.len() <= MAX_IOT_PAYLOAD_BYTES
        && !reading.values.is_empty()
        && reading.values.len() <= MAX_IOT_NUMERIC_FIELDS
        && reading.values.iter().all(|(key, value)| {
            !key.is_empty() && key.len() <= MAX_IOT_FIELD_NAME_BYTES && value.is_finite()
        })
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;
    use crate::swarm::mesh::sensor_iot::MAX_TRACKED_SENSOR_SERIES;
    use std::collections::HashMap;

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot::default()
    }

    fn critical_water_reading() -> IoTReading {
        IoTReading {
            sensor_id: "tank".to_string(),
            values: HashMap::from([("water_level".to_string(), 0.05)]),
            platform: crate::swarm::mesh::sensor_iot::IoTPlatform::Generic,
            timestamp_secs: 1000,
            raw_payload: String::new(),
        }
    }

    fn verified_water_reading(sequence: u64) -> VerifiedIoTReading {
        VerifiedIoTReading::from_trusted_ingress(
            critical_water_reading(),
            "device:tank-controller",
            [0xA5; 32],
            sequence,
        )
        .unwrap()
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
    fn untrusted_sensor_reading_is_observation_only() {
        let mut mgr = SurvivalManager::new(true);
        mgr.inject_event(SurvivalEvent::SensorReading(critical_water_reading()));
        let output = mgr.process(&default_snapshot());

        assert!(mgr.recent_alerts().is_empty());
        assert!(!mgr.untrusted_alerts().is_empty());
        assert!(!mgr.is_emergency());
        assert_eq!(mgr.telemetry().water_pct, 0.0);
        assert_eq!(mgr.telemetry().forecast_count, 0);
        assert_eq!(mgr.telemetry().sensor_count, 0);
        assert_eq!(mgr.telemetry().untrusted_sensor_count, 1);
        assert_eq!(output.arousal_delta, 0.0);
    }

    #[test]
    fn untrusted_series_churn_cannot_fill_verified_history() {
        let mut mgr = SurvivalManager::new(true);
        for i in 0..=MAX_TRACKED_SENSOR_SERIES {
            mgr.inject_event(SurvivalEvent::SensorReading(IoTReading {
                sensor_id: format!("untrusted-{i}"),
                values: HashMap::from([("x".to_string(), i as f64)]),
                platform: crate::swarm::mesh::sensor_iot::IoTPlatform::Generic,
                timestamp_secs: i as u64,
                raw_payload: String::new(),
            }));
        }
        mgr.process(&default_snapshot());

        assert_eq!(mgr.sensor_adapter().sensor_count(), 0);
        assert_eq!(
            mgr.untrusted_sensor_adapter().sensor_count(),
            MAX_TRACKED_SENSOR_SERIES
        );
        assert_eq!(mgr.untrusted_sensor_adapter().rejected_readings(), 1);
        assert!(!mgr.is_emergency());

        mgr.inject_event(SurvivalEvent::VerifiedSensorReading(
            verified_water_reading(1),
        ));
        mgr.process(&default_snapshot());
        assert_eq!(mgr.sensor_adapter().sensor_count(), 1);
        assert!(mgr.is_emergency());
    }

    #[test]
    fn verified_sensor_reading_can_drive_emergency_state() {
        let mut mgr = SurvivalManager::new(true);
        mgr.inject_event(SurvivalEvent::VerifiedSensorReading(
            verified_water_reading(1),
        ));
        let output = mgr.process(&default_snapshot());

        assert!(!mgr.recent_alerts().is_empty());
        assert!(mgr.untrusted_alerts().is_empty());
        assert!(mgr.is_emergency());
        assert!(mgr.telemetry().water_pct > 0.0);
        assert!(mgr.telemetry().forecast_count > 0);
        assert_eq!(mgr.telemetry().sensor_count, 1);
        assert_eq!(mgr.telemetry().untrusted_sensor_count, 0);
        assert!(output.arousal_delta > 0.0);
    }

    #[test]
    fn verified_telemetry_replay_does_not_retrigger_authority_path() {
        let mut mgr = SurvivalManager::new(true);
        mgr.inject_event(SurvivalEvent::VerifiedSensorReading(
            verified_water_reading(7),
        ));
        mgr.process(&default_snapshot());
        assert!(mgr.is_emergency());

        mgr.inject_event(SurvivalEvent::EmergencyResolved);
        mgr.process(&default_snapshot());
        assert!(!mgr.is_emergency());

        mgr.inject_event(SurvivalEvent::VerifiedSensorReading(
            verified_water_reading(7),
        ));
        mgr.process(&default_snapshot());
        assert!(!mgr.is_emergency());
        assert!(mgr.recent_alerts().is_empty());
    }

    #[test]
    fn verified_ingress_rejects_empty_or_unbounded_self_asserted_evidence() {
        assert!(
            VerifiedIoTReading::from_trusted_ingress(
                critical_water_reading(),
                "device:tank-controller",
                [0; 32],
                1,
            )
            .is_none()
        );
        assert!(
            VerifiedIoTReading::from_trusted_ingress(
                critical_water_reading(),
                "",
                [1; 32],
                1,
            )
            .is_none()
        );

        let mut oversized = critical_water_reading();
        oversized.sensor_id = "x".repeat(MAX_IOT_TOPIC_BYTES + 1);
        assert!(
            VerifiedIoTReading::from_trusted_ingress(
                oversized,
                "device:tank-controller",
                [1; 32],
                1,
            )
            .is_none()
        );
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
