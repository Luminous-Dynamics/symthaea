// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Positioning bridge: decentralized cooperative positioning for mobile.
//!
//! Integrates the `positioning` library's EKF with phone sensors
//! (GPS, BLE RSSI, UWB ToF, WiFi RTT) to provide GPS-independent
//! positioning that works on Earth, Moon, and Mars.
//!
//! # Modes
//!
//! - **GpsPrimary**: GPS available — phone acts as anchor for others via BLE mesh
//! - **CooperativeMesh**: GPS denied — trilateration from peer BLE/UWB ranges only
//! - **Hybrid**: GPS + peer ranging fused for maximum accuracy
//! - **Offline**: No connectivity — dead reckoning from last known position
//!
//! # Integration with Consciousness
//!
//! Position uncertainty maps to neuromodulation:
//! - High uncertainty → norepinephrine (heightened alertness)
//! - Novel location → dopamine (exploration reward)
//! - Known safe location → serotonin (calm)

use positioning::{
    CelestialBody, Earth, FilterConfig, GaussianEstimate3D, Measurement, MeasurementProvenance,
    PdrConfig, PedestrianDeadReckoning, PeerEstimate3D, PositionFilter, PublishableEstimate3D,
    RangeEstimate, ReferenceFrame, gdop,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Positioning bridge configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositioningConfig {
    /// Maximum age of a GPS fix before it's considered stale (seconds).
    pub gps_max_age_s: f64,
    /// BLE RSSI reference power at 1m (dBm). Phone-specific calibration.
    pub ble_ref_rssi_dbm: f64,
    /// BLE path loss exponent (2.0=free space, 3.0=typical indoor).
    pub ble_path_loss_exponent: f64,
    /// Minimum anchors required before attempting trilateration.
    pub min_anchors: usize,
    /// EKF process noise for position (m²/s).
    pub ekf_position_noise: f64,
    /// Whether to broadcast position as BLE anchor when GPS is available.
    pub broadcast_as_anchor: bool,
}

impl Default for PositioningConfig {
    fn default() -> Self {
        Self {
            gps_max_age_s: 30.0,
            ble_ref_rssi_dbm: -50.0,
            ble_path_loss_exponent: 2.5,
            min_anchors: 3,
            ekf_position_noise: 0.5,
            broadcast_as_anchor: true,
        }
    }
}

// ============================================================================
// POSITIONING MODE
// ============================================================================

/// Current positioning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositioningMode {
    /// GPS available — phone acts as anchor for others.
    GpsPrimary = 0,
    /// GPS denied — using peer-to-peer ranging only.
    CooperativeMesh = 1,
    /// GPS + peer ranging fused together.
    Hybrid = 2,
    /// No connectivity — dead reckoning from last known.
    Offline = 3,
    /// Dead reckoning from IMU (no GPS, no peers).
    DeadReckoning = 4,
    /// Not enough data to determine position.
    Initializing = 5,
}

// ============================================================================
// ANCHOR ENTRY
// ============================================================================

/// A nearby anchor node (another phone or fixed beacon with known position).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorEntry {
    pub peer_id: String,
    /// Position in ECEF meters.
    pub position_ecef: [f64; 3],
    /// Position accuracy (1-sigma meters).
    pub accuracy_m: f64,
    /// Trust level (0.0-1.0). GPS-anchored phones get high trust.
    pub trust: f64,
    /// Last update tick.
    pub last_seen_tick: u64,
}

/// A GPS reading from the phone's built-in receiver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpsReading {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f64,
    pub accuracy_m: f32,
    pub timestamp_s: f64,
}

// ============================================================================
// POSITIONING BRIDGE
// ============================================================================

/// Mobile positioning engine.
pub struct PositioningBridge {
    config: PositioningConfig,
    filter: PositionFilter,
    pdr: PedestrianDeadReckoning,
    /// Last known ECEF position for PDR origin.
    pdr_origin_ecef: Option<[f64; 3]>,
    anchors: Vec<AnchorEntry>,
    last_gps: Option<GpsReading>,
    gps_available: bool,
    mode: PositioningMode,
    current_tick: u64,
    body: Earth,
    /// Cached GDOP for current anchor geometry.
    cached_gdop: f64,
    /// Total position updates received.
    update_count: u64,
    /// Shared measurement stream for downstream all-domain routing.
    pending_measurements: Vec<Measurement>,
}

impl PublishableEstimate3D for PositioningBridge {
    fn position_mean_m(&self) -> [f64; 3] {
        self.filter.position()
    }

    fn position_sigma_m(&self) -> f64 {
        self.filter.position_sigma()
    }
}

impl PositioningBridge {
    /// Create a new positioning bridge.
    pub fn new(config: PositioningConfig) -> Self {
        let filter_config = FilterConfig {
            position_noise: config.ekf_position_noise,
            velocity_noise: 0.01,
            max_covariance: 1e8,
            innovation_gate_sigma: 4.0,
        };
        Self {
            config,
            filter: PositionFilter::new([0.0, 0.0, 0.0], 10_000.0, filter_config),
            pdr: PedestrianDeadReckoning::new(PdrConfig::default()),
            pdr_origin_ecef: None,
            anchors: Vec::new(),
            last_gps: None,
            gps_available: false,
            mode: PositioningMode::Initializing,
            current_tick: 0,
            body: Earth,
            cached_gdop: f64::INFINITY,
            update_count: 0,
            pending_measurements: Vec::new(),
        }
    }

    fn covariance_diag3(sigma_m: f64) -> [f64; 9] {
        let var = sigma_m * sigma_m;
        [var, 0.0, 0.0, 0.0, var, 0.0, 0.0, 0.0, var]
    }

    fn push_measurement(&mut self, measurement: Measurement) {
        self.pending_measurements.push(measurement);
    }

    /// Drain mobile-originating measurements in the shared positioning format.
    pub fn drain_measurements(&mut self) -> Vec<Measurement> {
        std::mem::take(&mut self.pending_measurements)
    }

    /// Feed a GPS reading from the phone's receiver.
    pub fn update_gps(&mut self, lat: f64, lon: f64, alt: f64, accuracy_m: f32) {
        let ecef = self.body.geodetic_to_cartesian(lat, lon, alt);
        let timestamp_s = self.current_tick as f64;
        self.push_measurement(Measurement::absolute_position(
            timestamp_s,
            ReferenceFrame::Ecef,
            ecef,
            Self::covariance_diag3(accuracy_m as f64),
            MeasurementProvenance::local("soma_gps", "gnss"),
        ));
        self.filter.update(
            &ecef, // Treat GPS as a "zero-range" measurement to the true position
            0.0,   // range = 0 (GPS gives position directly)
            accuracy_m as f64,
            1.0, // Full trust in GPS
        );
        // Actually, GPS gives position directly — just set the filter state
        self.filter.state.state[0] = ecef[0];
        self.filter.state.state[1] = ecef[1];
        self.filter.state.state[2] = ecef[2];
        // Set covariance from GPS accuracy
        let var = (accuracy_m as f64) * (accuracy_m as f64);
        self.filter.state.covariance[0] = var;
        self.filter.state.covariance[7] = var;
        self.filter.state.covariance[14] = var;

        self.last_gps = Some(GpsReading {
            latitude_deg: lat,
            longitude_deg: lon,
            altitude_m: alt,
            accuracy_m,
            timestamp_s,
        });
        self.gps_available = true;
        // Reset PDR origin on GPS fix (drift correction)
        self.pdr_origin_ecef = Some(ecef);
        self.pdr.reset_to_origin();
        self.pdr.set_reference_pressure(1013.25); // Could use actual sea-level pressure
        self.update_mode();
        self.update_count += 1;
    }

    /// Feed a BLE RSSI measurement from a nearby phone/beacon.
    pub fn update_ble_range(&mut self, peer_id: &str, rssi_dbm: i32) {
        let est = positioning::ranging::rssi_to_range(
            rssi_dbm as f64,
            self.config.ble_ref_rssi_dbm,
            self.config.ble_path_loss_exponent,
        );
        self.push_measurement(Measurement::range(
            self.current_tick as f64,
            ReferenceFrame::Ecef,
            est.distance_m,
            est.sigma_m,
            MeasurementProvenance {
                source_id: "soma_ble".to_string(),
                peer_id: Some(peer_id.to_string()),
                technology: "ble_rssi".to_string(),
                admitted: true,
                trust_hint: Some(0.4),
                los_confidence: Some(0.2),
            },
        ));
        self.apply_range(peer_id, est);
    }

    /// Feed a UWB time-of-flight measurement.
    pub fn update_uwb_range(&mut self, peer_id: &str, tof_ns: f64) {
        let est = positioning::ranging::uwb_tof_to_range(tof_ns);
        self.push_measurement(Measurement::range(
            self.current_tick as f64,
            ReferenceFrame::Ecef,
            est.distance_m,
            est.sigma_m,
            MeasurementProvenance {
                source_id: "soma_uwb".to_string(),
                peer_id: Some(peer_id.to_string()),
                technology: "uwb_tof".to_string(),
                admitted: true,
                trust_hint: Some(0.9),
                los_confidence: Some(0.95),
            },
        ));
        self.apply_range(peer_id, est);
    }

    /// Feed a WiFi RTT measurement.
    pub fn update_wifi_rtt(&mut self, ap_id: &str, rtt_ns: f64) {
        let est = positioning::ranging::wifi_rtt_to_range(rtt_ns);
        self.push_measurement(Measurement::range(
            self.current_tick as f64,
            ReferenceFrame::Ecef,
            est.distance_m,
            est.sigma_m,
            MeasurementProvenance {
                source_id: "soma_wifi".to_string(),
                peer_id: Some(ap_id.to_string()),
                technology: "wifi_rtt".to_string(),
                admitted: true,
                trust_hint: Some(0.7),
                los_confidence: Some(0.7),
            },
        ));
        self.apply_range(ap_id, est);
    }

    /// Register or update a nearby anchor.
    pub fn register_anchor(
        &mut self,
        peer_id: String,
        lat: f64,
        lon: f64,
        alt: f64,
        accuracy_m: f64,
        trust: f64,
    ) {
        let ecef = self.body.geodetic_to_cartesian(lat, lon, alt);
        self.push_measurement(Measurement::absolute_position(
            self.current_tick as f64,
            ReferenceFrame::Ecef,
            ecef,
            Self::covariance_diag3(accuracy_m),
            MeasurementProvenance {
                source_id: "soma_anchor_registry".to_string(),
                peer_id: Some(peer_id.clone()),
                technology: "peer_anchor".to_string(),
                admitted: trust >= 0.5,
                trust_hint: Some(trust),
                los_confidence: None,
            },
        ));
        // Update existing or insert new
        if let Some(anchor) = self.anchors.iter_mut().find(|a| a.peer_id == peer_id) {
            anchor.position_ecef = ecef;
            anchor.accuracy_m = accuracy_m;
            anchor.trust = trust;
            anchor.last_seen_tick = self.current_tick;
        } else {
            self.anchors.push(AnchorEntry {
                peer_id,
                position_ecef: ecef,
                accuracy_m,
                trust,
                last_seen_tick: self.current_tick,
            });
        }
        self.update_gdop();
    }

    /// Feed IMU sensor data for dead reckoning.
    /// Call each Soma cycle alongside tick().
    pub fn update_imu(
        &mut self,
        accel_magnitude: f32,
        rotation_rate: f32,
        barometer_hpa: f32,
        step_delta: u32,
        magnetic_heading_deg: Option<f32>,
        magnetic_accuracy_deg: Option<f32>,
        dt_s: f64,
    ) {
        // Always feed PDR (even with GPS — maintains stride calibration)
        self.pdr.step(
            accel_magnitude,
            rotation_rate,
            barometer_hpa,
            step_delta,
            dt_s,
        );

        // Apply magnetometer correction when available
        if let (Some(heading), Some(accuracy)) = (magnetic_heading_deg, magnetic_accuracy_deg) {
            self.pdr.correct_heading(heading, accuracy);
        }

        // Feed barometer to EKF for altitude
        if barometer_hpa > 100.0 {
            let altitude_m =
                positioning::dead_reckoning::barometric_altitude(barometer_hpa as f64, 1013.25);
            self.push_measurement(Measurement::absolute_position(
                self.current_tick as f64,
                ReferenceFrame::Enu,
                [0.0, 0.0, altitude_m],
                Self::covariance_diag3(1.0),
                MeasurementProvenance::local("soma_barometer", "barometer"),
            ));
            self.filter
                .update_barometer(barometer_hpa as f64, 1013.25, 1.0);
        }

        if step_delta > 0 {
            self.push_measurement(Measurement::relative_position(
                self.current_tick as f64,
                ReferenceFrame::VehicleBody,
                [step_delta as f64 * 0.75, 0.0, 0.0],
                Self::covariance_diag3(self.pdr.get_drift_estimate().max(1.0)),
                MeasurementProvenance::local("soma_pdr", "pdr"),
            ));
        }

        // In dead reckoning mode, use PDR to drive the EKF
        if self.mode == PositioningMode::DeadReckoning || self.mode == PositioningMode::Offline {
            if let Some(origin) = self.pdr_origin_ecef {
                let [east, north] = self.pdr.get_position();
                let alt = self.pdr.get_altitude();
                // Convert relative PDR position to ECEF (approximate: add local ENU offset)
                let estimated_ecef = [origin[0] + east, origin[1] + north, origin[2] + alt];
                // PDR sigma grows with drift
                let pdr_sigma = self.pdr.get_drift_estimate().max(2.0);
                // Feed to EKF as a position measurement
                self.filter.state.state[0] = estimated_ecef[0];
                self.filter.state.state[1] = estimated_ecef[1];
                self.filter.state.state[2] = estimated_ecef[2];
                // Grow covariance proportional to drift
                let var = pdr_sigma * pdr_sigma;
                self.filter.state.covariance[0] = var;
                self.filter.state.covariance[7] = var;
                self.filter.state.covariance[14] = var;
            }
        }
    }

    /// Advance time — call each Soma cycle.
    pub fn tick(&mut self, dt_s: f64) {
        self.current_tick += 1;
        self.filter.predict(dt_s);

        // Expire stale anchors (not seen for 60 ticks)
        self.anchors
            .retain(|a| self.current_tick - a.last_seen_tick < 60);

        // Check if GPS is still fresh
        if let Some(gps) = &self.last_gps {
            if (self.current_tick as f64 - gps.timestamp_s) > self.config.gps_max_age_s {
                self.gps_available = false;
                // Save PDR origin when GPS goes stale
                if self.pdr_origin_ecef.is_none() && self.update_count > 0 {
                    self.pdr_origin_ecef = Some(self.filter.position());
                    self.pdr.reset_to_origin();
                }
            }
        }
        self.update_mode();
    }

    /// Get current position as geodetic coordinates.
    pub fn get_position(&self) -> (f64, f64, f64) {
        let pos = self.filter.position();
        self.body.cartesian_to_geodetic(pos)
    }

    /// Get 1-sigma position uncertainty in meters.
    pub fn get_sigma(&self) -> f64 {
        self.filter.position_sigma()
    }

    /// Get the current local state in conservative Gaussian form for swarm sharing.
    pub fn gaussian_estimate(&self) -> GaussianEstimate3D {
        PublishableEstimate3D::gaussian_estimate(self)
    }

    /// Build a peer-shareable estimate for the current mobile state.
    pub fn peer_estimate(
        &self,
        peer_id: impl Into<String>,
        confidence: Option<f64>,
    ) -> PeerEstimate3D {
        PublishableEstimate3D::peer_estimate(self, peer_id, confidence)
    }

    /// Get current positioning mode.
    pub fn get_mode(&self) -> PositioningMode {
        self.mode
    }

    /// Get GDOP (lower is better, < 3 = good).
    pub fn get_gdop(&self) -> f64 {
        self.cached_gdop
    }

    /// Get number of active anchors.
    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    /// Generate BLE anchor payload for broadcasting position to peers.
    /// Returns None if we don't have a good enough position to share.
    pub fn get_anchor_payload(&self) -> Option<Vec<u8>> {
        if !self.config.broadcast_as_anchor {
            return None;
        }
        if self.get_sigma() > 100.0 {
            return None;
        } // Too uncertain

        let (lat, lon, alt) = self.get_position();
        let sigma = self.get_sigma() as f32;
        let trust = if self.gps_available { 0.9_f32 } else { 0.3 };

        let mut payload = Vec::with_capacity(25);
        payload.push(1u8); // message type: position_anchor
        payload.extend_from_slice(&lat.to_le_bytes());
        payload.extend_from_slice(&lon.to_le_bytes());
        payload.extend_from_slice(&sigma.to_le_bytes());
        payload.extend_from_slice(&trust.to_le_bytes());
        Some(payload)
    }

    /// Parse incoming BLE anchor payload from a peer.
    pub fn receive_anchor_payload(&mut self, peer_id: &str, payload: &[u8]) {
        if payload.len() < 25 || payload[0] != 1 {
            return;
        }
        let lat = f64::from_le_bytes(payload[1..9].try_into().unwrap_or([0; 8]));
        let lon = f64::from_le_bytes(payload[9..17].try_into().unwrap_or([0; 8]));
        let sigma = f32::from_le_bytes(payload[17..21].try_into().unwrap_or([0; 4]));
        let trust = f32::from_le_bytes(payload[21..25].try_into().unwrap_or([0; 4]));

        if lat.abs() <= 90.0 && lon.abs() <= 180.0 && sigma > 0.0 {
            self.register_anchor(
                peer_id.to_string(),
                lat,
                lon,
                0.0,
                sigma as f64,
                trust as f64,
            );
        }
    }

    /// Get PDR drift estimate in meters (how uncertain dead reckoning is).
    pub fn get_pdr_drift(&self) -> f64 {
        self.pdr.get_drift_estimate()
    }

    /// Get PDR total steps.
    pub fn get_pdr_steps(&self) -> u64 {
        self.pdr.total_steps()
    }

    /// Get current heading in degrees (from PDR gyro + magnetometer fusion).
    pub fn get_heading_deg(&self) -> f64 {
        self.pdr.get_heading_deg()
    }

    /// Get neuromodulator nudges based on positioning state.
    /// Returns (norepinephrine_delta, dopamine_delta, serotonin_delta).
    pub fn neuromod_nudges(&self) -> (f32, f32, f32) {
        let sigma = self.get_sigma();
        let drift = self.pdr.get_drift_estimate();
        // High uncertainty OR high drift → NE (alertness)
        let ne = if sigma > 50.0 || drift > 30.0 {
            0.1
        } else if sigma > 10.0 || drift > 10.0 {
            0.03
        } else {
            0.0
        };
        // Novel location → DA (exploration)
        let da = if self.update_count < 10 { 0.05 } else { 0.0 };
        // Low uncertainty AND low drift → 5-HT (calm/safety)
        let serotonin = if sigma < 5.0 && drift < 5.0 {
            0.02
        } else {
            0.0
        };
        (ne, da, serotonin)
    }

    // ========================================================================
    // INTERNAL
    // ========================================================================

    fn apply_range(&mut self, peer_id: &str, range_est: RangeEstimate) {
        if let Some(anchor) = self.anchors.iter().find(|a| a.peer_id == peer_id) {
            self.filter.update(
                &anchor.position_ecef,
                range_est.distance_m,
                range_est.sigma_m,
                anchor.trust,
            );
            self.update_count += 1;
            self.update_mode();
        }
        // If peer_id not in anchors, we can't use this range yet
        // (need to discover their position first via BLE anchor payload)
    }

    fn update_mode(&mut self) {
        self.mode = if self.gps_available && !self.anchors.is_empty() {
            PositioningMode::Hybrid
        } else if self.gps_available {
            PositioningMode::GpsPrimary
        } else if self.anchors.len() >= self.config.min_anchors {
            PositioningMode::CooperativeMesh
        } else if self.pdr_origin_ecef.is_some() && self.pdr.total_steps() > 0 {
            PositioningMode::DeadReckoning
        } else if self.update_count > 0 {
            PositioningMode::Offline
        } else {
            PositioningMode::Initializing
        };
    }

    fn update_gdop(&mut self) {
        if self.anchors.len() < 4 {
            self.cached_gdop = f64::INFINITY;
            return;
        }
        let anchor_positions: Vec<[f64; 3]> =
            self.anchors.iter().map(|a| a.position_ecef).collect();
        let pos = self.filter.position();
        self.cached_gdop = gdop(&anchor_positions, &pos);
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_bridge_is_initializing() {
        let bridge = PositioningBridge::new(PositioningConfig::default());
        assert_eq!(bridge.get_mode(), PositioningMode::Initializing);
        assert!(bridge.get_sigma() > 1000.0);
    }

    #[test]
    fn gps_update_sets_position() {
        let mut bridge = PositioningBridge::new(PositioningConfig::default());
        // Johannesburg
        bridge.update_gps(-26.2041, 28.0473, 1753.0, 5.0);
        assert_eq!(bridge.get_mode(), PositioningMode::GpsPrimary);
        let (lat, lon, _alt) = bridge.get_position();
        assert!((lat - (-26.2041)).abs() < 0.001);
        assert!((lon - 28.0473).abs() < 0.001);
        assert!(bridge.get_sigma() < 10.0);
    }

    #[test]
    fn gps_expires_after_max_age() {
        let mut bridge = PositioningBridge::new(PositioningConfig::default());
        bridge.update_gps(-26.2041, 28.0473, 1753.0, 5.0);
        assert!(bridge.gps_available);
        // Tick past max age
        for _ in 0..40 {
            bridge.tick(1.0);
        }
        assert!(!bridge.gps_available);
    }

    #[test]
    fn anchor_registration_and_expiry() {
        let mut bridge = PositioningBridge::new(PositioningConfig::default());
        bridge.register_anchor("node1".into(), -26.0, 28.0, 1700.0, 5.0, 0.8);
        assert_eq!(bridge.anchor_count(), 1);
        // Expire after 60 ticks
        for _ in 0..65 {
            bridge.tick(1.0);
        }
        assert_eq!(bridge.anchor_count(), 0);
    }

    #[test]
    fn anchor_payload_roundtrip() {
        let mut bridge = PositioningBridge::new(PositioningConfig::default());
        bridge.update_gps(-26.2041, 28.0473, 1753.0, 5.0);

        let payload = bridge.get_anchor_payload().expect("Should have payload");
        assert!(payload.len() >= 25);
        assert_eq!(payload[0], 1); // anchor type

        let mut bridge2 = PositioningBridge::new(PositioningConfig::default());
        bridge2.receive_anchor_payload("peer1", &payload);
        assert_eq!(bridge2.anchor_count(), 1);
    }

    #[test]
    fn mobile_bridge_emits_shared_measurements() {
        let mut bridge = PositioningBridge::new(PositioningConfig::default());
        bridge.update_gps(-26.2041, 28.0473, 1753.0, 5.0);
        bridge.update_imu(1.2, 0.1, 1008.0, 2, Some(90.0), Some(10.0), 1.0);
        let measurements = bridge.drain_measurements();
        assert!(measurements.iter().any(|m| matches!(
            m.modality,
            positioning::MeasurementModality::AbsolutePosition
        )));
        assert!(measurements.iter().any(|m| matches!(
            m.modality,
            positioning::MeasurementModality::RelativePosition
        )));
    }

    #[test]
    fn cooperative_mesh_mode() {
        let mut bridge = PositioningBridge::new(PositioningConfig {
            min_anchors: 3,
            ..Default::default()
        });
        // No GPS, but 3 anchors
        bridge.register_anchor("n1".into(), -26.0, 28.0, 1700.0, 5.0, 0.9);
        bridge.register_anchor("n2".into(), -26.001, 28.001, 1700.0, 5.0, 0.9);
        bridge.register_anchor("n3".into(), -26.001, 27.999, 1700.0, 5.0, 0.9);
        bridge.update_ble_range("n1", -60);
        assert_eq!(bridge.get_mode(), PositioningMode::CooperativeMesh);
    }

    #[test]
    fn hybrid_mode_with_gps_and_anchors() {
        let mut bridge = PositioningBridge::new(PositioningConfig::default());
        bridge.update_gps(-26.2041, 28.0473, 1753.0, 5.0);
        bridge.register_anchor("n1".into(), -26.0, 28.0, 1700.0, 5.0, 0.9);
        bridge.update_ble_range("n1", -60);
        assert_eq!(bridge.get_mode(), PositioningMode::Hybrid);
    }

    #[test]
    fn neuromod_nudges_scale_with_uncertainty() {
        let bridge = PositioningBridge::new(PositioningConfig::default());
        let (ne, _da, _sht) = bridge.neuromod_nudges();
        assert!(ne > 0.0, "High uncertainty should trigger NE");
    }
}
