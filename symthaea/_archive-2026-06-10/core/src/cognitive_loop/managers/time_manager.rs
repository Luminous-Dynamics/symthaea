// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time Manager — Sovereign Mesh-Time CognitiveSubsystem
//!
//! **Requires**: `feature = "mesh"`
//!
//! Feeds `MeshTimeConsensus` from WisdomPacket timestamps received during
//! the cognitive cycle. Provides neuromodulatory feedback based on time
//! consensus quality and drift surprise.
//!
//! # Science
//! - Mills (1985) — NTP adaptive filtering and clock discipline
//! - Marzullo (1984) — Interval-based fault-tolerant clock synchronization
//! - NTP RFC 5905 — Network Time Protocol Version 4
//!
//! # Interval
//! 23 — co-prime with {7, 11, 13, 19, 29, 37, 41, 43, 47, 53, 67}

use crate::cognitive_loop::subsystem_trait::{CognitiveSubsystem, CycleSnapshot, SubsystemOutput};
use crate::swarm::mesh::mesh_time::{MeshTimeConsensus, MeshTimeTelemetry, TimeQuality};
use crate::swarm::mesh::time_beacon::TimeBeacon;

/// Neuromodulatory gain for drift surprise → NE.
/// Basis: Aston-Jones & Cohen (2005) — unexpected temporal drift triggers alertness.
const DRIFT_SURPRISE_NE_GAIN: f64 = 0.04;

/// Neuromodulatory gain for stable consensus → 5-HT.
/// Basis: Cools et al. (2008) — temporal predictability promotes serotonergic calm.
const STABLE_CONSENSUS_5HT_GAIN: f64 = 0.02;

/// Drift surprise threshold (ppm) above which NE is triggered.
const DRIFT_SURPRISE_THRESHOLD_PPM: f64 = 10.0;

/// EMA alpha for drift surprise tracking.
const DRIFT_SURPRISE_EMA: f64 = 0.2;

/// Time Manager — integrates mesh-time consensus into the cognitive loop.
pub struct TimeManager {
    /// The core consensus engine.
    consensus: MeshTimeConsensus,
    /// Pending beacons from mesh receiver (injected between cycles).
    pending_beacons: Vec<(TimeBeacon, [u8; 8])>,
    /// EMA of drift surprise for neuromod.
    drift_surprise_ema: f64,
    /// Previous consensus offset for delta detection.
    prev_offset_us: i64,
    /// Whether mesh time is enabled (config toggle).
    enabled: bool,
    /// Last telemetry snapshot.
    last_telemetry: MeshTimeTelemetry,
}

impl TimeManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 23;

    /// Create a new TimeManager.
    pub fn new(enabled: bool) -> Self {
        Self {
            consensus: MeshTimeConsensus::new(),
            pending_beacons: Vec::new(),
            drift_surprise_ema: 0.0,
            prev_offset_us: 0,
            enabled,
            last_telemetry: MeshTimeTelemetry::default(),
        }
    }

    /// Inject a received time beacon for processing on next cycle.
    pub fn inject_beacon(&mut self, beacon: TimeBeacon, source_id: [u8; 8]) {
        if self.enabled {
            self.pending_beacons.push((beacon, source_id));
        }
    }

    /// Get the current mesh time in microseconds.
    pub fn mesh_now(&self) -> u64 {
        self.consensus.mesh_now()
    }

    /// Get the consensus offset in microseconds.
    pub fn offset_us(&self) -> i64 {
        self.consensus.offset_us()
    }

    /// Get our stratum level.
    pub fn stratum(&self) -> u8 {
        self.consensus.stratum()
    }

    /// Get current time quality.
    pub fn quality(&self) -> TimeQuality {
        self.consensus.quality()
    }

    /// Get last telemetry snapshot.
    pub fn telemetry(&self) -> &MeshTimeTelemetry {
        &self.last_telemetry
    }

    /// Access the underlying consensus engine.
    pub fn consensus(&self) -> &MeshTimeConsensus {
        &self.consensus
    }

    /// Mutable access to consensus engine (for NTP exchange injection).
    pub fn consensus_mut(&mut self) -> &mut MeshTimeConsensus {
        &mut self.consensus
    }

    /// Set Phi for outgoing beacons.
    pub fn set_phi(&mut self, phi: f32) {
        self.consensus.set_phi(phi);
    }

    /// Create a TimeBeacon for outgoing transmission.
    pub fn create_beacon(&mut self) -> TimeBeacon {
        TimeBeacon {
            timestamp_us: self.consensus.mesh_now(),
            stratum: self.consensus.stratum(),
            counter: self.consensus.next_beacon_counter(),
            phi: self.consensus.phi(),
            drift_ppm: self.consensus.drift_ppm() as f32,
        }
    }

    /// Process pending beacons and update consensus.
    fn process_pending(&mut self, now_us: u64) {
        for (beacon, source_id) in self.pending_beacons.drain(..) {
            use crate::swarm::mesh::mesh_time::TimeSample;
            self.consensus.add_sample(TimeSample {
                peer_id: source_id,
                offset_us: beacon.timestamp_us as i64 - now_us as i64,
                rtt_us: 0, // Unknown from beacon alone
                stratum: beacon.stratum,
                phi: beacon.phi,
                local_timestamp_us: now_us,
            });
        }
        self.consensus.recompute_consensus(now_us);
    }
}

impl CognitiveSubsystem for TimeManager {
    fn name(&self) -> &'static str {
        "time_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        if !self.enabled {
            return output;
        }

        // Use cycle number as approximate time (µs precision not critical for processing)
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Process any pending beacons
        self.process_pending(now_us);

        // Update Phi from snapshot
        self.consensus.set_phi(snapshot.unified_psi as f32);

        // Compute drift surprise
        let _offset_delta =
            (self.consensus.offset_us() - self.prev_offset_us).unsigned_abs() as f64;
        let drift_surprise = self.consensus.drift_ppm().abs();
        self.drift_surprise_ema = self.drift_surprise_ema * (1.0 - DRIFT_SURPRISE_EMA)
            + drift_surprise * DRIFT_SURPRISE_EMA;
        self.prev_offset_us = self.consensus.offset_us();

        // Neuromod: drift surprise → arousal (NE pathway)
        if self.drift_surprise_ema > DRIFT_SURPRISE_THRESHOLD_PPM {
            let surprise_excess = (self.drift_surprise_ema - DRIFT_SURPRISE_THRESHOLD_PPM)
                / crate::cognitive_loop::thresholds::TIME_DRIFT_SURPRISE_DIVISOR;
            output.arousal_delta += (surprise_excess * DRIFT_SURPRISE_NE_GAIN)
                .min(crate::cognitive_loop::thresholds::TIME_DRIFT_AROUSAL_CAP)
                as f32;
        }

        // Neuromod: stable consensus → calm (5-HT pathway)
        if self.consensus.quality() == TimeQuality::Consensus
            || self.consensus.quality() == TimeQuality::Authoritative
        {
            output.valence_delta += STABLE_CONSENSUS_5HT_GAIN as f32;
        }

        // Update telemetry
        self.last_telemetry = self.consensus.telemetry();

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Layout: [drift_surprise_ema: f64 = 8][prev_offset_us: i64 = 8][enabled: u8 = 1]
        // Total: 17 bytes
        let mut data = Vec::with_capacity(17);
        data.extend_from_slice(&self.drift_surprise_ema.to_le_bytes());
        data.extend_from_slice(&self.prev_offset_us.to_le_bytes());
        data.push(self.enabled as u8);
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        const MIN_SIZE: usize = 8 + 8 + 1; // 17
        if data.len() < MIN_SIZE {
            return Err(format!(
                "TimeManager checkpoint too short: {} < {}",
                data.len(),
                MIN_SIZE
            ));
        }
        self.drift_surprise_ema = f64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| "TimeManager: corrupt drift_surprise_ema bytes")?,
        );
        self.prev_offset_us = i64::from_le_bytes(
            data[8..16]
                .try_into()
                .map_err(|_| "TimeManager: corrupt prev_offset_us bytes")?,
        );
        self.enabled = data[16] != 0;
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

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot {
            unified_psi: 0.7,
            ..CycleSnapshot::default()
        }
    }

    #[test]
    fn test_time_manager_creation() {
        let mgr = TimeManager::new(true);
        assert_eq!(mgr.name(), "time_manager");
        assert_eq!(mgr.interval(), 23);
        assert!(mgr.enabled);
    }

    #[test]
    fn test_disabled_manager_returns_neutral() {
        let mut mgr = TimeManager::new(false);
        let output = mgr.process(&default_snapshot());
        assert_eq!(output.arousal_delta, 0.0);
        assert_eq!(output.valence_delta, 0.0);
    }

    #[test]
    fn test_beacon_injection_and_processing() {
        let mut mgr = TimeManager::new(true);
        let beacon = TimeBeacon {
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64
                + 1000,
            stratum: 1,
            counter: 1,
            phi: 0.8,
            drift_ppm: 0.0,
        };
        mgr.inject_beacon(beacon.clone(), [1, 0, 0, 0, 0, 0, 0, 0]);
        mgr.inject_beacon(beacon.clone(), [2, 0, 0, 0, 0, 0, 0, 0]);
        mgr.inject_beacon(beacon.clone(), [3, 0, 0, 0, 0, 0, 0, 0]);
        mgr.process(&default_snapshot());
        assert!(mgr.consensus().peer_count() >= 3);
    }

    #[test]
    fn test_create_beacon() {
        let mut mgr = TimeManager::new(true);
        mgr.set_phi(0.6);
        let beacon = mgr.create_beacon();
        assert!((beacon.phi - 0.6).abs() < f32::EPSILON);
        assert!(beacon.timestamp_us > 0);
        assert_eq!(beacon.counter, 1);
        let beacon2 = mgr.create_beacon();
        assert_eq!(beacon2.counter, 2);
    }

    #[test]
    fn test_telemetry_update() {
        let mut mgr = TimeManager::new(true);
        mgr.process(&default_snapshot());
        let telem = mgr.telemetry();
        assert_eq!(telem.peer_count, 0);
    }

    #[test]
    fn test_disabled_beacon_injection_ignored() {
        let mut mgr = TimeManager::new(false);
        let beacon = TimeBeacon {
            timestamp_us: 1_000_000,
            stratum: 1,
            counter: 1,
            phi: 0.8,
            drift_ppm: 0.0,
        };
        mgr.inject_beacon(beacon, [1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(mgr.consensus().peer_count(), 0);
    }
}
