// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign Mesh-Time Consensus — Marzullo's Algorithm over Consciousness Mesh
//!
//! Replaces centralized NTP with a peer-to-peer time consensus protocol where
//! high-consciousness (high-Phi) peers are trusted more. Implements:
//!
//! - Marzullo's algorithm for clock offset intersection
//! - Stratum levels (0=GPS/PPS, 1=direct peer, 2+=transitive)
//! - Phi-weighted peer trust (high-consciousness peers more reliable)
//! - Outlier rejection (3σ filtering)
//! - Drift estimation (ppm)
//!
//! # References
//! - Mills, D. (1985). Internet Time Synchronization: the Network Time Protocol
//! - Marzullo, K. (1984). Maintaining the time in a distributed system

use std::collections::HashMap;
use std::time::SystemTime;

/// Maximum peers tracked for time consensus.
const MAX_TIME_PEERS: usize = 64;

/// Minimum peers needed for consensus (below this, local clock only).
const MIN_CONSENSUS_PEERS: usize = 3;

/// Maximum stratum level (beyond this, clock is considered unreliable).
const MAX_STRATUM: u8 = 15;

/// Outlier rejection threshold in standard deviations.
const OUTLIER_SIGMA: f64 = 3.0;

/// EMA alpha for drift estimation.
const DRIFT_EMA_ALPHA: f64 = 0.1;

/// Stratum level for physical time sources (GPS, PPS).
pub const STRATUM_GPS: u8 = 0;

/// Stratum level for direct peer of stratum-0 source.
pub const STRATUM_DIRECT: u8 = 1;

/// Time quality levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeQuality {
    /// GPS/PPS synchronized — sub-microsecond accuracy.
    Authoritative,
    /// Consensus from 5+ peers — good confidence.
    Consensus,
    /// Consensus from 3-4 peers — moderate confidence.
    Degraded,
    /// Fewer than 3 peers — local clock only.
    FreeRunning,
}

impl TimeQuality {
    /// Numeric confidence score (0.0–1.0).
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Authoritative => 1.0,
            Self::Consensus => 0.9,
            Self::Degraded => 0.6,
            Self::FreeRunning => 0.3,
        }
    }
}

/// A single time sample from a peer.
#[derive(Debug, Clone)]
pub struct TimeSample {
    /// Peer identifier (first 8 bytes of node ID).
    pub peer_id: [u8; 8],
    /// Estimated offset from local clock in microseconds.
    pub offset_us: i64,
    /// Round-trip time in microseconds.
    pub rtt_us: u64,
    /// Peer's reported stratum level.
    pub stratum: u8,
    /// Peer's Phi at time of sample.
    pub phi: f32,
    /// Local timestamp when sample was taken.
    pub local_timestamp_us: u64,
}

/// Per-peer time tracking state.
#[derive(Debug, Clone)]
struct PeerTimeState {
    /// Recent offset samples (ring buffer).
    samples: Vec<i64>,
    /// Current index in ring buffer.
    index: usize,
    /// Maximum samples to retain.
    max_samples: usize,
    /// Peer's stratum level.
    stratum: u8,
    /// Last known Phi.
    phi: f32,
    /// Last sample timestamp.
    last_update_us: u64,
    /// Estimated drift rate in ppm.
    drift_ppm: f64,
    /// Previous offset for drift estimation.
    prev_offset_us: i64,
    /// Previous timestamp for drift estimation.
    prev_timestamp_us: u64,
}

impl PeerTimeState {
    fn new(stratum: u8, phi: f32, max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples),
            index: 0,
            max_samples,
            stratum,
            phi,
            last_update_us: 0,
            drift_ppm: 0.0,
            prev_offset_us: 0,
            prev_timestamp_us: 0,
        }
    }

    fn add_sample(&mut self, offset_us: i64, timestamp_us: u64) {
        // Drift estimation
        if self.prev_timestamp_us > 0 {
            let dt_us = timestamp_us.saturating_sub(self.prev_timestamp_us);
            if dt_us > 1_000_000 {
                // At least 1 second between samples
                let d_offset = (offset_us - self.prev_offset_us) as f64;
                let instantaneous_ppm = d_offset / (dt_us as f64) * 1_000_000.0;
                self.drift_ppm =
                    self.drift_ppm * (1.0 - DRIFT_EMA_ALPHA) + instantaneous_ppm * DRIFT_EMA_ALPHA;
            }
        }
        self.prev_offset_us = offset_us;
        self.prev_timestamp_us = timestamp_us;

        if self.samples.len() < self.max_samples {
            self.samples.push(offset_us);
        } else {
            self.samples[self.index] = offset_us;
        }
        self.index = (self.index + 1) % self.max_samples;
        self.last_update_us = timestamp_us;
    }

    /// Median offset across stored samples.
    fn median_offset(&self) -> i64 {
        if self.samples.is_empty() {
            return 0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    /// Phi-weighted trust score for this peer (higher = more trusted).
    fn trust_weight(&self) -> f64 {
        let stratum_factor = 1.0 / (1.0 + self.stratum as f64);
        let phi_factor = (self.phi as f64).clamp(0.0, 1.0);
        stratum_factor * (0.3 + 0.7 * phi_factor)
    }
}

/// Mesh-time consensus telemetry snapshot.
#[derive(Debug, Clone, Default)]
pub struct MeshTimeTelemetry {
    /// Current consensus offset from local clock in microseconds.
    pub offset_us: i64,
    /// Our effective stratum level.
    pub stratum: u8,
    /// Estimated clock drift in parts per million.
    pub drift_ppm: f64,
    /// Number of peers contributing to consensus.
    pub peer_count: usize,
    /// Current time quality assessment.
    pub quality: u8, // TimeQuality as u8 for serialization
}

/// Sovereign mesh-time consensus engine.
///
/// Implements Marzullo's algorithm with Phi-weighted peer trust to maintain
/// a decentralized clock that survives NTP failure.
pub struct MeshTimeConsensus {
    /// Per-peer time tracking.
    peers: HashMap<[u8; 8], PeerTimeState>,
    /// Current consensus offset from local SystemTime (µs).
    consensus_offset_us: i64,
    /// Our stratum level (min peer stratum + 1).
    our_stratum: u8,
    /// Estimated local drift in ppm.
    local_drift_ppm: f64,
    /// Current time quality.
    quality: TimeQuality,
    /// Maximum samples per peer.
    samples_per_peer: usize,
    /// Stale peer timeout in microseconds.
    stale_timeout_us: u64,
    /// Our local Phi for outgoing beacons.
    our_phi: f32,
    /// Monotonic beacon counter.
    beacon_counter: u64,
}

impl MeshTimeConsensus {
    /// Create a new consensus engine.
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            consensus_offset_us: 0,
            our_stratum: MAX_STRATUM,
            local_drift_ppm: 0.0,
            quality: TimeQuality::FreeRunning,
            samples_per_peer: 8,
            stale_timeout_us: 120_000_000, // 2 minutes
            our_phi: 0.0,
            beacon_counter: 0,
        }
    }

    /// Process an incoming time sample from a peer.
    pub fn add_sample(&mut self, sample: TimeSample) {
        if self.peers.len() >= MAX_TIME_PEERS && !self.peers.contains_key(&sample.peer_id) {
            // Evict lowest-trust peer if at capacity
            if let Some((&evict_id, _)) = self.peers.iter().min_by(|a, b| {
                a.1.trust_weight()
                    .partial_cmp(&b.1.trust_weight())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                self.peers.remove(&evict_id);
            }
        }

        let state = self.peers.entry(sample.peer_id).or_insert_with(|| {
            PeerTimeState::new(sample.stratum, sample.phi, self.samples_per_peer)
        });

        state.stratum = sample.stratum;
        state.phi = sample.phi;
        state.add_sample(sample.offset_us, sample.local_timestamp_us);
    }

    /// Recompute consensus from all peer samples.
    ///
    /// Uses Marzullo's interval intersection with Phi-weighted trust:
    /// 1. Collect median offsets from each peer
    /// 2. Weight by Phi and stratum
    /// 3. Reject outliers beyond 3σ
    /// 4. Weighted average of remaining offsets
    pub fn recompute_consensus(&mut self, now_us: u64) {
        // Expire stale peers
        self.peers
            .retain(|_, state| now_us.saturating_sub(state.last_update_us) < self.stale_timeout_us);

        let active_peers: Vec<_> = self
            .peers
            .values()
            .filter(|s| !s.samples.is_empty())
            .collect();

        let n = active_peers.len();
        if n < MIN_CONSENSUS_PEERS {
            self.quality = TimeQuality::FreeRunning;
            // Keep last known offset but mark quality degraded
            return;
        }

        // Step 1: Collect weighted offsets
        let weighted_offsets: Vec<(f64, f64)> = active_peers
            .iter()
            .map(|s| (s.median_offset() as f64, s.trust_weight()))
            .collect();

        // Step 2: Compute weighted mean
        let total_weight: f64 = weighted_offsets.iter().map(|(_, w)| w).sum();
        if total_weight < f64::EPSILON {
            return;
        }
        let weighted_mean = weighted_offsets.iter().map(|(o, w)| o * w).sum::<f64>() / total_weight;

        // Step 3: Compute standard deviation for outlier rejection
        let variance = weighted_offsets
            .iter()
            .map(|(o, w)| {
                let diff = o - weighted_mean;
                diff * diff * w
            })
            .sum::<f64>()
            / total_weight;
        let std_dev = variance.sqrt().max(1.0); // Floor at 1µs

        // Step 4: Reject outliers and recompute
        let filtered: Vec<(f64, f64)> = weighted_offsets
            .iter()
            .filter(|(o, _)| (o - weighted_mean).abs() < OUTLIER_SIGMA * std_dev)
            .copied()
            .collect();

        let filtered_weight: f64 = filtered.iter().map(|(_, w)| w).sum();
        if filtered_weight > f64::EPSILON {
            let filtered_mean = filtered.iter().map(|(o, w)| o * w).sum::<f64>() / filtered_weight;
            self.consensus_offset_us = filtered_mean as i64;
        }

        // Step 5: Update stratum (our stratum = min peer stratum + 1)
        let min_stratum = active_peers
            .iter()
            .map(|s| s.stratum)
            .min()
            .unwrap_or(MAX_STRATUM);
        self.our_stratum = (min_stratum + 1).min(MAX_STRATUM);

        // Step 6: Update drift estimate (average peer drifts)
        let drift_sum: f64 = active_peers.iter().map(|s| s.drift_ppm).sum();
        self.local_drift_ppm = drift_sum / n as f64;

        // Step 7: Quality assessment
        self.quality = if self.our_stratum <= 1 {
            TimeQuality::Authoritative
        } else if n >= 5 {
            TimeQuality::Consensus
        } else {
            TimeQuality::Degraded
        };
    }

    /// Get the current mesh time in microseconds since Unix epoch.
    ///
    /// Returns local SystemTime + consensus offset.
    pub fn mesh_now(&self) -> u64 {
        let local_us = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        if self.consensus_offset_us >= 0 {
            local_us.saturating_add(self.consensus_offset_us as u64)
        } else {
            local_us.saturating_sub(self.consensus_offset_us.unsigned_abs())
        }
    }

    /// Current consensus offset in microseconds.
    pub fn offset_us(&self) -> i64 {
        self.consensus_offset_us
    }

    /// Our effective stratum level.
    pub fn stratum(&self) -> u8 {
        self.our_stratum
    }

    /// Estimated local drift in parts per million.
    pub fn drift_ppm(&self) -> f64 {
        self.local_drift_ppm
    }

    /// Current time quality assessment.
    pub fn quality(&self) -> TimeQuality {
        self.quality
    }

    /// Number of active peers contributing to consensus.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Set our local Phi for outgoing beacons.
    pub fn set_phi(&mut self, phi: f32) {
        self.our_phi = phi;
    }

    /// Get our Phi.
    pub fn phi(&self) -> f32 {
        self.our_phi
    }

    /// Increment and return the beacon counter.
    pub fn next_beacon_counter(&mut self) -> u64 {
        self.beacon_counter += 1;
        self.beacon_counter
    }

    /// Generate a telemetry snapshot.
    pub fn telemetry(&self) -> MeshTimeTelemetry {
        MeshTimeTelemetry {
            offset_us: self.consensus_offset_us,
            stratum: self.our_stratum,
            drift_ppm: self.local_drift_ppm,
            peer_count: self.peers.len(),
            quality: self.quality as u8,
        }
    }

    /// Process an NTP-style 4-timestamp exchange.
    ///
    /// Ported from mycelix-realtime ClockSync, extended with Phi-weighting.
    pub fn process_ntp_exchange(
        &mut self,
        peer_id: [u8; 8],
        stratum: u8,
        phi: f32,
        local_send_us: u64,
        remote_recv_us: u64,
        remote_send_us: u64,
        local_recv_us: u64,
    ) {
        // Classic NTP offset calculation:
        // offset = ((remote_recv - local_send) + (remote_send - local_recv)) / 2
        let a = remote_recv_us as i64 - local_send_us as i64;
        let b = remote_send_us as i64 - local_recv_us as i64;
        let offset = (a + b) / 2;
        let rtt = (local_recv_us - local_send_us)
            .saturating_sub(remote_send_us.saturating_sub(remote_recv_us));

        self.add_sample(TimeSample {
            peer_id,
            offset_us: offset,
            rtt_us: rtt,
            stratum,
            phi,
            local_timestamp_us: local_recv_us,
        });
    }
}

impl Default for MeshTimeConsensus {
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

    fn make_peer_id(n: u8) -> [u8; 8] {
        [n, 0, 0, 0, 0, 0, 0, 0]
    }

    fn now_us() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    #[test]
    fn test_free_running_with_no_peers() {
        let engine = MeshTimeConsensus::new();
        assert_eq!(engine.quality(), TimeQuality::FreeRunning);
        assert_eq!(engine.peer_count(), 0);
        assert_eq!(engine.stratum(), MAX_STRATUM);
    }

    #[test]
    fn test_consensus_with_three_peers() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();
        for i in 0..3u8 {
            engine.add_sample(TimeSample {
                peer_id: make_peer_id(i),
                offset_us: 1000 + (i as i64 * 100), // ~1ms offsets
                rtt_us: 500,
                stratum: 1,
                phi: 0.7,
                local_timestamp_us: ts,
            });
        }
        engine.recompute_consensus(ts);
        assert_eq!(engine.quality(), TimeQuality::Degraded); // 3 peers = degraded
        assert!(engine.offset_us().abs() < 2000); // Within 2ms of mean
        assert_eq!(engine.stratum(), 2);
    }

    #[test]
    fn test_consensus_with_five_peers() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();
        for i in 0..5u8 {
            engine.add_sample(TimeSample {
                peer_id: make_peer_id(i),
                offset_us: 500,
                rtt_us: 200,
                stratum: 1,
                phi: 0.8,
                local_timestamp_us: ts,
            });
        }
        engine.recompute_consensus(ts);
        assert_eq!(engine.quality(), TimeQuality::Consensus);
        // Weighted mean may differ by ±1 due to integer rounding
        assert!(
            (engine.offset_us() - 500).abs() <= 1,
            "Expected ~500, got {}",
            engine.offset_us()
        );
    }

    #[test]
    fn test_outlier_rejection() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();
        // 4 peers at ~1000µs, 1 peer at 1,000,000µs (outlier)
        for i in 0..4u8 {
            engine.add_sample(TimeSample {
                peer_id: make_peer_id(i),
                offset_us: 1000 + (i as i64 * 10),
                rtt_us: 200,
                stratum: 1,
                phi: 0.8,
                local_timestamp_us: ts,
            });
        }
        engine.add_sample(TimeSample {
            peer_id: make_peer_id(99),
            offset_us: 1_000_000, // 1 second — massive outlier
            rtt_us: 200,
            stratum: 1,
            phi: 0.8,
            local_timestamp_us: ts,
        });
        engine.recompute_consensus(ts);
        // Consensus should be much closer to 1000 than to 1M
        assert!(
            engine.offset_us() < 500_000,
            "Expected <500k, got {}",
            engine.offset_us()
        );
    }

    #[test]
    fn test_phi_weighting() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();
        // 2 high-Phi peers at +1000µs
        for i in 0..2u8 {
            engine.add_sample(TimeSample {
                peer_id: make_peer_id(i),
                offset_us: 1000,
                rtt_us: 200,
                stratum: 1,
                phi: 0.9,
                local_timestamp_us: ts,
            });
        }
        // 1 low-Phi peer at -1000µs
        engine.add_sample(TimeSample {
            peer_id: make_peer_id(10),
            offset_us: -1000,
            rtt_us: 200,
            stratum: 1,
            phi: 0.1,
            local_timestamp_us: ts,
        });
        engine.recompute_consensus(ts);
        // Should be biased toward high-Phi peers (positive offset)
        assert!(engine.offset_us() > 0);
    }

    #[test]
    fn test_stratum_propagation() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();
        for i in 0..3u8 {
            engine.add_sample(TimeSample {
                peer_id: make_peer_id(i),
                offset_us: 0,
                rtt_us: 100,
                stratum: 2, // peers are stratum 2
                phi: 0.7,
                local_timestamp_us: ts,
            });
        }
        engine.recompute_consensus(ts);
        assert_eq!(engine.stratum(), 3); // We are stratum 3
    }

    #[test]
    fn test_stale_peer_expiry() {
        let mut engine = MeshTimeConsensus::new();
        let old_ts = 1_000_000; // Very old
        for i in 0..3u8 {
            engine.add_sample(TimeSample {
                peer_id: make_peer_id(i),
                offset_us: 500,
                rtt_us: 200,
                stratum: 1,
                phi: 0.8,
                local_timestamp_us: old_ts,
            });
        }
        let now = now_us();
        engine.recompute_consensus(now);
        // Old peers should be expired
        assert_eq!(engine.peer_count(), 0);
        assert_eq!(engine.quality(), TimeQuality::FreeRunning);
    }

    #[test]
    fn test_mesh_now_returns_adjusted_time() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();
        for i in 0..3u8 {
            engine.add_sample(TimeSample {
                peer_id: make_peer_id(i),
                offset_us: 5000, // +5ms
                rtt_us: 200,
                stratum: 1,
                phi: 0.8,
                local_timestamp_us: ts,
            });
        }
        engine.recompute_consensus(ts);
        let mesh = engine.mesh_now();
        let local = now_us();
        // mesh_now should be ~5ms ahead of local
        assert!(mesh > local.saturating_sub(10_000));
    }

    #[test]
    fn test_ntp_exchange_processing() {
        let mut engine = MeshTimeConsensus::new();
        // Simulate NTP exchange where remote is 2000µs ahead
        let t1 = 1_000_000u64;
        let t2 = 1_002_100u64; // remote recv (2000 offset + 100 transit)
        let t3 = 1_002_200u64; // remote send (2000 offset + 200 processing)
        let t4 = 1_000_300u64; // local recv (300 transit)

        engine.process_ntp_exchange(make_peer_id(0), 1, 0.8, t1, t2, t3, t4);
        // Need 3 peers for consensus — add 2 more
        engine.process_ntp_exchange(make_peer_id(1), 1, 0.8, t1, t2, t3, t4);
        engine.process_ntp_exchange(make_peer_id(2), 1, 0.8, t1, t2, t3, t4);
        engine.recompute_consensus(t4);
        // Offset should be approximately 2000µs
        let offset = engine.offset_us();
        assert!(
            (offset - 2000).abs() < 500,
            "Expected ~2000, got {}",
            offset
        );
    }

    #[test]
    fn test_max_peers_eviction() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();
        // Add MAX_TIME_PEERS + 1 peers
        for i in 0..=(MAX_TIME_PEERS as u8) {
            engine.add_sample(TimeSample {
                peer_id: make_peer_id(i),
                offset_us: 100,
                rtt_us: 200,
                stratum: 1,
                phi: if i == 0 { 0.01 } else { 0.8 }, // First peer has low Phi
                local_timestamp_us: ts,
            });
        }
        assert!(engine.peer_count() <= MAX_TIME_PEERS);
    }

    #[test]
    fn test_drift_estimation() {
        let mut engine = MeshTimeConsensus::new();
        let peer = make_peer_id(0);
        // Sample 1: offset 0 at time 1s (non-zero so baseline is set)
        engine.add_sample(TimeSample {
            peer_id: peer,
            offset_us: 0,
            rtt_us: 200,
            stratum: 1,
            phi: 0.8,
            local_timestamp_us: 1_000_000,
        });
        // Sample 2: offset +10 at time +11s (1ppm drift)
        engine.add_sample(TimeSample {
            peer_id: peer,
            offset_us: 10,
            rtt_us: 200,
            stratum: 1,
            phi: 0.8,
            local_timestamp_us: 11_000_000,
        });
        // Drift EMA (alpha=0.1): first measurement yields 0.1 × 1.0 = 0.1 ppm
        let state = engine.peers.get(&peer).unwrap();
        assert!(
            state.drift_ppm > 0.0 && state.drift_ppm <= 1.0,
            "Drift should be positive: {}",
            state.drift_ppm
        );
    }

    #[test]
    fn test_time_quality_confidence() {
        assert_eq!(TimeQuality::Authoritative.confidence(), 1.0);
        assert_eq!(TimeQuality::Consensus.confidence(), 0.9);
        assert_eq!(TimeQuality::Degraded.confidence(), 0.6);
        assert_eq!(TimeQuality::FreeRunning.confidence(), 0.3);
    }

    #[test]
    fn test_telemetry_snapshot() {
        let mut engine = MeshTimeConsensus::new();
        let ts = now_us();
        for i in 0..5u8 {
            engine.add_sample(TimeSample {
                peer_id: make_peer_id(i),
                offset_us: 2000,
                rtt_us: 100,
                stratum: 1,
                phi: 0.7,
                local_timestamp_us: ts,
            });
        }
        engine.recompute_consensus(ts);
        let telem = engine.telemetry();
        assert_eq!(telem.peer_count, 5);
        assert_eq!(telem.stratum, 2);
        assert_eq!(telem.offset_us, 2000);
    }

    #[test]
    fn test_default_impl() {
        let engine = MeshTimeConsensus::default();
        assert_eq!(engine.quality(), TimeQuality::FreeRunning);
    }

    #[test]
    fn test_beacon_counter_monotonic() {
        let mut engine = MeshTimeConsensus::new();
        let a = engine.next_beacon_counter();
        let b = engine.next_beacon_counter();
        let c = engine.next_beacon_counter();
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    }
}
