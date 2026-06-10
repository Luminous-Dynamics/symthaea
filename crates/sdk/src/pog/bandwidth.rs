// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Bandwidth Infrastructure PoG
//!
//! Network connectivity verification for Proof of Grounding.

use super::{PogScore, VerificationLevel};
use serde::{Deserialize, Serialize};

/// Connection type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Residential fiber
    ResidentialFiber,
    /// Business dedicated
    BusinessDedicated,
    /// Community mesh network
    CommunityMesh,
    /// Satellite uplink
    SatelliteUplink,
}

impl ConnectionType {
    /// Get PoG multiplier
    pub fn multiplier(&self) -> f64 {
        match self {
            ConnectionType::ResidentialFiber => 1.1,
            ConnectionType::BusinessDedicated => 1.2,
            ConnectionType::CommunityMesh => 1.4, // Highest for community infrastructure
            ConnectionType::SatelliteUplink => 1.3,
        }
    }
}

/// Bandwidth attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthAttestation {
    /// Provider DID
    pub provider_did: String,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Peer measurements
    pub measurements: Vec<PeerMeasurement>,
    /// Sustained Mbps (median of measurements)
    pub sustained_mbps: f64,
    /// Average latency in ms
    pub latency_ms_avg: u32,
    /// Uptime percentage (0-100)
    pub uptime_percentage: f64,
    /// Verification level
    pub verification_level: VerificationLevel,
}

/// Individual peer measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerMeasurement {
    /// Measuring peer DID
    pub peer_did: String,
    /// Download speed in Mbps
    pub download_mbps: f64,
    /// Upload speed in Mbps
    pub upload_mbps: f64,
    /// Latency in ms
    pub latency_ms: u32,
    /// Measurement timestamp
    pub measured_at: u64,
}

impl BandwidthAttestation {
    /// Aggregate measurements using median (Sybil-resistant)
    pub fn aggregate_measurements(&self) -> (f64, f64) {
        if self.measurements.is_empty() {
            return (0.0, 0.0);
        }

        let mut downloads: Vec<f64> = self.measurements.iter().map(|m| m.download_mbps).collect();
        let mut uploads: Vec<f64> = self.measurements.iter().map(|m| m.upload_mbps).collect();

        downloads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        uploads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_download = downloads[downloads.len() / 2];
        let median_upload = uploads[uploads.len() / 2];

        (median_download, median_upload)
    }

    /// Calculate PoG score
    pub fn calculate_pog(&self) -> PogScore {
        const MIN_MBPS: f64 = 10.0;

        let (download, upload) = self.aggregate_measurements();
        let combined_mbps = (download + upload) / 2.0;

        if combined_mbps < MIN_MBPS {
            return PogScore::zero();
        }

        // Logarithmic scaling (100 Mbps = ~2.3x score of 10 Mbps)
        let bandwidth_factor = (combined_mbps / MIN_MBPS).ln() + 1.0;

        // Uptime factor
        let uptime_factor = (self.uptime_percentage / 100.0).clamp(0.0, 1.0);

        // Latency bonus (lower is better)
        let latency_bonus = if self.latency_ms_avg < 50 {
            1.2
        } else if self.latency_ms_avg < 100 {
            1.1
        } else {
            1.0
        };

        // Connection type multiplier
        let type_mult = self.connection_type.multiplier();

        // Verification weight
        let verification = self.verification_level.weight();

        let score =
            (bandwidth_factor * uptime_factor * latency_bonus * type_mult * verification * 0.1)
                .min(1.0);

        PogScore::new(score)
    }

    /// Get minimum required peer measurements for validation
    pub fn required_peer_count(&self) -> usize {
        // More peers needed for higher value claims
        if self.sustained_mbps > 100.0 {
            7
        } else if self.sustained_mbps > 50.0 {
            5
        } else {
            3
        }
    }

    /// Check if enough peer measurements exist
    pub fn has_sufficient_peers(&self) -> bool {
        self.measurements.len() >= self.required_peer_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_measurements() -> Vec<PeerMeasurement> {
        vec![
            PeerMeasurement {
                peer_did: "did:test:peer1".to_string(),
                download_mbps: 45.0,
                upload_mbps: 40.0,
                latency_ms: 30,
                measured_at: 1000,
            },
            PeerMeasurement {
                peer_did: "did:test:peer2".to_string(),
                download_mbps: 50.0,
                upload_mbps: 45.0,
                latency_ms: 35,
                measured_at: 1001,
            },
            PeerMeasurement {
                peer_did: "did:test:peer3".to_string(),
                download_mbps: 48.0,
                upload_mbps: 42.0,
                latency_ms: 28,
                measured_at: 1002,
            },
        ]
    }

    #[test]
    fn test_bandwidth_pog() {
        let attestation = BandwidthAttestation {
            provider_did: "did:test:1".to_string(),
            connection_type: ConnectionType::CommunityMesh,
            measurements: sample_measurements(),
            sustained_mbps: 45.0,
            latency_ms_avg: 31,
            uptime_percentage: 99.5,
            verification_level: VerificationLevel::CommunityWitnessed,
        };

        let pog = attestation.calculate_pog();
        assert!(pog.value() > 0.0);
    }

    #[test]
    fn test_median_aggregation() {
        let attestation = BandwidthAttestation {
            provider_did: "did:test:1".to_string(),
            connection_type: ConnectionType::ResidentialFiber,
            measurements: sample_measurements(),
            sustained_mbps: 45.0,
            latency_ms_avg: 30,
            uptime_percentage: 99.0,
            verification_level: VerificationLevel::SelfReported,
        };

        let (download, upload) = attestation.aggregate_measurements();
        assert!((download - 48.0).abs() < 0.1); // Median of 45, 48, 50
        assert!((upload - 42.0).abs() < 0.1); // Median of 40, 42, 45
    }
}
