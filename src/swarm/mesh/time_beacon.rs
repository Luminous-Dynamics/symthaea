// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time Beacon — High-precision mesh time synchronization packets.
//!
//! Extends the mesh packet system with `PayloadType::TimeBeacon` for clock
//! synchronization. Beacons carry µs-precision timestamps, stratum levels,
//! and monotonic counters for ordering.
//!
//! # Wire Format (in WisdomPacket BinaryHV field)
//!
//! ```text
//! Bytes 0-7:   timestamp_us   (u64 LE) — sender's mesh time in µs since epoch
//! Bytes 8:     stratum        (u8)     — sender's stratum level
//! Bytes 9-16:  counter        (u64 LE) — monotonic beacon counter
//! Bytes 17-20: phi_bits       (f32 LE) — sender's Phi at emission time
//! Bytes 21-24: drift_ppm_bits (f32 LE) — sender's estimated drift
//! Bytes 25-2047: zeroed
//! ```

use symthaea_core::hdc::BinaryHV;

/// Parsed time beacon payload.
#[derive(Debug, Clone)]
pub struct TimeBeacon {
    /// Sender's mesh time in microseconds since Unix epoch.
    pub timestamp_us: u64,
    /// Sender's stratum level (0=GPS, 1=direct peer, 2+=transitive).
    pub stratum: u8,
    /// Monotonic beacon counter (for ordering).
    pub counter: u64,
    /// Sender's Phi at emission time.
    pub phi: f32,
    /// Sender's estimated clock drift in ppm.
    pub drift_ppm: f32,
}

impl TimeBeacon {
    /// Encode a time beacon into a BinaryHV for transmission in a WisdomPacket.
    pub fn encode(&self) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        bytes[0..8].copy_from_slice(&self.timestamp_us.to_le_bytes());
        bytes[8] = self.stratum;
        bytes[9..17].copy_from_slice(&self.counter.to_le_bytes());
        bytes[17..21].copy_from_slice(&self.phi.to_le_bytes());
        bytes[21..25].copy_from_slice(&self.drift_ppm.to_le_bytes());
        BinaryHV(bytes)
    }

    /// Decode a time beacon from a BinaryHV received in a WisdomPacket.
    pub fn decode(hv: &BinaryHV) -> Option<Self> {
        let bytes = &hv.0;
        if bytes.len() < 25 {
            return None;
        }

        let timestamp_us = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let stratum = bytes[8];
        let counter = u64::from_le_bytes(bytes[9..17].try_into().ok()?);
        let phi = f32::from_le_bytes(bytes[17..21].try_into().ok()?);
        let drift_ppm = f32::from_le_bytes(bytes[21..25].try_into().ok()?);

        // Sanity checks
        if !phi.is_finite() || !drift_ppm.is_finite() {
            return None;
        }
        if stratum > 15 {
            return None;
        }

        Some(Self {
            timestamp_us,
            stratum,
            counter,
            phi,
            drift_ppm,
        })
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beacon_roundtrip() {
        let original = TimeBeacon {
            timestamp_us: 1_710_000_000_000_000,
            stratum: 2,
            counter: 42,
            phi: 0.75,
            drift_ppm: -0.5,
        };
        let hv = original.encode();
        let decoded = TimeBeacon::decode(&hv).expect("decode should succeed");
        assert_eq!(decoded.timestamp_us, original.timestamp_us);
        assert_eq!(decoded.stratum, original.stratum);
        assert_eq!(decoded.counter, original.counter);
        assert!((decoded.phi - original.phi).abs() < f32::EPSILON);
        assert!((decoded.drift_ppm - original.drift_ppm).abs() < f32::EPSILON);
    }

    #[test]
    fn test_beacon_stratum_zero() {
        let beacon = TimeBeacon {
            timestamp_us: 1_000_000,
            stratum: 0,
            counter: 1,
            phi: 1.0,
            drift_ppm: 0.0,
        };
        let hv = beacon.encode();
        let decoded = TimeBeacon::decode(&hv).unwrap();
        assert_eq!(decoded.stratum, 0);
    }

    #[test]
    fn test_beacon_rejects_invalid_stratum() {
        let beacon = TimeBeacon {
            timestamp_us: 1_000_000,
            stratum: 16, // Invalid — max is 15
            counter: 1,
            phi: 0.5,
            drift_ppm: 0.0,
        };
        let hv = beacon.encode();
        assert!(TimeBeacon::decode(&hv).is_none());
    }

    #[test]
    fn test_beacon_rejects_nan_phi() {
        let beacon = TimeBeacon {
            timestamp_us: 1_000_000,
            stratum: 1,
            counter: 1,
            phi: f32::NAN,
            drift_ppm: 0.0,
        };
        let hv = beacon.encode();
        assert!(TimeBeacon::decode(&hv).is_none());
    }

    #[test]
    fn test_beacon_large_counter() {
        let beacon = TimeBeacon {
            timestamp_us: u64::MAX - 1,
            stratum: 5,
            counter: u64::MAX,
            phi: 0.3,
            drift_ppm: 100.0,
        };
        let hv = beacon.encode();
        let decoded = TimeBeacon::decode(&hv).unwrap();
        assert_eq!(decoded.counter, u64::MAX);
        assert_eq!(decoded.timestamp_us, u64::MAX - 1);
    }
}
