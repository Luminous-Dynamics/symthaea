// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Morphological transfer: quadrotor/helicopter → AUV.
//!
//! 10 shared channels (position, orientation, velocity) enable transfer
//! from aerial platforms. AUV-specific channels (depth, pressure, chemical
//! sensors, thrusters) are initialized from genesis.

/// Channel mapping for morphological transfer.
#[derive(Debug, Clone)]
pub struct ChannelMapping {
    pub source_ch: usize,
    pub target_ch: usize,
    pub weight: f32,
}

/// Default quadrotor (13ch) → AUV (32ch) mapping.
/// 10 shared channels: position(3), quaternion(4), velocity(3).
pub fn default_quadrotor_to_auv_mapping() -> Vec<ChannelMapping> {
    vec![
        // Position: flight[0..3] → AUV[0..3]
        ChannelMapping {
            source_ch: 0,
            target_ch: 0,
            weight: 1.0,
        },
        ChannelMapping {
            source_ch: 1,
            target_ch: 1,
            weight: 1.0,
        },
        ChannelMapping {
            source_ch: 2,
            target_ch: 2,
            weight: 1.5,
        }, // Altitude→depth (critical)
        // Quaternion: flight[3..7] → AUV[3..7]
        ChannelMapping {
            source_ch: 3,
            target_ch: 3,
            weight: 1.8,
        },
        ChannelMapping {
            source_ch: 4,
            target_ch: 4,
            weight: 1.8,
        },
        ChannelMapping {
            source_ch: 5,
            target_ch: 5,
            weight: 1.8,
        },
        ChannelMapping {
            source_ch: 6,
            target_ch: 6,
            weight: 1.8,
        },
        // Linear velocity: flight[7..10] → AUV[7..10]
        ChannelMapping {
            source_ch: 7,
            target_ch: 7,
            weight: 0.8,
        }, // Lower weight (drag-dominated)
        ChannelMapping {
            source_ch: 8,
            target_ch: 8,
            weight: 0.8,
        },
        ChannelMapping {
            source_ch: 9,
            target_ch: 9,
            weight: 0.8,
        },
    ]
}

/// Helicopter (18ch) → AUV (32ch) mapping (same shared channels).
pub fn default_helicopter_to_auv_mapping() -> Vec<ChannelMapping> {
    default_quadrotor_to_auv_mapping() // Same shared channels
}

pub const SHARED_CHANNELS: usize = 10;
pub const RECOMMENDED_BLEND_FACTOR: f32 = 0.4; // Lower than aerial-to-aerial (different physics)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_count() {
        assert_eq!(default_quadrotor_to_auv_mapping().len(), SHARED_CHANNELS);
    }

    #[test]
    fn test_mapping_valid_indices() {
        for m in default_quadrotor_to_auv_mapping() {
            assert!(m.source_ch < 13);
            assert!(m.target_ch < 32);
            assert!(m.weight > 0.0);
        }
    }
}
