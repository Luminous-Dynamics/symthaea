// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Morphological transfer: quadrotor flight → helicopter.
//!
//! ## Status: DATA-ONLY, NOT YET CONSUMED (2026-07 robotics deep review)
//!
//! Nothing instantiates these mappings — no transfer routine reads
//! `default_flight_to_helicopter_mapping()` or `RECOMMENDED_BLEND_FACTOR`
//! anywhere in the workspace. This module is the *specification* of the
//! flight→helicopter channel correspondence, kept for the future transfer
//! implementation (wire-or-drop is tracked in
//! SYMTHAEA_ROBOTICS_IMPROVEMENT_PLAN_2026-07-06.md Tier 2.5).
//!
//! 13 shared channels (position, orientation, velocities) enable direct
//! concept projection from a trained quadrotor controller to a new helicopter.
//! The 5 helicopter-specific channels (rotor RPM ×2, swashplate feedback ×3)
//! are initialized from genesis.
//! (An earlier version of this header said "11 shared channels" while the
//! table and `SHARED_CHANNELS` said 13 — 13 is correct: 3 position +
//! 4 quaternion + 3 linear velocity + 3 angular velocity.)

/// Channel mapping entry for flight → helicopter morphological transfer.
#[derive(Debug, Clone)]
pub struct ChannelMapping {
    /// Source channel index (flight encoder).
    pub source_ch: usize,
    /// Target channel index (helicopter encoder).
    pub target_ch: usize,
    /// Contribution weight for this channel pair.
    pub weight: f32,
}

/// Default flight (13 channels) → helicopter (18 channels) mapping.
///
/// Shared concepts (13 channels):
/// - Position: flight[0..3] → helicopter[0..3]
/// - Quaternion: flight[3..7] → helicopter[3..7]
/// - Linear velocity: flight[7..10] → helicopter[7..10]
/// - Angular velocity: flight[10..13] → helicopter[10..13]
///
/// Unshared (5 channels, helicopter-only):
/// - main_rotor_rpm [13], tail_rotor_rpm [14], collective_pitch [15],
///   cyclic_lon_feedback [16], cyclic_lat_feedback [17]
pub fn default_flight_to_helicopter_mapping() -> Vec<ChannelMapping> {
    vec![
        // Position (weight 1.5 — spatial awareness transfers directly)
        ChannelMapping {
            source_ch: 0,
            target_ch: 0,
            weight: 1.5,
        },
        ChannelMapping {
            source_ch: 1,
            target_ch: 1,
            weight: 1.5,
        },
        ChannelMapping {
            source_ch: 2,
            target_ch: 2,
            weight: 1.5,
        },
        // Quaternion (weight 2.0 — orientation is the most transferable concept)
        ChannelMapping {
            source_ch: 3,
            target_ch: 3,
            weight: 2.0,
        },
        ChannelMapping {
            source_ch: 4,
            target_ch: 4,
            weight: 2.0,
        },
        ChannelMapping {
            source_ch: 5,
            target_ch: 5,
            weight: 2.0,
        },
        ChannelMapping {
            source_ch: 6,
            target_ch: 6,
            weight: 2.0,
        },
        // Linear velocity (weight 1.0 — dynamics differ due to mass/drag)
        ChannelMapping {
            source_ch: 7,
            target_ch: 7,
            weight: 1.0,
        },
        ChannelMapping {
            source_ch: 8,
            target_ch: 8,
            weight: 1.0,
        },
        ChannelMapping {
            source_ch: 9,
            target_ch: 9,
            weight: 1.0,
        },
        // Angular velocity (weight 1.5 — rotational dynamics partially shared)
        ChannelMapping {
            source_ch: 10,
            target_ch: 10,
            weight: 1.5,
        },
        ChannelMapping {
            source_ch: 11,
            target_ch: 11,
            weight: 1.5,
        },
        ChannelMapping {
            source_ch: 12,
            target_ch: 12,
            weight: 1.5,
        },
    ]
}

/// Number of shared channels between flight and helicopter.
pub const SHARED_CHANNELS: usize = 13;

/// Recommended blend factor for flight → helicopter transfer.
/// Higher than humanoid (0.5) because aerodynamics are more directly shared.
pub const RECOMMENDED_BLEND_FACTOR: f32 = 0.7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_covers_shared_channels() {
        let mapping = default_flight_to_helicopter_mapping();
        assert_eq!(mapping.len(), SHARED_CHANNELS);
    }

    #[test]
    fn test_mapping_indices_valid() {
        let mapping = default_flight_to_helicopter_mapping();
        for m in &mapping {
            assert!(
                m.source_ch < 13,
                "Source channel must be < 13 (flight channels)"
            );
            assert!(
                m.target_ch < 18,
                "Target channel must be < 18 (helicopter channels)"
            );
            assert!(m.weight > 0.0);
        }
    }

    #[test]
    fn test_shared_channels_are_identity_mapped() {
        let mapping = default_flight_to_helicopter_mapping();
        // First 13 channels share the same indices
        for m in &mapping {
            assert_eq!(
                m.source_ch, m.target_ch,
                "Shared channels should have same index"
            );
        }
    }
}
