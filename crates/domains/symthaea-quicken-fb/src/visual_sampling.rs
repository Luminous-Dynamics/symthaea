// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic negative-space sampling for Spore visual nodes.
//!
//! This module makes `MorphologyParameters.node_density` usable without changing
//! topology identity. It is pure, allocation-free, platform-stable for the same
//! seed/index inputs, and contains no boot-state authority.

const INDEX_MIX: u64 = 0xd1b5_4a32_d192_ed03;
const SEED_BIAS: u64 = 0x9e37_79b9_7f4a_7c15;

/// Return whether an ordinary endpoint/node should receive a visible glow.
///
/// `node_density` is clamped to `[0, 1]`. Selection is monotonic: increasing
/// density can only add nodes for the same seed/index pair; it never removes one.
pub fn ordinary_node_visible(seed: &[u8; 32], curve_index: usize, node_density: f32) -> bool {
    let density = finite_unit(node_density);
    if density <= 0.0 {
        return false;
    }
    if density >= 1.0 {
        return true;
    }

    let threshold = (density * 65_536.0).floor() as u32;
    u32::from(node_score(seed, curve_index)) < threshold
}

/// Semantic event nodes (for example explicit repair marks) remain visible even
/// when ordinary decorative node density is low.
pub fn node_visible(
    seed: &[u8; 32],
    curve_index: usize,
    node_density: f32,
    semantic_override: bool,
) -> bool {
    semantic_override || ordinary_node_visible(seed, curve_index, node_density)
}

fn node_score(seed: &[u8; 32], curve_index: usize) -> u16 {
    let mut state = SEED_BIAS ^ (curve_index as u64).wrapping_mul(INDEX_MIX);
    for chunk_index in 0..4 {
        let start = chunk_index * 8;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&seed[start..start + 8]);
        let word = u64::from_le_bytes(bytes).rotate_left((chunk_index * 17) as u32);
        state = mix64(state ^ word);
    }
    (state >> 48) as u16
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_has_frozen_golden_vector() {
        assert_eq!(node_score(&[0x42; 32], 17), 0x2517);
    }

    #[test]
    fn selection_is_deterministic() {
        let seed = [0xa5; 32];
        for index in 0..2_400 {
            assert_eq!(
                ordinary_node_visible(&seed, index, 0.47),
                ordinary_node_visible(&seed, index, 0.47)
            );
        }
    }

    #[test]
    fn zero_density_hides_and_full_density_shows_all_ordinary_nodes() {
        let seed = [0x19; 32];
        for index in 0..2_400 {
            assert!(!ordinary_node_visible(&seed, index, 0.0));
            assert!(ordinary_node_visible(&seed, index, 1.0));
        }
    }

    #[test]
    fn density_is_monotonic_for_every_node() {
        let seed = [0x42; 32];
        for index in 0..2_400 {
            let low = ordinary_node_visible(&seed, index, 0.25);
            let medium = ordinary_node_visible(&seed, index, 0.50);
            let high = ordinary_node_visible(&seed, index, 0.75);
            assert!(!low || medium, "index {index}");
            assert!(!medium || high, "index {index}");
        }
    }

    #[test]
    fn density_materially_changes_negative_space() {
        let seed = [0x42; 32];
        let count = |density| {
            (0..2_400)
                .filter(|index| ordinary_node_visible(&seed, *index, density))
                .count()
        };
        assert_eq!(count(0.25), 606);
        assert_eq!(count(0.50), 1_197);
        assert_eq!(count(0.75), 1_789);
    }

    #[test]
    fn semantic_override_survives_zero_decorative_density() {
        let seed = [0x7c; 32];
        assert!(!node_visible(&seed, 91, 0.0, false));
        assert!(node_visible(&seed, 91, 0.0, true));
    }

    #[test]
    fn invalid_density_fails_quietly_closed_for_decorative_nodes() {
        let seed = [0x7c; 32];
        assert!(!ordinary_node_visible(&seed, 4, f32::NAN));
        assert!(!ordinary_node_visible(&seed, 4, f32::NEG_INFINITY));
    }
}
