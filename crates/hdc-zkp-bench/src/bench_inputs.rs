// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stable synthetic input fixtures for reproducible HDC proof demonstrations.
//!
//! SplitMix64 is used only to generate benchmark/test words. It is **not** a
//! cryptographic random-number generator and must never be used for secrets,
//! keys, blinding, or proof randomness.

const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX64_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX64_M2: u64 = 0x94D0_49BB_1331_11EB;

pub const SEED_HDC_XOR_A: u64 = 0x4844_435F_584F_5241;
pub const SEED_HDC_XOR_B: u64 = 0x4844_435F_584F_5242;
pub const SEED_BUNDLING_BASE: u64 = 0x4255_4E44_4C45_0000;
pub const SEED_CFC_INITIAL_STATE: u64 = 0x4346_435F_4830_0001;
pub const SEED_CFC_INPUT: u64 = 0x4346_435F_4830_0002;
pub const SEED_CFC_GATE: u64 = 0x4346_435F_4830_0003;
pub const SEED_FL_TRIPLE_BASE: u64 = 0x464C_5452_4950_0000;
pub const SEED_HAMMING_A: u64 = 0x4841_4D4D_494E_0041;
pub const SEED_HAMMING_B: u64 = 0x4841_4D4D_494E_0042;
pub const SEED_MASK_INITIAL_STATE: u64 = 0x5349_474D_4F49_4401;
pub const SEED_MASK_INPUT: u64 = 0x5349_474D_4F49_4402;
pub const SEED_MASK_GATE: u64 = 0x5349_474D_4F49_4403;
pub const SEED_TRIPLE_XOR_A: u64 = 0x5452_4950_4C45_0041;
pub const SEED_TRIPLE_XOR_B: u64 = 0x5452_4950_4C45_0042;

/// Generate a stable sequence of synthetic words from a public seed.
///
/// The output is independent of host entropy and stable for this specified
/// algorithm. Changing the algorithm or seed constants intentionally changes
/// the benchmark input corpus.
pub fn deterministic_words(seed: u64, count: usize) -> Vec<u64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(SPLITMIX64_GAMMA);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX64_M1);
            value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX64_M2);
            value ^ (value >> 31)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_fixture_matches_reference_vector() {
        assert_eq!(
            deterministic_words(0, 4),
            vec![
                0xE220_A839_7B1D_CDAF,
                0x6E78_9E6A_A1B9_65F4,
                0x06C4_5D18_8009_454F,
                0xF88B_B8A8_724C_81EC,
            ],
        );
    }

    #[test]
    fn fixture_generation_is_repeatable_and_seed_sensitive() {
        assert_eq!(deterministic_words(17, 32), deterministic_words(17, 32));
        assert_ne!(deterministic_words(17, 32), deterministic_words(18, 32));
        assert!(deterministic_words(9, 0).is_empty());
    }
}
