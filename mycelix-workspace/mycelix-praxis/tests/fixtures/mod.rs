// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Test fixtures and generators for Mycelix EduNet
//
// This module provides utilities for generating realistic test data across all
// components of the system. All generators support deterministic random generation
// via seeded RNG for reproducible tests.

pub mod aggregation;
pub mod courses;
pub mod credentials;
pub mod fl_rounds;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Default seed for reproducible tests
pub const DEFAULT_SEED: u64 = 42;

/// Create a seeded RNG for deterministic random generation
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Generate a random alphanumeric string of given length
pub fn random_string(rng: &mut ChaCha8Rng, len: usize) -> String {
    use rand::distributions::Alphanumeric;
    use rand::Rng;

    rng.sample_iter(&Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

/// Generate a random hex string of given length
pub fn random_hex(rng: &mut ChaCha8Rng, len: usize) -> String {
    (0..len)
        .map(|_| format!("{:02x}", rng.gen::<u8>()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeded_rng_deterministic() {
        let mut rng1 = seeded_rng(DEFAULT_SEED);
        let mut rng2 = seeded_rng(DEFAULT_SEED);

        assert_eq!(rng1.gen::<u32>(), rng2.gen::<u32>());
    }

    #[test]
    fn test_random_string() {
        let mut rng = seeded_rng(DEFAULT_SEED);
        let s = random_string(&mut rng, 10);

        assert_eq!(s.len(), 10);
        assert!(s.chars().all(|c| c.is_alphanumeric()));
    }

    #[test]
    fn test_random_hex() {
        let mut rng = seeded_rng(DEFAULT_SEED);
        let h = random_hex(&mut rng, 16);

        assert_eq!(h.len(), 32); // 16 bytes = 32 hex chars
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
