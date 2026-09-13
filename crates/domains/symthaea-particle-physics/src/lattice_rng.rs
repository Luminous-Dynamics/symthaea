// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible random streams for lattice Monte Carlo.
//!
//! This module binds an explicit ChaCha8 implementation/version to an injective
//! 64-bit stream-coordinate scheme. It provides deterministic replay and stream
//! separation; it does not prove statistical independence or ensemble mixing.

use crate::lattice_sweep::Uniform01Source;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const LATTICE_RNG_ALGORITHM: &str = "chacha8";
pub const LATTICE_RNG_IMPLEMENTATION: &str = "rand_chacha::ChaCha8Rng";
pub const LATTICE_RNG_VERSION: &str = "0.3.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LatticeStreamDomain {
    GaugeTransition = 1,
    Bootstrap = 2,
    Qualification = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatticeStreamCoordinates {
    pub domain: LatticeStreamDomain,
    /// Campaign-local ensemble slot. Limited to 24 bits by the packed format.
    pub ensemble_slot: u32,
    pub replica: u16,
    pub rank: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeRngError {
    EnsembleSlotTooLarge(u32),
}

impl LatticeStreamCoordinates {
    /// Injective bit packing:
    /// `[domain:8 | ensemble_slot:24 | replica:16 | rank:16]`.
    pub fn stream_id(self) -> Result<u64, LatticeRngError> {
        if self.ensemble_slot >= (1 << 24) {
            return Err(LatticeRngError::EnsembleSlotTooLarge(self.ensemble_slot));
        }
        Ok(((self.domain as u64) << 56)
            | ((self.ensemble_slot as u64) << 32)
            | ((self.replica as u64) << 16)
            | self.rank as u64)
    }
}

#[derive(Clone)]
pub struct LatticeChaCha8Stream {
    rng: ChaCha8Rng,
    stream_id: u64,
}

impl LatticeChaCha8Stream {
    pub fn new(
        seed: [u8; 32],
        coordinates: LatticeStreamCoordinates,
    ) -> Result<Self, LatticeRngError> {
        let stream_id = coordinates.stream_id()?;
        let mut rng = ChaCha8Rng::from_seed(seed);
        rng.set_stream(stream_id);
        Ok(Self { rng, stream_id })
    }

    pub fn stream_id(&self) -> u64 {
        self.stream_id
    }

    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    /// Discrete symmetric approximation to U(0,1) that excludes both endpoints.
    ///
    /// We keep the top 52 random bits and map k in [0,2^52-1] to
    /// `(k+1)/(2^52+1)`. Complementary 52-bit integers map exactly to
    /// complementary grid points, matching the sweep proposal's `u -> 1-u`
    /// inverse-measure argument without ever emitting 0 or 1.
    pub fn next_open01(&mut self) -> f64 {
        let k = (self.next_u64() >> 12) + 1;
        let denom = (1u64 << 52) + 1;
        k as f64 / denom as f64
    }
}

impl Uniform01Source for LatticeChaCha8Stream {
    fn next_uniform(&mut self) -> f64 {
        self.next_open01()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_coordinates_pack_injectively_for_neighbors() {
        let a = LatticeStreamCoordinates {
            domain: LatticeStreamDomain::GaugeTransition,
            ensemble_slot: 0x123456,
            replica: 0x789a,
            rank: 0xbcde,
        };
        let b = LatticeStreamCoordinates { rank: 0xbcdf, ..a };
        assert_eq!(a.stream_id().unwrap(), 0x0112_3456_789a_bcde);
        assert_ne!(a.stream_id().unwrap(), b.stream_id().unwrap());
    }

    #[test]
    fn oversized_ensemble_slot_fails_closed() {
        let coordinates = LatticeStreamCoordinates {
            domain: LatticeStreamDomain::GaugeTransition,
            ensemble_slot: 1 << 24,
            replica: 0,
            rank: 0,
        };
        assert!(matches!(
            coordinates.stream_id(),
            Err(LatticeRngError::EnsembleSlotTooLarge(0x0100_0000))
        ));
    }

    #[test]
    fn chacha8_matches_independent_language_level_oracle() {
        let mut seed = [0u8; 32];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = i as u8;
        }
        let coordinates = LatticeStreamCoordinates {
            domain: LatticeStreamDomain::GaugeTransition,
            ensemble_slot: 0x123456,
            replica: 0x789a,
            rank: 0xbcde,
        };
        let mut stream = LatticeChaCha8Stream::new(seed, coordinates).unwrap();
        let mut got = [0u8; 64];
        stream.rng.fill_bytes(&mut got);
        let expected: [u8; 64] = [
            0x13,0x8c,0xca,0x69,0x40,0x32,0x5d,0x71,0xf9,0xab,0x52,0xed,0x1f,0x9a,0x97,0x28,
            0x3e,0x01,0xc5,0xf6,0x99,0xdc,0x6a,0x43,0x54,0x50,0x12,0xc5,0x5c,0x4e,0x22,0x37,
            0x19,0x7b,0xb7,0x88,0x0b,0xd3,0x22,0xdc,0x5f,0x20,0x7b,0x91,0x6e,0x1b,0x50,0x09,
            0xea,0xd7,0x16,0xb1,0x8b,0xcd,0x82,0xa5,0x25,0x92,0x90,0xfd,0x33,0xdb,0x2f,0x9d,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn open01_never_hits_endpoints_and_replays() {
        let seed = [0x42; 32];
        let coordinates = LatticeStreamCoordinates {
            domain: LatticeStreamDomain::Qualification,
            ensemble_slot: 7,
            replica: 3,
            rank: 11,
        };
        let mut a = LatticeChaCha8Stream::new(seed, coordinates).unwrap();
        let mut b = a.clone();
        for _ in 0..10_000 {
            let x = a.next_open01();
            let y = b.next_open01();
            assert_eq!(x, y);
            assert!(x > 0.0 && x < 1.0);
        }
    }
}
