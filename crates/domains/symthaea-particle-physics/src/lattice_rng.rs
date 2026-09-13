// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible random streams for lattice Monte Carlo.
//!
//! This module binds an explicit ChaCha8 implementation/version to injective
//! stream-coordinate schemes. It provides deterministic replay and stream
//! separation; it does not prove statistical independence or ensemble mixing.

use crate::lattice_sweep::Uniform01Source;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const LATTICE_RNG_ALGORITHM: &str = "chacha8";
pub const LATTICE_RNG_IMPLEMENTATION: &str = "rand_chacha::ChaCha8Rng";
pub const LATTICE_RNG_VERSION: &str = "0.3.1";
pub const LATTICE_RNG_STREAM_COORDINATES_V2_ID: &str =
    "lattice_stream_coordinates_v2_phase_purpose";
pub const LATTICE_RNG_REPLAY_STATE_ID: &str = "lattice_chacha8_replay_state_v1";

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

/// Campaign phase for the successor stream namespace frozen by LQCD-021D.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LatticeCampaignPhase {
    Throughput = 1,
    Pilot = 2,
    Final = 3,
    Qualification = 4,
}

/// Randomness purpose within one campaign phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LatticeStreamPurpose {
    GaugeTransition = 1,
    Diagnostic = 2,
    Analysis = 3,
    Bootstrap = 4,
}

/// Successor coordinates with explicit phase/purpose separation.
///
/// Packing is `[phase:4 | purpose:4 | ensemble_slot:24 | replica:16 | rank:16]`.
/// Existing `LatticeStreamCoordinates` IDs remain unchanged and are not
/// reinterpreted under this scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatticeStreamCoordinatesV2 {
    pub phase: LatticeCampaignPhase,
    pub purpose: LatticeStreamPurpose,
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

impl LatticeStreamCoordinatesV2 {
    pub fn stream_id(self) -> Result<u64, LatticeRngError> {
        if self.ensemble_slot >= (1 << 24) {
            return Err(LatticeRngError::EnsembleSlotTooLarge(self.ensemble_slot));
        }
        let domain = ((self.phase as u8) << 4) | self.purpose as u8;
        Ok(((domain as u64) << 56)
            | ((self.ensemble_slot as u64) << 32)
            | ((self.replica as u64) << 16)
            | self.rank as u64)
    }
}

/// Exact in-process ChaCha8 replay state.
///
/// This is deliberately narrower than a scientific campaign checkpoint: it
/// contains only the seed preimage, packed stream ID and ChaCha word position.
/// Higher-level checkpoint authority must additionally bind chain/campaign,
/// gauge-field, sampler, environment and numerical-profile commitments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatticeRngReplayState {
    seed: [u8; 32],
    stream_id: u64,
    word_pos: u128,
}

impl LatticeRngReplayState {
    pub fn seed(self) -> [u8; 32] {
        self.seed
    }

    pub fn stream_id(self) -> u64 {
        self.stream_id
    }

    pub fn word_pos(self) -> u128 {
        self.word_pos
    }
}

#[derive(Clone)]
pub struct LatticeChaCha8Stream {
    rng: ChaCha8Rng,
    seed: [u8; 32],
    stream_id: u64,
}

impl LatticeChaCha8Stream {
    fn from_seed_and_stream_id(seed: [u8; 32], stream_id: u64) -> Self {
        let mut rng = ChaCha8Rng::from_seed(seed);
        rng.set_stream(stream_id);
        Self {
            rng,
            seed,
            stream_id,
        }
    }

    pub fn new(
        seed: [u8; 32],
        coordinates: LatticeStreamCoordinates,
    ) -> Result<Self, LatticeRngError> {
        Ok(Self::from_seed_and_stream_id(seed, coordinates.stream_id()?))
    }

    pub fn new_v2(
        seed: [u8; 32],
        coordinates: LatticeStreamCoordinatesV2,
    ) -> Result<Self, LatticeRngError> {
        Ok(Self::from_seed_and_stream_id(seed, coordinates.stream_id()?))
    }

    /// Restore the exact ChaCha8 byte position represented by a replay state.
    pub fn from_replay_state(state: LatticeRngReplayState) -> Self {
        let mut stream = Self::from_seed_and_stream_id(state.seed, state.stream_id);
        stream.rng.set_word_pos(state.word_pos);
        stream
    }

    /// Capture the exact ChaCha8 stream coordinates required for byte replay.
    pub fn replay_state(&self) -> LatticeRngReplayState {
        LatticeRngReplayState {
            seed: self.seed,
            stream_id: self.stream_id,
            word_pos: self.rng.get_word_pos(),
        }
    }

    pub fn stream_id(&self) -> u64 {
        self.stream_id
    }

    /// Current offset from the start of the ChaCha stream in 32-bit words.
    pub fn word_pos(&self) -> u128 {
        self.rng.get_word_pos()
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

        let coordinates_v2 = LatticeStreamCoordinatesV2 {
            phase: LatticeCampaignPhase::Final,
            purpose: LatticeStreamPurpose::GaugeTransition,
            ensemble_slot: 1 << 24,
            replica: 0,
            rank: 0,
        };
        assert!(matches!(
            coordinates_v2.stream_id(),
            Err(LatticeRngError::EnsembleSlotTooLarge(0x0100_0000))
        ));
    }

    #[test]
    fn successor_namespace_reproduces_independent_oracle_ids() {
        let base = LatticeStreamCoordinatesV2 {
            phase: LatticeCampaignPhase::Pilot,
            purpose: LatticeStreamPurpose::GaugeTransition,
            ensemble_slot: 0x021d,
            replica: 7,
            rank: 0,
        };
        assert_eq!(base.stream_id().unwrap(), 0x2100_021d_0007_0000);
        assert_eq!(
            LatticeStreamCoordinatesV2 {
                phase: LatticeCampaignPhase::Throughput,
                ..base
            }
            .stream_id()
            .unwrap(),
            0x1100_021d_0007_0000
        );
        assert_eq!(
            LatticeStreamCoordinatesV2 {
                phase: LatticeCampaignPhase::Final,
                ..base
            }
            .stream_id()
            .unwrap(),
            0x3100_021d_0007_0000
        );
    }

    #[test]
    fn all_frozen_phase_purpose_combinations_are_distinct() {
        let phases = [
            LatticeCampaignPhase::Throughput,
            LatticeCampaignPhase::Pilot,
            LatticeCampaignPhase::Final,
            LatticeCampaignPhase::Qualification,
        ];
        let purposes = [
            LatticeStreamPurpose::GaugeTransition,
            LatticeStreamPurpose::Diagnostic,
            LatticeStreamPurpose::Analysis,
            LatticeStreamPurpose::Bootstrap,
        ];
        let mut ids = Vec::new();
        for phase in phases {
            for purpose in purposes {
                ids.push(
                    LatticeStreamCoordinatesV2 {
                        phase,
                        purpose,
                        ensemble_slot: 0x021d,
                        replica: 7,
                        rank: 0,
                    }
                    .stream_id()
                    .unwrap(),
                );
            }
        }
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 16);
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
            0x13, 0x8c, 0xca, 0x69, 0x40, 0x32, 0x5d, 0x71, 0xf9, 0xab, 0x52, 0xed, 0x1f,
            0x9a, 0x97, 0x28, 0x3e, 0x01, 0xc5, 0xf6, 0x99, 0xdc, 0x6a, 0x43, 0x54, 0x50,
            0x12, 0xc5, 0x5c, 0x4e, 0x22, 0x37, 0x19, 0x7b, 0xb7, 0x88, 0x0b, 0xd3, 0x22,
            0xdc, 0x5f, 0x20, 0x7b, 0x91, 0x6e, 0x1b, 0x50, 0x09, 0xea, 0xd7, 0x16, 0xb1,
            0x8b, 0xcd, 0x82, 0xa5, 0x25, 0x92, 0x90, 0xfd, 0x33, 0xdb, 0x2f, 0x9d,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn replay_state_reproduces_exact_subsequent_rng_bytes() {
        let seed = [0xa5; 32];
        let coordinates = LatticeStreamCoordinatesV2 {
            phase: LatticeCampaignPhase::Pilot,
            purpose: LatticeStreamPurpose::GaugeTransition,
            ensemble_slot: 0x021d,
            replica: 7,
            rank: 0,
        };
        let mut uninterrupted = LatticeChaCha8Stream::new_v2(seed, coordinates).unwrap();

        for _ in 0..37 {
            let _ = uninterrupted.next_u64();
        }
        let state = uninterrupted.replay_state();
        assert_eq!(state.stream_id(), 0x2100_021d_0007_0000);
        assert_eq!(state.word_pos(), uninterrupted.word_pos());
        assert_eq!(state.seed(), seed);

        let expected: Vec<u64> = (0..128).map(|_| uninterrupted.next_u64()).collect();
        let mut restored = LatticeChaCha8Stream::from_replay_state(state);
        assert_eq!(restored.word_pos(), state.word_pos());
        let got: Vec<u64> = (0..128).map(|_| restored.next_u64()).collect();
        assert_eq!(got, expected);
        assert_eq!(restored.word_pos(), uninterrupted.word_pos());
    }

    #[test]
    fn replay_state_also_preserves_open01_sequence() {
        let seed = [0x3c; 32];
        let coordinates = LatticeStreamCoordinatesV2 {
            phase: LatticeCampaignPhase::Final,
            purpose: LatticeStreamPurpose::GaugeTransition,
            ensemble_slot: 0x021d,
            replica: 2,
            rank: 0,
        };
        let mut uninterrupted = LatticeChaCha8Stream::new_v2(seed, coordinates).unwrap();
        for _ in 0..19 {
            let _ = uninterrupted.next_open01();
        }
        let state = uninterrupted.replay_state();
        let expected: Vec<u64> = (0..64)
            .map(|_| uninterrupted.next_open01().to_bits())
            .collect();
        let mut restored = LatticeChaCha8Stream::from_replay_state(state);
        let got: Vec<u64> = (0..64).map(|_| restored.next_open01().to_bits()).collect();
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
