// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal legacy NumPy `RandomState` machinery needed for Matbench v0.1.
//!
//! Matbench generated its shuffled KFold indices through scikit-learn's
//! `check_random_state`, which uses NumPy's legacy MT19937 `RandomState` for an
//! integer seed. Historical `RandomState.shuffle` uses an inclusive bounded
//! integer sampler with mask-and-reject semantics; modulo reduction would not
//! reproduce the same permutation.

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

pub(crate) struct LegacyMt19937 {
    key: [u32; MT_N],
    pos: usize,
}

impl LegacyMt19937 {
    pub(crate) fn seeded(seed: u32) -> Self {
        let mut key = [0u32; MT_N];
        key[0] = seed;
        for pos in 1..MT_N {
            let previous = key[pos - 1];
            key[pos] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(pos as u32);
        }
        Self { key, pos: MT_N }
    }

    fn random_u32(&mut self) -> u32 {
        if self.pos >= MT_N {
            self.twist();
        }

        let mut value = self.key[self.pos];
        self.pos += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    /// NumPy RandomKit-style inclusive integer sample in `[0, max]`.
    pub(crate) fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }

        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;

        loop {
            let value = self.random_u32() & mask;
            if value <= max {
                return value;
            }
        }
    }

    fn twist(&mut self) {
        for index in 0..(MT_N - MT_M) {
            let value =
                (self.key[index] & MT_UPPER_MASK) | (self.key[index + 1] & MT_LOWER_MASK);
            self.key[index] = self.key[index + MT_M]
                ^ (value >> 1)
                ^ if value & 1 != 0 { MT_MATRIX_A } else { 0 };
        }

        for index in (MT_N - MT_M)..(MT_N - 1) {
            let value =
                (self.key[index] & MT_UPPER_MASK) | (self.key[index + 1] & MT_LOWER_MASK);
            self.key[index] = self.key[index - (MT_N - MT_M)]
                ^ (value >> 1)
                ^ if value & 1 != 0 { MT_MATRIX_A } else { 0 };
        }

        let value =
            (self.key[MT_N - 1] & MT_UPPER_MASK) | (self.key[0] & MT_LOWER_MASK);
        self.key[MT_N - 1] = self.key[MT_M - 1]
            ^ (value >> 1)
            ^ if value & 1 != 0 { MT_MATRIX_A } else { 0 };
        self.pos = 0;
    }
}

pub(crate) fn shuffled_indices<const N: usize>(seed: u32) -> [usize; N] {
    let mut permutation = std::array::from_fn(|index| index);
    let mut rng = LegacyMt19937::seeded(seed);

    for index in (1..N).rev() {
        let swap_with = rng.interval(index as u32) as usize;
        permutation.swap(index, swap_with);
    }

    permutation
}
