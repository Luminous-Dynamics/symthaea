// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Small dependency-free CRC64-ECMA helpers used by durable metadata formats.

const CRC64_ECMA_POLYNOMIAL: u64 = 0x42F0_E1EB_A9EA_3693;

/// Incremental CRC64-ECMA accumulator.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Crc64Ecma {
    value: u64,
}

impl Crc64Ecma {
    pub(crate) const fn new() -> Self {
        Self { value: 0 }
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.value ^= (byte as u64) << 56;
            for _ in 0..8 {
                self.value = if self.value & (1 << 63) != 0 {
                    (self.value << 1) ^ CRC64_ECMA_POLYNOMIAL
                } else {
                    self.value << 1
                };
            }
        }
    }

    pub(crate) const fn finalize(self) -> u64 {
        self.value
    }
}

pub(crate) fn crc64_ecma(bytes: &[u8]) -> u64 {
    let mut crc = Crc64Ecma::new();
    crc.update(bytes);
    crc.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_and_one_shot_checksums_match() {
        let bytes = b"symthaea deterministic storage";
        let mut incremental = Crc64Ecma::new();
        incremental.update(&bytes[..9]);
        incremental.update(&bytes[9..]);
        assert_eq!(incremental.finalize(), crc64_ecma(bytes));
    }

    #[test]
    fn ecma_check_value_matches_standard_vector() {
        assert_eq!(crc64_ecma(b"123456789"), 0x6C40_DF5F_0B49_7347);
    }
}
