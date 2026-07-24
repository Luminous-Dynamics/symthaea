// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic logical checksums for live HDC content.

use std::fmt;

use symthaea_core::hdc::BinaryHV;

use crate::checksum::Crc64Ecma;

const DOMAIN: &[u8] = b"symthaea-hdc-store:logical-content:v1\0";

/// Versioned, non-cryptographic checksum of the logical live vector set.
///
/// The checksum is computed over ascending `(id, BinaryHV bytes)` records and
/// intentionally excludes physical entry indexes, tombstones, capacity,
/// header generation, and index sidecars. Equal values therefore establish
/// deterministic logical equivalence across compaction and migration, but they
/// are not suitable as an authenticity or adversarial-integrity primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StoreContentChecksum {
    pub version: u32,
    pub live_count: u64,
    pub crc64_ecma: u64,
}

impl StoreContentChecksum {
    pub const VERSION: u32 = 1;

    /// Fixed-width lowercase hexadecimal checksum.
    pub fn hex(self) -> String {
        format!("{:016x}", self.crc64_ecma)
    }
}

impl fmt::Display for StoreContentChecksum {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "hdc-content-v{}:{}:{:016x}",
            self.version, self.live_count, self.crc64_ecma
        )
    }
}

/// Compute a deterministic logical checksum from records already ordered by ID.
pub(crate) struct ContentChecksumBuilder {
    checksum: Crc64Ecma,
    live_count: u64,
    records_seen: u64,
}

impl ContentChecksumBuilder {
    pub(crate) fn new(live_count: u64) -> Self {
        let mut checksum = Crc64Ecma::new();
        checksum.update(DOMAIN);
        checksum.update(&StoreContentChecksum::VERSION.to_le_bytes());
        checksum.update(&live_count.to_le_bytes());
        Self {
            checksum,
            live_count,
            records_seen: 0,
        }
    }

    pub(crate) fn update(&mut self, id: u64, hv_bytes: &[u8; 2048]) {
        self.checksum.update(&id.to_le_bytes());
        self.checksum.update(hv_bytes);
        self.records_seen = self.records_seen.saturating_add(1);
    }

    pub(crate) fn finalize(self) -> Option<StoreContentChecksum> {
        (self.records_seen == self.live_count).then(|| StoreContentChecksum {
            version: StoreContentChecksum::VERSION,
            live_count: self.live_count,
            crc64_ecma: self.checksum.finalize(),
        })
    }
}

pub(crate) fn checksum_ordered<'a>(
    records: impl IntoIterator<Item = (u64, &'a BinaryHV)>,
    live_count: u64,
) -> StoreContentChecksum {
    let mut checksum = ContentChecksumBuilder::new(live_count);
    for (id, hv) in records {
        checksum.update(id, &hv.0);
    }
    checksum
        .finalize()
        .expect("ordered iterator length must match declared live count")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_is_order_sensitive_by_construction() {
        let first = BinaryHV::random(1);
        let second = BinaryHV::random(2);
        let ordered = checksum_ordered([(1, &first), (2, &second)], 2);
        let reversed = checksum_ordered([(2, &second), (1, &first)], 2);
        assert_ne!(ordered, reversed);
    }

    #[test]
    fn display_is_stable_and_explicit() {
        let checksum = StoreContentChecksum {
            version: 1,
            live_count: 3,
            crc64_ecma: 0x1234,
        };
        assert_eq!(checksum.hex(), "0000000000001234");
        assert_eq!(checksum.to_string(), "hdc-content-v1:3:0000000000001234");
    }
}
