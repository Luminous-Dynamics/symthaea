// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Checksummed, page-separated file headers for HdcStore format version 2.

/// Magic bytes identifying an HdcStore file.
pub const MAGIC: [u8; 8] = *b"HDCSTORE";

/// Legacy format version used by the original single-header layout.
pub const LEGACY_VERSION: u32 = 1;

/// Current file format version.
pub const VERSION: u32 = 2;

/// Size of the serialized header payload in bytes.
pub const HEADER_SIZE: usize = 128;

/// Each redundant header occupies its own operating-system page.
pub const HEADER_PAGE_SIZE: usize = 4096;

/// Number of independently checksummed header pages.
pub const HEADER_SLOT_COUNT: usize = 2;

/// Byte offset where format-v2 entries begin.
pub const DATA_OFFSET: usize = HEADER_PAGE_SIZE * HEADER_SLOT_COUNT;

/// Legacy format-v1 entry offset.
pub const LEGACY_DATA_OFFSET: usize = HEADER_SIZE;

/// Size of each entry: 32-byte metadata + 2048-byte BinaryHV.
/// The metadata prefix ensures BinaryHV data is 32-byte aligned.
pub const ENTRY_SIZE: usize = 2080;

/// Byte offset of HV data within an entry.
pub const ENTRY_HV_OFFSET: usize = 32;

/// Status byte values for committed entries.
pub const STATUS_LIVE: u8 = 1;
pub const STATUS_TOMBSTONE: u8 = 2;

const CHECKSUM_OFFSET: usize = 64;
const CHECKSUM_END: usize = CHECKSUM_OFFSET + 8;

/// One of the two independently flushed header pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderSlot {
    Primary,
    Secondary,
}

impl HeaderSlot {
    /// Byte offset of this slot's page in the file.
    pub const fn page_offset(self) -> usize {
        match self {
            Self::Primary => 0,
            Self::Secondary => HEADER_PAGE_SIZE,
        }
    }

    /// The other header slot, used for alternating commits.
    pub const fn other(self) -> Self {
        match self {
            Self::Primary => Self::Secondary,
            Self::Secondary => Self::Primary,
        }
    }
}

/// Parsed representation of the manually serialized format-v2 header.
///
/// This struct is never cast directly onto mapped bytes. `to_bytes` and
/// `from_bytes` define the stable little-endian wire representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreHeader {
    /// Magic bytes: b"HDCSTORE"
    pub magic: [u8; 8],
    /// Format version (currently 2)
    pub version: u32,
    /// Reserved format flags; must be zero in version 2
    pub flags: u32,
    /// Monotonic header generation used to select the newest valid slot
    pub generation: u64,
    /// Total committed entries, including tombstones
    pub vector_count: u64,
    /// Currently live entries
    pub live_count: u64,
    /// Tombstoned entries
    pub tombstone_count: u64,
    /// Offset where a persisted LSH index begins (unsupported in version 2)
    pub lsh_offset: u64,
    /// Number of LSH bands
    pub lsh_bands: u32,
    /// Number of LSH rows per band
    pub lsh_rows: u32,
    /// CRC64-ECMA checksum of the complete header with this field zeroed
    pub header_checksum: u64,
    /// Reserved for future use
    pub _reserved: [u8; 56],
}

impl StoreHeader {
    /// Create a new format-v2 header with the default LSH configuration.
    pub fn new() -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            flags: 0,
            generation: 1,
            vector_count: 0,
            live_count: 0,
            tombstone_count: 0,
            lsh_offset: 0,
            lsh_bands: 32,
            lsh_rows: 8,
            header_checksum: 0,
            _reserved: [0; 56],
        }
        .sealed()
    }

    /// Validate format-level invariants that do not require scanning entries.
    pub fn validate(&self) -> Result<(), crate::HdcStoreError> {
        if self.magic != MAGIC {
            return Err(crate::HdcStoreError::InvalidHeader {
                reason: "bad magic bytes".into(),
            });
        }
        if self.version != VERSION {
            return Err(crate::HdcStoreError::VersionMismatch {
                expected: VERSION,
                found: self.version,
            });
        }
        if self.flags != 0 {
            return Err(crate::HdcStoreError::InvalidHeader {
                reason: format!("unsupported format-v2 flags: {:#x}", self.flags),
            });
        }
        if self.generation == 0 {
            return Err(crate::HdcStoreError::InvalidHeader {
                reason: "header generation must be greater than zero".into(),
            });
        }
        if self.lsh_offset != 0 {
            return Err(crate::HdcStoreError::InvalidHeader {
                reason: format!(
                    "format version {VERSION} does not support persisted LSH data (offset={})",
                    self.lsh_offset
                ),
            });
        }
        let committed = self.live_count.checked_add(self.tombstone_count).ok_or(
            crate::HdcStoreError::ArithmeticOverflow {
                context: "header live_count + tombstone_count",
            },
        )?;
        if committed != self.vector_count {
            return Err(crate::HdcStoreError::InvalidHeader {
                reason: format!(
                    "count invariant violated: vector_count={}, live_count={}, tombstone_count={}",
                    self.vector_count, self.live_count, self.tombstone_count
                ),
            });
        }
        Ok(())
    }

    /// Validate a serialized header, including its checksum.
    pub fn validate_serialized(buf: &[u8; HEADER_SIZE]) -> Result<Self, crate::HdcStoreError> {
        let header = Self::from_bytes(buf);
        header.validate()?;
        let expected = checksum_for_serialized_header(buf);
        if header.header_checksum != expected {
            return Err(crate::HdcStoreError::HeaderChecksumMismatch {
                generation: header.generation,
                expected,
                found: header.header_checksum,
            });
        }
        Ok(header)
    }

    /// Return a copy with a checksum matching its current fields.
    pub fn sealed(mut self) -> Self {
        self.header_checksum = 0;
        let bytes = self.to_bytes_unchecked();
        self.header_checksum = crate::checksum::crc64_ecma(&bytes);
        self
    }

    /// Byte offset where the data section begins.
    pub const fn data_offset(&self) -> usize {
        DATA_OFFSET
    }

    /// Checked byte offset for the entry at `index`.
    pub fn checked_entry_offset(&self, index: u64) -> Result<usize, crate::HdcStoreError> {
        checked_entry_offset(DATA_OFFSET, index)
    }

    /// Byte length required to contain every committed entry.
    pub fn required_file_len(&self) -> Result<usize, crate::HdcStoreError> {
        required_file_len(DATA_OFFSET, self.vector_count)
    }

    /// Convert the header to its stable checksummed byte representation.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        self.sealed().to_bytes_unchecked()
    }

    fn to_bytes_unchecked(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.flags.to_le_bytes());
        buf[16..24].copy_from_slice(&self.generation.to_le_bytes());
        buf[24..32].copy_from_slice(&self.vector_count.to_le_bytes());
        buf[32..40].copy_from_slice(&self.live_count.to_le_bytes());
        buf[40..48].copy_from_slice(&self.tombstone_count.to_le_bytes());
        buf[48..56].copy_from_slice(&self.lsh_offset.to_le_bytes());
        buf[56..60].copy_from_slice(&self.lsh_bands.to_le_bytes());
        buf[60..64].copy_from_slice(&self.lsh_rows.to_le_bytes());
        buf[64..72].copy_from_slice(&self.header_checksum.to_le_bytes());
        buf[72..128].copy_from_slice(&self._reserved);
        buf
    }

    /// Parse a header from its stable little-endian byte representation.
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Self {
        let mut reserved = [0u8; 56];
        reserved.copy_from_slice(&buf[72..128]);
        Self {
            magic: buf[0..8].try_into().expect("fixed-size magic slice"),
            version: u32::from_le_bytes(buf[8..12].try_into().expect("fixed-size version slice")),
            flags: u32::from_le_bytes(buf[12..16].try_into().expect("fixed-size flags slice")),
            generation: u64::from_le_bytes(
                buf[16..24].try_into().expect("fixed-size generation slice"),
            ),
            vector_count: u64::from_le_bytes(
                buf[24..32]
                    .try_into()
                    .expect("fixed-size vector_count slice"),
            ),
            live_count: u64::from_le_bytes(
                buf[32..40].try_into().expect("fixed-size live_count slice"),
            ),
            tombstone_count: u64::from_le_bytes(
                buf[40..48]
                    .try_into()
                    .expect("fixed-size tombstone_count slice"),
            ),
            lsh_offset: u64::from_le_bytes(
                buf[48..56].try_into().expect("fixed-size lsh_offset slice"),
            ),
            lsh_bands: u32::from_le_bytes(
                buf[56..60].try_into().expect("fixed-size lsh_bands slice"),
            ),
            lsh_rows: u32::from_le_bytes(
                buf[60..64].try_into().expect("fixed-size lsh_rows slice"),
            ),
            header_checksum: u64::from_le_bytes(
                buf[64..72]
                    .try_into()
                    .expect("fixed-size header checksum slice"),
            ),
            _reserved: reserved,
        }
    }
}

impl Default for StoreHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Checked entry offset for either the legacy or current data region.
pub fn checked_entry_offset(data_offset: usize, index: u64) -> Result<usize, crate::HdcStoreError> {
    let index = usize::try_from(index).map_err(|_| crate::HdcStoreError::ArithmeticOverflow {
        context: "entry index conversion",
    })?;
    let entry_bytes =
        index
            .checked_mul(ENTRY_SIZE)
            .ok_or(crate::HdcStoreError::ArithmeticOverflow {
                context: "entry byte offset",
            })?;
    data_offset
        .checked_add(entry_bytes)
        .ok_or(crate::HdcStoreError::ArithmeticOverflow {
            context: "absolute entry byte offset",
        })
}

/// Checked file length for a fixed-size entry region.
pub fn required_file_len(
    data_offset: usize,
    vector_count: u64,
) -> Result<usize, crate::HdcStoreError> {
    let entries =
        usize::try_from(vector_count).map_err(|_| crate::HdcStoreError::ArithmeticOverflow {
            context: "vector_count conversion",
        })?;
    let entry_bytes =
        entries
            .checked_mul(ENTRY_SIZE)
            .ok_or(crate::HdcStoreError::ArithmeticOverflow {
                context: "committed entry region length",
            })?;
    data_offset
        .checked_add(entry_bytes)
        .ok_or(crate::HdcStoreError::ArithmeticOverflow {
            context: "required store file length",
        })
}

fn checksum_for_serialized_header(buf: &[u8; HEADER_SIZE]) -> u64 {
    let mut copy = *buf;
    copy[CHECKSUM_OFFSET..CHECKSUM_END].fill(0);
    crate::checksum::crc64_ecma(&copy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip_and_checksum() {
        let mut h = StoreHeader::new();
        h.vector_count = 42;
        h.live_count = 30;
        h.tombstone_count = 12;
        let bytes = h.to_bytes();
        let h2 = StoreHeader::validate_serialized(&bytes).unwrap();
        assert_eq!(h2, h.sealed());
    }

    #[test]
    fn checksum_detects_single_byte_corruption() {
        let mut bytes = StoreHeader::new().to_bytes();
        bytes[100] ^= 0x40;
        assert!(matches!(
            StoreHeader::validate_serialized(&bytes),
            Err(crate::HdcStoreError::HeaderChecksumMismatch { .. })
        ));
    }

    #[test]
    fn header_validate_ok() {
        assert!(StoreHeader::new().validate().is_ok());
    }

    #[test]
    fn header_validate_bad_magic() {
        let mut h = StoreHeader::new();
        h.magic = *b"BADMAGIC";
        assert!(h.validate().is_err());
    }

    #[test]
    fn header_validate_bad_version() {
        let mut h = StoreHeader::new();
        h.version = 99;
        assert!(h.validate().is_err());
    }

    #[test]
    fn header_validate_bad_counts() {
        let mut h = StoreHeader::new();
        h.vector_count = 2;
        h.live_count = 1;
        assert!(h.validate().is_err());
    }

    #[test]
    fn checked_entry_offset_calculation() {
        let h = StoreHeader::new();
        assert_eq!(h.checked_entry_offset(0).unwrap(), DATA_OFFSET);
        assert_eq!(h.checked_entry_offset(1).unwrap(), DATA_OFFSET + ENTRY_SIZE);
        assert_eq!(
            h.checked_entry_offset(10).unwrap(),
            DATA_OFFSET + 10 * ENTRY_SIZE
        );
    }

    #[test]
    fn header_slots_and_data_are_page_separated() {
        assert_eq!(HeaderSlot::Primary.page_offset(), 0);
        assert_eq!(HeaderSlot::Secondary.page_offset(), HEADER_PAGE_SIZE);
        assert_eq!(DATA_OFFSET, HEADER_PAGE_SIZE * 2);
    }

    #[test]
    fn hv_data_alignment() {
        let h = StoreHeader::new();
        for i in 0..100 {
            let hv_offset = h.checked_entry_offset(i).unwrap() + ENTRY_HV_OFFSET;
            assert_eq!(
                hv_offset % 32,
                0,
                "HV data at index {i} not 32-byte aligned"
            );
        }
    }
}
