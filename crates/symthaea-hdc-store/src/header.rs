// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 128-byte file header for HdcStore files.

/// Magic bytes identifying an HdcStore file.
pub const MAGIC: [u8; 8] = *b"HDCSTORE";

/// Current file format version.
pub const VERSION: u32 = 1;

/// Size of the header in bytes (128 bytes, cache-line aligned).
pub const HEADER_SIZE: usize = 128;

/// Size of each entry: 32-byte metadata + 2048-byte BinaryHV.
/// The 32-byte metadata prefix ensures BinaryHV data is 32-byte aligned
/// (matching BinaryHV's `#[repr(align(32))]`).
pub const ENTRY_SIZE: usize = 2080;

/// Byte offset of HV data within an entry (after metadata prefix).
pub const ENTRY_HV_OFFSET: usize = 32;

/// Status byte values for entries.
pub const STATUS_LIVE: u8 = 1;
pub const STATUS_TOMBSTONE: u8 = 2;

/// On-disk header, mapped directly from the first 128 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StoreHeader {
    /// Magic bytes: b"HDCSTORE"
    pub magic: [u8; 8],
    /// Format version (currently 1)
    pub version: u32,
    /// Total entries ever written (including tombstones)
    pub vector_count: u64,
    /// Currently live (non-tombstoned) entries
    pub live_count: u64,
    /// Number of tombstoned entries
    pub tombstone_count: u64,
    /// Offset where LSH index begins (0 = no LSH index)
    pub lsh_offset: u64,
    /// Number of LSH bands
    pub lsh_bands: u32,
    /// Number of LSH rows per band
    pub lsh_rows: u32,
    /// Reserved for future use
    pub _reserved: [u8; 64],
}

impl StoreHeader {
    /// Create a new header with default LSH config.
    pub fn new() -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            vector_count: 0,
            live_count: 0,
            tombstone_count: 0,
            lsh_offset: 0,
            lsh_bands: 10,
            lsh_rows: 32,
            _reserved: [0; 64],
        }
    }

    /// Validate the header against expected values.
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
        Ok(())
    }

    /// Byte offset where the data section begins.
    pub fn data_offset(&self) -> usize {
        HEADER_SIZE
    }

    /// Byte offset for entry at the given index.
    pub fn entry_offset(&self, index: u64) -> usize {
        HEADER_SIZE + (index as usize) * ENTRY_SIZE
    }

    /// Convert header to bytes for writing.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..20].copy_from_slice(&self.vector_count.to_le_bytes());
        buf[20..28].copy_from_slice(&self.live_count.to_le_bytes());
        buf[28..36].copy_from_slice(&self.tombstone_count.to_le_bytes());
        buf[36..44].copy_from_slice(&self.lsh_offset.to_le_bytes());
        buf[44..48].copy_from_slice(&self.lsh_bands.to_le_bytes());
        buf[48..52].copy_from_slice(&self.lsh_rows.to_le_bytes());
        buf
    }

    /// Parse header from bytes.
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Self {
        Self {
            magic: buf[0..8].try_into().unwrap(),
            version: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            vector_count: u64::from_le_bytes(buf[12..20].try_into().unwrap()),
            live_count: u64::from_le_bytes(buf[20..28].try_into().unwrap()),
            tombstone_count: u64::from_le_bytes(buf[28..36].try_into().unwrap()),
            lsh_offset: u64::from_le_bytes(buf[36..44].try_into().unwrap()),
            lsh_bands: u32::from_le_bytes(buf[44..48].try_into().unwrap()),
            lsh_rows: u32::from_le_bytes(buf[48..52].try_into().unwrap()),
            _reserved: buf[52..116].try_into().unwrap_or([0; 64]),
        }
    }
}

impl Default for StoreHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let mut h = StoreHeader::new();
        h.vector_count = 42;
        h.live_count = 30;
        h.tombstone_count = 12;
        let bytes = h.to_bytes();
        let h2 = StoreHeader::from_bytes(&bytes);
        assert_eq!(h2.magic, MAGIC);
        assert_eq!(h2.version, VERSION);
        assert_eq!(h2.vector_count, 42);
        assert_eq!(h2.live_count, 30);
        assert_eq!(h2.tombstone_count, 12);
    }

    #[test]
    fn header_validate_ok() {
        let h = StoreHeader::new();
        assert!(h.validate().is_ok());
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
    fn entry_offset_calculation() {
        let h = StoreHeader::new();
        assert_eq!(h.entry_offset(0), HEADER_SIZE);
        assert_eq!(h.entry_offset(1), HEADER_SIZE + ENTRY_SIZE);
        assert_eq!(h.entry_offset(10), HEADER_SIZE + 10 * ENTRY_SIZE);
    }

    #[test]
    fn hv_data_alignment() {
        let h = StoreHeader::new();
        // Verify HV data is 32-byte aligned for all entry indices
        for i in 0..100 {
            let hv_offset = h.entry_offset(i) + ENTRY_HV_OFFSET;
            assert_eq!(
                hv_offset % 32,
                0,
                "HV data at index {i} not 32-byte aligned"
            );
        }
    }
}
