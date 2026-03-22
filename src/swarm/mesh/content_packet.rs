// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Content Announce Packet — Resonance-Based Content Discovery
//!
//! Defines `PayloadType::ContentAnnounce` (8) for gossiping content
//! announcements over the mesh. Carries a truncated HDV embedding
//! for resonance-based filtering at the network layer.
//!
//! # Wire Format (in BinaryHV)
//! ```text
//! Bytes 0-31:   content_hash    (BLAKE3 hash of content)
//! Bytes 32-63:  truncated_hdv   (first 256 bits of content HDV embedding)
//! Bytes 64:     domain_len      (u8)
//! Bytes 65-128: domain_str      (UTF-8, max 64 bytes)
//! Bytes 129-136: created_at     (u64 LE, Unix seconds)
//! ```

use symthaea_core::hdc::BinaryHV;

/// Parsed content announcement.
#[derive(Debug, Clone)]
pub struct ContentAnnounce {
    /// BLAKE3 hash of the content.
    pub content_hash: [u8; 32],
    /// Truncated HDV embedding (256 bits = 32 bytes).
    pub truncated_hdv: [u8; 32],
    /// Content domain.
    pub domain: String,
    /// Creation timestamp (Unix seconds).
    pub created_at: u64,
}

impl ContentAnnounce {
    /// Create from a full BinaryHV embedding (truncates to 256 bits).
    pub fn from_full_hdv(
        content_hash: [u8; 32],
        full_hdv: &BinaryHV,
        domain: String,
        created_at: u64,
    ) -> Self {
        let bytes = &full_hdv.0;
        let mut truncated_hdv = [0u8; 32];
        let copy_len = bytes.len().min(32);
        truncated_hdv[..copy_len].copy_from_slice(&bytes[..copy_len]);
        Self {
            content_hash,
            truncated_hdv,
            domain,
            created_at,
        }
    }

    /// Encode into a BinaryHV for mesh transmission.
    pub fn encode(&self) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        bytes[0..32].copy_from_slice(&self.content_hash);
        bytes[32..64].copy_from_slice(&self.truncated_hdv);
        let domain_bytes = self.domain.as_bytes();
        let domain_len = domain_bytes.len().min(64) as u8;
        bytes[64] = domain_len;
        bytes[65..65 + domain_len as usize].copy_from_slice(&domain_bytes[..domain_len as usize]);
        bytes[129..137].copy_from_slice(&self.created_at.to_le_bytes());
        BinaryHV(bytes)
    }

    /// Decode from a BinaryHV.
    pub fn decode(hv: &BinaryHV) -> Option<Self> {
        let bytes = &hv.0;
        if bytes.len() < 137 {
            return None;
        }
        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&bytes[0..32]);
        let mut truncated_hdv = [0u8; 32];
        truncated_hdv.copy_from_slice(&bytes[32..64]);
        let domain_len = bytes[64] as usize;
        if domain_len > 64 {
            return None;
        }
        let domain = std::str::from_utf8(&bytes[65..65 + domain_len])
            .ok()?
            .to_string();
        let created_at = u64::from_le_bytes(bytes[129..137].try_into().ok()?);
        Some(Self {
            content_hash,
            truncated_hdv,
            domain,
            created_at,
        })
    }

    /// Quick resonance check using truncated HDV (coarse filter).
    pub fn truncated_hamming_distance(&self, other_truncated: &[u8; 32]) -> u32 {
        self.truncated_hdv
            .iter()
            .zip(other_truncated.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_announce_roundtrip() {
        let ann = ContentAnnounce {
            content_hash: [0xAA; 32],
            truncated_hdv: [0xBB; 32],
            domain: "water".to_string(),
            created_at: 1234567890,
        };
        let hv = ann.encode();
        let decoded = ContentAnnounce::decode(&hv).unwrap();
        assert_eq!(decoded.content_hash, [0xAA; 32]);
        assert_eq!(decoded.truncated_hdv, [0xBB; 32]);
        assert_eq!(decoded.domain, "water");
        assert_eq!(decoded.created_at, 1234567890);
    }

    #[test]
    fn test_from_full_hdv() {
        let full = BinaryHV::random(777);
        let ann = ContentAnnounce::from_full_hdv([0; 32], &full, "test".to_string(), 0);
        assert_eq!(ann.truncated_hdv[..], full.0[..32]);
    }

    #[test]
    fn test_hamming_distance() {
        let a = ContentAnnounce {
            content_hash: [0; 32],
            truncated_hdv: [0xFF; 32],
            domain: String::new(),
            created_at: 0,
        };
        // All zeros vs all ones -> 256 bits different
        assert_eq!(a.truncated_hamming_distance(&[0x00; 32]), 256);
        // Identical
        assert_eq!(a.truncated_hamming_distance(&[0xFF; 32]), 0);
    }

    #[test]
    fn test_empty_domain() {
        let ann = ContentAnnounce {
            content_hash: [0; 32],
            truncated_hdv: [0; 32],
            domain: String::new(),
            created_at: 0,
        };
        let hv = ann.encode();
        let decoded = ContentAnnounce::decode(&hv).unwrap();
        assert!(decoded.domain.is_empty());
    }

    #[test]
    fn test_max_domain_length() {
        let ann = ContentAnnounce {
            content_hash: [0; 32],
            truncated_hdv: [0; 32],
            domain: "a".repeat(64),
            created_at: 0,
        };
        let hv = ann.encode();
        let decoded = ContentAnnounce::decode(&hv).unwrap();
        assert_eq!(decoded.domain.len(), 64);
    }
}
