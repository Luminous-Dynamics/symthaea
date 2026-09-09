// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cryptographic commitments for humanoid evidence and authority lineages.
//!
//! The humanoid stack historically used compact 64-bit FNV fingerprints as
//! deterministic local checksums. Those remain useful for diagnostics and fast
//! corruption detection, but they are not collision-resistant identities for an
//! adversarial operational-authority boundary. This module provides an explicit,
//! domain-separated SHA-256 commitment primitive for promotion-grade evidence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const HUMANOID_EVIDENCE_DIGEST_SCHEMA_VERSION: u32 = 1;

/// Collision-resistant identity for canonical humanoid evidence encodings.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HumanoidEvidenceDigest([u8; 32]);

impl HumanoidEvidenceDigest {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub const fn is_zero(self) -> bool {
        let bytes = self.0;
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] != 0 {
                return false;
            }
            index += 1;
        }
        true
    }

    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }
}

impl std::fmt::Debug for HumanoidEvidenceDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("HumanoidEvidenceDigest")
            .field(&self.to_hex())
            .finish()
    }
}

impl std::fmt::Display for HumanoidEvidenceDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Length-delimited, domain-separated canonical SHA-256 encoder.
///
/// Callers must feed fields in schema order. Every byte/string value is prefixed
/// with its length, preventing concatenation ambiguity. Floating-point values are
/// committed by IEEE-754 bit pattern so the digest follows the exact values used
/// by the runtime policies.
pub struct HumanoidEvidenceHasher {
    hasher: Sha256,
}

impl HumanoidEvidenceHasher {
    pub fn new(domain: &str) -> Self {
        let mut this = Self { hasher: Sha256::new() };
        this.bytes(b"symthaea.humanoid.evidence.sha256");
        this.u32(HUMANOID_EVIDENCE_DIGEST_SCHEMA_VERSION);
        this.string(domain);
        this
    }

    pub fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.u64(value.len() as u64);
        self.hasher.update(value);
        self
    }

    pub fn string(&mut self, value: &str) -> &mut Self {
        self.bytes(value.as_bytes())
    }

    pub fn bool(&mut self, value: bool) -> &mut Self {
        self.hasher.update([u8::from(value)]);
        self
    }

    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.hasher.update(value.to_le_bytes());
        self
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.hasher.update(value.to_le_bytes());
        self
    }

    pub fn usize(&mut self, value: usize) -> &mut Self {
        self.u64(value as u64)
    }

    pub fn f32(&mut self, value: f32) -> &mut Self {
        self.u32(value.to_bits())
    }

    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.u64(value.to_bits())
    }

    pub fn digest(&mut self, value: HumanoidEvidenceDigest) -> &mut Self {
        self.hasher.update(value.0);
        self
    }

    pub fn finish(self) -> HumanoidEvidenceDigest {
        let bytes: [u8; 32] = self.hasher.finalize().into();
        HumanoidEvidenceDigest(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_separation_changes_digest() {
        let mut a = HumanoidEvidenceHasher::new("reach.protocol.v1");
        a.string("same-payload");
        let mut b = HumanoidEvidenceHasher::new("reach.stage.v1");
        b.string("same-payload");
        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn length_delimiting_prevents_concatenation_ambiguity() {
        let mut a = HumanoidEvidenceHasher::new("test.v1");
        a.string("ab").string("c");
        let mut b = HumanoidEvidenceHasher::new("test.v1");
        b.string("a").string("bc");
        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn digest_hex_is_canonical_lowercase() {
        let mut hasher = HumanoidEvidenceHasher::new("hex.v1");
        hasher.string("payload");
        let hex = hasher.finish().to_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
    }
}
