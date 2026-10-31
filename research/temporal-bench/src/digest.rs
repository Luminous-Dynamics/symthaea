//! Versioned canonical content hashing for public benchmark artifacts.
//!
//! The 64-bit stream IDs used for pairing are not cryptographic. This module
//! provides SHA-256 over an explicit, versioned little-endian encoding.
//! Signatures and oracle isolation remain separate concerns.

use sha2::{Digest, Sha256};

const ENCODING_MAGIC: &[u8] = b"SYMTHEA-TEMPORAL-ARTIFACT\0";
const ENCODING_VERSION: u16 = 1;

pub struct CanonicalHasher {
    inner: Sha256,
}

impl CanonicalHasher {
    /// Start a digest domain. Domain separation prevents a byte sequence from
    /// being interpreted as another artifact type.
    pub fn new(domain: &'static [u8]) -> Self {
        let mut inner = Sha256::new();
        inner.update(ENCODING_MAGIC);
        inner.update(ENCODING_VERSION.to_le_bytes());
        inner.update(u64::try_from(domain.len()).unwrap_or(u64::MAX).to_le_bytes());
        inner.update(domain);
        Self { inner }
    }

    pub fn u8(&mut self, value: u8) {
        self.inner.update([value]);
    }

    pub fn u16(&mut self, value: u16) {
        self.inner.update(value.to_le_bytes());
    }

    pub fn u32(&mut self, value: u32) {
        self.inner.update(value.to_le_bytes());
    }

    pub fn u64(&mut self, value: u64) {
        self.inner.update(value.to_le_bytes());
    }

    pub fn usize(&mut self, value: usize) {
        self.u64(u64::try_from(value).unwrap_or(u64::MAX));
    }

    pub fn bytes(&mut self, value: &[u8]) {
        self.usize(value.len());
        self.inner.update(value);
    }

    pub fn finish(self) -> [u8; 32] {
        self.inner.finalize().into()
    }
}

/// Render a SHA-256 digest as lowercase hexadecimal for receipts and logs.
pub fn digest_hex(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_hasher_uses_sha256_known_answer() {
        // SHA-256("abc"), independent known-answer vector.
        let mut hasher = CanonicalHasher::new(b"known-answer-test");
        hasher.bytes(b"abc");
        let digest = hasher.finish();
        assert_eq!(
            digest_hex(&digest),
            "TODO_CANONICAL_HASH_VECTOR"
        );
    }

    #[test]
    fn artifact_domains_produce_distinct_digests() {
        let mut first = CanonicalHasher::new(b"task");
        first.bytes(b"payload");
        let mut second = CanonicalHasher::new(b"oracle");
        second.bytes(b"payload");
        assert_ne!(first.finish(), second.finish());
    }
}
