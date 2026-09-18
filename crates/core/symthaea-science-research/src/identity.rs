// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// Validated lowercase SHA-256 identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn parse(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(IdentityError::InvalidSha256);
        }
        Ok(Self(value.to_ascii_lowercase()))
    }

    pub fn of_bytes(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        let mut out = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        Self(out)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable human-readable research identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ResearchId(String);

impl ResearchId {
    pub const MAX_LEN: usize = 128;

    pub fn parse(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        if value.is_empty() || value.len() > Self::MAX_LEN {
            return Err(IdentityError::InvalidResearchId);
        }
        if !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        }) {
            return Err(IdentityError::InvalidResearchId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ResearchId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for ResearchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityError {
    InvalidSha256,
    InvalidResearchId,
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSha256 => f.write_str("expected a 64-character hexadecimal SHA-256 digest"),
            Self::InvalidResearchId => f.write_str(
                "research id must be 1..=128 ASCII characters from [A-Za-z0-9._:/-]",
            ),
        }
    }
}

impl std::error::Error for IdentityError {}

/// Domain-separated, length-prefixed SHA-256 builder.
///
/// Length-prefixing prevents concatenation ambiguity and gives every identity
/// constructor the same small deterministic framing primitive.
pub(crate) struct FramedDigest {
    bytes: Vec<u8>,
}

impl FramedDigest {
    pub(crate) fn new(domain: &str) -> Self {
        let mut this = Self { bytes: Vec::new() };
        this.text(domain);
        this
    }

    pub(crate) fn text(&mut self, value: &str) {
        self.bytes
            .extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    pub(crate) fn digest(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_deserialization_fails_closed() {
        assert!(Sha256Digest::parse("abc").is_err());
        assert!(Sha256Digest::parse("z".repeat(64)).is_err());
        assert!(serde_json::from_str::<Sha256Digest>("\"abc\"").is_err());
    }

    #[test]
    fn sha256_normalizes_hex_case() {
        let value = Sha256Digest::parse("AA".repeat(32)).unwrap();
        assert_eq!(value.as_str(), "aa".repeat(32));
    }

    #[test]
    fn research_id_deserialization_fails_closed() {
        assert!(ResearchId::parse("").is_err());
        assert!(ResearchId::parse("contains space").is_err());
        assert!(ResearchId::parse("x".repeat(ResearchId::MAX_LEN + 1)).is_err());
        assert!(serde_json::from_str::<ResearchId>("\"bad id\"").is_err());
    }

    #[test]
    fn framed_digest_is_order_and_boundary_sensitive() {
        let mut a = FramedDigest::new("test-domain");
        a.text("ab");
        a.text("c");
        let a = a.digest();

        let mut b = FramedDigest::new("test-domain");
        b.text("a");
        b.text("bc");
        let b = b.digest();

        assert_ne!(a, b);
    }
}
