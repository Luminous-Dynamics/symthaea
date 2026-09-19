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

/// Canonical domain/purpose identifier used to scope key authority.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct TrustUsage(String);

impl TrustUsage {
    pub const MAX_LEN: usize = 128;

    pub fn parse(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        if value.is_empty() || value.len() > Self::MAX_LEN {
            return Err(IdentityError::InvalidTrustUsage);
        }
        if !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
        }) {
            return Err(IdentityError::InvalidTrustUsage);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for TrustUsage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for TrustUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityError {
    InvalidSha256,
    InvalidTrustUsage,
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSha256 => {
                f.write_str("expected a 64-character hexadecimal SHA-256 digest")
            }
            Self::InvalidTrustUsage => f.write_str(
                "trust usage must be 1..=128 ASCII characters from [A-Za-z0-9._:/-]",
            ),
        }
    }
}

impl std::error::Error for IdentityError {}

/// Domain-separated, length-prefixed SHA-256 builder.
///
/// Kept crate-private so every trust identity uses one ambiguity-resistant
/// framing primitive.
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

    pub(crate) fn optional_sha(&mut self, value: Option<&Sha256Digest>) {
        match value {
            Some(value) => {
                self.text("some");
                self.text(value.as_str());
            }
            None => self.text("none"),
        }
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
    fn trust_usage_is_canonical_and_whitespace_free() {
        assert!(TrustUsage::parse("science.qualification").is_ok());
        assert!(TrustUsage::parse(" science.qualification").is_err());
        assert!(TrustUsage::parse("science qualification").is_err());
    }

    #[test]
    fn framed_digest_is_boundary_sensitive() {
        let mut left = FramedDigest::new("test");
        left.text("ab");
        left.text("c");
        let left = left.digest();

        let mut right = FramedDigest::new("test");
        right.text("a");
        right.text("bc");
        let right = right.digest();

        assert_ne!(left, right);
    }
}
