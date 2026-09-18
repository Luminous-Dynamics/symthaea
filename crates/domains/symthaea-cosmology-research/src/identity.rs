// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityError {
    InvalidSha256,
    InvalidGitObjectId,
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSha256 => write!(f, "expected exactly 64 hexadecimal SHA-256 digits"),
            Self::InvalidGitObjectId => {
                write!(f, "expected a 40- or 64-digit hexadecimal Git object id")
            }
        }
    }
}

impl std::error::Error for IdentityError {}

fn is_hex(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// A validated SHA-256 digest.
///
/// The constructor and custom deserializer preserve the invariant that this
/// type always contains exactly 64 hexadecimal digits. Values are normalized
/// to lowercase so textual equality is canonical.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn parse(value: &str) -> Result<Self, IdentityError> {
        if value.len() != 64 || !is_hex(value) {
            return Err(IdentityError::InvalidSha256);
        }
        Ok(Self(value.to_ascii_lowercase()))
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
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A validated Git object identifier.
///
/// SHA-1 repositories use 40 hexadecimal digits; SHA-256 repositories use 64.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct GitObjectId(String);

impl GitObjectId {
    pub fn parse(value: &str) -> Result<Self, IdentityError> {
        if !matches!(value.len(), 40 | 64) || !is_hex(value) {
            return Err(IdentityError::InvalidGitObjectId);
        }
        Ok(Self(value.to_ascii_lowercase()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for GitObjectId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for GitObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_normalizes_case_and_rejects_malformed_values() {
        let upper = "A".repeat(64);
        let digest = Sha256Digest::parse(&upper).unwrap();
        assert_eq!(digest.as_str(), "a".repeat(64));
        assert_eq!(
            Sha256Digest::parse("abc"),
            Err(IdentityError::InvalidSha256)
        );
        assert_eq!(
            Sha256Digest::parse(&"z".repeat(64)),
            Err(IdentityError::InvalidSha256)
        );
    }

    #[test]
    fn git_identity_supports_sha1_and_sha256_lengths() {
        assert!(GitObjectId::parse(&"a".repeat(40)).is_ok());
        assert!(GitObjectId::parse(&"b".repeat(64)).is_ok());
        assert_eq!(
            GitObjectId::parse(&"c".repeat(39)),
            Err(IdentityError::InvalidGitObjectId)
        );
    }

    #[test]
    fn serde_cannot_bypass_digest_validation() {
        assert!(serde_json::from_str::<Sha256Digest>("\"short\"").is_err());
        let encoded = format!("\"{}\"", "d".repeat(64));
        let decoded: Sha256Digest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.as_str(), "d".repeat(64));
    }
}
