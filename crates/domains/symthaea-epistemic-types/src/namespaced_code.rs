// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical machine identifiers for epistemic claims, loss codes, and profiles.
//!
//! These identifiers are intentionally narrower than arbitrary display strings.
//! They are byte-stable, lower-case ASCII, and safe to embed in authority-bearing
//! transcripts after the containing object has been independently validated.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

pub const NAMESPACED_CODE_V1_MAX_BYTES: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NamespacedCodeError {
    Empty,
    TooLong { len: usize },
    InvalidByte { index: usize, byte: u8 },
    InvalidSeparatorPosition { index: usize },
}

impl fmt::Display for NamespacedCodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "namespaced code must not be empty"),
            Self::TooLong { len } => write!(
                f,
                "namespaced code is {len} bytes; maximum is {NAMESPACED_CODE_V1_MAX_BYTES}"
            ),
            Self::InvalidByte { index, byte } => write!(
                f,
                "namespaced code contains invalid byte 0x{byte:02x} at index {index}"
            ),
            Self::InvalidSeparatorPosition { index } => write!(
                f,
                "namespaced code contains a leading, trailing, or repeated separator at index {index}"
            ),
        }
    }
}

impl std::error::Error for NamespacedCodeError {}

/// Canonical V1 machine code used for claim/profile/loss identities.
///
/// Grammar:
///
/// ```text
/// 1..128 bytes
/// lower-case ASCII only
/// alphanumeric tokens separated by '.', '-' or '_'
/// no leading/trailing separator
/// no adjacent separators
/// ```
///
/// This type is deliberately distinct from human-readable labels. Display text
/// may change without changing the machine identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct NamespacedCodeV1(String);

impl NamespacedCodeV1 {
    pub fn new(value: impl Into<String>) -> Result<Self, NamespacedCodeError> {
        let value = value.into();
        validate_namespaced_code(&value)?;
        Ok(Self(value))
    }

    pub fn validate(&self) -> Result<(), NamespacedCodeError> {
        validate_namespaced_code(&self.0)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for NamespacedCodeV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for NamespacedCodeV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

fn validate_namespaced_code(value: &str) -> Result<(), NamespacedCodeError> {
    let bytes = value.as_bytes();
    if bytes.is_empty() {
        return Err(NamespacedCodeError::Empty);
    }
    if bytes.len() > NAMESPACED_CODE_V1_MAX_BYTES {
        return Err(NamespacedCodeError::TooLong { len: bytes.len() });
    }

    let mut previous_was_separator = false;
    for (index, byte) in bytes.iter().copied().enumerate() {
        let alphanumeric = byte.is_ascii_lowercase() || byte.is_ascii_digit();
        let separator = matches!(byte, b'.' | b'-' | b'_');
        if !alphanumeric && !separator {
            return Err(NamespacedCodeError::InvalidByte { index, byte });
        }
        if separator
            && (index == 0
                || index + 1 == bytes.len()
                || previous_was_separator)
        {
            return Err(NamespacedCodeError::InvalidSeparatorPosition { index });
        }
        previous_was_separator = separator;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_canonical_machine_codes() {
        for value in [
            "muse.lifecycle.collection-irreversibly-closed",
            "qualification.engineering.rust-tests-pass",
            "projection.loss.display_metadata",
            "v1",
        ] {
            let code = NamespacedCodeV1::new(value).unwrap();
            assert_eq!(code.as_str(), value);
            code.validate().unwrap();
        }
    }

    #[test]
    fn rejects_noncanonical_codes() {
        for value in [
            "",
            "Muse.lifecycle.valid",
            "muse..valid",
            "muse.-valid",
            ".muse",
            "muse.",
            "muse valid",
            "muse/valid",
            "muse.é",
        ] {
            assert!(NamespacedCodeV1::new(value).is_err(), "accepted {value:?}");
        }
    }

    #[test]
    fn rejects_overlong_code() {
        assert!(matches!(
            NamespacedCodeV1::new("a".repeat(NAMESPACED_CODE_V1_MAX_BYTES + 1)),
            Err(NamespacedCodeError::TooLong { .. })
        ));
    }

    #[test]
    fn serde_deserialization_is_fail_closed() {
        let good: NamespacedCodeV1 =
            serde_json::from_str("\"muse.claim.valid\"").unwrap();
        assert_eq!(good.as_str(), "muse.claim.valid");
        assert!(serde_json::from_str::<NamespacedCodeV1>("\"Muse.claim.valid\"").is_err());
    }
}
