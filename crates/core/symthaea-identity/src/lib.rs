// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Versioned canonical lexical identity primitives.
//!
//! This crate is the production Rust mirror of qualified CORE-ID-001 profile
//! `symthaea.core.canonical-lexical-id.v1`. It defines lexical admission only:
//! exact bytes are preserved; there is no Unicode normalization, case folding,
//! namespace uniqueness, spoof defense, provenance, or authorization semantics.

#![deny(unsafe_code)]

use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::error::Error;
use std::fmt;
use std::str::FromStr;

/// Qualified CORE-ID-001 lexical-profile identifier.
pub const CANONICAL_LEXICAL_ID_V1_PROFILE: &str = "symthaea.core.canonical-lexical-id.v1";

/// Failures produced by the CORE-ID-001 lexical admission predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalIdentifierError {
    /// The domain supplied an unusable zero-byte identifier budget.
    ZeroByteLimit,
    /// The identifier contains no bytes.
    Empty,
    /// The identifier exceeds its domain-specific UTF-8 byte budget.
    TooLong {
        /// Maximum admitted UTF-8 bytes for this type.
        max_utf8_bytes: usize,
        /// Actual encoded UTF-8 byte length.
        actual_utf8_bytes: usize,
    },
    /// Raw bytes do not form strict UTF-8.
    InvalidUtf8,
    /// The first or last scalar is in the frozen V1 edge-whitespace set.
    EdgeWhitespace {
        /// Rejected Unicode scalar value.
        codepoint: u32,
    },
    /// An ASCII C0 control or DEL appears anywhere in the identifier.
    ForbiddenControl {
        /// Rejected Unicode scalar value.
        codepoint: u32,
    },
}

impl fmt::Display for CanonicalIdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroByteLimit => write!(f, "canonical identifier byte limit must be positive"),
            Self::Empty => write!(f, "canonical identifier is required"),
            Self::TooLong {
                max_utf8_bytes,
                actual_utf8_bytes,
            } => write!(
                f,
                "canonical identifier exceeds {max_utf8_bytes} UTF-8 bytes: {actual_utf8_bytes}"
            ),
            Self::InvalidUtf8 => write!(f, "canonical identifier must be valid UTF-8"),
            Self::EdgeWhitespace { codepoint } => write!(
                f,
                "canonical identifier has forbidden edge whitespace U+{codepoint:04X}"
            ),
            Self::ForbiddenControl { codepoint } => write!(
                f,
                "canonical identifier contains forbidden control U+{codepoint:04X}"
            ),
        }
    }
}

impl Error for CanonicalIdentifierError {}

/// Exact lexical identifier admitted by CORE-ID-001 V1.
///
/// `MAX_UTF8_BYTES` is a domain policy parameter, not part of the shared lexical
/// algorithm. For example, PIE may use 4,096 while another namespace uses 256.
///
/// The inner string is private. Construction and deserialization both pass
/// through the same admission predicate, so Serde cannot manufacture an invalid
/// `CanonicalIdentifier`. Generic parser allocation limits remain a transport /
/// parser concern and are not claimed by this type.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalIdentifier<const MAX_UTF8_BYTES: usize> {
    value: String,
}

impl<const MAX_UTF8_BYTES: usize> CanonicalIdentifier<MAX_UTF8_BYTES> {
    /// Canonical lexical profile implemented by this type.
    pub const PROFILE: &'static str = CANONICAL_LEXICAL_ID_V1_PROFILE;

    /// Domain-specific maximum encoded byte length.
    pub const MAX_UTF8_BYTES: usize = MAX_UTF8_BYTES;

    /// Admit an owned UTF-8 string without trimming, normalization, or mutation.
    pub fn new(value: impl Into<String>) -> Result<Self, CanonicalIdentifierError> {
        let value = value.into();
        validate_decoded::<MAX_UTF8_BYTES>(&value)?;
        Ok(Self { value })
    }

    /// Admit raw bytes as strict UTF-8 without changing accepted bytes.
    pub fn from_utf8_bytes(bytes: Vec<u8>) -> Result<Self, CanonicalIdentifierError> {
        validate_size::<MAX_UTF8_BYTES>(&bytes)?;
        let value = String::from_utf8(bytes).map_err(|_| CanonicalIdentifierError::InvalidUtf8)?;
        validate_decoded::<MAX_UTF8_BYTES>(&value)?;
        Ok(Self { value })
    }

    /// Borrow the admitted identifier as text.
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Borrow the exact admitted UTF-8 bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.value.as_bytes()
    }

    /// Consume the validated wrapper and return its exact string.
    pub fn into_string(self) -> String {
        self.value
    }
}

impl<const MAX_UTF8_BYTES: usize> fmt::Display for CanonicalIdentifier<MAX_UTF8_BYTES> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<const MAX_UTF8_BYTES: usize> AsRef<str> for CanonicalIdentifier<MAX_UTF8_BYTES> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<const MAX_UTF8_BYTES: usize> FromStr for CanonicalIdentifier<MAX_UTF8_BYTES> {
    type Err = CanonicalIdentifierError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl<const MAX_UTF8_BYTES: usize> TryFrom<String> for CanonicalIdentifier<MAX_UTF8_BYTES> {
    type Error = CanonicalIdentifierError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<const MAX_UTF8_BYTES: usize> Serialize for CanonicalIdentifier<MAX_UTF8_BYTES> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de, const MAX_UTF8_BYTES: usize> Deserialize<'de> for CanonicalIdentifier<MAX_UTF8_BYTES> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

fn validate_size<const MAX_UTF8_BYTES: usize>(
    bytes: &[u8],
) -> Result<(), CanonicalIdentifierError> {
    if MAX_UTF8_BYTES == 0 {
        return Err(CanonicalIdentifierError::ZeroByteLimit);
    }
    if bytes.is_empty() {
        return Err(CanonicalIdentifierError::Empty);
    }
    if bytes.len() > MAX_UTF8_BYTES {
        return Err(CanonicalIdentifierError::TooLong {
            max_utf8_bytes: MAX_UTF8_BYTES,
            actual_utf8_bytes: bytes.len(),
        });
    }
    Ok(())
}

fn validate_decoded<const MAX_UTF8_BYTES: usize>(
    value: &str,
) -> Result<(), CanonicalIdentifierError> {
    validate_size::<MAX_UTF8_BYTES>(value.as_bytes())?;

    let first = value
        .chars()
        .next()
        .ok_or(CanonicalIdentifierError::Empty)?;
    let last = value
        .chars()
        .next_back()
        .ok_or(CanonicalIdentifierError::Empty)?;

    if is_v1_edge_whitespace(first) {
        return Err(CanonicalIdentifierError::EdgeWhitespace {
            codepoint: first as u32,
        });
    }
    if is_v1_edge_whitespace(last) {
        return Err(CanonicalIdentifierError::EdgeWhitespace {
            codepoint: last as u32,
        });
    }

    if let Some(control) = value.chars().find(|ch| is_forbidden_control(*ch)) {
        return Err(CanonicalIdentifierError::ForbiddenControl {
            codepoint: control as u32,
        });
    }

    Ok(())
}

/// Frozen CORE-ID-001 V1 edge-whitespace membership.
fn is_v1_edge_whitespace(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0009..=0x000D
            | 0x0020
            | 0x0085
            | 0x00A0
            | 0x1680
            | 0x2000..=0x200A
            | 0x2028
            | 0x2029
            | 0x202F
            | 0x205F
            | 0x3000
    )
}

fn is_forbidden_control(ch: char) -> bool {
    let codepoint = ch as u32;
    codepoint < 0x20 || codepoint == 0x7F
}

#[cfg(test)]
mod tests {
    use super::*;

    type Id4096 = CanonicalIdentifier<4096>;
    type Id256 = CanonicalIdentifier<256>;

    const EDGE_WHITESPACE: &[u32] = &[
        0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x0020, 0x0085, 0x00A0, 0x1680, 0x2000, 0x2001,
        0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029,
        0x202F, 0x205F, 0x3000,
    ];

    #[test]
    fn profile_and_exact_bytes_are_preserved() {
        assert_eq!(Id4096::PROFILE, CANONICAL_LEXICAL_ID_V1_PROFILE);
        let value = Id4096::new("Process-A").unwrap();
        assert_eq!(value.as_str(), "Process-A");
        assert_eq!(value.as_bytes(), b"Process-A");
        assert_ne!(
            value.as_bytes(),
            Id4096::new("process-a").unwrap().as_bytes()
        );
        assert_ne!(
            Id4096::new("é").unwrap().as_bytes(),
            Id4096::new("e\u{301}").unwrap().as_bytes()
        );
    }

    #[test]
    fn every_frozen_edge_whitespace_codepoint_is_rejected_at_both_edges() {
        assert_eq!(EDGE_WHITESPACE.len(), 25);
        for codepoint in EDGE_WHITESPACE {
            let ch = char::from_u32(*codepoint).unwrap();
            let leading = format!("{ch}x");
            let trailing = format!("x{ch}");
            assert!(Id4096::new(leading).is_err(), "leading U+{codepoint:04X}");
            assert!(Id4096::new(trailing).is_err(), "trailing U+{codepoint:04X}");
        }
    }

    #[test]
    fn interior_non_control_whitespace_remains_admitted() {
        for value in ["p\u{00A0}1", "p\u{2003}1", "p\u{0085}1"] {
            assert!(Id4096::new(value).is_ok(), "{value:?}");
        }
    }

    #[test]
    fn deliberate_legacy_compatibility_edge_cases_remain_admitted() {
        for value in [
            "\u{200B}x",
            "x\u{200B}",
            "\u{FEFF}x",
            "x\u{FEFF}",
            "\u{180E}x",
            "\u{00AD}x",
        ] {
            assert!(Id4096::new(value).is_ok(), "{value:?}");
        }
    }

    #[test]
    fn c0_controls_and_del_are_rejected_anywhere() {
        for value in ["p\0x", "p\u{0001}x", "p\nx", "p\u{001F}x", "p\u{007F}x"] {
            assert!(matches!(
                Id4096::new(value),
                Err(CanonicalIdentifierError::ForbiddenControl { .. })
                    | Err(CanonicalIdentifierError::EdgeWhitespace { .. })
            ));
        }
    }

    #[test]
    fn exact_domain_byte_budgets_are_enforced() {
        assert_eq!(
            Id4096::new("a".repeat(4096)).unwrap().as_bytes().len(),
            4096
        );
        assert!(matches!(
            Id4096::new("a".repeat(4097)),
            Err(CanonicalIdentifierError::TooLong { .. })
        ));
        assert_eq!(
            Id4096::new("é".repeat(2048)).unwrap().as_bytes().len(),
            4096
        );
        assert!(Id4096::new("é".repeat(2049)).is_err());

        assert_eq!(Id256::new("a".repeat(256)).unwrap().as_bytes().len(), 256);
        assert!(Id256::new("a".repeat(257)).is_err());
        assert!(matches!(
            CanonicalIdentifier::<0>::new("x"),
            Err(CanonicalIdentifierError::ZeroByteLimit)
        ));
    }

    #[test]
    fn raw_bytes_require_strict_utf8_before_construction() {
        assert!(matches!(
            Id4096::from_utf8_bytes(vec![0xFF]),
            Err(CanonicalIdentifierError::InvalidUtf8)
        ));
        let exact = vec![b'p', b'1'];
        assert_eq!(
            Id4096::from_utf8_bytes(exact.clone()).unwrap().as_bytes(),
            exact.as_slice()
        );
    }

    #[test]
    fn serde_deserialization_reuses_the_constructor() {
        let value = Id4096::new("p1").unwrap();
        let encoded = serde_json::to_string(&value).unwrap();
        assert_eq!(encoded, "\"p1\"");
        let decoded: Id4096 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, value);

        assert!(serde_json::from_str::<Id4096>("\" p1\"").is_err());
        assert!(serde_json::from_str::<Id4096>("\"p1\\u0000\"").is_err());
        assert!(serde_json::from_str::<Id256>(&format!("\"{}\"", "a".repeat(257))).is_err());
    }
}
