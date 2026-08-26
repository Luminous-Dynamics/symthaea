// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit identity for the clock domain attached to a chemical timestamp.
//!
//! A clock-domain ID states only that timestamps are expressed against the same
//! declared timebase. It does not prove synchronization accuracy, authenticity,
//! monotonicity, or agreement with wall-clock time.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

pub const MAX_CHEMICAL_CLOCK_DOMAIN_LEN: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalClockDomainError {
    Empty,
    TooLong { actual: usize, max: usize },
    NonCanonical,
}

impl fmt::Display for ChemicalClockDomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "chemical clock-domain ID must not be empty"),
            Self::TooLong { actual, max } => write!(
                f,
                "chemical clock-domain ID length {actual} exceeds maximum {max}"
            ),
            Self::NonCanonical => write!(
                f,
                "chemical clock-domain ID must use lowercase ASCII letters, digits, '.', '_', '-', '/', or ':'"
            ),
        }
    }
}

impl std::error::Error for ChemicalClockDomainError {}

/// Opaque identifier for one timestamp-comparison domain.
///
/// The token is intentionally semantic rather than device-derived. Several
/// sensors may share one ID when their timestamps were produced by the same
/// acquisition clock or were explicitly normalized into one common timebase.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChemicalClockDomainId(String);

impl ChemicalClockDomainId {
    pub fn new(value: impl Into<String>) -> Result<Self, ChemicalClockDomainError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ChemicalClockDomainError::Empty);
        }
        if value.len() > MAX_CHEMICAL_CLOCK_DOMAIN_LEN {
            return Err(ChemicalClockDomainError::TooLong {
                actual: value.len(),
                max: MAX_CHEMICAL_CLOCK_DOMAIN_LEN,
            });
        }
        if !value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b'-' | b'/' | b':')
        }) {
            return Err(ChemicalClockDomainError::NonCanonical);
        }
        Ok(Self(value))
    }

    /// Well-known domain for timestamps explicitly measured as microseconds
    /// since the Unix epoch. This is an assertion by the producer, not proof
    /// that the producer's clock is accurate.
    pub fn unix_epoch() -> Self {
        Self("unix-epoch".into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ChemicalClockDomainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for ChemicalClockDomainId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ChemicalClockDomainId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_clock_domains_round_trip() {
        let id = ChemicalClockDomainId::new("capture-rig-01/monotonic").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        let decoded: ChemicalClockDomainId = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, id);
    }

    #[test]
    fn invalid_wire_clock_domain_fails_closed() {
        assert!(serde_json::from_str::<ChemicalClockDomainId>("\"Bad Clock\"").is_err());
        assert!(ChemicalClockDomainId::new("").is_err());
    }

    #[test]
    fn unix_epoch_is_explicit_not_implicit() {
        assert_eq!(ChemicalClockDomainId::unix_epoch().as_str(), "unix-epoch");
    }
}
