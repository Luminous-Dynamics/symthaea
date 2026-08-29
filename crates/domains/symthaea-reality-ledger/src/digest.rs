// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed, domain-separated digests used at reality boundaries.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DigestAlgorithm {
    Blake3,
    Sha256,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TypedDigest {
    /// Stable semantic namespace for the bytes that were hashed, for example
    /// `symtropy.scene-state.v1` or `symthaea.reality-ledger-head.v1`.
    pub domain: String,
    pub algorithm: DigestAlgorithm,
    /// Lower-level encoding is owned by the algorithm/host. The reality layer
    /// only requires a non-empty stable representation.
    pub value: String,
}

impl TypedDigest {
    pub fn new(
        domain: impl Into<String>,
        algorithm: DigestAlgorithm,
        value: impl Into<String>,
    ) -> Result<Self, TypedDigestError> {
        let digest = Self {
            domain: domain.into(),
            algorithm,
            value: value.into(),
        };
        digest.validate()?;
        Ok(digest)
    }

    pub fn blake3(domain: impl Into<String>, bytes: &[u8]) -> Result<Self, TypedDigestError> {
        Self::new(
            domain,
            DigestAlgorithm::Blake3,
            blake3::hash(bytes).to_hex().to_string(),
        )
    }

    pub fn validate(&self) -> Result<(), TypedDigestError> {
        validate_component(&self.domain, TypedDigestError::EmptyDomain)?;
        validate_component(&self.value, TypedDigestError::EmptyValue)?;
        if let DigestAlgorithm::Other(name) = &self.algorithm {
            validate_component(name, TypedDigestError::EmptyAlgorithm)?;
        }
        Ok(())
    }

    /// Equality suitable for state-materialization gates. Both the semantic
    /// domain and algorithm must match; equal-looking hexadecimal strings from
    /// different serializers are deliberately not interchangeable.
    pub fn same_typed_value(&self, other: &Self) -> bool {
        self.domain == other.domain
            && self.algorithm == other.algorithm
            && self.value == other.value
    }
}

fn validate_component(value: &str, empty: TypedDigestError) -> Result<(), TypedDigestError> {
    if value.trim().is_empty() {
        return Err(empty);
    }
    if value.len() > 2048 {
        return Err(TypedDigestError::ComponentTooLong);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TypedDigestError {
    #[error("digest domain may not be empty")]
    EmptyDomain,
    #[error("digest value may not be empty")]
    EmptyValue,
    #[error("custom digest algorithm name may not be empty")]
    EmptyAlgorithm,
    #[error("digest component exceeds supported length")]
    ComponentTooLong,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_hex_in_different_domains_is_not_same_state() {
        let a = TypedDigest::new("scene.v1", DigestAlgorithm::Blake3, "abc").unwrap();
        let b = TypedDigest::new("memory.v1", DigestAlgorithm::Blake3, "abc").unwrap();
        assert!(!a.same_typed_value(&b));
    }

    #[test]
    fn blake3_constructor_is_deterministic() {
        let a = TypedDigest::blake3("payload.v1", b"hello").unwrap();
        let b = TypedDigest::blake3("payload.v1", b"hello").unwrap();
        assert_eq!(a, b);
    }
}
