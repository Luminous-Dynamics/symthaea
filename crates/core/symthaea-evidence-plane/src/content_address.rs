// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed runtime references to externally computed content identities.
//!
//! [`ContentAddress32`] is deliberately smaller in scope than an evidence
//! signature or trust assertion. It says only:
//!
//! > under this named digest algorithm and semantic namespace, these 32 bytes
//! > identify some content.
//!
//! The producing domain remains responsible for canonicalizing the content and
//! computing the digest. This crate does not hash arbitrary Rust values, does not
//! assign epistemic authority, and does not authenticate who produced the bytes.
//! Authenticity/signatures and causal provenance are separate layers.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};

const MAX_ALGORITHM_LEN: usize = 32;
const MAX_NAMESPACE_LEN: usize = 160;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentAddressError {
    EmptyAlgorithm,
    AlgorithmTooLong { actual: usize, max: usize },
    InvalidAlgorithmCharacter,
    EmptyNamespace,
    NamespaceTooLong { actual: usize, max: usize },
    InvalidNamespaceCharacter,
}

impl fmt::Display for ContentAddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAlgorithm => write!(f, "content-address algorithm must not be empty"),
            Self::AlgorithmTooLong { actual, max } => write!(
                f,
                "content-address algorithm length {actual} exceeds maximum {max}"
            ),
            Self::InvalidAlgorithmCharacter => write!(
                f,
                "content-address algorithm contains a character outside [a-z0-9_-]"
            ),
            Self::EmptyNamespace => write!(f, "content-address namespace must not be empty"),
            Self::NamespaceTooLong { actual, max } => write!(
                f,
                "content-address namespace length {actual} exceeds maximum {max}"
            ),
            Self::InvalidNamespaceCharacter => write!(
                f,
                "content-address namespace contains a character outside [A-Za-z0-9_-]"
            ),
        }
    }
}

impl std::error::Error for ContentAddressError {}

/// A namespaced 32-byte content address whose digest was computed externally.
///
/// `algorithm` identifies the digest algorithm (for example `blake3-256` or
/// `sha256`). `namespace` identifies the semantic canonicalization contract (for
/// example `symthaea-chemosensation-evidence-bundle-v1`). The same digest bytes
/// under a different algorithm or namespace are intentionally different
/// addresses.
///
/// This type is **content identity, not authenticity**. A caller that requires
/// cryptographic provenance must separately verify a signature, trusted timestamp,
/// attestation, or other authentication evidence over the addressed content.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct ContentAddress32 {
    algorithm: String,
    namespace: String,
    digest: [u8; 32],
}

impl ContentAddress32 {
    pub fn new(
        algorithm: impl Into<String>,
        namespace: impl Into<String>,
        digest: [u8; 32],
    ) -> Result<Self, ContentAddressError> {
        let algorithm = algorithm.into();
        let namespace = namespace.into();
        validate_algorithm(&algorithm)?;
        validate_namespace(&namespace)?;
        Ok(Self {
            algorithm,
            namespace,
            digest,
        })
    }

    pub fn algorithm(&self) -> &str {
        &self.algorithm
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn into_digest(self) -> [u8; 32] {
        self.digest
    }

    /// A stable textual rendering suitable for logs and receipts.
    ///
    /// Parsing is intentionally not provided yet: machine interchange should use
    /// the structured serde form so an unescaped delimiter cannot become part of
    /// an identity contract by accident.
    pub fn to_canonical_string(&self) -> String {
        self.to_string()
    }
}

#[derive(Deserialize)]
struct ContentAddress32Wire {
    algorithm: String,
    namespace: String,
    digest: [u8; 32],
}

impl<'de> Deserialize<'de> for ContentAddress32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ContentAddress32Wire::deserialize(deserializer)?;
        Self::new(wire.algorithm, wire.namespace, wire.digest).map_err(de::Error::custom)
    }
}

impl fmt::Display for ContentAddress32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:", self.algorithm, self.namespace)?;
        for byte in self.digest {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

fn validate_algorithm(value: &str) -> Result<(), ContentAddressError> {
    if value.is_empty() {
        return Err(ContentAddressError::EmptyAlgorithm);
    }
    if value.len() > MAX_ALGORITHM_LEN {
        return Err(ContentAddressError::AlgorithmTooLong {
            actual: value.len(),
            max: MAX_ALGORITHM_LEN,
        });
    }
    // Keep algorithm identifiers compact and unambiguous in logs/receipts.
    // Lowercase ASCII, digits, '-' and '_' cover the algorithms currently used
    // across the workspace without imposing a cryptographic registry here.
    if !value.bytes().all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
    }) {
        return Err(ContentAddressError::InvalidAlgorithmCharacter);
    }
    Ok(())
}

fn validate_namespace(value: &str) -> Result<(), ContentAddressError> {
    if value.is_empty() {
        return Err(ContentAddressError::EmptyNamespace);
    }
    if value.len() > MAX_NAMESPACE_LEN {
        return Err(ContentAddressError::NamespaceTooLong {
            actual: value.len(),
            max: MAX_NAMESPACE_LEN,
        });
    }
    // Namespace is intentionally ASCII and delimiter-safe so Display remains
    // unambiguous. Dots/slashes/colons are excluded; use '-' or '_' for hierarchy.
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(ContentAddressError::InvalidNamespaceCharacter);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address(namespace: &str, byte: u8) -> ContentAddress32 {
        ContentAddress32::new("blake3-256", namespace, [byte; 32]).unwrap()
    }

    #[test]
    fn same_components_are_equal_and_display_deterministically() {
        let a = address("symthaea-test-v1", 0xab);
        let b = address("symthaea-test-v1", 0xab);
        assert_eq!(a, b);
        assert_eq!(
            a.to_canonical_string(),
            format!("blake3-256:symthaea-test-v1:{}", "ab".repeat(32))
        );
    }

    #[test]
    fn namespace_is_part_of_identity() {
        let evidence = address("symthaea-evidence-v1", 7);
        let representation = address("symthaea-representation-v1", 7);
        assert_ne!(evidence, representation);
    }

    #[test]
    fn algorithm_is_part_of_identity() {
        let blake = ContentAddress32::new("blake3-256", "symthaea-test-v1", [7; 32]).unwrap();
        let sha = ContentAddress32::new("sha256", "symthaea-test-v1", [7; 32]).unwrap();
        assert_ne!(blake, sha);
    }

    #[test]
    fn malformed_identifiers_are_rejected() {
        assert_eq!(
            ContentAddress32::new("", "symthaea-test-v1", [0; 32]),
            Err(ContentAddressError::EmptyAlgorithm)
        );
        assert_eq!(
            ContentAddress32::new("BLAKE3", "symthaea-test-v1", [0; 32]),
            Err(ContentAddressError::InvalidAlgorithmCharacter)
        );
        assert_eq!(
            ContentAddress32::new("blake3-256", "", [0; 32]),
            Err(ContentAddressError::EmptyNamespace)
        );
        assert_eq!(
            ContentAddress32::new("blake3-256", "symthaea:test", [0; 32]),
            Err(ContentAddressError::InvalidNamespaceCharacter)
        );
    }

    #[test]
    fn serde_round_trip_preserves_exact_identity() {
        let original = address("symthaea-test-v1", 0x42);
        let encoded = serde_json::to_string(&original).unwrap();
        let decoded: ContentAddress32 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn serde_cannot_bypass_constructor_validation() {
        let mut encoded = serde_json::to_value(address("symthaea-test-v1", 1)).unwrap();
        encoded["algorithm"] = serde_json::Value::String("BLAKE3".into());
        assert!(serde_json::from_value::<ContentAddress32>(encoded).is_err());

        let mut encoded = serde_json::to_value(address("symthaea-test-v1", 1)).unwrap();
        encoded["namespace"] = serde_json::Value::String("symthaea:test".into());
        assert!(serde_json::from_value::<ContentAddress32>(encoded).is_err());
    }
}
