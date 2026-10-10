// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Content verification primitives shared by instrument evidence adapters.
//!
//! These helpers prove only that supplied bytes match a declared SHA-256 digest.
//! They do not prove who authored the bytes, whether a review is independent, or
//! whether parsed metadata is authoritative. Those claims require separate
//! signature/trust-anchor checks and content-to-claim binding by the adapter.

use sha2::{Digest, Sha256};
use std::fmt;

use crate::ArtifactReference;

/// Return the lowercase hexadecimal SHA-256 digest of the exact supplied bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// An artifact's bytes after the exact content digest has been checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedArtifactBytes {
    reference: ArtifactReference,
    bytes: Vec<u8>,
}

impl VerifiedArtifactBytes {
    /// Verify owned bytes against an immutable artifact reference.
    ///
    /// The bytes are retained so an adapter can parse exactly the content that
    /// was verified, avoiding a separate later read that could change the input.
    pub fn verify(
        reference: ArtifactReference,
        bytes: Vec<u8>,
    ) -> Result<Self, ArtifactVerificationError> {
        let actual_sha256 = sha256_hex(&bytes);
        let expected_sha256 = reference.sha256_hex().to_owned();
        if actual_sha256 != expected_sha256 {
            return Err(ArtifactVerificationError::DigestMismatch {
                artifact_id: reference.artifact_id().to_owned(),
                expected_sha256,
                actual_sha256,
            });
        }

        Ok(Self { reference, bytes })
    }

    /// The exact artifact identity and digest that were checked.
    pub fn reference(&self) -> &ArtifactReference {
        &self.reference
    }

    /// The exact bytes whose digest was checked.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Number of verified bytes.
    pub fn byte_length(&self) -> usize {
        self.bytes.len()
    }
}

/// Content-integrity failures. Digest matching is not an authenticity proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactVerificationError {
    DigestMismatch {
        artifact_id: String,
        expected_sha256: String,
        actual_sha256: String,
    },
}

impl fmt::Display for ArtifactVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DigestMismatch {
                artifact_id,
                expected_sha256,
                actual_sha256,
            } => write!(
                f,
                "SHA-256 mismatch for artifact {artifact_id}: expected {expected_sha256}, got \
                 {actual_sha256}"
            ),
        }
    }
}

impl std::error::Error for ArtifactVerificationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(id: &str, digest: &str) -> ArtifactReference {
        ArtifactReference::new(id, digest).unwrap()
    }

    #[test]
    fn sha256_hex_matches_the_standard_abc_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn verified_artifact_retains_the_exact_bytes_and_reference() {
        let bytes = b"immutable fixture bytes".to_vec();
        let digest = sha256_hex(&bytes);
        let reference = reference("fixture-1", &digest);

        let verified = VerifiedArtifactBytes::verify(reference.clone(), bytes.clone()).unwrap();

        assert_eq!(verified.reference(), &reference);
        assert_eq!(verified.bytes(), bytes.as_slice());
        assert_eq!(verified.byte_length(), bytes.len());
    }

    #[test]
    fn one_byte_mutation_is_rejected_with_expected_and_actual_digests() {
        let expected_bytes = b"certificate".to_vec();
        let expected = sha256_hex(&expected_bytes);
        let reference = reference("certificate-1", &expected);
        let mut mutated = expected_bytes;
        mutated[0] ^= 0x01;
        let actual = sha256_hex(&mutated);

        assert_eq!(
            VerifiedArtifactBytes::verify(reference, mutated).unwrap_err(),
            ArtifactVerificationError::DigestMismatch {
                artifact_id: "certificate-1".to_owned(),
                expected_sha256: expected,
                actual_sha256: actual,
            }
        );
    }

    #[test]
    fn empty_content_is_hashed_exactly_instead_of_treated_as_verified_by_default() {
        let digest = sha256_hex(b"");
        let verified = VerifiedArtifactBytes::verify(reference("empty", &digest), Vec::new())
            .expect("matching empty content has a well-defined digest");
        assert_eq!(verified.byte_length(), 0);

        let incorrect = reference(
            "empty",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        assert!(matches!(
            VerifiedArtifactBytes::verify(incorrect, Vec::new()),
            Err(ArtifactVerificationError::DigestMismatch { .. })
        ));
    }
}
