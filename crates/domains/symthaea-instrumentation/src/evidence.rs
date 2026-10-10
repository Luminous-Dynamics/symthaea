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
use std::io::Read;

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

/// Verify a reader incrementally without retaining a second copy of its bytes.
///
/// The returned digest binds the bytes consumed through EOF. Callers must ensure
/// the underlying object remains immutable (or persist the same stream) before
/// using data addressed by this digest; this result does not lock a mutable file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedArtifactDigest {
    reference: ArtifactReference,
    byte_length: u64,
}

impl VerifiedArtifactDigest {
    /// Hash a complete reader using a bounded buffer and compare it to the reference.
    pub fn verify_reader<R: Read>(
        reference: ArtifactReference,
        mut reader: R,
    ) -> Result<Self, ArtifactVerificationError> {
        let artifact_id = reference.artifact_id().to_owned();
        let mut hasher = Sha256::new();
        let mut byte_length = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];

        loop {
            let count = reader.read(&mut buffer).map_err(|_| {
                ArtifactVerificationError::ReadFailure {
                    artifact_id: artifact_id.clone(),
                }
            })?;
            if count == 0 {
                break;
            }

            let count = u64::try_from(count).map_err(|_| {
                ArtifactVerificationError::LengthOverflow {
                    artifact_id: artifact_id.clone(),
                }
            })?;
            byte_length = byte_length.checked_add(count).ok_or_else(|| {
                ArtifactVerificationError::LengthOverflow {
                    artifact_id: artifact_id.clone(),
                }
            })?;
            hasher.update(&buffer[..count as usize]);
        }

        let actual_sha256 = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let expected_sha256 = reference.sha256_hex().to_owned();
        if actual_sha256 != expected_sha256 {
            return Err(ArtifactVerificationError::DigestMismatch {
                artifact_id,
                expected_sha256,
                actual_sha256,
            });
        }

        Ok(Self {
            reference,
            byte_length,
        })
    }

    /// Exact artifact identity and digest that were checked.
    pub fn reference(&self) -> &ArtifactReference {
        &self.reference
    }

    /// Number of bytes consumed through EOF.
    pub fn byte_length(&self) -> u64 {
        self.byte_length
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
    ReadFailure {
        artifact_id: String,
    },
    LengthOverflow {
        artifact_id: String,
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
            Self::ReadFailure { artifact_id } => {
                write!(f, "failed to read bytes for artifact {artifact_id}")
            }
            Self::LengthOverflow { artifact_id } => {
                write!(f, "byte length overflow while verifying artifact {artifact_id}")
            }
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

    #[test]
    fn streaming_verifier_hashes_large_content_without_buffering_the_whole_artifact() {
        let bytes = (0..200_000_usize)
            .map(|index| (index as u8).wrapping_mul(13))
            .collect::<Vec<_>>();
        let digest = sha256_hex(&bytes);
        let reference = reference("large-raw-acquisition", &digest);

        let verified = VerifiedArtifactDigest::verify_reader(
            reference.clone(),
            std::io::Cursor::new(bytes.clone()),
        )
        .unwrap();

        assert_eq!(verified.reference(), &reference);
        assert_eq!(verified.byte_length(), bytes.len() as u64);
    }

    #[test]
    fn streaming_verifier_rejects_a_mutated_reader() {
        let expected = b"original acquisition".to_vec();
        let reference = reference("raw-stream", &sha256_hex(&expected));
        let mutated = b"mutated acquisition".to_vec();

        assert!(matches!(
            VerifiedArtifactDigest::verify_reader(
                reference,
                std::io::Cursor::new(mutated),
            ),
            Err(ArtifactVerificationError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn streaming_verifier_fails_closed_when_the_reader_errors() {
        struct FailingReader;

        impl Read for FailingReader {
            fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("fixture read failure"))
            }
        }

        assert_eq!(
            VerifiedArtifactDigest::verify_reader(
                reference("unavailable", &sha256_hex(b"expected")),
                FailingReader,
            )
            .unwrap_err(),
            ArtifactVerificationError::ReadFailure {
                artifact_id: "unavailable".to_owned(),
            }
        );
    }
}
