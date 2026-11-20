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

/// Maximum encoded size of a backend's immutable source-version identifier.
pub const MAX_IMMUTABLE_SOURCE_VERSION_ID_BYTES: usize = 512;

/// A reader pinned to one immutable object version or content snapshot.
///
/// Implementations must bind both the returned version ID and all reads to the same
/// immutable object version. A path, mutable database row, or latest-version alias
/// does not satisfy this contract.
pub trait ImmutableArtifactReader: Read {
    /// Logical artifact ID that this pinned source version belongs to.
    fn artifact_id(&self) -> &str;

    /// Stable version/generation ID for the same bytes returned by Read.
    fn immutable_version_id(&self) -> &str;
}

/// Verify a reader incrementally without retaining a second copy of its bytes.
///
/// This digest proves content integrity only. It does not assert that a later read
/// comes from the same object version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedArtifactDigest {
    reference: ArtifactReference,
    byte_length: u64,
    immutable_source_version_id: Option<String>,
}

impl VerifiedArtifactDigest {
    /// Hash a complete reader using a bounded buffer and compare it to the reference.
    /// Verify bytes against their digest without asserting source immutability.
    pub fn verify_reader<R: Read>(
        reference: ArtifactReference,
        reader: R,
    ) -> Result<Self, ArtifactVerificationError> {
        Self::verify_reader_inner(reference, reader, None)
    }

    /// Verify bytes and retain the immutable version token associated with that read.
    ///
    /// This is the only streaming verification path accepted by ResolvedRawData.
    pub fn verify_immutable_reader<R: ImmutableArtifactReader>(
        reference: ArtifactReference,
        reader: R,
    ) -> Result<Self, ArtifactVerificationError> {
        let artifact_id = reference.artifact_id().to_owned();
        let actual_artifact_id = reader.artifact_id().to_owned();
        if actual_artifact_id != artifact_id {
            return Err(ArtifactVerificationError::ImmutableSourceIdentityMismatch {
                expected_artifact_id: artifact_id,
                actual_artifact_id,
            });
        }
        let source_version_id = reader.immutable_version_id().to_owned();
        if source_version_id.is_empty()
            || source_version_id.trim() != source_version_id
            || source_version_id.len() > MAX_IMMUTABLE_SOURCE_VERSION_ID_BYTES
            || source_version_id.chars().any(char::is_control)
        {
            return Err(ArtifactVerificationError::InvalidImmutableSourceVersion {
                artifact_id,
            });
        }
        Self::verify_reader_inner(reference, reader, Some(source_version_id))
    }

    fn verify_reader_inner<R: Read>(
        reference: ArtifactReference,
        mut reader: R,
        immutable_source_version_id: Option<String>,
    ) -> Result<Self, ArtifactVerificationError> {
        let artifact_id = reference.artifact_id().to_owned();
        let mut hasher = Sha256::new();
        let mut byte_length = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];

        loop {
            let count = match reader.read(&mut buffer) {
                Ok(count) => count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    return Err(ArtifactVerificationError::ReadFailure {
                        artifact_id: artifact_id.clone(),
                    });
                }
            };
            if count == 0 {
                break;
            }

            let count_u64 = u64::try_from(count).map_err(|_| {
                ArtifactVerificationError::LengthOverflow {
                    artifact_id: artifact_id.clone(),
                }
            })?;
            byte_length = byte_length.checked_add(count_u64).ok_or_else(|| {
                ArtifactVerificationError::LengthOverflow {
                    artifact_id: artifact_id.clone(),
                }
            })?;
            hasher.update(&buffer[..count]);
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
            immutable_source_version_id,
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

    /// Version token captured from the immutable reader, if one was supplied.
    pub fn immutable_source_version_id(&self) -> Option<&str> {
        self.immutable_source_version_id.as_deref()
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
    InvalidImmutableSourceVersion {
        artifact_id: String,
    },
    ImmutableSourceIdentityMismatch {
        expected_artifact_id: String,
        actual_artifact_id: String,
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
            Self::InvalidImmutableSourceVersion { artifact_id } => {
                write!(f, "invalid immutable source version for artifact {artifact_id}")
            }
            Self::ImmutableSourceIdentityMismatch {
                expected_artifact_id,
                actual_artifact_id,
            } => write!(
                f,
                "identity mismatch: {expected_artifact_id} != {actual_artifact_id}"
            ),
        }
    }
}

impl std::error::Error for ArtifactVerificationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct VersionedCursor {
        inner: std::io::Cursor<Vec<u8>>,
        artifact_id: String,
        version_id: String,
    }

    impl VersionedCursor {
        fn new(artifact_id: &str, version_id: &str, bytes: Vec<u8>) -> Self {
            Self {
                inner: std::io::Cursor::new(bytes),
                artifact_id: artifact_id.to_owned(),
                version_id: version_id.to_owned(),
            }
        }
    }

    impl std::io::Read for VersionedCursor {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            std::io::Read::read(&mut self.inner, buffer)
        }
    }

    impl ImmutableArtifactReader for VersionedCursor {
        fn artifact_id(&self) -> &str {
            &self.artifact_id
        }

        fn immutable_version_id(&self) -> &str {
            &self.version_id
        }
    }

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
    fn immutable_reader_verification_retains_source_version_identity() {
        let bytes = b"snapshot-bound raw acquisition".to_vec();
        let reference = reference("versioned-raw", &sha256_hex(&bytes));
        let verified = VerifiedArtifactDigest::verify_immutable_reader(
            reference.clone(),
            VersionedCursor::new("versioned-raw", "object-version-17", bytes.clone()),
        )
        .unwrap();

        assert_eq!(verified.reference(), &reference);
        assert_eq!(verified.immutable_source_version_id(), Some("object-version-17"));
        assert_eq!(verified.byte_length(), u64::try_from(bytes.len()).unwrap());
    }

    #[test]
    fn immutable_reader_rejects_blank_or_noncanonical_version_identifiers() {
        let bytes = b"raw".to_vec();
        let reference = reference("versioned-raw", &sha256_hex(&bytes));
        let too_long_version = "v".repeat(MAX_IMMUTABLE_SOURCE_VERSION_ID_BYTES + 1);
        for version_id in ["", " ", "version\n2", too_long_version.as_str()] {
            assert_eq!(
                VerifiedArtifactDigest::verify_immutable_reader(
                    reference.clone(),
                    VersionedCursor::new("versioned-raw", version_id, bytes.clone()),
                )
                .unwrap_err(),
                ArtifactVerificationError::InvalidImmutableSourceVersion {
                    artifact_id: "versioned-raw".to_owned(),
                }
            );
        }
    }

    #[test]
    fn immutable_reader_rejects_artifact_identity_substitution() {
        let bytes = b"raw".to_vec();
        let reference = reference("expected-raw", &sha256_hex(&bytes));
        assert_eq!(
            VerifiedArtifactDigest::verify_immutable_reader(
                reference,
                VersionedCursor::new("different-raw", "object-version-17", bytes),
            )
            .unwrap_err(),
            ArtifactVerificationError::ImmutableSourceIdentityMismatch {
                expected_artifact_id: "expected-raw".to_owned(),
                actual_artifact_id: "different-raw".to_owned(),
            }
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
        assert_eq!(verified.byte_length(), u64::try_from(bytes.len()).unwrap());
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
    fn streaming_verifier_retries_interrupted_reads() {
        struct InterruptedOnceReader {
            interrupted: bool,
            inner: std::io::Cursor<Vec<u8>>,
        }

        impl std::io::Read for InterruptedOnceReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if !self.interrupted {
                    self.interrupted = true;
                    return Err(std::io::Error::from(
                        std::io::ErrorKind::Interrupted,
                    ));
                }
                std::io::Read::read(&mut self.inner, buffer)
            }
        }

        let bytes = b"resumed stream".to_vec();
        let reference = reference("interrupted-once", &sha256_hex(&bytes));
        let reader = InterruptedOnceReader {
            interrupted: false,
            inner: std::io::Cursor::new(bytes.clone()),
        };

        let verified = VerifiedArtifactDigest::verify_reader(reference, reader).unwrap();
        assert_eq!(verified.byte_length(), u64::try_from(bytes.len()).unwrap());
    }

    #[test]
    fn streaming_verifier_fails_closed_when_the_reader_errors() {
        struct FailingReader;

        impl std::io::Read for FailingReader {
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
