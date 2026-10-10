// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable, request-bound circuit netlist input artifacts.
//!
//! This module identifies one primary netlist byte sequence. It does not claim
//! that referenced model files are included in a dependency closure, that the
//! netlist is safe to execute, or that a solver run has occurred. The executor
//! must separately validate include/library closure and run the solver inside
//! an explicitly configured sandbox before this artifact can become evidence.

use std::error::Error;
use std::fmt;

use symthaea_sim_bridge::SimulationRequest;

/// Maximum primary-netlist size accepted by this artifact boundary (4 MiB).
pub const MAX_NETLIST_BYTES: usize = 4 * 1024 * 1024;
/// Maximum size of a stable request identity.
pub const MAX_REQUEST_ID_BYTES: usize = 256;

/// A primary netlist bound to one request ID and identified by BLAKE3 of its
/// exact UTF-8 bytes. Fields are private so callers cannot mutate the bytes or
/// digest independently after construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetlistArtifact {
    request_id: String,
    bytes: Vec<u8>,
    blake3_digest: String,
}

/// Construction or request-binding failure for a netlist artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetlistArtifactError {
    EmptyRequestId,
    NonCanonicalRequestId,
    RequestIdTooLarge { actual: usize, maximum: usize },
    RequestIdControlCharacter,
    RequestIdMismatch { artifact: String, request: String },
    EmptyNetlist,
    NetlistTooLarge { actual: usize, maximum: usize },
    InvalidUtf8,
    NulByte,
    DigestMismatch,
}

impl fmt::Display for NetlistArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRequestId => f.write_str("netlist artifact request ID cannot be empty"),
            Self::NonCanonicalRequestId => f.write_str("request ID must not have leading or trailing whitespace"),
            Self::RequestIdTooLarge { actual, maximum } => write!(f, "request ID is {actual} bytes; maximum accepted size is {maximum}"),
            Self::RequestIdControlCharacter => f.write_str("request ID must not contain control characters"),
            Self::RequestIdMismatch { artifact, request } => write!(
                f,
                "netlist artifact belongs to request {artifact:?}, not {request:?}"
            ),
            Self::EmptyNetlist => f.write_str("netlist artifact cannot be empty"),
            Self::NetlistTooLarge { actual, maximum } => write!(
                f,
                "netlist is {actual} bytes; maximum accepted size is {maximum}"
            ),
            Self::InvalidUtf8 => f.write_str("netlist artifact must be valid UTF-8"),
            Self::NulByte => f.write_str("netlist artifact must not contain NUL bytes"),
            Self::DigestMismatch => {
                f.write_str("netlist artifact digest does not match its exact bytes")
            }
        }
    }
}

impl Error for NetlistArtifactError {}

impl NetlistArtifact {
    /// Construct an immutable artifact from exact netlist bytes.
    pub fn new(
        request_id: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<Self, NetlistArtifactError> {
        let request_id = request_id.into();
        if request_id.trim().is_empty() {
            return Err(NetlistArtifactError::EmptyRequestId);
        }
        if request_id.trim() != request_id.as_str() {
            return Err(NetlistArtifactError::NonCanonicalRequestId);
        }
        if request_id.len() > MAX_REQUEST_ID_BYTES {
            return Err(NetlistArtifactError::RequestIdTooLarge {
                actual: request_id.len(),
                maximum: MAX_REQUEST_ID_BYTES,
            });
        }
        if request_id.chars().any(char::is_control) {
            return Err(NetlistArtifactError::RequestIdControlCharacter);
        }

        let bytes = bytes.into();
        validate_bytes(&bytes)?;
        let blake3_digest = blake3::hash(&bytes).to_hex().to_string();

        Ok(Self {
            request_id,
            bytes,
            blake3_digest,
        })
    }

    /// Stable request identifier to which this netlist was bound at creation.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Exact UTF-8 source bytes; no line-ending or whitespace normalization.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Lowercase hexadecimal BLAKE3 digest over the exact bytes.
    pub fn blake3_digest(&self) -> &str {
        &self.blake3_digest
    }

    /// Verify the stored digest and bind the artifact to the supplied request.
    pub fn verify_for_request(
        &self,
        request: &SimulationRequest,
    ) -> Result<(), NetlistArtifactError> {
        if self.request_id != request.id {
            return Err(NetlistArtifactError::RequestIdMismatch {
                artifact: self.request_id.clone(),
                request: request.id.clone(),
            });
        }
        validate_bytes(&self.bytes)?;
        let computed = blake3::hash(&self.bytes).to_hex().to_string();
        if computed != self.blake3_digest {
            return Err(NetlistArtifactError::DigestMismatch);
        }
        Ok(())
    }
}

fn validate_bytes(bytes: &[u8]) -> Result<(), NetlistArtifactError> {
    if bytes.is_empty() || bytes.iter().all(|byte| byte.is_ascii_whitespace()) {
        return Err(NetlistArtifactError::EmptyNetlist);
    }
    if bytes.len() > MAX_NETLIST_BYTES {
        return Err(NetlistArtifactError::NetlistTooLarge {
            actual: bytes.len(),
            maximum: MAX_NETLIST_BYTES,
        });
    }
    std::str::from_utf8(bytes).map_err(|_| NetlistArtifactError::InvalidUtf8)?;
    if bytes.contains(&0) {
        return Err(NetlistArtifactError::NulByte);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{EngineeringDomain, SolverKind};

    fn request(id: &str) -> SimulationRequest {
        SimulationRequest::new(
            id,
            EngineeringDomain::Electrical,
            SolverKind::Circuit,
            "test netlist identity",
        )
    }

    #[test]
    fn digest_is_over_exact_source_bytes() {
        let bytes = b"R1 in out 1k\nC1 out 0 1u\n".to_vec();
        let artifact = NetlistArtifact::new("circuit-1", bytes.clone()).unwrap();

        assert_eq!(artifact.bytes(), bytes);
        let expected_digest = blake3::hash(&bytes).to_hex().to_string();
        assert_eq!(artifact.blake3_digest(), expected_digest);
        artifact.verify_for_request(&request("circuit-1")).unwrap();
    }

    #[test]
    fn line_ending_changes_change_the_artifact_digest() {
        let lf = NetlistArtifact::new("same-request", b"R1 a b 1k\n".to_vec()).unwrap();
        let crlf = NetlistArtifact::new("same-request", b"R1 a b 1k\r\n".to_vec()).unwrap();
        assert_ne!(lf.blake3_digest(), crlf.blake3_digest());
    }

    #[test]
    fn rejects_empty_or_whitespace_only_netlists() {
        assert_eq!(
            NetlistArtifact::new("circuit-1", Vec::<u8>::new()).unwrap_err(),
            NetlistArtifactError::EmptyNetlist
        );
        assert_eq!(
            NetlistArtifact::new("circuit-1", b" \n\t".to_vec()).unwrap_err(),
            NetlistArtifactError::EmptyNetlist
        );
    }

    #[test]
    fn rejects_invalid_utf8_and_nul_bytes() {
        assert_eq!(
            NetlistArtifact::new("circuit-1", vec![0xff]).unwrap_err(),
            NetlistArtifactError::InvalidUtf8
        );
        assert_eq!(
            NetlistArtifact::new("circuit-1", b"R1 a b 1k\0".to_vec()).unwrap_err(),
            NetlistArtifactError::NulByte
        );
    }

    #[test]
    fn rejects_oversized_netlist_before_hashing() {
        let bytes = vec![b'R'; MAX_NETLIST_BYTES + 1];
        assert!(matches!(
            NetlistArtifact::new("circuit-1", bytes),
            Err(NetlistArtifactError::NetlistTooLarge { .. })
        ));
    }

    #[test]
    fn artifact_cannot_be_used_for_a_different_request() {
        let artifact = NetlistArtifact::new("circuit-a", b"R1 a b 1k\n".to_vec()).unwrap();
        assert!(matches!(
            artifact.verify_for_request(&request("circuit-b")),
            Err(NetlistArtifactError::RequestIdMismatch { .. })
        ));
    }

    #[test]
    fn detects_digest_tampering() {
        let mut artifact =
            NetlistArtifact::new("circuit-a", b"R1 a b 1k\n".to_vec()).unwrap();
        artifact.blake3_digest.push('0');
        assert_eq!(
            artifact.verify_for_request(&request("circuit-a")),
            Err(NetlistArtifactError::DigestMismatch)
        );
    }

    #[test]
    fn rejects_empty_request_identity() {
        assert_eq!(
            NetlistArtifact::new("  ", b"R1 a b 1k\n".to_vec()).unwrap_err(),
            NetlistArtifactError::EmptyRequestId
        );
    }

    #[test]
    fn rejects_noncanonical_or_unbounded_request_identity() {
        assert_eq!(
            NetlistArtifact::new(" circuit-1", b"R1 a b 1k\n".to_vec()).unwrap_err(),
            NetlistArtifactError::NonCanonicalRequestId
        );
        assert!(matches!(
            NetlistArtifact::new(
                "x".repeat(MAX_REQUEST_ID_BYTES + 1),
                b"R1 a b 1k\n".to_vec()
            ),
            Err(NetlistArtifactError::RequestIdTooLarge { .. })
        ));
        assert_eq!(
            NetlistArtifact::new("circuit\n1", b"R1 a b 1k\n".to_vec()).unwrap_err(),
            NetlistArtifactError::RequestIdControlCharacter
        );
    }
}
