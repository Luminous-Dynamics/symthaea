//! Persistent pinned-peer enrollment and dual-signed endpoint-key rollover.
//!
//! Iroh endpoint identities are cryptographic keys. Production deployments
//! therefore need a bounded, serializable allowlist and an explicit rollover
//! ceremony rather than silently replacing endpoint IDs after a reinstall.

use crate::networking::{MAX_CLOCK_SKEW_MS, decode_bounded, encode_bounded, system_time_ms};
use iroh::{EndpointId, SecretKey, Signature};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const DIRECT_ENROLLMENT_BOOK_VERSION: u16 = 1;
pub const DIRECT_ROLLOVER_VERSION: u16 = 1;
pub const DEFAULT_ROLLOVER_TTL_MS: u64 = 15 * 60 * 1_000;
pub const MAX_ROLLOVER_TTL_MS: u64 = 24 * 60 * 60 * 1_000;
pub const MAX_ENROLLMENT_BOOK_BYTES: usize = 512 * 1024;
pub const MAX_ENROLLMENT_PEERS: usize = 16_384;

/// Versioned, bounded persistence format for pinned endpoint identities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectEnrollmentBook {
    pub protocol_version: u16,
    pub peers: Vec<EndpointId>,
}

impl DirectEnrollmentBook {
    pub fn empty() -> Self {
        Self {
            protocol_version: DIRECT_ENROLLMENT_BOOK_VERSION,
            peers: Vec::new(),
        }
    }

    pub fn from_peers<I>(peers: I) -> Result<Self, DirectEnrollmentError>
    where
        I: IntoIterator<Item = EndpointId>,
    {
        let mut peers = peers.into_iter().collect::<Vec<_>>();
        peers.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        peers.dedup();
        let book = Self {
            protocol_version: DIRECT_ENROLLMENT_BOOK_VERSION,
            peers,
        };
        book.validate()?;
        Ok(book)
    }

    pub fn validate(&self) -> Result<(), DirectEnrollmentError> {
        if self.protocol_version != DIRECT_ENROLLMENT_BOOK_VERSION {
            return Err(DirectEnrollmentError::UnsupportedBookVersion {
                received: self.protocol_version,
            });
        }
        if self.peers.len() > MAX_ENROLLMENT_PEERS {
            return Err(DirectEnrollmentError::TooManyPeers {
                count: self.peers.len(),
                maximum: MAX_ENROLLMENT_PEERS,
            });
        }
        let mut unique = HashSet::with_capacity(self.peers.len());
        if self.peers.iter().any(|peer| !unique.insert(*peer)) {
            return Err(DirectEnrollmentError::DuplicatePeer);
        }
        Ok(())
    }

    pub fn contains(&self, peer: EndpointId) -> bool {
        self.peers.contains(&peer)
    }

    pub fn encode(&self) -> Result<Vec<u8>, DirectEnrollmentError> {
        self.validate()?;
        encode_bounded(self, MAX_ENROLLMENT_BOOK_BYTES).map_err(DirectEnrollmentError::Codec)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DirectEnrollmentError> {
        let book: Self = decode_bounded(bytes, MAX_ENROLLMENT_BOOK_BYTES)
            .map_err(DirectEnrollmentError::Codec)?;
        book.validate()?;
        Ok(book)
    }
}

/// Proof that both the old and replacement endpoint keys consent to rollover.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectPeerRolloverProof {
    pub protocol_version: u16,
    pub old_peer: EndpointId,
    pub new_peer: EndpointId,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub old_signature: Signature,
    pub new_signature: Signature,
}

#[derive(Serialize)]
struct RolloverSigningView {
    protocol_version: u16,
    old_peer: EndpointId,
    new_peer: EndpointId,
    issued_at_ms: u64,
    expires_at_ms: u64,
}

impl DirectPeerRolloverProof {
    pub fn sign(
        old_key: &SecretKey,
        new_key: &SecretKey,
        issued_at_ms: u64,
        ttl_ms: u64,
    ) -> Result<Self, DirectEnrollmentError> {
        if ttl_ms == 0 || ttl_ms > MAX_ROLLOVER_TTL_MS {
            return Err(DirectEnrollmentError::InvalidRolloverTtl { ttl_ms });
        }
        let old_peer = old_key.public();
        let new_peer = new_key.public();
        if old_peer == new_peer {
            return Err(DirectEnrollmentError::RolloverIdentityUnchanged);
        }
        let expires_at_ms = issued_at_ms
            .checked_add(ttl_ms)
            .ok_or(DirectEnrollmentError::TimestampOverflow)?;
        let bytes = rollover_signing_bytes(
            DIRECT_ROLLOVER_VERSION,
            old_peer,
            new_peer,
            issued_at_ms,
            expires_at_ms,
        )?;
        Ok(Self {
            protocol_version: DIRECT_ROLLOVER_VERSION,
            old_peer,
            new_peer,
            issued_at_ms,
            expires_at_ms,
            old_signature: old_key.sign(&bytes),
            new_signature: new_key.sign(&bytes),
        })
    }

    pub fn sign_now(
        old_key: &SecretKey,
        new_key: &SecretKey,
    ) -> Result<Self, DirectEnrollmentError> {
        let now_ms = system_time_ms().map_err(DirectEnrollmentError::Clock)?;
        Self::sign(old_key, new_key, now_ms, DEFAULT_ROLLOVER_TTL_MS)
    }

    pub fn verify_now(&self) -> Result<(), DirectEnrollmentError> {
        self.verify_at(system_time_ms().map_err(DirectEnrollmentError::Clock)?)
    }

    pub fn verify_at(&self, now_ms: u64) -> Result<(), DirectEnrollmentError> {
        if self.protocol_version != DIRECT_ROLLOVER_VERSION {
            return Err(DirectEnrollmentError::UnsupportedRolloverVersion {
                received: self.protocol_version,
            });
        }
        if self.old_peer == self.new_peer {
            return Err(DirectEnrollmentError::RolloverIdentityUnchanged);
        }
        if self.expires_at_ms < self.issued_at_ms {
            return Err(DirectEnrollmentError::InvalidRolloverTtl { ttl_ms: 0 });
        }
        let ttl_ms = self.expires_at_ms - self.issued_at_ms;
        if ttl_ms == 0 || ttl_ms > MAX_ROLLOVER_TTL_MS {
            return Err(DirectEnrollmentError::InvalidRolloverTtl { ttl_ms });
        }
        if self.issued_at_ms > now_ms.saturating_add(MAX_CLOCK_SKEW_MS) {
            return Err(DirectEnrollmentError::RolloverIssuedInFuture {
                issued_at_ms: self.issued_at_ms,
                now_ms,
            });
        }
        if now_ms > self.expires_at_ms.saturating_add(MAX_CLOCK_SKEW_MS) {
            return Err(DirectEnrollmentError::RolloverExpired {
                expires_at_ms: self.expires_at_ms,
                now_ms,
            });
        }
        let bytes = rollover_signing_bytes(
            self.protocol_version,
            self.old_peer,
            self.new_peer,
            self.issued_at_ms,
            self.expires_at_ms,
        )?;
        self.old_peer
            .verify(&bytes, &self.old_signature)
            .map_err(|_| DirectEnrollmentError::InvalidOldSignature)?;
        self.new_peer
            .verify(&bytes, &self.new_signature)
            .map_err(|_| DirectEnrollmentError::InvalidNewSignature)
    }
}

fn rollover_signing_bytes(
    protocol_version: u16,
    old_peer: EndpointId,
    new_peer: EndpointId,
    issued_at_ms: u64,
    expires_at_ms: u64,
) -> Result<Vec<u8>, DirectEnrollmentError> {
    encode_bounded(
        &RolloverSigningView {
            protocol_version,
            old_peer,
            new_peer,
            issued_at_ms,
            expires_at_ms,
        },
        MAX_ENROLLMENT_BOOK_BYTES,
    )
    .map_err(DirectEnrollmentError::Codec)
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum DirectEnrollmentError {
    #[error("unsupported direct enrollment-book version {received}")]
    UnsupportedBookVersion { received: u16 },
    #[error("unsupported direct rollover version {received}")]
    UnsupportedRolloverVersion { received: u16 },
    #[error("direct enrollment book contains {count} peers; maximum is {maximum}")]
    TooManyPeers { count: usize, maximum: usize },
    #[error("direct enrollment book contains a duplicate endpoint")]
    DuplicatePeer,
    #[error("rollover must replace the old endpoint with a different endpoint")]
    RolloverIdentityUnchanged,
    #[error("invalid direct rollover TTL: {ttl_ms} ms")]
    InvalidRolloverTtl { ttl_ms: u64 },
    #[error("direct rollover timestamp arithmetic overflow")]
    TimestampOverflow,
    #[error("direct rollover issue time {issued_at_ms} is too far ahead of {now_ms}")]
    RolloverIssuedInFuture { issued_at_ms: u64, now_ms: u64 },
    #[error("direct rollover expired at {expires_at_ms}; current time is {now_ms}")]
    RolloverExpired { expires_at_ms: u64, now_ms: u64 },
    #[error("the old endpoint signature on the rollover proof is invalid")]
    InvalidOldSignature,
    #[error("the new endpoint signature on the rollover proof is invalid")]
    InvalidNewSignature,
    #[error("the rollover old endpoint is not currently enrolled")]
    OldPeerNotEnrolled,
    #[error("the rollover new endpoint is already enrolled")]
    NewPeerAlreadyEnrolled,
    #[error("direct enrollment codec error: {0}")]
    Codec(String),
    #[error("system clock error: {0}")]
    Clock(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enrollment_round_trip_is_canonical() {
        let a = SecretKey::from_bytes(&[41u8; 32]).public();
        let b = SecretKey::from_bytes(&[42u8; 32]).public();
        let book = DirectEnrollmentBook::from_peers([b, a, b]).unwrap();
        assert_eq!(book.peers.len(), 2);
        let decoded = DirectEnrollmentBook::decode(&book.encode().unwrap()).unwrap();
        assert_eq!(decoded, book);
    }

    #[test]
    fn rollover_requires_both_signatures() {
        let old = SecretKey::from_bytes(&[43u8; 32]);
        let new = SecretKey::from_bytes(&[44u8; 32]);
        let mut proof = DirectPeerRolloverProof::sign(&old, &new, 1_000, 1_000).unwrap();
        assert!(proof.verify_at(1_000).is_ok());
        proof.new_signature = old.sign(b"wrong");
        assert_eq!(
            proof.verify_at(1_000),
            Err(DirectEnrollmentError::InvalidNewSignature)
        );
    }
}
