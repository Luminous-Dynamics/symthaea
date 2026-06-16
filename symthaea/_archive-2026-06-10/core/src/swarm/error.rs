// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Swarm Error Types
//!
//! Comprehensive error handling for the swarm module.

use std::fmt;

/// Result type for swarm operations
pub type SwarmResult<T> = Result<T, SwarmError>;

/// Errors that can occur in swarm operations
#[derive(Debug)]
pub enum SwarmError {
    /// Peer is not trusted (Holochain verification failed)
    UntrustedPeer {
        peer_id: String,
        trust_level: f64,
        required: f64,
    },

    /// Connection to peer failed
    ConnectionFailed { peer_id: String, reason: String },

    /// Connection timed out
    ConnectionTimeout { peer_id: String, timeout_ms: u64 },

    /// Peer not found in network
    PeerNotFound { peer_id: String },

    /// Invalid connection ticket
    InvalidTicket { reason: String },

    /// Channel closed unexpectedly
    ChannelClosed { peer_id: String },

    /// Failed to send message
    SendFailed { peer_id: String, reason: String },

    /// Failed to receive message
    ReceiveFailed { peer_id: String, reason: String },

    /// Tensor streaming error
    TensorStreamError { reason: String },

    /// Holochain trust verification error
    TrustVerificationError { reason: String },

    /// Node not initialized
    NotInitialized,

    /// Maximum peers reached
    MaxPeersReached { current: usize, max: usize },

    /// Feature not enabled
    FeatureNotEnabled { feature: String },

    /// Generic IO error
    IoError(std::io::Error),

    /// Serialization error
    SerializationError(String),

    /// Encryption failed (ChaCha20-Poly1305 AEAD error).
    EncryptionFailed { reason: String },

    /// Decryption failed (authentication tag mismatch or malformed ciphertext).
    DecryptionFailed { reason: String },

    /// Attestation required but no attestation manager configured.
    ///
    /// In release builds, `recv_verified_consciousness()` rejects unsigned
    /// ConsciousnessVectors when no `AttestationManager` is set on the node.
    /// Use `recv_consciousness()` explicitly if you intend to accept unverified CVs.
    AttestationRequired,

    /// Internal error
    Internal(String),
}

impl fmt::Display for SwarmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UntrustedPeer {
                peer_id,
                trust_level,
                required,
            } => {
                write!(
                    f,
                    "Peer {peer_id} is not trusted (level: {trust_level:.2}, required: {required:.2})"
                )
            }
            Self::ConnectionFailed { peer_id, reason } => {
                write!(f, "Connection to {peer_id} failed: {reason}")
            }
            Self::ConnectionTimeout {
                peer_id,
                timeout_ms,
            } => {
                write!(f, "Connection to {peer_id} timed out after {timeout_ms}ms")
            }
            Self::PeerNotFound { peer_id } => {
                write!(f, "Peer {peer_id} not found in network")
            }
            Self::InvalidTicket { reason } => {
                write!(f, "Invalid connection ticket: {reason}")
            }
            Self::ChannelClosed { peer_id } => {
                write!(f, "Channel to {peer_id} closed unexpectedly")
            }
            Self::SendFailed { peer_id, reason } => {
                write!(f, "Failed to send to {peer_id}: {reason}")
            }
            Self::ReceiveFailed { peer_id, reason } => {
                write!(f, "Failed to receive from {peer_id}: {reason}")
            }
            Self::TensorStreamError { reason } => {
                write!(f, "Tensor streaming error: {reason}")
            }
            Self::TrustVerificationError { reason } => {
                write!(f, "Trust verification failed: {reason}")
            }
            Self::NotInitialized => {
                write!(f, "Swarm node not initialized")
            }
            Self::MaxPeersReached { current, max } => {
                write!(f, "Maximum peers reached ({current}/{max})")
            }
            Self::FeatureNotEnabled { feature } => {
                write!(f, "Feature '{feature}' not enabled")
            }
            Self::IoError(e) => {
                write!(f, "IO error: {e}")
            }
            Self::SerializationError(msg) => {
                write!(f, "Serialization error: {msg}")
            }
            Self::EncryptionFailed { reason } => {
                write!(f, "Encryption failed: {reason}")
            }
            Self::DecryptionFailed { reason } => {
                write!(f, "Decryption failed: {reason}")
            }
            Self::AttestationRequired => {
                write!(
                    f,
                    "Attestation required: no AttestationManager configured. \
                     Use recv_consciousness() for unverified reception, or \
                     configure an AttestationManager via set_attestation()."
                )
            }
            Self::Internal(msg) => {
                write!(f, "Internal error: {msg}")
            }
        }
    }
}

impl std::error::Error for SwarmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SwarmError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SwarmError::UntrustedPeer {
            peer_id: "abc123".to_string(),
            trust_level: 0.3,
            required: 0.5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("abc123"));
        assert!(msg.contains("0.30"));
        assert!(msg.contains("0.50"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let swarm_err: SwarmError = io_err.into();
        assert!(matches!(swarm_err, SwarmError::IoError(_)));
    }
}
