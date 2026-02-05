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
    ConnectionFailed {
        peer_id: String,
        reason: String,
    },

    /// Connection timed out
    ConnectionTimeout {
        peer_id: String,
        timeout_ms: u64,
    },

    /// Peer not found in network
    PeerNotFound {
        peer_id: String,
    },

    /// Invalid connection ticket
    InvalidTicket {
        reason: String,
    },

    /// Channel closed unexpectedly
    ChannelClosed {
        peer_id: String,
    },

    /// Failed to send message
    SendFailed {
        peer_id: String,
        reason: String,
    },

    /// Failed to receive message
    ReceiveFailed {
        peer_id: String,
        reason: String,
    },

    /// Tensor streaming error
    TensorStreamError {
        reason: String,
    },

    /// Holochain trust verification error
    TrustVerificationError {
        reason: String,
    },

    /// Node not initialized
    NotInitialized,

    /// Maximum peers reached
    MaxPeersReached {
        current: usize,
        max: usize,
    },

    /// Feature not enabled
    FeatureNotEnabled {
        feature: String,
    },

    /// Generic IO error
    IoError(std::io::Error),

    /// Serialization error
    SerializationError(String),

    /// Internal error
    Internal(String),
}

impl fmt::Display for SwarmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UntrustedPeer { peer_id, trust_level, required } => {
                write!(f, "Peer {} is not trusted (level: {:.2}, required: {:.2})",
                       peer_id, trust_level, required)
            }
            Self::ConnectionFailed { peer_id, reason } => {
                write!(f, "Connection to {} failed: {}", peer_id, reason)
            }
            Self::ConnectionTimeout { peer_id, timeout_ms } => {
                write!(f, "Connection to {} timed out after {}ms", peer_id, timeout_ms)
            }
            Self::PeerNotFound { peer_id } => {
                write!(f, "Peer {} not found in network", peer_id)
            }
            Self::InvalidTicket { reason } => {
                write!(f, "Invalid connection ticket: {}", reason)
            }
            Self::ChannelClosed { peer_id } => {
                write!(f, "Channel to {} closed unexpectedly", peer_id)
            }
            Self::SendFailed { peer_id, reason } => {
                write!(f, "Failed to send to {}: {}", peer_id, reason)
            }
            Self::ReceiveFailed { peer_id, reason } => {
                write!(f, "Failed to receive from {}: {}", peer_id, reason)
            }
            Self::TensorStreamError { reason } => {
                write!(f, "Tensor streaming error: {}", reason)
            }
            Self::TrustVerificationError { reason } => {
                write!(f, "Trust verification failed: {}", reason)
            }
            Self::NotInitialized => {
                write!(f, "Swarm node not initialized")
            }
            Self::MaxPeersReached { current, max } => {
                write!(f, "Maximum peers reached ({}/{})", current, max)
            }
            Self::FeatureNotEnabled { feature } => {
                write!(f, "Feature '{}' not enabled", feature)
            }
            Self::IoError(e) => {
                write!(f, "IO error: {}", e)
            }
            Self::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            Self::Internal(msg) => {
                write!(f, "Internal error: {}", msg)
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
