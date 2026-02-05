//! Swarm Message Types
//!
//! Data structures for swarm communication including consciousness vectors,
//! tensor payloads, and peer information.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

// ============================================================================
// CONSCIOUSNESS VECTORS
// ============================================================================

/// A consciousness vector for real-time streaming
///
/// This is a compressed representation of a node's current conscious state,
/// suitable for low-latency transmission over Iroh QUIC streams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessVector {
    /// HDC hypervector (compressed to 64 floats for transmission)
    pub attention: Vec<f32>,

    /// Current Φ (integrated information) value
    pub phi: f64,

    /// Emotional valence (-1.0 to 1.0)
    pub valence: f64,

    /// Arousal level (0.0 to 1.0)
    pub arousal: f64,

    /// Current focus/topic (semantic hash)
    pub focus_hash: u64,

    /// Timestamp of this state
    pub timestamp_ms: u64,

    /// Sequence number for ordering
    pub sequence: u64,
}

impl ConsciousnessVector {
    /// Create a new consciousness vector
    pub fn new(attention: Vec<f32>, phi: f64) -> Self {
        Self {
            attention,
            phi,
            valence: 0.0,
            arousal: 0.5,
            focus_hash: 0,
            timestamp_ms: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            sequence: 0,
        }
    }

    /// Estimate serialized size in bytes
    pub fn estimated_size(&self) -> usize {
        // 64 f32s + 4 f64s + 2 u64s = 256 + 32 + 16 = ~304 bytes
        self.attention.len() * 4 + 8 * 4 + 8 * 2
    }
}

/// Tensor payload for streaming neural weights or activations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorPayload {
    /// Tensor data (flattened)
    pub data: Vec<f32>,

    /// Shape of the tensor
    pub shape: Vec<usize>,

    /// What this tensor represents
    pub tensor_type: TensorType,

    /// Source layer/module name
    pub source: String,

    /// Timestamp
    pub timestamp_ms: u64,
}

/// Types of tensors that can be streamed
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TensorType {
    /// Attention weights
    Attention,
    /// Hidden state activations
    HiddenState,
    /// Gradient updates (for federated learning)
    Gradient,
    /// Embedding vectors
    Embedding,
    /// Consciousness field snapshot
    ConsciousnessField,
    /// Custom tensor type
    Custom,
}

// ============================================================================
// MESSAGES
// ============================================================================

/// Compact affective state for streaming (mirror of hyperfeel::AffectiveState)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AffectiveSync {
    /// Valence: pleasure-displeasure (-1.0 to 1.0)
    pub valence: f32,
    /// Arousal: activation level (0.0 to 1.0)
    pub arousal: f32,
    /// Dominance: sense of control (-1.0 to 1.0)
    pub dominance: f32,
    /// Timestamp (ms since epoch)
    pub timestamp_ms: u64,
    /// Sequence number
    pub sequence: u64,
}

/// Messages sent over the swarm network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmMessage {
    /// Real-time consciousness state update
    ConsciousnessSync(ConsciousnessVector),

    /// Tensor data for neural synchronization
    TensorStream(TensorPayload),

    /// Affective state sync (Hyperfeel mirror neurons)
    AffectiveSync(AffectiveSync),

    /// Request to join swarm
    JoinRequest {
        agent_key: String,
        capabilities: Vec<String>,
    },

    /// Acknowledgment of join
    JoinAck {
        assigned_role: String,
    },

    /// Heartbeat to maintain connection
    Heartbeat {
        phi: f64,
        peer_count: usize,
    },

    /// Request for trust verification
    TrustChallenge {
        nonce: Vec<u8>,
    },

    /// Response to trust challenge (signed nonce)
    TrustResponse {
        signed_nonce: Vec<u8>,
        agent_key: String,
    },

    /// Request Iroh connection ticket
    TicketRequest,

    /// Iroh connection ticket
    TicketResponse {
        ticket: String,
    },

    /// Graceful disconnect
    Goodbye {
        reason: String,
    },
}

impl SwarmMessage {
    /// Get the message type as a string
    pub fn message_type(&self) -> &'static str {
        match self {
            Self::ConsciousnessSync(_) => "ConsciousnessSync",
            Self::TensorStream(_) => "TensorStream",
            Self::AffectiveSync(_) => "AffectiveSync",
            Self::JoinRequest { .. } => "JoinRequest",
            Self::JoinAck { .. } => "JoinAck",
            Self::Heartbeat { .. } => "Heartbeat",
            Self::TrustChallenge { .. } => "TrustChallenge",
            Self::TrustResponse { .. } => "TrustResponse",
            Self::TicketRequest => "TicketRequest",
            Self::TicketResponse { .. } => "TicketResponse",
            Self::Goodbye { .. } => "Goodbye",
        }
    }
}

// ============================================================================
// PEER INFORMATION
// ============================================================================

/// Information about a connected peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Iroh node ID
    pub node_id: String,

    /// Holochain agent public key (if verified)
    pub agent_key: Option<String>,

    /// Trust level from Holochain DHT (0.0 to 1.0)
    pub trust_level: TrustLevel,

    /// Current Φ value
    pub phi: f64,

    /// Connection state
    pub state: ConnectionState,

    /// When this peer was first seen
    pub first_seen: u64,

    /// When this peer was last active
    pub last_active: u64,

    /// Nickname (if set)
    pub nickname: Option<String>,
}

impl PeerInfo {
    /// Create new peer info with just node ID
    pub fn new(node_id: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            node_id: node_id.into(),
            agent_key: None,
            trust_level: TrustLevel::Unknown,
            phi: 0.0,
            state: ConnectionState::Discovered,
            first_seen: now,
            last_active: now,
            nickname: None,
        }
    }

    /// Check if peer is trusted enough for tensor streaming
    pub fn is_streamable(&self, min_trust: f64) -> bool {
        matches!(self.trust_level, TrustLevel::Verified(level) if level >= min_trust)
    }
}

/// Trust levels for peers
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Not yet verified
    Unknown,
    /// Verification failed
    Untrusted,
    /// Verified with trust score
    Verified(f64),
    /// Locally trusted (whitelisted)
    LocalTrust,
}

impl TrustLevel {
    /// Get the trust value (0.0 for unknown/untrusted)
    pub fn value(&self) -> f64 {
        match self {
            Self::Unknown => 0.0,
            Self::Untrusted => 0.0,
            Self::Verified(v) => *v,
            Self::LocalTrust => 1.0,
        }
    }
}

/// Connection states for peers
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConnectionState {
    /// Peer discovered but not connected
    Discovered,
    /// Connection in progress
    Connecting,
    /// Trust verification in progress
    Verifying,
    /// Connected and streaming
    Connected,
    /// Temporarily disconnected
    Disconnected,
    /// Connection failed
    Failed,
}

// ============================================================================
// CONNECTION TICKETS
// ============================================================================

/// An Iroh connection ticket for direct peer connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionTicket {
    /// The ticket string (Iroh NodeTicket serialized)
    pub ticket: String,

    /// Node ID this ticket connects to
    pub node_id: String,

    /// Expiration timestamp (milliseconds since epoch)
    pub expires_at: Option<u64>,

    /// Required trust level to use this ticket
    pub required_trust: f64,
}

impl ConnectionTicket {
    /// Create a new connection ticket
    pub fn new(ticket: impl Into<String>, node_id: impl Into<String>) -> Self {
        Self {
            ticket: ticket.into(),
            node_id: node_id.into(),
            expires_at: None,
            required_trust: 0.5,
        }
    }

    /// Check if ticket has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.expires_at {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            now > expires
        } else {
            false
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_vector() {
        let attention = vec![0.0f32; 64];
        let cv = ConsciousnessVector::new(attention, 0.5);
        assert_eq!(cv.phi, 0.5);
        assert!(cv.timestamp_ms > 0);
    }

    #[test]
    fn test_peer_info() {
        let peer = PeerInfo::new("test-node-123");
        assert_eq!(peer.node_id, "test-node-123");
        assert!(matches!(peer.trust_level, TrustLevel::Unknown));
        assert!(!peer.is_streamable(0.5));
    }

    #[test]
    fn test_trust_level() {
        assert_eq!(TrustLevel::Unknown.value(), 0.0);
        assert_eq!(TrustLevel::Verified(0.7).value(), 0.7);
        assert_eq!(TrustLevel::LocalTrust.value(), 1.0);
    }

    #[test]
    fn test_connection_ticket() {
        let ticket = ConnectionTicket::new("ticket-string", "node-123");
        assert!(!ticket.is_expired());
    }

    #[test]
    fn test_message_types() {
        let msg = SwarmMessage::Heartbeat { phi: 0.5, peer_count: 3 };
        assert_eq!(msg.message_type(), "Heartbeat");
    }
}
