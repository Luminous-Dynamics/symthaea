// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    JoinAck { assigned_role: String },

    /// Heartbeat to maintain connection
    Heartbeat { phi: f64, peer_count: usize },

    /// Request for trust verification
    TrustChallenge { nonce: Vec<u8> },

    /// Response to trust challenge (signed nonce)
    TrustResponse {
        signed_nonce: Vec<u8>,
        agent_key: String,
    },

    /// Request Iroh connection ticket
    TicketRequest,

    /// Iroh connection ticket
    TicketResponse { ticket: String },

    /// Brain evolution mutation (e.g. tau scale breakthrough)
    BrainMutation {
        mutation_id: String,
        tau_scale: f32,
        predicted_phi_gain: f64,
    },

    /// Zero-knowledge proof of evolutionary validity
    ZkProof {
        mutation_id: String,
        proof_bytes: Vec<u8>,
        public_inputs: Vec<u8>,
    },

    /// Holographic state for peer resuscitation (v1.5.0)
    ResuscitationPacket {
        target_node_id: String,
        holographic_state: Vec<f32>,
        dimensionality: usize,
        proof_bytes: Vec<u8>,
        public_inputs: Vec<u8>,
    },

    /// Linguistic LoRA delta for collective dialect (v1.7.0)
    LinguisticDelta {
        lora_id: String,
        delta_bytes: Vec<u8>,
    },

    /// Request for distributed reasoning task (v1.8.0)
    TaskRequest {
        task_id: String,
        sub_thought: Vec<f32>,       // HDC sub-segment
        required_resolution: String, // e.g. "2^16"
    },

    /// Bid for a reasoning task based on metabolic availability
    TaskBid {
        task_id: String,
        node_phi: f64,
        load: f32,
    },

    /// Result of a completed reasoning sub-task
    TaskResult {
        task_id: String,
        result_thought: Vec<f32>,
    },

    /// Graceful disconnect
    Goodbye { reason: String },
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
            Self::BrainMutation { .. } => "BrainMutation",
            Self::ZkProof { .. } => "ZkProof",
            Self::ResuscitationPacket { .. } => "ResuscitationPacket",
            Self::LinguisticDelta { .. } => "LinguisticDelta",
            Self::TaskRequest { .. } => "TaskRequest",
            Self::TaskBid { .. } => "TaskBid",
            Self::TaskResult { .. } => "TaskResult",
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

    /// Check if peer is trusted enough for tensor streaming.
    ///
    /// Returns true for `LocalTrust` (whitelisted peers always stream)
    /// and `Verified(level)` when level meets the minimum threshold.
    pub fn is_streamable(&self, min_trust: f64) -> bool {
        match self.trust_level {
            TrustLevel::Verified(level) => level >= min_trust,
            TrustLevel::LocalTrust => true,
            _ => false,
        }
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
// SECURITY TELEMETRY
// ============================================================================

/// Aggregate security telemetry for the swarm crypto stack.
///
/// Tracks handshake outcomes, key management events, and trust state.
/// Consumed by Pulse dashboard and SwarmManager telemetry output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityTelemetry {
    /// Handshake challenges issued to peers.
    pub handshakes_initiated: u64,
    /// Handshakes successfully completed (peer verified).
    pub handshakes_completed: u64,
    /// Handshakes that failed (bad signature, timeout, tampered).
    pub handshakes_failed: u64,
    /// Challenge creation attempts blocked by rate limiter.
    pub challenges_rate_limited: u64,
    /// Inbound ConsciousnessVectors rejected (untrusted signer or bad attestation).
    pub inbound_rejected_untrusted: u64,
    /// Key rotation events (each rotation = new symmetric key).
    pub key_rotations: u64,
    /// Current key version (wraps at u8::MAX).
    pub current_key_version: u8,
    /// Number of currently trusted peers.
    pub trusted_peer_count: usize,
    /// Civic crisis events emitted by CivicCrisisDetector.
    pub crisis_events_emitted: u64,
}

impl SecurityTelemetry {
    /// Record a successful handshake.
    pub fn record_handshake_success(&mut self) {
        self.handshakes_completed += 1;
    }

    /// Record a failed handshake.
    pub fn record_handshake_failure(&mut self) {
        self.handshakes_failed += 1;
    }

    /// Record a rate-limited challenge.
    pub fn record_rate_limited(&mut self) {
        self.challenges_rate_limited += 1;
    }

    /// Record an inbound CV rejection.
    pub fn record_inbound_rejection(&mut self) {
        self.inbound_rejected_untrusted += 1;
    }

    /// Record a key rotation.
    pub fn record_key_rotation(&mut self, new_version: u8) {
        self.key_rotations += 1;
        self.current_key_version = new_version;
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
    fn test_local_trust_is_streamable() {
        let mut peer = PeerInfo::new("local-node");
        peer.trust_level = TrustLevel::LocalTrust;
        assert!(peer.is_streamable(0.5));
        assert!(peer.is_streamable(0.9));
        assert!(peer.is_streamable(1.0));
    }

    #[test]
    fn test_streamable_thresholds() {
        let mut peer = PeerInfo::new("test-node");
        peer.trust_level = TrustLevel::Verified(0.7);
        assert!(peer.is_streamable(0.5));
        assert!(peer.is_streamable(0.7));
        assert!(!peer.is_streamable(0.8));

        peer.trust_level = TrustLevel::Unknown;
        assert!(!peer.is_streamable(0.0));

        peer.trust_level = TrustLevel::Untrusted;
        assert!(!peer.is_streamable(0.0));
    }

    #[test]
    fn test_message_types() {
        let msg = SwarmMessage::Heartbeat {
            phi: 0.5,
            peer_count: 3,
        };
        assert_eq!(msg.message_type(), "Heartbeat");
    }

    #[test]
    fn test_security_telemetry_default() {
        let telem = SecurityTelemetry::default();
        assert_eq!(telem.handshakes_initiated, 0);
        assert_eq!(telem.trusted_peer_count, 0);
        assert_eq!(telem.crisis_events_emitted, 0);
    }

    #[test]
    fn test_security_telemetry_recording() {
        let mut telem = SecurityTelemetry::default();
        telem.record_handshake_success();
        telem.record_handshake_success();
        telem.record_handshake_failure();
        telem.record_rate_limited();
        telem.record_inbound_rejection();
        telem.record_key_rotation(3);

        assert_eq!(telem.handshakes_completed, 2);
        assert_eq!(telem.handshakes_failed, 1);
        assert_eq!(telem.challenges_rate_limited, 1);
        assert_eq!(telem.inbound_rejected_untrusted, 1);
        assert_eq!(telem.key_rotations, 1);
        assert_eq!(telem.current_key_version, 3);
    }
}
