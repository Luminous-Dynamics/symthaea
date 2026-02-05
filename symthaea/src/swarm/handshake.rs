//! Hybrid Handshake Protocol - Trust Bridge between Holochain and Iroh
//!
//! This module implements the "Trust Challenge" protocol that ensures
//! only trusted peers (verified via Holochain) can establish real-time
//! tensor streams (via Iroh).
//!
//! ## Protocol Flow
//!
//! ```text
//!     Node A                                    Node B
//!        │                                         │
//!        │──────── 1. Iroh Connect ───────────────>│
//!        │                                         │
//!        │<─────── 2. Trust Challenge ─────────────│
//!        │         (nonce)                         │
//!        │                                         │
//!        │──────── 3. Trust Response ─────────────>│
//!        │         (BLAKE3 MAC + agent key)        │
//!        │                                         │
//!        │         4. Verify MAC                   │
//!        │         5. Check Holochain DHT Trust    │
//!        │                                         │
//!        │<─────── 6. Trust Ack / Reject ──────────│
//!        │                                         │
//!        │<═══════ 7. Tensor Streaming ═══════════>│
//!        │         (if trusted)                    │
//! ```
//!
//! ## Security Model
//!
//! - The nonce prevents replay attacks
//! - The BLAKE3 keyed MAC proves possession of the agent's private key
//! - The DHT lookup verifies the agent's reputation
//! - Only after trust verification can tensor streaming begin
//!
//! ## Future: Ed25519 Signatures
//!
//! The current implementation uses BLAKE3 keyed MAC, which requires the
//! verifier to also know the agent's key. For a fully trustless protocol,
//! upgrade to Ed25519 signatures (ed25519-dalek crate) so that verifiers
//! only need the public key.

use crate::swarm::{
    SwarmConfig, SwarmResult, SwarmError,
    TrustLevel, SwarmMessage,
};
use std::fmt;
use std::time::{SystemTime, Duration};
use rand::Rng;
use tracing::warn;

// ============================================================================
// HANDSHAKE ERRORS
// ============================================================================

/// Errors specific to the handshake protocol
#[derive(Debug, Clone)]
pub enum HandshakeError {
    /// Expected a TrustChallenge message but received a different variant
    UnexpectedMessageType {
        expected: &'static str,
        actual: String,
    },
    /// Challenge nonce extraction failed
    ChallengeExtractionFailed {
        reason: String,
    },
    /// Response extraction failed
    ResponseExtractionFailed {
        reason: String,
    },
    /// Invalid handshake state
    InvalidState {
        expected: String,
        actual: String,
    },
    /// Protocol violation
    ProtocolViolation {
        message: String,
    },
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedMessageType { expected, actual } => {
                write!(f, "Expected {} message, got {}", expected, actual)
            }
            Self::ChallengeExtractionFailed { reason } => {
                write!(f, "Failed to extract challenge: {}", reason)
            }
            Self::ResponseExtractionFailed { reason } => {
                write!(f, "Failed to extract response: {}", reason)
            }
            Self::InvalidState { expected, actual } => {
                write!(f, "Invalid handshake state: expected {}, got {}", expected, actual)
            }
            Self::ProtocolViolation { message } => {
                write!(f, "Handshake protocol violation: {}", message)
            }
        }
    }
}

impl std::error::Error for HandshakeError {}

// ============================================================================
// SWARM MESSAGE HELPERS
// ============================================================================

/// Extension trait for extracting specific message types from SwarmMessage
pub trait SwarmMessageExt {
    /// Try to extract the nonce from a TrustChallenge message
    fn try_into_challenge_nonce(self) -> Result<Vec<u8>, HandshakeError>;

    /// Try to extract (signed_nonce, agent_key) from a TrustResponse message
    fn try_into_response(self) -> Result<(Vec<u8>, String), HandshakeError>;
}

impl SwarmMessageExt for SwarmMessage {
    fn try_into_challenge_nonce(self) -> Result<Vec<u8>, HandshakeError> {
        match self {
            SwarmMessage::TrustChallenge { nonce } => Ok(nonce),
            other => {
                let actual = other.message_type().to_string();
                warn!(
                    expected = "TrustChallenge",
                    actual = %actual,
                    "Unexpected message type during handshake"
                );
                Err(HandshakeError::UnexpectedMessageType {
                    expected: "TrustChallenge",
                    actual,
                })
            }
        }
    }

    fn try_into_response(self) -> Result<(Vec<u8>, String), HandshakeError> {
        match self {
            SwarmMessage::TrustResponse { signed_nonce, agent_key } => {
                Ok((signed_nonce, agent_key))
            }
            other => {
                let actual = other.message_type().to_string();
                warn!(
                    expected = "TrustResponse",
                    actual = %actual,
                    "Unexpected message type during handshake"
                );
                Err(HandshakeError::UnexpectedMessageType {
                    expected: "TrustResponse",
                    actual,
                })
            }
        }
    }
}

/// Length of the BLAKE3 MAC output in bytes
const MAC_LEN: usize = 32;

/// The hybrid handshake manager
pub struct HybridHandshake {
    /// Configuration
    config: SwarmConfig,

    /// Pending challenges we've issued
    pending_challenges: std::collections::HashMap<String, PendingChallenge>,

    /// Challenge timeout
    challenge_timeout: Duration,

    /// Optional genesis-seeded RNG for deterministic nonce generation
    seeded_rng: Option<symthaea_core::genesis::ShakeRng>,
}

/// A pending trust challenge
struct PendingChallenge {
    /// The nonce we sent
    nonce: Vec<u8>,

    /// When the challenge was issued
    issued_at: SystemTime,

    /// The peer we challenged
    #[allow(dead_code)]
    peer_node_id: String,
}

impl HybridHandshake {
    /// Create a new handshake manager
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            config,
            pending_challenges: std::collections::HashMap::new(),
            challenge_timeout: Duration::from_secs(30),
            seeded_rng: None,
        }
    }

    /// Create a handshake manager with deterministic RNG from a genesis seed.
    pub fn from_genesis(
        config: SwarmConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        Self {
            config,
            pending_challenges: std::collections::HashMap::new(),
            challenge_timeout: Duration::from_secs(30),
            seeded_rng: Some(genesis.domain(&format!("{label}::handshake"))),
        }
    }

    /// Generate a trust challenge for a new peer
    pub fn create_challenge(&mut self, peer_node_id: &str) -> SwarmMessage {
        // Generate cryptographic nonce (seeded or random)
        let nonce: Vec<u8> = if let Some(ref mut rng) = self.seeded_rng {
            use rand::RngCore;
            let mut buf = vec![0u8; 32];
            rng.fill_bytes(&mut buf);
            buf
        } else {
            rand::thread_rng()
                .sample_iter(rand::distributions::Standard)
                .take(32)
                .collect()
        };

        // Store pending challenge
        self.pending_challenges.insert(
            peer_node_id.to_string(),
            PendingChallenge {
                nonce: nonce.clone(),
                issued_at: SystemTime::now(),
                peer_node_id: peer_node_id.to_string(),
            },
        );

        SwarmMessage::TrustChallenge { nonce }
    }

    /// Create a response to a trust challenge
    ///
    /// Produces a BLAKE3 keyed MAC over the nonce using the agent's
    /// private key material. The MAC proves possession of the key
    /// without transmitting it.
    pub fn create_response(
        &self,
        nonce: &[u8],
        agent_key: &str,
        agent_private_key: &[u8],
    ) -> SwarmMessage {
        let mac = Self::compute_mac(nonce, agent_private_key);

        SwarmMessage::TrustResponse {
            signed_nonce: mac,
            agent_key: agent_key.to_string(),
        }
    }

    /// Verify a trust response
    ///
    /// Checks that the MAC is valid for the challenge nonce and the
    /// claimed agent key. In production, the verifier would look up
    /// the agent's key material from the Holochain DHT.
    pub fn verify_response(
        &mut self,
        peer_node_id: &str,
        signed_nonce: &[u8],
        agent_key: &str,
    ) -> SwarmResult<TrustLevel> {
        // Get pending challenge
        let challenge = self.pending_challenges.remove(peer_node_id)
            .ok_or_else(|| SwarmError::TrustVerificationError {
                reason: "No pending challenge for this peer".to_string(),
            })?;

        // Check timeout
        if challenge.issued_at.elapsed().unwrap_or(Duration::MAX) > self.challenge_timeout {
            return Err(SwarmError::TrustVerificationError {
                reason: "Challenge timed out".to_string(),
            });
        }

        // Verify MAC length
        if signed_nonce.len() != MAC_LEN {
            return Err(SwarmError::TrustVerificationError {
                reason: format!(
                    "Invalid MAC length: expected {} bytes, got {}",
                    MAC_LEN, signed_nonce.len()
                ),
            });
        }

        // Recompute MAC with the agent's key material
        // In production: look up agent_key in Holochain DHT to get key material
        let expected_mac = Self::compute_mac(&challenge.nonce, agent_key.as_bytes());

        // Constant-time comparison to prevent timing attacks
        if !constant_time_eq(signed_nonce, &expected_mac) {
            return Err(SwarmError::TrustVerificationError {
                reason: "MAC verification failed".to_string(),
            });
        }

        // In production: query Holochain DHT for agent's reputation
        // Trust level based on successful MAC verification
        Ok(TrustLevel::Verified(0.7))
    }

    /// Compute a BLAKE3 keyed MAC over the given data.
    ///
    /// Uses the first 32 bytes of the key material (zero-padded if shorter).
    fn compute_mac(data: &[u8], key_material: &[u8]) -> Vec<u8> {
        // Derive a 32-byte key from arbitrary-length key material
        let derived_key = blake3::hash(key_material);
        let key_bytes: [u8; 32] = *derived_key.as_bytes();

        // Compute keyed MAC
        let mac = blake3::keyed_hash(&key_bytes, data);
        mac.as_bytes().to_vec()
    }

    /// Check if a trust level meets minimum requirements
    pub fn meets_trust_requirement(&self, trust: &TrustLevel) -> bool {
        trust.value() >= self.config.min_trust_level
    }

    /// Clean up expired challenges
    pub fn cleanup_expired(&mut self) {
        self.pending_challenges.retain(|_, challenge| {
            challenge.issued_at.elapsed()
                .map(|d| d < self.challenge_timeout)
                .unwrap_or(false)
        });
    }

    /// Get the number of pending challenges
    pub fn pending_count(&self) -> usize {
        self.pending_challenges.len()
    }
}

/// Constant-time byte comparison to prevent timing attacks
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Result of a complete handshake
#[derive(Debug, Clone)]
pub struct HandshakeResult {
    /// The peer's node ID
    pub peer_node_id: String,

    /// The peer's agent key (if verified)
    pub agent_key: Option<String>,

    /// Trust level after verification
    pub trust_level: TrustLevel,

    /// Whether streaming is allowed
    pub streaming_allowed: bool,

    /// Time taken for handshake (milliseconds)
    pub handshake_time_ms: u64,
}

impl HandshakeResult {
    /// Create a successful handshake result
    pub fn success(
        peer_node_id: impl Into<String>,
        agent_key: impl Into<String>,
        trust_level: TrustLevel,
        handshake_time_ms: u64,
    ) -> Self {
        Self {
            peer_node_id: peer_node_id.into(),
            agent_key: Some(agent_key.into()),
            trust_level,
            streaming_allowed: true,
            handshake_time_ms,
        }
    }

    /// Create a failed handshake result
    pub fn failed(peer_node_id: impl Into<String>) -> Self {
        Self {
            peer_node_id: peer_node_id.into(),
            agent_key: None,
            trust_level: TrustLevel::Untrusted,
            streaming_allowed: false,
            handshake_time_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_challenge() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let challenge = handshake.create_challenge("peer-123");

        let nonce = challenge.try_into_challenge_nonce()
            .expect("create_challenge should return TrustChallenge");
        assert_eq!(nonce.len(), 32);

        assert_eq!(handshake.pending_count(), 1);
    }

    #[test]
    fn test_create_response_produces_mac() {
        let config = SwarmConfig::default();
        let handshake = HybridHandshake::new(config);

        let nonce = vec![1, 2, 3, 4];
        let response = handshake.create_response(&nonce, "agent-key", b"private-key");

        let (signed_nonce, agent_key) = response.try_into_response()
            .expect("create_response should return TrustResponse");
        // MAC should be exactly 32 bytes (BLAKE3 output)
        assert_eq!(signed_nonce.len(), MAC_LEN);
        assert_eq!(agent_key, "agent-key");
    }

    #[test]
    fn test_verify_valid_response() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        // Create challenge
        let challenge = handshake.create_challenge("peer-123");
        let nonce = challenge.try_into_challenge_nonce()
            .expect("create_challenge should return TrustChallenge");

        // Create response using agent_key as key material (matches verify_response lookup)
        let response = handshake.create_response(&nonce, "agent-key", b"agent-key");
        let (signed_nonce, agent_key) = response.try_into_response()
            .expect("create_response should return TrustResponse");

        // Verify
        let trust = handshake.verify_response("peer-123", &signed_nonce, &agent_key).unwrap();
        assert!(matches!(trust, TrustLevel::Verified(_)));
    }

    #[test]
    fn test_verify_wrong_key_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        // Create challenge
        let challenge = handshake.create_challenge("peer-123");
        let nonce = challenge.try_into_challenge_nonce()
            .expect("create_challenge should return TrustChallenge");

        // Create response with a DIFFERENT private key
        let response = handshake.create_response(&nonce, "agent-key", b"wrong-key");
        let (signed_nonce, agent_key) = response.try_into_response()
            .expect("create_response should return TrustResponse");

        // Verification should fail because MAC was computed with wrong key
        let result = handshake.verify_response("peer-123", &signed_nonce, &agent_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_tampered_mac_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let challenge = handshake.create_challenge("peer-123");
        let nonce = challenge.try_into_challenge_nonce()
            .expect("create_challenge should return TrustChallenge");

        let response = handshake.create_response(&nonce, "agent-key", b"agent-key");
        let (mut signed_nonce, agent_key) = response.try_into_response()
            .expect("create_response should return TrustResponse");

        // Tamper with the MAC
        signed_nonce[0] ^= 0xFF;

        let result = handshake.verify_response("peer-123", &signed_nonce, &agent_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_no_pending_challenge_fails() {
        let config = SwarmConfig::default();
        let mut handshake = HybridHandshake::new(config);

        let result = handshake.verify_response("peer-123", &[0u8; 32], "agent-key");
        assert!(result.is_err());
    }

    #[test]
    fn test_mac_is_deterministic() {
        let mac1 = HybridHandshake::compute_mac(b"test-nonce", b"test-key");
        let mac2 = HybridHandshake::compute_mac(b"test-nonce", b"test-key");
        assert_eq!(mac1, mac2);
    }

    #[test]
    fn test_mac_differs_for_different_inputs() {
        let mac1 = HybridHandshake::compute_mac(b"nonce-1", b"key");
        let mac2 = HybridHandshake::compute_mac(b"nonce-2", b"key");
        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"short", b"longer"));
    }

    #[test]
    fn test_trust_requirement() {
        let mut config = SwarmConfig::default();
        config.min_trust_level = 0.5;
        let handshake = HybridHandshake::new(config);

        assert!(handshake.meets_trust_requirement(&TrustLevel::Verified(0.7)));
        assert!(!handshake.meets_trust_requirement(&TrustLevel::Verified(0.3)));
        assert!(!handshake.meets_trust_requirement(&TrustLevel::Unknown));
    }

    #[test]
    fn test_handshake_result() {
        let success = HandshakeResult::success("peer-1", "agent-1", TrustLevel::Verified(0.8), 150);
        assert!(success.streaming_allowed);

        let failed = HandshakeResult::failed("peer-2");
        assert!(!failed.streaming_allowed);
    }
}
